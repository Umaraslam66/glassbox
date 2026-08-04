"""Open-answer fusion for the Stage 3 person encoder (Gate 3).

    python -m src.model.open_fusion \
        --transcripts data/runs/interview_v1/transcripts.jsonl \
        --features data/runs/stage3_qwen/open_answer_features.npz \
        --fit results/stage2_v2_fit.npz \
        --splits experiments/splits_v1.json \
        --out results/stage3_person_fused.json

What this is
------------
The person-encoder backbone (:mod:`src.model.person_encoder`) is the exact
posterior over a persona's latent vector given the closed interview answers.
This module adds the second channel: the three open-ended answers, read by the
frozen system-side model into 24 pole judgments (3 prompts x 8 poles), enter as
**one extra Gaussian measurement** of the same latent vector.

The measurement model is the pseudo-item form the backbone's extension point
already speaks:

    y = B theta + c + eps,    eps ~ N(0, Psi)

with ``y`` the 24 standardized judgments, ``B`` a 24x8 matrix, ``c`` an offset
and ``Psi`` the noise covariance (diagonal, or diagonal plus a low-rank factor).
:meth:`src.model.person_encoder.GaussianBlock.from_linear` turns that into the
information pair ``Lambda = B^T Psi^-1 B``, ``h = B^T Psi^-1 (y - c)`` which the
posterior adds to its negative log posterior. Two consequences matter:

* the result is still a genuine posterior, so the blur means the same thing it
  meant before and stays comparable across N; and
* ``Lambda`` does not depend on N, so the open answers are present at **every**
  prefix length -- which is the truth of the interview: the persona gave those
  three answers up front.

Fitting, and what it is fitted against
--------------------------------------
``B``, ``c`` and ``Psi`` are estimated on the **400 training personas only**,
against those personas' fitted ``theta_hat`` from the full training-item matrix
(``theta_train`` in the Stage 2 fit archive). That target is a system-side
fitted quantity, not a planted value.

Regularization, with 400 samples and 24 features:

* ridge on ``B`` (the intercept is never penalized), lambda chosen by 5-fold
  cross-validation on the training personas;
* ``Psi`` estimated from **out-of-fold** residuals, so the channel cannot look
  more precise than it is, and chosen between a diagonal and a low-rank-plus-
  diagonal form by the same cross-validation;
* the selection criterion is the out-of-fold negative log predictive density of
  ``theta_hat_full`` under the open-answers-only posterior. It scores the mean
  and the spread at once, so an over-confident ``Psi`` is punished.

Nothing about the 100 held-out personas takes part in any of that.

Two tracks, one run
-------------------
Every number is produced twice: **closed-only** (no block -- the backbone,
recomputed here by the same code so the two artifacts can be compared cell by
cell) and **fused** (with the block). The blur calibration is re-run exactly as
the backbone ran it -- one scalar per dimension from the spread of
``theta_hat_N - theta_hat_full`` over the training personas at N = 15 -- but now
with the block in place, and the fused constants are written beside the
backbone's rather than over them.

N = 0 is scored as well. On the closed-only track it is the prior; on the fused
track it is the open answers alone, which is what makes "the three open answers
are worth X closed items" a measurable statement rather than a figure of speech.

WALL RULES. This module reads the assembled transcripts, the open-answer
judgment features, the fitted item parameters and the frozen split manifest.
It never reads a planted trait value, a designed loading, a persona card or an
answer distribution, and no path to any of those appears in it. Every path is a
command-line argument. Nothing here is compared to planted truth -- that is the
Gate 3 grader's job (``src/eval/gate3.py``).
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import mirt, person_encoder as pe

#: Schema tag on the output artifact.
SCHEMA = "glassbox.stage3.person_fused/1"

#: Ridge grid for the measurement model, log-spaced.
DEFAULT_LAMBDA_GRID = tuple(float(v) for v in np.logspace(-3, 4, 29))

#: Noise-covariance shapes offered to model selection. ``0`` is plain diagonal.
DEFAULT_FACTOR_GRID = (0, 1, 2, 3)

#: Cross-validation folds and their seed. Training personas only.
DEFAULT_FOLDS = 5
DEFAULT_CV_SEED = 2026

#: Floor on the diagonal of Psi, as a fraction of the mean residual variance.
_PSI_FLOOR = 1e-3

#: Iterations of the principal-factor pass that fits the low-rank Psi.
_FACTOR_ITERS = 200

#: How far past the end of the interview an equivalence may be extrapolated,
#: as a multiple of the interview length. Past this the answer is "more than
#: the whole interview", which is more honest than a number.
_EXTRAPOLATION_CAP = 3.0

_EPS = mirt._EPS


# --------------------------------------------------------------------------
# the open-answer features
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class OpenFeatures:
    """The judgment features, one row per persona."""

    pids: list[str]
    values: np.ndarray  # (n_personas, m)
    columns: list[str]  # human-readable name per column
    prompt_ids: list[str]
    poles: list[str]
    coverage: float

    def rows_for(self, pids: Sequence[str]) -> np.ndarray:
        index = {pid: i for i, pid in enumerate(self.pids)}
        missing = [p for p in pids if p not in index]
        if missing:
            raise ValueError(
                f"{len(missing)} personas have no open-answer features: {missing[:5]}"
            )
        return self.values[[index[p] for p in pids]]


def load_open_features(path: str | Path) -> OpenFeatures:
    """Read the judgment archive: 3 open prompts x 8 poles per persona."""
    archive = np.load(Path(path), allow_pickle=False)
    pids = [str(p) for p in archive["pids"]]
    values = np.asarray(archive["features"], dtype=float)
    prompt_ids = [str(p) for p in archive["prompt_ids"]]
    poles = [str(p) for p in archive["poles"]]
    observed = np.asarray(archive["observed"], dtype=bool)
    expected = len(prompt_ids) * len(poles)
    if values.shape[1] != expected:
        raise ValueError(
            f"features has {values.shape[1]} columns but "
            f"{len(prompt_ids)} prompts x {len(poles)} poles = {expected}"
        )
    columns = [f"{prompt}:{pole}" for prompt in prompt_ids for pole in poles]
    return OpenFeatures(
        pids=pids,
        values=values,
        columns=columns,
        prompt_ids=prompt_ids,
        poles=poles,
        coverage=float(observed.mean()),
    )


# --------------------------------------------------------------------------
# the measurement model: y = B theta + c + eps
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MeasurementModel:
    """``y = B theta + c + eps``, ``eps ~ N(0, Psi)``, on standardized features.

    ``center`` and ``scale`` are the per-feature standardization constants,
    learned on the training personas. ``loading`` is ``B``, ``offset`` is ``c``
    and ``noise_cov`` is ``Psi``, all in standardized units. The posterior does
    not notice the standardization -- rescaling ``y`` rescales ``B`` and ``Psi``
    together and leaves ``Lambda`` and ``h`` untouched -- but it makes one ridge
    penalty mean the same thing for every column.
    """

    center: np.ndarray  # (m,)
    scale: np.ndarray  # (m,)
    loading: np.ndarray  # (m, dims)
    offset: np.ndarray  # (m,)
    noise_cov: np.ndarray  # (m, m)
    ridge: float
    n_factors: int
    n_personas: int

    def standardize(self, raw: np.ndarray) -> np.ndarray:
        """Put raw judgment scores on the units the model was fitted in."""
        return (np.asarray(raw, dtype=float) - self.center) / self.scale

    def block(self, raw: np.ndarray) -> pe.GaussianBlock:
        """The Gaussian measurement these personas' open answers make."""
        return pe.GaussianBlock.from_linear(
            self.loading, self.offset, self.standardize(raw), self.noise_cov
        )

    @property
    def precision(self) -> np.ndarray:
        """``Lambda = B^T Psi^-1 B``, the information one open-answer set adds."""
        weighted = np.linalg.solve(self.noise_cov, self.loading)
        lam = self.loading.T @ weighted
        return 0.5 * (lam + lam.T)

    def to_json(self) -> dict[str, Any]:
        lam = self.precision
        return {
            "form": "y = B theta + c + eps, eps ~ N(0, Psi); y standardized",
            "ridge_lambda": float(self.ridge),
            "n_factors_in_psi": int(self.n_factors),
            "psi_form": (
                "diagonal" if self.n_factors == 0
                else f"diagonal plus rank-{self.n_factors} factor"
            ),
            "fitted_on_n_personas": int(self.n_personas),
            "fitted_against": "theta_hat from the full training-item matrix",
            "feature_center": [float(v) for v in self.center],
            "feature_scale": [float(v) for v in self.scale],
            "B": self.loading.tolist(),
            "c": [float(v) for v in self.offset],
            "psi_diagonal": [float(v) for v in np.diag(self.noise_cov)],
            "psi_max_abs_offdiagonal": float(
                np.max(np.abs(self.noise_cov - np.diag(np.diag(self.noise_cov))))
            ),
            "information_matrix_lambda": lam.tolist(),
            "information_eigenvalues": [float(v) for v in np.linalg.eigvalsh(lam)],
            "information_trace": float(np.trace(lam)),
        }


def ridge_fit(
    features: np.ndarray, theta: np.ndarray, ridge: float
) -> tuple[np.ndarray, np.ndarray]:
    """Ridge regression of ``features`` on ``theta``. Intercept unpenalized.

    Returns ``B`` with shape ``(m, dims)`` and ``c`` with shape ``(m,)``.
    """
    features = np.asarray(features, dtype=float)
    theta = np.asarray(theta, dtype=float)
    theta_mean = theta.mean(axis=0)
    feature_mean = features.mean(axis=0)
    tc = theta - theta_mean
    fc = features - feature_mean
    gram = tc.T @ tc + ridge * np.eye(theta.shape[1])
    coefficients = np.linalg.solve(gram, tc.T @ fc)  # (dims, m)
    loading = coefficients.T
    offset = feature_mean - loading @ theta_mean
    return loading, offset


def fit_noise_covariance(residuals: np.ndarray, n_factors: int) -> np.ndarray:
    """``Psi`` from residuals: diagonal, or diagonal plus a rank-k factor.

    The low-rank form exists because the 24 judgments are 3 answers scored on 8
    poles by the same reader: their residuals are correlated, and a diagonal
    ``Psi`` would read correlated noise as 24 independent measurements and hand
    the posterior information that is not there. Fitted by the standard
    principal-factor iteration on the residual covariance.
    """
    residuals = np.asarray(residuals, dtype=float)
    m = residuals.shape[1]
    cov = residuals.T @ residuals / residuals.shape[0]
    cov = 0.5 * (cov + cov.T)
    floor = _PSI_FLOOR * float(np.mean(np.diag(cov)))
    if n_factors <= 0:
        return np.diag(np.maximum(np.diag(cov), floor))

    unique = np.maximum(np.diag(cov), floor)
    factor = np.zeros((m, n_factors))
    for _ in range(_FACTOR_ITERS):
        reduced = cov - np.diag(unique)
        eigvals, eigvecs = np.linalg.eigh(reduced)
        keep = np.argsort(eigvals)[::-1][:n_factors]
        factor = eigvecs[:, keep] * np.sqrt(np.maximum(eigvals[keep], 0.0))
        new_unique = np.maximum(np.diag(cov) - np.sum(factor**2, axis=1), floor)
        if np.max(np.abs(new_unique - unique)) < 1e-10:
            unique = new_unique
            break
        unique = new_unique
    psi = np.diag(unique) + factor @ factor.T
    return 0.5 * (psi + psi.T)


def open_only_posterior(
    model_loading: np.ndarray,
    model_offset: np.ndarray,
    noise_cov: np.ndarray,
    standardized: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form ``N(0, I)`` posterior given the open answers alone.

    Returns the per-persona mean and the shared posterior precision ``I +
    Lambda``. Used by model selection, where recomputing the full Newton sweep
    for every candidate would say nothing extra: with no items in play the
    posterior is exactly Gaussian.
    """
    weighted = np.linalg.solve(noise_cov, model_loading)
    lam = model_loading.T @ weighted
    lam = 0.5 * (lam + lam.T)
    info = (standardized - model_offset) @ weighted
    precision = np.eye(model_loading.shape[1]) + lam
    mean = np.linalg.solve(precision, info.T).T
    return mean, precision


def predictive_nll(
    theta: np.ndarray, mean: np.ndarray, precision: np.ndarray
) -> float:
    """Mean negative log density of ``theta`` under ``N(mean, precision^-1)``."""
    delta = theta - mean
    sign, logdet = np.linalg.slogdet(precision)
    if sign <= 0:
        return float("inf")
    quad = np.einsum("ni,ij,nj->n", delta, precision, delta)
    dims = theta.shape[1]
    return float(np.mean(0.5 * quad) - 0.5 * logdet + 0.5 * dims * np.log(2 * np.pi))


def fold_assignments(n: int, folds: int, seed: int) -> np.ndarray:
    """Balanced fold labels for ``n`` rows, shuffled with a recorded seed."""
    rng = np.random.default_rng(seed)
    labels = np.arange(n) % folds
    rng.shuffle(labels)
    return labels


def select_measurement_model(
    features: np.ndarray,
    theta: np.ndarray,
    *,
    lambda_grid: Sequence[float] = DEFAULT_LAMBDA_GRID,
    factor_grid: Sequence[int] = DEFAULT_FACTOR_GRID,
    folds: int = DEFAULT_FOLDS,
    seed: int = DEFAULT_CV_SEED,
) -> tuple[MeasurementModel, dict[str, Any]]:
    """Choose the ridge and the ``Psi`` shape by cross-validation, then refit.

    Everything here sees training personas only. The chosen ``B`` and ``c`` are
    refitted on all of them; ``Psi`` is built from the pooled **out-of-fold**
    residuals, so the noise the block declares is noise it actually made on data
    it had not seen.
    """
    features = np.asarray(features, dtype=float)
    theta = np.asarray(theta, dtype=float)
    n, m = features.shape
    center = features.mean(axis=0)
    scale = features.std(axis=0)
    if np.any(scale < 1e-9):
        raise ValueError("a judgment feature is constant across training personas")
    standardized = (features - center) / scale

    labels = fold_assignments(n, folds, seed)
    curve: list[dict[str, Any]] = []
    oof_residuals: dict[float, np.ndarray] = {}

    for ridge in lambda_grid:
        residuals = np.zeros_like(standardized)
        for fold in range(folds):
            train = labels != fold
            test = ~train
            loading, offset = ridge_fit(standardized[train], theta[train], ridge)
            residuals[test] = standardized[test] - (theta[test] @ loading.T + offset)
        oof_residuals[float(ridge)] = residuals

    for ridge in lambda_grid:
        for n_factors in factor_grid:
            total = 0.0
            for fold in range(folds):
                train = labels != fold
                test = ~train
                loading, offset = ridge_fit(standardized[train], theta[train], ridge)
                inner = standardized[train] - (theta[train] @ loading.T + offset)
                psi = fit_noise_covariance(inner, n_factors)
                mean, precision = open_only_posterior(
                    loading, offset, psi, standardized[test]
                )
                total += predictive_nll(theta[test], mean, precision) * int(test.sum())
            curve.append(
                {
                    "ridge_lambda": float(ridge),
                    "n_factors": int(n_factors),
                    "cv_nll": total / n,
                }
            )

    best = min(curve, key=lambda row: row["cv_nll"])
    ridge = float(best["ridge_lambda"])
    n_factors = int(best["n_factors"])

    loading, offset = ridge_fit(standardized, theta, ridge)
    psi = fit_noise_covariance(oof_residuals[ridge], n_factors)
    model = MeasurementModel(
        center=center,
        scale=scale,
        loading=loading,
        offset=offset,
        noise_cov=psi,
        ridge=ridge,
        n_factors=n_factors,
        n_personas=n,
    )

    in_sample = standardized - (theta @ loading.T + offset)
    diagnostics = {
        "criterion": (
            "mean out-of-fold negative log predictive density of theta_hat_full "
            "under the open-answers-only posterior; scores the mean and the "
            "spread together, so an over-confident Psi is punished"
        ),
        "folds": folds,
        "seed": seed,
        "lambda_grid": [float(v) for v in lambda_grid],
        "factor_grid": [int(v) for v in factor_grid],
        "selected": {"ridge_lambda": ridge, "n_factors": n_factors, "cv_nll": best["cv_nll"]},
        "selected_is_grid_edge": bool(
            ridge in (float(lambda_grid[0]), float(lambda_grid[-1]))
            or n_factors == int(max(factor_grid))
        ),
        "cv_curve": curve,
        "psi_source": "pooled out-of-fold residuals at the selected lambda",
        "r2_per_feature_in_sample": [
            float(v)
            for v in 1.0 - in_sample.var(axis=0) / np.maximum(standardized.var(axis=0), _EPS)
        ],
        "r2_per_feature_out_of_fold": [
            float(v)
            for v in 1.0
            - oof_residuals[ridge].var(axis=0) / np.maximum(standardized.var(axis=0), _EPS)
        ],
        "feature_columns": None,  # filled in by the caller, which knows the names
    }
    return model, diagnostics


# --------------------------------------------------------------------------
# how many closed items the open answers are worth
# --------------------------------------------------------------------------


def interpolate_n(curve_n: Sequence[int], curve_value: Sequence[float], target: float) -> dict[str, Any]:
    """Where a decreasing curve of N reaches ``target``, by linear interpolation.

    ``curve_value`` must be the closed-only quantity at ``curve_n``. When the
    target is below everything on the curve the answer is extrapolated from the
    last step and flagged, because "worth more than the whole interview" is a
    real answer and refusing to give it would hide the finding.
    """
    n = np.asarray(curve_n, dtype=float)
    value = np.asarray(curve_value, dtype=float)
    if target >= value[0]:
        return {"equivalent_n": float(n[0]), "extrapolated": False, "note": "at or above N=0"}
    for k in range(len(n) - 1):
        hi, lo = value[k], value[k + 1]
        if lo <= target <= hi:
            if abs(hi - lo) < 1e-15:
                return {"equivalent_n": float(n[k + 1]), "extrapolated": False}
            frac = (hi - target) / (hi - lo)
            return {
                "equivalent_n": float(n[k] + frac * (n[k + 1] - n[k])),
                "extrapolated": False,
            }
    step = value[-2] - value[-1]
    cap = float(n[-1]) * _EXTRAPOLATION_CAP
    if step <= 1e-15:
        return {
            "equivalent_n": None,
            "extrapolated": True,
            "note": "the closed-only curve is flat at its end, so no equivalent N exists",
        }
    extra = (value[-1] - target) / step
    equivalent = float(n[-1] + extra)
    if equivalent > cap:
        return {
            "equivalent_n": None,
            "extrapolated": True,
            "note": (
                f"more than {cap:.0f} closed items -- past the point where "
                "extrapolating the interview's final step means anything"
            ),
        }
    return {
        "equivalent_n": equivalent,
        "extrapolated": True,
        "note": (
            "below the closed-only curve's last point; extrapolated at the rate "
            "of its final step, so this is an estimate, not a measurement"
        ),
    }


def equivalence_table(
    n_grid: Sequence[int],
    closed_curve: Sequence[float],
    fused_curve: Sequence[float],
    *,
    quantity: str,
) -> dict[str, Any]:
    """"The three open answers were worth X closed items", at every N."""
    rows = []
    for k, n in enumerate(n_grid):
        hit = interpolate_n(n_grid, closed_curve, float(fused_curve[k]))
        equivalent = hit["equivalent_n"]
        row = {
            "n": int(n),
            "fused_value": float(fused_curve[k]),
            "closed_only_value": float(closed_curve[k]),
            "equivalent_closed_n": equivalent,
            "worth_in_closed_items": None if equivalent is None else equivalent - float(n),
            "extrapolated": hit["extrapolated"],
        }
        if hit.get("note"):
            row["note"] = hit["note"]
        rows.append(row)
    solid = [r for r in rows if not r["extrapolated"] and r["worth_in_closed_items"] is not None]
    return {
        "quantity": quantity,
        "reading": (
            "the fused posterior at N is as sharp as the closed-only posterior "
            "would be after 'equivalent_closed_n' scripted items; the difference "
            "is what the three open answers bought"
        ),
        "per_n": rows,
        "at_n0": rows[0]["worth_in_closed_items"] if rows else None,
        "at_n15": rows[-1]["worth_in_closed_items"] if rows else None,
        "median_over_measured_n": (
            float(np.median([r["worth_in_closed_items"] for r in solid])) if solid else None
        ),
        "n_extrapolated_rows": int(sum(1 for r in rows if r["extrapolated"])),
    }


# --------------------------------------------------------------------------
# the two-track sweep
# --------------------------------------------------------------------------


def _agreement(estimate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    """:func:`src.model.person_encoder.agreement`, quietly.

    At N = 0 on the closed-only track the estimate is the prior mean for every
    persona, so its correlation with anything is undefined and comes out NaN.
    That is the right answer, not a fault, and it is left in the artifact -- the
    warning is what is suppressed, not the value.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return pe.agreement(estimate, reference)


def track_sweep(
    codes: np.ndarray,
    params: mirt.ItemParams,
    n_grid: Sequence[int],
    block: pe.GaussianBlock | None,
) -> dict[int, pe.Posterior]:
    """Posterior at every N in ``n_grid``, with or without the open-answer block."""
    extra = () if block is None else (block,)
    return pe.encode_prefixes(codes, params, n_grid, extra=extra)


def mean_logdet_precision(posteriors: dict[int, pe.Posterior], n_grid: Sequence[int]) -> np.ndarray:
    """Mean log-determinant of the posterior precision at each N.

    A scale-free measure of total information, and the natural second reading of
    the equivalence question: blur is one diagonal at a time, this is the whole
    matrix.
    """
    out = np.zeros(len(n_grid))
    for k, n in enumerate(n_grid):
        cov = posteriors[int(n)].cov
        signs, logdets = np.linalg.slogdet(cov)
        out[k] = float(np.mean(-logdets[signs > 0]))
    return out


def run(
    transcripts_path: Path,
    features_path: Path,
    fit_path: Path,
    splits_path: Path,
    out_path: Path,
    *,
    estimates_path: Path | None = None,
    calibrate_at: int = 15,
    verbose: bool = True,
) -> dict[str, Any]:
    started = time.time()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if estimates_path is None:
        estimates_path = out_path.with_suffix(".npz")
    estimates_path = Path(estimates_path)

    fit = np.load(fit_path, allow_pickle=False)
    dims = int(fit["dims"])
    params, train_item_ids = pe.item_params_from_fit(fit, block="train")
    column_of = {iid: j for j, iid in enumerate(train_item_ids)}

    transcripts = pe.load_transcripts(transcripts_path)
    splits = json.loads(Path(splits_path).read_text(encoding="utf-8"))
    features = load_open_features(features_path)

    scripted = list(transcripts.item_ids)
    declared = [str(i) for i in splits.get("interview_closed") or []]
    if declared and declared != scripted:
        raise ValueError(
            "the transcripts' scripted order does not match the frozen split manifest"
        )
    missing = [iid for iid in scripted if iid not in column_of]
    if missing:
        raise ValueError(f"scripted items absent from the fitted training items: {missing}")

    interview_params = pe.prefix_params(params, [column_of[iid] for iid in scripted])
    kinds = list(interview_params.kinds)

    train_pids = [str(p) for p in fit["persona_train_ids"]]
    holdout_pids = [str(p) for p in fit["persona_holdout_ids"]]
    split_holdout = {str(p) for p in splits.get("persona_holdout") or []}
    if split_holdout and split_holdout != set(holdout_pids):
        raise ValueError("the fit's held-out personas differ from the split manifest")
    overlap = set(train_pids) & set(holdout_pids)
    if overlap:
        raise ValueError(f"training and held-out personas overlap: {sorted(overlap)[:5]}")

    theta_full_train = np.asarray(fit["theta_train"], dtype=float)
    theta_full_holdout = np.asarray(fit["theta_holdout"], dtype=float)

    raw_train = features.rows_for(train_pids)
    raw_holdout = features.rows_for(holdout_pids)

    # ---- the measurement model, training personas only --------------------
    if verbose:
        print(
            f"fitting y = B theta + c on {len(train_pids)} training personas, "
            f"{raw_train.shape[1]} judgment features ..."
        )
    model, selection = select_measurement_model(raw_train, theta_full_train)
    selection["feature_columns"] = list(features.columns)
    if verbose:
        print(
            f"  selected ridge {model.ridge:.4g}, Psi = "
            f"{'diagonal' if model.n_factors == 0 else f'diagonal + rank-{model.n_factors}'}"
            f", cv nll {selection['selected']['cv_nll']:.4f}"
        )
        eig = np.linalg.eigvalsh(model.precision)
        print(
            "  information added by the open answers: eigenvalues "
            + " ".join(f"{v:.3f}" for v in eig)
        )

    block_train = model.block(raw_train)
    block_holdout = model.block(raw_holdout)

    codes_train = pe.code_matrix(transcripts, train_pids, kinds)
    codes_holdout = pe.code_matrix(transcripts, holdout_pids, kinds)
    n_grid = list(range(1, len(scripted) + 1))
    full_grid = [0] + n_grid

    if verbose:
        print(f"scoring N = 0..{n_grid[-1]}, closed-only and fused ...")

    posts: dict[str, dict[str, dict[int, pe.Posterior]]] = {}
    for track, block in (("closed_only", None), ("fused", block_train)):
        posts.setdefault(track, {})["train"] = track_sweep(
            codes_train, interview_params, full_grid, block
        )
    for track, block in (("closed_only", None), ("fused", block_holdout)):
        posts[track]["holdout"] = track_sweep(
            codes_holdout, interview_params, full_grid, block
        )

    # ---- blur calibration, once per track, training residuals at N=15 -----
    calibrations = {}
    for track in ("closed_only", "fused"):
        post = posts[track]["train"][calibrate_at]
        calibrations[track] = pe.fit_blur_calibration(
            post.theta, post.blur, theta_full_train, at_n=calibrate_at
        )
        if verbose:
            print(
                f"  blur calibration ({track}) at N={calibrate_at}: kappa "
                + " ".join(f"{v:.3f}" for v in calibrations[track].kappa)
            )

    # ---- per-N table ------------------------------------------------------
    mean_blur = {
        track: {
            split: np.stack(
                [posts[track][split][n].blur.mean(axis=0) for n in full_grid]
            )
            for split in ("train", "holdout")
        }
        for track in ("closed_only", "fused")
    }

    per_n: list[dict[str, Any]] = []
    for k, n in enumerate(full_grid):
        row: dict[str, Any] = {"n": int(n), "item_ids": scripted[:n], "item_kinds": kinds[:n]}
        for track in ("closed_only", "fused"):
            ho = posts[track]["holdout"][n]
            tr = posts[track]["train"][n]
            calibrated = calibrations[track].apply(ho.blur)
            row[track] = {
                "holdout": {
                    "mean_blur_per_dim": [float(v) for v in mean_blur[track]["holdout"][k]],
                    "mean_blur": float(mean_blur[track]["holdout"][k].mean()),
                    "mean_blur_calibrated_per_dim": [
                        float(v) for v in calibrated.mean(axis=0)
                    ],
                    "mean_blur_calibrated": float(calibrated.mean()),
                    "agreement_with_full_matrix": _agreement(ho.theta, theta_full_holdout),
                    "n_non_pd_hessians": len(ho.non_pd_rows),
                    "newton_iters": ho.n_iters,
                    "max_gradient": ho.max_grad,
                },
                "train": {
                    "mean_blur_per_dim": [float(v) for v in mean_blur[track]["train"][k]],
                    "mean_blur": float(mean_blur[track]["train"][k].mean()),
                    "agreement_with_full_matrix": _agreement(tr.theta, theta_full_train),
                    "n_non_pd_hessians": len(tr.non_pd_rows),
                },
            }
        per_n.append(row)
        if verbose:
            closed = row["closed_only"]["holdout"]
            fused = row["fused"]["holdout"]
            print(
                f"  N={n:2d}  blur {closed['mean_blur']:.4f} -> {fused['mean_blur']:.4f}"
                f"   r {closed['agreement_with_full_matrix']['r_mean']:.4f}"
                f" -> {fused['agreement_with_full_matrix']['r_mean']:.4f}"
            )

    # ---- monotonicity, both tracks, on the N >= 1 grid the backbone used ---
    monotonicity = {}
    for track in ("closed_only", "fused"):
        monotonicity[track] = {
            "holdout": pe.monotonicity_report(
                n_grid,
                mean_blur[track]["holdout"][1:],
                blur_by_n=np.stack([posts[track]["holdout"][n].blur for n in n_grid]),
            ),
            "train": pe.monotonicity_report(n_grid, mean_blur[track]["train"][1:]),
        }
    fixed_blur = pe.blur_at_fixed_theta(
        codes_holdout,
        interview_params,
        posts["fused"]["holdout"][n_grid[-1]].theta,
        n_grid,
        extra=(block_holdout,),
    )
    monotonicity["fused"]["holdout_at_fixed_theta"] = pe.monotonicity_report(
        n_grid, fixed_blur.mean(axis=1), blur_by_n=fixed_blur
    )

    # ---- what the open answers were worth, in closed items ----------------
    closed_blur_curve = mean_blur["closed_only"]["holdout"].mean(axis=1)
    fused_blur_curve = mean_blur["fused"]["holdout"].mean(axis=1)
    logdet_closed = mean_logdet_precision(posts["closed_only"]["holdout"], full_grid)
    logdet_fused = mean_logdet_precision(posts["fused"]["holdout"], full_grid)
    equivalence = {
        "by_mean_blur": equivalence_table(
            full_grid, closed_blur_curve, fused_blur_curve,
            quantity="mean posterior blur over held-out personas and dimensions",
        ),
        "by_log_determinant": equivalence_table(
            full_grid, -logdet_closed, -logdet_fused,
            quantity="negative mean log-determinant of the posterior precision",
        ),
        "per_dimension_by_blur": {},
        "note": (
            "blur matching is the number the design promised; the "
            "log-determinant version is the same question asked of the whole "
            "information matrix instead of one diagonal at a time"
        ),
    }
    for d in range(dims):
        equivalence["per_dimension_by_blur"][f"dim_{d}"] = equivalence_table(
            full_grid,
            mean_blur["closed_only"]["holdout"][:, d],
            mean_blur["fused"]["holdout"][:, d],
            quantity=f"mean posterior blur, dimension {d}",
        )["at_n15"]

    if verbose:
        head = equivalence["by_mean_blur"]

        def _fmt(value: float | None) -> str:
            return "more than the interview" if value is None else f"{value:.1f}"

        print(
            f"\nthe three open answers are worth about "
            f"{_fmt(head['at_n0'])} closed items at N=0 and "
            f"{_fmt(head['at_n15'])} at N=15 (blur matching"
            + ("; extrapolated" if head["n_extrapolated_rows"] else "")
            + ")"
        )

    # ---- write ------------------------------------------------------------
    payload: dict[str, np.ndarray] = {
        "n_grid": np.array(n_grid),
        "full_grid": np.array(full_grid),
        "dims": np.array(dims),
        "scripted_item_ids": np.array(scripted),
        "persona_train_ids": np.array(train_pids),
        "persona_holdout_ids": np.array(holdout_pids),
        "measurement_B": model.loading,
        "measurement_c": model.offset,
        "measurement_psi": model.noise_cov,
        "measurement_center": model.center,
        "measurement_scale": model.scale,
        "open_information_lambda": model.precision,
    }
    for track in ("closed_only", "fused"):
        calib = calibrations[track]
        payload[f"blur_kappa_{track}"] = calib.kappa
        for split, pids in (("train", train_pids), ("holdout", holdout_pids)):
            grid_posts = posts[track][split]
            payload[f"theta_{split}_{track}_by_n"] = np.stack(
                [grid_posts[n].theta for n in n_grid]
            )
            payload[f"blur_{split}_{track}_by_n"] = np.stack(
                [grid_posts[n].blur for n in n_grid]
            )
            payload[f"theta_{split}_{track}_n0"] = grid_posts[0].theta
            payload[f"blur_{split}_{track}_n0"] = grid_posts[0].blur
        payload[f"cov_holdout_{track}_by_n"] = np.stack(
            [posts[track]["holdout"][n].cov for n in n_grid]
        )
        payload[f"cov_holdout_{track}_n0"] = posts[track]["holdout"][0].cov
        payload[f"blur_holdout_{track}_calibrated_by_n"] = np.stack(
            [calib.apply(posts[track]["holdout"][n].blur) for n in n_grid]
        )
        payload[f"blur_holdout_{track}_calibrated_n0"] = calib.apply(
            posts[track]["holdout"][0].blur
        )
    np.savez_compressed(estimates_path, **payload)

    report = {
        "schema": SCHEMA,
        "module": "src.model.open_fusion",
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {
            "transcripts": str(transcripts_path),
            "open_answer_features": str(features_path),
            "fit": str(fit_path),
            "splits": str(splits_path),
            "feature_coverage": features.coverage,
            "open_prompts": list(features.prompt_ids),
            "poles": list(features.poles),
        },
        "config": {
            "dims": dims,
            "n_grid": n_grid,
            "full_grid": full_grid,
            "tracks": ["closed_only", "fused"],
            "prior": "theta ~ N(0, I), the prior the Stage 2 fit uses",
            "item_parameters": "fitted TRAINING items, linked metric, frozen",
            "fusion": (
                "one Gaussian block per persona, y = B theta + c + eps, added to "
                "the negative log posterior at every N -- the open answers are "
                "given up front, so they are present at every prefix length"
            ),
            "open_answers_used": True,
            "llm_used": False,
            "llm_used_upstream": (
                "the 24 judgment features were produced by the frozen system-side "
                "model reading the persona's own open answers; no model runs here"
            ),
        },
        "interview": {
            "scripted_closed": scripted,
            "item_kinds": kinds,
            "all_from_training_items": True,
        },
        "splits": {
            "n_train_personas": len(train_pids),
            "n_holdout_personas": len(holdout_pids),
            "source": str(splits_path),
            "train_holdout_overlap": 0,
        },
        "measurement_model": model.to_json(),
        "model_selection": selection,
        "blur_calibration": {
            track: calibrations[track].to_json() for track in ("closed_only", "fused")
        },
        "per_n": per_n,
        "monotonicity": {
            "bar": (
                "PREREGISTRATION Gate 3: blur shrinks monotonically in expectation "
                "as N grows. Graded on the mean over held-out personas, per "
                "dimension, at zero tolerance; the persona-level block is the "
                "strict per-step reading."
            ),
            **monotonicity,
            "reading": (
                "the open-answer block adds a positive semi-definite term that "
                "does not depend on N, so it cannot break monotonicity; at a "
                "fixed theta the information matrices still nest and the curve is "
                "monotone by construction. Any rise in the real sweep is the MAP "
                "moving between prefixes, not information being lost."
            ),
        },
        "open_answers_worth_in_closed_items": equivalence,
        "reference": (
            "every agreement number here is against the Stage 2 fit's own "
            "theta_hat from the full training-item matrix. Nothing in this "
            "artifact is compared to planted truth."
        ),
        "artifacts": {"estimates": str(estimates_path)},
        "runtime_seconds": time.time() - started,
    }
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if verbose:
        print(f"\nreport    -> {out_path}")
        print(f"estimates -> {estimates_path}")
        print(f"total runtime {report['runtime_seconds']:.1f}s")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.model.open_fusion",
        description=(
            "Fuse the three open-answer judgments into the person-encoder "
            "posterior as one Gaussian pseudo-item block, and sweep N."
        ),
    )
    parser.add_argument("--transcripts", type=Path, required=True, help="transcripts jsonl")
    parser.add_argument("--features", type=Path, required=True, help="open-answer feature npz")
    parser.add_argument("--fit", type=Path, required=True, help="Stage 2 fit npz")
    parser.add_argument("--splits", type=Path, required=True, help="frozen splits manifest")
    parser.add_argument("--out", type=Path, required=True, help="report json to write")
    parser.add_argument("--estimates", type=Path, default=None, help="npz to write")
    parser.add_argument("--calibrate-at", type=int, default=15)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    run(
        args.transcripts,
        args.features,
        args.fit,
        args.splits,
        args.out,
        estimates_path=args.estimates,
        calibrate_at=args.calibrate_at,
        verbose=not args.quiet,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
