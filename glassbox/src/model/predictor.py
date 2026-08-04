"""Stage 4 predictor: theta_hat + item parameters -> a distribution over answers.

    python -m src.model.predictor build \\
        --fit results/stage2_v2_fit.npz \\
        --fused results/stage3_person_fused.npz \\
        --item-predictions data/runs/stage3_qwen/item_encoder_predictions.npz \\
        --answers data/runs/full_sweep_v2/answers_noised.jsonl \\
        --out results/stage4_predictions.npz

PREREGISTRATION.md section 5: "Predictor: ordinal logit (graded response form)
combining theta_hat with item loadings. No LLM anywhere in the predictor."

The response function is the same one ``src.model.mirt`` fits, restated here as
a forward pass rather than re-derived: with ``z = a_i . theta_n``,

  * Likert-5, graded response, four ordered thresholds tau_i1 < ... < tau_i4:
        P(Y >= k) = sigmoid(z - tau_i,k-1) for k = 2..5
        P(Y = k)  = P(Y >= k) - P(Y >= k+1)
  * binary, 2PL:
        P(yes) = sigmoid(z - b_i)

``tests/test_predictor.py`` proves the agreement numerically on random
parameters against ``mirt``'s own likelihood code, so the two can never drift.

The confirmatory path
---------------------
theta_hat
    the fused person-encoder posterior at N = 15 for the 100 held-out personas
    (``results/stage3_person_fused.npz``): posterior mean plus the CALIBRATED
    covariance ``K Sigma K``, ``K = diag(kappa_fused)``. That is the same track
    that produced the Gate 3 coverage of 0.649, and the kappa was fitted on
    training-split residuals only, so it is a system-side quantity.

loadings
    ZERO-SHOT item-encoder predictions for the 50 probe items, direction head
    times magnitude head. Fitted loadings for a held-out item are never used in
    the confirmatory path -- that is the whole point of the item encoder.

intercepts / thresholds
    CONFIRMATORY DESIGN CHOICE, recorded verbatim from the orchestrator's
    instruction:

        "Item intercepts/thresholds for the 50 probe items (CONFIRMATORY design
        choice by the orchestrator): fit per-item by maximum likelihood on the
        400 TRAINING personas' noised main-round answers on that item, with
        training-persona theta FIXED at stage2_v2_fit.npz theta_train and
        loadings FROZEN at the encoder predictions; optimize
        intercepts/thresholds only. Rationale: threshold information is exactly
        what the population-marginal baseline already sees; person-specific
        signal stays with the zero-shot loadings and theta-hat."

    A fully-zero-shot-item variant -- thresholds predicted from item text by a
    ridge head -- is built too and is labelled EXPLORATORY everywhere.

predictive distribution
    the posterior is integrated over, not collapsed: 512 seeded draws
    theta ~ N(theta_hat, Sigma_calibrated), shared across items within a
    persona, and the category probabilities are averaged. The plug-in variant
    at the posterior mean is computed alongside as an exploratory comparison.

WALL. System-side module. It reads fitted parameters, encoder predictions and
noised answers, and nothing else. No planted value, no persona card, no answer
distribution.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import mirt

REPO_ROOT = Path(__file__).resolve().parents[2]

LIKERT = mirt.LIKERT
BINARY = mirt.BINARY

#: Widest category count in the bank. Binary items use columns 0 (no) and 1 (yes).
MAX_CATEGORIES = mirt.LIKERT_CATEGORIES

#: Monte-Carlo draws per persona when integrating over the posterior.
DEFAULT_DRAWS = 512

#: Seed for the posterior draws. Recorded, never tuned.
DEFAULT_SEED = 2026

#: The confirmatory interview length (PREREGISTRATION.md section 5).
CONFIRMATORY_N = 15

#: The confirmatory person-encoder track (Gate 3's confirmatory arm).
CONFIRMATORY_TRACK = "fused"

SCHEMA = "glassbox.stage4.predictor/1"

#: The orchestrator's confirmatory ruling on where thresholds come from, stored
#: verbatim in the output so the choice travels with the numbers.
THRESHOLD_RULING = (
    "Item intercepts/thresholds for the 50 probe items (CONFIRMATORY design "
    "choice by the orchestrator): fit per-item by maximum likelihood on the 400 "
    "TRAINING personas' noised main-round answers on that item, with "
    "training-persona theta FIXED at stage2_v2_fit.npz theta_train and loadings "
    "FROZEN at the encoder predictions; optimize intercepts/thresholds only. "
    "Rationale: threshold information is exactly what the population-marginal "
    "baseline already sees; person-specific signal stays with the zero-shot "
    "loadings and theta-hat."
)


# --------------------------------------------------------------------------
# item parameters
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ItemBlock:
    """One set of item parameters, in the form the MIRT artifacts store them.

    ``thresholds`` is (n_items, 4) and is NaN on binary rows; ``difficulty`` is
    (n_items,) and is NaN on Likert rows. That is exactly the layout of
    ``stage2_v2_fit.npz``, so a fitted block loads with no translation.
    """

    item_ids: list[str]
    kinds: list[str]
    loadings: np.ndarray
    thresholds: np.ndarray
    difficulty: np.ndarray

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    @property
    def is_likert(self) -> np.ndarray:
        return np.array([k == LIKERT for k in self.kinds], dtype=bool)

    def n_categories(self) -> np.ndarray:
        return np.where(self.is_likert, MAX_CATEGORIES, 2)

    def cuts(self) -> np.ndarray:
        """(n_items, 4) cut points: the thresholds, or the difficulty in column 0."""
        out = np.zeros((self.n_items, mirt.LIKERT_CUTS), dtype=float)
        likert = self.is_likert
        out[likert] = self.thresholds[likert]
        out[~likert, 0] = self.difficulty[~likert]
        return out

    def subset(self, item_ids: Sequence[str]) -> "ItemBlock":
        index = {iid: i for i, iid in enumerate(self.item_ids)}
        rows = [index[iid] for iid in item_ids]
        return ItemBlock(
            item_ids=list(item_ids),
            kinds=[self.kinds[r] for r in rows],
            loadings=self.loadings[rows],
            thresholds=self.thresholds[rows],
            difficulty=self.difficulty[rows],
        )

    def with_loadings(self, loadings: np.ndarray) -> "ItemBlock":
        return ItemBlock(
            item_ids=list(self.item_ids),
            kinds=list(self.kinds),
            loadings=np.asarray(loadings, dtype=float),
            thresholds=self.thresholds.copy(),
            difficulty=self.difficulty.copy(),
        )

    def to_mirt_params(self) -> mirt.ItemParams:
        """The same block in ``mirt``'s own unconstrained parameterization."""
        return mirt.ItemParams(
            loadings=self.loadings.copy(),
            cut_raw=cut_raw_from_cuts(self.cuts(), self.is_likert),
            kinds=list(self.kinds),
        )


def cut_raw_from_cuts(cuts: np.ndarray, is_likert: np.ndarray) -> np.ndarray:
    """Invert ``mirt.ItemParams.thresholds``: ordered taus -> the free vector.

    ``tau_1 = cut_raw[0]`` and ``tau_j = tau_{j-1} + softplus(cut_raw[j])``, so
    the gaps go back through ``inv_softplus``. Binary rows carry the difficulty
    in column 0 and zeros elsewhere, which is what ``mirt`` reads off them.
    """
    cuts = np.asarray(cuts, dtype=float)
    out = np.zeros_like(cuts)
    out[:, 0] = cuts[:, 0]
    gaps = np.diff(cuts, axis=1)
    likert = np.asarray(is_likert, dtype=bool)
    if likert.any():
        out[likert, 1:] = mirt.inv_softplus(np.maximum(gaps[likert], 1e-9))
    return out


def block_from_fit(archive: Any, which: str) -> ItemBlock:
    """A block from a ``mirt`` artifact: ``which`` is ``train`` or ``holdout``."""
    prefix = "train" if which == "train" else "holdout"
    ids_key = "item_train_ids" if which == "train" else "item_holdout_ids"
    return ItemBlock(
        item_ids=[str(x) for x in archive[ids_key]],
        kinds=[str(k) for k in archive[f"{prefix}_item_kinds"]],
        loadings=np.asarray(archive[f"{prefix}_loadings"], dtype=float),
        thresholds=np.asarray(archive[f"{prefix}_thresholds"], dtype=float),
        difficulty=np.asarray(archive[f"{prefix}_difficulty"], dtype=float),
    )


# --------------------------------------------------------------------------
# the response function
# --------------------------------------------------------------------------


def category_probabilities(theta: np.ndarray, block: ItemBlock) -> np.ndarray:
    """(n_people, n_items, 5) category probabilities under the graded response.

    Columns past an item's category count are exactly zero: a binary item has
    P(no) in column 0, P(yes) in column 1 and nothing else.
    """
    theta = np.atleast_2d(np.asarray(theta, dtype=float))
    z = theta @ block.loadings.T  # (n_people, n_items)
    cuts = block.cuts()
    likert = block.is_likert

    out = np.zeros((z.shape[0], block.n_items, MAX_CATEGORIES), dtype=float)

    if likert.any():
        zl = z[:, likert][:, :, None]
        taus = cuts[likert][None, :, :]
        seen = mirt.sigmoid(zl - taus)  # P(Y >= k), k = 2..5
        ones = np.ones((z.shape[0], int(likert.sum()), 1))
        zeros = np.zeros_like(ones)
        cumulative = np.concatenate([ones, seen, zeros], axis=2)  # (n, i, 6)
        out[:, likert, :] = cumulative[:, :, :-1] - cumulative[:, :, 1:]

    if (~likert).any():
        p_yes = mirt.sigmoid(z[:, ~likert] - cuts[~likert, 0][None, :])
        out[:, ~likert, 0] = 1.0 - p_yes
        out[:, ~likert, 1] = p_yes

    return out


# --------------------------------------------------------------------------
# fitting intercepts / thresholds with theta and loadings frozen
# --------------------------------------------------------------------------


@dataclass
class CutFit:
    """The result of the confirmatory threshold fit."""

    cuts: np.ndarray  # (n_items, 4): thresholds, or difficulty in column 0
    thresholds: np.ndarray
    difficulty: np.ndarray
    loglik: float
    iters: int
    converged: bool
    max_abs_gradient: float
    loss_curve: list[float]


def fit_cuts(
    codes: np.ndarray,
    theta: np.ndarray,
    block: ItemBlock,
    *,
    lr: float = 0.05,
    max_iters: int = 6000,
    tol: float = 1e-10,
    patience: int = 50,
) -> CutFit:
    """Maximum-likelihood cut points, with theta and the loadings held fixed.

    ``codes`` is (n_people, n_items) with -1 for a missing cell. Everything
    except ``cut_raw`` is frozen, and the optimizer is the same full-batch Adam
    ``mirt`` uses, over ``mirt``'s own gradient -- so this is the same estimator
    that fit the training items, restricted to the thresholds.
    """
    data = mirt.Dataset(
        persona_ids=[str(i) for i in range(codes.shape[0])],
        item_ids=list(block.item_ids),
        item_kinds=list(block.kinds),
        codes=np.asarray(codes, dtype=np.int16),
    )
    blocks = mirt.make_blocks(data)
    params = mirt.init_item_params(data, block.loadings.shape[1], np.random.default_rng(0))
    params.loadings = np.asarray(block.loadings, dtype=float).copy()

    opt = mirt._Adam([params.cut_raw.shape], lr)
    curve: list[float] = []
    best = np.inf
    stale = 0
    converged = False
    loglik = float("nan")
    grad_cut = np.zeros_like(params.cut_raw)

    for _ in range(max_iters):
        loglik, _, _, grad_cut = mirt._loglik_and_grads(theta, params, blocks)
        loss = -loglik
        curve.append(loss)
        opt.step([params.cut_raw], [-grad_cut])
        if mirt._relative_gain(best, loss) < tol:
            stale += 1
            if stale >= patience:
                converged = True
                break
        else:
            stale = 0
        best = min(best, loss)

    loglik, _, _, grad_cut = mirt._loglik_and_grads(theta, params, blocks)
    likert = block.is_likert
    live = np.zeros_like(grad_cut, dtype=bool)
    live[likert, :] = True
    live[~likert, 0] = True

    taus = params.thresholds()
    cuts = np.zeros_like(params.cut_raw)
    cuts[likert] = taus[likert]
    cuts[~likert, 0] = params.difficulty()[~likert]
    return CutFit(
        cuts=cuts,
        thresholds=np.where(likert[:, None], taus, np.nan),
        difficulty=np.where(likert, np.nan, params.difficulty()),
        loglik=float(loglik),
        iters=len(curve),
        converged=converged,
        max_abs_gradient=float(np.max(np.abs(grad_cut[live]))),
        loss_curve=[float(v) for v in curve[::200][:40]] + [float(curve[-1])],
    )


# --------------------------------------------------------------------------
# integrating over the posterior
# --------------------------------------------------------------------------


def calibrated_covariance(cov: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    """``K Sigma K`` with ``K = diag(kappa)`` -- the blur calibration on the full matrix.

    The person encoder publishes one scalar per dimension, fitted on training
    residuals. Applied this way it reproduces the artifact's calibrated blur on
    the diagonal and keeps the off-diagonal structure, which the Monte Carlo
    needs because a draw has to respect the correlations between dimensions.
    """
    k = np.diag(np.asarray(kappa, dtype=float))
    return np.einsum("ij,njk,kl->nil", k, np.asarray(cov, dtype=float), k)


def _sqrt_psd(matrix: np.ndarray, floor: float = 1e-10) -> np.ndarray:
    """A square root of a symmetric matrix, eigenvalues clipped up to ``floor``."""
    sym = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(sym)
    return vectors @ np.diag(np.sqrt(np.maximum(values, floor)))


def predict_posterior(
    theta_hat: np.ndarray,
    cov: np.ndarray,
    block: ItemBlock,
    *,
    n_draws: int = DEFAULT_DRAWS,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """(n_people, n_items, 5), averaging the response function over the posterior.

    One person's draws are shared across all items, which is what makes the
    output a coherent predictive distribution for that person rather than an
    item-by-item smoothing.
    """
    theta_hat = np.asarray(theta_hat, dtype=float)
    cov = np.asarray(cov, dtype=float)
    out = np.zeros((theta_hat.shape[0], block.n_items, MAX_CATEGORIES), dtype=float)
    for row in range(theta_hat.shape[0]):
        rng = np.random.default_rng([seed, row])
        draws = theta_hat[row][None, :] + rng.standard_normal(
            (n_draws, theta_hat.shape[1])
        ) @ _sqrt_psd(cov[row]).T
        out[row] = category_probabilities(draws, block).mean(axis=0)
    return out


def predict_plug_in(theta_hat: np.ndarray, block: ItemBlock) -> np.ndarray:
    """(n_people, n_items, 5) at the posterior mean -- no uncertainty carried."""
    return category_probabilities(theta_hat, block)


# --------------------------------------------------------------------------
# EXPLORATORY: thresholds from item text
# --------------------------------------------------------------------------


def _standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return mean, scale


def _design(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    z = (x - mean) / scale
    return np.hstack([z, np.ones((z.shape[0], 1))])


def ridge_multi_fit(x: np.ndarray, y: np.ndarray, lam: float) -> dict[str, Any]:
    mean, scale = _standardize(x)
    d = _design(x, mean, scale)
    penalty = np.eye(d.shape[1]) * lam
    penalty[-1, -1] = 0.0
    weights = np.linalg.solve(d.T @ d + penalty, d.T @ y)
    return {"mean": mean, "scale": scale, "W": weights, "lam": float(lam)}


def ridge_multi_predict(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    return _design(x, model["mean"], model["scale"]) @ model["W"]


def ridge_multi_loo(x: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Exact leave-one-out predictions, the hat-matrix identity as in the item encoder."""
    mean, scale = _standardize(x)
    d = _design(x, mean, scale)
    penalty = np.eye(d.shape[1]) * lam
    penalty[-1, -1] = 0.0
    g = np.linalg.solve(d.T @ d + penalty, d.T)
    h = d @ g
    leverage = np.clip(np.diag(h), 0.0, 1.0 - 1e-9)
    return y - (y - h @ y) / (1.0 - leverage)[:, None]


THRESHOLD_LAMBDA_GRID = tuple(float(10.0**e) for e in np.arange(-4.0, 6.01, 0.25))


def train_threshold_head(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Ridge from item features onto fitted cut points, lambda by leave-one-out."""
    curve = []
    for lam in THRESHOLD_LAMBDA_GRID:
        pred = ridge_multi_loo(x, y, lam)
        curve.append({"lambda": lam, "loo_mse": float(np.mean((pred - y) ** 2))})
    best = min(curve, key=lambda row: row["loo_mse"])
    model = ridge_multi_fit(x, y, best["lambda"])
    model["loo_curve"] = curve
    model["loo_mse"] = best["loo_mse"]
    return model


# --------------------------------------------------------------------------
# loading the pieces
# --------------------------------------------------------------------------


def load_codes(
    answers_path: Path,
    persona_ids: Sequence[str],
    item_ids: Sequence[str],
    kinds: Sequence[str],
    *,
    round_name: str = "main",
) -> np.ndarray:
    """(n_people, n_items) coded answers, -1 where a cell is missing.

    Only the ``answer`` field is read, exactly as ``mirt.load_answers`` does.
    """
    prow = {pid: i for i, pid in enumerate(persona_ids)}
    icol = {iid: j for j, iid in enumerate(item_ids)}
    kind_of = dict(zip(item_ids, kinds))
    codes = np.full((len(persona_ids), len(item_ids)), -1, dtype=np.int16)
    with Path(answers_path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("round") != round_name:
                continue
            row = prow.get(str(record.get("pid")))
            col = icol.get(str(record.get("item_id")))
            if row is None or col is None:
                continue
            codes[row, col] = mirt._code_answer(record["answer"], kind_of[item_ids[col]])
    return codes


def load_item_texts(bank_path: Path) -> tuple[dict[str, dict], list[str]]:
    payload = json.loads(Path(bank_path).read_text(encoding="utf-8"))
    return {item["item_id"]: item for item in payload["items"]}, list(
        payload["topic_domains"]
    )


# --------------------------------------------------------------------------
# the build
# --------------------------------------------------------------------------


def build(
    fit_path: Path,
    fused_path: Path,
    item_predictions_path: Path,
    answers_path: Path,
    bank_path: Path,
    out_path: Path,
    *,
    embeddings_path: Path | None = None,
    n_draws: int = DEFAULT_DRAWS,
    seed: int = DEFAULT_SEED,
    verbose: bool = True,
) -> dict[str, Any]:
    started = time.time()
    fit = np.load(fit_path, allow_pickle=False)
    fused = np.load(fused_path, allow_pickle=False)
    encoded = np.load(item_predictions_path, allow_pickle=False)

    holdout_pids = [str(p) for p in fit["persona_holdout_ids"]]
    train_pids = [str(p) for p in fit["persona_train_ids"]]
    probe_ids = [str(i) for i in fit["item_holdout_ids"]]

    fitted_probe = block_from_fit(fit, "holdout")
    fitted_train_items = block_from_fit(fit, "train")

    # ---- zero-shot loadings for the probe items --------------------------
    encoder_ids = [str(i) for i in encoded["item_ids"]]
    if encoder_ids != probe_ids:
        raise ValueError(
            "the item-encoder predictions are not in the probe-item order of the fit"
        )
    encoder_loadings = np.asarray(encoded["predicted_loadings"], dtype=float)

    # ---- theta_hat -------------------------------------------------------
    grid = [int(v) for v in fused["n_grid"]]
    at = grid.index(CONFIRMATORY_N)
    theta_fused = np.asarray(
        fused[f"theta_holdout_{CONFIRMATORY_TRACK}_by_n"][at], dtype=float
    )
    cov_raw = np.asarray(fused[f"cov_holdout_{CONFIRMATORY_TRACK}_by_n"][at], dtype=float)
    kappa = np.asarray(fused[f"blur_kappa_{CONFIRMATORY_TRACK}"], dtype=float)
    cov_fused = calibrated_covariance(cov_raw, kappa)
    theta_full = np.asarray(fit["theta_holdout"], dtype=float)
    cov_full = np.asarray(fit["cov_holdout"], dtype=float)
    theta_train = np.asarray(fit["theta_train"], dtype=float)

    # a cheap integrity check on the calibration: the diagonal of K Sigma K has
    # to be the calibrated blur the person encoder already published.
    calibrated_blur_matches = bool(
        np.allclose(
            np.sqrt(np.diagonal(cov_fused, axis1=1, axis2=2)),
            np.asarray(fused[f"blur_holdout_{CONFIRMATORY_TRACK}_calibrated_by_n"][at]),
        )
    )

    # ---- the confirmatory thresholds -------------------------------------
    # THRESHOLD_RULING, above, in code.
    codes_train_probe = load_codes(
        answers_path, train_pids, probe_ids, fitted_probe.kinds
    )
    if int((codes_train_probe < 0).sum()):
        raise ValueError("the training-persona probe answer matrix has holes in it")

    encoder_block_seed = fitted_probe.with_loadings(encoder_loadings)
    if verbose:
        print("fitting probe thresholds on the 400 training personas ...")
    cut_fit = fit_cuts(codes_train_probe, theta_train, encoder_block_seed)
    trained_block = ItemBlock(
        item_ids=probe_ids,
        kinds=list(fitted_probe.kinds),
        loadings=encoder_loadings,
        thresholds=cut_fit.thresholds,
        difficulty=cut_fit.difficulty,
    )
    if verbose:
        print(
            f"  loglik {cut_fit.loglik:,.2f} after {cut_fit.iters} iters "
            f"(converged={cut_fit.converged}, |grad|max {cut_fit.max_abs_gradient:.2e})"
        )

    # ---- EXPLORATORY: thresholds from item text --------------------------
    text_block, text_head_report = _text_threshold_block(
        fitted_train_items,
        fitted_probe,
        encoder_loadings,
        bank_path,
        embeddings_path,
        verbose=verbose,
    )

    # ---- predictions -----------------------------------------------------
    if verbose:
        print(f"integrating over the posterior, {n_draws} draws per persona ...")
    arms: dict[str, np.ndarray] = {
        # confirmatory
        "model": predict_posterior(
            theta_fused, cov_fused, trained_block, n_draws=n_draws, seed=seed
        ),
        # exploratory: the same predictor without the posterior integral
        "model_plugin": predict_plug_in(theta_fused, trained_block),
        # exploratory: fully zero-shot item side
        "model_text_thresholds": predict_posterior(
            theta_fused, cov_fused, text_block, n_draws=n_draws, seed=seed
        ),
        # exploratory 2x2 decomposition, plug-in throughout so that the only
        # things varying across the four cells are theta_hat and the item block
        "dec_fused_encoder": predict_plug_in(theta_fused, trained_block),
        "dec_fused_fitted": predict_plug_in(theta_fused, fitted_probe),
        "dec_full_encoder": predict_plug_in(theta_full, trained_block),
        "dec_full_fitted": predict_plug_in(theta_full, fitted_probe),
        # exploratory: the Gate-2 full-matrix theta with its own posterior
        "full_matrix_posterior": predict_posterior(
            theta_full, cov_full, fitted_probe, n_draws=n_draws, seed=seed
        ),
    }

    payload: dict[str, Any] = {
        "persona_ids": np.array(holdout_pids),
        "item_ids": np.array(probe_ids),
        "item_kinds": np.array(fitted_probe.kinds),
        "n_categories": trained_block.n_categories(),
        "encoder_loadings": encoder_loadings,
        "trained_cuts": cut_fit.cuts,
        "text_cuts": text_block.cuts(),
        "fitted_cuts": fitted_probe.cuts(),
        "theta_fused_n15": theta_fused,
        "cov_fused_n15_calibrated": cov_fused,
        "theta_full_matrix": theta_full,
    }
    for name, probs in arms.items():
        payload[f"probs_{name}"] = probs

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)

    report = {
        "schema": SCHEMA,
        "module": "src.model.predictor",
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "response_function": (
            "graded response for Likert-5, 2PL for binary -- identical to "
            "src.model.mirt, proven numerically in tests/test_predictor.py"
        ),
        "confirmatory_arm": {
            "name": "model",
            "theta_hat": (
                f"fused person-encoder posterior at N = {CONFIRMATORY_N}, "
                "posterior mean + calibrated covariance K Sigma K"
            ),
            "loadings": "zero-shot item encoder (direction head x magnitude head)",
            "thresholds": "trained on the 400 training personas, see threshold_ruling",
            "integration": f"Monte Carlo, {n_draws} seeded draws per persona, seed {seed}",
        },
        "threshold_ruling": THRESHOLD_RULING,
        "threshold_fit": {
            "n_training_personas": len(train_pids),
            "n_items": len(probe_ids),
            "loglik": cut_fit.loglik,
            "iters": cut_fit.iters,
            "converged": cut_fit.converged,
            "max_abs_gradient_at_optimum": cut_fit.max_abs_gradient,
            "loss_curve_sampled": cut_fit.loss_curve,
        },
        "exploratory_text_threshold_head": text_head_report,
        "arms": {
            "model": "CONFIRMATORY",
            "model_plugin": "EXPLORATORY -- posterior collapsed to its mean",
            "model_text_thresholds": "EXPLORATORY -- item side fully zero-shot",
            "dec_fused_encoder": "EXPLORATORY -- 2x2 decomposition cell (plug-in)",
            "dec_fused_fitted": "EXPLORATORY -- 2x2 decomposition cell (plug-in)",
            "dec_full_encoder": "EXPLORATORY -- 2x2 decomposition cell (plug-in)",
            "dec_full_fitted": "EXPLORATORY -- 2x2 decomposition cell (plug-in)",
            "full_matrix_posterior": "EXPLORATORY -- Gate 2 theta with its own posterior",
        },
        "integrity": {
            "calibrated_covariance_diagonal_matches_published_blur": calibrated_blur_matches,
            "encoder_prediction_order_matches_probe_items": True,
            "kappa_per_dim": [float(v) for v in kappa],
        },
        "counts": {
            "holdout_personas": len(holdout_pids),
            "train_personas": len(train_pids),
            "probe_items": len(probe_ids),
            "n_draws": n_draws,
            "seed": seed,
        },
        "inputs": {
            "fit": str(fit_path),
            "fused": str(fused_path),
            "item_predictions": str(item_predictions_path),
            "answers": str(answers_path),
            "bank": str(bank_path),
            "embeddings": str(embeddings_path) if embeddings_path else None,
        },
        "artifacts": {"predictions": str(out_path)},
        "runtime_seconds": time.time() - started,
    }
    report_path = out_path.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if verbose:
        print(f"predictions -> {out_path}")
        print(f"report      -> {report_path}")
    return report


def _text_threshold_block(
    train_items: ItemBlock,
    probe_items: ItemBlock,
    encoder_loadings: np.ndarray,
    bank_path: Path,
    embeddings_path: Path | None,
    *,
    verbose: bool = True,
) -> tuple[ItemBlock, dict[str, Any]]:
    """EXPLORATORY. Cut points predicted from item text, never from probe answers.

    Two heads, because the two item types have different parameters: four
    ordered thresholds for a Likert item, one difficulty for a binary one. Both
    are trained on the fitted cut points of the training items only. Predicted
    Likert thresholds are sorted, which is the cheapest way to keep the ordering
    the response function needs.
    """
    from .item_encoder import reduce_embeddings, topic_features, wording_features

    items_by_id, domains = load_item_texts(bank_path)

    def features(ids: Sequence[str]) -> np.ndarray:
        wording = np.vstack([wording_features(items_by_id[i]) for i in ids])
        topic = topic_features(items_by_id, ids, domains)
        return np.hstack([wording, topic])

    x_train = features(train_items.item_ids)
    x_probe = features(probe_items.item_ids)
    embedding_source = "none"
    embed_components = 0
    if embeddings_path is not None and Path(embeddings_path).is_file():
        emb = np.load(embeddings_path, allow_pickle=False)
        lookup = {str(v): i for i, v in enumerate(emb["item_ids"])}
        raw_train = np.vstack([emb["embeddings"][lookup[i]] for i in train_items.item_ids])
        raw_probe = np.vstack([emb["embeddings"][lookup[i]] for i in probe_items.item_ids])
        embed_components = 16
        z_train, z_probe = reduce_embeddings(raw_train, raw_probe, embed_components)
        x_train = np.hstack([x_train, z_train])
        x_probe = np.hstack([x_probe, z_probe])
        embedding_source = str(embeddings_path)

    train_likert = train_items.is_likert
    probe_likert = probe_items.is_likert

    likert_head = train_threshold_head(
        x_train[train_likert], train_items.thresholds[train_likert]
    )
    binary_head = train_threshold_head(
        x_train[~train_likert], train_items.difficulty[~train_likert][:, None]
    )

    thresholds = np.full((probe_items.n_items, mirt.LIKERT_CUTS), np.nan)
    difficulty = np.full(probe_items.n_items, np.nan)
    if probe_likert.any():
        predicted = ridge_multi_predict(likert_head, x_probe[probe_likert])
        thresholds[probe_likert] = np.sort(predicted, axis=1)
    if (~probe_likert).any():
        difficulty[~probe_likert] = ridge_multi_predict(binary_head, x_probe[~probe_likert])[:, 0]

    block = ItemBlock(
        item_ids=list(probe_items.item_ids),
        kinds=list(probe_items.kinds),
        loadings=np.asarray(encoder_loadings, dtype=float),
        thresholds=thresholds,
        difficulty=difficulty,
    )
    report = {
        "label": "EXPLORATORY",
        "what": (
            "ridge from item text features onto the FITTED cut points of the 202 "
            "training items, applied zero-shot to the 50 probe items"
        ),
        "features": ["wording (4)", "topic one-hot", f"reduced embeddings ({embed_components})"],
        "embedding_source": embedding_source,
        "embed_components": embed_components,
        "likert_head": {
            "n_train_items": int(train_likert.sum()),
            "lambda": likert_head["lam"],
            "loo_mse": likert_head["loo_mse"],
        },
        "binary_head": {
            "n_train_items": int((~train_likert).sum()),
            "lambda": binary_head["lam"],
            "loo_mse": binary_head["loo_mse"],
        },
        "holdout_cut_rmse_vs_fitted": {
            "likert_thresholds": float(
                np.sqrt(
                    np.nanmean((thresholds[probe_likert] - probe_items.thresholds[probe_likert]) ** 2)
                )
            )
            if probe_likert.any()
            else None,
            "binary_difficulty": float(
                np.sqrt(
                    np.nanmean(
                        (difficulty[~probe_likert] - probe_items.difficulty[~probe_likert]) ** 2
                    )
                )
            )
            if (~probe_likert).any()
            else None,
        },
        "ordering_note": "predicted Likert thresholds are sorted ascending",
    }
    if verbose:
        print(
            f"  exploratory text threshold head: likert lambda {likert_head['lam']:.4g}, "
            f"binary lambda {binary_head['lam']:.4g}"
        )
    return block, report


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.model.predictor",
        description="Stage 4 ordinal-logit predictor over held-out cells.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    cmd = sub.add_parser("build", help="Predict every held-out persona x probe item cell.")
    cmd.add_argument("--fit", type=Path, default=REPO_ROOT / "results" / "stage2_v2_fit.npz")
    cmd.add_argument("--fused", type=Path, default=REPO_ROOT / "results" / "stage3_person_fused.npz")
    cmd.add_argument(
        "--item-predictions",
        type=Path,
        default=REPO_ROOT / "data" / "runs" / "stage3_qwen" / "item_encoder_predictions.npz",
    )
    cmd.add_argument(
        "--answers",
        type=Path,
        default=REPO_ROOT / "data" / "runs" / "full_sweep_v2" / "answers_noised.jsonl",
    )
    cmd.add_argument("--bank", type=Path, default=REPO_ROOT / "data" / "public" / "bank_items.json")
    cmd.add_argument(
        "--embeddings",
        type=Path,
        default=REPO_ROOT / "data" / "runs" / "stage3_qwen" / "item_embeddings.npz",
    )
    cmd.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "stage4_predictions.npz")
    cmd.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    cmd.add_argument("--seed", type=int, default=DEFAULT_SEED)
    cmd.add_argument("--quiet", action="store_true")

    args = parser.parse_args(argv)
    build(
        args.fit,
        args.fused,
        args.item_predictions,
        args.answers,
        args.bank,
        args.out,
        embeddings_path=args.embeddings,
        n_draws=args.draws,
        seed=args.seed,
        verbose=not args.quiet,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
