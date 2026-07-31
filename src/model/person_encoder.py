"""Person-encoder backbone for Stage 3 (PREREGISTRATION section 6, Gate 3).

    python -m src.model.person_encoder \
        --transcripts data/runs/interview_v1/transcripts.jsonl \
        --fit results/stage2_v2_fit.npz \
        --splits experiments/splits_v1.json \
        --out results/stage3_person_backbone.json

What this is
------------
The exact posterior over a persona's latent vector given the closed-form
interview answers they actually gave, under the item-response model Stage 2
already fitted. Nothing is learned here: the item parameters are frozen inputs,
so for a given set of answers this posterior is the whole of what those answers
say under that model.

For every persona and every prefix length N = 1..15 of the frozen scripted
order it produces

  * ``theta`` -- the MAP estimate, 8 numbers, in the linked metric the Stage 2
    fit stores (see :func:`src.model.mirt.link_scale`);
  * ``blur`` -- the per-dimension square root of the Laplace posterior
    covariance, raw and after the calibration below.

There is **no LLM anywhere in this file**. It is arithmetic on the transcript
and on the fitted item parameters.

Model, and why it matches the fit exactly
-----------------------------------------
Prior theta ~ N(0, I), the same prior :func:`src.model.mirt.fit_joint` uses.
Likelihood: the graded-response model on Likert-5 items and 2PL on binary ones,
imported from ``src.model.mirt`` rather than restated, so the posterior cannot
drift away from the likelihood that produced the parameters. The item
parameters are read out of the fit archive and put back into
:class:`src.model.mirt.ItemParams` by inverting the ordered-threshold
parameterization, which is exact.

The archive stores the loadings *after* linking, and the held-out person
vectors in it were scored with those linked loadings under the same N(0, I)
prior. Doing the same here is what puts this module's ``theta`` on the same
axis and the same unit as ``theta_train`` / ``theta_holdout``.

The MAP is found by damped Newton with per-persona backtracking (the negative
log posterior is strictly convex, so the mode is unique). The posterior
covariance is the inverse of the Hessian of the per-person negative log
posterior, taken by central differences on the analytic gradient with the same
step ``src.model.mirt.laplace_covariance`` uses, so "blur" means the same thing
here as it does in the Stage 2 artifacts.

Extension point for the open answers
------------------------------------
:class:`GaussianBlock` is the hook the open-answer fusion block plugs into.
A block contributes

    0.5 * theta^T Lambda theta  -  theta^T h

to the negative log posterior -- a Gaussian measurement written in information
form (precision ``Lambda``, information vector ``h``). Pass any number of them
to :func:`posterior` and the mode, the gradient and the Hessian all pick them
up; nothing in the core has to change.
:meth:`GaussianBlock.from_linear` builds one from the pseudo-item form
``y = B theta + c + eps``, ``eps ~ N(0, Psi)``, which is how the three open
answers are meant to enter. Because ``Lambda`` is positive semi-definite and
does not depend on N, adding a block keeps the blur monotone in N.

Blur calibration
----------------
One scalar per dimension, nothing more, and fitted on **training personas
only**: the ratio of the spread of ``theta_hat(N=15) - theta_hat(full matrix)``
across the 400 training personas to the mean posterior blur at N=15. Both terms
are system-side fitted quantities. The constants are written into the output
artifact and the identical constants are applied to held-out personas.

WALL RULES. This module reads three things: the assembled transcript
(public profile + the answers actually given), the fitted item parameters, and
the frozen split manifest. It never reads a planted trait value, a designed
loading, a persona card or an answer distribution, and no path to any of those
appears anywhere in it. Every path is a command-line argument. The Procrustes
rotation in the recovery artifact is fitted against planted truth and is
grader-side; nothing here touches it, and nothing here compares anything to
planted truth -- that is the Gate 3 grader's job.
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
from .profile_baseline import evaluate_profile_baseline

#: Latent width. Read off the fit archive at run time; this is only the default.
DEFAULT_DIMS = mirt.DEFAULT_DIMS

#: Step for the central-difference Hessian. Same value as
#: :func:`src.model.mirt.laplace_covariance`, so the blur is the same quantity.
HESSIAN_STEP = 1e-4

#: Newton stops when the largest gradient component of any persona is below this.
DEFAULT_TOL = 1e-10

#: Hard cap on Newton steps. Convergence is typically reached in well under ten.
DEFAULT_MAX_ITERS = 60

#: Eigenvalue floor used only to make a Newton *step* well posed. The Hessian
#: reported for the covariance is never floored -- a non-positive-definite one
#: is reported instead, exactly as the Stage 2 fit does.
_STEP_EIG_FLOOR = 1e-8

#: Backtracking line-search halvings per Newton step.
_MAX_BACKTRACK = 30

_EPS = mirt._EPS

#: Schema tag on the output artifact.
SCHEMA = "glassbox.stage3.person_backbone/1"


# --------------------------------------------------------------------------
# the extension point: extra Gaussian information about theta
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GaussianBlock:
    """One extra Gaussian measurement of theta, in information form.

    ``precision`` is ``Lambda``, either one ``(dims, dims)`` matrix shared by
    every persona or one per persona, ``(n, dims, dims)``. ``info`` is ``h``,
    either ``(dims,)`` or ``(n, dims)``. Together they add

        0.5 * theta^T Lambda theta - theta^T h

    to the negative log posterior, i.e. a Gaussian likelihood whose mean
    contribution is ``Lambda^-1 h``.

    This is the seam the open-answer channel plugs into. Nothing about the
    closed-item likelihood changes when a block is present.
    """

    precision: np.ndarray
    info: np.ndarray

    @classmethod
    def from_linear(
        cls,
        loading: np.ndarray,
        offset: np.ndarray,
        observation: np.ndarray,
        noise_cov: np.ndarray,
    ) -> "GaussianBlock":
        """Build a block from ``y = B theta + c + eps``, ``eps ~ N(0, Psi)``.

        ``loading`` is ``B`` with shape ``(m, dims)``, ``offset`` is ``c`` with
        shape ``(m,)``, ``observation`` is ``y`` with shape ``(m,)`` or
        ``(n, m)``, and ``noise_cov`` is ``Psi``: either the full ``(m, m)``
        matrix or its ``(m,)`` diagonal.

        Then ``Lambda = B^T Psi^-1 B`` and ``h = B^T Psi^-1 (y - c)``.
        """
        loading = np.asarray(loading, dtype=float)
        offset = np.asarray(offset, dtype=float)
        observation = np.asarray(observation, dtype=float)
        noise_cov = np.asarray(noise_cov, dtype=float)
        if loading.ndim != 2:
            raise ValueError(f"loading must be (m, dims), got {loading.shape}")
        m = loading.shape[0]
        if offset.shape != (m,):
            raise ValueError(f"offset must be ({m},), got {offset.shape}")

        if noise_cov.ndim == 1:
            if noise_cov.shape != (m,):
                raise ValueError(f"diagonal noise must be ({m},), got {noise_cov.shape}")
            weighted = loading / noise_cov[:, None]  # Psi^-1 B
        elif noise_cov.shape == (m, m):
            weighted = np.linalg.solve(noise_cov, loading)
        else:
            raise ValueError(f"noise_cov must be ({m},) or ({m}, {m}), got {noise_cov.shape}")

        precision = loading.T @ weighted
        precision = 0.5 * (precision + precision.T)
        residual = observation - offset  # broadcasts over a leading persona axis
        info = residual @ weighted
        return cls(precision=precision, info=info)

    def validate(self, n_personas: int, dims: int) -> None:
        """Refuse a block whose shapes do not fit the population being scored."""
        prec = np.asarray(self.precision, dtype=float)
        info = np.asarray(self.info, dtype=float)
        if prec.shape not in ((dims, dims), (n_personas, dims, dims)):
            raise ValueError(
                f"block precision must be {(dims, dims)} or "
                f"{(n_personas, dims, dims)}, got {prec.shape}"
            )
        if info.shape not in ((dims,), (n_personas, dims)):
            raise ValueError(
                f"block info must be {(dims,)} or {(n_personas, dims)}, got {info.shape}"
            )
        square = prec if prec.ndim == 2 else prec.reshape(-1, dims, dims)
        if not np.allclose(square, np.swapaxes(square, -1, -2), atol=1e-8):
            raise ValueError("block precision must be symmetric")
        if float(np.min(np.linalg.eigvalsh(square))) < -1e-8:
            raise ValueError("block precision must be positive semi-definite")

    def gradient(self, theta: np.ndarray) -> np.ndarray:
        """d/dtheta of the block's contribution, ``Lambda theta - h``."""
        prec = np.asarray(self.precision, dtype=float)
        if prec.ndim == 2:
            quad = theta @ prec
        else:
            quad = np.einsum("nij,nj->ni", prec, theta)
        return quad - np.asarray(self.info, dtype=float)

    def penalty(self, theta: np.ndarray) -> np.ndarray:
        """The block's contribution to each persona's negative log posterior."""
        prec = np.asarray(self.precision, dtype=float)
        if prec.ndim == 2:
            quad = np.einsum("ni,ij,nj->n", theta, prec, theta)
        else:
            quad = np.einsum("ni,nij,nj->n", theta, prec, theta)
        return 0.5 * quad - np.sum(theta * np.asarray(self.info, dtype=float), axis=1)


# --------------------------------------------------------------------------
# likelihood pieces
# --------------------------------------------------------------------------


def blocks_from_codes(codes: np.ndarray, kinds: Sequence[str]) -> mirt._Blocks:
    """Split an answer matrix into the Likert and binary blocks the fit uses."""
    codes = np.asarray(codes)
    kind_array = np.array(list(kinds))
    if codes.shape[1] != kind_array.size:
        raise ValueError(
            f"codes has {codes.shape[1]} columns but {kind_array.size} item kinds"
        )
    likert_cols = np.flatnonzero(kind_array == mirt.LIKERT)
    binary_cols = np.flatnonzero(kind_array == mirt.BINARY)
    unknown = set(kind_array.tolist()) - {mirt.LIKERT, mirt.BINARY}
    if unknown:
        raise ValueError(f"unknown item kinds: {sorted(unknown)}")
    return mirt._Blocks(
        likert_cols=likert_cols,
        binary_cols=binary_cols,
        likert_codes=codes[:, likert_cols].astype(np.int64),
        binary_codes=codes[:, binary_cols].astype(np.int64),
    )


def per_person_loglik(
    theta: np.ndarray, params: mirt.ItemParams, blocks: mirt._Blocks
) -> np.ndarray:
    """Log-likelihood of each person's answers, one number per person.

    The same graded-response and 2PL probabilities
    :func:`src.model.mirt._loglik_and_grads` sums; kept per-person here because
    the line search needs each persona's own objective. A unit test asserts
    this sums to exactly what the fit's own routine returns.
    """
    theta = np.asarray(theta, dtype=float)
    n_persons = theta.shape[0]
    z = theta @ params.loadings.T
    total = np.zeros(n_persons)

    lc = blocks.likert_cols
    if lc.size:
        taus = params.thresholds()[lc]
        seen = mirt.sigmoid(z[:, lc][:, :, None] - taus[None, :, :])
        cumulative = np.concatenate(
            [np.ones((n_persons, lc.size, 1)), seen, np.zeros((n_persons, lc.size, 1))],
            axis=2,
        )
        codes = blocks.likert_codes
        observed = codes >= 0
        safe = np.where(observed, codes, 0)
        p_lo = np.take_along_axis(cumulative, safe[:, :, None], axis=2)[:, :, 0]
        p_hi = np.take_along_axis(cumulative, safe[:, :, None] + 1, axis=2)[:, :, 0]
        prob = np.maximum(p_lo - p_hi, _EPS)
        total += np.sum(np.where(observed, np.log(prob), 0.0), axis=1)

    bc = blocks.binary_cols
    if bc.size:
        b = params.difficulty()[bc]
        p = mirt.sigmoid(z[:, bc] - b[None, :])
        codes = blocks.binary_codes
        observed = codes >= 0
        y = np.where(observed, codes, 0).astype(np.float64)
        total += np.sum(
            np.where(
                observed,
                y * np.log(np.maximum(p, _EPS)) + (1 - y) * np.log(np.maximum(1 - p, _EPS)),
                0.0,
            ),
            axis=1,
        )
    return total


def neg_log_posterior(
    theta: np.ndarray,
    params: mirt.ItemParams,
    blocks: mirt._Blocks,
    extra: Sequence[GaussianBlock] = (),
) -> np.ndarray:
    """Each person's negative log posterior, up to a constant."""
    value = -per_person_loglik(theta, params, blocks) + 0.5 * np.sum(theta**2, axis=1)
    for block in extra:
        value = value + block.penalty(theta)
    return value


def neg_log_posterior_grad(
    theta: np.ndarray,
    params: mirt.ItemParams,
    blocks: mirt._Blocks,
    extra: Sequence[GaussianBlock] = (),
) -> np.ndarray:
    """Gradient of :func:`neg_log_posterior`, one row per person."""
    _, g_theta, _, _ = mirt._loglik_and_grads(theta, params, blocks, want_item_grads=False)
    grad = -g_theta + theta
    for block in extra:
        grad = grad + block.gradient(theta)
    return grad


def observed_hessian(
    theta: np.ndarray,
    params: mirt.ItemParams,
    blocks: mirt._Blocks,
    extra: Sequence[GaussianBlock] = (),
    *,
    step: float = HESSIAN_STEP,
) -> np.ndarray:
    """Per-person Hessian of the negative log posterior, ``(n, dims, dims)``.

    Central differences on the analytic gradient, then symmetrized -- the same
    recipe and the same step as :func:`src.model.mirt.laplace_covariance`. The
    per-person objectives are separable, so differencing the whole population
    at once gives every persona's Hessian in ``2 * dims`` gradient evaluations.
    """
    theta = np.asarray(theta, dtype=float)
    n_persons, dims = theta.shape
    hess = np.zeros((n_persons, dims, dims))
    for d in range(dims):
        bump = np.zeros(dims)
        bump[d] = step
        g_plus = neg_log_posterior_grad(theta + bump, params, blocks, extra)
        g_minus = neg_log_posterior_grad(theta - bump, params, blocks, extra)
        hess[:, :, d] = (g_plus - g_minus) / (2 * step)
    return 0.5 * (hess + np.swapaxes(hess, 1, 2))


# --------------------------------------------------------------------------
# the posterior
# --------------------------------------------------------------------------


@dataclass
class Posterior:
    """MAP point estimate, Laplace covariance and blur for one population."""

    theta: np.ndarray  # (n, dims)
    cov: np.ndarray  # (n, dims, dims)
    blur: np.ndarray  # (n, dims)
    n_iters: int
    max_grad: float
    non_pd_rows: list[int]

    @property
    def dims(self) -> int:
        return self.theta.shape[1]


def posterior(
    codes: np.ndarray,
    params: mirt.ItemParams,
    *,
    extra: Sequence[GaussianBlock] = (),
    dims: int | None = None,
    max_iters: int = DEFAULT_MAX_ITERS,
    tol: float = DEFAULT_TOL,
    hessian_step: float = HESSIAN_STEP,
) -> Posterior:
    """MAP theta and Laplace covariance for every row of ``codes``.

    ``codes`` is ``(n_personas, n_items)`` with the category index in each cell
    (0..4 on Likert-5, 0/1 on binary, ``-1`` for an item that was not asked).
    ``params`` carries those same items in the same column order.

    ``extra`` is the extension point: any number of :class:`GaussianBlock`
    measurements, which enter the mode, the gradient and the Hessian alongside
    the items. With ``codes`` empty and one block, the answer is the exact
    Gaussian posterior ``N((I + Lambda)^-1 h, (I + Lambda)^-1)``.
    """
    codes = np.atleast_2d(np.asarray(codes))
    dims = int(params.loadings.shape[1] if dims is None else dims)
    n_personas = codes.shape[0]
    blocks = blocks_from_codes(codes, params.kinds)
    extra = list(extra)
    for block in extra:
        block.validate(n_personas, dims)

    theta = np.zeros((n_personas, dims))  # start at the prior mean
    value = neg_log_posterior(theta, params, blocks, extra)
    grad = neg_log_posterior_grad(theta, params, blocks, extra)
    used = 0

    while used < max_iters and float(np.max(np.abs(grad))) > tol:
        used += 1
        hess = observed_hessian(theta, params, blocks, extra, step=hessian_step)
        eigvals, eigvecs = np.linalg.eigh(hess)
        safe = np.maximum(eigvals, _STEP_EIG_FLOOR)
        # step = H^-1 g, with the eigenvalues floored so the step is a descent
        # direction even where the numerical Hessian is not positive definite
        proj = np.einsum("nji,nj->ni", eigvecs, grad)
        step_dir = np.einsum("nij,nj->ni", eigvecs, proj / safe)

        # backtrack per persona: halve the step until nobody's objective rose
        scale = np.ones(n_personas)
        candidate = theta - step_dir
        new_value = neg_log_posterior(candidate, params, blocks, extra)
        for _ in range(_MAX_BACKTRACK):
            worse = new_value > value + 1e-12
            if not worse.any():
                break
            scale[worse] *= 0.5
            candidate = theta - scale[:, None] * step_dir
            new_value = neg_log_posterior(candidate, params, blocks, extra)
        theta = candidate
        value = new_value
        grad = neg_log_posterior_grad(theta, params, blocks, extra)

    hess = observed_hessian(theta, params, blocks, extra, step=hessian_step)
    cov = np.zeros((n_personas, dims, dims))
    non_pd: list[int] = []
    for row in range(n_personas):
        eigs = np.linalg.eigvalsh(hess[row])
        if eigs.min() <= 0:
            non_pd.append(row)
            cov[row] = np.linalg.pinv(hess[row])
        else:
            cov[row] = np.linalg.inv(hess[row])
    blur = np.sqrt(np.maximum(np.diagonal(cov, axis1=1, axis2=2), 0.0))
    return Posterior(
        theta=theta,
        cov=cov,
        blur=blur,
        n_iters=used,
        max_grad=float(np.max(np.abs(grad))),
        non_pd_rows=non_pd,
    )


# --------------------------------------------------------------------------
# reading the frozen inputs
# --------------------------------------------------------------------------


def item_params_from_fit(
    fit: Any, *, block: str = "train"
) -> tuple[mirt.ItemParams, list[str]]:
    """Rebuild :class:`src.model.mirt.ItemParams` from the fit archive.

    The archive stores ordered thresholds and binary difficulties rather than
    the raw parameterization, so the raw cut points are recovered by inverting
    it: ``cut_raw[0] = tau_1`` and ``cut_raw[j] = inv_softplus(tau_j -
    tau_{j-1})``. That inversion is exact, and a unit test round-trips it.

    The loadings in the archive are the linked ones, which is what puts this
    module's theta on the same metric as the fit's own theta.
    """
    loadings = np.asarray(fit[f"{block}_loadings"], dtype=float)
    kinds = [str(k) for k in fit[f"{block}_item_kinds"]]
    thresholds = np.asarray(fit[f"{block}_thresholds"], dtype=float)
    difficulty = np.asarray(fit[f"{block}_difficulty"], dtype=float)
    item_ids = [str(i) for i in fit[f"item_{block}_ids"]]

    is_likert = np.array([k == mirt.LIKERT for k in kinds])
    cut_raw = np.zeros((len(kinds), mirt.LIKERT_CUTS))
    if is_likert.any():
        taus = thresholds[is_likert]
        gaps = np.diff(taus, axis=1)
        if np.any(gaps <= 0):
            raise ValueError("stored Likert thresholds are not strictly increasing")
        cut_raw[is_likert, 0] = taus[:, 0]
        cut_raw[is_likert, 1:] = mirt.inv_softplus(gaps)
    if (~is_likert).any():
        cut_raw[~is_likert, 0] = difficulty[~is_likert]

    params = mirt.ItemParams(loadings=loadings.copy(), cut_raw=cut_raw, kinds=kinds)
    return params, item_ids


@dataclass(frozen=True)
class Transcripts:
    """The system side's whole view of the interviewed population."""

    pids: list[str]
    profiles: dict[str, dict[str, Any]]
    item_ids: list[str]  # scripted order, taken from the transcripts themselves
    answers: dict[str, list[Any]]  # pid -> answer per scripted position


def load_transcripts(path: str | Path) -> Transcripts:
    """Read the assembled transcripts: profile and the closed turns in order.

    The open answers are read but not used here -- they are the later fusion
    block's input. Every persona must carry the same scripted order, which is
    what makes "the first N items" mean the same thing for everyone.
    """
    pids: list[str] = []
    profiles: dict[str, dict[str, Any]] = {}
    answers: dict[str, list[Any]] = {}
    scripted: list[str] | None = None

    with Path(path).open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            pid = str(record["pid"])
            turns = sorted(record.get("turns") or [], key=lambda t: int(t["n"]))
            order = [str(turn["item_id"]) for turn in turns]
            if scripted is None:
                scripted = order
            elif order != scripted:
                raise ValueError(
                    f"{path}: line {lineno} ({pid}) has a different scripted order"
                )
            pids.append(pid)
            profiles[pid] = dict(record.get("profile") or {})
            answers[pid] = [turn["answer"] for turn in turns]

    if scripted is None:
        raise ValueError(f"{path}: no transcripts found")
    return Transcripts(pids=pids, profiles=profiles, item_ids=scripted, answers=answers)


def code_matrix(
    transcripts: Transcripts, pids: Sequence[str], kinds: Sequence[str]
) -> np.ndarray:
    """Code the scripted answers for ``pids`` into the fit's category indices.

    Uses the fit's own answer coding (:func:`src.model.mirt._code_answer`), so a
    "yes" and a Likert 4 mean here exactly what they meant when the item
    parameters were estimated.
    """
    rows = []
    for pid in pids:
        given = transcripts.answers[pid]
        if len(given) != len(kinds):
            raise ValueError(
                f"{pid}: {len(given)} answers but {len(kinds)} scripted items"
            )
        rows.append([mirt._code_answer(a, k) for a, k in zip(given, kinds)])
    return np.array(rows, dtype=np.int16)


# --------------------------------------------------------------------------
# the prefix sweep
# --------------------------------------------------------------------------


def prefix_params(
    params: mirt.ItemParams, columns: Sequence[int]
) -> mirt.ItemParams:
    """The item parameters for a chosen subset of columns, order preserved."""
    columns = list(columns)
    return mirt.ItemParams(
        loadings=params.loadings[columns].copy(),
        cut_raw=params.cut_raw[columns].copy(),
        kinds=[params.kinds[c] for c in columns],
    )


def encode_prefixes(
    codes: np.ndarray,
    params: mirt.ItemParams,
    n_grid: Sequence[int],
    *,
    extra: Sequence[GaussianBlock] = (),
    **kwargs: Any,
) -> dict[int, Posterior]:
    """Posterior after the first N scripted items, for every N in ``n_grid``.

    ``extra`` is passed through to every N, which is exactly how the open
    answers behave: they are given up front, so they are present at every
    prefix length.
    """
    out: dict[int, Posterior] = {}
    for n in n_grid:
        n = int(n)
        cols = list(range(n))
        out[n] = posterior(
            codes[:, :n], prefix_params(params, cols), extra=extra, **kwargs
        )
    return out


# --------------------------------------------------------------------------
# blur calibration -- training residuals only
# --------------------------------------------------------------------------


#: How the calibration constant is defined, recorded in the artifact verbatim.
CALIBRATION_DEFINITION = (
    "kappa_d = rms_over_training_personas(theta_hat_N - theta_hat_full)_d / "
    "rms_over_training_personas(blur_N)_d, at N = 15, on the 400 training "
    "personas only. theta_hat_full is the Stage 2 fit's own point estimate "
    "for the same persona from the full training-item matrix. Both terms are "
    "system-side fitted quantities; no planted value is involved. The same "
    "constants are applied unchanged to held-out personas."
)


@dataclass(frozen=True)
class BlurCalibration:
    """One inflation factor per dimension, and the numbers behind it."""

    kappa: np.ndarray  # (dims,)
    residual_rms: np.ndarray  # (dims,)
    blur_rms: np.ndarray  # (dims,)
    fitted_at_n: int
    n_personas: int

    def apply(self, blur: np.ndarray) -> np.ndarray:
        """Inflate a blur matrix by the per-dimension constants."""
        return np.asarray(blur, dtype=float) * self.kappa[None, :]

    def to_json(self) -> dict[str, Any]:
        return {
            "definition": CALIBRATION_DEFINITION,
            "fitted_at_n": self.fitted_at_n,
            "fitted_on": "training personas only",
            "n_personas": self.n_personas,
            "kappa_per_dim": [float(v) for v in self.kappa],
            "residual_rms_per_dim": [float(v) for v in self.residual_rms],
            "posterior_blur_rms_per_dim": [float(v) for v in self.blur_rms],
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "BlurCalibration":
        return cls(
            kappa=np.asarray(payload["kappa_per_dim"], dtype=float),
            residual_rms=np.asarray(payload["residual_rms_per_dim"], dtype=float),
            blur_rms=np.asarray(payload["posterior_blur_rms_per_dim"], dtype=float),
            fitted_at_n=int(payload["fitted_at_n"]),
            n_personas=int(payload["n_personas"]),
        )


def fit_blur_calibration(
    theta_n: np.ndarray,
    blur_n: np.ndarray,
    theta_full: np.ndarray,
    *,
    at_n: int,
) -> BlurCalibration:
    """Per-dimension blur inflation from training residuals. Nothing else.

    ``theta_n`` / ``blur_n`` are this module's posterior at N for the training
    personas; ``theta_full`` is the same personas' fitted vector from the full
    training-item matrix. One scalar per dimension, no intercept, no slope, no
    dependence on the persona.
    """
    theta_n = np.asarray(theta_n, dtype=float)
    theta_full = np.asarray(theta_full, dtype=float)
    blur_n = np.asarray(blur_n, dtype=float)
    if theta_n.shape != theta_full.shape or theta_n.shape != blur_n.shape:
        raise ValueError(
            f"shape mismatch: theta_n {theta_n.shape}, blur_n {blur_n.shape}, "
            f"theta_full {theta_full.shape}"
        )
    residual_rms = np.sqrt(np.mean((theta_n - theta_full) ** 2, axis=0))
    blur_rms = np.sqrt(np.mean(blur_n**2, axis=0))
    kappa = residual_rms / np.maximum(blur_rms, _EPS)
    return BlurCalibration(
        kappa=kappa,
        residual_rms=residual_rms,
        blur_rms=blur_rms,
        fitted_at_n=int(at_n),
        n_personas=int(theta_n.shape[0]),
    )


# --------------------------------------------------------------------------
# agreement and monotonicity
# --------------------------------------------------------------------------


def agreement(estimate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    """Per-dimension Pearson r and RMSE between two theta matrices."""
    estimate = np.asarray(estimate, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if estimate.shape != reference.shape:
        raise ValueError(f"shape mismatch: {estimate.shape} vs {reference.shape}")
    r = mirt._column_correlations(estimate, reference)
    rmse = np.sqrt(np.mean((estimate - reference) ** 2, axis=0))
    return {
        "r_per_dim": [float(v) for v in r],
        "rmse_per_dim": [float(v) for v in rmse],
        "r_mean": float(np.nanmean(r)),
        "rmse_mean": float(np.mean(rmse)),
        "n_personas": int(estimate.shape[0]),
    }


def monotonicity_report(
    n_grid: Sequence[int],
    mean_blur: np.ndarray,
    *,
    blur_by_n: np.ndarray | None = None,
    tol: float = 0.0,
) -> dict[str, Any]:
    """Does the mean blur weakly shrink as N grows, dimension by dimension?

    ``mean_blur`` is ``(len(n_grid), dims)``; ``blur_by_n``, if given, is the
    full ``(len(n_grid), n_personas, dims)`` stack, used for the persona-level
    count.

    Adding an item adds a positive semi-definite term to the information
    matrix, so **at a fixed theta** every diagonal element of its inverse
    falls -- that part is a theorem, and :func:`blur_at_fixed_theta` checks it
    directly. The MAP itself moves between prefixes, though, and the observed
    information is read at the new MAP, so the curve of a real sweep is checked
    here rather than assumed. ``tol`` is zero by default: a rise is a rise, and
    its size is reported next to the shrinkage it sits on.
    """
    mean_blur = np.asarray(mean_blur, dtype=float)
    steps = np.diff(mean_blur, axis=0)
    dims = mean_blur.shape[1]
    worst = steps.max(axis=0) if steps.size else np.zeros(dims)
    per_dim = []
    for d in range(dims):
        rises = [
            {
                "from_n": int(n_grid[k]),
                "to_n": int(n_grid[k + 1]),
                "increase": float(steps[k, d]),
            }
            for k in range(steps.shape[0])
            if steps[k, d] > tol
        ]
        shrinkage = float(mean_blur[0, d] - mean_blur[-1, d])
        per_dim.append(
            {
                "dim": d,
                "monotone": not rises,
                "n_violations": len(rises),
                "largest_increase": float(worst[d]),
                "largest_increase_as_fraction_of_total_shrinkage": (
                    float(worst[d] / shrinkage) if shrinkage > 0 else float("nan")
                ),
                "total_shrinkage": shrinkage,
                "first_blur": float(mean_blur[0, d]),
                "last_blur": float(mean_blur[-1, d]),
                "violations": rises,
            }
        )

    report = {
        "tolerance": tol,
        "verdict": "PASS" if all(row["monotone"] for row in per_dim) else "FAIL",
        "n_dims_monotone": int(sum(row["monotone"] for row in per_dim)),
        "largest_increase_any_dim": float(worst.max()) if steps.size else 0.0,
        "mean_blur_per_dim_by_n": [[float(v) for v in row] for row in mean_blur],
        "n_grid": [int(n) for n in n_grid],
        "per_dim": per_dim,
    }
    if blur_by_n is not None:
        blur_by_n = np.asarray(blur_by_n, dtype=float)
        per_step = np.diff(blur_by_n, axis=0)
        report["persona_level"] = {
            "n_steps_checked": int(per_step.size),
            "n_increases": int((per_step > tol).sum()),
            "fraction_increasing": float((per_step > tol).mean()),
            "largest_increase": float(per_step.max()) if per_step.size else 0.0,
        }
    return report


def blur_at_fixed_theta(
    codes: np.ndarray,
    params: mirt.ItemParams,
    theta: np.ndarray,
    n_grid: Sequence[int],
    *,
    extra: Sequence[GaussianBlock] = (),
    hessian_step: float = HESSIAN_STEP,
) -> np.ndarray:
    """Blur after the first N items with theta held still, ``(len(n_grid), n, dims)``.

    The theorem check. Both the graded-response and the 2PL log-likelihood are
    concave in ``a . theta``, so each item contributes ``c_i(theta) a_i a_i^T``
    with ``c_i >= 0`` -- positive semi-definite whatever the answer was. The
    information matrices therefore nest exactly across prefixes and every
    diagonal of the inverse must fall. Any rise here is a numerical fault, not
    a property of the data, which is what makes this worth computing next to
    the real sweep.
    """
    theta = np.asarray(theta, dtype=float)
    out = np.zeros((len(n_grid), theta.shape[0], theta.shape[1]))
    for k, n in enumerate(n_grid):
        n = int(n)
        sub = prefix_params(params, range(n))
        blocks = blocks_from_codes(codes[:, :n], sub.kinds)
        hess = observed_hessian(theta, sub, blocks, extra, step=hessian_step)
        cov = np.linalg.inv(hess)
        out[k] = np.sqrt(np.maximum(np.diagonal(cov, axis1=1, axis2=2), 0.0))
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def run(
    transcripts_path: Path,
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
    params, train_item_ids = item_params_from_fit(fit, block="train")
    column_of = {iid: j for j, iid in enumerate(train_item_ids)}

    transcripts = load_transcripts(transcripts_path)
    splits = json.loads(Path(splits_path).read_text(encoding="utf-8"))

    scripted = list(transcripts.item_ids)
    declared = [str(i) for i in splits.get("interview_closed") or []]
    if declared and declared != scripted:
        raise ValueError(
            "the transcripts' scripted order does not match the frozen split "
            f"manifest: {scripted} vs {declared}"
        )
    missing = [iid for iid in scripted if iid not in column_of]
    if missing:
        raise ValueError(
            f"scripted items absent from the fitted training items: {missing}"
        )

    interview_cols = [column_of[iid] for iid in scripted]
    interview_params = prefix_params(params, interview_cols)
    kinds = list(interview_params.kinds)

    train_pids = [str(p) for p in fit["persona_train_ids"]]
    holdout_pids = [str(p) for p in fit["persona_holdout_ids"]]
    split_holdout = {str(p) for p in splits.get("persona_holdout") or []}
    if split_holdout and split_holdout != set(holdout_pids):
        raise ValueError("the fit's held-out personas differ from the split manifest")
    seen = set(transcripts.pids)
    absent = [p for p in train_pids + holdout_pids if p not in seen]
    if absent:
        raise ValueError(f"{len(absent)} fitted personas have no transcript: {absent[:5]}")

    theta_full_train = np.asarray(fit["theta_train"], dtype=float)
    theta_full_holdout = np.asarray(fit["theta_holdout"], dtype=float)

    codes_train = code_matrix(transcripts, train_pids, kinds)
    codes_holdout = code_matrix(transcripts, holdout_pids, kinds)
    n_grid = list(range(1, len(scripted) + 1))

    if verbose:
        print(
            f"transcripts: {len(transcripts.pids)} personas, {len(scripted)} scripted items"
        )
        print(f"fit: {len(train_pids)} training / {len(holdout_pids)} held-out personas, dims {dims}")
        print(f"scoring prefixes N = {n_grid[0]}..{n_grid[-1]} ...")

    train_post = encode_prefixes(codes_train, interview_params, n_grid)
    holdout_post = encode_prefixes(codes_holdout, interview_params, n_grid)

    # ---- blur calibration, training residuals only -----------------------
    calib = fit_blur_calibration(
        train_post[calibrate_at].theta,
        train_post[calibrate_at].blur,
        theta_full_train,
        at_n=calibrate_at,
    )
    if verbose:
        print(
            "blur calibration (training residuals at N="
            f"{calibrate_at}): kappa "
            + " ".join(f"{v:.3f}" for v in calib.kappa)
        )

    # ---- per-N table ------------------------------------------------------
    per_n: list[dict[str, Any]] = []
    holdout_mean_blur = np.zeros((len(n_grid), dims))
    train_mean_blur = np.zeros((len(n_grid), dims))
    for k, n in enumerate(n_grid):
        ho = holdout_post[n]
        tr = train_post[n]
        holdout_mean_blur[k] = ho.blur.mean(axis=0)
        train_mean_blur[k] = tr.blur.mean(axis=0)
        calibrated = calib.apply(ho.blur)
        per_n.append(
            {
                "n": n,
                "item_ids": scripted[:n],
                "item_kinds": kinds[:n],
                "holdout": {
                    "mean_blur_per_dim": [float(v) for v in holdout_mean_blur[k]],
                    "mean_blur": float(holdout_mean_blur[k].mean()),
                    "mean_blur_calibrated_per_dim": [
                        float(v) for v in calibrated.mean(axis=0)
                    ],
                    "mean_blur_calibrated": float(calibrated.mean()),
                    "agreement_with_full_matrix": agreement(ho.theta, theta_full_holdout),
                    "n_non_pd_hessians": len(ho.non_pd_rows),
                    "newton_iters": ho.n_iters,
                    "max_gradient": ho.max_grad,
                },
                "train": {
                    "mean_blur_per_dim": [float(v) for v in train_mean_blur[k]],
                    "mean_blur": float(train_mean_blur[k].mean()),
                    "agreement_with_full_matrix": agreement(tr.theta, theta_full_train),
                    "n_non_pd_hessians": len(tr.non_pd_rows),
                },
            }
        )
        if verbose:
            row = per_n[-1]["holdout"]
            print(
                f"  N={n:2d}  blur {row['mean_blur']:.4f}"
                f"  r {row['agreement_with_full_matrix']['r_mean']:.4f}"
                f"  rmse {row['agreement_with_full_matrix']['rmse_mean']:.4f}"
            )

    mono_holdout = monotonicity_report(
        n_grid,
        holdout_mean_blur,
        blur_by_n=np.stack([holdout_post[n].blur for n in n_grid]),
    )
    mono_train = monotonicity_report(n_grid, train_mean_blur)

    # the theorem check: the same sweep with theta pinned at its N=15 value, so
    # the information matrices nest exactly and monotonicity cannot fail unless
    # the arithmetic is wrong. Any rise in the sweep above and not here is the
    # MAP moving, not the information shrinking.
    fixed_blur = blur_at_fixed_theta(
        codes_holdout, interview_params, holdout_post[n_grid[-1]].theta, n_grid
    )
    mono_fixed = monotonicity_report(
        n_grid, fixed_blur.mean(axis=1), blur_by_n=fixed_blur
    )
    if verbose:
        print(
            f"monotonicity (held-out): {mono_holdout['verdict']}, "
            f"{mono_holdout['n_dims_monotone']}/{dims} dimensions weakly shrinking; "
            f"largest rise {mono_holdout['largest_increase_any_dim']:.2e}"
        )
        print(
            f"  at a fixed theta (nested information): {mono_fixed['verdict']}, "
            f"largest rise {mono_fixed['largest_increase_any_dim']:.2e}"
        )

    # ---- public-profile-only baseline ------------------------------------
    if verbose:
        print("public-profile-only ridge baseline ...")
    baseline = evaluate_profile_baseline(
        transcripts.profiles,
        train_pids,
        holdout_pids,
        theta_full_train,
        theta_full_holdout,
    )
    if verbose:
        print(
            f"  baseline held-out r {baseline['holdout']['agreement_with_full_matrix']['r_mean']:.4f}"
            f"  rmse {baseline['holdout']['agreement_with_full_matrix']['rmse_mean']:.4f}"
        )

    # ---- write ------------------------------------------------------------
    np.savez_compressed(
        estimates_path,
        n_grid=np.array(n_grid),
        dims=np.array(dims),
        scripted_item_ids=np.array(scripted),
        persona_train_ids=np.array(train_pids),
        persona_holdout_ids=np.array(holdout_pids),
        theta_train_by_n=np.stack([train_post[n].theta for n in n_grid]),
        blur_train_by_n=np.stack([train_post[n].blur for n in n_grid]),
        theta_holdout_by_n=np.stack([holdout_post[n].theta for n in n_grid]),
        blur_holdout_by_n=np.stack([holdout_post[n].blur for n in n_grid]),
        cov_holdout_by_n=np.stack([holdout_post[n].cov for n in n_grid]),
        blur_kappa=calib.kappa,
        blur_holdout_calibrated_by_n=np.stack(
            [calib.apply(holdout_post[n].blur) for n in n_grid]
        ),
        theta_baseline_holdout=np.asarray(baseline["holdout"]["theta"], dtype=float),
    )

    report = {
        "schema": SCHEMA,
        "module": "src.model.person_encoder",
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {
            "transcripts": str(transcripts_path),
            "fit": str(fit_path),
            "splits": str(splits_path),
            "fields_read_from_transcripts": ["pid", "profile", "turns.item_id", "turns.answer"],
        },
        "config": {
            "dims": dims,
            "n_grid": n_grid,
            "prior": "theta ~ N(0, I), the prior the Stage 2 fit uses",
            "likert_model": "graded response, 4 ordered thresholds (src.model.mirt)",
            "binary_model": "2PL (src.model.mirt)",
            "item_parameters": "fitted TRAINING items, linked metric, frozen",
            "map_solver": "damped Newton with per-persona backtracking",
            "hessian": f"central differences on the analytic gradient, step {HESSIAN_STEP}",
            "metric_note": (
                "loadings are read after linking, so theta here is on the same "
                "axis and unit as theta_train / theta_holdout in the fit archive"
            ),
            "open_answers_used": False,
            "llm_used": False,
        },
        "interview": {
            "scripted_closed": scripted,
            "item_kinds": kinds,
            "fitted_discrimination": [
                float(v) for v in np.linalg.norm(interview_params.loadings, axis=1)
            ],
            "all_from_training_items": True,
        },
        "splits": {
            "n_train_personas": len(train_pids),
            "n_holdout_personas": len(holdout_pids),
            "source": str(splits_path),
        },
        "blur_calibration": calib.to_json(),
        "per_n": per_n,
        "monotonicity": {
            "bar": (
                "PREREGISTRATION Gate 3: blur shrinks monotonically in "
                "expectation as N grows. Graded on the mean over held-out "
                "personas, per dimension, at zero tolerance."
            ),
            "holdout": mono_holdout,
            "train": mono_train,
            "holdout_at_fixed_theta": mono_fixed,
            "reading": (
                "at a fixed theta the information matrices nest, so the blur "
                "curve is monotone by construction and the fixed-theta block "
                "shows it. In the real sweep the MAP moves between prefixes and "
                "the observed information is read at the new MAP, which is the "
                "only source of the small rises in the held-out block."
            ),
        },
        "profile_baseline": baseline["report"],
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
        prog="python -m src.model.person_encoder",
        description=(
            "Posterior over a persona's latent vector from the closed interview "
            "answers, for every prefix length of the frozen scripted order."
        ),
    )
    parser.add_argument("--transcripts", type=Path, required=True, help="transcripts jsonl")
    parser.add_argument("--fit", type=Path, required=True, help="Stage 2 fit npz")
    parser.add_argument("--splits", type=Path, required=True, help="frozen splits manifest")
    parser.add_argument("--out", type=Path, required=True, help="report json to write")
    parser.add_argument(
        "--estimates",
        type=Path,
        default=None,
        help="npz of per-persona theta and blur (default: --out with .npz)",
    )
    parser.add_argument(
        "--calibrate-at", type=int, default=15, help="prefix length the blur calibration uses"
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    run(
        args.transcripts,
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
