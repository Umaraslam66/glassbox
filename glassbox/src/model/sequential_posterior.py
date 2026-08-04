"""Sequential (one-answer-at-a-time) version of the Stage 3 person-encoder backbone.

What this is
------------
Stage 3 scored a *fixed* list of 15 questions at every prefix length. Stage 5
asks questions one at a time and may pick the next one from the answers so far,
so the posterior has to be carried forward and updated after each answer instead
of rebuilt from a script. This module is that carried-forward posterior.

It is the **same** model as :mod:`src.model.person_encoder`:

* the same prior, ``theta ~ N(0, I)``;
* the same likelihood -- graded response on Likert-5 items, 2PL on binary ones,
  imported from :mod:`src.model.mirt` rather than restated;
* the same item parameters, the Stage 2 **fitted** loadings and thresholds of
  the closed training items (the owner's Stage 5 ruling addendum: fitted
  parameters are the confirmatory path, because the Gate 5 target is defined as
  the Gate 3 accuracy level and the two have to be comparable);
* the same Laplace covariance at the MAP.

Because the model is the same, replaying the Stage 3 scripted order through this
module reproduces the Stage 3 artifact -- ``tests/test_sequential_posterior.py``
asserts exactly that at every N, which is the integrity check for everything
Stage 5 measures.

How "one at a time" is represented
----------------------------------
The answer matrix has one column per *candidate* item and is filled with ``-1``
("not asked"). Asking an item writes its category index into that persona's
cell. Missing cells contribute nothing to the log-likelihood or its gradient
(:func:`src.model.mirt._loglik_and_grads` masks them), so a persona who has
answered five items is scored on exactly those five. Different personas may
therefore hold completely different item sets -- which is what an adaptive
interview produces -- with no change to the arithmetic.

The mode is re-found by damped Newton warm-started at the previous mode, which
is what makes the update cheap; the mode is unique (the negative log posterior
is strictly convex), so the warm start changes the route taken, not the answer.

Two Hessians, one number
------------------------
``hessian="central"`` is the Stage 3 recipe: central differences on the analytic
gradient, step 1e-4. ``hessian="analytic"`` is the closed form: both the graded
response and the 2PL log-likelihood depend on theta only through ``z = a . theta``,
so each answered item contributes ``c_i a_i a_i^T`` with a scalar curvature
``c_i >= 0`` and the Hessian is ``I + sum_i c_i a_i a_i^T`` exactly. They agree
to ~1e-10 on the blur (a unit test pins this) and the analytic one is five times
faster, which is what makes a 150-question sweep over 51 replicates a CPU job
rather than an overnight one. Both are offered; the integrity check runs in both.

WALL. Arithmetic on system-side quantities only: coded answers that were
actually given, and fitted item parameters. No planted value, no answer
distribution, no persona card, and no path to any of them appears here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from . import mirt
from .person_encoder import (
    DEFAULT_MAX_ITERS,
    DEFAULT_TOL,
    HESSIAN_STEP,
    GaussianBlock,
    Posterior,
    blocks_from_codes,
    neg_log_posterior,
    neg_log_posterior_grad,
    observed_hessian,
)

#: Eigenvalue floor used only to make a Newton *step* well posed, exactly as in
#: :func:`src.model.person_encoder.posterior`.
_STEP_EIG_FLOOR = 1e-8

#: Backtracking line-search halvings per Newton step. Same as Stage 3.
_MAX_BACKTRACK = 30

_EPS = mirt._EPS

#: The two Hessian recipes. "central" is the Stage 3 one; "analytic" is the
#: closed form, and is the default here because the sweep is long.
HESSIAN_MODES = ("analytic", "central")

#: Cell value meaning "this item has not been asked yet".
UNASKED = -1


# --------------------------------------------------------------------------
# curvature: the scalar each answered item contributes along its own direction
# --------------------------------------------------------------------------


def item_curvature(
    theta: np.ndarray, params: mirt.ItemParams, codes: np.ndarray
) -> np.ndarray:
    """``c_i`` for every persona and item: ``-d^2/dz^2 log P(answer | z)``.

    ``(n_personas, n_items)``, zero wherever the item was not asked. The full
    Hessian of the negative log posterior is then
    ``I + sum_i c_i a_i a_i^T``, because both response models see theta only
    through ``z = a_i . theta``.

    For the graded-response model with ``P = C_lo - C_hi`` (cumulative
    probabilities either side of the observed category),
    ``c = (P'/P)^2 - P''/P`` with ``C' = C(1 - C)`` and
    ``C'' = C(1 - C)(1 - 2C)``. For 2PL, ``c = p(1 - p)`` whatever the answer
    was.
    """
    theta = np.asarray(theta, dtype=float)
    codes = np.asarray(codes)
    z = theta @ params.loadings.T
    curvature = np.zeros_like(z)
    kinds = np.array(list(params.kinds))
    likert_cols = np.flatnonzero(kinds == mirt.LIKERT)
    binary_cols = np.flatnonzero(kinds == mirt.BINARY)

    if likert_cols.size:
        taus = params.thresholds()[likert_cols]
        seen = mirt.sigmoid(z[:, likert_cols][:, :, None] - taus[None, :, :])
        ones = np.ones((theta.shape[0], likert_cols.size, 1))
        cumulative = np.concatenate([ones, seen, np.zeros_like(ones)], axis=2)
        first = cumulative * (1.0 - cumulative)
        second = first * (1.0 - 2.0 * cumulative)

        block = codes[:, likert_cols].astype(np.int64)
        observed = block >= 0
        safe = np.where(observed, block, 0)[:, :, None]
        p_lo = np.take_along_axis(cumulative, safe, axis=2)[:, :, 0]
        p_hi = np.take_along_axis(cumulative, safe + 1, axis=2)[:, :, 0]
        prob = np.maximum(p_lo - p_hi, _EPS)
        d1 = (
            np.take_along_axis(first, safe, axis=2)[:, :, 0]
            - np.take_along_axis(first, safe + 1, axis=2)[:, :, 0]
        )
        d2 = (
            np.take_along_axis(second, safe, axis=2)[:, :, 0]
            - np.take_along_axis(second, safe + 1, axis=2)[:, :, 0]
        )
        curvature[:, likert_cols] = np.where(observed, (d1 / prob) ** 2 - d2 / prob, 0.0)

    if binary_cols.size:
        b = params.difficulty()[binary_cols]
        p = mirt.sigmoid(z[:, binary_cols] - b[None, :])
        block = codes[:, binary_cols].astype(np.int64)
        curvature[:, binary_cols] = np.where(block >= 0, p * (1.0 - p), 0.0)

    return curvature


def analytic_hessian(
    theta: np.ndarray,
    params: mirt.ItemParams,
    codes: np.ndarray,
    extra: Sequence[GaussianBlock] = (),
) -> np.ndarray:
    """``I + sum_i c_i a_i a_i^T`` per persona, plus any Gaussian blocks.

    The closed form of what :func:`src.model.person_encoder.observed_hessian`
    computes by differencing. Symmetrized the same way, so the two are
    interchangeable inputs to the covariance.
    """
    theta = np.asarray(theta, dtype=float)
    loadings = params.loadings
    curvature = item_curvature(theta, params, codes)
    weighted = curvature[:, :, None] * loadings[None, :, :]
    hess = np.einsum("nki,kj->nij", weighted, loadings)
    hess += np.eye(loadings.shape[1])[None, :, :]
    for block in extra:
        precision = np.asarray(block.precision, dtype=float)
        hess = hess + (precision[None, :, :] if precision.ndim == 2 else precision)
    return 0.5 * (hess + np.swapaxes(hess, 1, 2))


def hessian_at(
    theta: np.ndarray,
    params: mirt.ItemParams,
    codes: np.ndarray,
    blocks: mirt._Blocks,
    extra: Sequence[GaussianBlock] = (),
    *,
    mode: str = "analytic",
    step: float = HESSIAN_STEP,
) -> np.ndarray:
    """The Hessian of the negative log posterior, by either recipe."""
    if mode == "analytic":
        return analytic_hessian(theta, params, codes, extra)
    if mode == "central":
        return observed_hessian(theta, params, blocks, extra, step=step)
    raise ValueError(f"unknown hessian mode {mode!r}; known: {HESSIAN_MODES}")


# --------------------------------------------------------------------------
# the solver
# --------------------------------------------------------------------------


def solve(
    codes: np.ndarray,
    params: mirt.ItemParams,
    *,
    theta0: np.ndarray | None = None,
    extra: Sequence[GaussianBlock] = (),
    hessian: str = "analytic",
    max_iters: int = DEFAULT_MAX_ITERS,
    tol: float = DEFAULT_TOL,
    hessian_step: float = HESSIAN_STEP,
) -> Posterior:
    """MAP and Laplace covariance for a partly filled answer matrix.

    Identical to :func:`src.model.person_encoder.posterior` except that the
    Newton search may start somewhere other than the prior mean (``theta0``) and
    the Hessian recipe is selectable. The objective is strictly convex, so
    neither choice changes where it lands -- only how long it takes to get
    there.
    """
    codes = np.atleast_2d(np.asarray(codes))
    dims = int(params.loadings.shape[1])
    n_personas = codes.shape[0]
    blocks = blocks_from_codes(codes, params.kinds)
    extra = list(extra)
    for block in extra:
        block.validate(n_personas, dims)

    if theta0 is None:
        theta = np.zeros((n_personas, dims))
    else:
        theta = np.array(theta0, dtype=float, copy=True)
        if theta.shape != (n_personas, dims):
            raise ValueError(
                f"theta0 must be {(n_personas, dims)}, got {theta.shape}"
            )

    value = neg_log_posterior(theta, params, blocks, extra)
    grad = neg_log_posterior_grad(theta, params, blocks, extra)
    used = 0

    while used < max_iters and float(np.max(np.abs(grad))) > tol:
        used += 1
        hess = hessian_at(
            theta, params, codes, blocks, extra, mode=hessian, step=hessian_step
        )
        eigvals, eigvecs = np.linalg.eigh(hess)
        safe = np.maximum(eigvals, _STEP_EIG_FLOOR)
        proj = np.einsum("nji,nj->ni", eigvecs, grad)
        step_dir = np.einsum("nij,nj->ni", eigvecs, proj / safe)

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

    hess = hessian_at(
        theta, params, codes, blocks, extra, mode=hessian, step=hessian_step
    )
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
# what a strategy is allowed to see
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class InterviewState:
    """Everything a question-picking strategy may look at. Nothing else exists.

    ``theta`` is the current MAP, ``cov`` the Laplace covariance, ``blur`` its
    per-dimension square root, ``answered`` a boolean mask of which candidate
    items each persona has already been asked, and ``n_asked`` the count. All
    four arrays are read-only copies, so a strategy cannot reach back into the
    simulator through them.

    There is deliberately no answer matrix, no answer distribution, no persona
    identifier and no planted value on this object. A strategy sees its own
    state, the item it chose, and the answer that came back
    (:meth:`src.interview.strategies.Strategy.observe`) -- that is the whole
    interface, and ``tests/test_episodes.py`` asserts it.
    """

    theta: np.ndarray
    cov: np.ndarray
    blur: np.ndarray
    answered: np.ndarray
    n_asked: np.ndarray
    step: int

    @property
    def n_personas(self) -> int:
        return int(self.theta.shape[0])

    @property
    def n_items(self) -> int:
        return int(self.answered.shape[1])

    @property
    def dims(self) -> int:
        return int(self.theta.shape[1])


def _frozen(array: np.ndarray) -> np.ndarray:
    out = np.array(array, copy=True)
    out.setflags(write=False)
    return out


# --------------------------------------------------------------------------
# the carried-forward posterior
# --------------------------------------------------------------------------


class SequentialPosterior:
    """A posterior over every persona that is updated one answer at a time.

    Construct it with the candidate items' fitted parameters and a population
    size; it starts at the prior. :meth:`observe` records one answer per persona
    and refreshes the mode and the covariance.
    """

    def __init__(
        self,
        params: mirt.ItemParams,
        n_personas: int,
        *,
        extra: Sequence[GaussianBlock] = (),
        hessian: str = "analytic",
        max_iters: int = DEFAULT_MAX_ITERS,
        tol: float = DEFAULT_TOL,
        hessian_step: float = HESSIAN_STEP,
    ) -> None:
        if hessian not in HESSIAN_MODES:
            raise ValueError(f"unknown hessian mode {hessian!r}; known: {HESSIAN_MODES}")
        self.params = params
        self.n_personas = int(n_personas)
        self.n_items = int(params.loadings.shape[0])
        self.dims = int(params.loadings.shape[1])
        self.extra = list(extra)
        self.hessian = hessian
        self.max_iters = int(max_iters)
        self.tol = float(tol)
        self.hessian_step = float(hessian_step)
        self.reset()

    # ---- state ---------------------------------------------------------

    def reset(self) -> None:
        """Back to the prior, with nothing asked."""
        self._codes = np.full((self.n_personas, self.n_items), UNASKED, dtype=np.int16)
        self.step = 0
        self._refresh(theta0=None)

    def _refresh(self, theta0: np.ndarray | None) -> None:
        post = solve(
            self._codes,
            self.params,
            theta0=theta0,
            extra=self.extra,
            hessian=self.hessian,
            max_iters=self.max_iters,
            tol=self.tol,
            hessian_step=self.hessian_step,
        )
        self.theta = post.theta
        self.cov = post.cov
        self.blur = post.blur
        self.last_iters = post.n_iters
        self.last_max_grad = post.max_grad
        self.non_pd_rows = post.non_pd_rows

    @property
    def answered(self) -> np.ndarray:
        """Boolean mask of which items each persona has been asked."""
        return self._codes >= 0

    @property
    def n_asked(self) -> np.ndarray:
        return (self._codes >= 0).sum(axis=1)

    def state(self) -> InterviewState:
        """The read-only view a strategy is handed."""
        return InterviewState(
            theta=_frozen(self.theta),
            cov=_frozen(self.cov),
            blur=_frozen(self.blur),
            answered=_frozen(self.answered),
            n_asked=_frozen(self.n_asked),
            step=self.step,
        )

    # ---- the update ----------------------------------------------------

    def observe(self, columns: np.ndarray, answers: np.ndarray) -> None:
        """Record one answer per persona and re-find the mode.

        ``columns`` is the item column each persona was just asked and
        ``answers`` the category index each gave. Asking the same persona the
        same item twice is an error: an interview does not re-ask, and a
        strategy that tried to would be scoring one answer twice.
        """
        columns = np.asarray(columns, dtype=np.int64).reshape(-1)
        answers = np.asarray(answers, dtype=np.int64).reshape(-1)
        if columns.shape != (self.n_personas,) or answers.shape != (self.n_personas,):
            raise ValueError(
                f"expected one column and one answer per persona "
                f"({self.n_personas}), got {columns.shape} and {answers.shape}"
            )
        if columns.min() < 0 or columns.max() >= self.n_items:
            raise ValueError("item column out of range")
        rows = np.arange(self.n_personas)
        if np.any(self._codes[rows, columns] >= 0):
            repeated = int(np.flatnonzero(self._codes[rows, columns] >= 0)[0])
            raise ValueError(
                f"persona row {repeated} was asked item column "
                f"{int(columns[repeated])} twice"
            )
        if np.any(answers < 0):
            raise ValueError("an observed answer cannot be the not-asked marker")
        self._codes[rows, columns] = answers.astype(np.int16)
        self.step += 1
        self._refresh(theta0=self.theta)

    def observe_matrix(self, codes: np.ndarray) -> None:
        """Replace the whole answer matrix at once (used by the full-bank arm)."""
        codes = np.asarray(codes, dtype=np.int16)
        if codes.shape != (self.n_personas, self.n_items):
            raise ValueError(
                f"codes must be {(self.n_personas, self.n_items)}, got {codes.shape}"
            )
        self._codes = codes.copy()
        self.step = int((codes >= 0).sum(axis=1).max())
        self._refresh(theta0=self.theta)


# --------------------------------------------------------------------------
# the information heuristic's arithmetic
# --------------------------------------------------------------------------


def answer_probabilities(
    theta: np.ndarray, params: mirt.ItemParams
) -> tuple[np.ndarray, np.ndarray]:
    """``(probabilities, curvatures)``, both ``(n_personas, n_items, 5)``.

    Column ``y`` of ``probabilities`` is ``P(answer = y | theta)`` under the
    fitted model, plugged in at the current MAP; ``curvatures[:, :, y]`` is the
    ``c`` that answer would contribute (see :func:`item_curvature`). Binary
    items use columns 0 and 1 and leave the rest at zero probability.

    The plug-in at the MAP is the approximation, and it is the only one: the
    posterior's own spread in ``z`` is not integrated over. Stated here because
    the heuristic's definition in the artifact quotes this docstring.
    """
    theta = np.asarray(theta, dtype=float)
    n = theta.shape[0]
    n_items = params.loadings.shape[0]
    z = theta @ params.loadings.T
    probs = np.zeros((n, n_items, mirt.LIKERT_CATEGORIES))
    curv = np.zeros((n, n_items, mirt.LIKERT_CATEGORIES))
    kinds = np.array(list(params.kinds))
    likert_cols = np.flatnonzero(kinds == mirt.LIKERT)
    binary_cols = np.flatnonzero(kinds == mirt.BINARY)

    if likert_cols.size:
        taus = params.thresholds()[likert_cols]
        seen = mirt.sigmoid(z[:, likert_cols][:, :, None] - taus[None, :, :])
        ones = np.ones((n, likert_cols.size, 1))
        cumulative = np.concatenate([ones, seen, np.zeros_like(ones)], axis=2)
        first = cumulative * (1.0 - cumulative)
        second = first * (1.0 - 2.0 * cumulative)
        prob = np.maximum(cumulative[:, :, :-1] - cumulative[:, :, 1:], _EPS)
        d1 = first[:, :, :-1] - first[:, :, 1:]
        d2 = second[:, :, :-1] - second[:, :, 1:]
        probs[:, likert_cols, :] = prob
        curv[:, likert_cols, :] = (d1 / prob) ** 2 - d2 / prob

    if binary_cols.size:
        b = params.difficulty()[binary_cols]
        p = mirt.sigmoid(z[:, binary_cols] - b[None, :])
        probs[:, binary_cols, 0] = 1.0 - p
        probs[:, binary_cols, 1] = p
        curv[:, binary_cols, 0] = p * (1.0 - p)
        curv[:, binary_cols, 1] = p * (1.0 - p)

    return probs, curv


#: How the heuristic's score is defined, quoted verbatim into the artifact.
INFO_GAIN_DEFINITION = (
    "score(item i, persona p) = sum_y P(y | theta_hat_p) * "
    "[ trace(Sigma_p) - trace((Sigma_p^-1 + c_{i,y} a_i a_i^T)^-1) ], where "
    "Sigma_p is the persona's current Laplace posterior covariance, a_i the "
    "item's fitted loading vector, c_{i,y} the curvature answer y would "
    "contribute along a_i, and P(y | theta_hat_p) the fitted model's answer "
    "probability plugged in at the current MAP. By Sherman-Morrison the "
    "bracket is c ||Sigma a||^2 / (1 + c a^T Sigma a), so the score is exact "
    "for the rank-one update and needs no matrix inverse. This is expected "
    "reduction in TOTAL posterior variance (A-optimality) on the raw, "
    "uncalibrated covariance -- the system's own declared uncertainty."
)


def expected_variance_reduction(
    theta: np.ndarray,
    cov: np.ndarray,
    params: mirt.ItemParams,
) -> np.ndarray:
    """Expected drop in ``trace(Sigma)`` from asking each item, ``(n, n_items)``.

    See :data:`INFO_GAIN_DEFINITION`. Every quantity is system-side: the
    posterior the system currently declares and the fitted item parameters.
    """
    theta = np.asarray(theta, dtype=float)
    cov = np.asarray(cov, dtype=float)
    loadings = params.loadings
    probs, curv = answer_probabilities(theta, params)

    sigma_a = np.einsum("nij,kj->nki", cov, loadings)  # (n, items, dims)
    quad = np.einsum("nki,ki->nk", sigma_a, loadings)  # a^T Sigma a
    norm = np.einsum("nki,nki->nk", sigma_a, sigma_a)  # ||Sigma a||^2

    denom = 1.0 + curv * quad[:, :, None]
    gain = curv * norm[:, :, None] / np.maximum(denom, _EPS)
    return np.sum(probs * gain, axis=2)
