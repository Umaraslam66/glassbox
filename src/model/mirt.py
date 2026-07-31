"""Multidimensional item-response fit for Stage 2 (PRD M3a, PREREGISTRATION section 6, Gate 2).

    python -m src.model.mirt \
        --answers data/runs/full_sweep/answers_noised.jsonl \
        --bank data/public/bank_items.json \
        --splits experiments/splits_v1.json \
        --out results/stage2

System-side code. It sees three things and nothing else: the noised answer
matrix, the public item bank (item id, type, options) and the frozen splits
manifest (which persona ids and which item ids are held out). It never sees a
planted trait value, a designed loading, a persona card or a response
distribution -- ``tests/test_wall.py`` enforces that statically, and the
loader here reads exactly one field off each answer record, ``answer``.

Model
-----
Person ``n`` has a latent vector theta_n with ``d`` dimensions (default 8).
Item ``i`` has a loading vector a_i with the same width.

  * Likert-5 item: graded response. With z = a_i . theta_n and four ordered
    thresholds tau_i1 < ... < tau_i4,

        P(Y >= k) = sigmoid(z - tau_i,k-1)   for k = 2..5,
        P(Y = k)  = P(Y >= k) - P(Y >= k+1).

  * Binary item: 2PL, P(yes) = sigmoid(z - b_i).

Ordering of the thresholds is built into the parameterization (a free base plus
three softplus gaps), so no constraint can be violated during the search.

Fitting
-------
Joint MAP over the training personas' theta and the training items' parameters,
full-batch Adam on plain numpy in float64. Prior theta ~ N(0, I); loadings get
a weak ridge (lambda = 0.01) purely for identifiability. Nothing else is
penalized.

The fitted axes are arbitrary up to an orthogonal rotation -- the likelihood,
the prior and the ridge are all rotation invariant. That is expected and is not
resolved here; the grader aligns the axes afterwards. Restart agreement below
is therefore measured after rotating the restarts onto each other.

Then, with the training item parameters frozen:

  * **held-out personas** get a MAP theta from their answers to the 202
    training items only, plus a Laplace posterior covariance (inverse of the
    observed Hessian of the per-person negative log posterior). The
    per-dimension square roots of its diagonal are the "blur". The full
    covariance is written out too, because the grader has to rotate it into the
    planted axes before the coverage question means anything.

  * **held-out items** get their parameters fitted from the 400 training
    personas' answers to that item, with those personas' theta frozen.

Diagnostics written alongside: three restarts of the main fit with their loss
curves and pairwise agreement, and a held-out likelihood curve over
d in {2, 4, 6, 8, 10, 12}.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

# --------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------

#: Planted width of the latent space (PREREGISTRATION section 2: eight dimensions).
DEFAULT_DIMS = 8

#: Weak ridge on the loadings; identifiability only, not shrinkage with intent.
DEFAULT_RIDGE = 0.01

#: Restart seeds for the main fit.
DEFAULT_SEEDS = (0, 1, 2)

#: Latent widths for the dimensionality curve.
DEFAULT_DIM_GRID = (2, 4, 6, 8, 10, 12)

#: Answer round used for fitting. The retest round is a Gate-1 artifact.
MAIN_ROUND = "main"

LIKERT = "likert5"
BINARY = "binary"

#: Category count for a Likert-5 item, hence four thresholds.
LIKERT_CATEGORIES = 5
LIKERT_CUTS = LIKERT_CATEGORIES - 1

_EPS = 1e-12
_MIN_GAP = 0.05


# --------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Dataset:
    """The answer matrix, coded, with its row and column labels.

    ``codes`` holds the category index: 0..4 for a Likert-5 item, 0/1 for a
    binary item (yes = 1). ``-1`` marks a cell with no answer; the matrix from
    the full sweep has none, and :func:`missing_report` says so out loud.
    """

    persona_ids: list[str]
    item_ids: list[str]
    item_kinds: list[str]
    codes: np.ndarray  # (n_personas, n_items) int16

    @property
    def n_personas(self) -> int:
        return len(self.persona_ids)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    def subset(self, persona_ids: Sequence[str], item_ids: Sequence[str]) -> "Dataset":
        prow = {pid: i for i, pid in enumerate(self.persona_ids)}
        icol = {iid: j for j, iid in enumerate(self.item_ids)}
        rows = np.array([prow[p] for p in persona_ids], dtype=np.int64)
        cols = np.array([icol[i] for i in item_ids], dtype=np.int64)
        kinds = [self.item_kinds[c] for c in cols]
        return Dataset(
            persona_ids=list(persona_ids),
            item_ids=list(item_ids),
            item_kinds=kinds,
            codes=self.codes[np.ix_(rows, cols)].copy(),
        )


def load_bank(path: Path) -> dict[str, dict[str, Any]]:
    """Item id -> the public metadata this module is allowed to use."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    items = payload["items"] if isinstance(payload, dict) else payload
    bank: dict[str, dict[str, Any]] = {}
    for item in items:
        kind = item["type"]
        if kind not in (LIKERT, BINARY):
            raise ValueError(f"item {item['item_id']}: unknown type {kind!r}")
        bank[item["item_id"]] = {
            "type": kind,
            "options": [option["value"] for option in item["options"]],
        }
    return bank


def _code_answer(answer: Any, kind: str) -> int:
    """Map a recorded answer onto its category index."""
    if kind == LIKERT:
        value = int(answer)
        if not 1 <= value <= LIKERT_CATEGORIES:
            raise ValueError(f"Likert answer out of range: {answer!r}")
        return value - 1
    text = str(answer).strip().lower()
    if text in ("yes", "1", "true"):
        return 1
    if text in ("no", "0", "false"):
        return 0
    raise ValueError(f"binary answer not understood: {answer!r}")


def load_answers(
    path: Path,
    bank: dict[str, dict[str, Any]],
    *,
    round_name: str = MAIN_ROUND,
) -> Dataset:
    """Read one round of the noised answer file into a coded matrix.

    Exactly one answer field is read (``answer``). Every other field on the
    record is ignored on purpose: the file also carries the responder's
    pre-noise string under ``raw``, which is upstream material this side of the
    Wall has no business fitting to.
    """
    persona_index: dict[str, int] = {}
    item_ids = sorted(bank)
    item_index = {iid: j for j, iid in enumerate(item_ids)}
    kinds = [bank[iid]["type"] for iid in item_ids]

    rows: list[list[int]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("round") != round_name:
                continue
            iid = record["item_id"]
            col = item_index.get(iid)
            if col is None:
                continue
            pid = record["pid"]
            row = persona_index.get(pid)
            if row is None:
                row = len(persona_index)
                persona_index[pid] = row
                rows.append([-1] * len(item_ids))
            rows[row][col] = _code_answer(record["answer"], kinds[col])

    persona_ids = sorted(persona_index)
    order = [persona_index[pid] for pid in persona_ids]
    codes = np.array([rows[i] for i in order], dtype=np.int16)
    return Dataset(persona_ids=persona_ids, item_ids=item_ids, item_kinds=kinds, codes=codes)


def load_splits(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def missing_report(data: Dataset) -> dict[str, Any]:
    """How complete the matrix is, cell by cell."""
    missing = np.argwhere(data.codes < 0)
    return {
        "n_personas": data.n_personas,
        "n_items": data.n_items,
        "n_cells_expected": data.n_personas * data.n_items,
        "n_cells_present": int((data.codes >= 0).sum()),
        "n_cells_missing": int(missing.shape[0]),
        "complete": bool(missing.shape[0] == 0),
        "missing_examples": [
            [data.persona_ids[r], data.item_ids[c]] for r, c in missing[:20]
        ],
    }


# --------------------------------------------------------------------------
# parameterization
# --------------------------------------------------------------------------


def _relative_gain(best: float, loss: float) -> float:
    """How much this step improved on the best loss so far, relatively."""
    if not np.isfinite(best):
        return float("inf")
    return (best - loss) / max(abs(best), 1.0)


def softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)


def inv_softplus(y: np.ndarray) -> np.ndarray:
    y = np.maximum(y, 1e-6)
    return y + np.log(-np.expm1(-y))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * x))


@dataclass
class ItemParams:
    """Loadings plus cut points, for one block of items.

    ``cut_raw`` is unconstrained. For a Likert item the four thresholds are
    ``tau_1 = cut_raw[0]`` and ``tau_j = tau_{j-1} + softplus(cut_raw[j])``, so
    they are ordered by construction. For a binary item only ``cut_raw[0]`` is
    used and it is the difficulty ``b``.
    """

    loadings: np.ndarray  # (n_items, dims)
    cut_raw: np.ndarray  # (n_items, LIKERT_CUTS)
    kinds: list[str] = field(default_factory=list)

    @property
    def is_likert(self) -> np.ndarray:
        return np.array([k == LIKERT for k in self.kinds], dtype=bool)

    def thresholds(self) -> np.ndarray:
        """Ordered Likert thresholds, (n_items, 4). Meaningless on binary rows."""
        gaps = softplus(self.cut_raw[:, 1:])
        return self.cut_raw[:, :1] + np.concatenate(
            [np.zeros((self.cut_raw.shape[0], 1)), np.cumsum(gaps, axis=1)], axis=1
        )

    def difficulty(self) -> np.ndarray:
        """Binary difficulty b, (n_items,). Meaningless on Likert rows."""
        return self.cut_raw[:, 0]

    def copy(self) -> "ItemParams":
        return ItemParams(self.loadings.copy(), self.cut_raw.copy(), list(self.kinds))


def init_item_params(
    data: Dataset, dims: int, rng: np.random.Generator, *, scale: float = 0.3
) -> ItemParams:
    """Small random loadings; cut points placed on the observed marginals."""
    n_items = data.n_items
    loadings = rng.normal(0.0, scale, size=(n_items, dims))
    cut_raw = np.zeros((n_items, LIKERT_CUTS))

    for j, kind in enumerate(data.item_kinds):
        column = data.codes[:, j]
        column = column[column >= 0]
        if column.size == 0:
            continue
        if kind == LIKERT:
            taus = []
            for k in range(1, LIKERT_CATEGORIES):
                share = float(np.mean(column >= k))
                share = min(max(share, 0.02), 0.98)
                taus.append(-np.log(share / (1.0 - share)))
            taus = np.maximum.accumulate(np.array(taus))
            gaps = np.maximum(np.diff(taus), _MIN_GAP)
            cut_raw[j, 0] = taus[0]
            cut_raw[j, 1:] = inv_softplus(gaps)
        else:
            share = float(np.mean(column == 1))
            share = min(max(share, 0.02), 0.98)
            cut_raw[j, 0] = -np.log(share / (1.0 - share))
    return ItemParams(loadings=loadings, cut_raw=cut_raw, kinds=list(data.item_kinds))


# --------------------------------------------------------------------------
# likelihood and gradients
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Blocks:
    """Column indices and observed codes split by item type, cached once."""

    likert_cols: np.ndarray
    binary_cols: np.ndarray
    likert_codes: np.ndarray  # (n_personas, n_likert) 0..4, -1 missing
    binary_codes: np.ndarray  # (n_personas, n_binary) 0/1, -1 missing


def make_blocks(data: Dataset) -> _Blocks:
    kinds = np.array(data.item_kinds)
    likert_cols = np.flatnonzero(kinds == LIKERT)
    binary_cols = np.flatnonzero(kinds == BINARY)
    return _Blocks(
        likert_cols=likert_cols,
        binary_cols=binary_cols,
        likert_codes=data.codes[:, likert_cols].astype(np.int64),
        binary_codes=data.codes[:, binary_cols].astype(np.int64),
    )


def _loglik_and_grads(
    theta: np.ndarray,
    params: ItemParams,
    blocks: _Blocks,
    *,
    want_item_grads: bool = True,
) -> tuple[float, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Total log-likelihood and its gradients.

    Returns ``(loglik, d/dtheta, d/dloadings, d/dcut_raw)``; the item gradients
    are ``None`` when not asked for.
    """
    n_persons = theta.shape[0]
    n_items = params.loadings.shape[0]
    z = theta @ params.loadings.T  # (n_persons, n_items)
    dz = np.zeros_like(z)
    grad_cut = np.zeros_like(params.cut_raw) if want_item_grads else None
    total = 0.0

    # ---- Likert block: graded response -----------------------------------
    lc = blocks.likert_cols
    if lc.size:
        taus = params.thresholds()[lc]  # (n_likert, 4)
        zl = z[:, lc]
        seen = sigmoid(zl[:, :, None] - taus[None, :, :])  # P(Y >= k), k = 2..5
        ones = np.ones((n_persons, lc.size, 1))
        zeros = np.zeros((n_persons, lc.size, 1))
        cumulative = np.concatenate([ones, seen, zeros], axis=2)  # (n, i, 6)
        slope = cumulative * (1.0 - cumulative)  # zero on the two padded ends

        codes = blocks.likert_codes
        observed = codes >= 0
        safe = np.where(observed, codes, 0)
        idx_lo = safe[:, :, None]
        idx_hi = safe[:, :, None] + 1
        p_lo = np.take_along_axis(cumulative, idx_lo, axis=2)[:, :, 0]
        p_hi = np.take_along_axis(cumulative, idx_hi, axis=2)[:, :, 0]
        prob = np.maximum(p_lo - p_hi, _EPS)
        total += float(np.sum(np.where(observed, np.log(prob), 0.0)))

        w_lo = np.take_along_axis(slope, idx_lo, axis=2)[:, :, 0]
        w_hi = np.take_along_axis(slope, idx_hi, axis=2)[:, :, 0]
        inv_p = np.where(observed, 1.0 / prob, 0.0)
        dz[:, lc] = inv_p * (w_lo - w_hi)

        if want_item_grads:
            # d logP / d tau_j: -w_j when j is the lower boundary of the
            # observed category, +w_j when it is the upper one.
            grad_tau = np.zeros((lc.size, LIKERT_CUTS))
            contrib_lo = -inv_p * w_lo  # applies to tau index safe (when >= 1)
            contrib_hi = inv_p * w_hi  # applies to tau index safe + 1 (when <= 4)
            for j in range(1, LIKERT_CATEGORIES):  # tau_j sits at cumulative column j
                mask_lo = observed & (safe == j)
                mask_hi = observed & (safe + 1 == j)
                grad_tau[:, j - 1] = np.sum(
                    np.where(mask_lo, contrib_lo, 0.0) + np.where(mask_hi, contrib_hi, 0.0),
                    axis=0,
                )
            # chain through the base-plus-softplus-gaps parameterization
            tail = np.cumsum(grad_tau[:, ::-1], axis=1)[:, ::-1]  # sum_{j >= m}
            grad_cut[lc, 0] = tail[:, 0]
            grad_cut[lc, 1:] = sigmoid(params.cut_raw[lc, 1:]) * tail[:, 1:]

    # ---- binary block: 2PL ----------------------------------------------
    bc = blocks.binary_cols
    if bc.size:
        b = params.difficulty()[bc]
        zb = z[:, bc]
        p = sigmoid(zb - b[None, :])
        codes = blocks.binary_codes
        observed = codes >= 0
        y = np.where(observed, codes, 0).astype(np.float64)
        total += float(
            np.sum(
                np.where(
                    observed,
                    y * np.log(np.maximum(p, _EPS)) + (1 - y) * np.log(np.maximum(1 - p, _EPS)),
                    0.0,
                )
            )
        )
        resid = np.where(observed, y - p, 0.0)
        dz[:, bc] = resid
        if want_item_grads:
            grad_cut[bc, 0] = -np.sum(resid, axis=0)

    grad_theta = dz @ params.loadings
    grad_load = dz.T @ theta if want_item_grads else None
    assert n_items == params.loadings.shape[0]
    return total, grad_theta, grad_load, grad_cut


# --------------------------------------------------------------------------
# Adam
# --------------------------------------------------------------------------


class _Adam:
    def __init__(self, shapes: Sequence[tuple[int, ...]], lr: float) -> None:
        self.lr = lr
        self.m = [np.zeros(s) for s in shapes]
        self.v = [np.zeros(s) for s in shapes]
        self.t = 0

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        bc1 = 1.0 - b1**self.t
        bc2 = 1.0 - b2**self.t
        for k, (p, g) in enumerate(zip(params, grads)):
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * (g * g)
            p -= self.lr * (self.m[k] / bc1) / (np.sqrt(self.v[k] / bc2) + eps)


# --------------------------------------------------------------------------
# the joint fit
# --------------------------------------------------------------------------


@dataclass
class FitResult:
    theta: np.ndarray
    params: ItemParams
    loss_curve: list[float]
    n_iters: int
    converged: bool
    final_loss: float
    final_loglik: float
    seed: int
    dims: int
    seconds: float


def objective(
    theta: np.ndarray, params: ItemParams, blocks: _Blocks, ridge: float
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
    """Negative log posterior and its gradients."""
    loglik, g_theta, g_load, g_cut = _loglik_and_grads(theta, params, blocks)
    prior = 0.5 * float(np.sum(theta**2))
    pen = 0.5 * ridge * float(np.sum(params.loadings**2))
    loss = -loglik + prior + pen
    return loss, -g_theta + theta, -g_load + ridge * params.loadings, -g_cut, loglik


def fit_joint(
    data: Dataset,
    *,
    dims: int = DEFAULT_DIMS,
    seed: int = 0,
    ridge: float = DEFAULT_RIDGE,
    lr: float = 0.05,
    max_iters: int = 4000,
    tol: float = 1e-7,
    patience: int = 50,
    verbose: bool = False,
) -> FitResult:
    """Joint MAP over person vectors and item parameters."""
    started = time.time()
    rng = np.random.default_rng(seed)
    blocks = make_blocks(data)
    theta = rng.normal(0.0, 0.3, size=(data.n_personas, dims))
    params = init_item_params(data, dims, rng)

    opt = _Adam([theta.shape, params.loadings.shape, params.cut_raw.shape], lr)
    curve: list[float] = []
    best = np.inf
    stale = 0
    converged = False
    loglik = float("nan")

    for step in range(1, max_iters + 1):
        loss, g_theta, g_load, g_cut, loglik = objective(theta, params, blocks, ridge)
        curve.append(loss)
        opt.step([theta, params.loadings, params.cut_raw], [g_theta, g_load, g_cut])
        if verbose and step % 500 == 0:
            print(f"    iter {step:5d}  loss {loss:,.2f}")
        rel = _relative_gain(best, loss)
        if rel < tol:
            stale += 1
            if stale >= patience:
                converged = True
                break
        else:
            stale = 0
        best = min(best, loss)

    loss, _, _, _, loglik = objective(theta, params, blocks, ridge)
    curve.append(loss)
    return FitResult(
        theta=theta,
        params=params,
        loss_curve=curve,
        n_iters=len(curve) - 1,
        converged=converged,
        final_loss=loss,
        final_loglik=loglik,
        seed=seed,
        dims=dims,
        seconds=time.time() - started,
    )


# --------------------------------------------------------------------------
# scoring people with the item parameters frozen
# --------------------------------------------------------------------------


def score_personas(
    data: Dataset,
    params: ItemParams,
    *,
    dims: int,
    seed: int = 0,
    lr: float = 0.05,
    max_iters: int = 3000,
    tol: float = 1e-9,
    patience: int = 50,
) -> tuple[np.ndarray, list[float]]:
    """MAP theta for every persona in ``data``, item parameters held fixed."""
    rng = np.random.default_rng(seed)
    blocks = make_blocks(data)
    theta = rng.normal(0.0, 0.3, size=(data.n_personas, dims))
    opt = _Adam([theta.shape], lr)
    curve: list[float] = []
    best = np.inf
    stale = 0
    for _ in range(max_iters):
        loglik, g_theta, _, _ = _loglik_and_grads(theta, params, blocks, want_item_grads=False)
        loss = -loglik + 0.5 * float(np.sum(theta**2))
        curve.append(loss)
        opt.step([theta], [-g_theta + theta])
        rel = _relative_gain(best, loss)
        if rel < tol:
            stale += 1
            if stale >= patience:
                break
        else:
            stale = 0
        best = min(best, loss)
    return theta, curve


def _person_grad(
    theta_n: np.ndarray, params: ItemParams, blocks: _Blocks, row: int
) -> np.ndarray:
    """Gradient of one person's negative log posterior at ``theta_n``."""
    one = _Blocks(
        likert_cols=blocks.likert_cols,
        binary_cols=blocks.binary_cols,
        likert_codes=blocks.likert_codes[row : row + 1],
        binary_codes=blocks.binary_codes[row : row + 1],
    )
    _, g_theta, _, _ = _loglik_and_grads(
        theta_n[None, :], params, one, want_item_grads=False
    )
    return -g_theta[0] + theta_n


def laplace_covariance(
    data: Dataset,
    params: ItemParams,
    theta: np.ndarray,
    *,
    step: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Per-person Laplace posterior covariance and its per-dimension blur.

    The Hessian of the per-person negative log posterior is taken by central
    differences on the analytic gradient, symmetrized, then inverted. Any
    person whose Hessian is not positive definite is returned in the third
    slot rather than quietly patched.
    """
    blocks = make_blocks(data)
    n, dims = theta.shape
    covs = np.zeros((n, dims, dims))
    blur = np.zeros((n, dims))
    bad: list[int] = []
    for row in range(n):
        base = theta[row]
        hess = np.zeros((dims, dims))
        for d in range(dims):
            bump = np.zeros(dims)
            bump[d] = step
            g_plus = _person_grad(base + bump, params, blocks, row)
            g_minus = _person_grad(base - bump, params, blocks, row)
            hess[:, d] = (g_plus - g_minus) / (2 * step)
        hess = 0.5 * (hess + hess.T)
        eigs = np.linalg.eigvalsh(hess)
        if eigs.min() <= 0:
            bad.append(row)
            cov = np.linalg.pinv(hess)
        else:
            cov = np.linalg.inv(hess)
        covs[row] = cov
        blur[row] = np.sqrt(np.maximum(np.diag(cov), 0.0))
    return blur, covs, bad


# --------------------------------------------------------------------------
# fixing the latent metric
# --------------------------------------------------------------------------


def link_scale(theta: np.ndarray, covs: np.ndarray) -> float:
    """The factor that puts the latent space back on the prior's scale.

    Joint MAP has a scale hole in it. Multiplying every person vector by ``c``
    and dividing every loading by ``c`` leaves the likelihood untouched, so the
    data alone cannot say what the unit of the latent axis is. What settles it
    in a joint fit is an accident of arithmetic: the N(0, I) prior pushes the
    person vectors toward zero and the ridge on the loadings pushes back, and
    they balance at roughly (lambda * n_items / n_personas) ** 0.25 -- about a
    quarter of the intended unit here. Correlations do not notice, but an
    interval does: a blur measured on a shrunken axis would look far too tight
    against a trait value measured on the intended one, and the coverage number
    would be an artifact of the ridge rather than a statement about the model.

    So the metric is pinned the way item-response models normally pin it: the
    latent variance is fixed at the prior's value of 1. The marginal variance
    of a trait splits into the spread of the fitted point estimates plus the
    average posterior variance around them,

        Var(theta) = Var(theta_hat) + E[posterior variance],

    and the scale factor is chosen to make that sum equal 1, averaged over
    dimensions. Both terms scale the same way, so one closed-form factor does
    it. This reads only fitted quantities -- point estimates and their Laplace
    covariances. No planted value is involved, and it is a change of unit, not
    a change of fit: the likelihood, every correlation and every ranking come
    out bit-for-bit the same.
    """
    spread = float(np.mean(theta**2))
    posterior = float(np.mean(np.diagonal(covs, axis1=1, axis2=2)))
    return float(np.sqrt(spread + posterior))


def apply_link(
    scale: float, theta: np.ndarray, params: ItemParams, covs: np.ndarray
) -> tuple[np.ndarray, ItemParams, np.ndarray]:
    """Apply the linking factor. Cut points live on the z scale and do not move."""
    linked = ItemParams(params.loadings * scale, params.cut_raw.copy(), list(params.kinds))
    return theta / scale, linked, covs / (scale**2)


# --------------------------------------------------------------------------
# fitting items with the person vectors frozen
# --------------------------------------------------------------------------


def fit_items(
    data: Dataset,
    theta: np.ndarray,
    *,
    dims: int,
    seed: int = 0,
    ridge: float = DEFAULT_RIDGE,
    lr: float = 0.05,
    max_iters: int = 4000,
    tol: float = 1e-9,
    patience: int = 50,
) -> tuple[ItemParams, list[float]]:
    """Fit item parameters for ``data``'s items with person vectors held fixed."""
    rng = np.random.default_rng(seed)
    blocks = make_blocks(data)
    params = init_item_params(data, dims, rng)
    opt = _Adam([params.loadings.shape, params.cut_raw.shape], lr)
    curve: list[float] = []
    best = np.inf
    stale = 0
    for _ in range(max_iters):
        loglik, _, g_load, g_cut = _loglik_and_grads(theta, params, blocks)
        loss = -loglik + 0.5 * ridge * float(np.sum(params.loadings**2))
        curve.append(loss)
        opt.step(
            [params.loadings, params.cut_raw],
            [-g_load + ridge * params.loadings, -g_cut],
        )
        rel = _relative_gain(best, loss)
        if rel < tol:
            stale += 1
            if stale >= patience:
                break
        else:
            stale = 0
        best = min(best, loss)
    return params, curve


# --------------------------------------------------------------------------
# likelihood scoring, restart agreement, dimensionality
# --------------------------------------------------------------------------


def mean_loglik_per_cell(data: Dataset, theta: np.ndarray, params: ItemParams) -> float:
    blocks = make_blocks(data)
    loglik, _, _, _ = _loglik_and_grads(theta, params, blocks, want_item_grads=False)
    n_cells = int((data.codes >= 0).sum())
    return loglik / n_cells


def orthogonal_map(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rotation R minimizing ||source @ R - target||, via SVD of source^T target."""
    u, _, vt = np.linalg.svd(source.T @ target)
    return u @ vt


def canonical_correlations(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Canonical correlations between two centered matrices with the same rows."""
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    qa, _ = np.linalg.qr(a)
    qb, _ = np.linalg.qr(b)
    return np.clip(np.linalg.svd(qa.T @ qb, compute_uv=False), 0.0, 1.0)


def _column_correlations(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros(a.shape[1])
    for d in range(a.shape[1]):
        x, y = a[:, d], b[:, d]
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            out[d] = float("nan")
        else:
            out[d] = float(np.corrcoef(x, y)[0, 1])
    return out


def restart_agreement(fits: Sequence[FitResult]) -> dict[str, Any]:
    """How close the restarts are, once rotated onto each other."""
    pairs = []
    for i in range(len(fits)):
        for j in range(i + 1, len(fits)):
            rot = orthogonal_map(fits[j].theta, fits[i].theta)
            theta_r = fits[j].theta @ rot
            load_r = fits[j].params.loadings @ rot
            theta_corr = _column_correlations(theta_r, fits[i].theta)
            load_corr = _column_correlations(load_r, fits[i].params.loadings)
            cca = canonical_correlations(fits[j].theta, fits[i].theta)
            pairs.append(
                {
                    "seeds": [fits[i].seed, fits[j].seed],
                    "theta_corr_per_dim": [float(v) for v in theta_corr],
                    "theta_corr_min": float(np.nanmin(theta_corr)),
                    "theta_corr_mean": float(np.nanmean(theta_corr)),
                    "loading_corr_per_dim": [float(v) for v in load_corr],
                    "loading_corr_min": float(np.nanmin(load_corr)),
                    "loading_corr_mean": float(np.nanmean(load_corr)),
                    "canonical_correlations": [float(v) for v in cca],
                    "canonical_min": float(np.min(cca)),
                }
            )
    mins = [p["theta_corr_min"] for p in pairs]
    return {
        "pairs": pairs,
        "worst_theta_corr": float(np.min(mins)) if mins else float("nan"),
        "worst_canonical": float(min(p["canonical_min"] for p in pairs)) if pairs else float("nan"),
        "worst_loading_corr": float(min(p["loading_corr_min"] for p in pairs)) if pairs else float("nan"),
    }


def dimensionality_curve(
    train: Dataset,
    holdout: Dataset,
    *,
    grid: Iterable[int] = DEFAULT_DIM_GRID,
    seed: int = 0,
    ridge: float = DEFAULT_RIDGE,
    max_iters: int = 4000,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """Held-out persona likelihood as the latent width changes.

    Two numbers per width:

      * ``mean_loglik_per_cell`` -- exactly what Stage 2 asks for: score the
        held-out personas on the training items, then read the likelihood of
        those same cells. It cannot fall as the width grows, because the same
        answers are used to place the person and to grade the placement.
      * ``split_half_loglik_per_cell`` -- the honest version: place the person
        on half the training items, grade on the other half, both directions.
        This one can and does turn over.
    """
    rows: list[dict[str, Any]] = []
    half_a = [iid for k, iid in enumerate(holdout.item_ids) if k % 2 == 0]
    half_b = [iid for k, iid in enumerate(holdout.item_ids) if k % 2 == 1]
    for dims in grid:
        started = time.time()
        fit = fit_joint(train, dims=dims, seed=seed, ridge=ridge, max_iters=max_iters)
        theta_ho, _ = score_personas(holdout, fit.params, dims=dims, seed=seed)
        in_sample = mean_loglik_per_cell(holdout, theta_ho, fit.params)

        halves = []
        for place, grade in ((half_a, half_b), (half_b, half_a)):
            place_cols = [holdout.item_ids.index(i) for i in place]
            grade_cols = [holdout.item_ids.index(i) for i in grade]
            place_data = holdout.subset(holdout.persona_ids, place)
            grade_data = holdout.subset(holdout.persona_ids, grade)
            place_params = ItemParams(
                fit.params.loadings[place_cols],
                fit.params.cut_raw[place_cols],
                [fit.params.kinds[c] for c in place_cols],
            )
            grade_params = ItemParams(
                fit.params.loadings[grade_cols],
                fit.params.cut_raw[grade_cols],
                [fit.params.kinds[c] for c in grade_cols],
            )
            theta_half, _ = score_personas(place_data, place_params, dims=dims, seed=seed)
            halves.append(mean_loglik_per_cell(grade_data, theta_half, grade_params))

        rows.append(
            {
                "dims": dims,
                "train_final_loss": fit.final_loss,
                "train_loglik_per_cell": fit.final_loglik / int((train.codes >= 0).sum()),
                "mean_loglik_per_cell": in_sample,
                "split_half_loglik_per_cell": float(np.mean(halves)),
                "split_half_each_direction": [float(v) for v in halves],
                "converged": fit.converged,
                "iters": fit.n_iters,
                "seconds": time.time() - started,
            }
        )
        if verbose:
            row = rows[-1]
            print(
                f"  d={dims:2d}  holdout loglik/cell {row['mean_loglik_per_cell']:.5f}"
                f"  split-half {row['split_half_loglik_per_cell']:.5f}"
                f"  ({row['seconds']:.1f}s)"
            )
    return rows


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _params_payload(params: ItemParams, item_ids: Sequence[str]) -> dict[str, Any]:
    taus = params.thresholds()
    diff = params.difficulty()
    is_likert = params.is_likert
    return {
        "item_ids": list(item_ids),
        "kinds": list(params.kinds),
        "loadings": params.loadings,
        "cut_raw": params.cut_raw,
        "thresholds": np.where(is_likert[:, None], taus, np.nan),
        "difficulty": np.where(is_likert, np.nan, diff),
        "discrimination": np.linalg.norm(params.loadings, axis=1),
    }


def run(
    answers: Path,
    bank_path: Path,
    splits_path: Path,
    out_dir: Path,
    *,
    dims: int = DEFAULT_DIMS,
    ridge: float = DEFAULT_RIDGE,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    max_iters: int = 4000,
    dim_grid: Sequence[int] = DEFAULT_DIM_GRID,
    skip_dim_curve: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    started = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bank = load_bank(bank_path)
    data = load_answers(answers, bank)
    splits = load_splits(splits_path)

    holdout_personas = [p for p in splits["persona_holdout"] if p in set(data.persona_ids)]
    train_personas = [p for p in data.persona_ids if p not in set(splits["persona_holdout"])]
    holdout_items = [i for i in splits["item_holdout"] if i in bank]
    train_items = [i for i in data.item_ids if i not in set(splits["item_holdout"])]

    completeness = missing_report(data)
    if verbose:
        print(
            f"matrix: {completeness['n_personas']} personas x {completeness['n_items']} items"
            f" = {completeness['n_cells_expected']:,} cells,"
            f" missing {completeness['n_cells_missing']}"
        )
        print(
            f"splits: train {len(train_personas)} personas / {len(train_items)} items,"
            f" held out {len(holdout_personas)} personas / {len(holdout_items)} items"
        )

    train = data.subset(train_personas, train_items)
    ho_persona = data.subset(holdout_personas, train_items)
    ho_items = data.subset(train_personas, holdout_items)

    # ---- main fit, three restarts ---------------------------------------
    fits: list[FitResult] = []
    for seed in seeds:
        if verbose:
            print(f"main fit, seed {seed} ...")
        fit = fit_joint(train, dims=dims, seed=seed, ridge=ridge, max_iters=max_iters, verbose=verbose)
        if verbose:
            print(
                f"  loss {fit.final_loss:,.2f} after {fit.n_iters} iters"
                f" (converged={fit.converged}, {fit.seconds:.1f}s)"
            )
        fits.append(fit)

    agreement = restart_agreement(fits)
    primary = min(fits, key=lambda f: f.final_loss)
    if verbose:
        print(
            f"restart agreement: worst per-dimension theta r {agreement['worst_theta_corr']:.4f},"
            f" worst canonical corr {agreement['worst_canonical']:.4f}"
        )
        print(f"primary solution: seed {primary.seed}")

    # ---- pin the latent metric (see link_scale) --------------------------
    _, raw_cov_train, bad_train = laplace_covariance(train, primary.params, primary.theta)
    scale = link_scale(primary.theta, raw_cov_train)
    raw_latent_sd = float(np.sqrt(np.mean(primary.theta**2)))
    theta_train, item_params, cov_train = apply_link(
        scale, primary.theta, primary.params, raw_cov_train
    )
    blur_train = np.sqrt(np.maximum(np.diagonal(cov_train, axis1=1, axis2=2), 0.0))
    if verbose:
        print(
            f"latent metric: raw fitted sd {raw_latent_sd:.4f}, linking factor {scale:.4f},"
            f" linked sd {float(np.sqrt(np.mean(theta_train ** 2))):.4f}"
        )

    # ---- held-out personas ----------------------------------------------
    if verbose:
        print("scoring held-out personas on the training items ...")
    theta_ho, ho_curve = score_personas(ho_persona, item_params, dims=dims, seed=0)
    blur_ho, cov_ho, bad_hessians = laplace_covariance(ho_persona, item_params, theta_ho)
    if verbose:
        print(
            f"  blur: median {np.median(blur_ho):.4f}, "
            f"non-positive-definite Hessians {len(bad_hessians)}"
        )

    # the same scoring in the unpinned metric, so the grader can show what the
    # raw joint-MAP scale would have done to the coverage number
    theta_ho_raw, _ = score_personas(ho_persona, primary.params, dims=dims, seed=0)
    blur_ho_raw, _, _ = laplace_covariance(ho_persona, primary.params, theta_ho_raw)

    # split-half scoring of the same personas: two independent estimates of the
    # same person from disjoint halves of the training items. How well they
    # agree is how repeatable the estimate is, which is the ceiling on how well
    # it could possibly match anything. Truth-free, so it belongs here.
    halves = []
    for parity in (0, 1):
        chosen = [iid for k, iid in enumerate(train_items) if k % 2 == parity]
        cols = [train_items.index(iid) for iid in chosen]
        half_params = ItemParams(
            item_params.loadings[cols], item_params.cut_raw[cols],
            [item_params.kinds[c] for c in cols],
        )
        theta_half, _ = score_personas(
            ho_persona.subset(holdout_personas, chosen), half_params, dims=dims, seed=0
        )
        halves.append(theta_half)

    # ---- held-out items --------------------------------------------------
    if verbose:
        print("fitting the held-out items on frozen training person vectors ...")
    ho_item_params, item_curve = fit_items(ho_items, theta_train, dims=dims, seed=0, ridge=ridge)

    # ---- dimensionality curve -------------------------------------------
    curve_rows: list[dict[str, Any]] = []
    if not skip_dim_curve:
        if verbose:
            print("dimensionality curve ...")
        curve_rows = dimensionality_curve(
            train, ho_persona, grid=dim_grid, seed=0, ridge=ridge,
            max_iters=max_iters, verbose=verbose,
        )

    # ---- write out -------------------------------------------------------
    train_payload = _params_payload(item_params, train_items)
    ho_payload = _params_payload(ho_item_params, holdout_items)

    npz_path = out_dir / "stage2_fit.npz"
    np.savez_compressed(
        npz_path,
        dims=np.array(dims),
        ridge=np.array(ridge),
        primary_seed=np.array(primary.seed),
        persona_train_ids=np.array(train_personas),
        persona_holdout_ids=np.array(holdout_personas),
        item_train_ids=np.array(train_items),
        item_holdout_ids=np.array(holdout_items),
        link_factor=np.array(scale),
        theta_train=theta_train,
        theta_holdout=theta_ho,
        blur_holdout=blur_ho,
        cov_holdout=cov_ho,
        blur_train=blur_train,
        cov_train=cov_train,
        theta_holdout_unlinked=theta_ho_raw,
        blur_holdout_unlinked=blur_ho_raw,
        theta_holdout_half_a=halves[0],
        theta_holdout_half_b=halves[1],
        train_item_kinds=np.array(train_payload["kinds"]),
        train_loadings=train_payload["loadings"],
        train_thresholds=train_payload["thresholds"],
        train_difficulty=train_payload["difficulty"],
        train_discrimination=train_payload["discrimination"],
        holdout_item_kinds=np.array(ho_payload["kinds"]),
        holdout_loadings=ho_payload["loadings"],
        holdout_thresholds=ho_payload["thresholds"],
        holdout_difficulty=ho_payload["difficulty"],
        holdout_discrimination=ho_payload["discrimination"],
        restart_theta=np.stack([f.theta for f in fits]),
        restart_loadings=np.stack([f.params.loadings for f in fits]),
        restart_seeds=np.array([f.seed for f in fits]),
    )

    diagnostics = {
        "module": "src.model.mirt",
        "inputs": {
            "answers": str(answers),
            "bank": str(bank_path),
            "splits": str(splits_path),
            "round": MAIN_ROUND,
            "answer_field_used": "answer",
        },
        "config": {
            "dims": dims,
            "ridge": ridge,
            "seeds": list(seeds),
            "max_iters": max_iters,
            "optimizer": "adam(lr=0.05) full batch, float64 numpy",
            "likert_model": "graded response, 4 ordered thresholds",
            "binary_model": "2PL",
            "theta_prior": "N(0, I)",
        },
        "matrix": completeness,
        "splits": {
            "n_train_personas": len(train_personas),
            "n_holdout_personas": len(holdout_personas),
            "n_train_items": len(train_items),
            "n_holdout_items": len(holdout_items),
        },
        "restarts": [
            {
                "seed": f.seed,
                "final_loss": f.final_loss,
                "final_loglik": f.final_loglik,
                "loglik_per_cell": f.final_loglik / int((train.codes >= 0).sum()),
                "iters": f.n_iters,
                "converged": f.converged,
                "seconds": f.seconds,
                "loss_curve_tail": f.loss_curve[-5:],
                "loss_curve_sampled": f.loss_curve[::100][:60],
            }
            for f in fits
        ],
        "restart_agreement": agreement,
        "primary_seed": primary.seed,
        "latent_metric": {
            "raw_fitted_sd": raw_latent_sd,
            "link_factor": scale,
            "linked_point_estimate_sd": float(np.sqrt(np.mean(theta_train**2))),
            "linked_mean_posterior_variance": float(
                np.mean(np.diagonal(cov_train, axis1=1, axis2=2))
            ),
            "note": (
                "latent variance pinned at the prior's value of 1 "
                "(point-estimate spread + mean posterior variance); "
                "likelihood and all correlations are unchanged"
            ),
        },
        "holdout_persona_scoring": {
            "final_loss": ho_curve[-1],
            "iters": len(ho_curve),
            "blur_median": float(np.median(blur_ho)),
            "blur_per_dim_median": [float(v) for v in np.median(blur_ho, axis=0)],
            "n_non_pd_hessians": len(bad_hessians),
            "non_pd_personas": [holdout_personas[i] for i in bad_hessians],
            "loglik_per_cell": mean_loglik_per_cell(ho_persona, theta_ho, primary.params),
        },
        "train_persona_blur": {
            "blur_median": float(np.median(blur_train)),
            "n_non_pd_hessians": len(bad_train),
        },
        "holdout_item_fit": {
            "final_loss": item_curve[-1],
            "iters": len(item_curve),
            "discrimination_median": float(np.median(ho_payload["discrimination"])),
        },
        "dimensionality_curve": curve_rows,
        "runtime_seconds": time.time() - started,
        "artifacts": {"fit": str(npz_path)},
    }

    diag_path = out_dir / "stage2_fit_diagnostics.json"
    diag_path.write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    if verbose:
        print(f"\nfit      -> {npz_path}")
        print(f"diagnostics -> {diag_path}")
        print(f"total runtime {diagnostics['runtime_seconds']:.1f}s")
    return diagnostics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.model.mirt",
        description="Fit the multidimensional item-response model for Stage 2.",
    )
    parser.add_argument("--answers", type=Path, required=True, help="noised answers, jsonl")
    parser.add_argument("--bank", type=Path, required=True, help="public item bank json")
    parser.add_argument("--splits", type=Path, required=True, help="frozen splits manifest")
    parser.add_argument("--out", type=Path, required=True, help="results directory")
    parser.add_argument("--dims", type=int, default=DEFAULT_DIMS)
    parser.add_argument("--ridge", type=float, default=DEFAULT_RIDGE)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--max-iters", type=int, default=4000)
    parser.add_argument("--dim-grid", type=int, nargs="+", default=list(DEFAULT_DIM_GRID))
    parser.add_argument("--skip-dim-curve", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    run(
        args.answers,
        args.bank,
        args.splits,
        args.out,
        dims=args.dims,
        ridge=args.ridge,
        seeds=args.seeds,
        max_iters=args.max_iters,
        dim_grid=args.dim_grid,
        skip_dim_curve=args.skip_dim_curve,
        verbose=not args.quiet,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
