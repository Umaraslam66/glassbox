"""Stage M2 estimator — per-traveler conditional logit (frozen prereg §5).

System-side code: consumes the scenario bank (attributes exactly as shown to
the traveler) and the official noised choices. Never touches planted truth.

The utility function is the frozen one, verbatim:

    V = b_time*time + b_cost*cost + b_crw*(crowding x in-vehicle time)
        + ASC_car + ASC_transit + b_late*P(late)*delay
        + b_switch*(non-habitual option)

Regressor construction per block is recorded in experiments/m2_recovery.json
before any fit ran. The 10 mode_prc scenarios' 'friction' attribute has no
term in the frozen utility and none is added (declared specification gap).

A ridge of lambda = 1e-4 keeps the penalized log-likelihood strictly concave
so every per-traveler optimum exists and is unique (numerical guard, recorded
in the experiment config; not a tuning knob).
"""

from __future__ import annotations

import numpy as np

PARAMS = ["b_time", "b_cost", "b_crw", "asc_car", "asc_transit", "b_late", "b_switch"]
RIDGE_LAMBDA = 1e-4


def option_features(block: str, attrs: dict) -> list[float]:
    """The 7 frozen regressors for one option, from attributes as shown."""
    if block == "dep_time":
        time = float(attrs.get("slack_min", 0.0))
        late = float(attrs.get("p_late", 0.0)) * float(attrs.get("late_min", 0.0))
    else:
        tmin = float(attrs.get("time_min", 0.0))
        time = tmin
        late = float(attrs.get("p_late", 0.0)) * (
            float(attrs.get("time_max", tmin)) - tmin)
    crw = float(attrs.get("crowding", 0.0)) * float(attrs.get("time_min", 0.0))
    mode = attrs.get("mode")
    switch = (1.0 - float(attrs["habitual"])) if "habitual" in attrs else 0.0
    return [time, float(attrs.get("cost_eur", 0.0)), crw,
            1.0 if mode == "car" else 0.0, 1.0 if mode == "transit" else 0.0,
            late, switch]


def build_design(bank: list[dict]) -> dict[str, dict]:
    """sid -> {X: (n_options, 7), keys: [option letters], informative: bool}."""
    design = {}
    for s in bank:
        X = np.array([option_features(s["block"], o["attrs"]) for o in s["options"]])
        design[s["sid"]] = {
            "X": X,
            "keys": [o["key"] for o in s["options"]],
            "informative": bool(np.any(np.ptp(X, axis=0) > 0)),
        }
    return design


def _fit_newton(Xs: np.ndarray, chosen: np.ndarray, mask: np.ndarray,
                weights: np.ndarray, beta0: np.ndarray | None = None,
                prior_mean: np.ndarray | None = None,
                prior_prec: np.ndarray | None = None) -> np.ndarray:
    """Penalized MNL MLE (or MAP when a prior is given) by damped Newton.

    Xs: (S, O, 7) padded design; mask: (S, O) valid options; chosen: (S,)
    index of the chosen option; weights: (S,) scenario weights (bootstrap).
    """
    n_par = Xs.shape[2]
    if prior_prec is None:
        prior_mean = np.zeros(n_par)
        prior_prec = RIDGE_LAMBDA * np.eye(n_par)
    beta = np.zeros(n_par) if beta0 is None else beta0.copy()

    def neg_pen(b):
        U = np.where(mask, Xs @ b, -np.inf)
        lse = np.log(np.exp(U - U.max(axis=1, keepdims=True)).sum(axis=1)) + U.max(axis=1)
        ll = float((weights * (U[np.arange(len(chosen)), chosen] - lse)).sum())
        d = b - prior_mean
        return -(ll - 0.5 * d @ prior_prec @ d)

    for _ in range(60):
        U = np.where(mask, Xs @ beta, -np.inf)
        P = np.exp(U - U.max(axis=1, keepdims=True))
        P /= P.sum(axis=1, keepdims=True)
        xbar = np.einsum("so,sop->sp", P, Xs)
        x_chosen = Xs[np.arange(len(chosen)), chosen]
        grad = (weights[:, None] * (x_chosen - xbar)).sum(axis=0) \
            - prior_prec @ (beta - prior_mean)
        cov = np.einsum("so,sop,soq->spq", P, Xs, Xs) \
            - np.einsum("sp,sq->spq", xbar, xbar)
        H = -(weights[:, None, None] * cov).sum(axis=0) - prior_prec
        if np.max(np.abs(grad)) < 1e-6:
            break
        step = np.linalg.solve(H, -grad)
        f0 = neg_pen(beta)
        t = 1.0
        while t > 1e-6 and neg_pen(beta + t * step) > f0 - 1e-12:
            t *= 0.5
        beta = beta + t * step
        if t <= 1e-6:
            break
    return beta


def traveler_arrays(design: dict, answers: dict[str, str],
                    fit_sids: list[str], min_scenarios: int = 20) -> tuple | None:
    """Stack one traveler's answered, informative training scenarios.

    min_scenarios guards the unregularized-ish M2 fits; short-survey MAP
    fits (M3) pass 1 because their proper prior carries identification.
    """
    rows = []
    for sid in fit_sids:
        ans = answers.get(sid)
        d = design[sid]
        if ans is None or not d["informative"] or ans not in d["keys"]:
            continue
        rows.append((d["X"], d["keys"].index(ans)))
    if len(rows) < min_scenarios:
        return None
    o_max = max(x.shape[0] for x, _ in rows)
    S = len(rows)
    Xs = np.zeros((S, o_max, len(PARAMS)))
    mask = np.zeros((S, o_max), dtype=bool)
    chosen = np.zeros(S, dtype=int)
    for i, (x, c) in enumerate(rows):
        Xs[i, : x.shape[0]] = x
        mask[i, : x.shape[0]] = True
        chosen[i] = c
    return Xs, chosen, mask


def fit_traveler(design: dict, answers: dict[str, str], fit_sids: list[str],
                 beta0: np.ndarray | None = None,
                 prior_mean: np.ndarray | None = None,
                 prior_prec: np.ndarray | None = None) -> np.ndarray | None:
    arrays = traveler_arrays(design, answers, fit_sids)
    if arrays is None:
        return None
    Xs, chosen, mask = arrays
    w = np.ones(len(chosen))
    return _fit_newton(Xs, chosen, mask, w, beta0, prior_mean, prior_prec)


def bootstrap_traveler(design: dict, answers: dict[str, str], fit_sids: list[str],
                       beta_hat: np.ndarray, n_boot: int,
                       rng: np.random.Generator) -> np.ndarray:
    """B refits with the traveler's scenarios resampled with replacement
    (frozen §5: parametric bootstrap resampling scenarios), warm-started."""
    arrays = traveler_arrays(design, answers, fit_sids)
    Xs, chosen, mask = arrays
    S = len(chosen)
    out = np.zeros((n_boot, len(PARAMS)))
    for b in range(n_boot):
        w = rng.multinomial(S, np.full(S, 1.0 / S)).astype(float)
        out[b] = _fit_newton(Xs, chosen, mask, w, beta0=beta_hat)
    return out


def shrinkage_refit(design: dict, all_answers: dict[str, dict[str, str]],
                    fit_sids: list[str], betas: dict[str, np.ndarray],
                    train_ids: list[str]) -> dict[str, np.ndarray]:
    """The frozen 'alongside' comparison fit, operationalized as declared in
    the experiment config: empirical-Bayes MAP with a Gaussian prior taken
    from the TRAINING travelers' per-traveler estimates."""
    B = np.array([betas[p] for p in train_ids if p in betas])
    mu = B.mean(axis=0)
    cov = np.cov(B.T) + 1e-6 * np.eye(len(PARAMS))
    prec = np.linalg.inv(cov)
    out = {}
    for pid, answers in all_answers.items():
        if pid not in betas:
            continue
        out[pid] = fit_traveler(design, answers, fit_sids, beta0=betas[pid],
                                prior_mean=mu, prior_prec=prec)
    return out
