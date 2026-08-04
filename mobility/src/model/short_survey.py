"""Stage M3 short-survey estimator — K choices + profile (frozen prereg §5).

System-side. The traveler is seen only through K noised answers and the
public profile; identification beyond that comes from a Gaussian prior
estimated on training travelers (hierarchical shrinkage toward the
population fit, empirical Bayes — operationalization pre-declared in
experiments/m3_calibration.json). The public profile is designed
signal-free (§2), so the posterior uses the unconditional prior; the
profile enters through the profile-only baseline computed here.
"""

from __future__ import annotations

import numpy as np

from mobility.src.model.mnl import PARAMS, _fit_newton, traveler_arrays


def map_fit_with_cov(design: dict, answers: dict[str, str], sids: list[str],
                     prior_mean: np.ndarray, prior_prec: np.ndarray,
                     ) -> tuple[np.ndarray, np.ndarray] | None:
    """MAP fit on the K answered scenarios + Laplace covariance at the MAP."""
    arrays = traveler_arrays(design, answers, sids, min_scenarios=1)
    if arrays is None:
        return None
    Xs, chosen, mask = arrays
    w = np.ones(len(chosen))
    beta = _fit_newton(Xs, chosen, mask, w, beta0=prior_mean.copy(),
                       prior_mean=prior_mean, prior_prec=prior_prec)
    U = np.where(mask, Xs @ beta, -np.inf)
    P = np.exp(U - U.max(axis=1, keepdims=True))
    P /= P.sum(axis=1, keepdims=True)
    xbar = np.einsum("so,sop->sp", P, Xs)
    cov = np.einsum("so,sop,soq->spq", P, Xs, Xs) - np.einsum("sp,sq->spq", xbar, xbar)
    H = -(w[:, None, None] * cov).sum(axis=0) - prior_prec
    return beta, np.linalg.inv(-H)


def predict_probs(design: dict, sid: str, beta: np.ndarray) -> dict[str, float]:
    d = design[sid]
    u = d["X"] @ beta
    p = np.exp(u - u.max())
    p /= p.sum()
    return dict(zip(d["keys"], p.tolist()))


def knn_shares(target_pid: str, survey_sids: list[str], predict_sid: str,
               answers: dict[str, dict[str, str]], train_ids: list[str],
               k: int = 20) -> dict[str, float] | None:
    """Neighbours = training travelers by exact-match agreement on the same
    K answers; prediction = their observed shares on the target scenario."""
    mine = answers[target_pid]
    scored = []
    for pid in train_ids:
        theirs = answers[pid]
        pairs = [(mine.get(s), theirs.get(s)) for s in survey_sids]
        pairs = [(a, b) for a, b in pairs if a and b]
        if pairs:
            scored.append((sum(a == b for a, b in pairs) / len(pairs), pid))
    scored.sort(reverse=True)
    counts: dict[str, int] = {}
    for _, pid in scored[:k]:
        a = answers[pid].get(predict_sid)
        if a:
            counts[a] = counts.get(a, 0) + 1
    total = sum(counts.values())
    if not total:
        return None
    return {o: c / total for o, c in counts.items()}


def profile_group_shares(profile: dict, predict_sid: str,
                         profiles: dict[str, dict],
                         answers: dict[str, dict[str, str]],
                         train_ids: list[str]) -> dict[str, float] | None:
    """Training shares within the traveler's profile group, with the
    pre-declared back-off ladder (4 fields -> drop area_type -> age x
    occupation -> population)."""
    ladders = [
        ("age_band", "occupation_type", "household", "area_type"),
        ("age_band", "occupation_type", "household"),
        ("age_band", "occupation_type"),
        (),
    ]
    for fields in ladders:
        members = [p for p in train_ids
                   if all(profiles[p][f] == profile[f] for f in fields)]
        if len(members) >= 10:
            counts: dict[str, int] = {}
            for pid in members:
                a = answers[pid].get(predict_sid)
                if a:
                    counts[a] = counts.get(a, 0) + 1
            total = sum(counts.values())
            if total:
                return {o: c / total for o, c in counts.items()}
    return None
