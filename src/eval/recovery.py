"""Stage 2 recovery grader (PRD M4, PREREGISTRATION.md section 6, Gate 2).

    python -m src.eval.recovery \
        --fit results/stage2/stage2_fit.npz \
        --diagnostics results/stage2/stage2_fit_diagnostics.json \
        --truth <planted dir> \
        --splits experiments/splits_v1.json \
        --out results/stage2

Grader-side code: it lives in ``src/eval/``, so it may read the planted design.
The directory holding that design is a command-line argument; no path to it is
spelled out here.

What it does, in order:

1. **Alignment.** The fitted axes are arbitrary up to an orthogonal rotation.
   The rotation is found on the TRAINING personas only -- SVD of
   ``theta_hat_train.T @ theta_train`` -- and then applied unchanged to the
   held-out person vectors, to their posterior covariances, and to every fitted
   loading vector. Nothing about the held-out set touches the alignment.

2. **The four frozen Gate-2 bars**, each with its value and a pass/fail:
   trait recovery, item recovery, blur honesty, weak-item ordering.

3. **The watch list** -- distractor discrimination, the sign of two named weak
   items, the recovered TRU-RSK correlation against a planted zero, and the two
   items that were nearly flat in the raw sweep.

4. **Recovered against planted correlation matrix**, both written out with
   their difference, plus a heatmap.

5. **Per-stratum item recovery** (near / same-domain / far). Reported, no bar.

The verdict block at the top of the output JSON is the whole gate in one place.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

#: Trait order, frozen in PREREGISTRATION.md section 2.
DIMENSIONS = ("TRU", "RSK", "ENV", "PRC", "TRD", "SOC", "TEC", "LOC")

#: The planted population correlation matrix (PREREGISTRATION.md section 2).
PLANTED_SIGMA = np.array(
    [
        [1.00, 0.00, 0.10, -0.10, 0.30, 0.15, 0.15, 0.10],
        [0.00, 1.00, 0.00, -0.30, -0.20, 0.15, 0.40, -0.15],
        [0.10, 0.00, 1.00, -0.15, -0.30, 0.10, 0.05, 0.00],
        [-0.10, -0.30, -0.15, 1.00, 0.10, 0.00, -0.15, 0.10],
        [0.30, -0.20, -0.30, 0.10, 1.00, 0.10, -0.35, 0.40],
        [0.15, 0.15, 0.10, 0.00, 0.10, 1.00, 0.10, 0.20],
        [0.15, 0.40, 0.05, -0.15, -0.35, 0.10, 1.00, -0.20],
        [0.10, -0.15, 0.00, 0.10, 0.40, 0.20, -0.20, 1.00],
    ]
)

# ---- the frozen bars -----------------------------------------------------
BAR_TRAIT_R = 0.8  # every dimension, held-out personas
BAR_ITEM_R = 0.7  # median across held-out items
BAR_COVERAGE = (0.60, 0.75)  # blur honesty band

#: Named in the Stage-2 brief as things to look at whatever the bars say.
WATCH_SIGN_ITEMS = ("q229", "q234")
WATCH_FLAT_ITEMS = ("q044", "q246")
WATCH_ZERO_PAIR = ("TRU", "RSK")


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def load_planted_bank(truth_dir: Path) -> dict[str, dict[str, Any]]:
    """Designed loadings and strength class, per item."""
    payload = json.loads((Path(truth_dir) / "bank_truth.json").read_text(encoding="utf-8"))
    return {item["item_id"]: item for item in payload["items"]}


def load_planted_theta(truth_dir: Path, persona_ids: Sequence[str]) -> np.ndarray:
    """Planted trait matrix for the given personas, in the frozen dimension order."""
    rows = []
    for pid in persona_ids:
        payload = json.loads(
            (Path(truth_dir) / "theta" / f"{pid}.json").read_text(encoding="utf-8")
        )
        values = payload["theta"] if "theta" in payload else payload
        rows.append([float(values[d]) for d in DIMENSIONS])
    return np.array(rows)


def load_planted_wobble(truth_dir: Path, persona_ids: Sequence[str]) -> np.ndarray:
    out = []
    for pid in persona_ids:
        payload = json.loads(
            (Path(truth_dir) / "noise" / f"{pid}.json").read_text(encoding="utf-8")
        )
        out.append(float(payload["wobble"]))
    return np.array(out)


def designed_matrix(bank: dict[str, dict[str, Any]], item_ids: Sequence[str]) -> np.ndarray:
    """Designed 8-dimensional loading vector per item, zeros where nothing was planted."""
    out = np.zeros((len(item_ids), len(DIMENSIONS)))
    for row, iid in enumerate(item_ids):
        for dim, value in bank[iid]["loadings"].items():
            out[row, DIMENSIONS.index(dim)] = float(value)
    return out


# --------------------------------------------------------------------------
# alignment
# --------------------------------------------------------------------------


def procrustes_rotation(fitted: np.ndarray, planted: np.ndarray) -> np.ndarray:
    """Orthogonal R minimizing ||fitted @ R - planted||, fitted on training rows."""
    u, _, vt = np.linalg.svd(fitted.T @ planted)
    return u @ vt


def rotate_covariances(covs: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Push posterior covariances through the same change of basis."""
    return np.einsum("ji,njk,kl->nil", rotation, covs, rotation)


# --------------------------------------------------------------------------
# small statistics
# --------------------------------------------------------------------------


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def cosine(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    if nx < 1e-12 or ny < 1e-12:
        return float("nan")
    return float(np.dot(x, y) / (nx * ny))


def describe(values: Sequence[float]) -> dict[str, float]:
    arr = np.asarray([v for v in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "min": float(arr.min()),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
        "p75": float(np.percentile(arr, 75)),
        "max": float(arr.max()),
    }


# --------------------------------------------------------------------------
# the bars
# --------------------------------------------------------------------------


def trait_recovery(theta_hat: np.ndarray, theta_true: np.ndarray) -> dict[str, Any]:
    """Bar (a): per-dimension Pearson r on the held-out personas."""
    per_dim = {
        dim: pearson(theta_hat[:, d], theta_true[:, d]) for d, dim in enumerate(DIMENSIONS)
    }
    values = list(per_dim.values())
    worst = min(per_dim, key=lambda k: per_dim[k])
    rmse = {
        dim: float(np.sqrt(np.mean((theta_hat[:, d] - theta_true[:, d]) ** 2)))
        for d, dim in enumerate(DIMENSIONS)
    }
    return {
        "per_dimension_r": per_dim,
        "min_r": float(np.min(values)),
        "min_r_dimension": worst,
        "mean_r": float(np.mean(values)),
        "per_dimension_rmse": rmse,
        "n_personas": int(theta_hat.shape[0]),
        "bar": BAR_TRAIT_R,
        "pass": bool(np.min(values) >= BAR_TRAIT_R),
    }


def item_recovery(
    fitted: np.ndarray, designed: np.ndarray, item_ids: Sequence[str]
) -> dict[str, Any]:
    """Bar (b): per-item correlation (and cosine) between fitted and designed vectors."""
    per_item = {}
    for row, iid in enumerate(item_ids):
        per_item[iid] = {
            "r": pearson(fitted[row], designed[row]),
            "cosine": cosine(fitted[row], designed[row]),
            "fitted_norm": float(np.linalg.norm(fitted[row])),
            "designed_norm": float(np.linalg.norm(designed[row])),
        }
    rs = [v["r"] for v in per_item.values()]
    cosines = [v["cosine"] for v in per_item.values()]
    median_r = float(np.nanmedian(rs))
    return {
        "per_item": per_item,
        "r_summary": describe(rs),
        "cosine_summary": describe(cosines),
        "median_r": median_r,
        "median_cosine": float(np.nanmedian(cosines)),
        "n_items": len(item_ids),
        "n_items_with_no_designed_loading": int(
            sum(1 for row in range(len(item_ids)) if np.linalg.norm(designed[row]) < 1e-12)
        ),
        "bar": BAR_ITEM_R,
        "pass": bool(median_r >= BAR_ITEM_R),
    }


def blur_honesty(
    theta_hat: np.ndarray, blur: np.ndarray, theta_true: np.ndarray
) -> dict[str, Any]:
    """Bar (c): fraction of person-by-dimension cells with truth inside one blur."""
    inside = np.abs(theta_hat - theta_true) <= blur
    pooled = float(inside.mean())
    per_dim = {dim: float(inside[:, d].mean()) for d, dim in enumerate(DIMENSIONS)}
    low, high = BAR_COVERAGE
    error = theta_hat - theta_true
    predicted_var = float(np.mean(blur**2))
    actual_var = float(np.mean(error**2))
    # what the blur would have to be multiplied by to hit the nominal 68%.
    # Reported to size the miss; the blur itself is left exactly as fitted.
    ratios = np.abs(error) / np.maximum(blur, 1e-12)
    return {
        "diagnostics": {
            "mean_predicted_variance": predicted_var,
            "mean_actual_squared_error": actual_var,
            "variance_ratio_actual_over_predicted": actual_var / predicted_var,
            "blur_multiplier_for_nominal_68pct": float(np.percentile(ratios, 68)),
            "per_dimension_variance_ratio": {
                dim: float(np.mean(error[:, d] ** 2) / np.mean(blur[:, d] ** 2))
                for d, dim in enumerate(DIMENSIONS)
            },
            "note": (
                "a ratio above 1 means the fitted model is more confident than it "
                "has earned; the multiplier is how much wider the interval would "
                "have to be to reach the nominal 68 percent"
            ),
        },
        "pooled_coverage": pooled,
        "per_dimension_coverage": per_dim,
        "n_cells": int(inside.size),
        "mean_blur": float(blur.mean()),
        "mean_blur_per_dimension": {
            dim: float(blur[:, d].mean()) for d, dim in enumerate(DIMENSIONS)
        },
        "mean_abs_error": float(np.mean(np.abs(theta_hat - theta_true))),
        "band": list(BAR_COVERAGE),
        "pass": bool(low <= pooled <= high),
    }


def reliability_ceiling(
    half_a: np.ndarray, half_b: np.ndarray, observed_r: dict[str, float]
) -> dict[str, Any]:
    """How much of the trait-recovery shortfall is noise and how much is bias.

    Two estimates of the same person from disjoint halves of the training items
    give the repeatability of a half-length estimate; Spearman-Brown steps that
    up to the full 202-item estimate. Dividing the observed correlation against
    the planted value by the square root of that gives the correlation the
    estimate would reach if it were perfectly repeatable -- the part of the gap
    that more items could never close.
    """
    out: dict[str, Any] = {"per_dimension": {}}
    for d, dim in enumerate(DIMENSIONS):
        half = pearson(half_a[:, d], half_b[:, d])
        full = 2 * half / (1 + half) if np.isfinite(half) else float("nan")
        observed = observed_r[dim]
        out["per_dimension"][dim] = {
            "half_sample_agreement": half,
            "reliability_full_length": full,
            "observed_r": observed,
            "disattenuated_r": observed / np.sqrt(full) if full > 0 else float("nan"),
        }
    rel = [v["reliability_full_length"] for v in out["per_dimension"].values()]
    dis = [v["disattenuated_r"] for v in out["per_dimension"].values()]
    out["reliability_summary"] = describe(rel)
    out["disattenuated_summary"] = describe(dis)
    out["reading"] = (
        "reliability well above the observed correlation means the estimate is "
        "repeatable but is tracking something that is not exactly the planted "
        "value -- the shortfall is in what the answers carry, not in how noisily "
        "they were read"
    )
    return out


def obedience_ceiling(
    answers_path: Path,
    bank: dict[str, dict[str, Any]],
    persona_ids: Sequence[str],
    item_ids: Sequence[str],
    theta_true: np.ndarray,
) -> dict[str, Any]:
    """What the answer matrix itself carries, before any model is fitted.

    The Gate-1 obedience measure, restricted to exactly the cells the Stage-2
    fit used: for each dimension, the correlation across personas between the
    planted value and the persona's mean standardized answer on the items
    designed to load on it, with negatively-keyed items flipped. This is the
    reference the fitted recovery should be compared against -- no model can
    read out more trait than the answers put in.
    """
    wanted_p = {pid: i for i, pid in enumerate(persona_ids)}
    wanted_i = {iid: j for j, iid in enumerate(item_ids)}
    matrix = np.full((len(persona_ids), len(item_ids)), np.nan)
    with Path(answers_path).open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("round") != "main":
                continue
            row = wanted_p.get(record["pid"])
            col = wanted_i.get(record["item_id"])
            if row is None or col is None:
                continue
            answer = record["answer"]
            if isinstance(answer, str):
                value = 1.0 if answer.strip().lower() in ("yes", "1", "true") else 0.0
            else:
                value = float(answer)
            matrix[row, col] = value

    standardized = np.zeros_like(matrix)
    for col in range(matrix.shape[1]):
        column = matrix[:, col]
        spread = np.nanstd(column)
        standardized[:, col] = (
            (column - np.nanmean(column)) / spread if spread > 1e-12 else 0.0
        )

    per_dim: dict[str, float] = {}
    counts: dict[str, int] = {}
    for d, dim in enumerate(DIMENSIONS):
        cols, signs = [], []
        for iid, col in wanted_i.items():
            loading = bank[iid]["loadings"].get(dim)
            if loading:
                cols.append(col)
                signs.append(np.sign(loading))
        counts[dim] = len(cols)
        if not cols:
            per_dim[dim] = float("nan")
            continue
        score = np.nanmean(standardized[:, cols] * np.array(signs)[None, :], axis=1)
        per_dim[dim] = pearson(score, theta_true[:, d])
    return {
        "per_dimension": per_dim,
        "median": float(np.nanmedian(list(per_dim.values()))),
        "min": float(np.nanmin(list(per_dim.values()))),
        "n_items_per_dimension": counts,
        "scope": "held-out personas, training items only -- the exact cells the fit saw",
        "reading": (
            "this is what the answer matrix itself carries; a fitted recovery "
            "that lands at or above these numbers has extracted what is there"
        ),
    }


def weak_item_ordering(
    discrimination: dict[str, float], bank: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Bar (d): the designed-weak items must sit low on fitted discrimination.

    Strong primaries are the comparison group: designed magnitude 1.0 on a
    single dimension. Cross-loading items and distractors are kept out of it
    and reported separately.
    """
    groups: dict[str, list[str]] = {"weak": [], "strong_primary": [], "cross": [], "distractor": []}
    for iid in discrimination:
        item = bank[iid]
        loadings = item["loadings"]
        cls = item["strength_class"]
        if cls == "weak":
            groups["weak"].append(iid)
        elif cls == "none":
            groups["distractor"].append(iid)
        elif len(loadings) == 1 and abs(list(loadings.values())[0]) == 1.0:
            groups["strong_primary"].append(iid)
        else:
            groups["cross"].append(iid)

    weak_vals = [discrimination[i] for i in groups["weak"]]
    strong_vals = [discrimination[i] for i in groups["strong_primary"]]

    # rank overlap: how many weak items outrank the weakest strong primary, and
    # how many strong primaries fall below the strongest weak item
    overlap_weak_above = [
        i for i in groups["weak"] if discrimination[i] > (min(strong_vals) if strong_vals else 0)
    ]
    overlap_strong_below = [
        i
        for i in groups["strong_primary"]
        if discrimination[i] < (max(weak_vals) if weak_vals else 0)
    ]
    combined = sorted(
        groups["weak"] + groups["strong_primary"], key=lambda i: discrimination[i]
    )
    weak_ranks = [combined.index(i) + 1 for i in groups["weak"]]

    # threshold-free version of the same question: pick one weak item and one
    # strong primary at random, how often is the weak one lower?
    if weak_vals and strong_vals:
        wins = sum(
            1.0 if w < s else 0.5 if w == s else 0.0 for w in weak_vals for s in strong_vals
        )
        auc = wins / (len(weak_vals) * len(strong_vals))
    else:
        auc = float("nan")

    return {
        "prob_weak_below_strong": float(auc),
        "weak_items": {i: discrimination[i] for i in sorted(groups["weak"])},
        "weak_summary": describe(weak_vals),
        "strong_primary_summary": describe(strong_vals),
        "cross_summary": describe([discrimination[i] for i in groups["cross"]]),
        "distractor_summary": describe([discrimination[i] for i in groups["distractor"]]),
        "n_weak": len(groups["weak"]),
        "n_strong_primary": len(groups["strong_primary"]),
        "weak_ranks_within_weak_plus_strong": sorted(weak_ranks),
        "n_ranked_below_all_strong": int(
            sum(1 for i in groups["weak"] if discrimination[i] < min(strong_vals))
        )
        if strong_vals
        else 0,
        "weak_items_above_weakest_strong": sorted(overlap_weak_above),
        "strong_primaries_below_strongest_weak": sorted(overlap_strong_below),
        "separation": (
            float(min(strong_vals) - max(weak_vals)) if strong_vals and weak_vals else float("nan")
        ),
        "pass": bool(
            strong_vals
            and weak_vals
            and float(np.median(weak_vals)) < float(np.percentile(strong_vals, 5))
        ),
        "pass_rule": "median weak discrimination below the 5th percentile of strong primaries",
        "groups": {k: sorted(v) for k, v in groups.items()},
    }


# --------------------------------------------------------------------------
# watch list
# --------------------------------------------------------------------------


def watch_list(
    discrimination: dict[str, float],
    fitted_loadings: dict[str, np.ndarray],
    bank: dict[str, dict[str, Any]],
    theta_train: np.ndarray,
    theta_holdout: np.ndarray,
    ordering: dict[str, Any],
) -> dict[str, Any]:
    strong_vals = [discrimination[i] for i in ordering["groups"]["strong_primary"]]
    weak_vals = [discrimination[i] for i in ordering["groups"]["weak"]]
    weak_median = float(np.median(weak_vals)) if weak_vals else float("nan")
    strong_median = float(np.median(strong_vals)) if strong_vals else float("nan")

    # -- distractors -------------------------------------------------------
    distractors = sorted(ordering["groups"]["distractor"])
    dvals = {i: discrimination[i] for i in distractors}
    flagged = [i for i, v in dvals.items() if v >= weak_median]
    distractor_block = {
        "per_item_discrimination": dvals,
        "summary": describe(list(dvals.values())),
        "weak_median_for_reference": weak_median,
        "strong_primary_median_for_reference": strong_median,
        "flagged_at_or_above_weak_median": flagged,
        "n_flagged": len(flagged),
        "verdict": (
            "clean -- every distractor sits below the median designed-weak item"
            if not flagged
            else "LEAKAGE HUNT: distractors carrying real discrimination, listed above"
        ),
    }

    # -- signs on the two named weak items ---------------------------------
    signs = {}
    for iid in WATCH_SIGN_ITEMS:
        if iid not in fitted_loadings:
            signs[iid] = {"note": "not in the fitted set"}
            continue
        designed = bank[iid]["loadings"]
        dim = max(designed, key=lambda d: abs(designed[d])) if designed else None
        if dim is None:
            signs[iid] = {"note": "no designed loading"}
            continue
        d = DIMENSIONS.index(dim)
        fitted_value = float(fitted_loadings[iid][d])
        designed_value = float(designed[dim])
        signs[iid] = {
            "designed_dimension": dim,
            "designed_loading": designed_value,
            "fitted_loading_on_that_dimension": fitted_value,
            "fitted_norm": float(np.linalg.norm(fitted_loadings[iid])),
            "sign_matches": bool(np.sign(fitted_value) == np.sign(designed_value)),
            "flag": (
                "ok"
                if np.sign(fitted_value) == np.sign(designed_value)
                else "SIGN FLIPPED against the designed direction"
            ),
        }

    # -- the planted zero --------------------------------------------------
    a, b = (DIMENSIONS.index(WATCH_ZERO_PAIR[0]), DIMENSIONS.index(WATCH_ZERO_PAIR[1]))
    zero_pair = {
        "pair": list(WATCH_ZERO_PAIR),
        "planted": float(PLANTED_SIGMA[a, b]),
        "recovered_train": pearson(theta_train[:, a], theta_train[:, b]),
        "recovered_holdout": pearson(theta_holdout[:, a], theta_holdout[:, b]),
    }
    zero_pair["abs_delta_train"] = abs(zero_pair["recovered_train"] - zero_pair["planted"])
    zero_pair["abs_delta_holdout"] = abs(zero_pair["recovered_holdout"] - zero_pair["planted"])

    # -- the two nearly flat raw items -------------------------------------
    flat = {}
    for iid in WATCH_FLAT_ITEMS:
        if iid not in discrimination:
            flat[iid] = {"note": "not in the fitted set"}
            continue
        flat[iid] = {
            "type": bank[iid]["type"],
            "strength_class": bank[iid]["strength_class"],
            "designed_loadings": bank[iid]["loadings"],
            "fitted_discrimination": discrimination[iid],
            "fitted_loading_vector": {
                dim: float(fitted_loadings[iid][d]) for d, dim in enumerate(DIMENSIONS)
            },
            "rank_among_all_fitted_items": int(
                sorted(discrimination.values()).index(discrimination[iid]) + 1
            ),
            "n_fitted_items": len(discrimination),
        }

    return {
        "distractors": distractor_block,
        "sign_check": signs,
        "planted_zero_pair": zero_pair,
        "near_flat_raw_items": flat,
    }


# --------------------------------------------------------------------------
# correlation matrices and strata
# --------------------------------------------------------------------------


def correlation_block(theta: np.ndarray) -> dict[str, Any]:
    empirical = np.corrcoef(theta, rowvar=False)
    delta = empirical - PLANTED_SIGMA
    off = ~np.eye(len(DIMENSIONS), dtype=bool)
    worst = np.unravel_index(np.argmax(np.abs(delta * off)), delta.shape)
    return {
        "recovered": empirical.tolist(),
        "delta": delta.tolist(),
        "max_abs_delta_offdiagonal": float(np.max(np.abs(delta[off]))),
        "max_abs_delta_cell": [DIMENSIONS[worst[0]], DIMENSIONS[worst[1]]],
        "mean_abs_delta_offdiagonal": float(np.mean(np.abs(delta[off]))),
        "frobenius_delta": float(np.linalg.norm(delta)),
    }


def stratum_summary(
    per_item: dict[str, dict[str, float]], strata: dict[str, str]
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for label in ("near", "same-domain", "far"):
        ids = [i for i in per_item if strata.get(i) == label]
        out[label] = {
            "n": len(ids),
            "r": describe([per_item[i]["r"] for i in ids]),
            "cosine": describe([per_item[i]["cosine"] for i in ids]),
            "fitted_norm": describe([per_item[i]["fitted_norm"] for i in ids]),
        }
    return out


# --------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------


def draw_heatmap(block_train: dict[str, Any], block_holdout: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("planted", PLANTED_SIGMA, "coolwarm", (-1, 1)),
        ("recovered (train)", np.array(block_train["recovered"]), "coolwarm", (-1, 1)),
        ("recovered (held-out)", np.array(block_holdout["recovered"]), "coolwarm", (-1, 1)),
        ("difference, held-out minus planted", np.array(block_holdout["delta"]), "PuOr", (-0.3, 0.3)),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(19, 5.2))
    for ax, (title, matrix, cmap, limits) in zip(axes, panels):
        image = ax.imshow(matrix, cmap=cmap, vmin=limits[0], vmax=limits[1])
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(len(DIMENSIONS)))
        ax.set_yticks(range(len(DIMENSIONS)))
        ax.set_xticklabels(DIMENSIONS, rotation=90, fontsize=8)
        ax.set_yticklabels(DIMENSIONS, fontsize=8)
        for i in range(len(DIMENSIONS)):
            for j in range(len(DIMENSIONS)):
                ax.text(
                    j, i, f"{matrix[i, j]:.2f}",
                    ha="center", va="center", fontsize=6.5,
                    color="black" if abs(matrix[i, j]) < 0.6 else "white",
                )
        fig.colorbar(image, ax=ax, fraction=0.046)
    fig.suptitle("Stage 2: planted against recovered trait correlations", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def draw_scatter(
    theta_hat: np.ndarray, theta_true: np.ndarray, per_dim_r: dict[str, float], path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True)
    for d, (dim, ax) in enumerate(zip(DIMENSIONS, axes.ravel())):
        ax.scatter(theta_true[:, d], theta_hat[:, d], s=14, alpha=0.65, edgecolor="none")
        low = min(theta_true[:, d].min(), theta_hat[:, d].min()) - 0.2
        high = max(theta_true[:, d].max(), theta_hat[:, d].max()) + 0.2
        ax.plot([low, high], [low, high], lw=1, color="0.4", ls="--")
        ax.set_title(f"{dim}   r = {per_dim_r[dim]:.3f}", fontsize=11)
        ax.grid(alpha=0.25)
        if d >= 4:
            ax.set_xlabel("planted")
        if d % 4 == 0:
            ax.set_ylabel("recovered (aligned)")
    fig.suptitle(
        f"Stage 2: recovered against planted traits, {theta_hat.shape[0]} held-out personas",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------
# the grade
# --------------------------------------------------------------------------


def grade(
    fit_path: Path,
    truth_dir: Path,
    splits_path: Path,
    out_dir: Path,
    *,
    diagnostics_path: Path | None = None,
    answers_path: Path | None = None,
    make_plots: bool = True,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fit = np.load(fit_path, allow_pickle=False)
    splits = json.loads(Path(splits_path).read_text(encoding="utf-8"))
    bank = load_planted_bank(truth_dir)

    train_ids = [str(x) for x in fit["persona_train_ids"]]
    holdout_ids = [str(x) for x in fit["persona_holdout_ids"]]
    train_items = [str(x) for x in fit["item_train_ids"]]
    holdout_items = [str(x) for x in fit["item_holdout_ids"]]

    theta_true_train = load_planted_theta(truth_dir, train_ids)
    theta_true_holdout = load_planted_theta(truth_dir, holdout_ids)

    # ---- 1. alignment, fitted on the training personas only --------------
    rotation = procrustes_rotation(fit["theta_train"], theta_true_train)
    theta_train = fit["theta_train"] @ rotation
    theta_holdout = fit["theta_holdout"] @ rotation
    cov_holdout = rotate_covariances(fit["cov_holdout"], rotation)
    blur_holdout = np.sqrt(np.maximum(np.diagonal(cov_holdout, axis1=1, axis2=2), 0.0))
    loadings_train = fit["train_loadings"] @ rotation
    loadings_holdout = fit["holdout_loadings"] @ rotation

    fitted_loadings = {iid: loadings_train[r] for r, iid in enumerate(train_items)}
    fitted_loadings.update({iid: loadings_holdout[r] for r, iid in enumerate(holdout_items)})
    discrimination = {iid: float(np.linalg.norm(vec)) for iid, vec in fitted_loadings.items()}

    # ---- 2. the bars ------------------------------------------------------
    traits = trait_recovery(theta_holdout, theta_true_holdout)
    items = item_recovery(
        loadings_holdout, designed_matrix(bank, holdout_items), holdout_items
    )
    blur = blur_honesty(theta_holdout, blur_holdout, theta_true_holdout)
    ordering = weak_item_ordering(discrimination, bank)

    # same numbers on the training items, for context only
    train_item_recovery = item_recovery(
        loadings_train, designed_matrix(bank, train_items), train_items
    )
    traits_train = trait_recovery(theta_train, theta_true_train)

    # ---- 3. watch list ----------------------------------------------------
    watch = watch_list(
        discrimination, fitted_loadings, bank, theta_train, theta_holdout, ordering
    )

    # ---- 4. correlation matrices -----------------------------------------
    corr_train = correlation_block(theta_train)
    corr_holdout = correlation_block(theta_holdout)

    # ---- 5. strata --------------------------------------------------------
    strata = stratum_summary(items["per_item"], splits.get("strata", {}))

    # ---- extras: blur without the metric fix, and wobble ------------------
    unlinked = None
    if "theta_holdout_unlinked" in fit.files:
        rot_raw = procrustes_rotation(fit["theta_train"], theta_true_train)
        raw_theta = fit["theta_holdout_unlinked"] @ rot_raw
        raw_blur = fit["blur_holdout_unlinked"]
        inside = np.abs(raw_theta - theta_true_holdout) <= raw_blur
        unlinked = {
            "pooled_coverage": float(inside.mean()),
            "mean_blur": float(raw_blur.mean()),
            "note": (
                "what the coverage would have been on the unpinned joint-MAP scale, "
                "where the latent unit is set by the ridge rather than by the prior"
            ),
        }

    # ---- extras: how much of the shortfall is repeatability ---------------
    ceiling = None
    if "theta_holdout_half_a" in fit.files:
        ceiling = reliability_ceiling(
            fit["theta_holdout_half_a"] @ rotation,
            fit["theta_holdout_half_b"] @ rotation,
            traits["per_dimension_r"],
        )

    obedience = None
    if answers_path is not None and Path(answers_path).is_file():
        obedience = obedience_ceiling(
            answers_path, bank, holdout_ids, train_items, theta_true_holdout
        )
        obedience["fitted_recovery_for_comparison"] = traits["per_dimension_r"]
        paired = [
            (obedience["per_dimension"][d], traits["per_dimension_r"][d]) for d in DIMENSIONS
        ]
        obedience["r_between_the_two_profiles"] = pearson(
            [p[0] for p in paired], [p[1] for p in paired]
        )
        obedience["fit_minus_answers_per_dimension"] = {
            d: traits["per_dimension_r"][d] - obedience["per_dimension"][d] for d in DIMENSIONS
        }

    wobble = load_planted_wobble(truth_dir, holdout_ids)
    error = np.sqrt(np.mean((theta_holdout - theta_true_holdout) ** 2, axis=1))
    wobble_block = {
        "r_wobble_vs_person_rmse": pearson(wobble, error),
        "r_wobble_vs_mean_blur": pearson(wobble, blur_holdout.mean(axis=1)),
        "wobble_summary": describe(wobble.tolist()),
    }

    verdict = {
        "gate": "Gate 2 -- static recovery",
        "bars": {
            "trait_recovery": {
                "statement": "per-dimension r(theta_hat, theta_true) >= 0.8 on all 8 dimensions, held-out personas",
                "value": traits["min_r"],
                "value_label": f"worst dimension {traits['min_r_dimension']}",
                "bar": BAR_TRAIT_R,
                "pass": traits["pass"],
            },
            "item_recovery": {
                "statement": "median per-item r(fitted, designed loadings) >= 0.7 on held-out items",
                "value": items["median_r"],
                "bar": BAR_ITEM_R,
                "pass": items["pass"],
            },
            "blur_honesty": {
                "statement": "pooled +/-1 blur coverage of planted theta inside 60-75%",
                "value": blur["pooled_coverage"],
                "bar": list(BAR_COVERAGE),
                "pass": blur["pass"],
            },
            "weak_item_ordering": {
                "statement": "designed-weak items come out with low fitted discrimination",
                "value": ordering["prob_weak_below_strong"],
                "value_label": (
                    "chance that a random designed-weak item has lower fitted "
                    "discrimination than a random strong primary"
                ),
                "bar": ordering["pass_rule"],
                "pass": ordering["pass"],
                "caveat": (
                    "PREREGISTRATION section 6 states this check qualitatively "
                    "('designed weak items must show low fitted discrimination'); "
                    "the rule above is this grader's reading of it, not a frozen "
                    "number. The two distributions are reported in full so the "
                    "call can be re-made."
                ),
            },
        },
    }
    verdict["all_bars_pass"] = all(b["pass"] for b in verdict["bars"].values())
    verdict["failed_bars"] = [k for k, b in verdict["bars"].items() if not b["pass"]]

    report: dict[str, Any] = {
        "module": "src.eval.recovery",
        "verdict": verdict,
        "alignment": {
            "method": "orthogonal Procrustes, fitted on the 400 training personas, applied unchanged to held-out",
            "rotation_determinant": float(np.linalg.det(rotation)),
            "rotation": rotation.tolist(),
        },
        "trait_recovery_holdout": traits,
        "trait_recovery_train": traits_train,
        "item_recovery_holdout": items,
        "item_recovery_train_for_context": {
            k: v for k, v in train_item_recovery.items() if k != "per_item"
        },
        "blur_honesty": blur,
        "blur_honesty_unpinned_metric": unlinked,
        "reliability_ceiling": ceiling,
        "answer_matrix_ceiling": obedience,
        "weak_item_ordering": ordering,
        "watch_list": watch,
        "correlations": {
            "planted": PLANTED_SIGMA.tolist(),
            "dimensions": list(DIMENSIONS),
            "train": corr_train,
            "holdout": corr_holdout,
        },
        "item_recovery_by_stratum": strata,
        "wobble_relationship": wobble_block,
        "fitted_discrimination_all_items": discrimination,
        "counts": {
            "train_personas": len(train_ids),
            "holdout_personas": len(holdout_ids),
            "train_items": len(train_items),
            "holdout_items": len(holdout_items),
        },
    }

    if diagnostics_path is not None and Path(diagnostics_path).is_file():
        report["fit_diagnostics"] = json.loads(
            Path(diagnostics_path).read_text(encoding="utf-8")
        )

    json_path = out_dir / "stage2_recovery.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if make_plots:
        draw_heatmap(corr_train, corr_holdout, out_dir / "stage2_heatmap.png")
        draw_scatter(
            theta_holdout,
            theta_true_holdout,
            traits["per_dimension_r"],
            out_dir / "stage2_scatter.png",
        )

    return report


def render(report: dict[str, Any]) -> str:
    lines = ["", "=" * 74, "GATE 2 VERDICT", "=" * 74]
    for name, bar in report["verdict"]["bars"].items():
        mark = "PASS" if bar["pass"] else "FAIL"
        value = bar["value"]
        lines.append(f"  [{mark}] {name:22s} value {value:.4f}   bar {bar['bar']}")
        lines.append(f"         {bar['statement']}")
    lines.append("")
    lines.append(
        "  ALL BARS PASS" if report["verdict"]["all_bars_pass"]
        else "  FAILED: " + ", ".join(report["verdict"]["failed_bars"])
    )
    lines.append("")
    lines.append("per-dimension trait recovery (held-out personas):")
    for dim, value in report["trait_recovery_holdout"]["per_dimension_r"].items():
        lines.append(f"    {dim}  r = {value:.4f}")
    if report.get("answer_matrix_ceiling"):
        ceiling = report["answer_matrix_ceiling"]
        lines.append("")
        lines.append("what the answer matrix itself carries (same cells, no model):")
        for dim, value in ceiling["per_dimension"].items():
            gap = ceiling["fit_minus_answers_per_dimension"][dim]
            lines.append(f"    {dim}  {value:.4f}   fit minus answers {gap:+.4f}")
        lines.append(
            f"    profiles agree across dimensions at r = {ceiling['r_between_the_two_profiles']:.3f}"
        )
    if report.get("reliability_ceiling"):
        rel = report["reliability_ceiling"]["reliability_summary"]
        dis = report["reliability_ceiling"]["disattenuated_summary"]
        lines.append("")
        lines.append(
            f"repeatability of theta-hat: median {rel['median']:.3f} "
            f"(range {rel['min']:.3f}-{rel['max']:.3f}); "
            f"disattenuated validity median {dis['median']:.3f}"
        )
    lines.append("")
    blur = report["blur_honesty"]
    lines.append(f"blur coverage pooled {blur['pooled_coverage']:.4f} (band 0.60-0.75)")
    for dim, value in blur["per_dimension_coverage"].items():
        lines.append(f"    {dim}  {value:.3f}")
    diag = blur["diagnostics"]
    lines.append(
        f"    actual error variance is {diag['variance_ratio_actual_over_predicted']:.2f}x "
        f"the predicted; blur would need x{diag['blur_multiplier_for_nominal_68pct']:.2f} to reach 68%"
    )
    lines.append("")
    order = report["weak_item_ordering"]
    lines.append(
        f"discrimination: weak median {order['weak_summary']['median']:.3f} "
        f"(max {order['weak_summary']['max']:.3f}) vs strong primary median "
        f"{order['strong_primary_summary']['median']:.3f} "
        f"(min {order['strong_primary_summary']['min']:.3f})"
    )
    lines.append(
        f"    a random weak item is below a random strong primary "
        f"{order['prob_weak_below_strong']:.1%} of the time; "
        f"{len(order['weak_items_above_weakest_strong'])} weak items overlap the strong range"
    )
    dis = report["watch_list"]["distractors"]
    lines.append(
        f"distractors: median {dis['summary']['median']:.3f}, max {dis['summary']['max']:.3f} "
        f"-- {dis['verdict']}"
    )
    zero = report["watch_list"]["planted_zero_pair"]
    lines.append(
        f"{'-'.join(zero['pair'])} planted {zero['planted']:.2f}, recovered "
        f"train {zero['recovered_train']:+.3f} / held-out {zero['recovered_holdout']:+.3f}"
    )
    lines.append(
        "correlation matrix: max |delta| off-diagonal "
        f"{report['correlations']['holdout']['max_abs_delta_offdiagonal']:.3f} at "
        f"{'-'.join(report['correlations']['holdout']['max_abs_delta_cell'])} (held-out)"
    )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.eval.recovery",
        description="Grade the Stage-2 fit against the planted design (Gate 2).",
    )
    parser.add_argument("--fit", type=Path, required=True, help="stage2_fit.npz from the model")
    parser.add_argument("--diagnostics", type=Path, default=None, help="fit diagnostics json")
    parser.add_argument("--truth", type=Path, required=True, help="the batch's planted directory")
    parser.add_argument("--splits", type=Path, required=True, help="frozen splits manifest")
    parser.add_argument("--out", type=Path, required=True, help="results directory")
    parser.add_argument(
        "--answers", type=Path, default=None,
        help="noised answers, for the answer-matrix ceiling comparison",
    )
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    report = grade(
        args.fit,
        args.truth,
        args.splits,
        args.out,
        diagnostics_path=args.diagnostics,
        answers_path=args.answers,
        make_plots=not args.no_plots,
    )
    print(render(report))
    print(f"report -> {args.out / 'stage2_recovery.json'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
