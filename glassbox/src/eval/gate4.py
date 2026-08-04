"""Gate 4 grader: end-to-end prediction and calibration (PREREGISTRATION.md 5-6).

    python -m src.eval.gate4 \\
        --truth50 <truth runs dir>/truth50 \\
        --predictions results/stage4_predictions.npz \\
        --baselines results/stage4_baselines.npz \\
        --splits experiments/splits_v1.json \\
        --out results

Grader-side code. It reads the 50-sample true answer distributions that
``src.eval.truth_distributions`` drew, and grades every arm against them.

The frozen bars (PREREGISTRATION.md section 6, Gate 4)
------------------------------------------------------
a. **Primary lift** -- Brier beats the population marginal by >= 25% relative,
   ``(B_marginal - B_model) / B_marginal``, AND beats k-NN.
b. **ECE <= 0.05** -- 10 equal-width bins over every (cell, category) predicted
   probability against the matching empirical 50-sample frequency,
   count-weighted.
c. **Correlation >= 0.9** between predicted probability and empirical 50-sample
   frequency, pooled over every (cell, category) pair.

Reported with no bar: everything per probe stratum (near / same-domain / far)
for the model and all baselines; the predicted-vs-true population marginals with
bootstrap error bars; the 2-way cross-tabs.

How the two scores are defined
------------------------------
Brier for one cell is the mean over **that item's own categories** of
``(p_pred - p_true)^2`` -- five for a Likert item, two for a binary one. Using
the item's own width rather than a fixed five keeps a binary item from looking
artificially accurate just because three of its columns are zero on both sides.
Log loss is ``-sum_c p_true,c log(max(p_pred,c, eps))`` over the same columns.

Conditional independence, stated out loud
-----------------------------------------
The 2-way cross-tab needs a joint distribution over two items for one person.
Neither side has one: the predictor emits per-cell marginals, and the truth
store's 50 sittings are drawn per cell with independent seeds. Both sides are
therefore multiplied out as ``p(a, b) = p(a) p(b)``. The assumption is the same
on both sides, so the comparison is fair -- but it is an assumption, and the
report says so where the numbers appear.

WALL. Nothing computed here goes back to the system side. The predictor and the
baselines were built before this ran and are read, not re-fitted.
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

from .truth_distributions import MAX_CATEGORIES, Truth50

# ---- the frozen bars (PREREGISTRATION.md section 6, Gate 4) ---------------
BAR_LIFT = 0.25  # relative Brier improvement over the population marginal
BAR_ECE = 0.05  # 10 equal-width bins, count-weighted
BAR_CORRELATION = 0.90  # predicted probability vs empirical 50-sample frequency

#: The leakage trigger. Above either of these the orchestrator investigates;
#: this grader raises the flag and never clears it.
LEAKAGE_CORRELATION = 0.95
LEAKAGE_LIFT = 0.45

#: Bins for the calibration curve (PREREGISTRATION.md section 5: 10 equal-width).
N_BINS = 10

#: Floor inside the logarithm, so a zero probability costs a large number
#: rather than infinity. Recorded because it changes the log-loss scale.
LOG_EPS = 1e-6

#: Bootstrap resamples for the aggregate marginal error bars.
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 2026

DEFAULT_OUT_PREFIX = "stage4"

#: The confirmatory arm, and the two references its bar is written against.
CONFIRMATORY_ARM = "model"
MARGINAL_ARM = "marginal"
KNN_ARM = "knn"


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def category_mask(n_categories: np.ndarray) -> np.ndarray:
    """(n_items, 5) boolean: which category columns this item actually has."""
    mask = np.zeros((len(n_categories), MAX_CATEGORIES), dtype=bool)
    for col, k in enumerate(n_categories):
        mask[col, : int(k)] = True
    return mask


def brier_cells(
    predicted: np.ndarray, truth: np.ndarray, n_categories: np.ndarray
) -> np.ndarray:
    """(n_people, n_items) multiclass Brier, mean over the item's own categories."""
    mask = category_mask(n_categories)[None, :, :]
    squared = np.where(mask, (predicted - truth) ** 2, 0.0)
    return squared.sum(axis=2) / np.asarray(n_categories, dtype=float)[None, :]


def log_loss_cells(
    predicted: np.ndarray, truth: np.ndarray, n_categories: np.ndarray
) -> np.ndarray:
    """(n_people, n_items) cross entropy of the true distribution under the predicted one."""
    mask = category_mask(n_categories)[None, :, :]
    safe = np.maximum(predicted, LOG_EPS)
    return -np.where(mask, truth * np.log(safe), 0.0).sum(axis=2)


def pair_arrays(
    predicted: np.ndarray, truth: np.ndarray, n_categories: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten to (cell, category) pairs, keeping only categories the item has."""
    mask = np.broadcast_to(category_mask(n_categories)[None, :, :], predicted.shape)
    return predicted[mask], truth[mask]


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def expected_calibration_error(
    predicted: np.ndarray, observed: np.ndarray, *, n_bins: int = N_BINS
) -> dict[str, Any]:
    """Count-weighted ECE over ``n_bins`` equal-width bins of the predicted probability.

    Both inputs are flat arrays of matched (predicted probability, empirical
    50-sample frequency) pairs. A bin's error is the absolute gap between the
    two means inside it; the ECE is those errors weighted by how many pairs each
    bin holds.
    """
    predicted = np.asarray(predicted, dtype=float)
    observed = np.asarray(observed, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # right-closed on the last bin so a probability of exactly 1 has a home
    index = np.clip(np.digitize(predicted, edges[1:-1], right=False), 0, n_bins - 1)

    bins = []
    total = 0.0
    for b in range(n_bins):
        inside = index == b
        count = int(inside.sum())
        if count == 0:
            bins.append(
                {
                    "bin": [float(edges[b]), float(edges[b + 1])],
                    "n": 0,
                    "mean_predicted": None,
                    "mean_observed": None,
                    "gap": None,
                }
            )
            continue
        mean_p = float(predicted[inside].mean())
        mean_o = float(observed[inside].mean())
        total += count * abs(mean_p - mean_o)
        bins.append(
            {
                "bin": [float(edges[b]), float(edges[b + 1])],
                "n": count,
                "mean_predicted": mean_p,
                "mean_observed": mean_o,
                "gap": float(mean_p - mean_o),
            }
        )
    return {
        "ece": float(total / max(len(predicted), 1)),
        "n_pairs": int(len(predicted)),
        "n_bins": n_bins,
        "bins": bins,
    }


# --------------------------------------------------------------------------
# one arm, graded
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Arm:
    name: str
    label: str
    probs: np.ndarray


def grade_arm(
    arm: Arm,
    truth: np.ndarray,
    n_categories: np.ndarray,
    kinds: Sequence[str],
    strata: Sequence[str],
) -> dict[str, Any]:
    """Every metric for one arm: pooled, per stratum, per item type."""
    brier = brier_cells(arm.probs, truth, n_categories)
    logloss = log_loss_cells(arm.probs, truth, n_categories)
    flat_pred, flat_true = pair_arrays(arm.probs, truth, n_categories)
    calibration = expected_calibration_error(flat_pred, flat_true)

    is_binary = np.array([k != "likert5" for k in kinds])
    per_type: dict[str, Any] = {}
    for name, mask in (("binary", is_binary), ("likert5", ~is_binary)):
        if not mask.any():
            continue
        p, t = pair_arrays(
            arm.probs[:, mask, :], truth[:, mask, :], np.asarray(n_categories)[mask]
        )
        per_type[name] = {
            "n_items": int(mask.sum()),
            "brier": float(brier[:, mask].mean()),
            "log_loss": float(logloss[:, mask].mean()),
            "correlation": pearson(p, t),
            "ece": expected_calibration_error(p, t)["ece"],
        }

    per_stratum: dict[str, Any] = {}
    for stratum in sorted(set(strata)):
        mask = np.array([s == stratum for s in strata])
        p, t = pair_arrays(
            arm.probs[:, mask, :], truth[:, mask, :], np.asarray(n_categories)[mask]
        )
        per_stratum[stratum] = {
            "n_items": int(mask.sum()),
            "brier": float(brier[:, mask].mean()),
            "log_loss": float(logloss[:, mask].mean()),
            "correlation": pearson(p, t),
            "ece": expected_calibration_error(p, t)["ece"],
        }

    return {
        "label": arm.label,
        "brier": float(brier.mean()),
        "brier_sd_over_cells": float(brier.std()),
        "log_loss": float(logloss.mean()),
        "correlation": pearson(flat_pred, flat_true),
        "ece": calibration["ece"],
        "calibration": calibration,
        "per_stratum": per_stratum,
        "per_item_type": per_type,
        "cells": {"brier": brier, "log_loss": logloss},
    }


# --------------------------------------------------------------------------
# aggregate checks
# --------------------------------------------------------------------------


def aggregate_marginals(
    predicted: np.ndarray,
    truth: np.ndarray,
    n_categories: np.ndarray,
    item_ids: Sequence[str],
    *,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Predicted vs true population marginal per probe item, with bootstrap bars.

    The resampling is over the 100 held-out personas -- the unit that actually
    varies -- and both sides are resampled together, so the error bar answers
    "how much would this gap move with a different 100 people".
    """
    n_people = predicted.shape[0]
    rng = np.random.default_rng(seed)
    mask = category_mask(n_categories)

    predicted_mean = predicted.mean(axis=0)
    truth_mean = truth.mean(axis=0)

    boot_pred = np.zeros((n_bootstrap,) + predicted_mean.shape)
    boot_true = np.zeros_like(boot_pred)
    for b in range(n_bootstrap):
        draw = rng.integers(0, n_people, size=n_people)
        boot_pred[b] = predicted[draw].mean(axis=0)
        boot_true[b] = truth[draw].mean(axis=0)

    def interval(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.percentile(values, 2.5, axis=0), np.percentile(values, 97.5, axis=0)

    pred_lo, pred_hi = interval(boot_pred)
    true_lo, true_hi = interval(boot_true)
    inside = (predicted_mean >= true_lo) & (predicted_mean <= true_hi)

    per_item = []
    for col, item_id in enumerate(item_ids):
        keep = mask[col]
        per_item.append(
            {
                "item_id": item_id,
                "predicted": [float(v) for v in predicted_mean[col][keep]],
                "true": [float(v) for v in truth_mean[col][keep]],
                "true_ci_low": [float(v) for v in true_lo[col][keep]],
                "true_ci_high": [float(v) for v in true_hi[col][keep]],
                "predicted_ci_low": [float(v) for v in pred_lo[col][keep]],
                "predicted_ci_high": [float(v) for v in pred_hi[col][keep]],
                "max_abs_gap": float(np.max(np.abs(predicted_mean[col][keep] - truth_mean[col][keep]))),
                "total_variation": float(
                    0.5 * np.abs(predicted_mean[col][keep] - truth_mean[col][keep]).sum()
                ),
            }
        )

    flat_pred = predicted_mean[mask]
    flat_true = truth_mean[mask]
    return {
        "what": "population marginal per probe item: predicted vs the 50-sample truth",
        "n_bootstrap": n_bootstrap,
        "bootstrap_seed": seed,
        "resampling_unit": "the 100 held-out personas, both sides resampled together",
        "correlation": pearson(flat_pred, flat_true),
        "mean_abs_gap": float(np.mean(np.abs(flat_pred - flat_true))),
        "rmse": float(np.sqrt(np.mean((flat_pred - flat_true) ** 2))),
        "mean_total_variation_per_item": float(np.mean([r["total_variation"] for r in per_item])),
        "fraction_inside_the_true_95_interval": float(inside[mask].mean()),
        "per_item": per_item,
        "arrays": {
            "predicted_mean": predicted_mean,
            "truth_mean": truth_mean,
            "true_ci_low": true_lo,
            "true_ci_high": true_hi,
        },
    }


def cross_tabs(
    predicted: np.ndarray,
    truth: np.ndarray,
    n_categories: np.ndarray,
    *,
    max_pairs: int | None = None,
) -> dict[str, Any]:
    """Population 2-way tables for every probe item pair, both sides.

    Each persona's joint over items (i, j) is the outer product of their two
    per-cell distributions -- the conditional-independence assumption, applied
    identically to the predicted and the true side. The population table is the
    mean of those joints over the 100 held-out personas.
    """
    n_items = predicted.shape[1]
    mask = category_mask(n_categories)
    pairs = [(i, j) for i in range(n_items) for j in range(i + 1, n_items)]
    if max_pairs is not None:
        pairs = pairs[:max_pairs]

    pred_cells: list[float] = []
    true_cells: list[float] = []
    per_pair_tv: list[float] = []
    for i, j in pairs:
        joint_pred = np.einsum("na,nb->ab", predicted[:, i, :], predicted[:, j, :]) / predicted.shape[0]
        joint_true = np.einsum("na,nb->ab", truth[:, i, :], truth[:, j, :]) / truth.shape[0]
        keep = np.outer(mask[i], mask[j])
        pred_cells.extend(joint_pred[keep].tolist())
        true_cells.extend(joint_true[keep].tolist())
        per_pair_tv.append(float(0.5 * np.abs(joint_pred[keep] - joint_true[keep]).sum()))

    pred_array = np.asarray(pred_cells)
    true_array = np.asarray(true_cells)
    return {
        "what": "population 2-way table per probe item pair, predicted vs the 50-sample truth",
        "conditional_independence": (
            "each persona's joint over two items is the product of their two "
            "per-cell distributions, on BOTH sides. Neither the predictor nor "
            "the 50 seeded sittings carry a within-person joint, so the "
            "assumption is shared and the comparison is fair -- but it is an "
            "assumption, and a real answer dependence beyond theta would not "
            "show up in either side of this plot"
        ),
        "n_pairs": len(pairs),
        "n_table_cells": int(pred_array.size),
        "correlation": pearson(pred_array, true_array),
        "rmse": float(np.sqrt(np.mean((pred_array - true_array) ** 2))),
        "mean_total_variation_per_pair": float(np.mean(per_pair_tv)),
        "worst_total_variation_per_pair": float(np.max(per_pair_tv)),
        "arrays": {"predicted": pred_array, "truth": true_array},
    }


# --------------------------------------------------------------------------
# the optional cluster arm
# --------------------------------------------------------------------------


def load_qwen_arm(
    path: Path,
    persona_ids: Sequence[str],
    item_ids: Sequence[str],
    n_categories: np.ndarray,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """The no-interview LLM baseline, if the cluster has produced it yet.

    Schema, one JSON object per line:
    ``{"persona_id": ..., "item_id": ..., "probs": [...]}`` with the
    probabilities in the item's category order.
    """
    path = Path(path)
    if not path.is_file():
        return None, {
            "status": "pending",
            "expected_at": str(path),
            "note": (
                "the Qwen no-interview baseline runs on Leonardo; the grader "
                "accepts it as soon as the parsed file lands and grades without "
                "it until then"
            ),
        }

    prow = {pid: i for i, pid in enumerate(persona_ids)}
    icol = {iid: j for j, iid in enumerate(item_ids)}
    probs = np.full((len(persona_ids), len(item_ids), MAX_CATEGORIES), np.nan)
    seen = 0
    malformed = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            row = prow.get(str(record.get("persona_id")))
            col = icol.get(str(record.get("item_id")))
            if row is None or col is None:
                continue
            values = np.asarray(record.get("probs") or [], dtype=float)
            if values.size != int(n_categories[col]) or not np.isfinite(values).all():
                malformed += 1
                continue
            total = values.sum()
            if total <= 0:
                malformed += 1
                continue
            probs[row, col, : values.size] = values / total
            probs[row, col, values.size :] = 0.0
            seen += 1

    expected = len(persona_ids) * len(item_ids)
    missing = np.isnan(probs).any(axis=2)
    if missing.any():
        marginal_fill = np.where(
            category_mask(n_categories), 1.0 / np.asarray(n_categories, float)[:, None], 0.0
        )
        rows, cols = np.nonzero(missing)
        probs[rows, cols] = marginal_fill[cols]
    return probs, {
        "status": "graded",
        "path": str(path),
        "cells_parsed": seen,
        "cells_expected": expected,
        "cells_malformed": malformed,
        "cells_filled_with_a_uniform_row": int(missing.sum()),
    }


# --------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------


def plot_reliability(path: Path, graded: dict[str, Any], arms: Sequence[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"model": "#4C72B0", "marginal": "#8C8C8C", "knn": "#C44E52",
              "profile": "#DD8452", "qwen_no_interview": "#55A868"}

    ax = axes[0]
    ax.plot([0, 1], [0, 1], color="#333333", lw=1, ls=":", label="perfect")
    for name in arms:
        bins = [b for b in graded[name]["calibration"]["bins"] if b["n"]]
        ax.plot(
            [b["mean_predicted"] for b in bins],
            [b["mean_observed"] for b in bins],
            marker="o",
            ms=4,
            lw=1.6,
            color=colors.get(name, None),
            label=f"{name} (ECE {graded[name]['ece']:.4f})",
        )
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("empirical 50-sample frequency")
    ax.set_title("Reliability, 10 equal-width bins")
    ax.legend(fontsize=8)

    ax = axes[1]
    bins = graded[CONFIRMATORY_ARM]["calibration"]["bins"]
    centers = [0.5 * (b["bin"][0] + b["bin"][1]) for b in bins]
    ax.bar(centers, [b["n"] for b in bins], width=0.09, color="#4C72B0")
    ax.set_yscale("log")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("(cell, category) pairs")
    ax.set_title(f"Where the {CONFIRMATORY_ARM} arm's mass sits")

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_stratum_lift(path: Path, graded: dict[str, Any], arms: Sequence[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    strata = ["near", "same-domain", "far"]
    strata = [s for s in strata if s in graded[CONFIRMATORY_ARM]["per_stratum"]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    width = 0.8 / len(arms)
    for k, name in enumerate(arms):
        values = [graded[name]["per_stratum"][s]["brier"] for s in strata]
        ax.bar(np.arange(len(strata)) + k * width, values, width=width, label=name)
    ax.set_xticks(np.arange(len(strata)) + 0.4 - width / 2)
    ax.set_xticklabels(
        [f"{s}\n(n={graded[CONFIRMATORY_ARM]['per_stratum'][s]['n_items']})" for s in strata]
    )
    ax.set_ylabel("Brier (lower is better)")
    ax.set_title("Brier per probe stratum")
    ax.legend(fontsize=8)

    ax = axes[1]
    marginal = [graded[MARGINAL_ARM]["per_stratum"][s]["brier"] for s in strata]
    for name in arms:
        if name == MARGINAL_ARM:
            continue
        values = [graded[name]["per_stratum"][s]["brier"] for s in strata]
        lift = [100.0 * (m - v) / m for m, v in zip(marginal, values)]
        ax.plot(range(len(strata)), lift, marker="o", lw=1.8, label=name)
    ax.axhline(100 * BAR_LIFT, color="#C44E52", lw=2, ls="--", label=f"bar {100*BAR_LIFT:.0f}%")
    ax.axhline(0, color="#8C8C8C", lw=1)
    ax.set_xticks(range(len(strata)))
    ax.set_xticklabels(strata)
    ax.set_ylabel("Brier lift over the marginal (%)")
    ax.set_title("Lift per probe stratum")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_aggregate(path: Path, marginals: dict[str, Any], tabs: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    predicted = np.concatenate([np.asarray(r["predicted"]) for r in marginals["per_item"]])
    true = np.concatenate([np.asarray(r["true"]) for r in marginals["per_item"]])
    lo = np.concatenate([np.asarray(r["true_ci_low"]) for r in marginals["per_item"]])
    hi = np.concatenate([np.asarray(r["true_ci_high"]) for r in marginals["per_item"]])
    ax.errorbar(
        true, predicted, xerr=[true - lo, hi - true],
        fmt="o", ms=4, lw=0.8, alpha=0.7, color="#4C72B0", ecolor="#B0B0B0",
    )
    ax.plot([0, 1], [0, 1], color="#333333", lw=1, ls=":")
    ax.set_xlabel("true population marginal (50-sample, 95% bootstrap)")
    ax.set_ylabel("predicted population marginal")
    ax.set_title(f"Aggregate marginals, r = {marginals['correlation']:.4f}")

    ax = axes[1]
    ax.scatter(
        tabs["arrays"]["truth"], tabs["arrays"]["predicted"],
        s=3, alpha=0.15, color="#4C72B0", edgecolors="none",
    )
    top = float(max(tabs["arrays"]["truth"].max(), tabs["arrays"]["predicted"].max()))
    ax.plot([0, top], [0, top], color="#333333", lw=1, ls=":")
    ax.set_xlabel("true 2-way cell probability")
    ax.set_ylabel("predicted 2-way cell probability")
    ax.set_title(
        f"2-way cross-tabs, {tabs['n_pairs']} item pairs, r = {tabs['correlation']:.4f}\n"
        "(both sides assume conditional independence)"
    )

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items() if k not in ("cells", "arrays")}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def run(
    truth50_dir: Path,
    predictions_path: Path,
    baselines_path: Path,
    splits_path: Path,
    out_dir: Path,
    *,
    qwen_path: Path | None = None,
    out_prefix: str = DEFAULT_OUT_PREFIX,
    verbose: bool = True,
) -> dict[str, Any]:
    started = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    truth = Truth50(truth50_dir)
    predictions = np.load(predictions_path, allow_pickle=False)
    baselines = np.load(baselines_path, allow_pickle=False)
    splits = json.loads(Path(splits_path).read_text(encoding="utf-8"))

    persona_ids = [str(p) for p in predictions["persona_ids"]]
    item_ids = [str(i) for i in predictions["item_ids"]]
    kinds = [str(k) for k in predictions["item_kinds"]]
    n_categories = np.asarray(predictions["n_categories"], dtype=int)
    strata = [str(splits["strata"][i]) for i in item_ids]

    # ---- alignment, checked rather than assumed ---------------------------
    alignment = {
        "persona_ids_match_truth": persona_ids == truth.persona_ids,
        "item_ids_match_truth": item_ids == truth.item_ids,
        "persona_ids_match_baselines": persona_ids == [str(p) for p in baselines["persona_ids"]],
        "item_ids_match_baselines": item_ids == [str(i) for i in baselines["item_ids"]],
        "n_categories_match_truth": bool(
            np.array_equal(n_categories, truth.n_categories)
        ),
        "strata_match_truth": strata == truth.strata,
        "checksums_verified": truth.verify_checksums(),
    }
    bad = [k for k, v in alignment.items() if v is False]
    if bad:
        raise ValueError(f"the arms and the truth store do not line up: {bad}")
    if not all(alignment["checksums_verified"].values()):
        raise ValueError("the 50-sample truth files do not match their recorded checksums")

    empirical = truth.empirical

    # ---- the arms ---------------------------------------------------------
    arms: list[Arm] = [
        Arm(CONFIRMATORY_ARM, "CONFIRMATORY -- fused theta_hat@15 + zero-shot loadings + trained thresholds, posterior-integrated",
            np.asarray(predictions["probs_model"], dtype=float)),
        Arm(MARGINAL_ARM, "baseline 1 -- population marginal (zero information)",
            np.asarray(baselines["probs_marginal"], dtype=float)),
        Arm("profile", "baseline 2 -- public profile only",
            np.asarray(baselines["probs_profile"], dtype=float)),
        Arm(KNN_ARM, f"baseline 3 -- k-NN on the 15 interview answers, k = {int(baselines['knn_k'])}",
            np.asarray(baselines["probs_knn"], dtype=float)),
    ]
    qwen_probs, qwen_status = load_qwen_arm(
        Path(qwen_path) if qwen_path else Path("data/runs/stage4_qwen_baseline/predictions.jsonl"),
        persona_ids,
        item_ids,
        n_categories,
    )
    if qwen_probs is not None:
        arms.append(Arm("qwen_no_interview", "baseline 4 -- LLM given the public profile, no interview", qwen_probs))

    exploratory_arms = [
        Arm("model_plugin", "EXPLORATORY -- the same predictor with the posterior collapsed to its mean",
            np.asarray(predictions["probs_model_plugin"], dtype=float)),
        Arm("model_text_thresholds", "EXPLORATORY -- item side fully zero-shot (thresholds from item text)",
            np.asarray(predictions["probs_model_text_thresholds"], dtype=float)),
        Arm("full_matrix_posterior", "EXPLORATORY -- Gate 2 full-matrix theta with fitted holdout item parameters",
            np.asarray(predictions["probs_full_matrix_posterior"], dtype=float)),
        Arm("dec_fused_encoder", "EXPLORATORY -- decomposition: fused theta x encoder items (plug-in)",
            np.asarray(predictions["probs_dec_fused_encoder"], dtype=float)),
        Arm("dec_fused_fitted", "EXPLORATORY -- decomposition: fused theta x fitted items (plug-in)",
            np.asarray(predictions["probs_dec_fused_fitted"], dtype=float)),
        Arm("dec_full_encoder", "EXPLORATORY -- decomposition: full-matrix theta x encoder items (plug-in)",
            np.asarray(predictions["probs_dec_full_encoder"], dtype=float)),
        Arm("dec_full_fitted", "EXPLORATORY -- decomposition: full-matrix theta x fitted items (plug-in)",
            np.asarray(predictions["probs_dec_full_fitted"], dtype=float)),
        Arm("oracle", "EXPLORATORY -- the analytic post-noise distribution itself, the ceiling 50 samples allow",
            truth.expected),
    ]

    graded: dict[str, Any] = {}
    for arm in list(arms) + exploratory_arms:
        graded[arm.name] = grade_arm(arm, empirical, n_categories, kinds, strata)
        if verbose:
            g = graded[arm.name]
            print(
                f"  {arm.name:24s} brier {g['brier']:.5f}  logloss {g['log_loss']:.5f}"
                f"  ece {g['ece']:.4f}  r {g['correlation']:.4f}"
            )

    # ---- the frozen bars --------------------------------------------------
    b_model = graded[CONFIRMATORY_ARM]["brier"]
    b_marginal = graded[MARGINAL_ARM]["brier"]
    b_knn = graded[KNN_ARM]["brier"]
    lift = (b_marginal - b_model) / b_marginal
    beats_knn = bool(b_model < b_knn)

    per_stratum_lift = {
        stratum: {
            "model_brier": graded[CONFIRMATORY_ARM]["per_stratum"][stratum]["brier"],
            "marginal_brier": graded[MARGINAL_ARM]["per_stratum"][stratum]["brier"],
            "knn_brier": graded[KNN_ARM]["per_stratum"][stratum]["brier"],
            "profile_brier": graded["profile"]["per_stratum"][stratum]["brier"],
            "lift_vs_marginal": (
                graded[MARGINAL_ARM]["per_stratum"][stratum]["brier"]
                - graded[CONFIRMATORY_ARM]["per_stratum"][stratum]["brier"]
            )
            / graded[MARGINAL_ARM]["per_stratum"][stratum]["brier"],
            "knn_lift_vs_marginal": (
                graded[MARGINAL_ARM]["per_stratum"][stratum]["brier"]
                - graded[KNN_ARM]["per_stratum"][stratum]["brier"]
            )
            / graded[MARGINAL_ARM]["per_stratum"][stratum]["brier"],
            "model_beats_knn": bool(
                graded[CONFIRMATORY_ARM]["per_stratum"][stratum]["brier"]
                < graded[KNN_ARM]["per_stratum"][stratum]["brier"]
            ),
            "n_items": graded[CONFIRMATORY_ARM]["per_stratum"][stratum]["n_items"],
        }
        for stratum in graded[CONFIRMATORY_ARM]["per_stratum"]
    }

    verdict = {
        "gate": "Gate 4 -- end-to-end prediction and calibration",
        "confirmatory_arm": CONFIRMATORY_ARM,
        "n_holdout_personas": len(persona_ids),
        "n_probe_items": len(item_ids),
        "n_cells": len(persona_ids) * len(item_ids),
        "truth_definition": (
            f"{truth.n_samples} independent seeded applications of the frozen "
            "noise layer per cell (PREREGISTRATION.md section 5, addendum)"
        ),
        "bars": {
            "primary_lift": {
                "statement": (
                    "Brier beats the population-marginal baseline by >= 25% "
                    "relative, and beats k-NN"
                ),
                "value": float(lift),
                "value_label": "relative Brier lift over the marginal",
                "bar": BAR_LIFT,
                "pass": bool(lift >= BAR_LIFT and beats_knn),
                "model_brier": b_model,
                "marginal_brier": b_marginal,
                "knn_brier": b_knn,
                "beats_knn": beats_knn,
                "knn_lift_vs_marginal": float((b_marginal - b_knn) / b_marginal),
            },
            "ece": {
                "statement": "ECE <= 0.05, 10 equal-width bins, count-weighted",
                "value": graded[CONFIRMATORY_ARM]["ece"],
                "bar": BAR_ECE,
                "pass": bool(graded[CONFIRMATORY_ARM]["ece"] <= BAR_ECE),
            },
            "correlation": {
                "statement": (
                    "correlation between predicted probability and empirical "
                    "50-sample frequency >= 0.9 across all persona x item cells"
                ),
                "value": graded[CONFIRMATORY_ARM]["correlation"],
                "bar": BAR_CORRELATION,
                "pass": bool(graded[CONFIRMATORY_ARM]["correlation"] >= BAR_CORRELATION),
                "binary_only": graded[CONFIRMATORY_ARM]["per_item_type"].get("binary", {}).get("correlation"),
                "likert_only": graded[CONFIRMATORY_ARM]["per_item_type"].get("likert5", {}).get("correlation"),
            },
        },
    }
    verdict["all_bars_pass"] = all(b["pass"] for b in verdict["bars"].values())
    verdict["failed_bars"] = [k for k, b in verdict["bars"].items() if not b["pass"]]

    # PREREGISTRATION.md section 6, Gate 4: "If k-NN is close, that is reported
    # prominently." So it sits in the verdict, not in a footnote.
    knn_gap = (b_knn - b_model) / b_knn
    verdict["knn_proximity"] = {
        "statement": "how much of the model's advantage is over k-NN rather than over nothing",
        "model_lift_over_marginal": float(lift),
        "knn_lift_over_marginal": float((b_marginal - b_knn) / b_marginal),
        "model_brier_improvement_over_knn_relative": float(knn_gap),
        "close": bool(knn_gap < 0.10),
        "per_stratum_model_beats_knn": {
            stratum: per_stratum_lift[stratum]["model_beats_knn"] for stratum in per_stratum_lift
        },
        "strata_where_knn_wins": [
            stratum for stratum, row in per_stratum_lift.items() if not row["model_beats_knn"]
        ],
        "reading": (
            "k-NN is a strong reference here: it sees the same 15 interview "
            "answers the encoder sees, and matches on them directly. The "
            "per-stratum split is the interesting part -- an encoder that only "
            "beat k-NN on items near the interview would be memorizing the "
            "interview rather than recovering a trait vector."
        ),
    }

    # ---- leakage trigger --------------------------------------------------
    triggers = []
    if graded[CONFIRMATORY_ARM]["correlation"] > LEAKAGE_CORRELATION:
        triggers.append(
            f"pooled correlation {graded[CONFIRMATORY_ARM]['correlation']:.4f} > {LEAKAGE_CORRELATION}"
        )
    if lift > LEAKAGE_LIFT:
        triggers.append(f"marginal lift {lift:.4f} > {LEAKAGE_LIFT}")
    leakage = {
        "leakage_hunt_required": bool(triggers),
        "triggers": triggers,
        "thresholds": {"correlation": LEAKAGE_CORRELATION, "lift": LEAKAGE_LIFT},
        "instruction": (
            "this flag is set by the grader and is NEVER cleared by it. The "
            "orchestrator investigates (PREREGISTRATION.md section 3, rule 5)."
        ),
    }

    # ---- aggregate checks -------------------------------------------------
    marginals = aggregate_marginals(
        np.asarray(predictions["probs_model"], dtype=float), empirical, n_categories, item_ids
    )
    tabs = cross_tabs(
        np.asarray(predictions["probs_model"], dtype=float), empirical, n_categories
    )

    # ---- exploratory ------------------------------------------------------
    noise_floor = float(
        np.mean(
            np.sum(truth.expected * (1.0 - truth.expected), axis=2)
            / np.asarray(n_categories, dtype=float)[None, :]
            / truth.n_samples
        )
    )
    decomposition = {
        "label": "EXPLORATORY",
        "what": (
            "2x2 over {fused theta_hat@15 vs the Gate 2 full-matrix theta_holdout} "
            "x {encoder loadings + trained thresholds vs the fitted held-out item "
            "parameters}. Plug-in throughout, so the only things that move are "
            "theta_hat and the item block"
        ),
        "brier": {
            name: graded[name]["brier"]
            for name in ("dec_fused_encoder", "dec_fused_fitted", "dec_full_encoder", "dec_full_fitted")
        },
        "theta_effect_at_encoder_items": graded["dec_fused_encoder"]["brier"] - graded["dec_full_encoder"]["brier"],
        "theta_effect_at_fitted_items": graded["dec_fused_fitted"]["brier"] - graded["dec_full_fitted"]["brier"],
        "item_effect_at_fused_theta": graded["dec_fused_encoder"]["brier"] - graded["dec_fused_fitted"]["brier"],
        "item_effect_at_full_theta": graded["dec_full_encoder"]["brier"] - graded["dec_full_fitted"]["brier"],
        "interaction": (
            graded["dec_fused_encoder"]["brier"] - graded["dec_fused_fitted"]["brier"]
        )
        - (graded["dec_full_encoder"]["brier"] - graded["dec_full_fitted"]["brier"]),
        "reading": "positive numbers are damage: the first arm scores worse than the second",
    }
    ceiling = {
        "label": "EXPLORATORY",
        "what": (
            "the analytic post-noise distribution of every cell, graded against "
            "its own 50-sample estimate. This is the best score any predictor "
            "could get, and the gap to it is the yardstick's own noise"
        ),
        "oracle_brier": graded["oracle"]["brier"],
        "oracle_log_loss": graded["oracle"]["log_loss"],
        "oracle_correlation": graded["oracle"]["correlation"],
        "oracle_ece": graded["oracle"]["ece"],
        "analytic_noise_floor_brier": noise_floor,
        "analytic_noise_floor_definition": "mean over cells of sum_c p(1-p) / (categories * 50)",
        "model_brier_over_oracle_brier": b_model / graded["oracle"]["brier"],
        "share_of_the_model_brier_that_is_yardstick_noise": graded["oracle"]["brier"] / b_model,
    }
    integration = {
        "label": "EXPLORATORY",
        "what": "posterior-integrated vs plug-in, the same model otherwise",
        "posterior_brier": b_model,
        "plugin_brier": graded["model_plugin"]["brier"],
        "posterior_ece": graded[CONFIRMATORY_ARM]["ece"],
        "plugin_ece": graded["model_plugin"]["ece"],
        "posterior_correlation": graded[CONFIRMATORY_ARM]["correlation"],
        "plugin_correlation": graded["model_plugin"]["correlation"],
        "brier_gain_from_integrating": graded["model_plugin"]["brier"] - b_model,
    }

    # ---- write ------------------------------------------------------------
    cells_path = out_dir / f"{out_prefix}_cells.npz"
    cell_payload: dict[str, Any] = {
        "persona_ids": np.array(persona_ids),
        "item_ids": np.array(item_ids),
        "item_kinds": np.array(kinds),
        "n_categories": n_categories,
        "strata": np.array(strata),
        "truth_empirical": empirical,
        "truth_counts": truth.counts,
    }
    for name, block in graded.items():
        cell_payload[f"brier_{name}"] = block["cells"]["brier"]
        cell_payload[f"log_loss_{name}"] = block["cells"]["log_loss"]
    cell_payload["aggregate_predicted_mean"] = marginals["arrays"]["predicted_mean"]
    cell_payload["aggregate_truth_mean"] = marginals["arrays"]["truth_mean"]
    cell_payload["crosstab_predicted"] = tabs["arrays"]["predicted"]
    cell_payload["crosstab_truth"] = tabs["arrays"]["truth"]
    np.savez_compressed(cells_path, **cell_payload)

    headline = [a.name for a in arms]
    plot_paths = {
        "reliability": out_dir / f"{out_prefix}_reliability.png",
        "stratum_lift": out_dir / f"{out_prefix}_stratum_lift.png",
        "aggregate": out_dir / f"{out_prefix}_aggregate.png",
    }
    plot_reliability(plot_paths["reliability"], graded, headline)
    plot_stratum_lift(plot_paths["stratum_lift"], graded, headline)
    plot_aggregate(plot_paths["aggregate"], marginals, tabs)

    report: dict[str, Any] = {
        "module": "src.eval.gate4",
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "verdict": verdict,
        "leakage": leakage,
        "metric_definitions": {
            "brier": (
                "per cell, the mean over that item's own categories of "
                "(predicted - empirical 50-sample frequency)^2; reported as the "
                "mean over the 5,000 cells"
            ),
            "log_loss": (
                "per cell, -sum_c p_true,c log(max(p_pred,c, eps)) over the "
                f"item's own categories, eps = {LOG_EPS}"
            ),
            "ece": (
                f"{N_BINS} equal-width bins over every (cell, category) predicted "
                "probability, count-weighted absolute gap to the mean empirical "
                "frequency in the bin"
            ),
            "correlation": (
                "Pearson between predicted probability and empirical 50-sample "
                "frequency, pooled over every (cell, category) pair"
            ),
        },
        "arms": {
            name: {k: v for k, v in block.items() if k != "cells"}
            for name, block in graded.items()
        },
        "per_stratum_comparison": per_stratum_lift,
        "qwen_no_interview_baseline": qwen_status,
        "aggregate_check": {k: v for k, v in marginals.items() if k != "arrays"},
        "cross_tabs": {k: v for k, v in tabs.items() if k != "arrays"},
        "exploratory": {
            "error_decomposition_2x2": decomposition,
            "oracle_ceiling_and_noise_floor": ceiling,
            "posterior_vs_plug_in": integration,
        },
        "alignment": {k: v for k, v in alignment.items()},
        "counts": {
            "holdout_personas": len(persona_ids),
            "probe_items": len(item_ids),
            "cells": len(persona_ids) * len(item_ids),
            "truth_samples_per_cell": truth.n_samples,
            "stratum_counts": {
                s: int(sum(1 for x in strata if x == s)) for s in sorted(set(strata))
            },
        },
        "inputs": {
            "truth50": str(truth50_dir),
            "predictions": str(predictions_path),
            "baselines": str(baselines_path),
            "splits": str(splits_path),
        },
        "artifacts": {
            "cells": str(cells_path),
            **{k: str(v) for k, v in plot_paths.items()},
        },
        "runtime_seconds": time.time() - started,
    }

    out_path = out_dir / f"{out_prefix}_gate4.json"
    out_path.write_text(json.dumps(_json_safe(report), indent=2) + "\n", encoding="utf-8")
    if verbose:
        print(json.dumps(_json_safe(verdict["bars"]), indent=2))
        print(f"verdict: all bars pass = {verdict['all_bars_pass']}, failed {verdict['failed_bars']}")
        print(f"leakage_hunt_required = {leakage['leakage_hunt_required']} {leakage['triggers']}")
        print(f"report -> {out_path}")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.eval.gate4",
        description="Grade the Gate 4 bars against the 50-sample true answer distributions.",
    )
    parser.add_argument("--truth50", type=Path, required=True,
                        help="directory written by src.eval.truth_distributions")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--baselines", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, default=None,
                        help="parsed no-interview LLM baseline; graded if present")
    parser.add_argument("--out-prefix", default=DEFAULT_OUT_PREFIX)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    run(
        args.truth50,
        args.predictions,
        args.baselines,
        args.splits,
        args.out,
        qwen_path=args.qwen,
        out_prefix=args.out_prefix,
        verbose=not args.quiet,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
