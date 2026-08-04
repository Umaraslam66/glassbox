"""EXPLORATORY -- why the Gate 4 correlation bar came in at 0.684 and not 0.90.

    python -m src.eval.corr_attribution \\
        --truth50 <truth runs dir>/truth50 \\
        --predictions results/stage4_predictions.npz \\
        --baselines results/stage4_baselines.npz \\
        --splits experiments/splits_v1.json \\
        --out results

Grader-side, read-only, and **entirely exploratory**. Nothing here is a
pre-registered bar, nothing here re-fits or re-calibrates anything, and nothing
here touches ``src/eval/gate4.py`` or ``results/stage4_gate4.json``. The frozen
number stays what the grader wrote: pooled Pearson 0.6843654025031514 against a
bar of 0.90.

What this module answers
------------------------
1. **The ladder.** The same pooled correlation the bar uses -- every (cell,
   category) pair, 100 held-out personas x 50 probe items -- computed for every
   arm the Stage 4 artifacts carry, ordered from the zero-information marginal
   up to the truth-side oracle. Split binary / Likert per rung.

   Three differences on that ladder name the three losses:

   ``oracle - full x fitted``   what an 8-dimensional graded-response model
                                cannot express even with the full answer matrix
                                behind theta and the held-out items fitted;
   ``full x fitted - fused x fitted``  what is lost by seeing 15 interview
                                answers instead of the whole matrix;
   ``fused x fitted - model``   what is lost by reading the item parameters off
                                the item text zero-shot instead of fitting them.

   They telescope, so they sum exactly to the oracle-to-model gap.

2. **Flatness.** Predictions spread over a narrower range than the truth does.
   Reported as the standard deviation of the predicted probabilities against the
   standard deviation of the empirical frequencies, the ordinary-least-squares
   line of empirical on predicted, mean entropies, and how much of the truth
   sits pinned at 0 or 1.

3. **The variance split.** The empirical frequencies vary for two reasons: items
   differ from each other, and people differ within an item. A one-way split
   over the (item, category) slots separates the two, which bounds what any
   item-level-only predictor -- the population marginal, for one -- can score,
   and says how much of the pooled correlation is person-level signal at all.
   The within-item lens is the median per-item correlation across personas.

4. **Sanity.** Every pooled number recomputed here is checked against
   ``results/stage4_gate4.json`` to 1e-9 before anything else is believed.

WALL. Same rule as the grader: this reads the truth store and the finished arms,
and writes only to ``results/``. Nothing computed here goes back to the system
side.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

# The bar's own helpers, imported rather than re-implemented so the pooling here
# is the pooling the grader used, by construction.
from .gate4 import BAR_CORRELATION, category_mask, pair_arrays, pearson
from .truth_distributions import MAX_CATEGORIES, Truth50

#: How close a recomputed pooled number has to be to the graded one before the
#: rest of this report is trusted at all.
SANITY_TOLERANCE = 1e-9

#: "Sharp" means an empirical frequency pinned near 0 or near 1. With 50 samples
#: the grid is 0.02 wide, so these cover counts 0-2 and 48-50.
SHARP_LOW = 0.05
SHARP_HIGH = 0.95

#: The rungs, bottom to top, as (arm name, short label). Every one of these is
#: an array the Stage 4 artifacts already carry.
LADDER: tuple[tuple[str, str], ...] = (
    ("marginal", "population marginal (zero information)"),
    ("profile", "public profile only"),
    ("model_text_thresholds", "fused theta x fully zero-shot item side"),
    ("knn", "k-NN on the 15 interview answers"),
    ("model_plugin", "CONFIRMATORY model, posterior collapsed to its mean"),
    ("model", "CONFIRMATORY model, posterior-integrated"),
    ("dec_full_encoder", "full-matrix theta x encoder item parameters"),
    ("dec_fused_fitted", "fused theta@15 x fitted item parameters"),
    ("full_matrix_posterior", "full-matrix theta x fitted items, posterior-integrated"),
    ("dec_full_fitted", "full-matrix theta x fitted item parameters (CEILING)"),
    ("oracle", "analytic post-noise distribution (truth-side oracle)"),
)

#: The three arms the attribution differences are taken between.
ARM_MODEL = "model"
ARM_FUSED_FITTED = "dec_fused_fitted"
ARM_FULL_FITTED = "dec_full_fitted"
ARM_ORACLE = "oracle"

DEFAULT_OUT_PREFIX = "stage4"


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def load_arms(
    predictions_path: Path, baselines_path: Path
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Every (100, 50, 5) probability array the Stage 4 artifacts hold, by name."""
    predictions = np.load(predictions_path, allow_pickle=False)
    baselines = np.load(baselines_path, allow_pickle=False)
    arms = {
        "model": predictions["probs_model"],
        "model_plugin": predictions["probs_model_plugin"],
        "model_text_thresholds": predictions["probs_model_text_thresholds"],
        "full_matrix_posterior": predictions["probs_full_matrix_posterior"],
        "dec_fused_encoder": predictions["probs_dec_fused_encoder"],
        "dec_fused_fitted": predictions["probs_dec_fused_fitted"],
        "dec_full_encoder": predictions["probs_dec_full_encoder"],
        "dec_full_fitted": predictions["probs_dec_full_fitted"],
        "marginal": baselines["probs_marginal"],
        "profile": baselines["probs_profile"],
        "knn": baselines["probs_knn"],
    }
    meta = {
        "persona_ids": [str(p) for p in predictions["persona_ids"]],
        "item_ids": [str(i) for i in predictions["item_ids"]],
        "item_kinds": [str(k) for k in predictions["item_kinds"]],
        "n_categories": np.asarray(predictions["n_categories"], dtype=int),
    }
    return {k: np.asarray(v, dtype=float) for k, v in arms.items()}, meta


# --------------------------------------------------------------------------
# the pieces
# --------------------------------------------------------------------------


def ols_line(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Least-squares fit of ``y`` on ``x``: slope, intercept, r, r squared.

    A slope below 1 with a positive intercept is the signature of a flat
    predictor: the truth moves further than the prediction does.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    r = pearson(x, y)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r": float(r),
        "r_squared": float(r * r),
    }


def entropy_cells(probs: np.ndarray, n_categories: np.ndarray) -> np.ndarray:
    """(people, items) Shannon entropy in nats over each item's own categories.

    Zero probabilities contribute zero, the usual ``0 log 0 = 0`` convention.
    A uniform binary cell is ln 2 = 0.693; a uniform Likert-5 cell is
    ln 5 = 1.609.
    """
    mask = category_mask(n_categories)[None, :, :]
    safe = np.where(probs > 0.0, probs, 1.0)
    return -np.where(mask & (probs > 0.0), probs * np.log(safe), 0.0).sum(axis=2)


def sharp_fraction(
    values: np.ndarray, *, low: float = SHARP_LOW, high: float = SHARP_HIGH
) -> dict[str, float]:
    """Share of a flat array of probabilities pinned near 0, near 1, or either."""
    values = np.asarray(values, dtype=float)
    near_zero = float(np.mean(values <= low))
    near_one = float(np.mean(values >= high))
    return {
        "at_or_below_0.05": near_zero,
        "at_or_above_0.95": near_one,
        "either_end": near_zero + near_one,
    }


def variance_split(truth: np.ndarray, n_categories: np.ndarray) -> dict[str, Any]:
    """Split the variance of the empirical frequencies over (item, category) slots.

    Every slot -- one category column of one probe item -- holds the same 100
    held-out personas, so the one-way split is exact:

        total = between-slot + within-slot

    *Between* is how much slots differ from each other: item-level signal, the
    only kind a predictor that ignores the person can carry. *Within* is how much
    people differ inside one slot: the person-level signal Stage 4 exists to
    recover. The square root of the between share is the pooled correlation a
    perfect item-level-only predictor would reach, which is the honest reference
    for the population marginal's rung on the ladder.
    """
    mask = category_mask(n_categories)
    people = truth.shape[0]
    slots = truth[:, mask]  # (people, valid slots)
    grand = float(slots.mean())
    slot_means = slots.mean(axis=0)

    total = float(((slots - grand) ** 2).mean())
    between = float(((slot_means - grand) ** 2).mean())
    within = float(((slots - slot_means[None, :]) ** 2).mean())
    return {
        "n_slots": int(slot_means.size),
        "n_personas_per_slot": int(people),
        "grand_mean": grand,
        "total_variance": total,
        "between_item_variance": between,
        "within_item_between_persona_variance": within,
        "between_item_share": between / total,
        "within_item_share": within / total,
        "identity_residual": total - (between + within),
        "best_item_level_only_correlation": float(np.sqrt(between / total)),
        "reading": (
            "between-slot variance is item-level signal; within-slot variance is "
            "person-level signal. sqrt(between share) is the pooled correlation "
            "a predictor that knew every item's population marginal exactly, and "
            "nothing about the person, would score."
        ),
    }


def per_item_correlations(
    predicted: np.ndarray, truth: np.ndarray, n_categories: np.ndarray
) -> np.ndarray:
    """One correlation per probe item, pooled over that item's (persona, category) pairs.

    Between-item variance is gone by construction, so what is left is how well
    the arm ranks *people* on one item (plus that item's own category shape).
    """
    out = np.full(predicted.shape[1], np.nan)
    for col in range(predicted.shape[1]):
        keep = slice(0, int(n_categories[col]))
        out[col] = pearson(
            predicted[:, col, keep].ravel(), truth[:, col, keep].ravel()
        )
    return out


def per_slot_correlations(
    predicted: np.ndarray, truth: np.ndarray, n_categories: np.ndarray
) -> np.ndarray:
    """One correlation per (item, category) slot, across the 100 personas.

    The strictest within-item lens: no between-item and no between-category
    structure survives it, so a slot correlation is person-level signal only.
    """
    values = []
    for col in range(predicted.shape[1]):
        for cat in range(int(n_categories[col])):
            values.append(pearson(predicted[:, col, cat], truth[:, col, cat]))
    return np.asarray(values, dtype=float)


def summarise(values: np.ndarray) -> dict[str, Any]:
    """Median / mean / quartiles of a set of correlations, NaNs dropped and counted."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"n": 0, "n_undefined": int(values.size), "median": None}
    return {
        "n": int(finite.size),
        "n_undefined": int(values.size - finite.size),
        "median": float(np.median(finite)),
        "mean": float(finite.mean()),
        "q25": float(np.percentile(finite, 25)),
        "q75": float(np.percentile(finite, 75)),
        "min": float(finite.min()),
        "max": float(finite.max()),
    }


# --------------------------------------------------------------------------
# the ladder
# --------------------------------------------------------------------------


def ladder_rung(
    probs: np.ndarray,
    truth: np.ndarray,
    n_categories: np.ndarray,
    kinds: Sequence[str],
) -> dict[str, Any]:
    """Pooled correlation for one arm, plus the binary / Likert split."""
    flat_pred, flat_true = pair_arrays(probs, truth, n_categories)
    is_binary = np.array([k != "likert5" for k in kinds])

    split: dict[str, Any] = {}
    for name, mask in (("binary", is_binary), ("likert5", ~is_binary)):
        if not mask.any():
            continue
        p, t = pair_arrays(
            probs[:, mask, :], truth[:, mask, :], np.asarray(n_categories)[mask]
        )
        split[name] = {"n_items": int(mask.sum()), "correlation": pearson(p, t)}

    return {
        "correlation": pearson(flat_pred, flat_true),
        "binary_only": split.get("binary", {}).get("correlation"),
        "likert_only": split.get("likert5", {}).get("correlation"),
        "n_pairs": int(flat_pred.size),
        "predicted_std": float(flat_pred.std()),
        "predicted_mean": float(flat_pred.mean()),
    }


def attribution(rungs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The three telescoping losses between the oracle and the confirmatory arm."""
    oracle = float(rungs[ARM_ORACLE]["correlation"])
    full_fitted = float(rungs[ARM_FULL_FITTED]["correlation"])
    fused_fitted = float(rungs[ARM_FUSED_FITTED]["correlation"])
    model = float(rungs[ARM_MODEL]["correlation"])

    gap = oracle - model
    terms = {
        "model_class_8dim": {
            "what": (
                "oracle minus full-matrix theta x fitted items: what an 8-dim "
                "graded-response model cannot express even given every answer "
                "and every held-out item fitted"
            ),
            "from": ARM_ORACLE,
            "to": ARM_FULL_FITTED,
            "loss": oracle - full_fitted,
        },
        "information_loss_at_n15": {
            "what": (
                "full-matrix theta minus fused theta@15, item side held fixed at "
                "the fitted parameters: the cost of 15 interview answers instead "
                "of the whole answer matrix"
            ),
            "from": ARM_FULL_FITTED,
            "to": ARM_FUSED_FITTED,
            "loss": full_fitted - fused_fitted,
        },
        "zero_shot_item_encoder": {
            "what": (
                "fitted item parameters minus the confirmatory arm's zero-shot "
                "loadings and trained thresholds, theta held fixed at fused@15"
            ),
            "from": ARM_FUSED_FITTED,
            "to": ARM_MODEL,
            "loss": fused_fitted - model,
        },
    }
    for term in terms.values():
        term["share_of_oracle_to_model_gap"] = term["loss"] / gap
    return {
        "label": "EXPLORATORY",
        "oracle_correlation": oracle,
        "model_correlation": model,
        "oracle_to_model_gap": gap,
        "terms": terms,
        "terms_sum": sum(t["loss"] for t in terms.values()),
        "telescopes_exactly": bool(
            abs(sum(t["loss"] for t in terms.values()) - gap) < 1e-12
        ),
        "distance_from_the_frozen_bar": {
            "bar": BAR_CORRELATION,
            "model_shortfall": BAR_CORRELATION - model,
            "ceiling_shortfall": BAR_CORRELATION - full_fitted,
            "reading": (
                "the ceiling arm is itself below the bar, so closing every "
                "loss this project controls still would not reach 0.90"
            ),
        },
    }


# --------------------------------------------------------------------------
# sanity
# --------------------------------------------------------------------------


def sanity_check(
    rungs: Mapping[str, Mapping[str, Any]],
    graded: Mapping[str, Any],
    *,
    tolerance: float = SANITY_TOLERANCE,
) -> dict[str, Any]:
    """Recomputed pooled correlations against the frozen Gate 4 report."""
    checks = []
    for name, rung in rungs.items():
        arm = graded["arms"].get(name)
        if arm is None:
            continue
        for key, mine in (
            ("correlation", rung["correlation"]),
            ("binary_only", rung["binary_only"]),
            ("likert_only", rung["likert_only"]),
        ):
            if key == "correlation":
                theirs = arm["correlation"]
            elif key == "binary_only":
                theirs = arm["per_item_type"]["binary"]["correlation"]
            else:
                theirs = arm["per_item_type"]["likert5"]["correlation"]
            checks.append(
                {
                    "arm": name,
                    "quantity": key,
                    "recomputed": float(mine),
                    "gate4": float(theirs),
                    "abs_difference": abs(float(mine) - float(theirs)),
                    "matches": bool(abs(float(mine) - float(theirs)) <= tolerance),
                }
            )

    bar = graded["verdict"]["bars"]["correlation"]
    checks.append(
        {
            "arm": ARM_MODEL,
            "quantity": "the frozen bar value itself",
            "recomputed": float(rungs[ARM_MODEL]["correlation"]),
            "gate4": float(bar["value"]),
            "abs_difference": abs(float(rungs[ARM_MODEL]["correlation"]) - float(bar["value"])),
            "matches": bool(
                abs(float(rungs[ARM_MODEL]["correlation"]) - float(bar["value"])) <= tolerance
            ),
        }
    )
    failures = [c for c in checks if not c["matches"]]
    return {
        "tolerance": tolerance,
        "n_checks": len(checks),
        "all_match": not failures,
        "failures": failures,
        "checks": checks,
    }


# --------------------------------------------------------------------------
# plot
# --------------------------------------------------------------------------


def plot_attribution(
    path: Path,
    predicted: np.ndarray,
    observed: np.ndarray,
    line: Mapping[str, float],
    rungs: Mapping[str, Mapping[str, Any]],
    std_ratio: float,
) -> None:
    """Left: the confirmatory arm's cloud with identity and OLS. Right: the ladder."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6), gridspec_kw={"width_ratios": [1.05, 1]})

    ax = axes[0]
    ax.scatter(predicted, observed, s=3, alpha=0.08, color="#4C72B0", edgecolors="none")
    grid = np.linspace(0.0, 1.0, 100)
    ax.plot(grid, grid, color="#333333", lw=1.2, ls=":", label="identity")
    ax.plot(
        grid,
        line["intercept"] + line["slope"] * grid,
        color="#C44E52",
        lw=1.8,
        label=f"OLS: {line['slope']:.3f} x + {line['intercept']:.3f}",
    )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("predicted probability (confirmatory arm)")
    ax.set_ylabel("empirical 50-sample frequency")
    ax.set_title(
        f"EXPLORATORY -- confirmatory arm, {predicted.size:,} (cell, category) pairs\n"
        f"pooled r = {line['r']:.4f}    spread {std_ratio:.2f}x the truth's\n"
        f"OLS slope {line['slope']:.3f}: noisy, not shrunk",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="upper left")

    ax = axes[1]
    labels = {n: label for n, label in LADDER}
    names = sorted(
        (n for n, _ in LADDER if n in rungs), key=lambda n: rungs[n]["correlation"]
    )
    values = [rungs[n]["correlation"] for n in names]
    y = np.arange(len(names))
    colors = []
    for n in names:
        if n == ARM_MODEL:
            colors.append("#4C72B0")
        elif n in (ARM_FULL_FITTED, ARM_ORACLE, ARM_FUSED_FITTED):
            colors.append("#55A868")
        else:
            colors.append("#B0B0B0")
    ax.barh(y, values, color=colors, height=0.66)
    for pos, value in zip(y, values):
        ax.text(value + 0.008, pos, f"{value:.3f}", va="center", fontsize=8.5)
    ax.axvline(BAR_CORRELATION, color="#C44E52", lw=2, ls="--",
               label=f"frozen bar {BAR_CORRELATION:.2f}")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{labels[n]}" for n in names], fontsize=8)
    ax.set_xlim(0.0, 1.08)
    ax.set_xlabel("pooled Pearson r (predicted vs empirical)")
    ax.set_title("EXPLORATORY -- the correlation ladder")
    ax.legend(fontsize=9, loc="lower right")

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
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
    gate4_report: Path,
    out_dir: Path,
    *,
    out_prefix: str = DEFAULT_OUT_PREFIX,
    verbose: bool = True,
) -> dict[str, Any]:
    started = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    truth = Truth50(truth50_dir)
    arms, meta = load_arms(Path(predictions_path), Path(baselines_path))
    splits = json.loads(Path(splits_path).read_text(encoding="utf-8"))
    graded = json.loads(Path(gate4_report).read_text(encoding="utf-8"))

    persona_ids = meta["persona_ids"]
    item_ids = meta["item_ids"]
    kinds = meta["item_kinds"]
    n_categories = meta["n_categories"]
    strata = [str(splits["strata"][i]) for i in item_ids]

    if persona_ids != truth.persona_ids or item_ids != truth.item_ids:
        raise ValueError("the arms and the truth store do not line up")
    if not np.array_equal(n_categories, truth.n_categories):
        raise ValueError("category counts disagree between the arms and the truth store")
    if strata != truth.strata:
        raise ValueError("strata disagree between the splits file and the truth store")
    if not all(truth.verify_checksums().values()):
        raise ValueError("the 50-sample truth files do not match their recorded checksums")

    empirical = truth.empirical
    arms[ARM_ORACLE] = np.asarray(truth.expected, dtype=float)

    # ---- 1. the ladder ----------------------------------------------------
    rungs = {
        name: ladder_rung(probs, empirical, n_categories, kinds)
        for name, probs in arms.items()
    }

    # ---- 4. sanity, before anything else is believed ----------------------
    sanity = sanity_check(rungs, graded)
    if not sanity["all_match"]:
        raise ValueError(
            "recomputed pooled correlations do not match results/stage4_gate4.json: "
            + json.dumps(_json_safe(sanity["failures"]), indent=2)
        )

    # ---- 2. flatness ------------------------------------------------------
    is_binary = np.array([k != "likert5" for k in kinds])
    truth_flat, _ = pair_arrays(empirical, empirical, n_categories)
    truth_binary, _ = pair_arrays(
        empirical[:, is_binary, :], empirical[:, is_binary, :], n_categories[is_binary]
    )
    truth_likert, _ = pair_arrays(
        empirical[:, ~is_binary, :], empirical[:, ~is_binary, :], n_categories[~is_binary]
    )
    truth_entropy = entropy_cells(empirical, n_categories)

    spread: dict[str, Any] = {}
    for name, probs in arms.items():
        p_all, _ = pair_arrays(probs, empirical, n_categories)
        p_bin, _ = pair_arrays(
            probs[:, is_binary, :], empirical[:, is_binary, :], n_categories[is_binary]
        )
        p_lik, _ = pair_arrays(
            probs[:, ~is_binary, :], empirical[:, ~is_binary, :], n_categories[~is_binary]
        )
        entropy = entropy_cells(probs, n_categories)
        spread[name] = {
            "predicted_std_pooled": float(p_all.std()),
            "predicted_std_binary": float(p_bin.std()),
            "predicted_std_likert5": float(p_lik.std()),
            "std_ratio_pooled": float(p_all.std() / truth_flat.std()),
            "std_ratio_binary": float(p_bin.std() / truth_binary.std()),
            "std_ratio_likert5": float(p_lik.std() / truth_likert.std()),
            "mean_entropy_nats": float(entropy.mean()),
            "mean_entropy_binary": float(entropy[:, is_binary].mean()),
            "mean_entropy_likert5": float(entropy[:, ~is_binary].mean()),
            "sharp_fraction": sharp_fraction(p_all),
            "ols_empirical_on_predicted": ols_line(p_all, truth_flat),
        }

    flatness = {
        "label": "EXPLORATORY",
        "what": (
            "how far the predicted probabilities spread compared with the "
            "empirical 50-sample frequencies they are graded against"
        ),
        "truth": {
            "empirical_std_pooled": float(truth_flat.std()),
            "empirical_std_binary": float(truth_binary.std()),
            "empirical_std_likert5": float(truth_likert.std()),
            "empirical_mean_pooled": float(truth_flat.mean()),
            "mean_entropy_nats": float(truth_entropy.mean()),
            "mean_entropy_binary": float(truth_entropy[:, is_binary].mean()),
            "mean_entropy_likert5": float(truth_entropy[:, ~is_binary].mean()),
            "sharp_fraction": sharp_fraction(truth_flat),
            "uniform_entropy_reference": {
                "binary_ln2": float(np.log(2.0)),
                "likert5_ln5": float(np.log(5.0)),
            },
        },
        "per_arm": spread,
        "confirmatory_ols": spread[ARM_MODEL]["ols_empirical_on_predicted"],
        "reading": (
            "read the two numbers together. The predicted spread is well below "
            "the truth's, but the OLS slope of empirical on predicted sits near "
            "1, which is what the low ECE already said: the arm is not "
            "systematically shrunk, it is noisy. Pearson is unchanged by any "
            "shared affine squash, so flatness is a description of the miss and "
            "not on its own a cause of it -- slope = r * sd(truth) / "
            "sd(predicted) ties the three together"
        ),
    }

    # ---- 3. the variance split and the within-item lens -------------------
    split = variance_split(empirical, n_categories)
    within_item: dict[str, Any] = {}
    for name, probs in arms.items():
        within_item[name] = {
            "per_item": summarise(per_item_correlations(probs, empirical, n_categories)),
            "per_item_category_slot": summarise(
                per_slot_correlations(probs, empirical, n_categories)
            ),
        }

    marginal_r = float(rungs["marginal"]["correlation"])
    model_r = float(rungs[ARM_MODEL]["correlation"])
    variance = {
        "label": "EXPLORATORY",
        "split": split,
        "what_the_marginal_already_captures": {
            "marginal_correlation": marginal_r,
            "model_correlation": model_r,
            "share_of_the_model_r_the_marginal_already_has": marginal_r / model_r,
            "share_of_the_model_r_squared_the_marginal_already_has": (
                marginal_r**2 / model_r**2
            ),
            "share_of_the_bar_the_marginal_already_has": marginal_r / BAR_CORRELATION,
            "item_level_only_ceiling": split["best_item_level_only_correlation"],
            "marginal_share_of_the_item_level_ceiling": (
                marginal_r / split["best_item_level_only_correlation"]
            ),
            "model_gain_over_the_item_level_ceiling": (
                model_r - split["best_item_level_only_correlation"]
            ),
            "reading": (
                "the population marginal knows nothing about any person, so "
                "everything it scores is item-level signal. What the "
                "confirmatory arm adds on top of it is the person-level part of "
                "the pooled correlation"
            ),
        },
        "within_item_correlations": {
            "what": (
                "EXPLORATORY lens only -- the frozen bar is the pooled number "
                "and stays the graded one. Per item, the correlation is taken "
                "over that item's (persona, category) pairs; per slot, over the "
                "100 personas of one (item, category) column"
            ),
            "per_arm": within_item,
        },
    }

    # ---- the attribution --------------------------------------------------
    attributed = attribution(rungs)

    # ---- plot -------------------------------------------------------------
    model_pred, model_true = pair_arrays(arms[ARM_MODEL], empirical, n_categories)
    plot_path = out_dir / f"{out_prefix}_corr_attribution.png"
    plot_attribution(
        plot_path,
        model_pred,
        model_true,
        spread[ARM_MODEL]["ols_empirical_on_predicted"],
        rungs,
        spread[ARM_MODEL]["std_ratio_pooled"],
    )

    summary = {
        "headline": (
            "the correlation bar fails structurally, not by a calibration "
            "slip: the 8-dim model class explains most of the gap, and the "
            "ceiling arm is itself below the bar"
        ),
        "bar": BAR_CORRELATION,
        "confirmatory_correlation": model_r,
        "ceiling_correlation": float(rungs[ARM_FULL_FITTED]["correlation"]),
        "oracle_correlation": float(rungs[ARM_ORACLE]["correlation"]),
        "losses": {
            key: {
                "loss": term["loss"],
                "share_of_oracle_to_model_gap": term["share_of_oracle_to_model_gap"],
            }
            for key, term in attributed["terms"].items()
        },
        "confirmatory_ols_slope": spread[ARM_MODEL]["ols_empirical_on_predicted"]["slope"],
        "confirmatory_std_ratio": spread[ARM_MODEL]["std_ratio_pooled"],
        "truth_sharp_fraction_either_end": flatness["truth"]["sharp_fraction"]["either_end"],
        "between_item_share_of_truth_variance": split["between_item_share"],
        "marginal_share_of_the_model_correlation": marginal_r / model_r,
        "median_per_item_correlation": {
            name: within_item[name]["per_item"]["median"]
            for name in (ARM_MODEL, "knn", ARM_FUSED_FITTED, ARM_FULL_FITTED, ARM_ORACLE)
        },
    }

    report = {
        "module": "src.eval.corr_attribution",
        "status": "EXPLORATORY -- not a pre-registered bar, nothing here is confirmatory",
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "question": (
            "the Gate 4 correlation bar failed at 0.684 against 0.90. Where do "
            "the missing 0.31 correlation points sit?"
        ),
        "summary": summary,
        "sanity_check": sanity,
        "ladder": {
            "label": "EXPLORATORY",
            "pooling": (
                "Pearson between predicted probability and empirical 50-sample "
                "frequency over every (cell, category) pair -- the same pooling "
                "the frozen bar uses, 100 held-out personas x 50 probe items"
            ),
            "order": [name for name, _ in LADDER if name in rungs],
            "labels": {name: label for name, label in LADDER},
            "rungs": rungs,
        },
        "attribution": attributed,
        "flatness": flatness,
        "variance": variance,
        "counts": {
            "holdout_personas": len(persona_ids),
            "probe_items": len(item_ids),
            "cells": len(persona_ids) * len(item_ids),
            "cell_category_pairs": int(model_pred.size),
            "binary_items": int(is_binary.sum()),
            "likert5_items": int((~is_binary).sum()),
            "truth_samples_per_cell": truth.n_samples,
        },
        "inputs": {
            "truth50": str(truth50_dir),
            "predictions": str(predictions_path),
            "baselines": str(baselines_path),
            "splits": str(splits_path),
            "gate4_report": str(gate4_report),
        },
        "artifacts": {"plot": str(plot_path)},
        "runtime_seconds": time.time() - started,
    }

    out_path = out_dir / f"{out_prefix}_corr_attribution.json"
    out_path.write_text(json.dumps(_json_safe(report), indent=2) + "\n", encoding="utf-8")

    if verbose:
        print(f"sanity: {sanity['n_checks']} pooled numbers match gate4 to {SANITY_TOLERANCE}")
        for name, _ in LADDER:
            if name not in rungs:
                continue
            rung = rungs[name]
            print(
                f"  {name:24s} r {rung['correlation']:.4f}"
                f"  binary {rung['binary_only']:.4f}  likert {rung['likert_only']:.4f}"
            )
        for key, term in attributed["terms"].items():
            print(
                f"  loss {key:24s} {term['loss']:+.4f}"
                f"  ({100 * term['share_of_oracle_to_model_gap']:.1f}% of the gap)"
            )
        print(f"report -> {out_path}")
        print(f"plot   -> {plot_path}")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.eval.corr_attribution",
        description=(
            "EXPLORATORY: attribute the Gate 4 correlation shortfall to model "
            "class, interview length, and the zero-shot item encoder."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--truth50", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--baselines", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--gate4", type=Path, default=Path("results/stage4_gate4.json"),
                        help="the frozen verdict, read only to sanity-check the recomputation")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--out-prefix", default=DEFAULT_OUT_PREFIX)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    run(
        args.truth50,
        args.predictions,
        args.baselines,
        args.splits,
        args.gate4,
        args.out,
        out_prefix=args.out_prefix,
        verbose=not args.quiet,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
