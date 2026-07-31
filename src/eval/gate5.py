"""Gate 5 grader: how many questions each strategy needs (PREREGISTRATION section 6).

    python -m src.eval.gate5 \
        --episodes data/runs/stage5/episode_codes.npz \
        --fit results/stage2_v2_fit.npz \
        --backbone results/stage3_person_backbone.npz \
        --splits experiments/splits_v1.json \
        --truth <planted dir> \
        --out results

Grader-side code: it lives in ``src/eval/``, so it may read the planted design.
The directory holding that design is a command-line argument; no path to it is
spelled out here.

What it measures
----------------
Three non-RL strategies -- random order, the Stage 3 scripted order, and the
information-gain heuristic -- on the 100 ORIGINAL held-out personas, over
independent seeded episodes. (The RL policy is added later; it plugs into the
same interface and is graded by the same code.) For each episode the strategy
asks one closed training item at a time, the persona answers through the frozen
noise layer, and the sequential posterior is updated
(:mod:`src.model.sequential_posterior`). After every question the point estimate
is rotated onto the planted axes by orthogonal Procrustes fitted on the 400
TRAINING personas' own estimate at that N -- the same convention, and the same
function, Gate 2 and Gate 3 used -- and the per-dimension RMSE against planted
theta is recorded.

The frozen bar, and how it is read
----------------------------------
PREREGISTRATION section 6, Gate 5(a): "Median questions-to-target for the
adaptive interviewer (RL or info-gain heuristic) <= 50% of median
questions-to-target for random ordering", with target = the Gate 3 accuracy
level, per-dimension RMSE <= 0.5.

"Questions-to-target" is not defined further in the frozen text, so the reading
used here is documented as a grader's reading, both lenses preserved, exactly as
the Gate 2 weak-item check and the Gate 3 monotonicity call were:

* PRIMARY (the one the bar is graded on) -- **population-curve crossing**: for
  each replicate, the first N at which the population's per-dimension RMSE is at
  or below the target on all eight dimensions; the median of that across
  replicates. This is the reading that matches how every other bar in this study
  is computed (a population number, per dimension, on the held-out set).
* SECONDARY (reported beside it, never graded) -- **per-persona crossing**: for
  each persona in each replicate, the first N at which that persona's own
  absolute error is at or below the target on all eight dimensions.

Owner RULING 1 (recorded before any Stage 5 compute) governs the case the
Stage 2 and Stage 3 numbers make likely: the target may be unreachable at any N,
because the full 252-item matrix itself only reaches RMSE 0.529. If a strategy
never reaches the target within the question budget, that is reported as it is
-- the bar is unmet or undefined, and is not re-read to make it passable.

Pre-declared exploratory analysis (RULING 1, and labeled as such everywhere it
appears): the same questions-to-target grid at reachable thresholds 0.60, 0.65
and 0.70 with the 50%-of-random comparison at each; the fraction of
full-matrix information reached at N = 5, 10, 15 and 25; and the declared-blur
against truth-RMSE tracking arrays, which are the watch for proxy gaming that
RULING 2 requires. One further owner-approved exploratory arm re-runs random and
the heuristic with item-encoder-PREDICTED item parameters in place of the fitted
ones, which prices the product's cold start.

WALL. Nothing computed here is written back to the system side. The episodes are
simulated by :mod:`src.interview.episodes`, which draws answers through the
frozen noise layer and never lets a strategy see anything but the answers it
asked for; this module adds planted truth only to score what came out.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..interview.episodes import AnswerBank, EpisodeSimulator, episode_seed, round_tag
from ..interview.strategies import FixedOrder, InfoGain, RandomOrder
from ..model import mirt
from ..model.person_encoder import item_params_from_fit
from ..model.sequential_posterior import (
    INFO_GAIN_DEFINITION,
    SequentialPosterior,
    solve,
)
from .gate3 import calibrated_covariance
from .recovery import (
    DIMENSIONS,
    load_planted_theta,
    procrustes_rotation,
    rotate_covariances,
)

SCHEMA = "glassbox.stage5.strategies/1"

# ---- the frozen bar (PREREGISTRATION.md section 6, Gate 5) ----------------

#: The target: the Gate 3 accuracy level, per dimension, on held-out personas.
BAR_RMSE = 0.5

#: The adaptive interviewer must need at most this fraction of random's median.
BAR_RATIO = 0.5

#: How the frozen wording is read, quoted verbatim into the artifact.
GRADER_READING = (
    "GRADER'S READING of the frozen wording 'median questions-to-target', both "
    "lenses preserved (house precedent: the Gate 2 weak-item ordering check and "
    "the Gate 3 monotonicity call). PRIMARY, and the one graded: population-curve "
    "crossing -- for each replicate, the first N at which the held-out "
    "population's per-dimension RMSE is <= the target on all eight dimensions; "
    "the median of that N across replicates; the bar is met when the adaptive "
    "median is <= 50% of the random median. SECONDARY, reported alongside and "
    "never graded: per-persona crossing -- for each persona in each replicate, "
    "the first N at which that persona's own absolute error is <= the target on "
    "all eight dimensions. If a strategy never reaches the target inside the "
    "question budget the bar is reported as unmet or undefined (owner RULING 1), "
    "not reinterpreted."
)

# ---- pre-declared exploratory (owner RULING 1) ---------------------------

EXPLORATORY_LABEL = (
    "EXPLORATORY -- pre-declared in results/PROJECT_LOG.md, entry 'Stage 5 "
    "opened', before any Stage 5 result existed. Not a frozen bar."
)

#: Reachable thresholds for the exploratory questions-to-target grid.
EXPLORATORY_THRESHOLDS = (0.60, 0.65, 0.70)

#: Where the fraction-of-full-matrix measure is read off.
FRACTION_GRID = (5, 10, 15, 25)

FRACTION_DEFINITION = (
    "fraction_of_full_matrix(N) = (RMSE_0 - RMSE_N) / (RMSE_0 - RMSE_full), on "
    "POOLED RMSE over the eight dimensions and the 100 held-out personas, after "
    "the same Procrustes alignment. RMSE_0 is the prior-mean estimate (no "
    "questions asked, theta_hat = 0). RMSE_full is the value the SAME machinery "
    "reaches from the full 202-item closed training bank on the same episode's "
    "answers -- i.e. every candidate item asked in that same sitting -- averaged "
    "over replicates. Values above 1 mean the strategy beat the full bank on "
    "that replicate's answers, which is possible because a partial matrix can "
    "shrink less toward the prior."
)

COLD_START_LABEL = (
    "EXPLORATORY, owner-approved -- the product cold start. The confirmatory "
    "arms use the Stage 2 FITTED loadings and thresholds of the closed training "
    "items (owner ruling addendum: fitted parameters are the confirmatory path, "
    "so the Gate 5 target stays comparable with the Gate 3 accuracy level). This "
    "arm instead uses the Stage 3 item encoder's PREDICTED loadings for the same "
    "items, produced by exact leave-one-out so that no item's own fitted loading "
    "helped predict it. Thresholds and binary difficulties are kept from the fit: "
    "the item encoder has no threshold head, and Stage 4 already established "
    "fitting thresholds from training answers as legal. The episodes are the same "
    "sittings as the first replicates of the confirmatory arms -- same round tags, "
    "same answers -- so the comparison is paired and the only thing that changes "
    "is what the system believes the items measure."
)

DEFAULT_N_MAX = 150
DEFAULT_REPLICATES = 51
DEFAULT_COLD_REPLICATES = 11
DEFAULT_OUT_PREFIX = "stage5"

#: The confirmatory strategies, and which of them is "the adaptive interviewer".
STRATEGIES = ("random", "fixed", "heuristic")
ADAPTIVE = "heuristic"
BASELINE = "random"


# --------------------------------------------------------------------------
# item parameters: fitted (confirmatory) and predicted (cold-start arm)
# --------------------------------------------------------------------------


def cold_start_params(
    fit: Any,
    judgments_path: Path,
    embeddings_path: Path | None,
    *,
    embed_components: int = 64,
) -> tuple[mirt.ItemParams, dict[str, Any]]:
    """Training-item parameters with the item encoder's loadings in place of the fit's.

    Rebuilds the Stage 3 item encoder's two primary heads on the 202 training
    items and reads off their **exact leave-one-out** predictions, so the
    loading vector used for an item was produced by a model that never saw that
    item's fitted loading. Direction and magnitude are predicted separately and
    multiplied, exactly as :func:`src.model.item_encoder.predict_loadings` does.
    """
    from ..model import item_encoder as ie

    item_ids = [str(x) for x in fit["item_train_ids"]]
    fitted_loadings = np.asarray(fit["train_loadings"], dtype=float)
    discrimination = np.asarray(fit["train_discrimination"], dtype=float)

    items, _ = ie.load_bank()
    items_by_id = {item["item_id"]: item for item in items}
    parsed = ie.load_judgments(Path(judgments_path))
    means, spreads = ie.judgment_features(parsed, item_ids)
    polar = ie.polarization_features(parsed, item_ids)
    wording = np.vstack([ie.wording_features(items_by_id[i]) for i in item_ids])

    embed = None
    if embeddings_path is not None:
        emb = np.load(embeddings_path, allow_pickle=False)
        emb_ids = [str(x) for x in emb["item_ids"]]
        lookup = {iid: row for row, iid in enumerate(emb_ids)}
        raw = np.vstack([emb["embeddings"][lookup[i]] for i in item_ids])
        embed, _ = ie.reduce_embeddings(raw, raw, int(embed_components))

    x_direction = np.hstack([means, spreads])
    magnitude_blocks = [means, spreads, polar, wording]
    if embed is not None:
        magnitude_blocks.append(embed)
    x_magnitude = np.hstack(magnitude_blocks)

    lam_dir, _ = ie.select_lambda_direction(x_direction, fitted_loadings)
    log_a = np.log(np.maximum(discrimination, 1e-6))
    lam_mag, _ = ie.select_lambda_scalar(x_magnitude, log_a)

    direction = ie.unit_rows(ie.ridge_loo(x_direction, fitted_loadings, lam_dir))
    norms = np.exp(ie.ridge_loo(x_magnitude, log_a[:, None], lam_mag)[:, 0])
    predicted = direction * norms[:, None]

    fitted_params, _ = item_params_from_fit(fit, block="train")
    params = mirt.ItemParams(
        loadings=predicted,
        cut_raw=fitted_params.cut_raw.copy(),
        kinds=list(fitted_params.kinds),
    )
    cosines = ie.row_cosines(predicted, fitted_loadings)
    diagnostics = {
        "label": COLD_START_LABEL,
        "n_items": len(item_ids),
        "lambda_direction": float(lam_dir),
        "lambda_magnitude": float(lam_mag),
        "embed_components": int(embed_components) if embed is not None else 0,
        "loo_median_cosine_vs_fitted": float(np.median(cosines)),
        "loo_mean_cosine_vs_fitted": float(np.mean(cosines)),
        "loo_discrimination_r": float(ie.pearson(norms, discrimination)),
        "loo_discrimination_r_log": float(ie.pearson(np.log(norms), log_a)),
        "thresholds": "kept from the Stage 2 fit (the encoder has no threshold head)",
    }
    return params, diagnostics


# --------------------------------------------------------------------------
# one episode, start to finish -- this is what a worker process runs
# --------------------------------------------------------------------------


def run_episode(task: dict[str, Any]) -> dict[str, Any]:
    """Simulate one episode and return its truth-side curves.

    Module-level and plain-dict in, plain-dict out, so it survives being sent to
    a worker process. Everything truth-side happens here rather than in
    ``src/interview``: the simulator hands over the posterior, and this function
    -- grader-side code -- is what compares it with planted theta.
    """
    params = mirt.ItemParams(
        loadings=np.asarray(task["loadings"], dtype=float),
        cut_raw=np.asarray(task["cut_raw"], dtype=float),
        kinds=list(task["kinds"]),
    )
    codes = np.asarray(task["codes"], dtype=np.int16)
    n_personas, n_items = codes.shape
    n_train = int(task["n_train"])
    theta_true_train = np.asarray(task["theta_true_train"], dtype=float)
    theta_true_holdout = np.asarray(task["theta_true_holdout"], dtype=float)
    kappa = np.asarray(task["kappa"], dtype=float)
    n_steps = int(task["n_steps"])
    dims = theta_true_holdout.shape[1]
    n_holdout = theta_true_holdout.shape[0]

    bank = AnswerBank(codes, task["round"])
    posterior = SequentialPosterior(params, n_personas, hessian=task["hessian"])

    name = str(task["strategy"])
    if name == "random":
        strategy = RandomOrder(n_personas, n_items, int(task["seed"]))
    elif name == "heuristic":
        strategy = InfoGain(n_personas, params)
    elif name == "fixed":
        strategy = FixedOrder(n_personas, n_items, task["scripted_columns"])
    else:  # pragma: no cover -- guarded by the caller
        raise ValueError(f"unknown strategy {name!r}")

    rmse = np.zeros((n_steps, dims))
    pooled = np.zeros(n_steps)
    blur = np.zeros((n_steps, dims))
    blur_calibrated = np.zeros((n_steps, dims))
    max_error = np.zeros((n_steps, n_holdout), dtype=np.float32)
    iterations = np.zeros(n_steps, dtype=np.int32)

    def score(theta: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, ...]:
        rotation = procrustes_rotation(theta[:n_train], theta_true_train)
        estimate = theta[n_train:] @ rotation
        error = estimate - theta_true_holdout
        rotated = rotate_covariances(cov, rotation)
        raw_blur = np.sqrt(np.maximum(np.diagonal(rotated, axis1=1, axis2=2), 0.0))
        rotated_cal = rotate_covariances(calibrated_covariance(cov, kappa), rotation)
        cal_blur = np.sqrt(np.maximum(np.diagonal(rotated_cal, axis1=1, axis2=2), 0.0))
        return error, raw_blur, cal_blur

    def on_step(step: int, state: SequentialPosterior) -> None:
        k = step - 1
        error, raw_blur, cal_blur = score(state.theta, state.cov[n_train:])
        rmse[k] = np.sqrt(np.mean(error**2, axis=0))
        pooled[k] = float(np.sqrt(np.mean(error**2)))
        blur[k] = raw_blur.mean(axis=0)
        blur_calibrated[k] = cal_blur.mean(axis=0)
        max_error[k] = np.abs(error).max(axis=1)
        iterations[k] = state.last_iters

    result = EpisodeSimulator(bank, posterior, strategy).run(n_steps, on_step=on_step)
    used = result.n_steps

    # the full-bank reference for the fraction measure: the same sitting's
    # answers, every candidate item asked.
    full = solve(bank.full_matrix_for_scoring(), params, hessian=task["hessian"])
    full_error, _, _ = score(full.theta, full.cov[n_train:])

    return {
        "strategy": name,
        "arm": str(task["arm"]),
        "replicate": int(task["replicate"]),
        "round": str(task["round"]),
        "n_steps": used,
        "capped_at": result.capped_at,
        "rmse_by_n": rmse[:used],
        "pooled_by_n": pooled[:used],
        "blur_by_n": blur[:used],
        "blur_calibrated_by_n": blur_calibrated[:used],
        "max_error_by_n": max_error[:used],
        "newton_iters": iterations[:used],
        "rmse_full_per_dim": np.sqrt(np.mean(full_error**2, axis=0)),
        "pooled_full": float(np.sqrt(np.mean(full_error**2))),
        "n_distinct_items": int(len(np.unique(result.chosen))),
    }


# --------------------------------------------------------------------------
# crossings
# --------------------------------------------------------------------------


def crossing(curve: np.ndarray, threshold: float) -> int | None:
    """First N (1-based) at which every column of ``curve`` is at or below ``threshold``."""
    curve = np.asarray(curve, dtype=float)
    ok = np.all(curve <= threshold, axis=1)
    hits = np.flatnonzero(ok)
    return int(hits[0] + 1) if hits.size else None


def censored_median(values: Sequence[int | None], budget: int) -> dict[str, Any]:
    """Median of a set of crossings, honest about the ones that never crossed.

    A replicate that never reached the target inside the budget is not dropped
    (that would flatter the median) and is not given a made-up number: it enters
    the median as "larger than the budget", and if the median itself lands in
    that region the median is reported as undefined with a lower bound.
    """
    reached = [v for v in values if v is not None]
    padded = [float(v) if v is not None else float(budget) + 1.0 for v in values]
    median = float(np.median(padded)) if padded else float("nan")
    censored = bool(median > budget)
    return {
        "n_replicates": len(values),
        "n_reached": len(reached),
        "fraction_reached": float(len(reached) / len(values)) if values else float("nan"),
        "median": None if censored else median,
        "median_is_censored": censored,
        "median_lower_bound": budget if censored else None,
        "min": int(min(reached)) if reached else None,
        "max": int(max(reached)) if reached else None,
        "q25": float(np.percentile(reached, 25)) if reached else None,
        "q75": float(np.percentile(reached, 75)) if reached else None,
        "budget": int(budget),
    }


def ratio_comparison(
    adaptive: dict[str, Any], baseline: dict[str, Any], *, bar: float = BAR_RATIO
) -> dict[str, Any]:
    """The bar's own arithmetic: adaptive median as a fraction of random's."""
    a, b = adaptive.get("median"), baseline.get("median")
    if a is None or b is None or not b:
        return {
            "ratio": None,
            "bar": bar,
            "verdict": "UNDEFINED",
            "why": (
                "at least one median is censored -- that strategy never reached "
                "the target inside the question budget, so the comparison the "
                "bar asks for does not exist (owner RULING 1)"
            ),
            "adaptive_median": a,
            "random_median": b,
        }
    ratio = float(a / b)
    return {
        "ratio": ratio,
        "bar": bar,
        "verdict": "PASS" if ratio <= bar else "FAIL",
        "adaptive_median": a,
        "random_median": b,
    }


def per_persona_crossings(
    max_error: np.ndarray, threshold: float, budget: int
) -> dict[str, Any]:
    """The secondary lens: when each individual persona's own error clears the target."""
    max_error = np.asarray(max_error, dtype=float)  # (replicates, steps, personas)
    ok = max_error <= threshold
    first = np.where(ok.any(axis=1), ok.argmax(axis=1) + 1, 0)
    reached = first[first > 0].astype(float)
    return {
        "threshold": float(threshold),
        "n_persona_episodes": int(first.size),
        "n_reached": int(reached.size),
        "fraction_reached": float(reached.size / first.size) if first.size else float("nan"),
        "median_of_reached": float(np.median(reached)) if reached.size else None,
        "q25_of_reached": float(np.percentile(reached, 25)) if reached.size else None,
        "q75_of_reached": float(np.percentile(reached, 75)) if reached.size else None,
        "budget": int(budget),
        "note": (
            "median is over the persona-episodes that reached the target; the "
            "fraction that never did is reported next to it and is not imputed"
        ),
    }


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------


def stack_curves(results: Sequence[dict[str, Any]], key: str) -> np.ndarray:
    """Stack one curve across replicates, ``(replicates, steps, ...)``."""
    return np.stack([np.asarray(r[key], dtype=float) for r in results])


def summarize_arm(
    results: Sequence[dict[str, Any]], *, budget: int, theta_true_holdout: np.ndarray
) -> dict[str, Any]:
    """Everything one strategy's replicates say, curves and crossings."""
    rmse = stack_curves(results, "rmse_by_n")  # (R, N, dims)
    pooled = stack_curves(results, "pooled_by_n")  # (R, N)
    blur = stack_curves(results, "blur_by_n")
    blur_cal = stack_curves(results, "blur_calibrated_by_n")
    max_error = stack_curves(results, "max_error_by_n")
    steps = rmse.shape[1]
    grid = list(range(1, steps + 1))
    # a strategy that ran out of questions is censored at its own last question,
    # not at the budget it never had: the Stage 3 script holds 15 items.
    budget = min(int(budget), steps)

    rmse0 = float(np.sqrt(np.mean(theta_true_holdout**2)))
    rmse0_per_dim = np.sqrt(np.mean(theta_true_holdout**2, axis=0))
    pooled_full = float(np.mean([r["pooled_full"] for r in results]))

    thresholds: dict[str, Any] = {}
    for value in (BAR_RMSE, *EXPLORATORY_THRESHOLDS):
        crossings = [crossing(r["rmse_by_n"], value) for r in results]
        thresholds[f"{value:.2f}"] = {
            **censored_median(crossings, budget),
            "per_replicate": crossings,
            "per_persona": per_persona_crossings(max_error, value, budget),
        }

    fraction = {}
    denominator = rmse0 - pooled_full
    mean_pooled = pooled.mean(axis=0)
    for n in FRACTION_GRID:
        if n > steps:
            fraction[str(n)] = None
            continue
        fraction[str(n)] = (
            float((rmse0 - mean_pooled[n - 1]) / denominator)
            if abs(denominator) > 1e-12
            else None
        )

    return {
        "n_replicates": len(results),
        "n_steps": steps,
        "capped_at": results[0]["capped_at"],
        "n_grid": grid,
        "rmse_per_dim_mean_by_n": rmse.mean(axis=0).tolist(),
        "rmse_worst_dim_mean_by_n": rmse.mean(axis=0).max(axis=1).tolist(),
        "pooled_rmse_mean_by_n": mean_pooled.tolist(),
        "pooled_rmse_q25_by_n": np.percentile(pooled, 25, axis=0).tolist(),
        "pooled_rmse_q75_by_n": np.percentile(pooled, 75, axis=0).tolist(),
        "declared_blur_mean_by_n": blur.mean(axis=0).mean(axis=1).tolist(),
        "declared_blur_per_dim_mean_by_n": blur.mean(axis=0).tolist(),
        "declared_blur_calibrated_mean_by_n": blur_cal.mean(axis=0).mean(axis=1).tolist(),
        "rmse_at_n0": rmse0,
        "rmse_at_n0_per_dim": rmse0_per_dim.tolist(),
        "pooled_rmse_full_bank": pooled_full,
        "rmse_full_bank_per_dim": np.mean(
            [r["rmse_full_per_dim"] for r in results], axis=0
        ).tolist(),
        "questions_to_target": thresholds,
        "fraction_of_full_matrix": fraction,
        "mean_distinct_items_used": float(
            np.mean([r["n_distinct_items"] for r in results])
        ),
        "mean_newton_iters": float(
            np.mean([np.mean(r["newton_iters"]) for r in results])
        ),
    }


def proxy_gaming_watch(arms: dict[str, Any]) -> dict[str, Any]:
    """Does the declared blur shrink faster than the truth-side error does?

    Owner RULING 2 asks for this explicitly and calls a divergence a first-class
    finding rather than a bug. The number reported is the ratio of what each
    strategy has shrunk by N: declared blur relative to its own start, against
    truth RMSE relative to its own start. A strategy whose blur falls much
    faster than its error is declaring confidence it has not earned.
    """
    watch = {}
    for name, arm in arms.items():
        blur = np.asarray(arm["declared_blur_mean_by_n"], dtype=float)
        rmse = np.asarray(arm["pooled_rmse_mean_by_n"], dtype=float)
        rmse0 = float(arm["rmse_at_n0"])
        blur0 = 1.0  # the prior blur is 1 by construction: theta ~ N(0, I)
        blur_share = blur / blur0
        rmse_share = rmse / rmse0
        rows = []
        for k, n in enumerate(arm["n_grid"]):
            rows.append(
                {
                    "n": int(n),
                    "declared_blur": float(blur[k]),
                    "truth_rmse": float(rmse[k]),
                    "blur_remaining": float(blur_share[k]),
                    "error_remaining": float(rmse_share[k]),
                    "gap": float(rmse_share[k] - blur_share[k]),
                }
            )
        watch[name] = {
            "prior_blur": blur0,
            "rmse_at_n0": rmse0,
            "by_n": rows,
            "largest_gap": max(rows, key=lambda r: r["gap"]),
            "note": (
                "gap > 0 means the system has declared away more of its "
                "uncertainty than it has actually removed from its error"
            ),
        }
    return watch


# --------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------

_COLOURS = {
    "random": "#4C72B0",
    "fixed": "#8172B2",
    "heuristic": "#C44E52",
    "cold_start_random": "#7BA3D0",
    "cold_start_heuristic": "#DD8452",
}


def draw_rmse_vs_n(report: dict[str, Any], path: Path) -> None:
    """Truth-side error against interview length, with the targets drawn on."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = report["curves"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    ax = axes[0]
    for name in ("random", "fixed", "heuristic"):
        arm = arms.get(name)
        if arm is None:
            continue
        grid = arm["n_grid"]
        worst = arm["rmse_worst_dim_mean_by_n"]
        ax.plot(grid, worst, color=_COLOURS[name], lw=1.8, label=name)
    for value, style in ((BAR_RMSE, "-"), *[(t, ":") for t in EXPLORATORY_THRESHOLDS]):
        ax.axhline(value, color="#555555", ls=style, lw=1.0)
        ax.text(
            len(arms["random"]["n_grid"]) * 0.99, value + 0.005, f"{value:.2f}",
            fontsize=7, ha="right", color="#555555",
        )
    ax.set_xlabel("N closed questions asked")
    ax.set_ylabel("worst per-dimension RMSE vs planted theta")
    ax.set_title("worst dimension: what the frozen bar is read on")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1]
    for name, arm in arms.items():
        grid = arm["n_grid"]
        ax.plot(
            grid, arm["pooled_rmse_mean_by_n"],
            color=_COLOURS.get(name, "#888888"),
            ls="--" if name.startswith("cold_start") else "-",
            lw=1.6, label=name,
        )
        if not name.startswith("cold_start"):
            ax.fill_between(
                grid, arm["pooled_rmse_q25_by_n"], arm["pooled_rmse_q75_by_n"],
                color=_COLOURS.get(name, "#888888"), alpha=0.15, lw=0,
            )
    full = arms["random"]["pooled_rmse_full_bank"]
    ax.axhline(full, color="#333333", lw=1.0, ls="-.")
    ax.text(
        len(arms["random"]["n_grid"]) * 0.99, full + 0.005,
        f"full 202-item bank {full:.3f}", fontsize=7, ha="right", color="#333333",
    )
    ax.set_xlabel("N closed questions asked")
    ax.set_ylabel("pooled RMSE vs planted theta")
    ax.set_title("pooled error, mean over replicates (band: inter-quartile)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    fig.suptitle(
        "Gate 5: truth-side error against interview length, by question-picking strategy",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def draw_blur_vs_truth(report: dict[str, Any], path: Path) -> None:
    """The proxy-gaming watch: declared confidence against actual error."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = report["curves"]
    watch = report["exploratory"]["blur_vs_truth"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    ax = axes[0]
    for name in ("random", "fixed", "heuristic"):
        arm = arms.get(name)
        if arm is None:
            continue
        grid = arm["n_grid"]
        ax.plot(grid, arm["declared_blur_mean_by_n"], color=_COLOURS[name], lw=1.6,
                label=f"{name}: declared blur")
        ax.plot(grid, arm["pooled_rmse_mean_by_n"], color=_COLOURS[name], lw=1.6,
                ls="--", label=f"{name}: truth RMSE")
    ax.set_xlabel("N closed questions asked")
    ax.set_ylabel("mean declared blur / pooled truth RMSE")
    ax.set_title("what the system says vs what it does")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)

    ax = axes[1]
    for name in ("random", "fixed", "heuristic"):
        rows = watch.get(name)
        if rows is None:
            continue
        grid = [row["n"] for row in rows["by_n"]]
        ax.plot(grid, [row["blur_remaining"] for row in rows["by_n"]],
                color=_COLOURS[name], lw=1.6, label=f"{name}: blur left")
        ax.plot(grid, [row["error_remaining"] for row in rows["by_n"]],
                color=_COLOURS[name], lw=1.6, ls="--", label=f"{name}: error left")
    ax.axhline(1.0, color="#999999", lw=0.8)
    ax.set_xlabel("N closed questions asked")
    ax.set_ylabel("share of the starting value still there")
    ax.set_title("proxy-gaming watch (RULING 2): blur falling faster than error")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)

    fig.suptitle(
        "Gate 5: declared uncertainty against truth-side error", fontsize=12
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------


def build_tasks(
    *,
    episodes: Any,
    params: mirt.ItemParams,
    cold_params: mirt.ItemParams | None,
    scripted_columns: Sequence[int],
    theta_true_train: np.ndarray,
    theta_true_holdout: np.ndarray,
    kappa: np.ndarray,
    n_train: int,
    n_max: int,
    replicates: int,
    cold_replicates: int,
    hessian: str,
) -> list[dict[str, Any]]:
    """One task per episode: the answers, the parameters, and nothing else."""
    codes = episodes["codes"]
    rounds = [str(r) for r in episodes["rounds"]]
    index = {name: k for k, name in enumerate(rounds)}

    tasks: list[dict[str, Any]] = []

    def add(arm: str, strategy: str, replicate: int, item_params: mirt.ItemParams) -> None:
        tag = round_tag(strategy, replicate)
        if tag not in index:
            raise ValueError(f"the episode cache has no round {tag!r}")
        tasks.append(
            {
                "arm": arm,
                "strategy": strategy,
                "replicate": replicate,
                "round": tag,
                "codes": codes[index[tag]],
                "loadings": item_params.loadings,
                "cut_raw": item_params.cut_raw,
                "kinds": list(item_params.kinds),
                "scripted_columns": list(scripted_columns),
                "seed": episode_seed(strategy, replicate),
                "theta_true_train": theta_true_train,
                "theta_true_holdout": theta_true_holdout,
                "kappa": kappa,
                "n_train": n_train,
                "n_steps": n_max,
                "hessian": hessian,
            }
        )

    for strategy in STRATEGIES:
        for replicate in range(replicates):
            add(strategy, strategy, replicate, params)
    if cold_params is not None:
        for strategy in ("random", "heuristic"):
            for replicate in range(cold_replicates):
                add(f"cold_start_{strategy}", strategy, replicate, cold_params)
    return tasks


def execute(tasks: Sequence[dict[str, Any]], workers: int, *, verbose: bool = True) -> list[dict[str, Any]]:
    """Run the episodes, in parallel when there is more than one core to use."""
    if workers <= 1:
        out = []
        for k, task in enumerate(tasks, start=1):
            out.append(run_episode(task))
            if verbose:
                print(f"  [{k}/{len(tasks)}] {task['arm']} {task['round']}")
        return out

    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    # one BLAS thread per worker: the arrays here are small and oversubscribing
    # the cores makes the whole sweep slower, not faster.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(name, "1")

    results: list[dict[str, Any]] = []
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        for k, result in enumerate(pool.map(run_episode, list(tasks)), start=1):
            results.append(result)
            if verbose and (k % 10 == 0 or k == len(tasks)):
                print(f"  {k}/{len(tasks)} episodes done")
    return results


def grade(
    episodes_path: Path,
    fit_path: Path,
    backbone_path: Path,
    splits_path: Path,
    truth_dir: Path,
    out_dir: Path,
    *,
    out_prefix: str = DEFAULT_OUT_PREFIX,
    n_max: int = DEFAULT_N_MAX,
    replicates: int = DEFAULT_REPLICATES,
    cold_replicates: int = DEFAULT_COLD_REPLICATES,
    judgments_path: Path | None = None,
    embeddings_path: Path | None = None,
    hessian: str = "analytic",
    workers: int = 1,
    make_plots: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    started = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes = np.load(episodes_path, allow_pickle=False)
    fit = np.load(fit_path, allow_pickle=False)
    backbone = np.load(backbone_path, allow_pickle=False)
    splits = json.loads(Path(splits_path).read_text(encoding="utf-8"))

    params, item_ids = item_params_from_fit(fit, block="train")
    column_of = {iid: j for j, iid in enumerate(item_ids)}
    cached_items = [str(i) for i in episodes["item_ids"]]
    if cached_items != item_ids:
        raise ValueError("the episode cache was drawn on a different candidate bank")

    train_ids = [str(p) for p in fit["persona_train_ids"]]
    holdout_ids = [str(p) for p in fit["persona_holdout_ids"]]
    cached_pids = [str(p) for p in episodes["persona_ids"]]
    if cached_pids != train_ids + holdout_ids:
        raise ValueError("the episode cache holds different personas than the fit")

    scripted = [str(i) for i in splits.get("interview_closed") or []]
    missing = [i for i in scripted if i not in column_of]
    if missing:
        raise ValueError(f"scripted items absent from the training bank: {missing}")
    scripted_columns = [column_of[i] for i in scripted]

    theta_true_train = load_planted_theta(truth_dir, train_ids)
    theta_true_holdout = load_planted_theta(truth_dir, holdout_ids)
    kappa = np.asarray(backbone["blur_kappa"], dtype=float)

    cold_params = None
    cold_diagnostics: dict[str, Any] | None = None
    if judgments_path is not None:
        cold_params, cold_diagnostics = cold_start_params(
            fit, Path(judgments_path), embeddings_path
        )

    tasks = build_tasks(
        episodes=episodes,
        params=params,
        cold_params=cold_params,
        scripted_columns=scripted_columns,
        theta_true_train=theta_true_train,
        theta_true_holdout=theta_true_holdout,
        kappa=kappa,
        n_train=len(train_ids),
        n_max=n_max,
        replicates=replicates,
        cold_replicates=cold_replicates,
        hessian=hessian,
    )
    if verbose:
        print(
            f"{len(tasks)} episodes: {len(train_ids)} training + "
            f"{len(holdout_ids)} held-out personas, {len(item_ids)} candidate "
            f"items, up to N = {n_max}, on {workers} worker(s)"
        )
    results = execute(tasks, workers, verbose=verbose)

    by_arm: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        by_arm.setdefault(result["arm"], []).append(result)
    for arm in by_arm.values():
        arm.sort(key=lambda r: r["replicate"])

    curves = {
        arm: summarize_arm(rows, budget=n_max, theta_true_holdout=theta_true_holdout)
        for arm, rows in by_arm.items()
    }

    # ---- the frozen bar --------------------------------------------------
    key = f"{BAR_RMSE:.2f}"
    frozen = {
        "target": {
            "definition": (
                "per-dimension RMSE(theta_hat, theta_true) <= 0.5 on all eight "
                "dimensions, held-out personas -- the Gate 3 accuracy level, "
                "PREREGISTRATION section 6"
            ),
            "value": BAR_RMSE,
        },
        "grader_reading": GRADER_READING,
        "question_budget": n_max,
        "population_curve_crossing": {
            arm: {k: v for k, v in curves[arm]["questions_to_target"][key].items()
                  if k != "per_persona"}
            for arm in STRATEGIES
        },
        "secondary_lens_per_persona_crossing": {
            arm: curves[arm]["questions_to_target"][key]["per_persona"]
            for arm in STRATEGIES
        },
        "comparison": {
            "heuristic_vs_random": ratio_comparison(
                curves[ADAPTIVE]["questions_to_target"][key],
                curves[BASELINE]["questions_to_target"][key],
            ),
            "fixed_vs_random": ratio_comparison(
                curves["fixed"]["questions_to_target"][key],
                curves[BASELINE]["questions_to_target"][key],
            ),
        },
        "ruling_1": (
            "owner RULING 1, recorded before any Stage 5 compute: if no strategy "
            "reaches the target within the budget, the bar is unmet or undefined "
            "and is reported exactly as written, not reinterpreted"
        ),
    }
    frozen["verdict"] = frozen["comparison"]["heuristic_vs_random"]["verdict"]

    # ---- pre-declared exploratory ---------------------------------------
    exploratory: dict[str, Any] = {
        "label": EXPLORATORY_LABEL,
        "threshold_grid": {
            f"{value:.2f}": {
                "per_strategy": {
                    arm: {k: v for k, v in curves[arm]["questions_to_target"][f"{value:.2f}"].items()
                          if k != "per_persona"}
                    for arm in curves
                },
                "per_persona": {
                    arm: curves[arm]["questions_to_target"][f"{value:.2f}"]["per_persona"]
                    for arm in curves
                },
                "heuristic_vs_random": ratio_comparison(
                    curves[ADAPTIVE]["questions_to_target"][f"{value:.2f}"],
                    curves[BASELINE]["questions_to_target"][f"{value:.2f}"],
                ),
                "fixed_vs_random": ratio_comparison(
                    curves["fixed"]["questions_to_target"][f"{value:.2f}"],
                    curves[BASELINE]["questions_to_target"][f"{value:.2f}"],
                ),
            }
            for value in EXPLORATORY_THRESHOLDS
        },
        "fraction_of_full_matrix": {
            "definition": FRACTION_DEFINITION,
            "grid": list(FRACTION_GRID),
            "per_strategy": {arm: curves[arm]["fraction_of_full_matrix"] for arm in curves},
            "full_bank_pooled_rmse": {
                arm: curves[arm]["pooled_rmse_full_bank"] for arm in curves
            },
            "rmse_at_n0": curves[BASELINE]["rmse_at_n0"],
        },
        "blur_vs_truth": proxy_gaming_watch(curves),
    }
    if cold_diagnostics is not None:
        exploratory["cold_start"] = {
            "label": COLD_START_LABEL,
            "item_encoder": cold_diagnostics,
            "n_replicates": cold_replicates,
            "pooled_rmse_at_n": {
                arm: {
                    str(n): curves[arm]["pooled_rmse_mean_by_n"][n - 1]
                    for n in FRACTION_GRID
                    if n <= curves[arm]["n_steps"]
                }
                for arm in curves
                if arm.startswith("cold_start")
            },
            "cost_vs_fitted": {
                strategy: {
                    str(n): (
                        curves[f"cold_start_{strategy}"]["pooled_rmse_mean_by_n"][n - 1]
                        - curves[strategy]["pooled_rmse_mean_by_n"][n - 1]
                    )
                    for n in FRACTION_GRID
                }
                for strategy in ("random", "heuristic")
                if f"cold_start_{strategy}" in curves
            },
            "questions_to_target_0.70": {
                arm: {k: v for k, v in curves[arm]["questions_to_target"]["0.70"].items()
                      if k != "per_persona"}
                for arm in curves
                if arm.startswith("cold_start")
            },
        }

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {
            "episodes": str(episodes_path),
            "fit": str(fit_path),
            "backbone": str(backbone_path),
            "splits": str(splits_path),
            "judgments": str(judgments_path) if judgments_path else None,
            "embeddings": str(embeddings_path) if embeddings_path else None,
        },
        "design": {
            "n_replicates": replicates,
            "n_cold_start_replicates": cold_replicates if cold_params is not None else 0,
            "question_budget": n_max,
            "n_candidate_items": len(item_ids),
            "n_personas_training": len(train_ids),
            "n_personas_holdout": len(holdout_ids),
            "scripted_order_length": len(scripted_columns),
            "item_parameters": (
                "Stage 2 FITTED loadings and thresholds of the 202 closed "
                "training items (owner ruling addendum), the same parameters "
                "the Stage 3 person-encoder backbone used"
            ),
            "episode_answers": (
                "each answer is a fresh, independently seeded application of the "
                "frozen noise layer (a=1.2, b=4.0, T_noise=16, seed base 548) to "
                "the persona's recorded answer distribution from the v2 sweep, "
                "under round tag ep_<strategy>_<replicate>; the responder is "
                "never re-queried and no episode can leave the persona's own "
                "answer distribution"
            ),
            "alignment": (
                "orthogonal Procrustes fitted on the 400 training personas' own "
                "estimate at that N and applied unchanged to the held-out "
                "personas and their covariances -- the same convention and the "
                "same function as Gate 2 and Gate 3 "
                "(src.eval.recovery.procrustes_rotation)"
            ),
            "info_gain_definition": INFO_GAIN_DEFINITION,
            "random_order": (
                "a fresh seeded permutation of the 202 candidate items per "
                "persona per replicate, so the baseline is not hostage to one "
                "lucky order"
            ),
            "fixed_order": (
                "the frozen Stage 3 scripted order; it holds 15 items, so this "
                "strategy simply stops at 15 -- the cap truncates the curve and "
                "is reported, never extended"
            ),
            "blur_calibration": (
                "the Stage 3 backbone's per-dimension constants, read and applied "
                "unchanged, never refitted here; raw blur is the primary number "
                "and the calibrated one is reported beside it"
            ),
            "hessian": hessian,
            "strategies_evaluated": sorted(curves),
        },
        "frozen_bar": frozen,
        "exploratory": exploratory,
        "curves": curves,
    }

    # ---- artifacts -------------------------------------------------------
    json_path = out_dir / f"{out_prefix}_strategies.json"
    npz_path = out_dir / f"{out_prefix}_strategies.npz"
    payload: dict[str, np.ndarray] = {}
    for arm, rows in by_arm.items():
        payload[f"{arm}_rmse_by_n"] = stack_curves(rows, "rmse_by_n")
        payload[f"{arm}_pooled_by_n"] = stack_curves(rows, "pooled_by_n")
        payload[f"{arm}_blur_by_n"] = stack_curves(rows, "blur_by_n")
        payload[f"{arm}_blur_calibrated_by_n"] = stack_curves(rows, "blur_calibrated_by_n")
        payload[f"{arm}_max_error_by_n"] = stack_curves(rows, "max_error_by_n").astype(np.float32)
        payload[f"{arm}_pooled_full"] = np.array([r["pooled_full"] for r in rows])
    payload["dimensions"] = np.array(list(DIMENSIONS))
    payload["persona_holdout_ids"] = np.array(holdout_ids)
    payload["item_train_ids"] = np.array(item_ids)
    np.savez_compressed(npz_path, **payload)

    plots: list[str] = []
    if make_plots:
        rmse_plot = out_dir / f"{out_prefix}_rmse_vs_n.png"
        blur_plot = out_dir / f"{out_prefix}_blur_vs_truth.png"
        draw_rmse_vs_n(report, rmse_plot)
        draw_blur_vs_truth(report, blur_plot)
        plots = [str(rmse_plot), str(blur_plot)]

    report["costs"] = {
        "wall_seconds": round(time.time() - started, 1),
        "n_episodes": len(tasks),
        "workers": workers,
        "gpu": "none",
        "api": "none",
    }
    report["artifacts"] = {"json": str(json_path), "npz": str(npz_path), "plots": plots}
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


# --------------------------------------------------------------------------
# text summary
# --------------------------------------------------------------------------


def render(report: dict[str, Any]) -> str:
    lines: list[str] = []
    add = lines.append
    design = report["design"]
    add("GATE 5 -- adaptive interviewing, non-RL strategies")
    add(
        f"  {design['n_replicates']} replicates x {design['n_personas_holdout']} "
        f"held-out personas, budget N = {design['question_budget']}, "
        f"{design['n_candidate_items']} candidate items"
    )
    add("")
    add("FROZEN BAR (a) -- questions to per-dimension RMSE <= 0.5 on all 8 dims")
    for arm, block in report["frozen_bar"]["population_curve_crossing"].items():
        if arm.startswith("cold_start"):
            continue
        if block["median"] is None:
            add(
                f"  {arm:<10} never reached inside N <= {block['budget']} "
                f"({block['n_reached']}/{block['n_replicates']} replicates reached)"
            )
        else:
            add(
                f"  {arm:<10} median {block['median']:.0f} questions "
                f"({block['n_reached']}/{block['n_replicates']} replicates reached)"
            )
    comparison = report["frozen_bar"]["comparison"]["heuristic_vs_random"]
    add(f"  heuristic vs random: {comparison['verdict']}"
        + (f" (ratio {comparison['ratio']:.3f}, bar <= {comparison['bar']})"
           if comparison["ratio"] is not None else f" -- {comparison['why']}"))
    add("")
    add("EXPLORATORY threshold grid (pre-declared, labeled)")
    for value, block in report["exploratory"]["threshold_grid"].items():
        row = [f"  RMSE <= {value}:"]
        for arm in ("random", "fixed", "heuristic"):
            stats = block["per_strategy"].get(arm)
            if stats is None:
                continue
            median = stats["median"]
            row.append(f"{arm} " + (f"{median:.0f}" if median is not None else "never"))
        ratio = block["heuristic_vs_random"]
        row.append(
            f"| ratio " + (f"{ratio['ratio']:.3f} {ratio['verdict']}"
                           if ratio["ratio"] is not None else "undefined")
        )
        add("  ".join(row))
    add("")
    add("EXPLORATORY fraction of full-matrix information")
    fraction = report["exploratory"]["fraction_of_full_matrix"]
    for arm, values in fraction["per_strategy"].items():
        parts = [
            f"N={n}: {values[str(n)]:.3f}" if values.get(str(n)) is not None else f"N={n}: -"
            for n in FRACTION_GRID
        ]
        add(f"  {arm:<22} " + "  ".join(parts))
    add("")
    add("CURVES (mean pooled RMSE)")
    for arm, block in report["curves"].items():
        grid = block["n_grid"]
        picks = [n for n in (1, 5, 10, 15, 25, 50, 100, 150) if n <= len(grid)]
        parts = [f"N={n}: {block['pooled_rmse_mean_by_n'][n - 1]:.3f}" for n in picks]
        add(f"  {arm:<22} " + "  ".join(parts))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.eval.gate5",
        description="Grade the Stage 5 question-picking strategies against the frozen bar.",
        allow_abbrev=False,
    )
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--out-prefix", default=DEFAULT_OUT_PREFIX)
    parser.add_argument("--n-max", type=int, default=DEFAULT_N_MAX)
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--cold-replicates", type=int, default=DEFAULT_COLD_REPLICATES)
    parser.add_argument("--judgments", type=Path, default=None)
    parser.add_argument("--embeddings", type=Path, default=None)
    parser.add_argument("--hessian", default="analytic", choices=("analytic", "central"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    report = grade(
        args.episodes,
        args.fit,
        args.backbone,
        args.splits,
        args.truth,
        args.out,
        out_prefix=args.out_prefix,
        n_max=args.n_max,
        replicates=args.replicates,
        cold_replicates=args.cold_replicates,
        judgments_path=args.judgments,
        embeddings_path=args.embeddings,
        hessian=args.hessian,
        workers=args.workers,
    )
    print(render(report))
    print()
    for name, path in report["artifacts"].items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
