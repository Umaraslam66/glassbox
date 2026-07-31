"""Gate 3 person-encoder grader (PREREGISTRATION.md sections 5-6).

    python -m src.eval.gate3 \
        --fused results/stage3_person_fused.json \
        --backbone results/stage3_person_backbone.npz \
        --fit results/stage2_v2_fit.npz \
        --truth <planted dir> \
        --splits experiments/splits_v1.json \
        --out results

Grader-side code: it lives in ``src/eval/``, so it may read the planted design.
The directory holding that design is a command-line argument; no path to it is
spelled out here.

What it grades
--------------
The four Gate 3 person-encoder bars, all on the 100 held-out personas, at
N = 15, for the **fused** track (closed answers + the three open answers -- the
confirmatory arm) with the **closed-only** track reported beside it:

1. per-dimension ``RMSE(theta_hat, theta_true) <= 0.5`` on all eight dimensions;
2. pooled coverage of the calibrated +/-1 blur, band 60-75%;
3. blur shrinks monotonically in expectation as N grows -- reported in **both**
   readings the backbone artifact documents, strict per-step and mean-curve,
   with the size of every uptick. This grader does not choose between them;
4. RMSE improvement over the public-profile-only baseline >= 30%.

Alignment -- identical to Stage 2's
-----------------------------------
The fitted axes are arbitrary up to an orthogonal rotation. The rotation is
found on the TRAINING personas only, by
:func:`src.eval.recovery.procrustes_rotation` -- the same function Gate 2 used,
imported rather than restated -- and applied unchanged to the held-out person
vectors and to their posterior covariances. One rotation per track per N, each
fitted on that track's own training estimate at that N, so every graded number
is aligned by a map that never saw a held-out persona.

Scale linking: none is applied here, and none is needed. The Stage 2 fit pinned
the latent metric once (:func:`src.model.mirt.link_scale`) and stored the
loadings **after** linking; the person encoder reads those linked loadings under
the same N(0, I) prior, so its theta is already on the axis and in the unit that
``theta_train`` / ``theta_holdout`` live in -- the same quantities Gate 2 graded
with a rotation and nothing else. This grader therefore does exactly what
``src.eval.recovery.grade`` does: rotate, then measure. The shrinkage of the MAP
toward the prior is reported as a diagnostic and is never corrected, because
correcting it would mean fitting a scale against planted truth.

WALL. Nothing in this file is written back to the system side. No number
produced here is used to calibrate anything: the blur constants were fitted on
system-side training residuals before this ran, and are read, not re-fitted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .recovery import (
    DIMENSIONS,
    describe,
    load_planted_theta,
    pearson,
    procrustes_rotation,
    rotate_covariances,
)

# ---- the frozen bars (PREREGISTRATION.md section 6, Gate 3) ---------------
BAR_RMSE = 0.5  # every dimension, held-out personas, N = 15
BAR_COVERAGE = (0.60, 0.75)  # calibrated +/-1 blur
BAR_LIFT = 0.30  # over the public-profile-only baseline

#: The confirmatory interview length (PREREGISTRATION.md section 5).
CONFIRMATORY_N = 15

#: The confirmatory track. The other one is reported beside it.
CONFIRMATORY_TRACK = "fused"

DEFAULT_OUT_PREFIX = "stage3"


# --------------------------------------------------------------------------
# the pre-filed projection, quoted so the outcome sits next to the prediction
# --------------------------------------------------------------------------

#: Verbatim from results/PROJECT_LOG.md, entry "Stage 3 design decided;
#: pre-measurement projection filed" -- written before any Stage 3 measurement.
PROJECTION_QUOTE = (
    "PROJECTION, FILED BEFORE ANY STAGE 3 MEASUREMENT (a prediction, not a "
    "result): from Stage 2's own numbers, the Gate 3 person-encoder RMSE bar "
    "(<= 0.5 per dimension at N=15) is projected unreachable -- the full "
    "252-item fit already sits at RMSE 0.458-0.580 with 5/8 dimensions above "
    "0.50, the transmission ceiling puts 4/8 dimensions above the bar even "
    "with infinite interview data, and Fisher-information plus a direct "
    "15-item MAP scoring of held-out personas (two independent methods that "
    "agree) put N=15 at RMSE 0.70-0.81 on every dimension. Projected lift "
    "from closed items alone: ~23% vs the 30% bar (the open answers are the "
    "only legal lever); projected coverage ~0.60 -- near the band floor, with "
    "the anticipation-note blind spot expected to surface as N grows rather "
    "than at N=15."
)

#: The per-dimension numbers behind the "0.70-0.81 on every dimension" range,
#: from the filed design proposal's table. Closed items only, N = 15.
PROJECTED_RMSE_CLOSED_ONLY = {
    "TRU": 0.762, "RSK": 0.752, "ENV": 0.718, "PRC": 0.712,
    "TRD": 0.797, "SOC": 0.742, "TEC": 0.697, "LOC": 0.807,
}

PROJECTED_LIFT_CLOSED_ONLY = 0.230
PROJECTED_COVERAGE = 0.598

#: Recorded in the backbone commit: the projection's agreement figure shared
#: answer noise with its own reference, so it was optimistic.
INDEPENDENT_SITTING_NOTE = (
    "the projection chained through r(theta_hat_15, theta_hat_full) = 0.758, "
    "which was measured by re-scoring the SAME noised answers the full-matrix "
    "fit had already seen. The interview is an independent sitting of the "
    "frozen noise layer, so its answers carry fresh noise; the backbone "
    "measured 0.686 instead. Numbers below are from the independent sitting."
)

#: When a confirmatory number counts as materially beating its projection, and
#: therefore triggers a documented leakage hunt before it is reported as a pass.
HUNT_RMSE_MARGIN = 0.10  # per dimension, below the projected value
HUNT_LIFT_MARGIN = 0.10  # pooled lift, above the projected value

#: What Stage 2's disattenuated recovery says the personas' own answers carry
#: about planted theta, per dimension. A system-side text feature that read
#: planted truth more sharply than this would not be reading the answers.
CEILING_BAND = (0.82, 0.92)
CEILING_ALARM = 0.92


# --------------------------------------------------------------------------
# loading the tracks
# --------------------------------------------------------------------------


def load_tracks(
    fused_json: Path, fused_npz: Path | None = None
) -> tuple[dict[str, Any], Any]:
    """The fusion report and its estimate archive."""
    report = json.loads(Path(fused_json).read_text(encoding="utf-8"))
    if fused_npz is None:
        fused_npz = Path(report["artifacts"]["estimates"])
    return report, np.load(Path(fused_npz), allow_pickle=False)


def calibrated_covariance(cov: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    """Inflate a posterior covariance by the per-dimension blur constants.

    The calibration is one scalar per dimension in the fitted basis. Applying it
    to the covariance as ``K Sigma K`` reproduces the artifact's calibrated blur
    exactly on the diagonal, and is the only form that survives a change of
    basis -- which matters here, because the grader rotates afterwards.
    """
    k = np.diag(np.asarray(kappa, dtype=float))
    return np.einsum("ij,njk,kl->nil", k, cov, k)


def aligned_track(
    archive: Any,
    track: str,
    n: int,
    theta_true_train: np.ndarray,
    *,
    calibrate: bool = True,
) -> dict[str, Any]:
    """Rotate one track at one N onto the planted axes, Stage 2's way.

    The rotation is fitted on the training personas' own estimate at this N and
    applied unchanged to the held-out personas, their covariances and the blur.
    """
    theta_train = np.asarray(archive[f"theta_train_{track}_by_n"][n - 1], dtype=float)
    theta_holdout = np.asarray(archive[f"theta_holdout_{track}_by_n"][n - 1], dtype=float)
    cov_holdout = np.asarray(archive[f"cov_holdout_{track}_by_n"][n - 1], dtype=float)
    kappa = np.asarray(archive[f"blur_kappa_{track}"], dtype=float)

    rotation = procrustes_rotation(theta_train, theta_true_train)
    cov = calibrated_covariance(cov_holdout, kappa) if calibrate else cov_holdout
    rotated_cov = rotate_covariances(cov, rotation)
    blur = np.sqrt(np.maximum(np.diagonal(rotated_cov, axis1=1, axis2=2), 0.0))
    return {
        "rotation": rotation,
        "theta_train": theta_train @ rotation,
        "theta": theta_holdout @ rotation,
        "blur": blur,
        "kappa": kappa,
    }


# --------------------------------------------------------------------------
# the bars
# --------------------------------------------------------------------------


def rmse_bar(theta_hat: np.ndarray, theta_true: np.ndarray) -> dict[str, Any]:
    """Bar (a): per-dimension RMSE <= 0.5 on every dimension."""
    per_dim = {
        dim: float(np.sqrt(np.mean((theta_hat[:, d] - theta_true[:, d]) ** 2)))
        for d, dim in enumerate(DIMENSIONS)
    }
    per_dim_r = {
        dim: pearson(theta_hat[:, d], theta_true[:, d]) for d, dim in enumerate(DIMENSIONS)
    }
    values = list(per_dim.values())
    worst = max(per_dim, key=lambda k: per_dim[k])
    return {
        "per_dimension_rmse": per_dim,
        "per_dimension_r": per_dim_r,
        "worst_dimension": worst,
        "worst_rmse": float(max(values)),
        "best_rmse": float(min(values)),
        "pooled_rmse": float(np.sqrt(np.mean((theta_hat - theta_true) ** 2))),
        "mean_of_per_dimension_rmse": float(np.mean(values)),
        "n_dimensions_passing": int(sum(1 for v in values if v <= BAR_RMSE)),
        "n_personas": int(theta_hat.shape[0]),
        "bar": BAR_RMSE,
        "pass": bool(max(values) <= BAR_RMSE),
    }


def coverage_bar(
    theta_hat: np.ndarray, blur: np.ndarray, theta_true: np.ndarray
) -> dict[str, Any]:
    """Bar (b): planted truth inside the calibrated +/-1 blur, 60-75% of cells.

    The diagnostics block is the Gate 2 one, reported the same way: how much
    wider the interval would have to be to reach the nominal 68 percent. It
    sizes the miss. Nothing here is fed back into the blur -- calibrating any
    system-side uncertainty against planted theta is the Wall breach the
    standing rule names.
    """
    inside = np.abs(theta_hat - theta_true) <= blur
    pooled = float(inside.mean())
    error = theta_hat - theta_true
    ratios = np.abs(error) / np.maximum(blur, 1e-12)
    low, high = BAR_COVERAGE
    return {
        "pooled_coverage": pooled,
        "per_dimension_coverage": {
            dim: float(inside[:, d].mean()) for d, dim in enumerate(DIMENSIONS)
        },
        "n_cells": int(inside.size),
        "mean_blur": float(blur.mean()),
        "mean_blur_per_dimension": {
            dim: float(blur[:, d].mean()) for d, dim in enumerate(DIMENSIONS)
        },
        "diagnostics": {
            "mean_predicted_variance": float(np.mean(blur**2)),
            "mean_actual_squared_error": float(np.mean(error**2)),
            "variance_ratio_actual_over_predicted": float(
                np.mean(error**2) / np.mean(blur**2)
            ),
            "blur_multiplier_for_nominal_68pct": float(np.percentile(ratios, 68)),
            "per_dimension_variance_ratio": {
                dim: float(np.mean(error[:, d] ** 2) / np.mean(blur[:, d] ** 2))
                for d, dim in enumerate(DIMENSIONS)
            },
            "note": (
                "a ratio above 1 means the posterior is more confident than it "
                "has earned; the multiplier is how much wider the interval "
                "would have to be to reach the nominal 68 percent. Reported to "
                "size the miss -- the blur itself is left exactly as calibrated "
                "on system-side training residuals."
            ),
        },
        "band": list(BAR_COVERAGE),
        "pass": bool(low <= pooled <= high),
    }


def lift_bar(
    theta_hat: np.ndarray, theta_baseline: np.ndarray, theta_true: np.ndarray
) -> dict[str, Any]:
    """Bar (d): ``1 - RMSE_encoder / RMSE_baseline >= 0.30``.

    The baseline is the public-profile-only ridge, which cross-validation
    collapsed to intercept-only: it predicts the training mean for everybody.
    Its RMSE against planted theta is therefore the population spread, which
    makes this bar and the PRD's zero-information baseline numerically the same
    bar. That is stated in the output rather than left implicit.
    """
    enc = {
        dim: float(np.sqrt(np.mean((theta_hat[:, d] - theta_true[:, d]) ** 2)))
        for d, dim in enumerate(DIMENSIONS)
    }
    base = {
        dim: float(np.sqrt(np.mean((theta_baseline[:, d] - theta_true[:, d]) ** 2)))
        for d, dim in enumerate(DIMENSIONS)
    }
    per_dim = {dim: 1.0 - enc[dim] / base[dim] for dim in DIMENSIONS}
    pooled_enc = float(np.sqrt(np.mean((theta_hat - theta_true) ** 2)))
    pooled_base = float(np.sqrt(np.mean((theta_baseline - theta_true) ** 2)))
    pooled = 1.0 - pooled_enc / pooled_base
    population_sd = {
        dim: float(np.sqrt(np.mean(theta_true[:, d] ** 2))) for d, dim in enumerate(DIMENSIONS)
    }
    return {
        "definition": "1 - RMSE(encoder) / RMSE(public-profile-only baseline)",
        "pooled_lift": pooled,
        "mean_of_per_dimension_lift": float(np.mean(list(per_dim.values()))),
        "per_dimension_lift": per_dim,
        "encoder_rmse_per_dimension": enc,
        "baseline_rmse_per_dimension": base,
        "pooled_encoder_rmse": pooled_enc,
        "pooled_baseline_rmse": pooled_base,
        "population_rms_per_dimension": population_sd,
        "baseline_is_intercept_only": True,
        "baseline_note": (
            "the public-profile ridge selected the intercept-only model in "
            "cross-validation, so it predicts the training mean for every "
            "persona and its RMSE is the population spread in planted units "
            "(compare baseline_rmse_per_dimension with "
            "population_rms_per_dimension). The Gate 3 lift bar and the PRD's "
            "zero-information baseline are the same bar here."
        ),
        "bar": BAR_LIFT,
        "pass": bool(pooled >= BAR_LIFT),
    }


def monotonicity_readings(report: dict[str, Any], track: str) -> dict[str, Any]:
    """Both readings of the shrinkage bar, presented, not adjudicated.

    PREREGISTRATION says "blur shrinks monotonically **in expectation** as N
    grows". Two things can be measured, and the backbone artifact records both:

    * the **mean-curve** reading -- the blur averaged over held-out personas,
      per dimension, at zero tolerance. This is the literal reading of "in
      expectation";
    * the **strict per-step** reading -- every persona, every dimension, every
      step of N, counted individually.

    Their uptick sizes are what settle the question, so they are reported next
    to the shrinkage they sit on, together with the fixed-theta control where
    the information matrices nest exactly and monotonicity is a theorem.
    """
    block = report["monotonicity"][track]
    mean_curve = block["holdout"]
    per_dim = mean_curve["per_dim"]
    total_shrinkage = [row["total_shrinkage"] for row in per_dim]
    largest = mean_curve["largest_increase_any_dim"]
    strict = mean_curve.get("persona_level", {})
    fixed = report["monotonicity"].get(track, {}).get("holdout_at_fixed_theta")
    return {
        "bar_wording": (
            "PREREGISTRATION Gate 3: blur shrinks monotonically in expectation "
            "as N grows (plot required)."
        ),
        "mean_curve_reading": {
            "what": "blur averaged over the 100 held-out personas, per dimension, zero tolerance",
            "verdict": mean_curve["verdict"],
            "n_dimensions_monotone": mean_curve["n_dims_monotone"],
            "largest_uptick": largest,
            "largest_uptick_as_fraction_of_smallest_total_shrinkage": (
                float(largest / min(total_shrinkage)) if min(total_shrinkage) > 0 else None
            ),
            "total_shrinkage_per_dimension": {
                DIMENSIONS[row["dim"]] if row["dim"] < len(DIMENSIONS) else str(row["dim"]):
                row["total_shrinkage"]
                for row in per_dim
            },
            "upticks_per_dimension": {
                DIMENSIONS[row["dim"]] if row["dim"] < len(DIMENSIONS) else str(row["dim"]): {
                    "n_upticks": row["n_violations"],
                    "largest": row["largest_increase"],
                    "as_fraction_of_total_shrinkage": row[
                        "largest_increase_as_fraction_of_total_shrinkage"
                    ],
                    "steps": row["violations"],
                }
                for row in per_dim
            },
            "axis_note": (
                "dimension labels here are the FITTED axes of the encoder, not "
                "the planted ones -- blur is a system-side quantity and the "
                "monotonicity bar is about it directly, before any rotation"
            ),
        },
        "strict_per_step_reading": {
            "what": "every persona x dimension x step of N counted individually",
            **strict,
            "verdict": (
                "PASS" if strict and strict.get("n_increases", 1) == 0 else "FAIL"
            ),
        },
        "fixed_theta_control": fixed
        and {
            "what": (
                "the same sweep with theta pinned at its N=15 value, so the "
                "information matrices nest exactly and monotonicity is a theorem"
            ),
            "verdict": fixed["verdict"],
            "largest_uptick": fixed["largest_increase_any_dim"],
        },
        "reading": report["monotonicity"]["reading"],
        "verdict": None,
        "note": (
            "this grader does not choose between the two readings. Both are "
            "reported with their uptick sizes; the call is the orchestrator's."
        ),
    }


# --------------------------------------------------------------------------
# curves
# --------------------------------------------------------------------------


def sweep_curves(
    archive: Any,
    track: str,
    theta_true_train: np.ndarray,
    theta_true_holdout: np.ndarray,
    theta_baseline: np.ndarray,
) -> list[dict[str, Any]]:
    """RMSE, coverage, lift and blur at every N on the graded grid."""
    n_grid = [int(v) for v in archive["n_grid"]]
    rows = []
    for n in n_grid:
        piece = aligned_track(archive, track, n, theta_true_train)
        theta = piece["theta"]
        blur = piece["blur"]
        rmse = np.sqrt(np.mean((theta - theta_true_holdout) ** 2, axis=0))
        base = np.sqrt(np.mean((theta_baseline @ piece["rotation"] - theta_true_holdout) ** 2))
        pooled = float(np.sqrt(np.mean((theta - theta_true_holdout) ** 2)))
        inside = np.abs(theta - theta_true_holdout) <= blur
        rows.append(
            {
                "n": n,
                "pooled_rmse": pooled,
                "rmse_per_dimension": {
                    dim: float(rmse[d]) for d, dim in enumerate(DIMENSIONS)
                },
                "worst_dimension_rmse": float(rmse.max()),
                "pooled_coverage": float(inside.mean()),
                "mean_calibrated_blur": float(blur.mean()),
                "pooled_lift": float(1.0 - pooled / base),
                "mean_r": float(
                    np.mean(
                        [pearson(theta[:, d], theta_true_holdout[:, d]) for d in range(len(DIMENSIONS))]
                    )
                ),
            }
        )
    return rows


def full_matrix_reference(
    fit: Any, theta_true_train: np.ndarray, theta_true_holdout: np.ndarray
) -> dict[str, Any]:
    """The Stage 2 full-matrix estimate, graded against planted truth again.

    This is the transcript-vs-full-matrix gap in the units the gate cares about:
    what 252 answered items recover, next to what a 15-question interview plus
    three open answers recovers.
    """
    rotation = procrustes_rotation(np.asarray(fit["theta_train"]), theta_true_train)
    theta = np.asarray(fit["theta_holdout"]) @ rotation
    rmse = {
        dim: float(np.sqrt(np.mean((theta[:, d] - theta_true_holdout[:, d]) ** 2)))
        for d, dim in enumerate(DIMENSIONS)
    }
    r = {dim: pearson(theta[:, d], theta_true_holdout[:, d]) for d, dim in enumerate(DIMENSIONS)}
    return {
        "what": "Stage 2 fit, all 202 training items answered, held-out personas",
        "per_dimension_rmse": rmse,
        "per_dimension_r": r,
        "pooled_rmse": float(np.sqrt(np.mean((theta - theta_true_holdout) ** 2))),
        "mean_r": float(np.mean(list(r.values()))),
        "n_items": int(len(fit["item_train_ids"])),
    }


def shrinkage_diagnostic(
    theta_hat: np.ndarray, theta_true: np.ndarray
) -> dict[str, Any]:
    """How much of the RMSE is the MAP pulling toward the prior mean.

    EXPLORATORY, and never applied. A posterior mode under an N(0, I) prior is
    shrunk toward zero, which costs RMSE even when the ranking is perfect. The
    numbers below say how much; rescaling to remove it would mean fitting a
    scale against planted truth, which the standing rule forbids.
    """
    out = {}
    for d, dim in enumerate(DIMENSIONS):
        x, y = theta_hat[:, d], theta_true[:, d]
        slope = float(np.sum(x * y) / max(np.sum(x * x), 1e-12))
        rescaled = float(np.sqrt(np.mean((slope * x - y) ** 2)))
        out[dim] = {
            "sd_estimate": float(x.std()),
            "sd_planted": float(y.std()),
            "regression_of_planted_on_estimate": slope,
            "rmse_as_measured": float(np.sqrt(np.mean((x - y) ** 2))),
            "rmse_if_optimally_rescaled": rescaled,
        }
    return {
        "label": "EXPLORATORY -- reported, never applied",
        "per_dimension": out,
        "note": (
            "the optimal rescaling is fitted against planted theta, so using it "
            "anywhere would be a Wall breach. It is here only to show how much "
            "of the measured RMSE is prior shrinkage rather than lost signal."
        ),
    }


# --------------------------------------------------------------------------
# leakage hunt
# --------------------------------------------------------------------------


def leakage_hunt(
    fused_report: dict[str, Any],
    archive: Any,
    backbone_archive: Any,
    features_path: Path | None,
    fit: Any,
    splits: dict[str, Any],
    theta_true_train: np.ndarray,
    theta_true_holdout: np.ndarray,
    bars: dict[str, Any],
    *,
    closed_only_rmse: dict[str, float] | None = None,
    closed_only_lift: float | None = None,
) -> dict[str, Any]:
    """Split checks, feature provenance and B-fit scope, run every time.

    The standing rule is that a result which beats its projection gets a
    documented hunt before it is reported. The trigger is computed here, and the
    checks run whether or not it fires, because the fused channel is new
    machinery and the cost of looking is a second.
    """
    train_ids = [str(p) for p in archive["persona_train_ids"]]
    holdout_ids = [str(p) for p in archive["persona_holdout_ids"]]
    fit_train = [str(p) for p in fit["persona_train_ids"]]
    fit_holdout = [str(p) for p in fit["persona_holdout_ids"]]
    split_holdout = [str(p) for p in splits.get("persona_holdout") or []]

    splits_block = {
        "n_train": len(train_ids),
        "n_holdout": len(holdout_ids),
        "train_holdout_overlap": sorted(set(train_ids) & set(holdout_ids)),
        "matches_the_fit_archive": bool(
            train_ids == fit_train and holdout_ids == fit_holdout
        ),
        "matches_the_frozen_split_manifest": bool(
            not split_holdout or set(holdout_ids) == set(split_holdout)
        ),
        "verdict": None,
    }
    splits_block["verdict"] = (
        "clean"
        if not splits_block["train_holdout_overlap"]
        and splits_block["matches_the_fit_archive"]
        and splits_block["matches_the_frozen_split_manifest"]
        else "SPLIT VIOLATION"
    )

    model = fused_report["measurement_model"]
    selection = fused_report["model_selection"]
    scope_block = {
        "fitted_on_n_personas": model["fitted_on_n_personas"],
        "fitted_against": model["fitted_against"],
        "expected_n_personas": len(train_ids),
        "cross_validation_folds": selection["folds"],
        "cross_validation_seed": selection["seed"],
        "psi_source": selection["psi_source"],
        "selection_saw_holdout": False,
        "verdict": (
            "clean -- B, c, Psi and both hyper-parameters come from the "
            "training personas only, against a system-side target"
            if model["fitted_on_n_personas"] == len(train_ids)
            else "SCOPE VIOLATION: the measurement model was not fitted on the training split alone"
        ),
    }

    # -- feature provenance: three questions of the judgments. Do they know
    #    more about planted theta on held-out personas than on training ones
    #    (a split leak)? Do they read planted theta more sharply than the
    #    persona's own answers ever carried it (a truth leak)? And do they
    #    track the fitted estimate, which is all the rest of the pipeline sees?
    provenance: dict[str, Any] = {"available": features_path is not None}
    if features_path is not None:
        from ..model.open_fusion import load_open_features

        feats = load_open_features(features_path)
        poles = feats.poles
        raw_train = feats.rows_for(train_ids)
        raw_holdout = feats.rows_for(holdout_ids)
        n_prompts = len(feats.prompt_ids)
        rotation = procrustes_rotation(np.asarray(fit["theta_train"]), theta_true_train)
        fitted_train = np.asarray(fit["theta_train"]) @ rotation
        fitted_holdout = np.asarray(fit["theta_holdout"]) @ rotation
        per_pole = {}
        for p, pole in enumerate(poles):
            cols = [q * len(poles) + p for q in range(n_prompts)]
            mean_train = raw_train[:, cols].mean(axis=1)
            mean_holdout = raw_holdout[:, cols].mean(axis=1)
            d = DIMENSIONS.index(pole) if pole in DIMENSIONS else p
            per_pole[pole] = {
                "r_with_planted_train": pearson(mean_train, theta_true_train[:, d]),
                "r_with_planted_holdout": pearson(mean_holdout, theta_true_holdout[:, d]),
                "r_with_full_matrix_estimate_train": pearson(mean_train, fitted_train[:, d]),
                "r_with_full_matrix_estimate_holdout": pearson(
                    mean_holdout, fitted_holdout[:, d]
                ),
            }
        gaps = [
            v["r_with_planted_holdout"] - v["r_with_planted_train"] for v in per_pole.values()
        ]
        planted_r = [abs(v["r_with_planted_holdout"]) for v in per_pole.values()]
        flags = []
        if max(abs(g) for g in gaps) >= 0.25:
            flags.append(
                "a pole reads planted truth materially better on held-out "
                "personas than on training ones"
            )
        if max(planted_r) > CEILING_ALARM:
            flags.append(
                f"a pole reads planted truth at r > {CEILING_ALARM}, above the "
                "transmission ceiling the persona's own answers carry"
            )
        provenance.update(
            {
                "what": (
                    "each pole's judgment averaged over the three open answers, "
                    "correlated with the planted value of that dimension and "
                    "with the full-matrix estimate on the same axes"
                ),
                "per_pole": per_pole,
                "holdout_minus_train_gap": describe(gaps),
                "largest_r_with_planted": float(max(planted_r)),
                "transmission_ceiling_for_reference": {
                    "value": list(CEILING_BAND),
                    "source": (
                        "Stage 2's disattenuated recovery: what an infinitely "
                        "long interview could read out of these personas' "
                        "answers. A text feature that beat it would be reading "
                        "something the answers do not contain."
                    ),
                    "alarm_at": CEILING_ALARM,
                },
                "verdict": (
                    "clean -- the judgments track planted truth about equally "
                    "on both splits and stay under the transmission ceiling, "
                    "which is what a reader with no access to planted truth "
                    "looks like"
                    if not flags
                    else "LOOK AGAIN: " + "; ".join(flags)
                ),
                "note": (
                    "the raw feature axes are the published pole names, so the "
                    "planted correlation is meaningful without any rotation; the "
                    "full-matrix estimate is rotated onto those axes by Stage "
                    "2's own convention. Positive correlation is expected and is "
                    "the point of the channel -- what would be alarming is a "
                    "SPLIT-DEPENDENT difference or a correlation above the "
                    "ceiling."
                ),
            }
        )

    # -- the closed-only arm against its own filed projection. If the backbone
    #    matches what was predicted and only the fused arm beats it, the gain
    #    came from the channel that was built to produce it.
    closed_rmse = closed_only_rmse or {}
    backbone_block = {
        "what": (
            "the closed-only arm is the thing the projection was actually about; "
            "if it lands where it was predicted, the fused arm's improvement is "
            "the open-answer channel doing its declared job"
        ),
        "per_dimension_outcome_minus_projection": {
            dim: closed_rmse[dim] - PROJECTED_RMSE_CLOSED_ONLY[dim]
            for dim in DIMENSIONS
            if dim in closed_rmse
        },
        "closed_only_pooled_lift": closed_only_lift,
        "projected_pooled_lift": PROJECTED_LIFT_CLOSED_ONLY,
    }
    deltas = list(backbone_block["per_dimension_outcome_minus_projection"].values())
    backbone_block["verdict"] = (
        "clean -- the closed-only arm sits within "
        f"{max(abs(v) for v in deltas):.3f} of its filed per-dimension projection "
        "on every dimension"
        if deltas and max(abs(v) for v in deltas) <= HUNT_RMSE_MARGIN
        else "LOOK AGAIN: the closed-only backbone itself departs from its projection"
    )

    # -- integrity: the fusion must not have moved the backbone it sits on.
    integrity = {"what": "the closed-only track, recomputed inside the fusion run, against the backbone artifact"}
    try:
        integrity["theta_holdout_identical"] = bool(
            np.array_equal(
                np.asarray(backbone_archive["theta_holdout_by_n"]),
                np.asarray(archive["theta_holdout_closed_only_by_n"]),
            )
        )
        integrity["blur_holdout_identical"] = bool(
            np.array_equal(
                np.asarray(backbone_archive["blur_holdout_by_n"]),
                np.asarray(archive["blur_holdout_closed_only_by_n"]),
            )
        )
        integrity["kappa_identical"] = bool(
            np.array_equal(
                np.asarray(backbone_archive["blur_kappa"]),
                np.asarray(archive["blur_kappa_closed_only"]),
            )
        )
        integrity["verdict"] = (
            "clean -- bit-identical"
            if all(v is True for k, v in integrity.items() if k.endswith("identical"))
            else "LOOK AGAIN: the closed-only track moved when the block was added"
        )
    except KeyError as exc:  # pragma: no cover - defensive
        integrity["verdict"] = f"could not check: {exc}"

    # -- the trigger --------------------------------------------------------
    rmse = bars["rmse"]["per_dimension_rmse"]
    beats_rmse = {
        dim: PROJECTED_RMSE_CLOSED_ONLY[dim] - rmse[dim]
        for dim in DIMENSIONS
        if PROJECTED_RMSE_CLOSED_ONLY[dim] - rmse[dim] > HUNT_RMSE_MARGIN
    }
    lift_excess = bars["lift"]["pooled_lift"] - PROJECTED_LIFT_CLOSED_ONLY
    fired = bool(beats_rmse) or lift_excess > HUNT_LIFT_MARGIN

    return {
        "trigger": {
            "rule": (
                f"a hunt is documented when a per-dimension RMSE lands more than "
                f"{HUNT_RMSE_MARGIN} below its filed projection, or the pooled "
                f"lift lands more than {HUNT_LIFT_MARGIN} above the filed "
                f"projection of {PROJECTED_LIFT_CLOSED_ONLY}"
            ),
            "dimensions_beating_projection_materially": beats_rmse,
            "pooled_lift_minus_projection": lift_excess,
            "fired": fired,
            "note": (
                "the filed projection covers the closed-only arm, so the fused "
                "arm is compared against it as the nearest filed prediction; "
                "the fused arm was expected to be better, not equal"
            ),
        },
        "checks_run_regardless": True,
        "splits": splits_block,
        "measurement_model_scope": scope_block,
        "feature_provenance": provenance,
        "closed_only_against_its_projection": backbone_block,
        "backbone_integrity": integrity,
        "verdict": (
            "no leakage found"
            if splits_block["verdict"] == "clean"
            and "VIOLATION" not in scope_block["verdict"]
            and provenance.get("verdict", "clean").startswith("clean")
            and backbone_block["verdict"].startswith("clean")
            and integrity["verdict"].startswith("clean")
            else "FOLLOW UP -- see the blocks above"
        ),
    }


# --------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------


def draw_blur_vs_n(report: dict[str, Any], path: Path) -> None:
    """The bar's required picture: blur against N, both tracks."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_n = report["blur_vs_n"]
    grid = [row["n"] for row in per_n]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    ax = axes[0]
    for track, colour in (("closed_only", "#4C72B0"), ("fused", "#C44E52")):
        ax.plot(
            grid, [row[track]["mean_blur"] for row in per_n],
            marker="o", ms=4, color=colour, label=f"{track.replace('_', '-')}, raw",
        )
        ax.plot(
            grid, [row[track]["mean_blur_calibrated"] for row in per_n],
            marker="s", ms=4, ls="--", color=colour,
            label=f"{track.replace('_', '-')}, calibrated",
        )
    ax.set_xlabel("N closed questions answered")
    ax.set_ylabel("mean posterior blur (held-out personas)")
    ax.set_title("blur against interview length")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1]
    for track, colour in (("closed_only", "#4C72B0"), ("fused", "#C44E52")):
        matrix = np.array([row[track]["mean_blur_per_dim"] for row in per_n])
        for d in range(matrix.shape[1]):
            ax.plot(grid, matrix[:, d], color=colour, alpha=0.55, lw=1.1)
        ax.plot([], [], color=colour, label=track.replace("_", "-"))
    ax.set_xlabel("N closed questions answered")
    ax.set_ylabel("mean blur, one line per fitted dimension")
    ax.set_title("per-dimension blur (fitted axes)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    fig.suptitle(
        "Gate 3: posterior blur against interview length, with and without the three open answers",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def draw_gate3_curves(report: dict[str, Any], path: Path) -> None:
    """RMSE, coverage and lift against N, with the bars drawn on."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = report["curves"]
    grid = [row["n"] for row in curves["fused"]]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.0))
    colours = {"closed_only": "#4C72B0", "fused": "#C44E52"}

    ax = axes[0]
    for track, colour in colours.items():
        ax.plot(grid, [r["pooled_rmse"] for r in curves[track]], marker="o", ms=4,
                color=colour, label=f"{track.replace('_', '-')} (pooled)")
        ax.plot(grid, [r["worst_dimension_rmse"] for r in curves[track]], ls=":",
                color=colour, alpha=0.7, label=f"{track.replace('_', '-')} (worst dim)")
    ax.axhline(BAR_RMSE, color="k", ls="--", lw=1.2, label=f"bar {BAR_RMSE}")
    ax.set_xlabel("N closed questions")
    ax.set_ylabel("RMSE against planted theta")
    ax.set_title("RMSE against N")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.axhspan(BAR_COVERAGE[0], BAR_COVERAGE[1], color="#55A868", alpha=0.16,
               label=f"band {BAR_COVERAGE[0]:.2f}-{BAR_COVERAGE[1]:.2f}")
    for track, colour in colours.items():
        ax.plot(grid, [r["pooled_coverage"] for r in curves[track]], marker="o", ms=4,
                color=colour, label=track.replace("_", "-"))
    ax.set_xlabel("N closed questions")
    ax.set_ylabel("coverage of the calibrated +/-1 blur")
    ax.set_title("coverage against N (the design predicted decay)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[2]
    for track, colour in colours.items():
        ax.plot(grid, [r["pooled_lift"] for r in curves[track]], marker="o", ms=4,
                color=colour, label=track.replace("_", "-"))
    ax.axhline(BAR_LIFT, color="k", ls="--", lw=1.2, label=f"bar {BAR_LIFT:.0%}")
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.set_xlabel("N closed questions")
    ax.set_ylabel("lift over the public-profile-only baseline")
    ax.set_title("lift against N")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    fig.suptitle(
        "Gate 3 person encoder: 100 held-out personas, alignment fitted on the 400 training personas",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------
# the grade
# --------------------------------------------------------------------------


def grade(
    fused_json: Path,
    backbone_npz: Path,
    fit_path: Path,
    truth_dir: Path,
    splits_path: Path,
    out_dir: Path,
    *,
    out_prefix: str = DEFAULT_OUT_PREFIX,
    fused_npz: Path | None = None,
    features_path: Path | None = None,
    make_plots: bool = True,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fused_report, archive = load_tracks(Path(fused_json), fused_npz)
    backbone = np.load(Path(backbone_npz), allow_pickle=False)
    fit = np.load(Path(fit_path), allow_pickle=False)
    splits = json.loads(Path(splits_path).read_text(encoding="utf-8"))

    train_ids = [str(p) for p in archive["persona_train_ids"]]
    holdout_ids = [str(p) for p in archive["persona_holdout_ids"]]
    theta_true_train = load_planted_theta(truth_dir, train_ids)
    theta_true_holdout = load_planted_theta(truth_dir, holdout_ids)
    theta_baseline = np.asarray(backbone["theta_baseline_holdout"], dtype=float)

    tracks: dict[str, Any] = {}
    for track in ("fused", "closed_only"):
        piece = aligned_track(archive, track, CONFIRMATORY_N, theta_true_train)
        bars = {
            "rmse": rmse_bar(piece["theta"], theta_true_holdout),
            "coverage": coverage_bar(piece["theta"], piece["blur"], theta_true_holdout),
            "lift": lift_bar(
                piece["theta"],
                theta_baseline @ piece["rotation"],
                theta_true_holdout,
            ),
            "monotonicity": monotonicity_readings(fused_report, track),
        }
        bars["rmse"]["projection"] = {
            "filed": PROJECTED_RMSE_CLOSED_ONLY,
            "per_dimension_outcome_minus_projection": {
                dim: bars["rmse"]["per_dimension_rmse"][dim] - PROJECTED_RMSE_CLOSED_ONLY[dim]
                for dim in DIMENSIONS
            },
            "quote": PROJECTION_QUOTE,
            "independent_sitting_correction": INDEPENDENT_SITTING_NOTE,
            "scope": (
                "the filed per-dimension projection is for the CLOSED-ONLY arm; "
                "it is shown against both arms because it is the only filed "
                "per-dimension prediction"
            ),
        }
        bars["coverage"]["projection"] = {
            "filed_pooled_coverage": PROJECTED_COVERAGE,
            "outcome_minus_projection": bars["coverage"]["pooled_coverage"] - PROJECTED_COVERAGE,
        }
        bars["lift"]["projection"] = {
            "filed_closed_only_lift": PROJECTED_LIFT_CLOSED_ONLY,
            "outcome_minus_projection": bars["lift"]["pooled_lift"] - PROJECTED_LIFT_CLOSED_ONLY,
        }
        tracks[track] = {
            "n": CONFIRMATORY_N,
            "bars": bars,
            "alignment": {
                "method": (
                    "orthogonal Procrustes, fitted on the 400 training personas' "
                    "own estimate at this N, applied unchanged to held-out -- the "
                    "same convention and the same code as Stage 2 "
                    "(src.eval.recovery.procrustes_rotation)"
                ),
                "rotation_determinant": float(np.linalg.det(piece["rotation"])),
                "rotation": piece["rotation"].tolist(),
                "scale_linking": (
                    "none applied. The Stage 2 fit pinned the latent metric "
                    "(src.model.mirt.link_scale) and stored the loadings after "
                    "linking; the encoder reads those linked loadings under the "
                    "same N(0, I) prior, so its theta is already in the unit "
                    "Gate 2 graded. Rotation only, exactly as recovery.py does."
                ),
            },
            "prior_shrinkage_diagnostic": shrinkage_diagnostic(
                piece["theta"], theta_true_holdout
            ),
        }

    curves = {
        track: sweep_curves(
            archive, track, theta_true_train, theta_true_holdout, theta_baseline
        )
        for track in ("fused", "closed_only")
    }

    hunt = leakage_hunt(
        fused_report,
        archive,
        backbone,
        features_path,
        fit,
        splits,
        theta_true_train,
        theta_true_holdout,
        tracks[CONFIRMATORY_TRACK]["bars"],
        closed_only_rmse=tracks["closed_only"]["bars"]["rmse"]["per_dimension_rmse"],
        closed_only_lift=tracks["closed_only"]["bars"]["lift"]["pooled_lift"],
    )

    confirmatory = tracks[CONFIRMATORY_TRACK]["bars"]
    verdict = {
        "gate": "Gate 3 -- person encoder (translators)",
        "confirmatory_track": CONFIRMATORY_TRACK,
        "confirmatory_n": CONFIRMATORY_N,
        "n_holdout_personas": len(holdout_ids),
        "bars": {
            "rmse": {
                "statement": (
                    "per-dimension RMSE(theta_hat, theta_true) <= 0.5 on all 8 "
                    "dimensions, held-out personas, N = 15"
                ),
                "value": confirmatory["rmse"]["worst_rmse"],
                "value_label": f"worst dimension {confirmatory['rmse']['worst_dimension']}",
                "bar": BAR_RMSE,
                "pass": confirmatory["rmse"]["pass"],
                "n_dimensions_passing": confirmatory["rmse"]["n_dimensions_passing"],
            },
            "blur_coverage": {
                "statement": "pooled coverage of the calibrated +/-1 blur inside 60-75%",
                "value": confirmatory["coverage"]["pooled_coverage"],
                "bar": list(BAR_COVERAGE),
                "pass": confirmatory["coverage"]["pass"],
            },
            "monotone_blur": {
                "statement": "blur shrinks monotonically in expectation as N grows",
                "value": confirmatory["monotonicity"]["mean_curve_reading"]["largest_uptick"],
                "value_label": "largest uptick in the mean blur curve, any dimension",
                "bar": "two readings reported; the call is the orchestrator's",
                "pass": None,
                "mean_curve_verdict": confirmatory["monotonicity"]["mean_curve_reading"]["verdict"],
                "strict_per_step_verdict": confirmatory["monotonicity"][
                    "strict_per_step_reading"
                ]["verdict"],
            },
            "lift": {
                "statement": (
                    "RMSE improvement over the public-profile-only baseline >= 30%"
                ),
                "value": confirmatory["lift"]["pooled_lift"],
                "bar": BAR_LIFT,
                "pass": confirmatory["lift"]["pass"],
            },
        },
    }
    decided = {k: b for k, b in verdict["bars"].items() if b["pass"] is not None}
    verdict["all_decided_bars_pass"] = all(b["pass"] for b in decided.values())
    verdict["failed_bars"] = [k for k, b in decided.items() if not b["pass"]]
    verdict["undecided_bars"] = [
        k for k, b in verdict["bars"].items() if b["pass"] is None
    ]
    verdict["closed_only_for_comparison"] = {
        "rmse_worst": tracks["closed_only"]["bars"]["rmse"]["worst_rmse"],
        "rmse_pass": tracks["closed_only"]["bars"]["rmse"]["pass"],
        "coverage": tracks["closed_only"]["bars"]["coverage"]["pooled_coverage"],
        "coverage_pass": tracks["closed_only"]["bars"]["coverage"]["pass"],
        "lift": tracks["closed_only"]["bars"]["lift"]["pooled_lift"],
        "lift_pass": tracks["closed_only"]["bars"]["lift"]["pass"],
    }

    coverage_failed = not confirmatory["coverage"]["pass"]
    report: dict[str, Any] = {
        "module": "src.eval.gate3",
        "verdict": verdict,
        "tracks": tracks,
        "curves": curves,
        "blur_vs_n": [
            {
                "n": row["n"],
                "closed_only": {
                    "mean_blur": row["closed_only"]["holdout"]["mean_blur"],
                    "mean_blur_calibrated": row["closed_only"]["holdout"]["mean_blur_calibrated"],
                    "mean_blur_per_dim": row["closed_only"]["holdout"]["mean_blur_per_dim"],
                },
                "fused": {
                    "mean_blur": row["fused"]["holdout"]["mean_blur"],
                    "mean_blur_calibrated": row["fused"]["holdout"]["mean_blur_calibrated"],
                    "mean_blur_per_dim": row["fused"]["holdout"]["mean_blur_per_dim"],
                },
            }
            for row in fused_report["per_n"]
        ],
        "full_matrix_reference": full_matrix_reference(
            fit, theta_true_train, theta_true_holdout
        ),
        "transcript_vs_full_matrix": {
            "what": (
                "the same held-out personas graded against planted truth twice: "
                "from all 202 answered training items, and from the 15-question "
                "interview plus three open answers"
            ),
            "full_matrix_pooled_rmse": None,  # filled below
            "n15_fused_pooled_rmse": confirmatory["rmse"]["pooled_rmse"],
            "n15_closed_only_pooled_rmse": tracks["closed_only"]["bars"]["rmse"]["pooled_rmse"],
            "system_side_agreement_at_n15": {
                "what": (
                    "r and RMSE of the interview estimate against the "
                    "full-matrix estimate -- no planted value involved, restated "
                    "from the fusion artifact for the comparison"
                ),
                "fused": fused_report["per_n"][CONFIRMATORY_N]["fused"]["holdout"][
                    "agreement_with_full_matrix"
                ],
                "closed_only": fused_report["per_n"][CONFIRMATORY_N]["closed_only"][
                    "holdout"
                ]["agreement_with_full_matrix"],
            },
            "independent_sitting_note": INDEPENDENT_SITTING_NOTE,
        },
        "open_answers_worth_in_closed_items": fused_report[
            "open_answers_worth_in_closed_items"
        ],
        "measurement_model": fused_report["measurement_model"],
        "model_selection": {
            k: v for k, v in fused_report["model_selection"].items() if k != "cv_curve"
        },
        "blur_calibration": fused_report["blur_calibration"],
        "leakage_hunt": hunt,
        "anticipation_note": {
            "recorded": (
                "PROJECT_LOG, Gate 2 close: if Stage 3 coverage fails while RMSE "
                "passes, that is reported as the same phenomenon as Gate 2, not "
                "patched, with the same exploratory decomposition -- and no "
                "truth-anchored calibration on the system side, ever."
            ),
            "coverage_bar_failed": coverage_failed,
            "branch_taken": (
                "the coverage bar did not fail at N = 15, so the anticipation "
                "branch does not fire. The mechanism it describes is still "
                "visible in the coverage-against-N curve, which decays as the "
                "interview lengthens -- reported below, no bar attached."
                if not coverage_failed
                else "the coverage bar failed; see coverage_decay and the "
                "exploratory diagnostics in tracks.*.bars.coverage.diagnostics"
            ),
            "coverage_decay": {
                track: {
                    "at_n1": curves[track][0]["pooled_coverage"],
                    "at_n5": curves[track][4]["pooled_coverage"],
                    "at_n10": curves[track][9]["pooled_coverage"],
                    "at_n15": curves[track][14]["pooled_coverage"],
                }
                for track in ("fused", "closed_only")
            },
            "full_matrix_coverage_for_reference": (
                "Gate 2 measured 0.358 on the same personas from all 202 items -- "
                "the end point of the same decay"
            ),
            "no_recalibration": (
                "no confirmatory number in this report was recalibrated against "
                "planted truth, and no quantity computed here was written back "
                "to the system side"
            ),
        },
        "counts": {
            "train_personas": len(train_ids),
            "holdout_personas": len(holdout_ids),
            "scripted_items": int(len(archive["scripted_item_ids"])),
            "open_prompts": len(fused_report["inputs"]["open_prompts"]),
        },
        "inputs": {
            "fused_report": str(fused_json),
            "fused_estimates": str(fused_npz or fused_report["artifacts"]["estimates"]),
            "backbone_estimates": str(backbone_npz),
            "fit": str(fit_path),
            "splits": str(splits_path),
        },
    }
    report["transcript_vs_full_matrix"]["full_matrix_pooled_rmse"] = report[
        "full_matrix_reference"
    ]["pooled_rmse"]

    json_path = out_dir / f"{out_prefix}_gate3.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if make_plots:
        draw_blur_vs_n(report, out_dir / f"{out_prefix}_blur_vs_n.png")
        draw_gate3_curves(report, out_dir / f"{out_prefix}_gate3_curves.png")

    return report


def render(report: dict[str, Any]) -> str:
    verdict = report["verdict"]
    lines = ["", "=" * 78, "GATE 3 VERDICT -- person encoder", "=" * 78]
    lines.append(
        f"  confirmatory: {verdict['confirmatory_track']} track, N = "
        f"{verdict['confirmatory_n']}, {verdict['n_holdout_personas']} held-out personas"
    )
    lines.append("")
    for name, bar in verdict["bars"].items():
        mark = "PASS" if bar["pass"] else ("FAIL" if bar["pass"] is False else " ?? ")
        value = bar["value"]
        lines.append(f"  [{mark}] {name:16s} value {value:.4f}   bar {bar['bar']}")
        lines.append(f"         {bar['statement']}")
    lines.append("")
    lines.append(
        "  ALL DECIDED BARS PASS" if verdict["all_decided_bars_pass"]
        else "  FAILED: " + ", ".join(verdict["failed_bars"])
    )
    if verdict["undecided_bars"]:
        lines.append("  FOR THE ORCHESTRATOR: " + ", ".join(verdict["undecided_bars"]))
    lines.append("")

    confirm = report["tracks"][verdict["confirmatory_track"]]["bars"]
    closed = report["tracks"]["closed_only"]["bars"]
    lines.append("per-dimension RMSE at N=15 (bar 0.5), against the filed projection:")
    lines.append("    dim    fused   closed   projected(closed)   fused-vs-bar")
    for dim in DIMENSIONS:
        f = confirm["rmse"]["per_dimension_rmse"][dim]
        c = closed["rmse"]["per_dimension_rmse"][dim]
        p = confirm["rmse"]["projection"]["filed"][dim]
        lines.append(
            f"    {dim}   {f:.3f}   {c:.3f}      {p:.3f}            "
            f"{'PASS' if f <= BAR_RMSE else f'over by {f - BAR_RMSE:+.3f}'}"
        )
    lines.append("")
    lines.append("lift over the public-profile-only baseline (bar 30%):")
    lines.append(
        f"    pooled   fused {confirm['lift']['pooled_lift']:.1%}   "
        f"closed-only {closed['lift']['pooled_lift']:.1%}   "
        f"(projected closed-only {report['tracks']['fused']['bars']['lift']['projection']['filed_closed_only_lift']:.1%})"
    )
    for dim in DIMENSIONS:
        lines.append(
            f"    {dim}   fused {confirm['lift']['per_dimension_lift'][dim]:+.1%}   "
            f"closed-only {closed['lift']['per_dimension_lift'][dim]:+.1%}"
        )
    lines.append("")
    lines.append(
        f"blur coverage: fused {confirm['coverage']['pooled_coverage']:.4f}, "
        f"closed-only {closed['coverage']['pooled_coverage']:.4f}   band 0.60-0.75"
    )
    decay = report["anticipation_note"]["coverage_decay"]["fused"]
    lines.append(
        "    fused coverage against N: "
        f"N=1 {decay['at_n1']:.3f}  N=5 {decay['at_n5']:.3f}  "
        f"N=10 {decay['at_n10']:.3f}  N=15 {decay['at_n15']:.3f}"
    )
    lines.append("")
    mono = confirm["monotonicity"]
    lines.append("monotone blur shrinkage -- both readings, no verdict picked:")
    lines.append(
        f"    mean-curve reading : {mono['mean_curve_reading']['verdict']}, "
        f"{mono['mean_curve_reading']['n_dimensions_monotone']}/8 dimensions weakly "
        f"shrinking, largest uptick {mono['mean_curve_reading']['largest_uptick']:.2e}"
    )
    strict = mono["strict_per_step_reading"]
    if strict.get("n_steps_checked"):
        lines.append(
            f"    strict per-step    : {strict['verdict']}, "
            f"{strict['n_increases']}/{strict['n_steps_checked']} steps rise "
            f"({strict['fraction_increasing']:.1%}), largest {strict['largest_increase']:.2e}"
        )
    if mono.get("fixed_theta_control"):
        control = mono["fixed_theta_control"]
        lines.append(
            f"    fixed-theta control: {control['verdict']}, largest uptick "
            f"{control['largest_uptick']:.2e} (the theorem check)"
        )
    lines.append("")
    worth = report["open_answers_worth_in_closed_items"]["by_mean_blur"]

    def _fmt(value: Any) -> str:
        return "more than the interview" if value is None else f"{value:.1f}"

    lines.append(
        f"the three open answers are worth {_fmt(worth['at_n0'])} closed items at N=0 "
        f"and {_fmt(worth['at_n15'])} at N=15 (blur matching)"
    )
    full = report["full_matrix_reference"]
    lines.append("")
    lines.append(
        f"transcript against full matrix, both against planted truth: "
        f"202 items pooled RMSE {full['pooled_rmse']:.3f} (mean r {full['mean_r']:.3f}) "
        f"vs N=15 fused {confirm['rmse']['pooled_rmse']:.3f}"
    )
    hunt = report["leakage_hunt"]
    lines.append("")
    lines.append(
        f"leakage hunt: trigger {'FIRED' if hunt['trigger']['fired'] else 'did not fire'} "
        f"-- {hunt['verdict']}"
    )
    lines.append(f"    splits: {hunt['splits']['verdict']}")
    lines.append(f"    measurement-model scope: {hunt['measurement_model_scope']['verdict']}")
    if hunt["feature_provenance"].get("verdict"):
        lines.append(f"    feature provenance: {hunt['feature_provenance']['verdict']}")
    lines.append(f"    closed-only vs its projection: {hunt['closed_only_against_its_projection']['verdict']}")
    lines.append(f"    backbone integrity: {hunt['backbone_integrity']['verdict']}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.eval.gate3",
        description="Grade the Stage 3 person encoder against the planted truth (Gate 3).",
    )
    parser.add_argument("--fused", type=Path, required=True, help="stage3_person_fused.json")
    parser.add_argument("--fused-npz", type=Path, default=None, help="its estimates npz")
    parser.add_argument(
        "--backbone", type=Path, required=True,
        help="stage3_person_backbone.npz -- the public-profile-only baseline lives here",
    )
    parser.add_argument("--fit", type=Path, required=True, help="Stage 2 fit npz")
    parser.add_argument("--truth", type=Path, required=True, help="the batch's planted directory")
    parser.add_argument("--splits", type=Path, required=True, help="frozen splits manifest")
    parser.add_argument(
        "--features", type=Path, default=None,
        help="open-answer feature npz, for the provenance half of the leakage hunt",
    )
    parser.add_argument("--out", type=Path, required=True, help="results directory")
    parser.add_argument("--out-prefix", default=DEFAULT_OUT_PREFIX)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    report = grade(
        args.fused,
        args.backbone,
        args.fit,
        args.truth,
        args.splits,
        args.out,
        out_prefix=args.out_prefix,
        fused_npz=args.fused_npz,
        features_path=args.features,
        make_plots=not args.no_plots,
    )
    print(render(report))
    print(f"report -> {args.out / f'{args.out_prefix}_gate3.json'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
