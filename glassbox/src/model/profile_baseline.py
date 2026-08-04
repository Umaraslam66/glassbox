"""Public-profile-only baseline for Stage 3 (PREREGISTRATION section 4, baseline 2).

    python -m src.model.profile_baseline \
        --transcripts data/runs/interview_v1/transcripts.jsonl \
        --fit results/stage2_v2_fit.npz \
        --out results/stage3_profile_baseline.json

The mandatory "demographics, no interview" reference. Ridge regression from the
public profile -- age, occupation, city size, household, region type -- onto the
Stage 2 fit's own theta_hat, trained on the 400 training personas and applied
unchanged to the 100 held-out ones.

Design choices, all boring on purpose:

  * Features: age standardized, age squared, and one-hot indicators for
    occupation, city size, household and region type. The level lists are
    learned on the **training** personas; a level that only appears held-out
    gets an all-zero block and is counted in the report rather than silently
    absorbed.
  * One shared ridge strength for all eight targets, chosen by 5-fold
    cross-validation on the training personas with a recorded shuffle seed.
    Held-out personas take no part in choosing it.
  * The intercept is never penalized: features and targets are centered on
    their training means and the intercept is put back afterwards.

No LLM, no planted truth, no interview answers. The target is a fitted
system-side quantity, which is what makes this baseline legal to train at all.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

#: Profile fields the baseline is allowed to use, and how each is encoded.
NUMERIC_FIELDS: tuple[str, ...] = ("age",)
CATEGORICAL_FIELDS: tuple[str, ...] = (
    "occupation",
    "city_size",
    "household",
    "region_type",
)

#: Ridge strengths searched by cross-validation. The grid runs far past the
#: point where the fit collapses onto the intercept, so that a selected lambda
#: at the top end is a real plateau rather than the grid running out.
LAMBDA_GRID: tuple[float, ...] = tuple(float(v) for v in np.logspace(-3.0, 9.0, 49))

#: Folds and the shuffle seed for the cross-validation. Recorded, not tuned.
DEFAULT_FOLDS = 5
DEFAULT_SEED = 2026

SCHEMA = "glassbox.stage3.profile_baseline/1"


# --------------------------------------------------------------------------
# design matrix
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureSpec:
    """The encoding, fixed on the training personas and then frozen."""

    levels: dict[str, list[str]]
    age_mean: float
    age_sd: float

    @property
    def columns(self) -> list[str]:
        names = ["age_z", "age_z_sq"]
        for field in CATEGORICAL_FIELDS:
            names += [f"{field}={level}" for level in self.levels[field]]
        return names


def fit_feature_spec(profiles: Sequence[Mapping[str, Any]]) -> FeatureSpec:
    """Learn the level lists and the age scaling from the training profiles."""
    ages = np.array([float(p["age"]) for p in profiles], dtype=float)
    levels = {
        field: sorted({str(p[field]) for p in profiles}) for field in CATEGORICAL_FIELDS
    }
    sd = float(ages.std())
    return FeatureSpec(
        levels=levels, age_mean=float(ages.mean()), age_sd=sd if sd > 0 else 1.0
    )


def build_design(
    profiles: Sequence[Mapping[str, Any]], spec: FeatureSpec
) -> tuple[np.ndarray, list[str]]:
    """Encode profiles with a frozen spec. Unseen levels give a zero block."""
    rows = []
    unseen: list[str] = []
    for profile in profiles:
        age_z = (float(profile["age"]) - spec.age_mean) / spec.age_sd
        row = [age_z, age_z**2]
        for field in CATEGORICAL_FIELDS:
            value = str(profile[field])
            block = [1.0 if value == level else 0.0 for level in spec.levels[field]]
            if not any(block):
                unseen.append(f"{field}={value}")
            row.extend(block)
        rows.append(row)
    return np.array(rows, dtype=float), unseen


# --------------------------------------------------------------------------
# ridge
# --------------------------------------------------------------------------


def ridge_fit(
    x: np.ndarray, y: np.ndarray, lam: float
) -> tuple[np.ndarray, np.ndarray]:
    """Ridge weights and intercept, with the intercept left unpenalized."""
    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    xc = x - x_mean
    yc = y - y_mean
    gram = xc.T @ xc + lam * np.eye(x.shape[1])
    weights = np.linalg.solve(gram, xc.T @ yc)
    intercept = y_mean - x_mean @ weights
    return weights, intercept


def ridge_predict(x: np.ndarray, weights: np.ndarray, intercept: np.ndarray) -> np.ndarray:
    return x @ weights + intercept


def cross_validate(
    x: np.ndarray,
    y: np.ndarray,
    *,
    grid: Sequence[float] = LAMBDA_GRID,
    folds: int = DEFAULT_FOLDS,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Pick one ridge strength by k-fold CV on the rows given. Training only."""
    n = x.shape[0]
    order = np.random.default_rng(seed).permutation(n)
    fold_of = np.zeros(n, dtype=int)
    fold_of[order] = np.arange(n) % folds

    curve = []
    for lam in grid:
        predicted = np.zeros_like(y)
        for k in range(folds):
            test = fold_of == k
            weights, intercept = ridge_fit(x[~test], y[~test], lam)
            predicted[test] = ridge_predict(x[test], weights, intercept)
        mse_per_dim = np.mean((predicted - y) ** 2, axis=0)
        var_per_dim = np.var(y, axis=0)
        curve.append(
            {
                "lambda": float(lam),
                "cv_mse": float(mse_per_dim.mean()),
                "cv_r2_per_dim": [
                    float(1.0 - m / v) if v > 0 else float("nan")
                    for m, v in zip(mse_per_dim, var_per_dim)
                ],
            }
        )
    # what predicting the training mean and nothing else would score, so the
    # report can say how much the profile actually buys over an empty model
    intercept_only = np.zeros_like(y)
    for k in range(folds):
        test = fold_of == k
        intercept_only[test] = y[~test].mean(axis=0)
    empty_mse = float(np.mean((intercept_only - y) ** 2))

    best = min(curve, key=lambda row: row["cv_mse"])
    return {
        "folds": folds,
        "seed": seed,
        "grid_size": len(grid),
        "lambda_range": [float(min(grid)), float(max(grid))],
        "best_lambda_is_grid_edge": bool(
            best["lambda"] in (float(min(grid)), float(max(grid)))
        ),
        "intercept_only_cv_mse": empty_mse,
        "best_lambda": best["lambda"],
        "best_cv_mse": best["cv_mse"],
        "best_cv_r2_per_dim": best["cv_r2_per_dim"],
        "best_cv_r2_mean": float(np.mean(best["cv_r2_per_dim"])),
        "curve": curve,
    }


# --------------------------------------------------------------------------
# the baseline end to end
# --------------------------------------------------------------------------


def evaluate_profile_baseline(
    profiles: Mapping[str, Mapping[str, Any]],
    train_pids: Sequence[str],
    holdout_pids: Sequence[str],
    theta_train: np.ndarray,
    theta_holdout: np.ndarray,
    *,
    folds: int = DEFAULT_FOLDS,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Train on the training personas, predict the held-out ones, score both.

    Returns ``{"report": <json-safe dict>, "holdout": {"theta": ndarray, ...}}``
    so a caller can both write the numbers out and keep the predictions.
    """
    from .person_encoder import agreement  # local import: they use each other

    theta_train = np.asarray(theta_train, dtype=float)
    theta_holdout = np.asarray(theta_holdout, dtype=float)
    train_profiles = [profiles[pid] for pid in train_pids]
    holdout_profiles = [profiles[pid] for pid in holdout_pids]

    spec = fit_feature_spec(train_profiles)
    x_train, _ = build_design(train_profiles, spec)
    x_holdout, unseen = build_design(holdout_profiles, spec)

    cv = cross_validate(x_train, theta_train, folds=folds, seed=seed)
    weights, intercept = ridge_fit(x_train, theta_train, cv["best_lambda"])
    fitted_train = ridge_predict(x_train, weights, intercept)
    predicted_holdout = ridge_predict(x_holdout, weights, intercept)

    variance = np.var(theta_holdout, axis=0)
    residual = np.mean((predicted_holdout - theta_holdout) ** 2, axis=0)
    r2 = [float(1.0 - m / v) if v > 0 else float("nan") for m, v in zip(residual, variance)]

    # how far the chosen model is from the empty one. If cross-validation lands
    # on the intercept, the profile carries no out-of-sample signal at all and
    # this baseline is literally the zero-information reference.
    empty = cv["intercept_only_cv_mse"]
    gap = float((cv["best_cv_mse"] - empty) / empty) if empty > 0 else float("nan")
    collapsed = bool(abs(gap) < 1e-6)

    report = {
        "schema": SCHEMA,
        "what": "ridge from the public profile onto the fitted theta_hat",
        "features": {
            "numeric": list(NUMERIC_FIELDS) + ["age squared (on the standardized age)"],
            "categorical": list(CATEGORICAL_FIELDS),
            "n_columns": int(x_train.shape[1]),
            "n_levels_per_field": {
                field: len(spec.levels[field]) for field in CATEGORICAL_FIELDS
            },
            "levels_learned_on": "training personas only",
            "holdout_rows_with_an_unseen_level": len(unseen),
            "unseen_levels": sorted(set(unseen)),
        },
        "cross_validation": {
            **{k: v for k, v in cv.items() if k != "curve"},
            "cv_mse_gap_to_intercept_only": gap,
            "collapsed_to_intercept": collapsed,
            "reading": (
                "cross-validation selects the intercept-only model: the public "
                "profile carries no out-of-sample signal about theta_hat"
                if collapsed
                else "cross-validation keeps a non-degenerate profile model"
            ),
        },
        "cv_curve": cv["curve"],
        "train": {"agreement_with_full_matrix": agreement(fitted_train, theta_train)},
        "holdout": {
            "agreement_with_full_matrix": agreement(predicted_holdout, theta_holdout),
            "out_of_sample_r2_per_dim": r2,
            "out_of_sample_r2_mean": float(np.mean(r2)),
            "population_sd_per_dim": [float(v) for v in np.sqrt(variance)],
            "rmse_over_population_sd_per_dim": [
                float(np.sqrt(m / v)) if v > 0 else float("nan")
                for m, v in zip(residual, variance)
            ],
        },
        "note": (
            "the reference is the Stage 2 fit's own theta_hat from the full "
            "training-item matrix, the same reference the person encoder is "
            "scored against. Nothing here touches planted truth."
        ),
    }
    return {
        "report": report,
        "holdout": {
            "theta": predicted_holdout,
            "agreement_with_full_matrix": report["holdout"]["agreement_with_full_matrix"],
        },
        "train": {"theta": fitted_train},
        "spec": spec,
        "weights": weights,
        "intercept": intercept,
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def run(
    transcripts_path: Path,
    fit_path: Path,
    out_path: Path,
    *,
    folds: int = DEFAULT_FOLDS,
    seed: int = DEFAULT_SEED,
    verbose: bool = True,
) -> dict[str, Any]:
    from .person_encoder import load_transcripts

    started = time.time()
    fit = np.load(fit_path, allow_pickle=False)
    transcripts = load_transcripts(transcripts_path)
    train_pids = [str(p) for p in fit["persona_train_ids"]]
    holdout_pids = [str(p) for p in fit["persona_holdout_ids"]]

    result = evaluate_profile_baseline(
        transcripts.profiles,
        train_pids,
        holdout_pids,
        np.asarray(fit["theta_train"], dtype=float),
        np.asarray(fit["theta_holdout"], dtype=float),
        folds=folds,
        seed=seed,
    )
    report = dict(result["report"])
    report["module"] = "src.model.profile_baseline"
    report["generated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report["inputs"] = {"transcripts": str(transcripts_path), "fit": str(fit_path)}
    report["runtime_seconds"] = time.time() - started

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if verbose:
        holdout = report["holdout"]["agreement_with_full_matrix"]
        print(f"lambda {report['cross_validation']['best_lambda']:.4g}")
        print(f"held-out r {holdout['r_mean']:.4f}, rmse {holdout['rmse_mean']:.4f}")
        print(f"held-out R2 {report['holdout']['out_of_sample_r2_mean']:.4f}")
        print(f"report -> {out_path}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.model.profile_baseline",
        description="Public-profile-only ridge baseline onto the fitted theta_hat.",
    )
    parser.add_argument("--transcripts", type=Path, required=True)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    run(
        args.transcripts,
        args.fit,
        args.out,
        folds=args.folds,
        seed=args.seed,
        verbose=not args.quiet,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
