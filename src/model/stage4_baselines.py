"""The mandatory Stage 4 baselines (PREREGISTRATION.md section 4).

    python -m src.model.stage4_baselines build \\
        --fit results/stage2_v2_fit.npz \\
        --transcripts data/runs/interview_v1/transcripts.jsonl \\
        --answers data/runs/full_sweep_v2/answers_noised.jsonl \\
        --out results/stage4_baselines.npz

Three arms are built here, all of them predicting the same thing the model
predicts: a distribution over the answer categories, for every one of the 100
held-out personas on every one of the 50 probe items.

1. **Population marginal** -- the zero-information reference. Per probe item,
   the empirical distribution of the 400 training personas' noised main-round
   answers, add-0.5 smoothed. The same row for every held-out persona.

2. **Profile-only** -- demographics, no interview. Per probe item, a ridge from
   the public-profile design matrix onto the one-hot training answer, so the
   prediction is a per-person distribution. The featurization is
   ``src.model.profile_baseline``'s, reused rather than restated. Expect this to
   collapse toward the marginal: Gate 3 already found the profile carries
   essentially no trait signal in this population.

3. **k-NN** -- the target persona is represented by the 15 answers they gave in
   the interview sitting; the neighbour pool is the 400 training personas'
   answers from the **same** sitting type, so neither side has an advantage in
   noise. Distance is the summed per-item disagreement: 0/1 on a binary item,
   ``|delta| / 4`` on a Likert-5 one. The prediction is the add-0.5-smoothed
   empirical distribution of the k nearest neighbours' noised main-round
   answers on the probe item. **k is chosen on training data only**, by
   leave-one-out over the 400 training personas against their own one-hot
   main-round answers, scored with the same multiclass Brier the gate uses.

The fourth mandatory baseline -- an LLM given the public profile and no
interview -- runs on the cluster and lands as a separate file; the grader picks
it up if it is there and says "pending" if it is not.

WALL. System-side module: noised answers, public profiles, the frozen splits
and the fitted item list. Nothing else.
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

from . import mirt
from .predictor import MAX_CATEGORIES, load_codes
from .profile_baseline import build_design, fit_feature_spec

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Laplace smoothing added to every category count before normalizing.
SMOOTHING = 0.5

#: The neighbourhood sizes searched. Chosen once, recorded, never re-opened.
K_GRID: tuple[int, ...] = (1, 3, 5, 10, 20, 40, 80, 160)

#: Ridge strengths for the profile arm. Runs far past collapse on purpose.
PROFILE_LAMBDA_GRID: tuple[float, ...] = tuple(float(v) for v in np.logspace(-2.0, 8.0, 41))

DEFAULT_FOLDS = 5
DEFAULT_SEED = 2026

SCHEMA = "glassbox.stage4.baselines/1"


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------


def one_hot(codes: np.ndarray, n_categories: np.ndarray) -> np.ndarray:
    """(n_people, n_items, 5) indicator of the answer actually given."""
    out = np.zeros((codes.shape[0], codes.shape[1], MAX_CATEGORIES), dtype=float)
    rows, cols = np.nonzero(codes >= 0)
    out[rows, cols, codes[rows, cols]] = 1.0
    return out * category_mask(n_categories)[None, :, :]


def category_mask(n_categories: np.ndarray) -> np.ndarray:
    """(n_items, 5) boolean: which columns this item actually has."""
    mask = np.zeros((len(n_categories), MAX_CATEGORIES), dtype=bool)
    for col, k in enumerate(n_categories):
        mask[col, : int(k)] = True
    return mask


def brier(predicted: np.ndarray, target: np.ndarray, n_categories: np.ndarray) -> np.ndarray:
    """(n_people, n_items) multiclass Brier, mean over the item's own categories."""
    mask = category_mask(n_categories)[None, :, :]
    squared = np.where(mask, (predicted - target) ** 2, 0.0)
    return squared.sum(axis=2) / np.asarray(n_categories, dtype=float)[None, :]


def smoothed_distribution(counts: np.ndarray, n_categories: np.ndarray) -> np.ndarray:
    """Add-0.5 smoothing, normalized over each item's own categories.

    ``counts`` is (..., n_items, 5); columns the item does not have stay zero.
    """
    mask = category_mask(n_categories)
    smoothed = np.where(mask, counts + SMOOTHING, 0.0)
    return smoothed / smoothed.sum(axis=-1, keepdims=True)


# --------------------------------------------------------------------------
# 1. population marginal
# --------------------------------------------------------------------------


def population_marginal(
    train_codes: np.ndarray, n_categories: np.ndarray
) -> np.ndarray:
    """(n_items, 5). The zero-information reference."""
    counts = np.zeros((train_codes.shape[1], MAX_CATEGORIES), dtype=float)
    for category in range(MAX_CATEGORIES):
        counts[:, category] = np.sum(train_codes == category, axis=0)
    return smoothed_distribution(counts, n_categories)


# --------------------------------------------------------------------------
# 2. profile-only
# --------------------------------------------------------------------------


def _renormalize(raw: np.ndarray, n_categories: np.ndarray, floor: float = 1e-4) -> np.ndarray:
    """A ridge fit onto one-hot targets can go negative; floor it and renormalize."""
    mask = category_mask(n_categories)[None, :, :]
    clipped = np.where(mask, np.maximum(raw, floor), 0.0)
    return clipped / clipped.sum(axis=2, keepdims=True)


def _ridge_multi(x: np.ndarray, y: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray]:
    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    xc = x - x_mean
    gram = xc.T @ xc + lam * np.eye(x.shape[1])
    weights = np.linalg.solve(gram, xc.T @ (y - y_mean))
    return weights, y_mean - x_mean @ weights


def profile_baseline_cells(
    profiles: Mapping[str, Mapping[str, Any]],
    train_pids: Sequence[str],
    holdout_pids: Sequence[str],
    train_codes: np.ndarray,
    n_categories: np.ndarray,
    *,
    folds: int = DEFAULT_FOLDS,
    seed: int = DEFAULT_SEED,
) -> tuple[np.ndarray, dict[str, Any]]:
    """(n_holdout, n_items, 5) plus a report. One shared lambda, chosen by CV."""
    spec = fit_feature_spec([profiles[pid] for pid in train_pids])
    x_train, _ = build_design([profiles[pid] for pid in train_pids], spec)
    x_holdout, unseen = build_design([profiles[pid] for pid in holdout_pids], spec)

    n_items = train_codes.shape[1]
    targets = one_hot(train_codes, n_categories)  # (n_train, n_items, 5)
    flat = targets.reshape(targets.shape[0], -1)  # ridge is per-column anyway

    order = np.random.default_rng(seed).permutation(len(train_pids))
    fold_of = np.zeros(len(train_pids), dtype=int)
    fold_of[order] = np.arange(len(train_pids)) % folds

    curve = []
    for lam in PROFILE_LAMBDA_GRID:
        predicted = np.zeros_like(flat)
        for k in range(folds):
            test = fold_of == k
            weights, intercept = _ridge_multi(x_train[~test], flat[~test], lam)
            predicted[test] = x_train[test] @ weights + intercept
        cells = _renormalize(predicted.reshape(targets.shape), n_categories)
        curve.append(
            {"lambda": float(lam), "cv_brier": float(brier(cells, targets, n_categories).mean())}
        )
    best = min(curve, key=lambda row: row["cv_brier"])

    weights, intercept = _ridge_multi(x_train, flat, best["lambda"])
    holdout = _renormalize(
        (x_holdout @ weights + intercept).reshape(len(holdout_pids), n_items, MAX_CATEGORIES),
        n_categories,
    )

    marginal = population_marginal(train_codes, n_categories)
    report = {
        "what": "per-item ridge from the public profile onto the one-hot training answer",
        "featurization": "src.model.profile_baseline (age, age^2, one-hot occupation / city size / household / region type)",
        "n_feature_columns": int(x_train.shape[1]),
        "holdout_rows_with_an_unseen_level": len(unseen),
        "cross_validation": {
            "folds": folds,
            "seed": seed,
            "grid_size": len(PROFILE_LAMBDA_GRID),
            "best_lambda": best["lambda"],
            "best_cv_brier": best["cv_brier"],
            "best_lambda_is_grid_edge": bool(
                best["lambda"] in (PROFILE_LAMBDA_GRID[0], PROFILE_LAMBDA_GRID[-1])
            ),
            "curve": curve,
        },
        "collapse_check": {
            "mean_abs_deviation_from_the_marginal": float(
                np.mean(np.abs(holdout - marginal[None, :, :]))
            ),
            "max_abs_deviation_from_the_marginal": float(
                np.max(np.abs(holdout - marginal[None, :, :]))
            ),
            "note": (
                "Gate 3 found the public profile carries essentially no trait "
                "signal in this population, so this arm is expected to sit on "
                "top of the marginal. The two numbers above say how close it got."
            ),
        },
    }
    return holdout, report


# --------------------------------------------------------------------------
# 3. k-NN over the interview answers
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class InterviewMatrix:
    """The 15 scripted answers per persona, coded, with the item types."""

    pids: list[str]
    item_ids: list[str]
    kinds: list[str]
    codes: np.ndarray  # (n_people, 15)


def interview_matrix(
    transcripts_path: Path, bank: Mapping[str, dict], pids: Sequence[str]
) -> InterviewMatrix:
    """Code the interview sitting for ``pids``, in the scripted order."""
    from .person_encoder import load_transcripts

    transcripts = load_transcripts(transcripts_path)
    item_ids = list(transcripts.item_ids)
    kinds = [bank[i]["type"] for i in item_ids]
    codes = np.full((len(pids), len(item_ids)), -1, dtype=np.int16)
    for row, pid in enumerate(pids):
        answers = transcripts.answers[pid]
        for col, (answer, kind) in enumerate(zip(answers, kinds)):
            codes[row, col] = mirt._code_answer(answer, kind)
    return InterviewMatrix(pids=list(pids), item_ids=item_ids, kinds=kinds, codes=codes)


def disagreement_matrix(a: np.ndarray, b: np.ndarray, kinds: Sequence[str]) -> np.ndarray:
    """(n_a, n_b) summed per-item disagreement between two coded interview blocks.

    Binary items contribute 0 or 1; Likert-5 items contribute ``|delta| / 4``,
    so one item can never contribute more than 1 whatever its type.
    """
    scale = np.array(
        [4.0 if kind == mirt.LIKERT else 1.0 for kind in kinds], dtype=float
    )
    delta = np.abs(a[:, None, :].astype(float) - b[None, :, :].astype(float))
    return (delta / scale[None, None, :]).sum(axis=2)


def knn_predict(
    distances: np.ndarray,
    neighbour_codes: np.ndarray,
    n_categories: np.ndarray,
    k: int,
    *,
    exclude_self: bool = False,
) -> np.ndarray:
    """(n_targets, n_items, 5) from the k nearest neighbours' main-round answers.

    Ties are broken by neighbour index, deterministically: ``argsort`` runs in
    stable mode over the distance row, so the same input always produces the
    same neighbourhood.

    ``exclude_self`` pushes row ``i``'s distance to neighbour ``i`` to infinity
    rather than dropping whichever neighbour happens to sort first. Two personas
    can and do give identical interview answers, and then "drop the nearest"
    would drop the twin and keep the persona itself -- which is the exact
    leakage the leave-one-out is there to prevent.
    """
    if exclude_self:
        if distances.shape[0] != distances.shape[1]:
            raise ValueError("exclude_self needs a square distance matrix")
        distances = distances.copy()
        np.fill_diagonal(distances, np.inf)
    order = np.argsort(distances, axis=1, kind="stable")
    chosen = order[:, :k]  # (n_targets, k)
    counts = np.zeros((distances.shape[0], neighbour_codes.shape[1], MAX_CATEGORIES))
    for category in range(MAX_CATEGORIES):
        hit = (neighbour_codes == category).astype(float)  # (n_neighbours, n_items)
        counts[:, :, category] = hit[chosen].sum(axis=1)
    return smoothed_distribution(counts, n_categories)


def choose_k(
    train_interview: np.ndarray,
    train_codes: np.ndarray,
    kinds: Sequence[str],
    n_categories: np.ndarray,
    *,
    grid: Sequence[int] = K_GRID,
) -> tuple[int, list[dict[str, Any]]]:
    """Leave-one-out over the 400 training personas. Held-out data takes no part.

    For each training persona the neighbourhood is drawn from the other 399:
    the persona's own row is removed by name, not by position, so a twin at
    distance zero cannot stand in for it.
    """
    distances = disagreement_matrix(train_interview, train_interview, kinds)
    target = one_hot(train_codes, n_categories)
    curve: list[dict[str, Any]] = []
    for k in grid:
        if k >= train_interview.shape[0]:
            continue
        predicted = knn_predict(
            distances, train_codes, n_categories, k, exclude_self=True
        )
        curve.append(
            {
                "k": int(k),
                "loo_brier": float(brier(predicted, target, n_categories).mean()),
            }
        )
    best = min(curve, key=lambda row: row["loo_brier"])
    return int(best["k"]), curve


# --------------------------------------------------------------------------
# the build
# --------------------------------------------------------------------------


def build(
    fit_path: Path,
    transcripts_path: Path,
    answers_path: Path,
    bank_path: Path,
    out_path: Path,
    *,
    verbose: bool = True,
) -> dict[str, Any]:
    started = time.time()
    fit = np.load(fit_path, allow_pickle=False)
    bank = mirt.load_bank(bank_path)

    train_pids = [str(p) for p in fit["persona_train_ids"]]
    holdout_pids = [str(p) for p in fit["persona_holdout_ids"]]
    probe_ids = [str(i) for i in fit["item_holdout_ids"]]
    kinds = [str(k) for k in fit["holdout_item_kinds"]]
    n_categories = np.array(
        [MAX_CATEGORIES if k == mirt.LIKERT else 2 for k in kinds], dtype=int
    )

    train_codes = load_codes(answers_path, train_pids, probe_ids, kinds)
    holdout_codes = load_codes(answers_path, holdout_pids, probe_ids, kinds)
    if int((train_codes < 0).sum()) or int((holdout_codes < 0).sum()):
        raise ValueError("the probe answer matrix has holes in it")

    # ---- 1. population marginal ------------------------------------------
    marginal = population_marginal(train_codes, n_categories)
    marginal_cells = np.repeat(marginal[None, :, :], len(holdout_pids), axis=0)

    # ---- 2. profile-only --------------------------------------------------
    from .person_encoder import load_transcripts

    transcripts = load_transcripts(transcripts_path)
    profile_cells, profile_report = profile_baseline_cells(
        transcripts.profiles, train_pids, holdout_pids, train_codes, n_categories
    )

    # ---- 3. k-NN ----------------------------------------------------------
    train_interview = interview_matrix(transcripts_path, bank, train_pids)
    holdout_interview = interview_matrix(transcripts_path, bank, holdout_pids)
    if verbose:
        print("choosing k by leave-one-out on the 400 training personas ...")
    k, k_curve = choose_k(
        train_interview.codes, train_codes, train_interview.kinds, n_categories
    )
    distances = disagreement_matrix(
        holdout_interview.codes, train_interview.codes, train_interview.kinds
    )
    knn_cells = knn_predict(distances, train_codes, n_categories, k)
    if verbose:
        print(f"  chosen k = {k}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        persona_ids=np.array(holdout_pids),
        item_ids=np.array(probe_ids),
        item_kinds=np.array(kinds),
        n_categories=n_categories,
        probs_marginal=marginal_cells,
        probs_profile=profile_cells,
        probs_knn=knn_cells,
        holdout_main_round_codes=holdout_codes,
        train_main_round_codes=train_codes,
        knn_distances=distances,
        knn_k=np.array(k),
    )

    report = {
        "schema": SCHEMA,
        "module": "src.model.stage4_baselines",
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "arms": {
            "marginal": {
                "what": (
                    "per probe item, the add-0.5-smoothed empirical distribution "
                    "of the 400 training personas' noised main-round answers"
                ),
                "smoothing": SMOOTHING,
                "role": "the zero-information reference the primary lift bar is measured against",
            },
            "profile": profile_report,
            "knn": {
                "what": (
                    "add-0.5-smoothed empirical distribution of the k nearest "
                    "training personas' noised main-round answers on the probe item"
                ),
                "representation": "the 15 interview_v1 answers, same sitting type on both sides",
                "distance": "sum over the 15 items of per-item disagreement: binary 0/1, likert-5 |delta| / 4",
                "k_chosen": k,
                "k_selection": (
                    "leave-one-out over the 400 TRAINING personas, mean multiclass "
                    "Brier against their own one-hot noised main-round probe "
                    "answers; held-out personas take no part"
                ),
                "k_curve": k_curve,
                "distance_summary": {
                    "min": float(distances.min()),
                    "median": float(np.median(distances)),
                    "max": float(distances.max()),
                    "median_nearest": float(np.median(np.sort(distances, axis=1)[:, 0])),
                },
            },
        },
        "counts": {
            "train_personas": len(train_pids),
            "holdout_personas": len(holdout_pids),
            "probe_items": len(probe_ids),
            "interview_items": len(train_interview.item_ids),
        },
        "inputs": {
            "fit": str(fit_path),
            "transcripts": str(transcripts_path),
            "answers": str(answers_path),
            "bank": str(bank_path),
        },
        "artifacts": {"baselines": str(out_path)},
        "runtime_seconds": time.time() - started,
    }
    report_path = out_path.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if verbose:
        print(f"baselines -> {out_path}")
        print(f"report    -> {report_path}")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.model.stage4_baselines",
        description="Build the mandatory Stage 4 baselines over the held-out cells.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    cmd = sub.add_parser("build")
    cmd.add_argument("--fit", type=Path, default=REPO_ROOT / "results" / "stage2_v2_fit.npz")
    cmd.add_argument(
        "--transcripts",
        type=Path,
        default=REPO_ROOT / "data" / "runs" / "interview_v1" / "transcripts.jsonl",
    )
    cmd.add_argument(
        "--answers",
        type=Path,
        default=REPO_ROOT / "data" / "runs" / "full_sweep_v2" / "answers_noised.jsonl",
    )
    cmd.add_argument("--bank", type=Path, default=REPO_ROOT / "data" / "public" / "bank_items.json")
    cmd.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "stage4_baselines.npz")
    cmd.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    build(
        args.fit,
        args.transcripts,
        args.answers,
        args.bank,
        args.out,
        verbose=not args.quiet,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
