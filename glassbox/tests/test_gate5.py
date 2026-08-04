"""Tests for the Gate 5 grader.

Most of these are about honesty rather than arithmetic: a strategy that never
reaches the target must not end up with a median that looks like it did, and the
frozen bar's comparison must come out UNDEFINED rather than PASS when one side of
it does not exist. Owner RULING 1 is what those tests are protecting.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval import gate5
from src.model import mirt

DIMS = 8
DIMENSION_NAMES = ("TRU", "RSK", "ENV", "PRC", "TRD", "SOC", "TEC", "LOC")


# --------------------------------------------------------------------------
# crossings
# --------------------------------------------------------------------------


def test_crossing_is_the_first_n_where_every_dimension_clears():
    curve = np.array([[0.9, 0.9], [0.4, 0.9], [0.4, 0.4], [0.6, 0.3]])
    assert gate5.crossing(curve, 0.5) == 3
    assert gate5.crossing(curve, 0.95) == 1
    assert gate5.crossing(curve, 0.1) is None


def test_censored_median_does_not_drop_the_replicates_that_never_crossed():
    values = [10, 12, None, None, None]
    stats = gate5.censored_median(values, budget=150)
    assert stats["n_reached"] == 2
    assert stats["median_is_censored"] is True
    assert stats["median"] is None
    assert stats["median_lower_bound"] == 150
    assert stats["min"] == 10 and stats["max"] == 12


def test_censored_median_reports_a_median_when_most_replicates_crossed():
    # the two that never crossed sit above the budget, so they pull the median
    # up to 8 rather than vanishing and leaving it at 6.
    stats = gate5.censored_median([4, 6, 8, None, None], budget=20)
    assert stats["median"] == pytest.approx(8.0)
    assert stats["median_is_censored"] is False
    assert stats["fraction_reached"] == pytest.approx(0.6)


def test_the_bar_comparison_is_undefined_when_a_median_is_missing():
    adaptive = gate5.censored_median([5, 5, 5], budget=50)
    never = gate5.censored_median([None, None, None], budget=50)
    assert gate5.ratio_comparison(adaptive, never)["verdict"] == "UNDEFINED"
    assert gate5.ratio_comparison(never, adaptive)["verdict"] == "UNDEFINED"


def test_the_bar_comparison_passes_only_at_half_or_better():
    slow = gate5.censored_median([20, 20, 20], budget=50)
    half = gate5.censored_median([10, 10, 10], budget=50)
    just_over = gate5.censored_median([11, 11, 11], budget=50)
    assert gate5.ratio_comparison(half, slow)["verdict"] == "PASS"
    assert gate5.ratio_comparison(half, slow)["ratio"] == pytest.approx(0.5)
    assert gate5.ratio_comparison(just_over, slow)["verdict"] == "FAIL"


def test_per_persona_crossings_report_the_ones_that_never_cleared():
    # (replicates, steps, personas)
    errors = np.array(
        [
            [[0.9, 0.9], [0.4, 0.9], [0.4, 0.9]],
            [[0.9, 0.2], [0.3, 0.2], [0.3, 0.2]],
        ]
    )
    stats = gate5.per_persona_crossings(errors, 0.5, budget=3)
    assert stats["n_persona_episodes"] == 4
    assert stats["n_reached"] == 3
    assert stats["fraction_reached"] == pytest.approx(0.75)
    assert stats["median_of_reached"] == pytest.approx(2.0)


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------


def fake_result(replicate: int, rmse: np.ndarray, *, pooled_full: float = 0.4) -> dict:
    steps, dims = rmse.shape
    return {
        "strategy": "random",
        "arm": "random",
        "replicate": replicate,
        "round": f"ep_random_{replicate:03d}",
        "n_steps": steps,
        "capped_at": None,
        "rmse_by_n": rmse,
        "pooled_by_n": rmse.mean(axis=1),
        "blur_by_n": rmse * 0.5,
        "blur_calibrated_by_n": rmse * 0.6,
        "max_error_by_n": np.tile(rmse.max(axis=1)[:, None], (1, 3)),
        "newton_iters": np.full(steps, 4),
        "rmse_full_per_dim": np.full(dims, pooled_full),
        "pooled_full": pooled_full,
        "n_distinct_items": steps,
    }


def test_summarize_arm_builds_the_curves_and_the_crossings():
    steps = 6
    curve = np.linspace(1.0, 0.3, steps)[:, None].repeat(DIMS, axis=1)
    results = [fake_result(r, curve.copy()) for r in range(3)]
    truth = np.zeros((3, DIMS))
    truth[:, 0] = 1.0
    summary = gate5.summarize_arm(results, budget=10, theta_true_holdout=truth)
    assert summary["n_replicates"] == 3
    assert summary["n_steps"] == steps
    assert summary["n_grid"] == list(range(1, steps + 1))
    assert len(summary["pooled_rmse_mean_by_n"]) == steps
    assert summary["questions_to_target"]["0.50"]["n_reached"] == 3
    assert summary["questions_to_target"]["0.50"]["median"] == pytest.approx(
        gate5.crossing(curve, 0.5)
    )


def test_summarize_arm_censors_at_the_strategy_s_own_last_question():
    """A 15-question script is censored at 15, not at the 150-question budget."""
    curve = np.full((15, DIMS), 0.9)
    results = [fake_result(r, curve.copy()) for r in range(3)]
    truth = np.zeros((3, DIMS))
    summary = gate5.summarize_arm(results, budget=150, theta_true_holdout=truth)
    block = summary["questions_to_target"]["0.50"]
    assert block["budget"] == 15
    assert block["median_lower_bound"] == 15


def test_fraction_of_full_matrix_is_one_at_the_full_bank_value():
    steps = 30
    curve = np.full((steps, DIMS), 0.4)
    results = [fake_result(r, curve.copy(), pooled_full=0.4) for r in range(2)]
    truth = np.ones((5, DIMS))  # rmse at N = 0 is exactly 1
    summary = gate5.summarize_arm(results, budget=steps, theta_true_holdout=truth)
    for n in (5, 10, 15, 25):
        assert summary["fraction_of_full_matrix"][str(n)] == pytest.approx(1.0)


def test_the_proxy_gaming_watch_spots_blur_falling_faster_than_error():
    arms = {
        "honest": {
            "n_grid": [1, 2],
            "declared_blur_mean_by_n": [0.8, 0.6],
            "pooled_rmse_mean_by_n": [0.8, 0.6],
            "rmse_at_n0": 1.0,
        },
        "gamer": {
            "n_grid": [1, 2],
            "declared_blur_mean_by_n": [0.4, 0.2],
            "pooled_rmse_mean_by_n": [0.95, 0.9],
            "rmse_at_n0": 1.0,
        },
    }
    watch = gate5.proxy_gaming_watch(arms)
    assert watch["honest"]["largest_gap"]["gap"] == pytest.approx(0.0)
    assert watch["gamer"]["largest_gap"]["gap"] > 0.5


# --------------------------------------------------------------------------
# a whole small run
# --------------------------------------------------------------------------


def build_world(tmp_path: Path, *, n_items: int = 8, n_train: int = 6, n_holdout: int = 4):
    """A miniature Stage 2 world: a fit, a backbone, splits, truth, episodes."""
    rng = np.random.default_rng(4)
    kinds = [mirt.LIKERT if j % 2 == 0 else mirt.BINARY for j in range(n_items)]
    loadings = rng.normal(0.0, 0.8, size=(n_items, DIMS))
    thresholds = np.sort(rng.normal(0.0, 0.5, size=(n_items, mirt.LIKERT_CUTS)), axis=1)
    thresholds += np.arange(mirt.LIKERT_CUTS) * 0.05  # strictly increasing
    difficulty = rng.normal(0.0, 0.4, size=n_items)

    train_ids = [f"p{i:04d}" for i in range(n_train)]
    holdout_ids = [f"p{i:04d}" for i in range(n_train, n_train + n_holdout)]
    item_ids = [f"q{j:03d}" for j in range(n_items)]

    fit_path = tmp_path / "fit.npz"
    np.savez(
        fit_path,
        dims=np.array(DIMS),
        item_train_ids=np.array(item_ids),
        item_holdout_ids=np.array([]),
        train_loadings=loadings,
        train_thresholds=thresholds,
        train_difficulty=difficulty,
        train_item_kinds=np.array(kinds),
        train_discrimination=np.linalg.norm(loadings, axis=1),
        persona_train_ids=np.array(train_ids),
        persona_holdout_ids=np.array(holdout_ids),
    )

    backbone_path = tmp_path / "backbone.npz"
    np.savez(backbone_path, blur_kappa=np.full(DIMS, 1.1))

    splits_path = tmp_path / "splits.json"
    splits_path.write_text(
        json.dumps({"interview_closed": item_ids[:3]}), encoding="utf-8"
    )

    truth_dir = tmp_path / "planted"
    (truth_dir / "theta").mkdir(parents=True)
    for pid in train_ids + holdout_ids:
        values = rng.normal(0.0, 1.0, size=DIMS)
        (truth_dir / "theta" / f"{pid}.json").write_text(
            json.dumps({"theta": dict(zip(DIMENSION_NAMES, values.tolist()))}),
            encoding="utf-8",
        )

    rounds = [
        gate5.round_tag(name, rep)
        for name in gate5.STRATEGIES
        for rep in range(2)
    ]
    codes = np.zeros((len(rounds), n_train + n_holdout, n_items), dtype=np.int8)
    for k in range(len(rounds)):
        for j, kind in enumerate(kinds):
            top = mirt.LIKERT_CATEGORIES if kind == mirt.LIKERT else 2
            codes[k, :, j] = rng.integers(0, top, size=n_train + n_holdout)
    episodes_path = tmp_path / "episodes.npz"
    np.savez(
        episodes_path,
        codes=codes,
        rounds=np.array(rounds),
        persona_ids=np.array(train_ids + holdout_ids),
        item_ids=np.array(item_ids),
    )
    return episodes_path, fit_path, backbone_path, splits_path, truth_dir


def test_a_whole_small_grade_runs_and_reports_every_required_block(tmp_path):
    episodes, fit, backbone, splits, truth = build_world(tmp_path)
    report = gate5.grade(
        episodes, fit, backbone, splits, truth, tmp_path / "out",
        n_max=5, replicates=2, workers=1, make_plots=False, verbose=False,
    )

    assert report["schema"] == gate5.SCHEMA
    assert set(report["curves"]) == set(gate5.STRATEGIES)
    assert report["design"]["n_replicates"] == 2

    frozen = report["frozen_bar"]
    assert frozen["target"]["value"] == 0.5
    assert "GRADER'S READING" in frozen["grader_reading"]
    assert set(frozen["population_curve_crossing"]) == set(gate5.STRATEGIES)
    assert frozen["comparison"]["heuristic_vs_random"]["verdict"] in {
        "PASS", "FAIL", "UNDEFINED"
    }

    exploratory = report["exploratory"]
    assert exploratory["label"].startswith("EXPLORATORY")
    assert set(exploratory["threshold_grid"]) == {"0.60", "0.65", "0.70"}
    assert "definition" in exploratory["fraction_of_full_matrix"]
    assert set(exploratory["blur_vs_truth"]) == set(gate5.STRATEGIES)

    # the scripted arm stops where the script stops
    assert report["curves"]["fixed"]["n_steps"] == 3
    assert report["curves"]["fixed"]["capped_at"] == 3
    assert report["curves"]["random"]["n_steps"] == 5

    written = json.loads((tmp_path / "out" / "stage5_strategies.json").read_text())
    assert written["schema"] == gate5.SCHEMA
    archive = np.load(tmp_path / "out" / "stage5_strategies.npz", allow_pickle=False)
    assert archive["random_rmse_by_n"].shape == (2, 5, DIMS)
    assert archive["fixed_rmse_by_n"].shape == (2, 3, DIMS)


def test_the_grade_refuses_a_cache_drawn_on_a_different_bank(tmp_path):
    episodes, fit, backbone, splits, truth = build_world(tmp_path)
    archive = dict(np.load(episodes, allow_pickle=False))
    archive["item_ids"] = np.array([f"z{j:03d}" for j in range(archive["codes"].shape[2])])
    other = tmp_path / "other.npz"
    np.savez(other, **archive)
    with pytest.raises(ValueError, match="different candidate bank"):
        gate5.grade(
            other, fit, backbone, splits, truth, tmp_path / "out2",
            n_max=3, replicates=2, workers=1, make_plots=False, verbose=False,
        )


def test_the_render_summarises_without_crashing(tmp_path):
    episodes, fit, backbone, splits, truth = build_world(tmp_path)
    report = gate5.grade(
        episodes, fit, backbone, splits, truth, tmp_path / "out",
        n_max=5, replicates=2, workers=1, make_plots=False, verbose=False,
    )
    text = gate5.render(report)
    assert "FROZEN BAR" in text
    assert "EXPLORATORY threshold grid" in text
    for name in gate5.STRATEGIES:
        assert name in text


def test_the_grader_documents_the_reading_and_the_rulings():
    """The frozen wording is qualitative; the reading has to be on the record."""
    assert "GRADER'S READING" in gate5.GRADER_READING
    assert "population-curve crossing" in gate5.GRADER_READING
    assert "per-persona crossing" in gate5.GRADER_READING
    assert "RULING 1" in gate5.GRADER_READING
    assert gate5.EXPLORATORY_LABEL.startswith("EXPLORATORY")
    assert "pre-declared" in gate5.EXPLORATORY_LABEL
    assert "RMSE_full" in gate5.FRACTION_DEFINITION
    assert "leave-one-out" in gate5.COLD_START_LABEL
