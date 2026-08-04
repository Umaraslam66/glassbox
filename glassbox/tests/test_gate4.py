"""Tests for the Gate 4 grader: the metrics, the binning, and the report schema."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval import gate4


# --------------------------------------------------------------------------
# the metrics
# --------------------------------------------------------------------------


def test_brier_is_the_mean_over_the_items_own_categories() -> None:
    predicted = np.zeros((1, 2, 5))
    predicted[0, 0] = [0.2, 0.2, 0.2, 0.2, 0.2]
    predicted[0, 1, :2] = [0.3, 0.7]
    truth = np.zeros((1, 2, 5))
    truth[0, 0] = [1.0, 0.0, 0.0, 0.0, 0.0]
    truth[0, 1, :2] = [0.5, 0.5]
    n_categories = np.array([5, 2])

    scores = gate4.brier_cells(predicted, truth, n_categories)
    assert np.isclose(scores[0, 0], (0.8**2 + 4 * 0.2**2) / 5)
    assert np.isclose(scores[0, 1], (0.2**2 + 0.2**2) / 2)


def test_log_loss_is_the_cross_entropy_and_never_infinite() -> None:
    predicted = np.zeros((1, 1, 5))
    predicted[0, 0] = [0.0, 0.0, 0.0, 0.0, 1.0]
    truth = np.zeros((1, 1, 5))
    truth[0, 0] = [1.0, 0.0, 0.0, 0.0, 0.0]
    value = gate4.log_loss_cells(predicted, truth, np.array([5]))[0, 0]
    assert np.isfinite(value)
    assert np.isclose(value, -np.log(gate4.LOG_EPS))

    truth[0, 0] = [0.0, 0.0, 0.0, 0.0, 1.0]
    assert np.isclose(gate4.log_loss_cells(predicted, truth, np.array([5]))[0, 0], 0.0)


def test_pair_arrays_drop_the_columns_an_item_does_not_have() -> None:
    predicted = np.zeros((2, 2, 5))
    predicted[:, 0] = 0.2
    predicted[:, 1, :2] = 0.5
    truth = predicted.copy()
    flat_p, flat_t = gate4.pair_arrays(predicted, truth, np.array([5, 2]))
    assert flat_p.size == 2 * (5 + 2)
    assert np.array_equal(flat_p, flat_t)


# --------------------------------------------------------------------------
# the calibration binning, hand-computed
# --------------------------------------------------------------------------


def test_ece_matches_a_hand_computed_example() -> None:
    """Four pairs, three occupied bins, one arithmetic answer.

    bin [0.0, 0.1): 0.05 vs 0.00           -> gap 0.05, n = 1
    bin [0.1, 0.2): 0.15, 0.15 vs 0.2, 0.4 -> gap |0.15 - 0.30| = 0.15, n = 2
    bin [0.9, 1.0]: 0.95 vs 1.00           -> gap 0.05, n = 1
    ECE = (1*0.05 + 2*0.15 + 1*0.05) / 4 = 0.10
    """
    predicted = np.array([0.05, 0.15, 0.15, 0.95])
    observed = np.array([0.0, 0.2, 0.4, 1.0])
    result = gate4.expected_calibration_error(predicted, observed)

    assert result["n_pairs"] == 4
    assert result["n_bins"] == 10
    assert np.isclose(result["ece"], 0.10)

    occupied = {tuple(b["bin"]): b for b in result["bins"] if b["n"]}
    assert set(occupied) == {(0.0, 0.1), (0.1, 0.2), (0.9, 1.0)}
    assert occupied[(0.1, 0.2)]["n"] == 2
    assert np.isclose(occupied[(0.1, 0.2)]["mean_observed"], 0.3)


def test_the_bin_edges_are_equal_width_and_include_one() -> None:
    result = gate4.expected_calibration_error(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    assert np.isclose(result["ece"], 0.0)
    counted = [b["n"] for b in result["bins"]]
    assert counted[0] == 1 and counted[-1] == 1 and sum(counted) == 2
    widths = {round(b["bin"][1] - b["bin"][0], 10) for b in result["bins"]}
    assert widths == {0.1}


def test_a_perfectly_calibrated_arm_scores_zero() -> None:
    rng = np.random.default_rng(1)
    values = rng.uniform(0.0, 1.0, size=5000)
    assert gate4.expected_calibration_error(values, values)["ece"] == 0.0


def test_a_uniformly_overconfident_arm_scores_its_own_offset() -> None:
    values = np.linspace(0.01, 0.89, 400)
    result = gate4.expected_calibration_error(values, values - 0.01)
    assert np.isclose(result["ece"], 0.01)


# --------------------------------------------------------------------------
# the frozen bars, as constants
# --------------------------------------------------------------------------


def test_the_bars_are_the_preregistered_numbers() -> None:
    assert gate4.BAR_LIFT == 0.25
    assert gate4.BAR_ECE == 0.05
    assert gate4.BAR_CORRELATION == 0.90
    assert gate4.N_BINS == 10
    assert gate4.LEAKAGE_CORRELATION == 0.95
    assert gate4.LEAKAGE_LIFT == 0.45


# --------------------------------------------------------------------------
# the optional cluster arm
# --------------------------------------------------------------------------


def test_the_qwen_arm_is_pending_when_its_file_is_absent(tmp_path: Path) -> None:
    probs, status = gate4.load_qwen_arm(
        tmp_path / "nothing.jsonl", ["p1"], ["q1"], np.array([2])
    )
    assert probs is None
    assert status["status"] == "pending"


def test_the_qwen_arm_is_read_and_renormalized_when_present(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    path.write_text(
        json.dumps({"persona_id": "p1", "item_id": "q1", "probs": [1.0, 3.0]}) + "\n"
        + json.dumps({"persona_id": "p1", "item_id": "q2", "probs": [1, 2, 3, 4, 5]}) + "\n"
        + json.dumps({"persona_id": "p2", "item_id": "q1", "probs": [0.5]}) + "\n",
        encoding="utf-8",
    )
    probs, status = gate4.load_qwen_arm(
        path, ["p1", "p2"], ["q1", "q2"], np.array([2, 5])
    )
    assert status["status"] == "graded"
    assert status["cells_parsed"] == 2
    assert status["cells_malformed"] == 1  # wrong width for a binary item
    assert np.allclose(probs[0, 0, :2], [0.25, 0.75])
    assert np.allclose(probs[0, 1], np.array([1, 2, 3, 4, 5]) / 15.0)
    # the two cells the file did not supply fall back to a uniform row
    assert np.allclose(probs[1, 0, :2], [0.5, 0.5])
    assert np.allclose(probs.sum(axis=2), 1.0)


# --------------------------------------------------------------------------
# aggregate checks
# --------------------------------------------------------------------------


def test_aggregate_marginals_are_exact_when_the_prediction_is() -> None:
    rng = np.random.default_rng(4)
    truth = rng.dirichlet(np.ones(5), size=(6, 3))
    n_categories = np.array([5, 5, 5])
    result = gate4.aggregate_marginals(
        truth, truth, n_categories, ["a", "b", "c"], n_bootstrap=25
    )
    assert np.isclose(result["mean_abs_gap"], 0.0)
    assert np.isclose(result["correlation"], 1.0)
    assert result["n_bootstrap"] == 25


def test_cross_tabs_multiply_out_the_same_way_on_both_sides() -> None:
    rng = np.random.default_rng(6)
    probs = rng.dirichlet(np.ones(5), size=(8, 4))
    tabs = gate4.cross_tabs(probs, probs, np.array([5, 5, 5, 5]))
    assert tabs["n_pairs"] == 6  # 4 choose 2
    assert tabs["n_table_cells"] == 6 * 25
    assert np.isclose(tabs["rmse"], 0.0)
    assert np.isclose(tabs["correlation"], 1.0)


def test_cross_tab_widths_follow_the_item_types() -> None:
    rng = np.random.default_rng(8)
    probs = np.zeros((5, 2, 5))
    probs[:, 0] = rng.dirichlet(np.ones(5), size=5)
    probs[:, 1, :2] = rng.dirichlet(np.ones(2), size=5)
    tabs = gate4.cross_tabs(probs, probs, np.array([5, 2]))
    assert tabs["n_pairs"] == 1
    assert tabs["n_table_cells"] == 10  # 5 x 2


# --------------------------------------------------------------------------
# the produced report, when it is present
# --------------------------------------------------------------------------

REPORT = Path(__file__).resolve().parents[1] / "results" / "stage4_gate4.json"
CELLS = Path(__file__).resolve().parents[1] / "results" / "stage4_cells.npz"


@pytest.mark.skipif(not REPORT.is_file(), reason="Gate 4 has not been graded here")
def test_the_report_carries_the_house_style_verdict_block() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    verdict = report["verdict"]
    assert verdict["gate"].startswith("Gate 4")
    assert set(verdict["bars"]) == {"primary_lift", "ece", "correlation"}
    for name, bar in verdict["bars"].items():
        assert set(bar) >= {"statement", "value", "bar", "pass"}, name
        assert isinstance(bar["pass"], bool), name
    assert verdict["all_bars_pass"] == (not verdict["failed_bars"])
    assert set(verdict["failed_bars"]) <= set(verdict["bars"])

    # the flag exists whatever it says, and carries its own instruction
    assert isinstance(report["leakage"]["leakage_hunt_required"], bool)
    assert "NEVER cleared" in report["leakage"]["instruction"]

    # PREREGISTRATION section 6 asks for the k-NN comparison prominently
    assert "knn_proximity" in verdict
    assert "per_stratum_model_beats_knn" in verdict["knn_proximity"]


@pytest.mark.skipif(not REPORT.is_file(), reason="Gate 4 has not been graded here")
def test_the_report_covers_every_stratum_and_every_arm() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert set(report["per_stratum_comparison"]) == {"near", "same-domain", "far"}
    for name in ("model", "marginal", "profile", "knn", "oracle"):
        arm = report["arms"][name]
        assert set(arm["per_stratum"]) == {"near", "same-domain", "far"}, name
        assert arm["brier"] >= 0.0
    assert set(report["exploratory"]) == {
        "error_decomposition_2x2",
        "oracle_ceiling_and_noise_floor",
        "posterior_vs_plug_in",
    }


@pytest.mark.skipif(not REPORT.is_file(), reason="Gate 4 has not been graded here")
def test_the_grader_checked_that_the_arms_line_up_with_the_truth_store() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    alignment = report["alignment"]
    for key, value in alignment.items():
        if key == "checksums_verified":
            assert all(value.values())
        else:
            assert value is True, key


@pytest.mark.skipif(not CELLS.is_file(), reason="Gate 4 has not been graded here")
def test_the_cell_arrays_match_the_reported_means() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    archive = np.load(CELLS, allow_pickle=False)
    for name in ("model", "marginal", "knn"):
        assert np.isclose(archive[f"brier_{name}"].mean(), report["arms"][name]["brier"])
        assert np.isclose(
            archive[f"log_loss_{name}"].mean(), report["arms"][name]["log_loss"]
        )
    assert np.all(archive["truth_counts"].sum(axis=2) == 50)
