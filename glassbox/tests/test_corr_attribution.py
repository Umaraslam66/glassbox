"""Tests for the EXPLORATORY Gate 4 correlation attribution.

The pooled numbers themselves are the grader's and are checked against
``results/stage4_gate4.json`` at run time; what is tested here is the new
machinery: the variance split, the OLS line, the entropies, the two within-item
lenses, the sharpness fraction, and the telescoping of the three losses.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.eval import corr_attribution as ca


# --------------------------------------------------------------------------
# the variance split
# --------------------------------------------------------------------------


def test_variance_split_is_exact_and_adds_up() -> None:
    rng = np.random.default_rng(0)
    truth = rng.random((30, 4, 5))
    truth[:, :2, 2:] = 0.0  # two binary items
    n_categories = np.array([2, 2, 5, 5])

    split = ca.variance_split(truth, n_categories)
    assert split["n_slots"] == 2 + 2 + 5 + 5
    assert split["n_personas_per_slot"] == 30
    assert split["between_item_variance"] + split["within_item_between_persona_variance"] == pytest.approx(
        split["total_variance"], abs=1e-12
    )
    assert split["between_item_share"] + split["within_item_share"] == pytest.approx(1.0)


def test_variance_split_is_all_between_when_nobody_differs() -> None:
    """Identical people: every slot is a constant, so no within-slot variance."""
    one_person = np.array([[[0.1, 0.9, 0.0, 0.0, 0.0], [0.2, 0.2, 0.2, 0.2, 0.2]]])
    truth = np.repeat(one_person, 12, axis=0)
    split = ca.variance_split(truth, np.array([2, 5]))
    assert split["within_item_share"] == pytest.approx(0.0, abs=1e-15)
    assert split["between_item_share"] == pytest.approx(1.0)
    assert split["best_item_level_only_correlation"] == pytest.approx(1.0)


def test_variance_split_is_all_within_when_every_slot_has_the_same_mean() -> None:
    truth = np.zeros((2, 1, 5))
    truth[0, 0, :2] = [0.0, 1.0]
    truth[1, 0, :2] = [1.0, 0.0]
    split = ca.variance_split(truth, np.array([2]))
    assert split["between_item_share"] == pytest.approx(0.0, abs=1e-15)
    assert split["within_item_share"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# the OLS line
# --------------------------------------------------------------------------


def test_ols_line_recovers_a_planted_line() -> None:
    x = np.linspace(0.0, 1.0, 50)
    y = 0.4 * x + 0.1
    line = ca.ols_line(x, y)
    assert line["slope"] == pytest.approx(0.4)
    assert line["intercept"] == pytest.approx(0.1)
    assert line["r"] == pytest.approx(1.0)


def test_ols_slope_equals_r_times_the_std_ratio() -> None:
    """The identity the flatness block leans on: slope = r * sd(y) / sd(x)."""
    rng = np.random.default_rng(7)
    x = rng.random(400)
    y = 0.3 * x + 0.5 * rng.random(400)
    line = ca.ols_line(x, y)
    assert line["slope"] == pytest.approx(line["r"] * y.std() / x.std())


# --------------------------------------------------------------------------
# entropy and sharpness
# --------------------------------------------------------------------------


def test_entropy_uses_the_items_own_categories() -> None:
    probs = np.zeros((1, 2, 5))
    probs[0, 0, :2] = [0.5, 0.5]  # uniform binary
    probs[0, 1] = 0.2  # uniform Likert-5
    values = ca.entropy_cells(probs, np.array([2, 5]))
    assert values[0, 0] == pytest.approx(np.log(2.0))
    assert values[0, 1] == pytest.approx(np.log(5.0))


def test_entropy_of_a_certain_cell_is_zero_and_never_nan() -> None:
    probs = np.zeros((1, 1, 5))
    probs[0, 0, 0] = 1.0
    value = ca.entropy_cells(probs, np.array([5]))[0, 0]
    assert np.isfinite(value)
    assert value == pytest.approx(0.0)


def test_sharp_fraction_counts_both_ends() -> None:
    values = np.array([0.0, 0.02, 0.5, 0.96, 1.0])
    sharp = ca.sharp_fraction(values)
    assert sharp["at_or_below_0.05"] == pytest.approx(2 / 5)
    assert sharp["at_or_above_0.95"] == pytest.approx(2 / 5)
    assert sharp["either_end"] == pytest.approx(4 / 5)


# --------------------------------------------------------------------------
# the within-item lenses
# --------------------------------------------------------------------------


def test_per_item_and_per_slot_correlations_have_the_right_shapes() -> None:
    rng = np.random.default_rng(3)
    truth = rng.random((25, 3, 5))
    predicted = truth + 0.05 * rng.standard_normal((25, 3, 5))
    n_categories = np.array([2, 5, 5])

    per_item = ca.per_item_correlations(predicted, truth, n_categories)
    per_slot = ca.per_slot_correlations(predicted, truth, n_categories)
    assert per_item.shape == (3,)
    assert per_slot.shape == (12,)
    assert np.all(per_item > 0.8)


def test_a_predictor_that_ignores_the_person_has_no_within_item_correlation() -> None:
    """The population marginal is constant down a slot, so a slot r is undefined."""
    rng = np.random.default_rng(11)
    truth = rng.random((20, 2, 5))
    n_categories = np.array([5, 5])
    marginal = np.repeat(truth.mean(axis=0)[None, :, :], 20, axis=0)

    per_slot = ca.per_slot_correlations(marginal, truth, n_categories)
    assert np.isnan(per_slot).all()
    assert ca.summarise(per_slot)["median"] is None
    assert ca.summarise(per_slot)["n_undefined"] == 10


def test_summarise_drops_nans_and_counts_them() -> None:
    out = ca.summarise(np.array([0.1, 0.3, np.nan, 0.5]))
    assert out["n"] == 3
    assert out["n_undefined"] == 1
    assert out["median"] == pytest.approx(0.3)


# --------------------------------------------------------------------------
# the attribution
# --------------------------------------------------------------------------


def test_the_three_losses_telescope_to_the_oracle_gap() -> None:
    rungs = {
        ca.ARM_ORACLE: {"correlation": 0.99},
        ca.ARM_FULL_FITTED: {"correlation": 0.78},
        ca.ARM_FUSED_FITTED: {"correlation": 0.73},
        ca.ARM_MODEL: {"correlation": 0.68},
    }
    out = ca.attribution(rungs)
    assert out["oracle_to_model_gap"] == pytest.approx(0.31)
    assert out["terms_sum"] == pytest.approx(0.31)
    assert out["telescopes_exactly"]
    shares = [t["share_of_oracle_to_model_gap"] for t in out["terms"].values()]
    assert sum(shares) == pytest.approx(1.0)
    assert out["terms"]["model_class_8dim"]["loss"] == pytest.approx(0.21)


def test_the_ladder_rung_splits_binary_from_likert() -> None:
    rng = np.random.default_rng(5)
    truth = np.zeros((15, 2, 5))
    truth[:, 0, :2] = rng.random((15, 2))
    truth[:, 1, :] = rng.random((15, 5))
    predicted = truth.copy()
    predicted[:, 1, :] = rng.random((15, 5))  # Likert item predicted at random

    rung = ca.ladder_rung(predicted, truth, np.array([2, 5]), ["binary", "likert5"])
    assert rung["n_pairs"] == 15 * (2 + 5)
    assert rung["binary_only"] == pytest.approx(1.0)
    assert abs(rung["likert_only"]) < 0.5


# --------------------------------------------------------------------------
# the sanity gate
# --------------------------------------------------------------------------


def _graded(correlation: float) -> dict:
    return {
        "arms": {
            "model": {
                "correlation": correlation,
                "per_item_type": {
                    "binary": {"correlation": 0.7},
                    "likert5": {"correlation": 0.6},
                },
            }
        },
        "verdict": {"bars": {"correlation": {"value": correlation}}},
    }


def test_sanity_check_passes_on_an_exact_match() -> None:
    rungs = {"model": {"correlation": 0.684, "binary_only": 0.7, "likert_only": 0.6}}
    out = ca.sanity_check(rungs, _graded(0.684))
    assert out["all_match"]
    assert out["n_checks"] == 4


def test_sanity_check_catches_a_drifted_number() -> None:
    rungs = {"model": {"correlation": 0.684 + 1e-6, "binary_only": 0.7, "likert_only": 0.6}}
    out = ca.sanity_check(rungs, _graded(0.684))
    assert not out["all_match"]
    assert {f["quantity"] for f in out["failures"]} == {
        "correlation",
        "the frozen bar value itself",
    }
