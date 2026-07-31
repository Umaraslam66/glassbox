"""Tests for the Stage 4 baselines: smoothing, distance, and the k-NN selection."""

from __future__ import annotations

import numpy as np
import pytest

from src.model import mirt
from src.model import stage4_baselines as B


def test_smoothing_adds_half_and_normalizes_over_the_items_own_width() -> None:
    counts = np.array([[3.0, 1.0, 0.0, 0.0, 0.0], [7.0, 3.0, 0.0, 0.0, 0.0]])
    n_categories = np.array([5, 2])
    out = B.smoothed_distribution(counts, n_categories)

    # Likert row: (3.5, 1.5, 0.5, 0.5, 0.5) / 6.5
    assert np.allclose(out[0], np.array([3.5, 1.5, 0.5, 0.5, 0.5]) / 6.5)
    # binary row: only two live columns, (7.5, 3.5) / 11
    assert np.allclose(out[1, :2], np.array([7.5, 3.5]) / 11.0)
    assert np.all(out[1, 2:] == 0.0)
    assert np.allclose(out.sum(axis=1), 1.0)


def test_the_marginal_is_the_training_populations_own_answer_mix() -> None:
    codes = np.array([[0, 1], [0, 0], [4, 1], [2, 1]], dtype=np.int16)
    n_categories = np.array([5, 2])
    marginal = B.population_marginal(codes, n_categories)
    # item 0: counts 2,0,1,0,1 -> +0.5 each, over 4 + 2.5
    assert np.allclose(marginal[0], np.array([2.5, 0.5, 1.5, 0.5, 1.5]) / 6.5)
    # item 1: no 1, yes 3
    assert np.allclose(marginal[1, :2], np.array([1.5, 3.5]) / 5.0)


def test_brier_averages_over_the_items_own_categories() -> None:
    predicted = np.zeros((1, 2, 5))
    predicted[0, 0] = [0.2, 0.2, 0.2, 0.2, 0.2]
    predicted[0, 1, :2] = [0.5, 0.5]
    target = np.zeros((1, 2, 5))
    target[0, 0, 0] = 1.0
    target[0, 1, 1] = 1.0
    n_categories = np.array([5, 2])
    scores = B.brier(predicted, target, n_categories)
    assert np.isclose(scores[0, 0], (0.8**2 + 4 * 0.2**2) / 5)
    assert np.isclose(scores[0, 1], (0.5**2 + 0.5**2) / 2)


def test_disagreement_is_bounded_by_one_per_item() -> None:
    kinds = [mirt.LIKERT, mirt.BINARY, mirt.LIKERT]
    a = np.array([[0, 0, 4]], dtype=np.int16)  # extremes
    b = np.array([[4, 1, 0], [0, 0, 4]], dtype=np.int16)
    distances = B.disagreement_matrix(a, b, kinds)
    assert np.isclose(distances[0, 0], 1.0 + 1.0 + 1.0)  # every item maximally apart
    assert np.isclose(distances[0, 1], 0.0)  # identical

    mid = np.array([[2, 0, 4]], dtype=np.int16)
    assert np.isclose(B.disagreement_matrix(a, mid, kinds)[0, 0], 2 / 4)


def test_knn_prediction_is_the_smoothed_neighbourhood_mix() -> None:
    """Hand-computed: three neighbours, k = 2, one Likert item."""
    distances = np.array([[0.5, 0.25, 3.0]])
    neighbour_codes = np.array([[0], [0], [4]], dtype=np.int16)  # (neighbours, items)
    n_categories = np.array([5])
    predicted = B.knn_predict(distances, neighbour_codes, n_categories, k=2)
    # nearest two are neighbours 1 and 0, both answering category 0
    assert np.allclose(predicted[0, 0], np.array([2.5, 0.5, 0.5, 0.5, 0.5]) / 4.5)

    all_three = B.knn_predict(distances, neighbour_codes, n_categories, k=3)
    assert np.allclose(all_three[0, 0], np.array([2.5, 0.5, 0.5, 0.5, 1.5]) / 5.5)


def test_knn_ties_break_deterministically() -> None:
    distances = np.array([[1.0, 1.0, 1.0, 1.0]])
    codes = np.array([[0], [1], [2], [3]], dtype=np.int16)
    first = B.knn_predict(distances, codes, np.array([5]), k=2)
    second = B.knn_predict(distances, codes, np.array([5]), k=2)
    assert np.array_equal(first, second)
    # stable argsort keeps index order, so the first two neighbours win
    assert np.allclose(first[0, 0], np.array([1.5, 1.5, 0.5, 0.5, 0.5]) / 4.5)


def test_leave_one_out_never_lets_a_persona_be_its_own_neighbour() -> None:
    """The self row is at distance zero, so it must be the one that is dropped."""
    interview = np.array([[0, 0], [0, 0], [4, 4], [4, 4]], dtype=np.int16)
    codes = np.array([[0], [1], [4], [4]], dtype=np.int16)
    kinds = [mirt.LIKERT, mirt.LIKERT]
    n_categories = np.array([5])

    distances = B.disagreement_matrix(interview, interview, kinds)
    predicted = B.knn_predict(distances, codes, n_categories, k=1, exclude_self=True)
    # persona 0's only twin is persona 1, who answered category 1
    assert int(np.argmax(predicted[0, 0])) == 1
    # persona 1's twin is persona 0, who answered category 0
    assert int(np.argmax(predicted[1, 0])) == 0

    without = B.knn_predict(distances, codes, n_categories, k=1, exclude_self=False)
    assert int(np.argmax(without[0, 0])) == 0  # itself, which is the bug being ruled out


def test_choosing_k_scores_every_grid_point_and_takes_the_best() -> None:
    rng = np.random.default_rng(2)
    interview = rng.integers(0, 5, size=(60, 6)).astype(np.int16)
    codes = rng.integers(0, 5, size=(60, 4)).astype(np.int16)
    kinds = [mirt.LIKERT] * 6
    n_categories = np.array([5, 5, 5, 5])

    k, curve = B.choose_k(interview, codes, kinds, n_categories, grid=(1, 3, 5, 10))
    assert [row["k"] for row in curve] == [1, 3, 5, 10]
    assert k == min(curve, key=lambda row: row["loo_brier"])["k"]
    # on noise, more neighbours is closer to the marginal and so scores better
    assert curve[0]["loo_brier"] > curve[-1]["loo_brier"]


def test_choosing_k_skips_grid_points_wider_than_the_pool() -> None:
    rng = np.random.default_rng(9)
    interview = rng.integers(0, 5, size=(12, 4)).astype(np.int16)
    codes = rng.integers(0, 5, size=(12, 2)).astype(np.int16)
    _, curve = B.choose_k(
        interview, codes, [mirt.LIKERT] * 4, np.array([5, 5]), grid=(1, 5, 40)
    )
    assert [row["k"] for row in curve] == [1, 5]


# --------------------------------------------------------------------------
# the built artifact, when it is present
# --------------------------------------------------------------------------

BASELINES = B.REPO_ROOT / "results" / "stage4_baselines.npz"


@pytest.mark.skipif(not BASELINES.is_file(), reason="Stage 4 baselines not built here")
def test_the_built_baselines_are_distributions() -> None:
    archive = np.load(BASELINES, allow_pickle=False)
    n_categories = np.asarray(archive["n_categories"], dtype=int)
    for name in ("probs_marginal", "probs_profile", "probs_knn"):
        probs = archive[name]
        assert probs.shape == (100, 50, 5), name
        assert (probs >= 0).all(), name
        assert np.allclose(probs.sum(axis=2), 1.0), name
        for col, width in enumerate(n_categories):
            assert np.all(probs[:, col, int(width):] == 0.0), (name, col)


@pytest.mark.skipif(not BASELINES.is_file(), reason="Stage 4 baselines not built here")
def test_the_marginal_arm_is_the_same_row_for_everybody() -> None:
    archive = np.load(BASELINES, allow_pickle=False)
    probs = archive["probs_marginal"]
    assert np.allclose(probs, probs[0][None, :, :])


@pytest.mark.skipif(not BASELINES.is_file(), reason="Stage 4 baselines not built here")
def test_the_chosen_k_is_on_the_recorded_grid() -> None:
    archive = np.load(BASELINES, allow_pickle=False)
    assert int(archive["knn_k"]) in B.K_GRID
