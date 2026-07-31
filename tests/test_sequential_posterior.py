"""Tests for the Stage 5 sequential posterior.

The one that matters most is :func:`test_scripted_replay_reproduces_the_backbone`:
if updating the posterior one answer at a time does not land exactly where Stage
3's batch scoring landed, every Stage 5 number is measured against a different
model than the one that defines the Gate 5 target, and the comparison is void.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.model import mirt, person_encoder as pe
from src.model import sequential_posterior as sp

REPO_ROOT = Path(__file__).resolve().parents[1]
FIT = REPO_ROOT / "results" / "stage2_v2_fit.npz"
BACKBONE = REPO_ROOT / "results" / "stage3_person_backbone.npz"
TRANSCRIPTS = REPO_ROOT / "data" / "runs" / "interview_v1" / "transcripts.jsonl"

#: The integrity replay has to be tight, not merely close. Both sides run the
#: Newton search to a gradient of 1e-10 and the two Hessian recipes differ by
#: about 1e-10 on the blur, so anything above this is a real disagreement.
INTEGRITY_ATOL = 1e-8


# --------------------------------------------------------------------------
# small synthetic fixtures
# --------------------------------------------------------------------------


def toy_params(seed: int = 0, n_items: int = 6, dims: int = 3) -> mirt.ItemParams:
    rng = np.random.default_rng(seed)
    kinds = [mirt.LIKERT if j % 2 == 0 else mirt.BINARY for j in range(n_items)]
    cut_raw = np.zeros((n_items, mirt.LIKERT_CUTS))
    cut_raw[:, 0] = rng.normal(0.0, 0.6, size=n_items)
    cut_raw[:, 1:] = rng.normal(0.0, 0.3, size=(n_items, mirt.LIKERT_CUTS - 1))
    return mirt.ItemParams(
        loadings=rng.normal(0.0, 0.9, size=(n_items, dims)),
        cut_raw=cut_raw,
        kinds=kinds,
    )


def toy_codes(params: mirt.ItemParams, n_personas: int = 7, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    codes = np.zeros((n_personas, len(params.kinds)), dtype=np.int16)
    for j, kind in enumerate(params.kinds):
        top = mirt.LIKERT_CATEGORIES if kind == mirt.LIKERT else 2
        codes[:, j] = rng.integers(0, top, size=n_personas)
    return codes


# --------------------------------------------------------------------------
# the solver is the Stage 3 solver
# --------------------------------------------------------------------------


def test_cold_solve_matches_the_stage3_posterior_exactly():
    params = toy_params()
    codes = toy_codes(params)
    house = pe.posterior(codes, params)
    mine = sp.solve(codes, params, hessian="central")
    assert np.allclose(mine.theta, house.theta, atol=1e-12, rtol=0)
    assert np.allclose(mine.blur, house.blur, atol=1e-12, rtol=0)
    assert np.allclose(mine.cov, house.cov, atol=1e-12, rtol=0)


def test_analytic_and_central_hessians_agree():
    params = toy_params(seed=3)
    codes = toy_codes(params, seed=4)
    theta = np.random.default_rng(5).normal(size=(codes.shape[0], params.loadings.shape[1]))
    blocks = pe.blocks_from_codes(codes, params.kinds)
    analytic = sp.analytic_hessian(theta, params, codes)
    central = pe.observed_hessian(theta, params, blocks)
    assert np.abs(analytic - central).max() < 1e-6
    blur_a = np.sqrt(np.diagonal(np.linalg.inv(analytic), axis1=1, axis2=2))
    blur_c = np.sqrt(np.diagonal(np.linalg.inv(central), axis1=1, axis2=2))
    assert np.abs(blur_a - blur_c).max() < 1e-7


def test_unasked_columns_change_nothing():
    """A masked wide matrix scores exactly like the narrow one it stands for."""
    params = toy_params(seed=6, n_items=8)
    codes = toy_codes(params, seed=7)
    wide = codes.copy()
    wide[:, 3:] = sp.UNASKED
    narrow_params = pe.prefix_params(params, range(3))
    wide_post = sp.solve(wide, params, hessian="central")
    narrow_post = pe.posterior(codes[:, :3], narrow_params)
    assert np.allclose(wide_post.theta, narrow_post.theta, atol=1e-10, rtol=0)
    assert np.allclose(wide_post.blur, narrow_post.blur, atol=1e-10, rtol=0)


# --------------------------------------------------------------------------
# incremental == batch
# --------------------------------------------------------------------------


@pytest.mark.parametrize("hessian", ["analytic", "central"])
def test_incremental_update_equals_a_batch_refit(hessian):
    """The whole premise: answers one at a time land where all-at-once lands."""
    params = toy_params(seed=8, n_items=6, dims=3)
    truth = toy_codes(params, n_personas=5, seed=9)
    n_personas = truth.shape[0]
    rng = np.random.default_rng(10)
    order = np.array([rng.permutation(len(params.kinds)) for _ in range(n_personas)])

    state = sp.SequentialPosterior(params, n_personas, hessian=hessian)
    rows = np.arange(n_personas)
    running = np.full_like(truth, sp.UNASKED)
    for step in range(len(params.kinds)):
        columns = order[:, step]
        answers = truth[rows, columns]
        state.observe(columns, answers)
        running[rows, columns] = answers

        batch = sp.solve(running, params, hessian=hessian)
        assert np.allclose(state.theta, batch.theta, atol=1e-9, rtol=0)
        assert np.allclose(state.blur, batch.blur, atol=1e-9, rtol=0)
        assert np.allclose(state.cov, batch.cov, atol=1e-9, rtol=0)


def test_different_personas_may_hold_different_item_sets():
    params = toy_params(seed=11, n_items=5, dims=2)
    truth = toy_codes(params, n_personas=3, seed=12)
    state = sp.SequentialPosterior(params, 3)
    state.observe(np.array([0, 1, 2]), truth[np.arange(3), [0, 1, 2]])
    state.observe(np.array([3, 3, 4]), truth[np.arange(3), [3, 3, 4]])
    assert state.n_asked.tolist() == [2, 2, 2]
    assert state.answered[0].tolist() == [True, False, False, True, False]
    assert state.answered[2].tolist() == [False, False, True, False, True]


def test_observe_refuses_to_ask_the_same_item_twice():
    params = toy_params(seed=13, n_items=4, dims=2)
    state = sp.SequentialPosterior(params, 2)
    state.observe(np.array([0, 1]), np.array([1, 1]))
    with pytest.raises(ValueError, match="twice"):
        state.observe(np.array([0, 2]), np.array([2, 0]))


def test_observe_checks_shapes_and_ranges():
    params = toy_params(seed=14, n_items=4, dims=2)
    state = sp.SequentialPosterior(params, 2)
    with pytest.raises(ValueError, match="one column and one answer"):
        state.observe(np.array([0]), np.array([1]))
    with pytest.raises(ValueError, match="out of range"):
        state.observe(np.array([0, 9]), np.array([1, 1]))
    with pytest.raises(ValueError, match="not-asked marker"):
        state.observe(np.array([0, 1]), np.array([1, -1]))


def test_reset_returns_to_the_prior():
    params = toy_params(seed=15, n_items=4, dims=2)
    state = sp.SequentialPosterior(params, 3)
    prior_blur = state.blur.copy()
    state.observe(np.array([0, 0, 0]), np.array([1, 2, 3]))
    assert not np.allclose(state.blur, prior_blur)
    state.reset()
    assert np.allclose(state.blur, prior_blur)
    assert state.n_asked.tolist() == [0, 0, 0]
    assert np.allclose(state.theta, 0.0)


def test_the_state_handed_to_a_strategy_is_read_only():
    params = toy_params(seed=16, n_items=4, dims=2)
    state = sp.SequentialPosterior(params, 2).state()
    for array in (state.theta, state.cov, state.blur, state.answered, state.n_asked):
        with pytest.raises(ValueError):
            array[...] = 0


# --------------------------------------------------------------------------
# the information heuristic's arithmetic
# --------------------------------------------------------------------------


def brute_force_expected_reduction(
    theta: np.ndarray, cov: np.ndarray, params: mirt.ItemParams
) -> np.ndarray:
    """The heuristic's definition, written out with explicit matrix inverses.

    Independent of the module under test: the answer probabilities come from
    the response model directly and the curvature from a numerical second
    derivative of the negative log probability in ``z``.
    """
    n, dims = theta.shape
    n_items = params.loadings.shape[0]
    out = np.zeros((n, n_items))
    taus = params.thresholds()
    diff = params.difficulty()
    step = 1e-5

    def prob(item: int, category: int, z: float) -> float:
        if params.kinds[item] == mirt.LIKERT:
            cuts = taus[item]
            lo = 1.0 if category == 0 else float(mirt.sigmoid(z - cuts[category - 1]))
            hi = 0.0 if category == 4 else float(mirt.sigmoid(z - cuts[category]))
            return lo - hi
        p = float(mirt.sigmoid(z - diff[item]))
        return p if category == 1 else 1.0 - p

    for row in range(n):
        precision = np.linalg.inv(cov[row])
        base = float(np.trace(cov[row]))
        for item in range(n_items):
            a = params.loadings[item]
            z = float(theta[row] @ a)
            categories = range(5) if params.kinds[item] == mirt.LIKERT else range(2)
            total = 0.0
            for category in categories:
                p = prob(item, category, z)
                curvature = -(
                    np.log(prob(item, category, z + step))
                    - 2.0 * np.log(p)
                    + np.log(prob(item, category, z - step))
                ) / step**2
                after = np.linalg.inv(precision + curvature * np.outer(a, a))
                total += p * (base - float(np.trace(after)))
            out[row, item] = total
    return out


def test_expected_variance_reduction_matches_an_independent_implementation():
    params = toy_params(seed=17, n_items=5, dims=3)
    codes = toy_codes(params, n_personas=4, seed=18)
    codes[:, 2:] = sp.UNASKED
    post = sp.solve(codes, params)
    fast = sp.expected_variance_reduction(post.theta, post.cov, params)
    slow = brute_force_expected_reduction(post.theta, post.cov, params)
    assert np.abs(fast - slow).max() < 1e-5


def test_the_heuristic_picks_the_analytically_best_of_two_items():
    """Two binary items, one twice as discriminating. The answer is not a guess.

    At the prior the posterior covariance is the identity, so asking item i
    reduces the trace by ``c ||a||^2 / (1 + c ||a||^2)`` with ``c = p(1 - p) =
    0.25`` at zero difficulty. Item 0 has ``||a||^2 = 1`` and scores 0.2; item 1
    has ``||a||^2 = 4`` and scores 0.5. Both numbers are checked, not just the
    ordering.
    """
    params = mirt.ItemParams(
        loadings=np.array([[1.0, 0.0], [0.0, 2.0]]),
        cut_raw=np.zeros((2, mirt.LIKERT_CUTS)),
        kinds=[mirt.BINARY, mirt.BINARY],
    )
    theta = np.zeros((1, 2))
    cov = np.eye(2)[None, :, :]
    score = sp.expected_variance_reduction(theta, cov, params)
    assert score.shape == (1, 2)
    assert score[0, 0] == pytest.approx(0.25 * 1.0 / (1.0 + 0.25 * 1.0))
    assert score[0, 1] == pytest.approx(0.25 * 4.0 / (1.0 + 0.25 * 4.0))
    assert int(np.argmax(score[0])) == 1


def test_answer_probabilities_are_a_distribution():
    params = toy_params(seed=19, n_items=6, dims=3)
    theta = np.random.default_rng(20).normal(size=(4, 3))
    probs, curv = sp.answer_probabilities(theta, params)
    for j, kind in enumerate(params.kinds):
        width = 5 if kind == mirt.LIKERT else 2
        assert np.allclose(probs[:, j, :width].sum(axis=1), 1.0)
        assert np.allclose(probs[:, j, width:], 0.0)
        assert (curv[:, j, :width] > 0).all()


def test_curvature_is_the_hessian_of_the_answer_that_was_given():
    """``item_curvature`` and ``answer_probabilities`` have to agree cell by cell."""
    params = toy_params(seed=21, n_items=5, dims=2)
    codes = toy_codes(params, n_personas=3, seed=22)
    theta = np.random.default_rng(23).normal(size=(3, 2))
    given = sp.item_curvature(theta, params, codes)
    _, table = sp.answer_probabilities(theta, params)
    picked = np.take_along_axis(table, codes.astype(np.int64)[:, :, None], axis=2)[:, :, 0]
    assert np.allclose(given, picked)


def test_curvature_is_zero_for_items_that_were_not_asked():
    params = toy_params(seed=24, n_items=4, dims=2)
    codes = np.full((3, 4), sp.UNASKED, dtype=np.int16)
    codes[:, 0] = 1
    theta = np.zeros((3, 2))
    curvature = sp.item_curvature(theta, params, codes)
    assert (curvature[:, 1:] == 0).all()
    assert (curvature[:, 0] > 0).all()


# --------------------------------------------------------------------------
# THE INTEGRITY CHECK
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    not (FIT.is_file() and BACKBONE.is_file() and TRANSCRIPTS.is_file()),
    reason="needs the Stage 2 fit, the Stage 3 backbone and the Stage 3 transcripts",
)
@pytest.mark.parametrize("hessian", ["analytic", "central"])
def test_scripted_replay_reproduces_the_backbone(hessian):
    """Replaying Stage 3's 15 scripted questions must land on Stage 3's numbers.

    Same personas, same answers, same fitted item parameters, same order --
    only the arithmetic is sequential instead of batched. Every N is checked,
    not just the last one, and both Hessian recipes are checked, because the
    sweep runs on the analytic one.
    """
    fit = np.load(FIT, allow_pickle=False)
    backbone = np.load(BACKBONE, allow_pickle=False)
    params, item_ids = pe.item_params_from_fit(fit, block="train")
    column_of = {iid: j for j, iid in enumerate(item_ids)}

    scripted = [str(i) for i in backbone["scripted_item_ids"]]
    columns = [column_of[i] for i in scripted]
    holdout = [str(p) for p in backbone["persona_holdout_ids"]]

    transcripts = pe.load_transcripts(TRANSCRIPTS)
    assert transcripts.item_ids == scripted
    kinds = [params.kinds[c] for c in columns]
    codes = pe.code_matrix(transcripts, holdout, kinds)

    theta_reference = np.asarray(backbone["theta_holdout_by_n"], dtype=float)
    blur_reference = np.asarray(backbone["blur_holdout_by_n"], dtype=float)

    state = sp.SequentialPosterior(params, len(holdout), hessian=hessian)
    for step in range(len(scripted)):
        state.observe(np.full(len(holdout), columns[step]), codes[:, step])
        assert np.abs(state.theta - theta_reference[step]).max() < INTEGRITY_ATOL, (
            f"theta drifted from the Stage 3 backbone at N = {step + 1}"
        )
        assert np.abs(state.blur - blur_reference[step]).max() < INTEGRITY_ATOL, (
            f"blur drifted from the Stage 3 backbone at N = {step + 1}"
        )


@pytest.mark.skipif(
    not (FIT.is_file() and BACKBONE.is_file()),
    reason="needs the Stage 2 fit and the Stage 3 backbone",
)
def test_the_scripted_items_are_all_training_items():
    """The candidate bank Stage 5 chooses from has to contain the Stage 3 script."""
    fit = np.load(FIT, allow_pickle=False)
    backbone = np.load(BACKBONE, allow_pickle=False)
    training = {str(i) for i in fit["item_train_ids"]}
    scripted = {str(i) for i in backbone["scripted_item_ids"]}
    assert scripted <= training
    holdout = {str(i) for i in fit["item_holdout_ids"]}
    assert not (scripted & holdout)


@pytest.mark.skipif(
    not (REPO_ROOT / "experiments" / "splits_v1.json").is_file(),
    reason="needs the frozen split manifest",
)
def test_the_scripted_order_is_the_frozen_one():
    splits = json.loads(
        (REPO_ROOT / "experiments" / "splits_v1.json").read_text(encoding="utf-8")
    )
    scripted = [str(i) for i in splits["interview_closed"]]
    assert len(scripted) == 15
    assert len(set(scripted)) == 15
