"""Tests for the Stage 4 predictor.

The load-bearing one is :func:`test_probabilities_agree_with_mirt`: the
predictor restates ``src.model.mirt``'s response function as a forward pass, and
if the two ever drift the whole stage is measuring one model with another
model's parameters.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.model import mirt
from src.model import predictor as P


def random_block(rng: np.random.Generator, n_items: int = 7, dims: int = 4) -> P.ItemBlock:
    kinds = [mirt.LIKERT if i % 2 == 0 else mirt.BINARY for i in range(n_items)]
    is_likert = np.array([k == mirt.LIKERT for k in kinds])
    base = rng.normal(0.0, 1.0, size=n_items)
    gaps = rng.uniform(0.15, 1.4, size=(n_items, 3))
    thresholds = base[:, None] + np.concatenate(
        [np.zeros((n_items, 1)), np.cumsum(gaps, axis=1)], axis=1
    )
    return P.ItemBlock(
        item_ids=[f"q{i:03d}" for i in range(n_items)],
        kinds=kinds,
        loadings=rng.normal(0.0, 0.9, size=(n_items, dims)),
        thresholds=np.where(is_likert[:, None], thresholds, np.nan),
        difficulty=np.where(is_likert, np.nan, base),
    )


# --------------------------------------------------------------------------
# the agreement that matters
# --------------------------------------------------------------------------


def test_probabilities_agree_with_mirt() -> None:
    """Every category probability equals the one implied by mirt's likelihood.

    ``mirt`` never exposes a response function -- it only ever evaluates the
    log-likelihood of an observed answer. So the check goes through that: pin a
    cell to each category in turn, read the log-likelihood mirt assigns, and
    exponentiate. Same parameters, same numbers, or the test fails.
    """
    rng = np.random.default_rng(11)
    block = random_block(rng, n_items=9, dims=5)
    theta = rng.normal(0.0, 1.0, size=(6, 5))

    mine = P.category_probabilities(theta, block)
    params = block.to_mirt_params()

    for row in range(theta.shape[0]):
        for col in range(block.n_items):
            width = int(block.n_categories()[col])
            for category in range(width):
                codes = np.full((1, block.n_items), -1, dtype=np.int16)
                codes[0, col] = category
                data = mirt.Dataset(
                    persona_ids=["x"],
                    item_ids=list(block.item_ids),
                    item_kinds=list(block.kinds),
                    codes=codes,
                )
                loglik, _, _, _ = mirt._loglik_and_grads(
                    theta[row : row + 1], params, mirt.make_blocks(data),
                    want_item_grads=False,
                )
                assert np.isclose(float(np.exp(loglik)), mine[row, col, category], atol=1e-10)


def test_probabilities_are_distributions_over_the_right_width() -> None:
    rng = np.random.default_rng(3)
    block = random_block(rng, n_items=11, dims=6)
    probs = P.category_probabilities(rng.normal(size=(20, 6)), block)

    assert probs.shape == (20, 11, 5)
    assert (probs >= 0).all()
    assert np.allclose(probs.sum(axis=2), 1.0)
    binary = ~block.is_likert
    assert np.all(probs[:, binary, 2:] == 0.0)


def test_cut_raw_round_trips_through_the_mirt_parameterization() -> None:
    """``cut_raw -> thresholds -> cut_raw`` must be the identity."""
    rng = np.random.default_rng(5)
    block = random_block(rng, n_items=8, dims=3)
    params = block.to_mirt_params()
    recovered = np.where(
        block.is_likert[:, None], params.thresholds(), np.nan
    )
    assert np.allclose(
        recovered[block.is_likert], block.thresholds[block.is_likert], atol=1e-6
    )
    assert np.allclose(
        params.difficulty()[~block.is_likert], block.difficulty[~block.is_likert]
    )
    assert np.all(np.diff(params.thresholds()[block.is_likert], axis=1) > 0)


# --------------------------------------------------------------------------
# fitting the cut points
# --------------------------------------------------------------------------


def test_fitting_cuts_recovers_planted_ones() -> None:
    """With theta and the loadings known, the thresholds are identifiable."""
    rng = np.random.default_rng(17)
    block = random_block(rng, n_items=6, dims=3)
    theta = rng.normal(0.0, 1.0, size=(4000, 3))

    probs = P.category_probabilities(theta, block)
    codes = np.zeros((theta.shape[0], block.n_items), dtype=np.int16)
    for col in range(block.n_items):
        width = int(block.n_categories()[col])
        for row in range(theta.shape[0]):
            codes[row, col] = rng.choice(width, p=probs[row, col, :width])

    fitted = P.fit_cuts(codes, theta, block)
    assert fitted.converged
    assert fitted.max_abs_gradient < 1.0
    likert = block.is_likert
    assert np.max(np.abs(fitted.thresholds[likert] - block.thresholds[likert])) < 0.25
    assert np.max(np.abs(fitted.difficulty[~likert] - block.difficulty[~likert])) < 0.25


def test_fitting_cuts_leaves_the_loadings_alone() -> None:
    rng = np.random.default_rng(23)
    block = random_block(rng, n_items=5, dims=3)
    theta = rng.normal(size=(200, 3))
    codes = np.zeros((200, 5), dtype=np.int16)
    fitted = P.fit_cuts(codes, theta, block, max_iters=50)
    assert fitted.cuts.shape == (5, 4)
    # binary rows carry only a difficulty; the other three columns are unused
    assert np.all(fitted.cuts[~block.is_likert, 1:] == 0.0)


# --------------------------------------------------------------------------
# integrating over the posterior
# --------------------------------------------------------------------------


def test_calibrated_covariance_scales_the_diagonal_by_kappa_squared() -> None:
    rng = np.random.default_rng(7)
    a = rng.normal(size=(3, 4, 4))
    cov = np.einsum("nij,nkj->nik", a, a)
    kappa = np.array([1.1, 0.8, 1.0, 1.3])
    scaled = P.calibrated_covariance(cov, kappa)
    assert np.allclose(
        np.diagonal(scaled, axis1=1, axis2=2),
        np.diagonal(cov, axis1=1, axis2=2) * kappa[None, :] ** 2,
    )
    assert np.allclose(scaled, np.transpose(scaled, (0, 2, 1)))


def test_posterior_integration_is_seeded_and_reproducible() -> None:
    rng = np.random.default_rng(31)
    block = random_block(rng, n_items=6, dims=4)
    theta = rng.normal(size=(5, 4))
    a = rng.normal(size=(5, 4, 4)) * 0.3
    cov = np.einsum("nij,nkj->nik", a, a) + np.eye(4)[None, :, :] * 0.05

    first = P.predict_posterior(theta, cov, block, n_draws=256, seed=99)
    second = P.predict_posterior(theta, cov, block, n_draws=256, seed=99)
    other = P.predict_posterior(theta, cov, block, n_draws=256, seed=100)

    assert np.array_equal(first, second)
    assert not np.allclose(first, other)
    assert np.allclose(first.sum(axis=2), 1.0)


def test_a_vanishing_posterior_collapses_onto_the_plug_in() -> None:
    rng = np.random.default_rng(41)
    block = random_block(rng, n_items=6, dims=4)
    theta = rng.normal(size=(4, 4))
    cov = np.zeros((4, 4, 4))
    integrated = P.predict_posterior(theta, cov, block, n_draws=64, seed=1)
    assert np.allclose(integrated, P.predict_plug_in(theta, block), atol=1e-7)


def test_integrating_pulls_probabilities_toward_the_middle() -> None:
    """Averaging a bounded S-curve over uncertainty cannot sharpen it."""
    rng = np.random.default_rng(53)
    block = random_block(rng, n_items=10, dims=4)
    theta = rng.normal(size=(30, 4)) * 1.5
    cov = np.repeat(np.eye(4)[None, :, :], 30, axis=0) * 0.6

    plug_in = P.predict_plug_in(theta, block)
    integrated = P.predict_posterior(theta, cov, block, n_draws=1024, seed=4)
    assert integrated.max(axis=2).mean() < plug_in.max(axis=2).mean()


# --------------------------------------------------------------------------
# the built artifact, when it is present
# --------------------------------------------------------------------------

PREDICTIONS = P.REPO_ROOT / "results" / "stage4_predictions.npz"


@pytest.mark.skipif(not PREDICTIONS.is_file(), reason="Stage 4 predictions not built here")
def test_the_built_predictions_are_distributions() -> None:
    archive = np.load(PREDICTIONS, allow_pickle=False)
    n_categories = np.asarray(archive["n_categories"], dtype=int)
    for name in [k for k in archive.files if k.startswith("probs_")]:
        probs = archive[name]
        assert probs.shape == (100, 50, 5), name
        assert (probs >= 0).all(), name
        assert np.allclose(probs.sum(axis=2), 1.0), name
        for col, width in enumerate(n_categories):
            assert np.all(probs[:, col, int(width):] == 0.0), (name, col)


@pytest.mark.skipif(not PREDICTIONS.is_file(), reason="Stage 4 predictions not built here")
def test_the_confirmatory_arm_uses_the_zero_shot_loadings() -> None:
    """The held-out items' fitted loadings must never reach the confirmatory arm."""
    archive = np.load(PREDICTIONS, allow_pickle=False)
    fit = np.load(P.REPO_ROOT / "results" / "stage2_v2_fit.npz", allow_pickle=False)
    encoded = np.load(
        P.REPO_ROOT / "data" / "runs" / "stage3_qwen" / "item_encoder_predictions.npz",
        allow_pickle=False,
    )
    assert np.allclose(archive["encoder_loadings"], encoded["predicted_loadings"])
    assert not np.allclose(archive["encoder_loadings"], fit["holdout_loadings"])
