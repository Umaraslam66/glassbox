"""Unit tests for the open-answer fusion block (Stage 3, Gate 3).

Most of this runs on a made-up world: a small item bank with known parameters,
a population drawn from a known latent model, and judgment features generated
from that population's own theta so the measurement model has something real to
find. Nothing there depends on the project's data.

Three tests at the end do read the real artifacts, because three of the claims
this module makes are about them and cannot be checked anywhere else:

  * the closed-only track inside the fusion artifact is bit-identical to the
    backbone artifact -- adding the block must not have moved the thing it sits
    on;
  * the calibration constants are persisted for both tracks and the stored
    calibrated blur is exactly those constants applied to the stored raw blur;
  * the measurement model in the artifact is reproducible from the 400 training
    personas alone, which is the split-discipline claim.

They skip when the artifacts are not in the checkout.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.model import mirt, open_fusion as of, person_encoder as pe

from tests.test_person_encoder import make_items, sample_answers

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKBONE_NPZ = REPO_ROOT / "results" / "stage3_person_backbone.npz"
FUSED_JSON = REPO_ROOT / "results" / "stage3_person_fused.json"
FUSED_NPZ = REPO_ROOT / "results" / "stage3_person_fused.npz"
FIT_NPZ = REPO_ROOT / "results" / "stage2_v2_fit.npz"
FEATURES_NPZ = REPO_ROOT / "data" / "runs" / "stage3_qwen" / "open_answer_features.npz"


# --------------------------------------------------------------------------
# a small world with a real open-answer channel in it
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def channel_world():
    """(theta, params, codes, features) where the features really do read theta."""
    dims = 3
    params = make_items(n_likert=10, n_binary=4, dims=dims, seed=5)
    rng = np.random.default_rng(101)
    theta = rng.normal(size=(300, dims))
    codes = sample_answers(theta, params, seed=6)
    loading = rng.normal(0.0, 1.0, size=(9, dims))
    offset = rng.normal(0.0, 0.5, size=9)
    noise = rng.normal(size=(300, 9)) * 0.7
    features = theta @ loading.T + offset + noise
    return theta, params, codes, features


# --------------------------------------------------------------------------
# the pieces
# --------------------------------------------------------------------------


def test_ridge_fit_reduces_to_least_squares_at_zero_penalty() -> None:
    rng = np.random.default_rng(3)
    theta = rng.normal(size=(80, 4))
    loading = rng.normal(size=(6, 4))
    offset = rng.normal(size=6)
    features = theta @ loading.T + offset

    fitted_b, fitted_c = of.ridge_fit(features, theta, 0.0)
    assert np.allclose(fitted_b, loading, atol=1e-8)
    assert np.allclose(fitted_c, offset, atol=1e-8)


def test_ridge_shrinks_the_loading_toward_zero() -> None:
    rng = np.random.default_rng(4)
    theta = rng.normal(size=(60, 3))
    features = theta @ rng.normal(size=(5, 3)).T + rng.normal(size=(60, 5)) * 0.3

    small, _ = of.ridge_fit(features, theta, 0.01)
    large, _ = of.ridge_fit(features, theta, 1e4)
    assert np.linalg.norm(large) < 0.05 * np.linalg.norm(small)


def test_diagonal_noise_covariance_is_the_residual_second_moment() -> None:
    """Around the model's own prediction, not around the residuals' mean.

    The offset ``c`` already carries the mean, so a residual that is off-centre
    out of fold is real error and must not be subtracted away.
    """
    rng = np.random.default_rng(5)
    residuals = rng.normal(size=(500, 6)) * np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    psi = of.fit_noise_covariance(residuals, 0)
    assert np.allclose(psi, np.diag(np.diag(psi)))
    assert np.allclose(np.diag(psi), np.mean(residuals**2, axis=0), atol=1e-12)

    biased = residuals + 0.5
    assert np.all(np.diag(of.fit_noise_covariance(biased, 0)) > np.diag(psi))


def test_low_rank_noise_covariance_refuses_to_bank_correlated_noise() -> None:
    """A shared factor in the residuals must not become independent evidence.

    Six judgments that all wobble together carry roughly one measurement's worth
    of information, not six. The diagonal ``Psi`` cannot say that; the low-rank
    one can, and the difference shows up where it matters -- in ``B^T Psi^-1 B``.
    """
    rng = np.random.default_rng(6)
    shared = rng.normal(size=(4000, 1))
    residuals = shared @ np.ones((1, 6)) + rng.normal(size=(4000, 6)) * 0.3

    diagonal = of.fit_noise_covariance(residuals, 0)
    factored = of.fit_noise_covariance(residuals, 1)

    assert np.allclose(diagonal, np.diag(np.diag(diagonal)))
    off = factored - np.diag(np.diag(factored))
    assert np.max(np.abs(off)) > 0.5, "the shared factor did not reach the covariance"
    assert np.min(np.linalg.eigvalsh(factored)) > 0, "Psi must stay positive definite"

    loading = np.ones((6, 1))  # a direction aligned with the shared wobble
    information_diagonal = float((loading.T @ np.linalg.solve(diagonal, loading))[0, 0])
    information_factored = float((loading.T @ np.linalg.solve(factored, loading))[0, 0])
    assert information_factored < 0.2 * information_diagonal, (
        information_diagonal, information_factored
    )


def test_measurement_model_block_is_the_information_form() -> None:
    """``MeasurementModel.block`` is exactly ``Lambda``, ``h`` from the formulas."""
    rng = np.random.default_rng(7)
    dims, m, n = 3, 5, 11
    model = of.MeasurementModel(
        center=rng.normal(size=m),
        scale=np.abs(rng.normal(size=m)) + 0.5,
        loading=rng.normal(size=(m, dims)),
        offset=rng.normal(size=m),
        noise_cov=np.diag(np.abs(rng.normal(size=m)) + 0.3),
        ridge=1.0,
        n_factors=0,
        n_personas=n,
    )
    raw = rng.normal(size=(n, m))
    block = model.block(raw)

    standardized = (raw - model.center) / model.scale
    weighted = np.linalg.solve(model.noise_cov, model.loading)
    assert np.allclose(block.precision, model.loading.T @ weighted)
    assert np.allclose(block.info, (standardized - model.offset) @ weighted)
    assert np.allclose(block.precision, model.precision)


def test_a_block_alone_gives_the_closed_form_posterior(channel_world) -> None:
    """With no items, the fused posterior is ``N((I+Lambda)^-1 h, (I+Lambda)^-1)``."""
    theta, params, _, features = channel_world
    model, _ = of.select_measurement_model(
        features[:200], theta[:200], lambda_grid=(1.0,), factor_grid=(0,)
    )
    block = model.block(features[200:])
    empty = np.zeros((100, 0), dtype=np.int16)
    post = pe.posterior(empty, pe.prefix_params(params, range(0)), extra=(block,))

    precision = np.eye(3) + model.precision
    expected_mean = np.linalg.solve(precision, np.asarray(block.info).T).T
    expected_cov = np.linalg.inv(precision)
    assert np.allclose(post.theta, expected_mean, atol=1e-7)
    assert np.allclose(post.cov, np.broadcast_to(expected_cov, post.cov.shape), atol=1e-6)


def test_the_blocks_information_does_not_depend_on_n(channel_world) -> None:
    """The open answers are given up front, so they weigh the same at every N."""
    theta, params, codes, features = channel_world
    model, _ = of.select_measurement_model(
        features[:200], theta[:200], lambda_grid=(1.0,), factor_grid=(0,)
    )
    block = model.block(features[200:])
    grid = [0, 3, 7, 14]
    posts = of.track_sweep(codes[200:], params, grid, block)

    for n in grid:
        blocks = pe.blocks_from_codes(codes[200:, :n], params.kinds[:n])
        hess = pe.observed_hessian(
            posts[n].theta, pe.prefix_params(params, range(n)), blocks, (block,)
        )
        without = pe.observed_hessian(
            posts[n].theta, pe.prefix_params(params, range(n)), blocks, ()
        )
        assert np.allclose(hess - without, model.precision, atol=1e-5)


# --------------------------------------------------------------------------
# the block earns its place
# --------------------------------------------------------------------------


def test_the_block_improves_agreement_with_the_fitting_target(channel_world) -> None:
    """Fused beats closed-only against the target the head was fitted on.

    Split discipline is respected inside the test: the measurement model is
    fitted on the first 200 people and the comparison is made on the other 100.
    """
    theta, params, codes, features = channel_world
    model, _ = of.select_measurement_model(features[:200], theta[:200])
    block = model.block(features[200:])

    closed = pe.posterior(codes[200:, :6], pe.prefix_params(params, range(6)))
    fused = pe.posterior(
        codes[200:, :6], pe.prefix_params(params, range(6)), extra=(block,)
    )

    closed_rmse = np.sqrt(np.mean((closed.theta - theta[200:]) ** 2))
    fused_rmse = np.sqrt(np.mean((fused.theta - theta[200:]) ** 2))
    assert fused_rmse < closed_rmse, (closed_rmse, fused_rmse)

    closed_r = np.mean(mirt._column_correlations(closed.theta, theta[200:]))
    fused_r = np.mean(mirt._column_correlations(fused.theta, theta[200:]))
    assert fused_r > closed_r


def test_the_block_never_widens_the_blur(channel_world) -> None:
    """Adding a positive semi-definite term cannot make any interval wider."""
    theta, params, codes, features = channel_world
    model, _ = of.select_measurement_model(
        features[:200], theta[:200], lambda_grid=(1.0,), factor_grid=(0,)
    )
    block = model.block(features[200:])
    pinned = np.zeros((100, 3))
    with_block = pe.blur_at_fixed_theta(
        codes[200:], params, pinned, [8], extra=(block,)
    )
    without = pe.blur_at_fixed_theta(codes[200:], params, pinned, [8])
    assert np.all(with_block <= without + 1e-12)


def test_model_selection_prefers_a_factor_when_the_noise_is_correlated() -> None:
    """The criterion notices correlated residuals rather than banking them as data."""
    rng = np.random.default_rng(8)
    dims, m, n = 3, 8, 400
    theta = rng.normal(size=(n, dims))
    loading = rng.normal(size=(m, dims))
    shared = rng.normal(size=(n, 1))
    features = theta @ loading.T + shared @ np.ones((1, m)) * 1.2 + rng.normal(size=(n, m)) * 0.3

    _, diagnostics = of.select_measurement_model(
        features, theta, lambda_grid=(1.0, 10.0), factor_grid=(0, 1)
    )
    assert diagnostics["selected"]["n_factors"] == 1, diagnostics["selected"]

    by_factor = {}
    for row in diagnostics["cv_curve"]:
        key = row["n_factors"]
        by_factor[key] = min(by_factor.get(key, np.inf), row["cv_nll"])
    assert by_factor[1] < by_factor[0]


def test_psi_comes_from_out_of_fold_residuals() -> None:
    """The declared noise is the noise the head made on data it had not seen."""
    rng = np.random.default_rng(9)
    theta = rng.normal(size=(200, 3))
    features = theta @ rng.normal(size=(4, 3)).T + rng.normal(size=(200, 4)) * 0.5

    model, diagnostics = of.select_measurement_model(
        features, theta, lambda_grid=(1.0,), factor_grid=(0,)
    )
    center = features.mean(axis=0)
    scale = features.std(axis=0)
    standardized = (features - center) / scale
    in_sample = standardized - (theta @ model.loading.T + model.offset)
    in_sample_psi = of.fit_noise_covariance(in_sample, 0)

    assert diagnostics["psi_source"].startswith("pooled out-of-fold")
    assert np.mean(np.diag(model.noise_cov)) > np.mean(np.diag(in_sample_psi)), (
        "an out-of-fold Psi must be at least as large as the in-sample one"
    )


# --------------------------------------------------------------------------
# the equivalence arithmetic
# --------------------------------------------------------------------------


def test_interpolate_n_finds_an_exact_point_and_an_interior_one() -> None:
    grid = [0, 1, 2, 3]
    curve = [1.0, 0.8, 0.6, 0.4]
    assert of.interpolate_n(grid, curve, 0.8)["equivalent_n"] == pytest.approx(1.0)
    assert of.interpolate_n(grid, curve, 0.7)["equivalent_n"] == pytest.approx(1.5)
    assert of.interpolate_n(grid, curve, 1.5)["equivalent_n"] == pytest.approx(0.0)


def test_interpolate_n_extrapolates_and_says_so() -> None:
    grid = [0, 1, 2, 3]
    curve = [1.0, 0.8, 0.6, 0.4]
    beyond = of.interpolate_n(grid, curve, 0.3)
    assert beyond["extrapolated"] is True
    assert beyond["equivalent_n"] == pytest.approx(3.5)


def test_interpolate_n_refuses_to_extrapolate_past_the_cap() -> None:
    grid = [0, 5, 10]
    curve = [1.0, 0.99, 0.98]
    far = of.interpolate_n(grid, curve, 0.1)
    assert far["extrapolated"] is True
    assert far["equivalent_n"] is None
    assert "more than" in far["note"]


def test_equivalence_table_prices_a_known_head_start() -> None:
    """A fused curve that is the closed curve shifted by two items reads as two."""
    grid = list(range(0, 8))
    closed = [1.0 - 0.05 * n for n in grid]
    fused = [1.0 - 0.05 * (n + 2) for n in grid]
    table = of.equivalence_table(grid, closed, fused, quantity="test")
    measured = [row for row in table["per_n"] if not row["extrapolated"]]
    assert measured, "nothing was measurable"
    for row in measured:
        assert row["worth_in_closed_items"] == pytest.approx(2.0, abs=1e-9)


# --------------------------------------------------------------------------
# the feature archive
# --------------------------------------------------------------------------


def test_load_open_features_refuses_a_shape_that_does_not_add_up(tmp_path: Path) -> None:
    path = tmp_path / "features.npz"
    np.savez_compressed(
        path,
        pids=np.array(["p0001", "p0002"]),
        prompt_ids=np.array(["oe01", "oe02"]),
        poles=np.array(["TRU", "RSK"]),
        features=np.zeros((2, 5)),  # should be 2 x 2 = 4
        observed=np.ones((2, 2, 2), dtype=bool),
    )
    with pytest.raises(ValueError, match="columns"):
        of.load_open_features(path)


def test_rows_for_refuses_an_unknown_persona(tmp_path: Path) -> None:
    path = tmp_path / "features.npz"
    np.savez_compressed(
        path,
        pids=np.array(["p0001"]),
        prompt_ids=np.array(["oe01"]),
        poles=np.array(["TRU", "RSK"]),
        features=np.zeros((1, 2)),
        observed=np.ones((1, 1, 2), dtype=bool),
    )
    feats = of.load_open_features(path)
    with pytest.raises(ValueError, match="no open-answer features"):
        feats.rows_for(["p0001", "p9999"])


# --------------------------------------------------------------------------
# the real artifacts
# --------------------------------------------------------------------------


needs_artifacts = pytest.mark.skipif(
    not (FUSED_NPZ.is_file() and BACKBONE_NPZ.is_file() and FUSED_JSON.is_file()),
    reason="Stage 3 artifacts are not in this checkout",
)


@needs_artifacts
def test_closed_only_track_is_bit_identical_to_the_backbone() -> None:
    """The fusion recomputes the backbone; it must not have changed it."""
    backbone = np.load(BACKBONE_NPZ, allow_pickle=False)
    fused = np.load(FUSED_NPZ, allow_pickle=False)

    pairs = [
        ("theta_train_by_n", "theta_train_closed_only_by_n"),
        ("blur_train_by_n", "blur_train_closed_only_by_n"),
        ("theta_holdout_by_n", "theta_holdout_closed_only_by_n"),
        ("blur_holdout_by_n", "blur_holdout_closed_only_by_n"),
        ("cov_holdout_by_n", "cov_holdout_closed_only_by_n"),
        ("blur_kappa", "blur_kappa_closed_only"),
        ("blur_holdout_calibrated_by_n", "blur_holdout_closed_only_calibrated_by_n"),
        ("persona_train_ids", "persona_train_ids"),
        ("persona_holdout_ids", "persona_holdout_ids"),
        ("scripted_item_ids", "scripted_item_ids"),
    ]
    for backbone_key, fused_key in pairs:
        assert np.array_equal(backbone[backbone_key], fused[fused_key]), (
            f"{fused_key} differs from the backbone's {backbone_key}"
        )


@needs_artifacts
def test_calibration_is_persisted_for_both_tracks_and_actually_applied() -> None:
    report = json.loads(FUSED_JSON.read_text(encoding="utf-8"))
    archive = np.load(FUSED_NPZ, allow_pickle=False)

    for track in ("closed_only", "fused"):
        block = report["blur_calibration"][track]
        assert block["fitted_at_n"] == 15
        assert block["fitted_on"] == "training personas only"
        assert block["n_personas"] == 400
        assert "training" in block["definition"]

        kappa = np.asarray(block["kappa_per_dim"], dtype=float)
        assert np.allclose(kappa, archive[f"blur_kappa_{track}"])
        raw = archive[f"blur_holdout_{track}_by_n"]
        calibrated = archive[f"blur_holdout_{track}_calibrated_by_n"]
        assert np.allclose(calibrated, raw * kappa[None, None, :])

    closed = np.asarray(report["blur_calibration"]["closed_only"]["kappa_per_dim"])
    fused = np.asarray(report["blur_calibration"]["fused"]["kappa_per_dim"])
    assert not np.allclose(closed, fused), (
        "the fused calibration was not re-fitted with the block in place"
    )


@needs_artifacts
@pytest.mark.skipif(
    not (FIT_NPZ.is_file() and FEATURES_NPZ.is_file()),
    reason="the fit archive or the feature archive is not in this checkout",
)
def test_the_measurement_model_is_reproducible_from_the_training_personas_alone() -> None:
    """Split discipline, checked by rebuilding B and c from the training rows."""
    report = json.loads(FUSED_JSON.read_text(encoding="utf-8"))
    fit = np.load(FIT_NPZ, allow_pickle=False)
    feats = of.load_open_features(FEATURES_NPZ)

    train_ids = [str(p) for p in fit["persona_train_ids"]]
    raw = feats.rows_for(train_ids)
    center = np.asarray(report["measurement_model"]["feature_center"])
    scale = np.asarray(report["measurement_model"]["feature_scale"])
    assert np.allclose(center, raw.mean(axis=0))
    assert np.allclose(scale, raw.std(axis=0))

    standardized = (raw - center) / scale
    loading, offset = of.ridge_fit(
        standardized, np.asarray(fit["theta_train"]), report["measurement_model"]["ridge_lambda"]
    )
    assert np.allclose(loading, np.asarray(report["measurement_model"]["B"]))
    assert np.allclose(offset, np.asarray(report["measurement_model"]["c"]))
    assert report["measurement_model"]["fitted_on_n_personas"] == len(train_ids)


@needs_artifacts
def test_the_fused_track_agrees_better_with_the_full_matrix_estimate() -> None:
    """The block earns its place on the real run, on held-out personas."""
    report = json.loads(FUSED_JSON.read_text(encoding="utf-8"))
    at_15 = report["per_n"][15]
    assert at_15["n"] == 15
    closed = at_15["closed_only"]["holdout"]["agreement_with_full_matrix"]
    fused = at_15["fused"]["holdout"]["agreement_with_full_matrix"]
    assert fused["r_mean"] > closed["r_mean"]
    assert fused["rmse_mean"] < closed["rmse_mean"]
    assert at_15["fused"]["holdout"]["mean_blur"] < at_15["closed_only"]["holdout"]["mean_blur"]
