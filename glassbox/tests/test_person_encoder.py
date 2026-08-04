"""Unit tests for the Stage-3 person-encoder backbone.

Everything here runs on fixtures this file makes up: a small item bank with
known parameters and a population drawn from a known latent model. No project
data is read, so the tests say something about the machinery rather than about
the run.

Three things are worth checking beyond "does it run":

  * the posterior is the *same* posterior the Stage 2 fit uses -- same
    likelihood, same prior, same answer coding -- so a couple of tests pin it
    against ``src.model.mirt`` directly rather than against a restatement;
  * the extension point behaves like a Gaussian measurement, which is checked
    against a closed form rather than a regression value;
  * the calibration constants survive a round trip and are applied, not
    recomputed, when they are used on a second population.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.model import mirt, person_encoder as pe
from src.model import profile_baseline as pb


# --------------------------------------------------------------------------
# fixtures: a small world with known parameters
# --------------------------------------------------------------------------


def make_items(
    *, n_likert: int = 24, n_binary: int = 12, dims: int = 3, seed: int = 11
) -> mirt.ItemParams:
    """A tiny bank with clear primary dimensions and well-spread thresholds."""
    rng = np.random.default_rng(seed)
    kinds = [mirt.LIKERT] * n_likert + [mirt.BINARY] * n_binary
    n_items = len(kinds)
    loadings = rng.normal(0.0, 0.35, size=(n_items, dims))
    primary = np.arange(n_items) % dims
    loadings[np.arange(n_items), primary] += rng.choice([-1.6, 1.6], size=n_items)
    cut_raw = np.zeros((n_items, mirt.LIKERT_CUTS))
    cut_raw[:, 0] = rng.normal(-1.5, 0.4, n_items)
    cut_raw[:, 1:] = mirt.inv_softplus(np.abs(rng.normal(1.0, 0.2, (n_items, 3))))
    return mirt.ItemParams(loadings=loadings, cut_raw=cut_raw, kinds=kinds)


def sample_answers(
    theta: np.ndarray, params: mirt.ItemParams, *, seed: int = 12
) -> np.ndarray:
    """Draw a category for every person and item from the model itself."""
    rng = np.random.default_rng(seed)
    z = theta @ params.loadings.T
    taus = params.thresholds()
    diff = params.difficulty()
    codes = np.zeros(z.shape, dtype=np.int16)
    for j, kind in enumerate(params.kinds):
        if kind == mirt.LIKERT:
            seen = mirt.sigmoid(z[:, j][:, None] - taus[j][None, :])
            padded = np.concatenate(
                [np.ones((z.shape[0], 1)), seen, np.zeros((z.shape[0], 1))], axis=1
            )
            probs = padded[:, :-1] - padded[:, 1:]
            draws = np.array(
                [rng.choice(mirt.LIKERT_CATEGORIES, p=row / row.sum()) for row in probs]
            )
            codes[:, j] = draws
        else:
            p = mirt.sigmoid(z[:, j] - diff[j])
            codes[:, j] = (rng.random(z.shape[0]) < p).astype(np.int16)
    return codes


@pytest.fixture(scope="module")
def world() -> tuple[np.ndarray, mirt.ItemParams, np.ndarray]:
    """(theta, item parameters, answers) for 120 people on 36 items in 3 dims."""
    params = make_items()
    rng = np.random.default_rng(7)
    theta = rng.normal(size=(120, 3))
    codes = sample_answers(theta, params)
    return theta, params, codes


# --------------------------------------------------------------------------
# the posterior recovers a synthetic persona
# --------------------------------------------------------------------------


def test_posterior_recovers_synthetic_theta(world) -> None:
    """A population generated from the model is placed back where it came from."""
    theta, params, codes = world
    post = pe.posterior(codes, params)

    assert post.theta.shape == theta.shape
    assert post.blur.shape == theta.shape
    assert post.non_pd_rows == []
    assert post.max_grad < 1e-8, "Newton did not reach the mode"

    r = mirt._column_correlations(post.theta, theta)
    assert np.all(r > 0.85), f"per-dimension recovery too weak: {r}"
    rmse = np.sqrt(np.mean((post.theta - theta) ** 2, axis=0))
    assert np.all(rmse < 0.45), f"per-dimension RMSE too large: {rmse}"
    # the blur is an honest reading of that error, not a decoration
    assert np.all(np.abs(post.blur.mean(axis=0) - rmse) < 0.12), (
        f"blur {post.blur.mean(axis=0)} does not track the error {rmse}"
    )


def test_posterior_matches_the_fit_scorer(world) -> None:
    """The mode agrees with ``mirt.score_personas`` on the same data.

    Different optimizer, same objective. If these disagree, this module is not
    computing the Stage 2 posterior.
    """
    _, params, codes = world
    data = mirt.Dataset(
        persona_ids=[f"s{i:03d}" for i in range(codes.shape[0])],
        item_ids=[f"i{j:03d}" for j in range(codes.shape[1])],
        item_kinds=list(params.kinds),
        codes=codes,
    )
    reference, _ = mirt.score_personas(data, params, dims=3, seed=0, max_iters=6000)
    post = pe.posterior(codes, params)
    assert np.allclose(post.theta, reference, atol=2e-3)

    blur, _, _ = mirt.laplace_covariance(data, params, post.theta)
    assert np.allclose(post.blur, blur, atol=1e-9)


def test_per_person_loglik_sums_to_the_fits_own_total(world) -> None:
    """The per-person likelihood is the fit's likelihood, split up."""
    theta, params, codes = world
    blocks = pe.blocks_from_codes(codes, params.kinds)
    total, grad, _, _ = mirt._loglik_and_grads(theta, params, blocks, want_item_grads=False)
    per_person = pe.per_person_loglik(theta, params, blocks)
    assert per_person.shape == (codes.shape[0],)
    assert per_person.sum() == pytest.approx(total, rel=1e-12, abs=1e-9)

    # and the gradient this module differentiates is the fit's gradient
    assert np.allclose(pe.neg_log_posterior_grad(theta, params, blocks), -grad + theta)


def test_unanswered_items_are_ignored(world) -> None:
    """A ``-1`` cell contributes nothing, so it is the same as a shorter form."""
    _, params, codes = world
    short = pe.posterior(codes[:, :8], pe.prefix_params(params, range(8)))
    padded = codes.copy()
    padded[:, 8:] = -1
    masked = pe.posterior(padded, params)
    assert np.allclose(short.theta, masked.theta, atol=1e-9)
    assert np.allclose(short.blur, masked.blur, atol=1e-9)


# --------------------------------------------------------------------------
# blur shrinks with more items
# --------------------------------------------------------------------------


def test_blur_shrinks_as_items_are_added(world) -> None:
    """More answers, less blur -- on the mean, and per dimension."""
    _, params, codes = world
    n_grid = [1, 2, 4, 8, 16, 24, 36]
    posteriors = pe.encode_prefixes(codes, params, n_grid)
    mean_blur = np.stack([posteriors[n].blur.mean(axis=0) for n in n_grid])

    assert mean_blur[0].mean() > mean_blur[-1].mean()
    report = pe.monotonicity_report(n_grid, mean_blur)
    assert report["largest_increase_any_dim"] < 0.02, report["per_dim"]
    # the prior is N(0, I), so blur can never exceed 1
    assert np.all(mean_blur <= 1.0 + 1e-9)


def test_blur_at_fixed_theta_is_exactly_monotone(world) -> None:
    """With theta held still the information matrices nest, so this is a theorem."""
    _, params, codes = world
    n_grid = list(range(1, codes.shape[1] + 1))
    theta = pe.posterior(codes, params).theta
    blur = pe.blur_at_fixed_theta(codes, params, theta, n_grid)
    steps = np.diff(blur, axis=0)
    assert np.all(steps <= 0), f"largest rise {steps.max():.3e}"

    report = pe.monotonicity_report(n_grid, blur.mean(axis=1), blur_by_n=blur)
    assert report["verdict"] == "PASS"
    assert report["persona_level"]["n_increases"] == 0


def test_monotonicity_report_catches_a_rise() -> None:
    """The check is not vacuous: a planted rise is found and located."""
    mean_blur = np.array([[0.9, 0.9], [0.8, 0.95], [0.7, 0.7]])
    report = pe.monotonicity_report([1, 2, 3], mean_blur)
    assert report["verdict"] == "FAIL"
    assert report["n_dims_monotone"] == 1
    assert report["per_dim"][0]["monotone"] is True
    assert report["per_dim"][1]["violations"][0]["from_n"] == 1
    assert report["per_dim"][1]["largest_increase"] == pytest.approx(0.05)


# --------------------------------------------------------------------------
# the extension point
# --------------------------------------------------------------------------


def test_gaussian_block_alone_gives_the_closed_form_posterior() -> None:
    """No items, one block: the answer must be the exact Gaussian posterior."""
    rng = np.random.default_rng(3)
    dims = 4
    root = rng.normal(size=(dims, dims))
    precision = root @ root.T + 0.5 * np.eye(dims)
    info = rng.normal(size=(6, dims))
    block = pe.GaussianBlock(precision=precision, info=info)

    empty = mirt.ItemParams(
        loadings=np.zeros((0, dims)), cut_raw=np.zeros((0, mirt.LIKERT_CUTS)), kinds=[]
    )
    post = pe.posterior(np.zeros((6, 0), dtype=np.int16), empty, extra=[block], dims=dims)

    expected_cov = np.linalg.inv(np.eye(dims) + precision)
    expected_mean = info @ expected_cov.T
    assert np.allclose(post.theta, expected_mean, atol=1e-7)
    assert np.allclose(post.cov, expected_cov[None, :, :], atol=1e-6)
    assert np.allclose(post.blur, np.sqrt(np.diag(expected_cov))[None, :], atol=1e-7)


def test_gaussian_block_shifts_an_item_posterior_toward_its_mean(world) -> None:
    """More weight on the block pulls the mode toward it, all the way there.

    With items in play the dimensions are coupled through the loadings, so no
    single coordinate is guaranteed to move toward the block's mean. What is
    guaranteed is the dose response: as the block's precision grows, the mode
    goes to the block's mean and the blur goes to zero.
    """
    _, params, codes = world
    dims = 3
    items = pe.prefix_params(params, range(5))
    target = np.full((codes.shape[0], dims), 1.5)

    distances = []
    previous_blur = None
    for weight in (0.0, 0.5, 2.0, 20.0, 2000.0):
        block = pe.GaussianBlock(
            precision=weight * np.eye(dims), info=weight * target
        )
        post = pe.posterior(codes[:, :5], items, extra=[block])
        distances.append(float(np.abs(post.theta - target).max()))
        blur = post.blur.mean(axis=0)
        if previous_blur is not None:
            assert np.all(blur <= previous_blur + 1e-9), "more information, wider blur"
        previous_blur = blur

    assert distances == sorted(distances, reverse=True), distances
    assert distances[-1] < 1e-2, "an overwhelming block did not win"


def test_a_block_is_a_stationary_point_of_the_combined_objective(world) -> None:
    """The fused mode really is the mode of items-plus-block, not of one of them."""
    _, params, codes = world
    dims = 3
    items = pe.prefix_params(params, range(7))
    rng = np.random.default_rng(9)
    root = rng.normal(size=(dims, dims))
    block = pe.GaussianBlock(
        precision=root @ root.T + 0.3 * np.eye(dims),
        info=rng.normal(size=(codes.shape[0], dims)),
    )

    fused = pe.posterior(codes[:, :7], items, extra=[block])
    blocks = pe.blocks_from_codes(codes[:, :7], items.kinds)
    grad = pe.neg_log_posterior_grad(fused.theta, items, blocks, [block])
    assert np.max(np.abs(grad)) < 1e-8

    # and it beats the items-only mode on the combined objective
    items_only = pe.posterior(codes[:, :7], items)
    combined = pe.neg_log_posterior(fused.theta, items, blocks, [block])
    worse = pe.neg_log_posterior(items_only.theta, items, blocks, [block])
    assert np.all(combined <= worse + 1e-12)

    # the block adds its precision to the Hessian exactly
    with_block = pe.observed_hessian(fused.theta, items, blocks, [block])
    without = pe.observed_hessian(fused.theta, items, blocks)
    assert np.allclose(with_block - without, block.precision[None, :, :], atol=1e-6)


def test_gaussian_block_from_linear_matches_the_information_form() -> None:
    """``y = B theta + c + eps`` becomes the right precision and information."""
    rng = np.random.default_rng(5)
    dims, m = 3, 2
    loading = rng.normal(size=(m, dims))
    offset = rng.normal(size=m)
    psi = np.array([0.4, 1.7])
    y = rng.normal(size=(4, m))

    block = pe.GaussianBlock.from_linear(loading, offset, y, psi)
    expected_precision = loading.T @ np.diag(1.0 / psi) @ loading
    expected_info = (y - offset) @ (np.diag(1.0 / psi) @ loading)
    assert np.allclose(block.precision, expected_precision)
    assert np.allclose(block.info, expected_info)

    # the full-covariance form must agree with the diagonal one
    same = pe.GaussianBlock.from_linear(loading, offset, y, np.diag(psi))
    assert np.allclose(same.precision, block.precision)
    assert np.allclose(same.info, block.info)

    # and the block's own penalty must equal the Gaussian negative log density
    theta = rng.normal(size=(4, dims))
    residual = y - offset - theta @ loading.T
    quad = 0.5 * np.sum(residual**2 / psi, axis=1)
    constant = 0.5 * np.sum((y - offset) ** 2 / psi, axis=1)
    assert np.allclose(block.penalty(theta), quad - constant)


def test_gaussian_block_rejects_bad_shapes_and_indefinite_precision() -> None:
    dims = 3
    good = pe.GaussianBlock(precision=np.eye(dims), info=np.zeros(dims))
    good.validate(5, dims)

    with pytest.raises(ValueError):
        pe.GaussianBlock(precision=np.eye(dims + 1), info=np.zeros(dims)).validate(5, dims)
    with pytest.raises(ValueError):
        pe.GaussianBlock(precision=np.eye(dims), info=np.zeros(dims + 1)).validate(5, dims)
    with pytest.raises(ValueError):
        pe.GaussianBlock(precision=-np.eye(dims), info=np.zeros(dims)).validate(5, dims)


def test_per_persona_precision_is_allowed(world) -> None:
    """A block may carry its own precision per persona, not just a shared one."""
    _, params, codes = world
    n = codes.shape[0]
    dims = 3
    shared = pe.GaussianBlock(precision=np.eye(dims), info=np.zeros((n, dims)))
    stacked = pe.GaussianBlock(
        precision=np.repeat(np.eye(dims)[None, :, :], n, axis=0),
        info=np.zeros((n, dims)),
    )
    a = pe.posterior(codes[:, :6], pe.prefix_params(params, range(6)), extra=[shared])
    b = pe.posterior(codes[:, :6], pe.prefix_params(params, range(6)), extra=[stacked])
    assert np.allclose(a.theta, b.theta)
    assert np.allclose(a.blur, b.blur)


# --------------------------------------------------------------------------
# item parameters read back out of a fit archive
# --------------------------------------------------------------------------


def test_item_params_round_trip_through_a_fit_archive(tmp_path: Path, world) -> None:
    """Thresholds and difficulties survive being stored and rebuilt, exactly."""
    _, params, _ = world
    payload = mirt._params_payload(params, [f"i{j:03d}" for j in range(len(params.kinds))])
    archive = tmp_path / "fit.npz"
    np.savez_compressed(
        archive,
        item_train_ids=np.array(payload["item_ids"]),
        train_item_kinds=np.array(payload["kinds"]),
        train_loadings=payload["loadings"],
        train_thresholds=payload["thresholds"],
        train_difficulty=payload["difficulty"],
    )
    rebuilt, item_ids = pe.item_params_from_fit(np.load(archive), block="train")

    assert item_ids == payload["item_ids"]
    assert rebuilt.kinds == params.kinds
    assert np.allclose(rebuilt.loadings, params.loadings)
    likert = rebuilt.is_likert
    assert np.allclose(rebuilt.thresholds()[likert], params.thresholds()[likert])
    assert np.allclose(rebuilt.difficulty()[~likert], params.difficulty()[~likert])


# --------------------------------------------------------------------------
# blur calibration
# --------------------------------------------------------------------------


def test_calibration_constants_are_persisted_and_applied(tmp_path: Path) -> None:
    """Fitted on one population, written down, applied unchanged to another."""
    rng = np.random.default_rng(21)
    dims = 3
    theta_full = rng.normal(size=(200, dims))
    blur = np.tile(np.array([0.4, 0.5, 0.6]), (200, 1))
    inflation = np.array([1.0, 1.5, 2.0])
    theta_n = theta_full + rng.normal(size=(200, dims)) * (blur * inflation)

    calib = pe.fit_blur_calibration(theta_n, blur, theta_full, at_n=15)
    assert calib.kappa.shape == (dims,)
    assert np.allclose(calib.kappa, inflation, atol=0.15), calib.kappa
    assert calib.fitted_at_n == 15
    assert calib.n_personas == 200

    payload = json.loads(json.dumps(calib.to_json()))
    assert "definition" in payload and "training" in payload["definition"]
    restored = pe.BlurCalibration.from_json(payload)
    assert np.allclose(restored.kappa, calib.kappa)

    # applied, not refitted: a second population gets the first one's constants
    other = np.tile(np.array([0.2, 0.2, 0.2]), (7, 1))
    assert np.allclose(restored.apply(other), other * calib.kappa[None, :])


def test_calibration_is_exactly_one_scalar_per_dimension() -> None:
    """Scaling one dimension's residuals scales only that dimension's constant."""
    rng = np.random.default_rng(22)
    dims = 3
    theta_full = rng.normal(size=(300, dims))
    blur = np.full((300, dims), 0.5)
    noise = rng.normal(size=(300, dims))

    first = pe.fit_blur_calibration(theta_full + noise, blur, theta_full, at_n=5)
    doubled = noise.copy()
    doubled[:, 1] *= 2.0
    second = pe.fit_blur_calibration(theta_full + doubled, blur, theta_full, at_n=5)

    assert np.allclose(first.kappa[[0, 2]], second.kappa[[0, 2]])
    assert second.kappa[1] == pytest.approx(2.0 * first.kappa[1])


def test_calibration_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError):
        pe.fit_blur_calibration(
            np.zeros((5, 3)), np.zeros((5, 3)), np.zeros((5, 4)), at_n=3
        )


# --------------------------------------------------------------------------
# agreement helper
# --------------------------------------------------------------------------


def test_agreement_is_perfect_against_itself() -> None:
    rng = np.random.default_rng(31)
    theta = rng.normal(size=(50, 4))
    out = pe.agreement(theta, theta)
    assert out["r_mean"] == pytest.approx(1.0)
    assert out["rmse_mean"] == pytest.approx(0.0, abs=1e-12)
    assert out["n_personas"] == 50


# --------------------------------------------------------------------------
# transcripts
# --------------------------------------------------------------------------


def _transcript(pid: str, answers: list) -> dict:
    return {
        "pid": pid,
        "profile": {
            "pid": pid,
            "age": 40,
            "occupation": "barista",
            "city_size": "village",
            "household": "alone",
            "region_type": "rural",
        },
        "turns": [
            {"n": i + 1, "item_id": f"q{i:03d}", "question": "t", "answer": a}
            for i, a in enumerate(answers)
        ],
        "open": [],
    }


def test_transcripts_are_read_in_scripted_order(tmp_path: Path) -> None:
    path = tmp_path / "transcripts.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(_transcript(pid, [3, "yes", 5]))
            for pid in ("p0001", "p0002")
        )
        + "\n",
        encoding="utf-8",
    )
    transcripts = pe.load_transcripts(path)
    assert transcripts.pids == ["p0001", "p0002"]
    assert transcripts.item_ids == ["q000", "q001", "q002"]

    codes = pe.code_matrix(
        transcripts, transcripts.pids, [mirt.LIKERT, mirt.BINARY, mirt.LIKERT]
    )
    assert codes.tolist() == [[2, 1, 4], [2, 1, 4]]


def test_a_transcript_with_a_different_order_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "transcripts.jsonl"
    first = _transcript("p0001", [1, 2])
    second = _transcript("p0002", [1, 2])
    second["turns"] = list(reversed(second["turns"]))
    for k, turn in enumerate(second["turns"]):
        turn["n"] = k + 1
    path.write_text(
        json.dumps(first) + "\n" + json.dumps(second) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="scripted order"):
        pe.load_transcripts(path)


# --------------------------------------------------------------------------
# the public-profile-only baseline
# --------------------------------------------------------------------------


def _profiles(n: int, seed: int = 41) -> dict[str, dict]:
    rng = np.random.default_rng(seed)
    jobs = ["barista", "nurse", "welder", "teacher"]
    sizes = ["village", "small town", "mid-size city", "big city"]
    homes = ["alone", "with partner", "with kids, no partner"]
    regions = ["rural", "suburban", "urban"]
    out = {}
    for i in range(n):
        pid = f"p{i:04d}"
        out[pid] = {
            "pid": pid,
            "age": int(rng.integers(18, 80)),
            "occupation": str(rng.choice(jobs)),
            "city_size": str(rng.choice(sizes)),
            "household": str(rng.choice(homes)),
            "region_type": str(rng.choice(regions)),
        }
    return out


def test_profile_baseline_shapes_and_split_discipline() -> None:
    """Trained on the training rows, applied to held-out, shapes as promised."""
    profiles = _profiles(250)
    pids = sorted(profiles)
    train_pids, holdout_pids = pids[:200], pids[200:]
    rng = np.random.default_rng(42)
    theta_train = rng.normal(size=(200, 8))
    theta_holdout = rng.normal(size=(50, 8))

    result = pb.evaluate_profile_baseline(
        profiles, train_pids, holdout_pids, theta_train, theta_holdout
    )
    report = result["report"]
    assert result["holdout"]["theta"].shape == (50, 8)
    assert result["train"]["theta"].shape == (200, 8)
    assert len(report["holdout"]["agreement_with_full_matrix"]["r_per_dim"]) == 8
    assert len(report["holdout"]["out_of_sample_r2_per_dim"]) == 8
    assert report["cross_validation"]["best_lambda"] in pb.LAMBDA_GRID
    # 2 age columns + one-hot blocks, levels learned on the training rows only
    expected = 2 + sum(len(v) for v in result["spec"].levels.values())
    assert report["features"]["n_columns"] == expected
    # pure noise targets: cross-validated R2 must not be meaningfully positive
    assert report["cross_validation"]["best_cv_r2_mean"] < 0.05
    assert json.loads(json.dumps(report))  # the report is JSON-safe


def test_profile_baseline_learns_a_real_signal() -> None:
    """Not vacuous: when the target really is a profile function, it is found."""
    profiles = _profiles(300, seed=43)
    pids = sorted(profiles)
    train_pids, holdout_pids = pids[:240], pids[240:]
    coding = {"rural": -1.0, "suburban": 0.0, "urban": 1.0}

    def target(pid: str) -> list[float]:
        p = profiles[pid]
        base = 0.05 * (p["age"] - 45) + 1.2 * coding[p["region_type"]]
        return [base] * 8

    theta_train = np.array([target(pid) for pid in train_pids])
    theta_holdout = np.array([target(pid) for pid in holdout_pids])
    result = pb.evaluate_profile_baseline(
        profiles, train_pids, holdout_pids, theta_train, theta_holdout
    )
    r2 = result["report"]["holdout"]["out_of_sample_r2_mean"]
    assert r2 > 0.9, f"a deterministic profile target was not recovered (R2 {r2})"


def test_profile_baseline_handles_a_level_seen_only_in_holdout() -> None:
    """An unseen occupation gets a zero block and is counted, not crashed on."""
    profiles = _profiles(120, seed=44)
    pids = sorted(profiles)
    train_pids, holdout_pids = pids[:100], pids[100:]
    profiles[holdout_pids[0]]["occupation"] = "lighthouse keeper"
    rng = np.random.default_rng(45)
    result = pb.evaluate_profile_baseline(
        profiles,
        train_pids,
        holdout_pids,
        rng.normal(size=(100, 8)),
        rng.normal(size=(20, 8)),
    )
    features = result["report"]["features"]
    assert features["holdout_rows_with_an_unseen_level"] == 1
    assert features["unseen_levels"] == ["occupation=lighthouse keeper"]


def test_ridge_reduces_to_least_squares_at_zero_penalty() -> None:
    rng = np.random.default_rng(46)
    x = rng.normal(size=(60, 4))
    truth = rng.normal(size=(4, 2))
    y = x @ truth + 3.0
    weights, intercept = pb.ridge_fit(x, y, 0.0)
    assert np.allclose(weights, truth, atol=1e-8)
    assert np.allclose(intercept, 3.0, atol=1e-8)
