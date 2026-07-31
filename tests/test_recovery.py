"""Unit tests for the Stage-2 recovery grader (``src/eval/recovery.py``).

Every planted quantity these tests use is invented here, in a temporary
directory, from this file's own random stream. Nothing reads the real batch.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.eval import recovery
from src.model import mirt
from tests.test_mirt import make_population

DIMS = recovery.DIMENSIONS


# --------------------------------------------------------------------------
# alignment
# --------------------------------------------------------------------------


def test_an_arbitrary_rotation_aligns_back() -> None:
    """A rotated copy of the answer must align back onto the answer."""
    rng = np.random.default_rng(0)
    planted = rng.normal(size=(200, 8))
    rotation, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    scrambled = planted @ rotation

    recovered = recovery.procrustes_rotation(scrambled, planted)
    aligned = scrambled @ recovered

    assert np.allclose(aligned, planted, atol=1e-8)
    assert np.allclose(recovered.T @ recovered, np.eye(8), atol=1e-10)


def test_alignment_is_fitted_on_train_and_carried_to_held_out() -> None:
    """The rotation learned on one block must fix the other block untouched."""
    rng = np.random.default_rng(1)
    planted = rng.normal(size=(300, 8))
    rotation, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    scrambled = planted @ rotation

    learned = recovery.procrustes_rotation(scrambled[:200], planted[:200])
    held_out = scrambled[200:] @ learned

    assert np.allclose(held_out, planted[200:], atol=1e-8)


def test_alignment_ignores_the_scale_of_the_fitted_space() -> None:
    rng = np.random.default_rng(2)
    planted = rng.normal(size=(150, 8))
    fitted = planted @ np.linalg.qr(rng.normal(size=(8, 8)))[0]

    small = recovery.procrustes_rotation(fitted * 0.05, planted)
    large = recovery.procrustes_rotation(fitted * 20.0, planted)
    assert np.allclose(small, large, atol=1e-9)


def test_covariances_travel_with_the_rotation() -> None:
    """A rotated posterior must keep its volume and match a hand-rolled rotation."""
    rng = np.random.default_rng(3)
    root = rng.normal(size=(5, 8, 8))
    covs = np.einsum("nij,nkj->nik", root, root)
    rotation, _ = np.linalg.qr(rng.normal(size=(8, 8)))

    rotated = recovery.rotate_covariances(covs, rotation)
    by_hand = np.stack([rotation.T @ c @ rotation for c in covs])

    assert np.allclose(rotated, by_hand)
    assert np.allclose(np.linalg.det(rotated), np.linalg.det(covs))


# --------------------------------------------------------------------------
# the bars, on inputs built to sit on either side of them
# --------------------------------------------------------------------------


def test_trait_recovery_reads_the_worst_dimension() -> None:
    rng = np.random.default_rng(4)
    planted = rng.normal(size=(300, 8))
    good = planted + rng.normal(0, 0.15, planted.shape)
    result = recovery.trait_recovery(good, planted)
    assert result["pass"] is True
    assert result["min_r"] >= recovery.BAR_TRAIT_R

    spoiled = good.copy()
    spoiled[:, 3] = rng.normal(size=300)  # one dimension is pure noise
    broken = recovery.trait_recovery(spoiled, planted)
    assert broken["pass"] is False
    assert broken["min_r_dimension"] == DIMS[3]


def test_item_recovery_uses_the_median_across_items() -> None:
    rng = np.random.default_rng(5)
    designed = np.zeros((40, 8))
    for row in range(40):
        designed[row, rng.integers(0, 8)] = rng.choice([-1.0, 1.0])
    fitted = designed * 2.3 + rng.normal(0, 0.1, designed.shape)

    result = recovery.item_recovery(fitted, designed, [f"q{i}" for i in range(40)])
    assert result["pass"] is True
    assert result["median_r"] >= recovery.BAR_ITEM_R
    assert result["median_cosine"] > 0.8

    noise = rng.normal(size=designed.shape)
    assert recovery.item_recovery(noise, designed, [f"q{i}" for i in range(40)])["pass"] is False


def test_item_recovery_is_blind_to_the_size_of_a_loading() -> None:
    """Only the shape of the vector is graded, so a rescale changes nothing."""
    rng = np.random.default_rng(6)
    designed = rng.normal(size=(20, 8))
    fitted = designed * 7.0
    ids = [f"q{i}" for i in range(20)]
    assert recovery.item_recovery(fitted, designed, ids)["median_r"] == pytest.approx(1.0)


def test_blur_honesty_finds_the_band() -> None:
    rng = np.random.default_rng(7)
    planted = rng.normal(size=(500, 8))
    sigma = 0.4
    estimate = planted + rng.normal(0, sigma, planted.shape)

    honest = recovery.blur_honesty(estimate, np.full_like(estimate, sigma), planted)
    assert honest["pass"] is True  # one sigma of a normal covers about 68 percent
    assert 0.60 <= honest["pooled_coverage"] <= 0.75
    assert honest["diagnostics"]["variance_ratio_actual_over_predicted"] == pytest.approx(1.0, abs=0.15)

    boastful = recovery.blur_honesty(estimate, np.full_like(estimate, sigma / 4), planted)
    assert boastful["pass"] is False
    assert boastful["diagnostics"]["variance_ratio_actual_over_predicted"] > 4
    assert boastful["diagnostics"]["blur_multiplier_for_nominal_68pct"] > 2

    timid = recovery.blur_honesty(estimate, np.full_like(estimate, sigma * 4), planted)
    assert timid["pass"] is False
    assert timid["pooled_coverage"] > 0.75


def test_weak_item_ordering_separates_the_two_groups() -> None:
    bank = {}
    discrimination = {}
    for i in range(30):
        iid = f"s{i}"
        bank[iid] = {"loadings": {DIMS[i % 8]: 1.0}, "strength_class": "strong"}
        discrimination[iid] = 2.0 + 0.1 * i
    for i in range(10):
        iid = f"w{i}"
        bank[iid] = {"loadings": {DIMS[i % 8]: 0.25}, "strength_class": "weak"}
        discrimination[iid] = 0.3 + 0.02 * i
    for i in range(5):
        iid = f"d{i}"
        bank[iid] = {"loadings": {}, "strength_class": "none"}
        discrimination[iid] = 0.05

    result = recovery.weak_item_ordering(discrimination, bank)
    assert result["pass"] is True
    assert result["prob_weak_below_strong"] == pytest.approx(1.0)
    assert result["separation"] > 0
    assert result["n_weak"] == 10 and result["n_strong_primary"] == 30
    assert result["weak_items_above_weakest_strong"] == []

    # now spread the weak items right across the strong range, so the two
    # groups are no longer telling apart
    for i in range(10):
        discrimination[f"w{i}"] = 2.0 + 0.3 * i
    muddled = recovery.weak_item_ordering(discrimination, bank)
    assert muddled["pass"] is False
    assert muddled["prob_weak_below_strong"] < 0.6


def test_reliability_ceiling_applies_spearman_brown() -> None:
    rng = np.random.default_rng(8)
    signal = rng.normal(size=(400, 8))
    half_a = signal + rng.normal(0, 0.5, signal.shape)
    half_b = signal + rng.normal(0, 0.5, signal.shape)
    observed = {dim: 0.8 for dim in DIMS}

    result = recovery.reliability_ceiling(half_a, half_b, observed)
    for dim in DIMS:
        entry = result["per_dimension"][dim]
        expected = 2 * entry["half_sample_agreement"] / (1 + entry["half_sample_agreement"])
        assert entry["reliability_full_length"] == pytest.approx(expected)
        assert entry["reliability_full_length"] > entry["half_sample_agreement"]
        assert entry["disattenuated_r"] == pytest.approx(
            0.8 / np.sqrt(entry["reliability_full_length"])
        )


def test_correlation_block_measures_the_gap_to_the_planted_matrix() -> None:
    rng = np.random.default_rng(9)
    root = np.linalg.cholesky(recovery.PLANTED_SIGMA)
    drawn = rng.normal(size=(20000, 8)) @ root.T

    block = recovery.correlation_block(drawn)
    assert block["max_abs_delta_offdiagonal"] < 0.05
    assert np.allclose(np.array(block["recovered"]), np.corrcoef(drawn, rowvar=False))
    assert np.allclose(
        np.array(block["delta"]), np.array(block["recovered"]) - recovery.PLANTED_SIGMA
    )


def test_stratum_summary_splits_by_label() -> None:
    per_item = {
        "a": {"r": 0.9, "cosine": 0.9, "fitted_norm": 1.0},
        "b": {"r": 0.5, "cosine": 0.5, "fitted_norm": 2.0},
        "c": {"r": 0.7, "cosine": 0.7, "fitted_norm": 3.0},
    }
    strata = {"a": "near", "b": "far", "c": "far"}
    summary = recovery.stratum_summary(per_item, strata)
    assert summary["near"]["n"] == 1
    assert summary["far"]["n"] == 2
    assert summary["same-domain"]["n"] == 0
    assert summary["far"]["r"]["median"] == pytest.approx(0.6)


# --------------------------------------------------------------------------
# end to end: fit an invented population, then grade it against its own truth
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def graded(tmp_path_factory):
    """Fit a made-up population and grade it, exactly as the CLI would."""
    workspace = tmp_path_factory.mktemp("stage2")
    planted_dir = workspace / "planted"
    (planted_dir / "theta").mkdir(parents=True)
    (planted_dir / "noise").mkdir(parents=True)

    rng = np.random.default_rng(555)

    # Designed loadings first, shaped like the real bank: one primary dimension
    # per item, a few cross-loaders, a few deliberately weak ones. The answers
    # are then generated from exactly these, so "designed" and "generating"
    # mean the same thing and item recovery has a real answer to find.
    n_items = 180
    designed = np.zeros((n_items, 8))
    classes = []
    for j in range(n_items):
        primary = rng.integers(0, 8)
        sign = rng.choice([-1.0, 1.0])
        if j % 17 == 0:
            designed[j, primary] = 0.25 * sign
            classes.append("weak")
        else:
            designed[j, primary] = 1.0 * sign
            if j % 7 == 0:
                secondary = (primary + rng.integers(1, 8)) % 8
                designed[j, primary] = 0.7 * sign
                designed[j, secondary] = 0.5 * rng.choice([-1.0, 1.0])
            classes.append("strong")

    data, theta8, params = make_population(
        n_personas=320, n_likert=120, n_binary=60, dims=8, seed=77,
        loadings=designed * 1.8,
    )

    train_ids = data.persona_ids[:250]
    holdout_ids = data.persona_ids[250:]
    train_items = data.item_ids[:140]
    holdout_items = data.item_ids[140:]

    for row, pid in enumerate(data.persona_ids):
        (planted_dir / "theta" / f"{pid}.json").write_text(
            json.dumps({"pid": pid, "theta": {d: float(theta8[row, i]) for i, d in enumerate(DIMS)}}),
            encoding="utf-8",
        )
        (planted_dir / "noise" / f"{pid}.json").write_text(
            json.dumps({"pid": pid, "wobble": float(rng.uniform(0.05, 0.4))}), encoding="utf-8"
        )

    items = []
    for j, iid in enumerate(data.item_ids):
        items.append(
            {
                "item_id": iid,
                "type": data.item_kinds[j],
                "loadings": {
                    DIMS[d]: float(designed[j, d]) for d in range(8) if designed[j, d] != 0
                },
                "strength_class": classes[j],
            }
        )
    (planted_dir / "bank_truth.json").write_text(json.dumps({"items": items}), encoding="utf-8")

    splits_file = workspace / "splits.json"
    splits_file.write_text(
        json.dumps(
            {
                "persona_holdout": holdout_ids,
                "item_holdout": holdout_items,
                "strata": {
                    iid: ["near", "same-domain", "far"][k % 3]
                    for k, iid in enumerate(holdout_items)
                },
            }
        ),
        encoding="utf-8",
    )

    train = data.subset(train_ids, train_items)
    fit = mirt.fit_joint(train, dims=8, seed=0, max_iters=3000)
    _, covs, _ = mirt.laplace_covariance(train, fit.params, fit.theta)
    scale = mirt.link_scale(fit.theta, covs)
    theta_train, linked, cov_train = mirt.apply_link(scale, fit.theta, fit.params, covs)

    held_out = data.subset(holdout_ids, train_items)
    theta_ho, _ = mirt.score_personas(held_out, linked, dims=8, seed=0)
    blur_ho, cov_ho, _ = mirt.laplace_covariance(held_out, linked, theta_ho)
    ho_params, _ = mirt.fit_items(
        data.subset(train_ids, holdout_items), theta_train, dims=8, seed=0
    )

    fit_file = workspace / "fit.npz"
    np.savez_compressed(
        fit_file,
        persona_train_ids=np.array(train_ids),
        persona_holdout_ids=np.array(holdout_ids),
        item_train_ids=np.array(train_items),
        item_holdout_ids=np.array(holdout_items),
        theta_train=theta_train,
        theta_holdout=theta_ho,
        blur_holdout=blur_ho,
        cov_holdout=cov_ho,
        train_loadings=linked.loadings,
        holdout_loadings=ho_params.loadings,
        train_item_kinds=np.array(linked.kinds),
        holdout_item_kinds=np.array(ho_params.kinds),
    )

    report = recovery.grade(
        fit_file, planted_dir, splits_file, workspace / "out", make_plots=True
    )
    return report, workspace


def test_grading_an_invented_population_passes_its_own_bars(graded) -> None:
    report, _ = graded
    verdict = report["verdict"]
    assert verdict["bars"]["trait_recovery"]["pass"] is True, verdict["bars"]["trait_recovery"]
    assert verdict["bars"]["item_recovery"]["pass"] is True, verdict["bars"]["item_recovery"]
    assert report["trait_recovery_holdout"]["min_r"] > 0.9


def test_the_grader_writes_its_report_and_both_plots(graded) -> None:
    _, workspace = graded
    out = workspace / "out"
    assert (out / "stage2_recovery.json").is_file()
    assert (out / "stage2_heatmap.png").stat().st_size > 5000
    assert (out / "stage2_scatter.png").stat().st_size > 5000
    payload = json.loads((out / "stage2_recovery.json").read_text(encoding="utf-8"))
    assert payload["verdict"]["bars"].keys() == {
        "trait_recovery", "item_recovery", "blur_honesty", "weak_item_ordering"
    }
    assert "failed_bars" in payload["verdict"]


def test_the_report_carries_every_block_the_gate_asks_for(graded) -> None:
    report, _ = graded
    for key in (
        "verdict",
        "alignment",
        "trait_recovery_holdout",
        "item_recovery_holdout",
        "blur_honesty",
        "weak_item_ordering",
        "watch_list",
        "correlations",
        "item_recovery_by_stratum",
    ):
        assert key in report, f"missing block: {key}"
    for key in ("distractors", "sign_check", "planted_zero_pair", "near_flat_raw_items"):
        assert key in report["watch_list"]
    assert set(report["item_recovery_by_stratum"]) == {"near", "same-domain", "far"}
    assert len(report["correlations"]["planted"]) == 8


def test_the_alignment_really_was_learned_on_the_training_personas(graded) -> None:
    report, workspace = graded
    fit = np.load(workspace / "fit.npz", allow_pickle=False)
    planted_train = recovery.load_planted_theta(
        workspace / "planted", [str(p) for p in fit["persona_train_ids"]]
    )
    expected = recovery.procrustes_rotation(fit["theta_train"], planted_train)
    assert np.allclose(np.array(report["alignment"]["rotation"]), expected)
    assert abs(abs(report["alignment"]["rotation_determinant"]) - 1.0) < 1e-8


def test_grading_survives_a_solution_that_is_rotated_before_it_arrives(graded) -> None:
    """Grading is invariant to the arbitrary rotation the fit happened to land on."""
    report, workspace = graded
    fit = dict(np.load(workspace / "fit.npz", allow_pickle=False))
    rng = np.random.default_rng(31)
    rotation, _ = np.linalg.qr(rng.normal(size=(8, 8)))

    fit["theta_train"] = fit["theta_train"] @ rotation
    fit["theta_holdout"] = fit["theta_holdout"] @ rotation
    fit["cov_holdout"] = recovery.rotate_covariances(fit["cov_holdout"], rotation)
    fit["train_loadings"] = fit["train_loadings"] @ rotation
    fit["holdout_loadings"] = fit["holdout_loadings"] @ rotation
    spun = workspace / "fit_rotated.npz"
    np.savez_compressed(spun, **fit)

    again = recovery.grade(
        spun, workspace / "planted", workspace / "splits.json", workspace / "out2", make_plots=False
    )
    assert again["trait_recovery_holdout"]["min_r"] == pytest.approx(
        report["trait_recovery_holdout"]["min_r"], abs=1e-8
    )
    assert again["item_recovery_holdout"]["median_r"] == pytest.approx(
        report["item_recovery_holdout"]["median_r"], abs=1e-8
    )
    assert again["blur_honesty"]["pooled_coverage"] == pytest.approx(
        report["blur_honesty"]["pooled_coverage"]
    )
