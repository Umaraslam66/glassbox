"""Unit tests for the Gate 3 person-encoder grader.

Everything here runs on a world this file makes up. One test builds the whole
pipeline end to end -- a synthetic population, a fit archive, transcripts,
open-answer features, a planted-truth directory -- runs the fusion and then the
grader over it, and checks the shape of what comes out. The rest pin the pieces
that a wrong answer would hide inside:

  * the alignment is **the same** orthogonal Procrustes convention Stage 2 used,
    checked by putting both graders' rotations on one shared fixture and
    demanding bit equality;
  * the calibration enters the rotated blur the only way that survives a change
    of basis, and reproduces the artifact's per-dimension numbers before it;
  * each bar's arithmetic is checked against a case whose answer is known by
    hand, including its edges;
  * the monotonicity block reports both readings and picks neither;
  * the leakage hunt actually catches the things it claims to look for.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval import gate3, recovery
from src.model import mirt, open_fusion as of

from tests.test_person_encoder import make_items, sample_answers

DIMS = len(gate3.DIMENSIONS)


# --------------------------------------------------------------------------
# alignment: the same convention as Stage 2, not a lookalike
# --------------------------------------------------------------------------


def _tiny_archive(tmp_path: Path, theta_train, theta_holdout, cov_holdout, kappa) -> Path:
    """An estimates archive with just the fields ``aligned_track`` reads."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "fused.npz"
    np.savez_compressed(
        path,
        n_grid=np.array([1]),
        theta_train_fused_by_n=theta_train[None, :, :],
        theta_holdout_fused_by_n=theta_holdout[None, :, :],
        cov_holdout_fused_by_n=cov_holdout[None, :, :, :],
        blur_kappa_fused=kappa,
    )
    return path


def test_alignment_is_recovery_pys_rotation_on_a_shared_fixture(tmp_path: Path) -> None:
    """Both graders' rotations must be the same matrix, to the bit."""
    rng = np.random.default_rng(1)
    n_train, n_holdout = 120, 40
    theta_true_train = rng.normal(size=(n_train, DIMS))
    twist = np.linalg.qr(rng.normal(size=(DIMS, DIMS)))[0]
    theta_train = theta_true_train @ twist + rng.normal(size=(n_train, DIMS)) * 0.2
    theta_holdout = rng.normal(size=(n_holdout, DIMS))
    cov = np.broadcast_to(np.eye(DIMS) * 0.25, (n_holdout, DIMS, DIMS)).copy()
    kappa = np.ones(DIMS)

    archive = np.load(
        _tiny_archive(tmp_path, theta_train, theta_holdout, cov, kappa), allow_pickle=False
    )
    piece = gate3.aligned_track(archive, "fused", 1, theta_true_train)

    stage2 = recovery.procrustes_rotation(theta_train, theta_true_train)
    assert np.array_equal(piece["rotation"], stage2), (
        "Gate 3 is not using Stage 2's alignment"
    )
    assert np.allclose(piece["theta"], theta_holdout @ stage2)
    assert np.allclose(piece["theta_train"], theta_train @ stage2)

    # and the covariance goes through the same change of basis recovery.py uses
    expected = recovery.rotate_covariances(cov, stage2)
    assert np.allclose(
        piece["blur"], np.sqrt(np.diagonal(expected, axis1=1, axis2=2))
    )


def test_the_alignment_never_looks_at_the_holdout_personas(tmp_path: Path) -> None:
    """Replacing every held-out vector leaves the rotation untouched."""
    rng = np.random.default_rng(2)
    theta_true_train = rng.normal(size=(100, DIMS))
    theta_train = rng.normal(size=(100, DIMS))
    cov = np.broadcast_to(np.eye(DIMS), (10, DIMS, DIMS)).copy()

    first = gate3.aligned_track(
        np.load(
            _tiny_archive(tmp_path / "a", theta_train, rng.normal(size=(10, DIMS)), cov, np.ones(DIMS)),
            allow_pickle=False,
        ),
        "fused", 1, theta_true_train,
    )
    second = gate3.aligned_track(
        np.load(
            _tiny_archive(tmp_path / "b", theta_train, rng.normal(size=(10, DIMS)) * 9, cov, np.ones(DIMS)),
            allow_pickle=False,
        ),
        "fused", 1, theta_true_train,
    )
    assert np.array_equal(first["rotation"], second["rotation"])


def test_calibrated_covariance_reproduces_the_artifacts_blur_before_rotation() -> None:
    """``K Sigma K`` on the diagonal is exactly ``kappa * blur``, per dimension."""
    rng = np.random.default_rng(3)
    n = 25
    root = rng.normal(size=(n, DIMS, DIMS))
    cov = np.einsum("nij,nkj->nik", root, root) + np.eye(DIMS) * 0.1
    kappa = np.linspace(0.9, 1.4, DIMS)

    raw_blur = np.sqrt(np.diagonal(cov, axis1=1, axis2=2))
    calibrated = gate3.calibrated_covariance(cov, kappa)
    assert np.allclose(
        np.sqrt(np.diagonal(calibrated, axis1=1, axis2=2)), raw_blur * kappa[None, :]
    )
    # and it stays a covariance
    assert np.all(np.linalg.eigvalsh(calibrated) > 0)


# --------------------------------------------------------------------------
# the bars
# --------------------------------------------------------------------------


def test_rmse_bar_passes_only_when_every_dimension_clears() -> None:
    rng = np.random.default_rng(4)
    truth = rng.normal(size=(400, DIMS))
    good = truth + rng.normal(size=(400, DIMS)) * 0.2
    block = gate3.rmse_bar(good, truth)
    assert block["pass"] is True
    assert block["n_dimensions_passing"] == DIMS

    spoiled = good.copy()
    spoiled[:, 3] = truth[:, 3] + rng.normal(size=400) * 2.0
    block = gate3.rmse_bar(spoiled, truth)
    assert block["pass"] is False
    assert block["worst_dimension"] == gate3.DIMENSIONS[3]
    assert block["n_dimensions_passing"] == DIMS - 1


def test_rmse_bar_arithmetic_is_the_plain_definition() -> None:
    truth = np.zeros((3, DIMS))
    estimate = np.full((3, DIMS), 0.5)
    block = gate3.rmse_bar(estimate, truth)
    assert block["pooled_rmse"] == pytest.approx(0.5)
    for dim in gate3.DIMENSIONS:
        assert block["per_dimension_rmse"][dim] == pytest.approx(0.5)
    assert block["pass"] is True, "0.5 is inside the bar, which is <= 0.5"


def test_coverage_bar_counts_cells_and_respects_the_band() -> None:
    truth = np.zeros((100, DIMS))
    blur = np.ones((100, DIMS))
    estimate = np.zeros((100, DIMS))
    estimate[:30] = 2.0  # 30% of every dimension's cells fall outside

    block = gate3.coverage_bar(estimate, blur, truth)
    assert block["pooled_coverage"] == pytest.approx(0.70)
    assert block["n_cells"] == 100 * DIMS
    assert block["pass"] is True

    estimate[:50] = 2.0
    assert gate3.coverage_bar(estimate, blur, truth)["pass"] is False


def test_coverage_diagnostics_size_the_miss_without_changing_the_blur() -> None:
    rng = np.random.default_rng(5)
    truth = np.zeros((2000, DIMS))
    estimate = rng.normal(size=(2000, DIMS)) * 2.0
    blur = np.ones((2000, DIMS))
    block = gate3.coverage_bar(estimate, blur, truth)
    diagnostics = block["diagnostics"]
    assert diagnostics["variance_ratio_actual_over_predicted"] == pytest.approx(4.0, rel=0.1)
    assert diagnostics["blur_multiplier_for_nominal_68pct"] == pytest.approx(2.0, rel=0.15)
    assert block["mean_blur"] == pytest.approx(1.0), "the blur itself must not move"


def test_lift_bar_is_one_minus_the_rmse_ratio() -> None:
    truth = np.zeros((50, DIMS))
    encoder = np.full((50, DIMS), 0.5)
    baseline = np.full((50, DIMS), 1.0)
    block = gate3.lift_bar(encoder, baseline, truth)
    assert block["pooled_lift"] == pytest.approx(0.5)
    assert block["mean_of_per_dimension_lift"] == pytest.approx(0.5)
    assert block["pass"] is True

    block = gate3.lift_bar(np.full((50, DIMS), 0.75), baseline, truth)
    assert block["pooled_lift"] == pytest.approx(0.25)
    assert block["pass"] is False


def test_lift_bar_says_the_baseline_is_the_population_spread() -> None:
    """An intercept-only baseline's RMSE is the population RMS, and it is shown."""
    rng = np.random.default_rng(6)
    truth = rng.normal(size=(500, DIMS))
    baseline = np.zeros((500, DIMS))  # predicts the population mean for everyone
    block = gate3.lift_bar(truth * 0.5, baseline, truth)
    for dim in gate3.DIMENSIONS:
        assert block["baseline_rmse_per_dimension"][dim] == pytest.approx(
            block["population_rms_per_dimension"][dim]
        )
    assert "zero-information" in block["baseline_note"]


def test_shrinkage_diagnostic_is_reported_and_not_applied() -> None:
    rng = np.random.default_rng(7)
    truth = rng.normal(size=(500, DIMS))
    shrunk = truth * 0.6  # a perfectly ranked but shrunken estimate
    block = gate3.shrinkage_diagnostic(shrunk, truth)
    assert block["label"].startswith("EXPLORATORY")
    for dim in gate3.DIMENSIONS:
        row = block["per_dimension"][dim]
        assert row["regression_of_planted_on_estimate"] == pytest.approx(1 / 0.6, rel=1e-6)
        assert row["rmse_if_optimally_rescaled"] < row["rmse_as_measured"]


# --------------------------------------------------------------------------
# monotonicity: two readings, no verdict
# --------------------------------------------------------------------------


def _monotonicity_report(track: str, upticks: bool) -> dict:
    grid = list(range(1, 6))
    curve = np.tile(np.linspace(1.0, 0.5, len(grid))[:, None], (1, DIMS))
    if upticks:
        curve[2, 1] += 0.2
    stack = np.repeat(curve[:, None, :], 4, axis=1)
    from src.model.person_encoder import monotonicity_report

    return {
        "monotonicity": {
            track: {
                "holdout": monotonicity_report(grid, curve, blur_by_n=stack),
                "train": monotonicity_report(grid, curve),
            },
            "reading": "a note",
        }
    }


def test_monotonicity_block_reports_both_readings_and_picks_neither() -> None:
    report = _monotonicity_report("fused", upticks=True)
    block = gate3.monotonicity_readings(report, "fused")

    assert block["verdict"] is None
    assert "orchestrator" in block["note"]
    assert block["mean_curve_reading"]["verdict"] == "FAIL"
    assert block["mean_curve_reading"]["largest_uptick"] == pytest.approx(0.075, abs=1e-9)
    assert block["strict_per_step_reading"]["verdict"] == "FAIL"
    assert block["strict_per_step_reading"]["n_increases"] == 4  # one step, four personas


def test_monotonicity_block_reports_a_clean_curve_as_clean() -> None:
    block = gate3.monotonicity_readings(_monotonicity_report("fused", upticks=False), "fused")
    assert block["mean_curve_reading"]["verdict"] == "PASS"
    assert block["strict_per_step_reading"]["verdict"] == "PASS"
    assert block["verdict"] is None, "the grader must still not choose"


def test_uptick_sizes_are_reported_next_to_the_shrinkage_they_sit_on() -> None:
    block = gate3.monotonicity_readings(_monotonicity_report("fused", upticks=True), "fused")
    per_dim = block["mean_curve_reading"]["upticks_per_dimension"]
    noisy = per_dim[gate3.DIMENSIONS[1]]
    assert noisy["n_upticks"] == 1
    assert noisy["as_fraction_of_total_shrinkage"] == pytest.approx(0.075 / 0.5, rel=1e-6)
    assert per_dim[gate3.DIMENSIONS[0]]["n_upticks"] == 0


# --------------------------------------------------------------------------
# the end-to-end run
# --------------------------------------------------------------------------


def _write_world(tmp_path: Path, *, n_train: int = 200, n_holdout: int = 60, n_items: int = 15):
    """A whole synthetic study: fit archive, transcripts, features, truth, splits."""
    rng = np.random.default_rng(11)
    params = make_items(n_likert=n_items - 4, n_binary=4, dims=DIMS, seed=13)
    theta_true = rng.normal(size=(n_train + n_holdout, DIMS))
    codes = sample_answers(theta_true, params, seed=14)

    pids = [f"p{i:04d}" for i in range(n_train + n_holdout)]
    train_ids, holdout_ids = pids[:n_train], pids[n_train:]
    item_ids = [f"q{j:03d}" for j in range(n_items)]

    # the "full matrix" estimate: theta plus a little noise, standing in for a
    # long fit. The encoder's target and the grader's reference are different
    # things, exactly as they are in the real run.
    theta_hat = theta_true + rng.normal(size=theta_true.shape) * 0.25

    payload = mirt._params_payload(params, item_ids)
    fit_path = tmp_path / "fit.npz"
    np.savez_compressed(
        fit_path,
        dims=np.array(DIMS),
        persona_train_ids=np.array(train_ids),
        persona_holdout_ids=np.array(holdout_ids),
        item_train_ids=np.array(item_ids),
        item_holdout_ids=np.array([]),
        theta_train=theta_hat[:n_train],
        theta_holdout=theta_hat[n_train:],
        train_loadings=payload["loadings"],
        train_thresholds=payload["thresholds"],
        train_difficulty=payload["difficulty"],
        train_item_kinds=np.array(payload["kinds"]),
    )

    transcripts = tmp_path / "transcripts.jsonl"
    with transcripts.open("w", encoding="utf-8") as fh:
        for row, pid in enumerate(pids):
            turns = []
            for j, (iid, kind) in enumerate(zip(item_ids, params.kinds)):
                code = int(codes[row, j])
                answer = code + 1 if kind == mirt.LIKERT else ("yes" if code else "no")
                turns.append({"n": j + 1, "item_id": iid, "question": "?", "answer": answer})
            fh.write(json.dumps({"pid": pid, "profile": {"pid": pid}, "turns": turns, "open": []}) + "\n")

    # judgments that really do read theta, plus noise -- three prompts, eight poles
    loading = rng.normal(size=(3 * DIMS, DIMS))
    features = (
        theta_true @ loading.T
        + rng.normal(size=(len(pids), 3 * DIMS)) * 1.2
        + rng.normal(size=(len(pids), 1))  # a shared reader wobble
    )
    features_path = tmp_path / "features.npz"
    np.savez_compressed(
        features_path,
        pids=np.array(pids),
        prompt_ids=np.array(["oe01", "oe02", "oe03"]),
        poles=np.array(list(gate3.DIMENSIONS)),
        scores=features.reshape(len(pids), 3, DIMS),
        observed=np.ones((len(pids), 3, DIMS), dtype=bool),
        features=features,
    )

    splits_path = tmp_path / "splits.json"
    splits_path.write_text(
        json.dumps(
            {
                "persona_holdout": holdout_ids,
                "interview_closed": item_ids,
                "strata": {},
            }
        ),
        encoding="utf-8",
    )

    truth_dir = tmp_path / "planted"
    (truth_dir / "theta").mkdir(parents=True)
    for row, pid in enumerate(pids):
        (truth_dir / "theta" / f"{pid}.json").write_text(
            json.dumps(
                {"theta": {dim: float(theta_true[row, d]) for d, dim in enumerate(gate3.DIMENSIONS)}}
            ),
            encoding="utf-8",
        )

    backbone_path = tmp_path / "backbone.npz"
    np.savez_compressed(
        backbone_path,
        theta_baseline_holdout=np.tile(theta_hat[:n_train].mean(axis=0), (n_holdout, 1)),
        theta_holdout_by_n=np.zeros((n_items, n_holdout, DIMS)),
        blur_holdout_by_n=np.ones((n_items, n_holdout, DIMS)),
        blur_kappa=np.ones(DIMS),
    )
    return {
        "fit": fit_path,
        "transcripts": transcripts,
        "features": features_path,
        "splits": splits_path,
        "truth": truth_dir,
        "backbone": backbone_path,
    }


@pytest.fixture(scope="module")
def graded(tmp_path_factory) -> dict:
    """Run the fusion and then the grader over one synthetic study."""
    tmp_path = tmp_path_factory.mktemp("gate3")
    world = _write_world(tmp_path)
    of.run(
        world["transcripts"],
        world["features"],
        world["fit"],
        world["splits"],
        tmp_path / "fused.json",
        verbose=False,
    )
    report = gate3.grade(
        tmp_path / "fused.json",
        world["backbone"],
        world["fit"],
        world["truth"],
        world["splits"],
        tmp_path / "out",
        fused_npz=tmp_path / "fused.npz",
        features_path=world["features"],
        make_plots=True,
    )
    return {"report": report, "out": tmp_path / "out", "world": world}


def test_the_verdict_block_has_all_four_bars_in_the_stage2_style(graded) -> None:
    verdict = graded["report"]["verdict"]
    assert set(verdict["bars"]) == {"rmse", "blur_coverage", "monotone_blur", "lift"}
    for name, bar in verdict["bars"].items():
        assert "statement" in bar and "value" in bar and "bar" in bar and "pass" in bar
    assert verdict["confirmatory_track"] == "fused"
    assert verdict["confirmatory_n"] == 15
    assert verdict["bars"]["monotone_blur"]["pass"] is None
    assert verdict["undecided_bars"] == ["monotone_blur"]
    assert isinstance(verdict["all_decided_bars_pass"], bool)
    assert "closed_only_for_comparison" in verdict


def test_both_tracks_are_graded_and_the_fused_one_is_better(graded) -> None:
    tracks = graded["report"]["tracks"]
    assert set(tracks) == {"fused", "closed_only"}
    fused = tracks["fused"]["bars"]["rmse"]["pooled_rmse"]
    closed = tracks["closed_only"]["bars"]["rmse"]["pooled_rmse"]
    assert fused < closed, (fused, closed)
    assert tracks["fused"]["bars"]["lift"]["pooled_lift"] > tracks["closed_only"]["bars"]["lift"]["pooled_lift"]


def test_the_projection_is_quoted_next_to_the_outcome(graded) -> None:
    projection = graded["report"]["tracks"]["fused"]["bars"]["rmse"]["projection"]
    assert "PROJECTION, FILED BEFORE ANY STAGE 3 MEASUREMENT" in projection["quote"]
    assert set(projection["filed"]) == set(gate3.DIMENSIONS)
    assert set(projection["per_dimension_outcome_minus_projection"]) == set(gate3.DIMENSIONS)
    assert "independent sitting" in projection["independent_sitting_correction"]


def test_the_curves_cover_every_n_and_the_plots_are_written(graded) -> None:
    curves = graded["report"]["curves"]
    for track in ("fused", "closed_only"):
        assert [row["n"] for row in curves[track]] == list(range(1, 16))
        for row in curves[track]:
            assert 0.0 <= row["pooled_coverage"] <= 1.0
    assert (graded["out"] / "stage3_blur_vs_n.png").is_file()
    assert (graded["out"] / "stage3_gate3_curves.png").is_file()
    assert (graded["out"] / "stage3_gate3.json").is_file()


def test_the_report_restates_the_full_matrix_gap(graded) -> None:
    gap = graded["report"]["transcript_vs_full_matrix"]
    assert gap["full_matrix_pooled_rmse"] == pytest.approx(
        graded["report"]["full_matrix_reference"]["pooled_rmse"]
    )
    assert gap["full_matrix_pooled_rmse"] < gap["n15_fused_pooled_rmse"]
    assert "r_per_dim" in gap["system_side_agreement_at_n15"]["fused"]


def test_the_leakage_hunt_runs_and_clears_a_clean_world(graded) -> None:
    hunt = graded["report"]["leakage_hunt"]
    assert hunt["checks_run_regardless"] is True
    assert hunt["splits"]["verdict"] == "clean"
    assert hunt["splits"]["train_holdout_overlap"] == []
    assert "clean" in hunt["measurement_model_scope"]["verdict"]
    assert hunt["feature_provenance"]["available"] is True
    assert set(hunt["feature_provenance"]["per_pole"]) == set(gate3.DIMENSIONS)


def test_the_renderer_prints_every_bar(graded) -> None:
    text = gate3.render(graded["report"])
    for name in ("rmse", "blur_coverage", "monotone_blur", "lift"):
        assert name in text
    assert "GATE 3 VERDICT" in text
    assert "leakage hunt" in text


# --------------------------------------------------------------------------
# the hunt catches what it says it catches
# --------------------------------------------------------------------------


def _hunt_inputs(graded):
    world = graded["world"]
    fused_report = json.loads((graded["out"].parent / "fused.json").read_text(encoding="utf-8"))
    archive = np.load(graded["out"].parent / "fused.npz", allow_pickle=False)
    backbone = np.load(world["backbone"], allow_pickle=False)
    fit = np.load(world["fit"], allow_pickle=False)
    splits = json.loads(world["splits"].read_text(encoding="utf-8"))
    train_ids = [str(p) for p in archive["persona_train_ids"]]
    holdout_ids = [str(p) for p in archive["persona_holdout_ids"]]
    truth_train = recovery.load_planted_theta(world["truth"], train_ids)
    truth_holdout = recovery.load_planted_theta(world["truth"], holdout_ids)
    bars = graded["report"]["tracks"]["fused"]["bars"]
    return fused_report, archive, backbone, fit, splits, truth_train, truth_holdout, bars


def test_the_hunt_catches_a_split_violation(graded, tmp_path: Path) -> None:
    fused_report, archive, backbone, fit, splits, truth_train, truth_holdout, bars = _hunt_inputs(graded)
    broken = dict(archive)
    broken["persona_holdout_ids"] = np.array(
        [str(archive["persona_train_ids"][0])] + [str(p) for p in archive["persona_holdout_ids"][1:]]
    )

    hunt = gate3.leakage_hunt(
        fused_report, broken, backbone, None, fit, splits, truth_train, truth_holdout, bars
    )
    assert hunt["splits"]["verdict"] == "SPLIT VIOLATION"
    assert hunt["verdict"].startswith("FOLLOW UP")


def test_the_hunt_catches_a_feature_that_reads_planted_truth_too_well(
    graded, tmp_path: Path
) -> None:
    """A judgment feature copied straight from planted theta must be caught."""
    fused_report, archive, backbone, fit, splits, truth_train, truth_holdout, bars = _hunt_inputs(graded)
    pids = [str(p) for p in archive["persona_train_ids"]] + [
        str(p) for p in archive["persona_holdout_ids"]
    ]
    planted = np.vstack([truth_train, truth_holdout])
    leaked = np.tile(planted, (1, 3)).reshape(len(pids), 3, DIMS)
    leaked = np.transpose(leaked, (0, 1, 2)).reshape(len(pids), 3 * DIMS)
    for q in range(3):
        leaked[:, q * DIMS : (q + 1) * DIMS] = planted

    path = tmp_path / "leaked.npz"
    np.savez_compressed(
        path,
        pids=np.array(pids),
        prompt_ids=np.array(["oe01", "oe02", "oe03"]),
        poles=np.array(list(gate3.DIMENSIONS)),
        scores=leaked.reshape(len(pids), 3, DIMS),
        observed=np.ones((len(pids), 3, DIMS), dtype=bool),
        features=leaked,
    )
    hunt = gate3.leakage_hunt(
        fused_report, archive, backbone, path, fit, splits, truth_train, truth_holdout, bars
    )
    assert hunt["feature_provenance"]["verdict"].startswith("LOOK AGAIN")
    assert hunt["feature_provenance"]["largest_r_with_planted"] > gate3.CEILING_ALARM
    assert hunt["verdict"].startswith("FOLLOW UP")


def test_the_hunt_catches_a_moved_backbone(graded) -> None:
    fused_report, archive, backbone, fit, splits, truth_train, truth_holdout, bars = _hunt_inputs(graded)
    moved = dict(backbone)
    moved["theta_holdout_by_n"] = np.asarray(archive["theta_holdout_closed_only_by_n"]) + 1.0
    moved["blur_holdout_by_n"] = np.asarray(archive["blur_holdout_closed_only_by_n"])
    moved["blur_kappa"] = np.asarray(archive["blur_kappa_closed_only"])

    hunt = gate3.leakage_hunt(
        fused_report, archive, moved, None, fit, splits, truth_train, truth_holdout, bars
    )
    assert hunt["backbone_integrity"]["verdict"].startswith("LOOK AGAIN")


def test_the_hunt_trigger_fires_when_a_number_beats_its_projection(graded) -> None:
    fused_report, archive, backbone, fit, splits, truth_train, truth_holdout, bars = _hunt_inputs(graded)
    spectacular = json.loads(json.dumps(bars))
    for dim in gate3.DIMENSIONS:
        spectacular["rmse"]["per_dimension_rmse"][dim] = 0.1
    hunt = gate3.leakage_hunt(
        fused_report, archive, backbone, None, fit, splits, truth_train, truth_holdout, spectacular
    )
    assert hunt["trigger"]["fired"] is True
    assert set(hunt["trigger"]["dimensions_beating_projection_materially"]) == set(
        gate3.DIMENSIONS
    )
