"""Smoke tests for the local dashboard (app/dashboard.py).

Streamlit's own ``AppTest`` runs the script in-process, so these check the one
thing that matters for a monitoring page: it renders without raising, both when
the results files are there and when they are not. Nothing here asserts a
research number -- those are graded in the eval tests, and the page only
displays what the report has already decided.

For the missing-files case the script is re-run with its repository root
pointed at an empty temporary directory, so the test does not depend on what
happens to sit in ``results/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = REPO_ROOT / "app" / "dashboard.py"

#: Gate 2 attempt 2 -- what the page shows as the current state.
RECOVERY_REPORT = REPO_ROOT / "results" / "stage2_v2_recovery.json"
#: Gate 2 attempt 1 -- kept on disk and shown beside it.
PREVIOUS_RECOVERY_REPORT = REPO_ROOT / "results" / "stage2_recovery.json"
RECOVERY_PICTURES = ("stage2_v2_scatter.png", "stage2_v2_heatmap.png")

#: Stage 3 -- the Gate 3 report and the item encoder graded beside it.
GATE3_REPORT = REPO_ROOT / "results" / "stage3_gate3.json"
ITEM_ENCODER_REPORT = REPO_ROOT / "results" / "stage3_item_encoder.json"
GATE3_PICTURES = ("stage3_blur_vs_n.png", "stage3_gate3_curves.png")

#: Stage 4 -- the Gate 4 report and the two pictures written next to it.
GATE4_REPORT = REPO_ROOT / "results" / "stage4_gate4.json"
GATE4_PICTURES = ("stage4_reliability.png", "stage4_aggregate.png")

#: The wording the monotonicity row must carry, exactly.
MONOTONICITY_RULING = "orchestrator ruling on qualitative wording; both readings reported"

RECOVERY_PAGE = "2. Recovery"
PERSON_ENCODER_PAGE = "3. Person encoder"
CALIBRATION_PAGE = "4. Calibration"

#: The line in the dashboard that decides where every file it opens lives.
ROOT_LINE = "REPO_ROOT = Path(__file__).resolve().parents[1]"


def run_app(root: Path | None = None) -> AppTest:
    """The dashboard, optionally rooted somewhere with no results in it."""
    if root is None:
        app = AppTest.from_file(str(DASHBOARD), default_timeout=60.0)
    else:
        source = DASHBOARD.read_text(encoding="utf-8")
        assert ROOT_LINE in source, "the dashboard no longer sets its root the way this test patches"
        app = AppTest.from_string(
            source.replace(ROOT_LINE, f"REPO_ROOT = Path({str(root)!r})"),
            default_timeout=60.0,
        )
    app.run()
    return app


def open_recovery(app: AppTest) -> AppTest:
    """Flip the sidebar selector to page 2 and re-run."""
    app.sidebar.radio[0].set_value(RECOVERY_PAGE).run()
    return app


def open_person_encoder(app: AppTest) -> AppTest:
    """Flip the sidebar selector to page 3 and re-run."""
    app.sidebar.radio[0].set_value(PERSON_ENCODER_PAGE).run()
    return app


def open_calibration(app: AppTest) -> AppTest:
    """Flip the sidebar selector to page 4 and re-run."""
    app.sidebar.radio[0].set_value(CALIBRATION_PAGE).run()
    return app


def all_text(app: AppTest) -> str:
    """Every string the page put on screen, headings and callouts included."""
    parts: list[str] = []
    for group in (app.title, app.header, app.subheader, app.markdown, app.caption):
        parts.extend(str(element.value) for element in group)
    for group in (app.info, app.warning, app.error, app.success):
        parts.extend(str(element.value) for element in group)
    return "\n".join(parts)


def test_dashboard_boots() -> None:
    """Page 1 renders against whatever is in results/ right now."""
    app = run_app()
    assert not app.exception
    assert app.title[0].value == "GLASSBOX monitor"
    assert "Population" in all_text(app)


def test_recovery_page_renders_with_the_real_report() -> None:
    """Page 2 against results/stage2_v2_recovery.json as it actually stands."""
    if not RECOVERY_REPORT.is_file():
        pytest.skip("no Stage 2 recovery report in this checkout")

    verdict = json.loads(RECOVERY_REPORT.read_text(encoding="utf-8")).get("verdict") or {}
    bars = verdict.get("bars") or {}

    app = open_recovery(run_app())
    assert not app.exception

    text = all_text(app)
    for heading in (
        "Gate 2 verdict",
        "Attempt 1 vs attempt 2",
        "Trait recovery per dimension",
        "How many dimensions the answers support",
        "Item recovery and blur",
        "Watch list",
    ):
        assert heading in text, f"the recovery page is missing its {heading!r} panel"

    # the page says which attempt it is showing, and names both reports
    assert "attempt 2" in text
    assert RECOVERY_REPORT.name in text
    assert PREVIOUS_RECOVERY_REPORT.name in text

    # the banner states the report's own verdict, and does not soften it
    if verdict.get("all_bars_pass"):
        assert any("Gate 2: PASS" in str(e.value) for e in app.success)
    else:
        assert any("Gate 2: FAIL" in str(e.value) for e in app.error)
        assert not app.success

    # one verdict row per frozen bar, carrying the report's own pass or fail
    frame = app.dataframe[0].value
    assert list(frame["bar"]) == [key.replace("_", " ") for key in bars]
    assert list(frame["verdict"]) == [
        "PASS" if entry.get("pass") is True
        else "FAIL" if entry.get("pass") is False
        else "not evaluated"
        for entry in bars.values()
    ]


def test_the_comparison_table_carries_both_attempts() -> None:
    """The four bars are shown attempt 1 next to attempt 2, from the two files."""
    if not (RECOVERY_REPORT.is_file() and PREVIOUS_RECOVERY_REPORT.is_file()):
        pytest.skip("both Stage 2 attempts are needed for the comparison")

    now = json.loads(RECOVERY_REPORT.read_text(encoding="utf-8"))
    before = json.loads(PREVIOUS_RECOVERY_REPORT.read_text(encoding="utf-8"))
    bars = (now.get("verdict") or {}).get("bars") or {}
    old_bars = (before.get("verdict") or {}).get("bars") or {}

    app = open_recovery(run_app())
    assert not app.exception

    # dataframe 0 is the verdict table, dataframe 1 the attempt comparison
    frame = app.dataframe[1].value
    assert list(frame["bar"]) == [key.replace("_", " ") for key in bars]
    for column, source in (
        ("attempt 1 value", old_bars),
        ("attempt 2 value", bars),
    ):
        assert column in frame.columns
        assert list(frame[column]) == [
            "--" if (source.get(key) or {}).get("value") is None
            else f"{float(source[key]['value']):.3f}"
            for key in bars
        ], f"{column} does not match the report it comes from"


def test_recovery_page_survives_missing_files(tmp_path: Path) -> None:
    """A fresh clone gets the how-to-produce-it note, not a traceback."""
    (tmp_path / "results").mkdir()

    app = open_recovery(run_app(root=tmp_path))
    assert not app.exception

    text = all_text(app)
    assert "python -m src.eval.recovery" in text
    for picture in RECOVERY_PICTURES:
        assert f"{picture} not found" in text


def test_recovery_page_survives_a_missing_first_attempt(tmp_path: Path) -> None:
    """Attempt 2 alone still renders; the comparison says what is missing."""
    if not RECOVERY_REPORT.is_file():
        pytest.skip("no Stage 2 recovery report in this checkout")
    results = tmp_path / "results"
    results.mkdir()
    (results / RECOVERY_REPORT.name).write_text(
        RECOVERY_REPORT.read_text(encoding="utf-8"), encoding="utf-8"
    )

    app = open_recovery(run_app(root=tmp_path))
    assert not app.exception

    text = all_text(app)
    assert "Gate 2 verdict" in text
    assert "Trait recovery per dimension" in text
    assert f"{PREVIOUS_RECOVERY_REPORT.name} is not in results/" in text


def test_person_encoder_page_renders_with_the_real_report() -> None:
    """Page 3 against results/stage3_gate3.json as it actually stands."""
    if not GATE3_REPORT.is_file():
        pytest.skip("no Gate 3 report in this checkout")

    verdict = json.loads(GATE3_REPORT.read_text(encoding="utf-8")).get("verdict") or {}
    bars = verdict.get("bars") or {}

    app = open_person_encoder(run_app())
    assert not app.exception

    text = all_text(app)
    for heading in ("Gate 3 verdict", "Pictures", "RMSE per dimension", "Item encoder"):
        assert heading in text, f"the person encoder page is missing its {heading!r} panel"
    assert GATE3_REPORT.name in text

    # the banner states the report's own verdict, and does not soften it
    if verdict.get("failed_bars"):
        assert any("Gate 3: FAIL" in str(e.value) for e in app.error)
        assert not app.success
    else:
        assert any("Gate 3: PASS" in str(e.value) for e in app.success)

    # one verdict row per frozen bar, carrying the report's own pass or fail --
    # except the undecided one, which carries the ruling and says so
    frame = app.dataframe[0].value
    assert list(frame["bar"]) == [key.replace("_", " ") for key in bars]
    for (key, entry), shown in zip(bars.items(), frame["verdict"]):
        if entry.get("pass") is True:
            assert shown == "PASS"
        elif entry.get("pass") is False:
            assert shown == "FAIL"
        else:
            assert shown == f"PASS -- {MONOTONICITY_RULING}"

    # the report's own strict readings stay on screen next to that ruling
    undecided = [entry for entry in bars.values() if entry.get("pass") is None]
    for entry in undecided:
        assert str(entry.get("mean_curve_verdict")) in text
        assert str(entry.get("strict_per_step_verdict")) in text


def test_the_rmse_table_carries_both_tracks() -> None:
    """Per-dimension RMSE is fused beside closed-only, from the two tracks."""
    if not GATE3_REPORT.is_file():
        pytest.skip("no Gate 3 report in this checkout")

    tracks = json.loads(GATE3_REPORT.read_text(encoding="utf-8")).get("tracks") or {}
    fused = ((tracks.get("fused") or {}).get("bars") or {}).get("rmse") or {}
    closed = ((tracks.get("closed_only") or {}).get("bars") or {}).get("rmse") or {}
    per_dimension = fused.get("per_dimension_rmse") or {}
    if not per_dimension:
        pytest.skip("this Gate 3 report has no per-dimension RMSE")

    app = open_person_encoder(run_app())
    assert not app.exception

    # dataframe 0 is the verdict table, dataframe 1 the per-dimension RMSE
    frame = app.dataframe[1].value
    assert list(frame["dimension"]) == list(per_dimension)
    assert list(frame["fused"]) == [f"{float(v):.3f}" for v in per_dimension.values()]
    assert list(frame["closed-only"]) == [
        f"{float((closed.get('per_dimension_rmse') or {})[d]):.3f}" for d in per_dimension
    ]


def test_the_item_encoder_bars_come_from_their_own_report() -> None:
    """The two item bars are shown compactly, with the report's own verdicts."""
    if not (GATE3_REPORT.is_file() and ITEM_ENCODER_REPORT.is_file()):
        pytest.skip("both Stage 3 reports are needed for the item bars")

    bars = json.loads(ITEM_ENCODER_REPORT.read_text(encoding="utf-8")).get("bars") or {}

    app = open_person_encoder(run_app())
    assert not app.exception

    # dataframe 2 is the item-encoder table
    frame = app.dataframe[2].value
    assert list(frame["bar"]) == [key.replace("_", " ") for key in bars]
    assert list(frame["value"]) == [f"{float(e['value']):.3f}" for e in bars.values()]
    assert list(frame["verdict"]) == [
        "PASS" if entry.get("pass") is True
        else "FAIL" if entry.get("pass") is False
        else "not evaluated"
        for entry in bars.values()
    ]


def test_person_encoder_page_survives_missing_files(tmp_path: Path) -> None:
    """A fresh clone gets the how-to-produce-it notes, not a traceback."""
    (tmp_path / "results").mkdir()

    app = open_person_encoder(run_app(root=tmp_path))
    assert not app.exception

    text = all_text(app)
    assert "python -m src.eval.gate3" in text
    assert "python -m src.model.item_encoder" in text
    for picture in GATE3_PICTURES:
        assert f"{picture} not found" in text


def test_person_encoder_page_survives_a_missing_item_report(tmp_path: Path) -> None:
    """The Gate 3 report alone still renders; the item panel says what is missing."""
    if not GATE3_REPORT.is_file():
        pytest.skip("no Gate 3 report in this checkout")
    results = tmp_path / "results"
    results.mkdir()
    (results / GATE3_REPORT.name).write_text(
        GATE3_REPORT.read_text(encoding="utf-8"), encoding="utf-8"
    )

    app = open_person_encoder(run_app(root=tmp_path))
    assert not app.exception

    text = all_text(app)
    assert "Gate 3 verdict" in text
    assert "RMSE per dimension" in text
    assert f"{ITEM_ENCODER_REPORT.name} not found" in text


def test_calibration_page_renders_with_the_real_report() -> None:
    """Page 4 against results/stage4_gate4.json as it actually stands."""
    if not GATE4_REPORT.is_file():
        pytest.skip("no Gate 4 report in this checkout")

    verdict = json.loads(GATE4_REPORT.read_text(encoding="utf-8")).get("verdict") or {}
    bars = verdict.get("bars") or {}

    app = open_calibration(run_app())
    assert not app.exception

    text = all_text(app)
    for heading in (
        "Gate 4 verdict",
        "Reliability diagram",
        "Per-stratum performance",
        "Aggregate check",
        "Qwen no-interview baseline",
    ):
        assert heading in text, f"the calibration page is missing its {heading!r} panel"
    assert GATE4_REPORT.name in text

    # the banner states the report's own verdict, and does not soften it
    if verdict.get("all_bars_pass"):
        assert any("Gate 4: PASS" in str(e.value) for e in app.success)
    else:
        assert any("Gate 4: FAIL" in str(e.value) for e in app.error)
        assert not app.success

    # one verdict row per frozen bar, carrying the report's own pass or fail
    frame = app.dataframe[0].value
    assert list(frame["bar"]) == [key.replace("_", " ") for key in bars]
    assert list(frame["verdict"]) == [
        "PASS" if entry.get("pass") is True
        else "FAIL" if entry.get("pass") is False
        else "not evaluated"
        for entry in bars.values()
    ]
    assert list(frame["value"]) == [
        f"{float(entry['value']):.3f}"
        + (f" ({entry['value_label']})" if entry.get("value_label") else "")
        for entry in bars.values()
    ]


def test_the_stratum_panel_carries_the_knn_reading() -> None:
    """Per-stratum lift, with the strata k-NN wins named on screen."""
    if not GATE4_REPORT.is_file():
        pytest.skip("no Gate 4 report in this checkout")

    report = json.loads(GATE4_REPORT.read_text(encoding="utf-8"))
    per_stratum = report.get("per_stratum_comparison") or {}
    proximity = ((report.get("verdict") or {}).get("knn_proximity")) or {}
    if not per_stratum:
        pytest.skip("this Gate 4 report has no per-stratum comparison")

    app = open_calibration(run_app())
    assert not app.exception

    text = all_text(app)
    assert str(proximity.get("reading", "")) in text
    for stratum in proximity.get("strata_where_knn_wins") or []:
        assert any(stratum in str(e.value) for e in app.warning), (
            f"the page does not say k-NN wins {stratum!r}"
        )

    # dataframes: 0 verdict, 1 arms, 2 calibration bins, 3 per-stratum
    frame = app.dataframe[3].value
    assert set(frame["stratum"]) == set(per_stratum)
    for stratum, shown in zip(frame["stratum"], frame["model lift"]):
        assert shown == f"{float(per_stratum[stratum]['lift_vs_marginal']):.3f}"
    for stratum, shown in zip(frame["stratum"], frame["model beats k-NN"]):
        assert shown == ("yes" if per_stratum[stratum]["model_beats_knn"] else "no")


def test_the_aggregate_numbers_come_from_the_report() -> None:
    """The two headline correlations are shown as the report wrote them."""
    if not GATE4_REPORT.is_file():
        pytest.skip("no Gate 4 report in this checkout")

    report = json.loads(GATE4_REPORT.read_text(encoding="utf-8"))
    aggregate = report.get("aggregate_check") or {}
    cross_tabs = report.get("cross_tabs") or {}
    if not (aggregate and cross_tabs):
        pytest.skip("this Gate 4 report has no aggregate check")

    app = open_calibration(run_app())
    assert not app.exception

    shown = [str(element.value) for element in app.metric]
    assert f"{float(aggregate['correlation']):.4f}" in shown
    assert f"{float(cross_tabs['correlation']):.4f}" in shown
    assert str(cross_tabs.get("conditional_independence")) in all_text(app)


def test_the_qwen_slot_renders_whether_or_not_the_file_has_landed() -> None:
    """The no-interview baseline: a pending note, or its numbers once graded."""
    if not GATE4_REPORT.is_file():
        pytest.skip("no Gate 4 report in this checkout")

    report = json.loads(GATE4_REPORT.read_text(encoding="utf-8"))
    block = report.get("qwen_no_interview_baseline") or {}
    arm = (report.get("arms") or {}).get("qwen_no_interview") or {}

    app = open_calibration(run_app())
    assert not app.exception

    text = all_text(app)
    assert "Qwen no-interview baseline" in text

    if str(block.get("status", "")).lower() == "pending":
        assert "Pending" in text
        assert str(block.get("note", "")).rstrip(" .") in text
        return

    # graded: the numbers come from the arm, never quoted without the marginal
    assert arm, "the report grades the baseline but carries no arm for it"
    frame = app.dataframe[4].value
    assert list(frame["arm"]) == ["qwen no-interview", "marginal"]
    assert frame["Brier"][0] == f"{float(arm['brier']):.4f}"
    marginal = (report.get("arms") or {}).get("marginal") or {}
    assert frame["Brier"][1] == f"{float(marginal['brier']):.4f}"


def test_the_arms_table_matches_the_reliability_picture() -> None:
    """Every pre-registered arm the report carries is in the table beside it."""
    if not GATE4_REPORT.is_file():
        pytest.skip("no Gate 4 report in this checkout")

    arms = json.loads(GATE4_REPORT.read_text(encoding="utf-8")).get("arms") or {}
    expected = [
        name
        for name in ("model", "marginal", "profile", "knn", "qwen_no_interview")
        if arms.get(name)
    ]

    app = open_calibration(run_app())
    assert not app.exception

    # dataframe 1 is the per-arm table next to the reliability diagram
    frame = app.dataframe[1].value
    assert list(frame["arm"]) == expected
    assert list(frame["Brier"]) == [f"{float(arms[name]['brier']):.4f}" for name in expected]


def test_calibration_page_survives_missing_files(tmp_path: Path) -> None:
    """A fresh clone gets the how-to-produce-it note, not a traceback."""
    (tmp_path / "results").mkdir()

    app = open_calibration(run_app(root=tmp_path))
    assert not app.exception

    text = all_text(app)
    assert "python -m src.eval.gate4" in text
    for picture in GATE4_PICTURES:
        assert f"{picture} not found" in text


def test_calibration_page_survives_missing_pictures(tmp_path: Path) -> None:
    """The Gate 4 report alone still renders; the panels say what is missing."""
    if not GATE4_REPORT.is_file():
        pytest.skip("no Gate 4 report in this checkout")
    results = tmp_path / "results"
    results.mkdir()
    (results / GATE4_REPORT.name).write_text(
        GATE4_REPORT.read_text(encoding="utf-8"), encoding="utf-8"
    )

    app = open_calibration(run_app(root=tmp_path))
    assert not app.exception

    text = all_text(app)
    assert "Gate 4 verdict" in text
    assert "Per-stratum performance" in text
    for picture in GATE4_PICTURES:
        assert f"{picture} not found" in text
