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

#: Stage 5 -- the Gate 5 report, the RL run's own two reports, and the pictures.
GATE5_REPORT = REPO_ROOT / "results" / "stage5_strategies.json"
RL_TRAINING_REPORT = REPO_ROOT / "results" / "stage5_rl_training.json"
RL_WATCH_REPORT = REPO_ROOT / "results" / "stage5_rl_proxy_watch.json"
ORACLE_WATCH_REPORT = REPO_ROOT / "results" / "stage5_rl_exploratory_oracle_proxy_watch.json"
GATE5_PICTURES = (
    "stage5_rmse_vs_n.png",
    "stage5_blur_vs_truth.png",
    "stage5_rl_training.png",
    "stage5_rl_proxy_watch.png",
)
#: The pre-declared exploratory thresholds the tie is read at.
GATE5_THRESHOLDS = ("0.60", "0.65", "0.70")

#: The wording the monotonicity row must carry, exactly.
MONOTONICITY_RULING = "orchestrator ruling on qualitative wording; both readings reported"

#: The front page -- the six gate verdicts, read out of the reports below it.
#: Gate 1's confirmatory report is the one the overview reads for that row.
GATE1_REPORT = REPO_ROOT / "results" / "full_sweep_v2_qa.json"
GATE_ROWS = ("Gate 0", "Gate 1", "Gate 2", "Gate 3", "Gate 4", "Gate 5")

POPULATION_PAGE = "1. Population"
RECOVERY_PAGE = "2. Recovery"
PERSON_ENCODER_PAGE = "3. Person encoder"
CALIBRATION_PAGE = "4. Calibration"
INTERVIEWER_PAGE = "5. Interviewer"

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


def open_population(app: AppTest) -> AppTest:
    """Flip the sidebar selector to page 1 and re-run."""
    app.sidebar.radio[0].set_value(POPULATION_PAGE).run()
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


def open_interviewer(app: AppTest) -> AppTest:
    """Flip the sidebar selector to page 5 and re-run."""
    app.sidebar.radio[0].set_value(INTERVIEWER_PAGE).run()
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
    """The dashboard opens on the overview, against whatever is in results/."""
    app = run_app()
    assert not app.exception
    assert app.title[0].value == "GLASSBOX monitor"
    assert "Gates" in all_text(app)


def test_population_page_still_renders() -> None:
    """Page 1 is one click away from the new front page."""
    app = open_population(run_app())
    assert not app.exception
    assert "Population" in all_text(app)


def test_the_overview_marks_come_from_the_gate_reports() -> None:
    """Every gate row is on screen and its mark matches its own report file."""
    app = run_app()
    assert not app.exception

    # dataframe 0 on the front page is the gate table
    frame = app.dataframe[0].value
    assert [row.split(" -- ")[0] for row in frame["gate"]] == list(GATE_ROWS)

    def mark_of(path: Path, block: str, flag: str) -> str:
        if not path.is_file():
            return "no report yet"
        value = (json.loads(path.read_text(encoding="utf-8")).get(block) or {}).get(flag)
        if isinstance(value, bool):
            return "PASS" if value else "FAIL"
        return value.strip().upper() if isinstance(value, str) and value.strip() else "no report yet"

    expected = [
        "PASS",  # Gate 0 has no report -- a setup checklist signed off in conversation
        mark_of(GATE1_REPORT, "verdict", "all_pass"),
        mark_of(RECOVERY_REPORT, "verdict", "all_bars_pass"),
        mark_of(GATE3_REPORT, "verdict", "all_decided_bars_pass"),
        mark_of(GATE4_REPORT, "verdict", "all_bars_pass"),
        mark_of(GATE5_REPORT, "frozen_bar", "verdict"),
    ]
    assert list(frame["verdict"]) == expected, "the front page disagrees with a gate report"

    # the failures are stated, not softened
    assert "nothing was re-graded after the fact" in all_text(app)


def test_the_overview_survives_missing_reports(tmp_path: Path) -> None:
    """A fresh clone shows six gates, five of them ungraded, and no traceback."""
    (tmp_path / "results").mkdir()

    app = run_app(root=tmp_path)
    assert not app.exception

    frame = app.dataframe[0].value
    assert list(frame["verdict"]) == ["PASS"] + ["no report yet"] * 5
    assert "no report yet" in all_text(app) or "Not found yet" in all_text(app)


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


def test_interviewer_page_renders_with_the_real_report() -> None:
    """Page 5 against results/stage5_strategies.json as it actually stands."""
    if not GATE5_REPORT.is_file():
        pytest.skip("no Gate 5 report in this checkout")

    report = json.loads(GATE5_REPORT.read_text(encoding="utf-8"))
    frozen = report.get("frozen_bar") or {}
    comparison = frozen.get("comparison") or {}

    app = open_interviewer(run_app())
    assert not app.exception

    text = all_text(app)
    for heading in (
        "Gate 5 verdict",
        "Error against interview length",
        "Questions to target, exploratory grid",
        "Proxy-gaming watch",
        "RL training",
    ):
        assert heading in text, f"the interviewer page is missing its {heading!r} panel"
    assert GATE5_REPORT.name in text

    # the banner states the report's own verdict on the frozen target
    assert any(
        f"Gate 5 frozen bar (a): **{frozen.get('verdict')}**" in str(e.value) for e in app.error
    )
    assert not app.success

    # one row per comparison the frozen bar asks for, with the report's verdict
    frame = app.dataframe[0].value
    assert list(frame["bar"]) == [key.replace("_", " ") for key in comparison]
    assert list(frame["verdict"]) == [entry.get("verdict") for entry in comparison.values()]
    assert list(frame["ratio"]) == [
        "--" if entry.get("ratio") is None else f"{float(entry['ratio']):.3f}"
        for entry in comparison.values()
    ]

    # the pre-declaration that makes an undefined bar reportable is on screen
    assert str(frozen.get("ruling_1")) in text


def test_the_censored_crossing_table_shows_no_strategy_reaching_the_target() -> None:
    """Every strategy's median is censored at the budget, and the page says so."""
    if not GATE5_REPORT.is_file():
        pytest.skip("no Gate 5 report in this checkout")

    crossing = (
        json.loads(GATE5_REPORT.read_text(encoding="utf-8")).get("frozen_bar") or {}
    ).get("population_curve_crossing") or {}
    if not crossing:
        pytest.skip("this Gate 5 report has no population-curve crossing block")

    app = open_interviewer(run_app())
    assert not app.exception

    # dataframe 0 is the verdict table, dataframe 1 the per-strategy crossing
    frame = app.dataframe[1].value
    assert len(frame) == len(crossing)
    for name, censored in zip(crossing, frame["censored"]):
        assert censored == ("yes" if crossing[name].get("median_is_censored") else "no")
    for name, reached in zip(crossing, frame["replicates reaching the target"]):
        assert reached == f"{crossing[name]['n_reached']} of {crossing[name]['n_replicates']}"


def test_the_exploratory_grid_carries_the_pre_declared_thresholds() -> None:
    """The grid table and the tie note, both read out of the report."""
    if not GATE5_REPORT.is_file():
        pytest.skip("no Gate 5 report in this checkout")

    exploratory = json.loads(GATE5_REPORT.read_text(encoding="utf-8")).get("exploratory") or {}
    grid = exploratory.get("threshold_grid") or {}
    if not grid:
        pytest.skip("this Gate 5 report has no exploratory grid")

    app = open_interviewer(run_app())
    assert not app.exception

    # dataframe 3 is the threshold grid, dataframe 4 the fraction of full matrix
    frame = app.dataframe[3].value
    for threshold, block in grid.items():
        per_strategy = block.get("per_strategy") or {}
        for name in ("random", "heuristic", "policy"):
            rows = frame[
                (frame["target (all dims)"] == threshold)
                & (frame["strategy"].str.startswith(name.replace("policy", "RL")))
            ]
            assert len(rows) == 1, f"{name} missing from the {threshold} row of the grid"
            median = per_strategy[name].get("median")
            assert rows["median questions"].iloc[0] == (
                "--" if median is None else f"{float(median):.0f}"
            )

    # the exploratory label is never dropped, and the tie is stated per threshold
    text = all_text(app)
    assert str(exploratory.get("label")) in text
    tie = [str(e.value) for e in app.warning if "frozen bar (b)" in str(e.value).lower()]
    assert tie, "the page does not state frozen bar (b)"
    for threshold in GATE5_THRESHOLDS:
        entry = (grid.get(threshold) or {}).get("policy_vs_heuristic") or {}
        if not entry:
            continue
        assert f"{threshold}: RL {float(entry['adaptive_median']):.0f}" in tie[0]
        assert f"heuristic {float(entry['random_median']):.0f}" in tie[0]

    # fraction of the full matrix at every pre-declared N
    fraction = (exploratory.get("fraction_of_full_matrix") or {}).get("per_strategy") or {}
    if fraction:
        frame = app.dataframe[4].value
        for column in ("N=5", "N=10", "N=15", "N=25"):
            assert column in frame.columns


def test_the_proxy_watch_states_the_finding() -> None:
    """Blur against truth: the gap, and that the trained policy is not the cause."""
    if not GATE5_REPORT.is_file():
        pytest.skip("no Gate 5 report in this checkout")

    watch = (
        json.loads(GATE5_REPORT.read_text(encoding="utf-8")).get("exploratory") or {}
    ).get("blur_vs_truth") or {}
    if not watch:
        pytest.skip("this Gate 5 report has no blur-against-truth watch")

    app = open_interviewer(run_app())
    assert not app.exception

    # dataframe 5 is the watch table, one row per strategy
    frame = app.dataframe[5].value
    assert len(frame) == len(watch)
    for name, shown in zip(watch, frame["widest gap"]):
        assert shown == f"{float(watch[name]['largest_gap']['gap']):.3f}"

    text = all_text(app)

    def gap_at_25(name: str) -> float:
        steps = (watch.get(name) or {}).get("by_n") or []
        return float(next(step for step in steps if step["n"] == 25)["gap"])

    # the finding: the never-trained heuristic's gap is the larger of the two
    assert gap_at_25("heuristic") > gap_at_25("policy")
    assert f"{gap_at_25('heuristic'):.3f}" in text
    assert f"{gap_at_25('policy'):.3f}" in text
    assert "not reward hacking" in text.lower()

    if ORACLE_WATCH_REPORT.is_file() and RL_WATCH_REPORT.is_file():
        oracle = json.loads(ORACLE_WATCH_REPORT.read_text(encoding="utf-8"))["checkpoints"][-1]
        blur = json.loads(RL_WATCH_REPORT.read_text(encoding="utf-8"))["checkpoints"][-1]
        assert f"{float(oracle['truth_rmse_final']):.3f}" in text
        assert f"{float(blur['truth_rmse_final']):.3f}" in text


def test_the_rl_panel_carries_the_training_numbers() -> None:
    """Reward, entropy and the static-order finding, from the training report."""
    if not (GATE5_REPORT.is_file() and RL_TRAINING_REPORT.is_file()):
        pytest.skip("both the Gate 5 and the RL training reports are needed here")

    training = json.loads(RL_TRAINING_REPORT.read_text(encoding="utf-8"))
    history = training.get("history") or []
    if not history:
        pytest.skip("this RL training report has no history")

    app = open_interviewer(run_app())
    assert not app.exception

    shown = [str(element.value) for element in app.metric]
    assert f"{float(history[-1]['episode_return']):.2f}" in shown
    assert f"{float(history[-1]['entropy']):.2f}" in shown
    assert "1.30 core-h" in shown

    text = all_text(app)
    assert str(training.get("reward")) in text
    assert f"discount {training['config']['discount']}" in text

    # the static-order finding, with the number it rests on
    declared = (training.get("checkpoints") or [{}])[-1].get("declared") or {}
    static = [str(e.value) for e in app.warning if "fixed order" in str(e.value)]
    assert static, "the page does not carry the static-order finding"
    assert f"{declared['n_distinct_items']} distinct items across all" in static[0]


def test_interviewer_page_survives_missing_files(tmp_path: Path) -> None:
    """A fresh clone gets the how-to-produce-it notes, not a traceback."""
    (tmp_path / "results").mkdir()

    app = open_interviewer(run_app(root=tmp_path))
    assert not app.exception

    text = all_text(app)
    assert "python -m src.eval.gate5" in text
    assert "python -m src.rl.train" in text
    for picture in GATE5_PICTURES:
        assert f"{picture} not found" in text


def test_interviewer_page_survives_missing_pictures(tmp_path: Path) -> None:
    """The Gate 5 report alone still renders; the panels say what is missing."""
    if not GATE5_REPORT.is_file():
        pytest.skip("no Gate 5 report in this checkout")
    results = tmp_path / "results"
    results.mkdir()
    (results / GATE5_REPORT.name).write_text(
        GATE5_REPORT.read_text(encoding="utf-8"), encoding="utf-8"
    )

    app = open_interviewer(run_app(root=tmp_path))
    assert not app.exception

    text = all_text(app)
    assert "Gate 5 verdict" in text
    assert "Questions to target, exploratory grid" in text
    assert "python -m src.rl.train" in text
    for picture in GATE5_PICTURES:
        assert f"{picture} not found" in text
