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

#: The wording the monotonicity row must carry, exactly.
MONOTONICITY_RULING = "orchestrator ruling on qualitative wording; both readings reported"

RECOVERY_PAGE = "2. Recovery"
PERSON_ENCODER_PAGE = "3. Person encoder"

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
