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
RECOVERY_REPORT = REPO_ROOT / "results" / "stage2_recovery.json"

RECOVERY_PAGE = "2. Recovery"

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
    """Page 2 against results/stage2_recovery.json as it actually stands."""
    if not RECOVERY_REPORT.is_file():
        pytest.skip("no Stage 2 recovery report in this checkout")

    verdict = json.loads(RECOVERY_REPORT.read_text(encoding="utf-8")).get("verdict") or {}
    bars = verdict.get("bars") or {}

    app = open_recovery(run_app())
    assert not app.exception

    text = all_text(app)
    for heading in (
        "Gate 2 verdict",
        "Trait recovery per dimension",
        "How many dimensions the answers support",
        "Item recovery and blur",
        "Watch list",
    ):
        assert heading in text, f"the recovery page is missing its {heading!r} panel"

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


def test_recovery_page_survives_missing_files(tmp_path: Path) -> None:
    """A fresh clone gets the how-to-produce-it note, not a traceback."""
    (tmp_path / "results").mkdir()

    app = open_recovery(run_app(root=tmp_path))
    assert not app.exception

    text = all_text(app)
    assert "python -m src.eval.recovery" in text
    assert "stage2_scatter.png not found" in text
    assert "stage2_heatmap.png not found" in text
