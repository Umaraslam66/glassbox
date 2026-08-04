"""Unit test for the truth-store accessor.

Grader-side code: this is one of the two files exempt from the Wall test, which
is why it may import the accessor at all. See ``tests/test_wall.py``.
"""

from __future__ import annotations

import pytest

from src.eval import truth_store


@pytest.fixture()
def empty_truth_dir(tmp_path, monkeypatch):
    """Point the accessor at an empty directory, so nothing is on file."""
    monkeypatch.setenv("GLASSBOX_TRUTH_DIR", str(tmp_path))
    return tmp_path


@pytest.mark.parametrize(
    "loader", [truth_store.load_theta, truth_store.load_card, truth_store.load_noise]
)
def test_loaders_raise_file_not_found_when_absent(loader, empty_truth_dir) -> None:
    with pytest.raises(FileNotFoundError) as excinfo:
        loader("persona-0001")
    assert "persona-0001" in str(excinfo.value)


def test_list_persona_ids_is_empty_when_absent(empty_truth_dir) -> None:
    assert truth_store.list_persona_ids() == []


def test_round_trip(tmp_path, monkeypatch) -> None:
    """A file that exists is read back as written."""
    monkeypatch.setenv("GLASSBOX_TRUTH_DIR", str(tmp_path))
    theta_dir = tmp_path / "theta"
    theta_dir.mkdir()
    (theta_dir / "persona-0001.json").write_text(
        '{"institution_trust": 1.4, "risk_appetite": -0.9}', encoding="utf-8"
    )

    assert truth_store.load_theta("persona-0001")["institution_trust"] == 1.4
    assert truth_store.list_persona_ids() == ["persona-0001"]
