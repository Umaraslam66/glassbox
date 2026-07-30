"""Tests for the population summary the dashboard reads.

The summary is the one file that crosses from the grader side to the dashboard
side, so two things have to hold and both are checked here:

  * the shape is exactly what the dashboard opens -- fixed histogram grid,
    fixed keys, one entry per dimension;
  * the contents are aggregate. The strongest form of that check is the last
    test in this file: no persona id and no individual planted value may appear
    anywhere in the written text.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.bank.schema import DIMENSIONS
from src.eval import export_population_summary as export

TOP_LEVEL_KEYS = {
    "schema",
    "generated_at",
    "n_personas",
    "n_with_wobble",
    "dimensions",
    "dimension_labels",
    "theta",
    "wobble",
    "demographics",
}

CITY_SIZES = ["village", "small town", "mid-size city", "big city"]
HOUSEHOLDS = ["alone", "with partner", "shared flat"]
REGIONS = ["rural", "suburban", "urban"]


def pid_of(index: int) -> str:
    return f"p{index:04d}"


def write_batch(tmp_path: Path, n_personas: int = 60, n_occupations: int = 30) -> tuple[Path, Path]:
    """A synthetic batch on disk: planted values plus public profiles.

    Trait values walk across the whole [-2, 2] range and occupations are more
    numerous than the top-20 cut, so both the histogram grid and the truncation
    are exercised by the same fixture.
    """
    planted = tmp_path / "planted"
    (planted / "theta").mkdir(parents=True)
    (planted / "noise").mkdir(parents=True)
    public = tmp_path / "public"
    (public / "profiles").mkdir(parents=True)

    for index in range(n_personas):
        pid = pid_of(index + 1)
        position = index / max(n_personas - 1, 1)  # 0 .. 1
        theta = {
            dimension: round(-2.0 + 4.0 * ((position + offset / len(DIMENSIONS)) % 1.0), 4)
            for offset, dimension in enumerate(DIMENSIONS)
        }
        (planted / "theta" / f"{pid}.json").write_text(
            json.dumps({"pid": pid, "theta": theta, "batch_seed": 548}), encoding="utf-8"
        )
        (planted / "noise" / f"{pid}.json").write_text(
            json.dumps({"pid": pid, "wobble": round(0.1 + 0.35 * position, 2)}), encoding="utf-8"
        )
        (public / "profiles" / f"{pid}.json").write_text(
            json.dumps(
                {
                    "pid": pid,
                    "age": 20 + index % 60,
                    "occupation": f"job {index % n_occupations:02d}",
                    "city_size": CITY_SIZES[index % len(CITY_SIZES)],
                    "household": HOUSEHOLDS[index % len(HOUSEHOLDS)],
                    "region_type": REGIONS[index % len(REGIONS)],
                }
            ),
            encoding="utf-8",
        )

    return planted, public


@pytest.fixture()
def summary(tmp_path: Path) -> dict:
    planted, public = write_batch(tmp_path)
    out = tmp_path / "results" / "population_summary.json"
    assert export.main(["--truth", str(planted), "--public", str(public), "--out", str(out)]) == 0
    return json.loads(out.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# shape
# --------------------------------------------------------------------------


def test_the_summary_has_exactly_the_documented_top_level_keys(summary: dict) -> None:
    assert set(summary) == TOP_LEVEL_KEYS
    assert summary["schema"] == export.SCHEMA
    assert summary["n_personas"] == 60
    assert summary["n_with_wobble"] == 60
    assert summary["dimensions"] == list(DIMENSIONS)
    assert set(summary["dimension_labels"]) == set(DIMENSIONS)


def test_every_dimension_gets_the_same_fixed_histogram_grid(summary: dict) -> None:
    block = summary["theta"]
    assert block["bins"] == 20
    assert block["range"] == [-2.0, 2.0]
    assert set(block["histograms"]) == set(DIMENSIONS)

    for dimension in DIMENSIONS:
        histogram = block["histograms"][dimension]
        assert set(histogram) == {"bin_edges", "counts", "n"}
        assert len(histogram["bin_edges"]) == 21
        assert len(histogram["counts"]) == 20
        assert histogram["bin_edges"][0] == -2.0
        assert histogram["bin_edges"][-1] == 2.0
        assert histogram["n"] == 60
        # Every persona lands in exactly one bin: nobody is lost, nobody doubled.
        assert sum(histogram["counts"]) == 60
        assert all(isinstance(count, int) for count in histogram["counts"])


def test_the_wobble_histogram_covers_every_persona(summary: dict) -> None:
    histogram = summary["wobble"]["histogram"]
    assert summary["wobble"]["bins"] == 20
    assert len(histogram["counts"]) == 20
    assert len(histogram["bin_edges"]) == 21
    assert sum(histogram["counts"]) == 60
    assert histogram["bin_edges"][0] < histogram["bin_edges"][-1]


def test_demographics_are_counted_per_field_most_common_first(summary: dict) -> None:
    block = summary["demographics"]
    assert block["n_profiles"] == 60
    assert set(block["fields"]) == {"occupation", "city_size", "household", "region_type"}

    city = block["fields"]["city_size"]
    assert city["n_distinct"] == 4
    assert city["truncated"] is False
    assert sum(row["count"] for row in city["counts"]) == 60
    counts = [row["count"] for row in city["counts"]]
    assert counts == sorted(counts, reverse=True)
    assert {row["value"] for row in city["counts"]} == set(CITY_SIZES)


def test_occupation_is_cut_at_the_top_twenty(summary: dict) -> None:
    occupation = summary["demographics"]["fields"]["occupation"]
    assert occupation["n_distinct"] == 30
    assert occupation["n_shown"] == 20
    assert occupation["truncated"] is True
    assert len(occupation["counts"]) == 20


# --------------------------------------------------------------------------
# aggregate-only
# --------------------------------------------------------------------------


#: Marked values for one persona: inside [-2, 2] so they still land in a bin,
#: but at a precision no bin edge can produce, so finding one in the output
#: could only mean an individual value was copied through.
MARKED_THETA = {
    dimension: value
    for dimension, value in zip(
        DIMENSIONS, [-1.7391, 0.6173, 1.4142, -0.3178, 1.9319, -1.1235, 0.8121, -0.5772]
    )
}
MARKED_WOBBLE = 0.2718


def test_no_persona_id_or_individual_value_reaches_the_file(tmp_path: Path) -> None:
    """The check that matters: the dashboard side must not be able to read one
    person's planted values out of this file."""
    planted, public = write_batch(tmp_path, n_personas=25)
    (planted / "theta" / "p0007.json").write_text(
        json.dumps({"pid": "p0007", "theta": MARKED_THETA, "batch_seed": 548}), encoding="utf-8"
    )
    (planted / "noise" / "p0007.json").write_text(
        json.dumps({"pid": "p0007", "wobble": MARKED_WOBBLE}), encoding="utf-8"
    )

    out = tmp_path / "summary.json"
    export.main(["--truth", str(planted), "--public", str(public), "--out", str(out)])
    text = out.read_text(encoding="utf-8")

    for index in range(25):
        assert pid_of(index + 1) not in text
    for value in MARKED_THETA.values():
        assert str(value) not in text
    assert str(MARKED_WOBBLE) not in text
    # It is there in the batch, and it is counted -- it is only never itemised.
    assert sum(summary_counts(text, "TRU")) == 25


def summary_counts(text: str, dimension: str) -> list[int]:
    return json.loads(text)["theta"]["histograms"][dimension]["counts"]


def test_a_missing_public_directory_is_not_a_crash(tmp_path: Path) -> None:
    planted, _ = write_batch(tmp_path, n_personas=10)
    out = tmp_path / "summary.json"

    assert export.main(["--truth", str(planted), "--public", str(tmp_path / "nowhere"), "--out", str(out)]) == 0
    summary = json.loads(out.read_text(encoding="utf-8"))

    assert summary["n_personas"] == 10
    assert summary["demographics"]["n_profiles"] == 0
    for field in summary["demographics"]["fields"].values():
        assert field["counts"] == []
        assert field["n_distinct"] == 0


def test_an_empty_batch_is_an_error_not_an_empty_file(tmp_path: Path, capsys) -> None:
    (tmp_path / "planted" / "theta").mkdir(parents=True)
    out = tmp_path / "summary.json"

    with pytest.raises(SystemExit):
        export.main(
            ["--truth", str(tmp_path / "planted"), "--public", str(tmp_path), "--out", str(out)]
        )
    assert not out.exists()


def test_the_cli_leaves_the_environment_as_it_found_it(tmp_path: Path) -> None:
    """The batch directory is a command-line argument, not a lasting setting."""
    before = dict(os.environ)
    planted, public = write_batch(tmp_path, n_personas=10)
    export.main(
        ["--truth", str(planted), "--public", str(public), "--out", str(tmp_path / "summary.json")]
    )
    assert dict(os.environ) == before


def test_an_existing_environment_override_is_put_back(tmp_path: Path, monkeypatch) -> None:
    """The same pattern as the store's own test: the override is honoured while
    the export runs and restored to whatever it was afterwards."""
    monkeypatch.setenv("GLASSBOX_TRUTH_DIR", "/somewhere/else")
    planted, public = write_batch(tmp_path, n_personas=10)

    export.main(
        ["--truth", str(planted), "--public", str(public), "--out", str(tmp_path / "summary.json")]
    )
    assert os.environ["GLASSBOX_TRUTH_DIR"] == "/somewhere/else"


def test_personas_without_planted_wobble_still_get_counted(tmp_path: Path) -> None:
    planted, public = write_batch(tmp_path, n_personas=10)
    (planted / "noise" / "p0003.json").unlink()
    out = tmp_path / "summary.json"

    export.main(["--truth", str(planted), "--public", str(public), "--out", str(out)])
    summary = json.loads(out.read_text(encoding="utf-8"))

    assert summary["n_personas"] == 10
    assert summary["n_with_wobble"] == 9
    assert sum(summary["wobble"]["histogram"]["counts"]) == 9
    assert sum(summary["theta"]["histograms"]["TRU"]["counts"]) == 10


def test_one_wobble_value_does_not_collapse_the_histogram(tmp_path: Path) -> None:
    """A batch where everybody wobbles the same amount still plots."""
    planted, public = write_batch(tmp_path, n_personas=10)
    for path in (planted / "noise").glob("*.json"):
        path.write_text(json.dumps({"pid": path.stem, "wobble": 0.2}), encoding="utf-8")
    out = tmp_path / "summary.json"

    export.main(["--truth", str(planted), "--public", str(public), "--out", str(out)])
    histogram = json.loads(out.read_text(encoding="utf-8"))["wobble"]["histogram"]

    assert sum(histogram["counts"]) == 10
    assert histogram["bin_edges"][0] < 0.2 < histogram["bin_edges"][-1]
