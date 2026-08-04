"""The pole descriptions must stay byte-identical to the frozen table.

``src/model/pole_descriptions.py`` copies the eight rows of PREREGISTRATION.md
section 2 into constants that go straight into system-side prompts. That is
legal only because those rows are *published design*: committed, written before
any persona existed, readable by anyone with the repository.

The moment a description is reworded -- to "improve" it, to make a stubborn item
classify correctly, to smooth an awkward phrase -- it stops being published
design and becomes an artifact tuned against observed behaviour. That is the
exact mechanism by which planted design turns into planted truth, and it is
listed as risk 8 in the Stage 3 design.

So this test re-reads the committed file on every run and compares character for
character. It fails on a changed word, a changed semicolon, a changed space. If
PREREGISTRATION.md itself ever changes here, that is a pre-registration
deviation and the owner decides -- not this file, and not the encoder.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.model.pole_descriptions import (
    DIMENSION_CODES,
    DIMENSION_NAMES,
    HIGH_POLE,
    LOW_POLE,
    SOURCE_SECTION,
    describe,
    pole_rows,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PREREG = REPO_ROOT / "PREREGISTRATION.md"


def parse_table() -> list[list[str]]:
    """The section 2 trait table, as lists of stripped cells, in file order.

    Deliberately dumb: find the section heading, walk forward, keep every line
    that looks like a table row whose first cell is a number. No regex over the
    contents, so a reworded description cannot slip through a clever parser.
    """
    text = PREREG.read_text(encoding="utf-8")
    start = text.index(SOURCE_SECTION)
    rows: list[list[str]] = []
    for line in text[start:].splitlines():
        if line.startswith("```"):
            break
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) == 5 and re.fullmatch(r"\d+", cells[0]):
            rows.append(cells)
    return rows


def test_the_source_section_is_where_we_think_it_is() -> None:
    """Guard against the parser silently finding nothing."""
    assert PREREG.is_file(), f"expected the frozen contract at {PREREG}"
    assert SOURCE_SECTION in PREREG.read_text(encoding="utf-8")


def test_the_table_still_has_eight_rows() -> None:
    rows = parse_table()
    assert len(rows) == 8, f"expected 8 trait rows in section 2, parsed {len(rows)}"


def test_every_cell_is_byte_identical() -> None:
    """Row by row, cell by cell, against the committed file."""
    parsed = parse_table()
    ours = pole_rows()
    assert len(parsed) == len(ours)

    problems: list[str] = []
    for (number, code, name, low, high), row in zip(ours, parsed):
        expected = [str(number), code, name, low, high]
        for field, mine, theirs in zip(
            ("number", "code", "name", "low pole", "high pole"), expected, row
        ):
            if mine != theirs:
                problems.append(
                    f"{code} {field}: constants have {mine!r}, "
                    f"PREREGISTRATION.md has {theirs!r}"
                )

    assert not problems, (
        "pole descriptions have drifted from the frozen table:\n"
        + "\n".join(f"  - {p}" for p in problems)
    )


def test_the_order_is_the_tables_order() -> None:
    """Column order of every loading vector in the study depends on this."""
    assert [row[1] for row in parse_table()] == list(DIMENSION_CODES)


def test_no_dimension_is_missing_from_any_map() -> None:
    for mapping, label in ((DIMENSION_NAMES, "names"), (LOW_POLE, "low"), (HIGH_POLE, "high")):
        assert set(mapping) == set(DIMENSION_CODES), f"{label} map does not cover the codes"


@pytest.mark.parametrize("code", DIMENSION_CODES)
def test_describe_quotes_the_published_text_verbatim(code: str) -> None:
    """What actually reaches a prompt is the published string, unaltered."""
    block = describe(code)
    assert DIMENSION_NAMES[code] in block
    assert LOW_POLE[code] in block
    assert HIGH_POLE[code] in block
    # and nothing of our own invention beyond the three fixed labels
    residue = block
    for piece in (DIMENSION_NAMES[code], LOW_POLE[code], HIGH_POLE[code]):
        residue = residue.replace(piece, "", 1)
    assert residue == (
        "TRAIT DIMENSION: \nLOW end of this dimension looks like: \n"
        "HIGH end of this dimension looks like: "
    )
