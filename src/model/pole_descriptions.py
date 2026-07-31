"""The eight trait poles, copied verbatim from PREREGISTRATION.md section 2.

Why this file exists at all
---------------------------
The item encoder and the person encoder describe each latent dimension to the
system-side model in words. The words have to come from somewhere, and the only
Wall-clean source is the study's own **published design**: the table in
PREREGISTRATION.md section 2, committed to the repository, written before any
persona existed, readable by anyone who reads the paper.

What is planted truth is *which item loads on which dimension and how much*, and
*what each persona's trait vector is*. Neither is in that table. A stranger with
the public repository could build the same prompts.

The rules that keep it that way
-------------------------------
1. Every string below is **byte-identical** to the corresponding cell of the
   committed table. ``tests/test_pole_descriptions.py`` re-parses
   PREREGISTRATION.md on every run and fails on any difference, including
   whitespace and punctuation.
2. The descriptions are **never reworded in response to encoder errors**. The
   moment they are tuned against observed item behaviour they stop being
   published design and become a truth-informed artifact. If the table ever
   changes in PREREGISTRATION.md, that is a pre-registration deviation and the
   owner decides, not this file.

Nothing here reads ``results/``, the fitted parameters, the item bank's design
labels or any persona quantity. It is a string table and an ordering.
"""

from __future__ import annotations

#: The eight dimension codes, in the order the table lists them. This order is
#: the column order of every loading vector in the study.
DIMENSION_CODES: tuple[str, ...] = (
    "TRU",
    "RSK",
    "ENV",
    "PRC",
    "TRD",
    "SOC",
    "TEC",
    "LOC",
)

#: code -> the "Dimension" column.
DIMENSION_NAMES: dict[str, str] = {
    "TRU": "Institution trust",
    "RSK": "Risk appetite",
    "ENV": "Environmentalism",
    "PRC": "Price sensitivity",
    "TRD": "Traditionalism",
    "SOC": "Sociability",
    "TEC": "Tech optimism",
    "LOC": "Locality attachment",
}

#: code -> the "-2 looks like" column (the low pole).
LOW_POLE: dict[str, str] = {
    "TRU": "assumes officials lie; avoids banks, doctors, forms",
    "RSK": "savings account only; insures everything",
    "ENV": "annoyed by green rules; won't pay extra",
    "PRC": "buys quality without checking price",
    "TRD": 'breaks conventions; distrusts "how it\'s done"',
    "SOC": "keeps to self; small circle",
    "TEC": "new tech is a scam or a threat",
    "LOC": "would move anywhere; no hometown pull",
}

#: code -> the "+2 looks like" column (the high pole).
HIGH_POLE: dict[str, str] = {
    "TRU": "defends public services; follows official advice",
    "RSK": "invests, gambles, quits jobs, moves abroad",
    "ENV": "organizes recycling; pays green premiums",
    "PRC": "compares unit prices; waits for discounts",
    "TRD": "keeps customs, family roles, established ways",
    "SOC": "joins clubs; knows the neighbors; hosts",
    "TEC": "early adopter; tech solves problems",
    "LOC": "rooted; local news, local pride, won't move",
}

#: The section of PREREGISTRATION.md this table is copied from, for the guard test.
SOURCE_SECTION = "## 2. Planted truth: traits, population, noise"


def pole_rows() -> list[tuple[int, str, str, str, str]]:
    """The table as ``(number, code, name, low_pole, high_pole)`` rows, in order."""
    return [
        (i + 1, code, DIMENSION_NAMES[code], LOW_POLE[code], HIGH_POLE[code])
        for i, code in enumerate(DIMENSION_CODES)
    ]


def describe(code: str) -> str:
    """The three-line block a prompt shows for one dimension.

    Deliberately dull: the dimension's published name, then its two published
    poles, with no extra adjectives of our own.
    """
    return (
        f"TRAIT DIMENSION: {DIMENSION_NAMES[code]}\n"
        f"LOW end of this dimension looks like: {LOW_POLE[code]}\n"
        f"HIGH end of this dimension looks like: {HIGH_POLE[code]}"
    )
