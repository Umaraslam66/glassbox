"""Unit tests for Gate 4 baseline 4 -- the LLM given the public profile only.

Three things are checked, in the order they can hurt:

1. **Category order.** ``probs`` must be in the category-index order the truth
   store and the grader use: 1..5 on Likert, **[no, yes]** on binary. The
   prompt's wire format lists a binary item's options the other way round
   ("yes:70 no:30"), so a parser that went by position instead of by name would
   silently invert 30% of the cells and still look perfectly healthy.
2. **What the prompt contains.** Only the whitelisted public-profile fields and
   the public item text. A prompt that picked up a persona card or a trait
   value would void the stage.
3. **Parsing.** The model writes free text under a 32-token budget, so the
   parser is the join between a 27B model's manners and a probability vector.
   Fixtures cover the clean case, percentages that miss 100 slightly,
   fractions, truncation, and the several ways a line is not an answer.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.model import llm_profile_baseline as lb

LIKERT_ITEM = {
    "item_id": "q001",
    "text": "I always follow the exact medication schedule my doctor writes down for me.",
    "type": "likert5",
    "options": [
        {"value": 1, "label": "strongly disagree"},
        {"value": 2, "label": "disagree"},
        {"value": 3, "label": "neither agree nor disagree"},
        {"value": 4, "label": "agree"},
        {"value": 5, "label": "strongly agree"},
    ],
    "topic_domain": "health",
}

BINARY_ITEM = {
    "item_id": "q017",
    "text": "I follow the detour signs set up by the road crew.",
    "type": "binary",
    "options": [{"value": 1, "label": "yes"}, {"value": 0, "label": "no"}],
    "topic_domain": "public-services",
}

PROFILE = {
    "age": 38,
    "occupation": "dairy farmer",
    "city_size": "small town",
    "household": "with partner",
    "region_type": "urban",
}


# ---------------------------------------------------------------------------
# category order
# ---------------------------------------------------------------------------


def test_binary_output_order_is_no_then_yes():
    assert lb.categories(BINARY_ITEM) == ["no", "yes"]
    assert lb.answer_values(BINARY_ITEM) == [0, 1]


def test_binary_wire_order_is_yes_then_no():
    """The prompt asks in the bank's order; only the stored vector is recoded."""
    assert lb.prompt_categories(BINARY_ITEM) == ["yes", "no"]
    assert lb.format_line(BINARY_ITEM) == "yes:<pct> no:<pct>"


def test_likert_order_is_one_to_five_both_ways():
    assert lb.categories(LIKERT_ITEM) == ["1", "2", "3", "4", "5"]
    assert lb.answer_values(LIKERT_ITEM) == [1, 2, 3, 4, 5]
    assert lb.format_line(LIKERT_ITEM) == "1:<pct> 2:<pct> 3:<pct> 4:<pct> 5:<pct>"


def test_a_yes_leaning_reply_lands_in_the_yes_slot():
    """The one that would silently invert the arm if it broke."""
    probs = lb.parse_distribution("yes:80 no:20", lb.categories(BINARY_ITEM))
    assert probs == pytest.approx([0.2, 0.8])


# ---------------------------------------------------------------------------
# what the prompt contains
# ---------------------------------------------------------------------------


def test_prompt_carries_the_profile_and_the_item_and_nothing_else():
    system, user = lb.build_prompt(PROFILE, LIKERT_ITEM)
    assert LIKERT_ITEM["text"] in user
    for value in PROFILE.values():
        assert str(value) in user
    for label in ("strongly disagree", "strongly agree"):
        assert label in user
    assert "interview" not in (system + user).lower()
    assert "trait" not in (system + user).lower()


def test_profile_fields_outside_the_whitelist_never_reach_the_prompt(tmp_path: Path):
    (tmp_path / "p9999.json").write_text(
        json.dumps({"pid": "p9999", "age": 40, "secret_theta": "TRU=+1.8"}),
        encoding="utf-8",
    )
    profile = lb.load_profile("p9999", tmp_path)
    assert profile == {"age": 40}
    _, user = lb.build_prompt(profile, BINARY_ITEM)
    assert "secret_theta" not in user and "TRU" not in user


def test_every_cell_gets_every_repeat_and_a_seed_of_its_own(tmp_path: Path):
    """The driver seeds from (pid, item_id, round), so those must be unique."""
    for pid in ("p0001", "p0002"):
        (tmp_path / f"{pid}.json").write_text(
            json.dumps({"pid": pid, **PROFILE}), encoding="utf-8"
        )
    bank = {"q001": LIKERT_ITEM, "q017": BINARY_ITEM}
    records = list(
        lb.iter_records(
            ["p0001", "p0002"], ["q001", "q017"], bank, profiles_dir=tmp_path, repeats=3
        )
    )
    assert len(records) == 2 * 2 * 3
    identities = {(r["pid"], r["item_id"], r["round"]) for r in records}
    assert len(identities) == len(records)
    assert {r["round"] for r in records} == {
        f"{lb.ROUND_TAG}|r{rep}" for rep in range(3)
    }


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

LIKERT_NAMES = ["1", "2", "3", "4", "5"]
BINARY_NAMES = ["no", "yes"]


@pytest.mark.parametrize(
    "text, names, expected",
    [
        ("1:15 2:20 3:30 4:20 5:15", LIKERT_NAMES, [0.15, 0.20, 0.30, 0.20, 0.15]),
        ("1: 15  2: 20 3: 30 4: 20 5: 15\n", LIKERT_NAMES, [0.15, 0.20, 0.30, 0.20, 0.15]),
        ("1:15% 2:20% 3:30% 4:20% 5:15%", LIKERT_NAMES, [0.15, 0.20, 0.30, 0.20, 0.15]),
        # truncated by the token budget mid-decimal, still a whole line
        ("1:15.0 2:20.0 3:30.0 4:20.0 5:15.", LIKERT_NAMES, [0.15, 0.20, 0.30, 0.20, 0.15]),
        # written as fractions rather than percentages
        ("1:0.15 2:0.2 3:0.3 4:0.2 5:0.15", LIKERT_NAMES, [0.15, 0.20, 0.30, 0.20, 0.15]),
        ("YES:70 NO:30", BINARY_NAMES, [0.30, 0.70]),
        ("yes:0 no:100", BINARY_NAMES, [1.0, 0.0]),
    ],
)
def test_parse_distribution_accepts(text, names, expected):
    assert lb.parse_distribution(text, names) == pytest.approx(expected)


@pytest.mark.parametrize(
    "text, names",
    [
        ("", LIKERT_NAMES),
        ("I cannot answer that.", LIKERT_NAMES),
        ("1:15 2:20 3:30 4:20", LIKERT_NAMES),  # a category missing
        ("1:15 2:20 3:30 4:20 5:15 6:5", LIKERT_NAMES),  # a category too many
        ("1:20 1:20 3:20 4:20 5:20", LIKERT_NAMES),  # one named twice
        ("1:0 2:0 3:0 4:0 5:0", LIKERT_NAMES),  # no mass anywhere
        ("1:10 2:10 3:10 4:10 5:10", LIKERT_NAMES),  # sums to 50, not an answer
        ("yes:700 no:300", BINARY_NAMES),  # sums to 1000
    ],
)
def test_parse_distribution_rejects(text, names):
    assert lb.parse_distribution(text, names) is None


def test_averaged_repeats_stay_a_distribution():
    vectors = [[0.1, 0.2, 0.3, 0.2, 0.2], [0.2, 0.2, 0.2, 0.2, 0.2]]
    mean = lb.average(vectors)
    assert sum(mean) == pytest.approx(1.0, abs=1e-12)
    assert mean[0] == pytest.approx(0.15)
