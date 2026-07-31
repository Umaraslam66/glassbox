"""Tests for the question bank (module M1).

No network anywhere. Fake clients read the topic list out of each drafting
prompt and hand back canned JSON, so a full 252-item bank can be built and
validated in a fraction of a second.
"""

from __future__ import annotations

import json
import re
from collections import Counter

import pytest

from src.bank import generate, validate
from src.bank.schema import (
    AMENDMENT_KEYS,
    DIMENSIONS,
    OPEN_PRIVATE_FILENAME,
    PRIVATE_FILENAME,
    PRIVATE_ONLY_KEYS,
    PUBLIC_FILENAME,
    TOPIC_DOMAINS,
    BankSpec,
    normalize_text,
    read_bank,
    read_json,
    write_bank,
)
from src.llm_client import LLMClient, LLMResponse

TOPIC_LINE = re.compile(r"^\s*\d+\.\s*([a-z-]+)\s*$", re.MULTILINE)

#: A sentence one fake client offers in every single call, so the deduplicator
#: has to notice it across design cells and not just inside one call.
TWIN = "I like round numbers on my receipts."


class FakeClient(LLMClient):
    """Answers a drafting prompt with one unique line per requested topic."""

    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []
        self.counter = 0

    def rows_for(self, user: str) -> list[dict[str, str]]:
        rows = []
        for domain in TOPIC_LINE.findall(user):
            self.counter += 1
            rows.append(
                {
                    "topic_domain": domain,
                    "text": f"I keep an eye on the {domain} side of things, case {self.counter}.",
                }
            )
        return rows

    def generate(self, system, user, temperature=0.7, max_tokens=512) -> LLMResponse:
        self.calls.append({"system": system or "", "user": user})
        body = json.dumps(self.rows_for(user))
        return LLMResponse(
            text=body, input_tokens=len(user) // 4, output_tokens=len(body) // 4, model="fake"
        )


class TwinClient(FakeClient):
    """Offers the same sentence first in every call. It may only be used once."""

    def rows_for(self, user: str) -> list[dict[str, str]]:
        rows = super().rows_for(user)
        return [{"topic_domain": rows[0]["topic_domain"], "text": TWIN}] + rows


class RepeatingClient(FakeClient):
    """First reply for any chunk is one sentence repeated; a retry gets real ones."""

    def __init__(self) -> None:
        super().__init__()
        self.first_replies = 0

    def rows_for(self, user: str) -> list[dict[str, str]]:
        if "RETRY" in user:
            return super().rows_for(user)
        self.first_replies += 1
        return [
            {"topic_domain": domain, "text": "I answer these things the very same way every time."}
            for domain in TOPIC_LINE.findall(user)
        ]


class FencedClient(FakeClient):
    """Wraps its JSON in a markdown fence and never labels the topic."""

    def generate(self, system, user, temperature=0.7, max_tokens=512) -> LLMResponse:
        self.calls.append({"system": system or "", "user": user})
        rows = [{"text": row["text"]} for row in self.rows_for(user)]
        body = "```json\n" + json.dumps(rows) + "\n```"
        return LLMResponse(text=body, input_tokens=1, output_tokens=1, model="fake")


@pytest.fixture()
def reduced() -> BankSpec:
    return BankSpec.reduced()


# --------------------------------------------------------------------------
# The design itself, with no model involved
# --------------------------------------------------------------------------


def test_full_design_has_the_frozen_cell_counts() -> None:
    spec = BankSpec()
    closed, open_slots = generate.build_slots(spec, seed=7)

    assert len(closed) == 252
    assert len(open_slots) == 20

    kinds = Counter(slot.cell.kind for slot in closed)
    assert kinds == {"primary": 192, "cross": 30, "weak": 15, "distractor": 15}

    for dim in DIMENSIONS:
        primaries = [s for s in closed if s.cell.kind == "primary" and s.cell.dims == (dim,)]
        assert len(primaries) == 24
        assert Counter(s.cell.item_type for s in primaries) == {"likert5": 16, "binary": 8}
        negative = Counter(s.cell.item_type for s in primaries if s.cell.negatively_keyed)
        assert negative == {"likert5": 5, "binary": 3}
        for slot in primaries:
            assert abs(slot.cell.loading_dict()[dim]) == spec.strong_loading


def test_cross_items_cover_every_dimension_pair_including_the_frozen_zeros() -> None:
    closed, _ = generate.build_slots(BankSpec(), seed=1)
    cross = [slot for slot in closed if slot.cell.kind == "cross"]

    pairs = {tuple(sorted(slot.cell.dims)) for slot in cross}
    assert len(pairs) == 28
    assert ("RSK", "TRU") in pairs
    assert ("ENV", "RSK") in pairs

    signs = set()
    for slot in cross:
        loadings = slot.cell.loading_dict()
        assert sorted(abs(v) for v in loadings.values()) == [0.5, 0.7]
        main = max(loadings, key=lambda d: abs(loadings[d]))
        assert (loadings[main] < 0) == slot.cell.negatively_keyed
        signs.add(tuple(v > 0 for v in loadings.values()))
    assert len(signs) == 4, "all four primary/secondary sign combinations should appear"


def test_weak_items_are_low_magnitude_and_distractors_are_flat() -> None:
    closed, _ = generate.build_slots(BankSpec(), seed=3)

    weak = [slot for slot in closed if slot.cell.kind == "weak"]
    assert len(weak) == 15
    for slot in weak:
        loadings = slot.cell.loading_dict()
        assert len(loadings) == 1
        assert abs(next(iter(loadings.values()))) == 0.25
        assert slot.cell.strength_class == "weak"

    distractors = [slot for slot in closed if slot.cell.kind == "distractor"]
    assert len(distractors) == 15
    for slot in distractors:
        assert slot.cell.loading_dict() == {}
        assert slot.cell.strength_class == "none"


def test_every_dimension_and_every_topic_domain_is_used() -> None:
    closed, open_slots = generate.build_slots(BankSpec(), seed=11)

    touching: Counter[str] = Counter()
    for slot in closed:
        for dim in slot.cell.dims:
            touching[dim] += 1
    assert all(touching[dim] >= 24 for dim in DIMENSIONS)

    assert {slot.topic_domain for slot in closed} == set(TOPIC_DOMAINS)

    open_dims = Counter(d for slot in open_slots for d in slot.cell.dims)
    assert all(open_dims[dim] >= 1 for dim in DIMENSIONS)
    assert all(1 <= len(slot.cell.dims) <= 2 for slot in open_slots)


def test_slot_assignment_is_seed_deterministic() -> None:
    first, first_open = generate.build_slots(BankSpec(), seed=42)
    again, again_open = generate.build_slots(BankSpec(), seed=42)
    other, _ = generate.build_slots(BankSpec(), seed=43)

    assert first == again
    assert first_open == again_open
    assert [s.topic_domain for s in first] != [s.topic_domain for s in other]
    # Only the topic assignment moves with the seed; the design cells are fixed.
    assert [s.cell for s in first] == [s.cell for s in other]


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------


def test_prompts_demand_the_house_style() -> None:
    prompts = generate.dry_run_prompts(BankSpec(), seed=5)
    by_kind: dict[tuple[str, str], dict] = {}
    for prompt in prompts:
        by_kind.setdefault((prompt["kind"], prompt["strength_class"]), prompt)

    strong = by_kind[("primary", "strong")]["user"]
    assert "I generally trust what city officials tell us" in strong
    assert "One clause only" in strong
    assert "no trait words" in strong.lower()
    assert "First person" in strong
    assert "JSON array" in strong

    weak = by_kind[("weak", "weak")]["user"]
    assert "DELIBERATELY WEAK" in weak
    assert "double-barreled" in weak
    assert "vague quantifier" in weak

    open_prompt = by_kind[("open", "none")]["user"]
    assert "Tell me about a time a public service let you down" in open_prompt
    assert "Never answerable with yes or no" in open_prompt

    distractor = by_kind[("distractor", "none")]["user"]
    assert "DISTRACTORS" in distractor
    assert "no information about any of" in distractor

    negative = next(p for p in prompts if p["kind"] == "primary" and p["negatively_keyed"])
    assert "AGREES with the statement is LOW" in negative["user"]

    cross = by_kind[("cross", "strong")]["user"]
    assert "SECOND TRAIT" in cross


def test_dry_run_emits_prompts_without_touching_the_client(tmp_path) -> None:
    client = FakeClient()
    target = tmp_path / "prompts" / "drafting.json"

    exit_code = generate.main(
        ["--dry-run", "--prompts-out", str(target), "--seed", "5"], client=client
    )

    assert exit_code == 0
    assert client.calls == []
    prompts = json.loads(target.read_text(encoding="utf-8"))
    assert len(prompts) > 50
    assert all(prompt["system"] and prompt["user"] for prompt in prompts)
    # One prompt per chunk, and the chunks cover every slot exactly once.
    assert sum(prompt["n_slots"] for prompt in prompts) == 252 + 20


def test_parse_candidates_survives_fences_bare_strings_and_bullets() -> None:
    fenced = '```json\n[{"topic_domain": "money", "text": "I check the price first."}]\n```'
    assert generate.parse_candidates(fenced) == [
        {"topic_domain": "money", "text": "I check the price first."}
    ]

    wrapped = '{"items": ["I check the price first.", "I never look at receipts."]}'
    assert [c["text"] for c in generate.parse_candidates(wrapped)] == [
        "I check the price first.",
        "I never look at receipts.",
    ]

    bullets = "1. I check the price first.\n- I never look at receipts.\n"
    assert [c["text"] for c in generate.parse_candidates(bullets)] == [
        "I check the price first.",
        "I never look at receipts.",
    ]

    assert generate.parse_candidates("") == []
    assert generate.parse_candidates("[]") == []
    assert generate.parse_candidates(
        '[{"topic_domain": "not-a-domain", "text": "hello there friend"}]'
    ) == [{"topic_domain": None, "text": "hello there friend"}]


# --------------------------------------------------------------------------
# Round trip with a fake client
# --------------------------------------------------------------------------


def test_reduced_round_trip_validates(tmp_path, reduced) -> None:
    bank = generate.generate_bank(FakeClient(), spec=reduced, seed=13)

    paths = write_bank(bank, public_dir=tmp_path / "public", private_dir=tmp_path / "keys")
    reloaded = read_bank(tmp_path / "keys")
    public_payload = read_json(paths["public"])

    assert validate.validate_bank(reloaded, public_payload=public_payload) == []
    assert len(reloaded.closed) == reduced.n_closed == 45
    assert len(reloaded.open_items) == 8
    assert reloaded.seed == 13
    assert reloaded.spec == reduced
    assert [item.item_id for item in reloaded.closed][:3] == ["q001", "q002", "q003"]
    assert reloaded.closed[-1].item_id == "q045"
    assert reloaded.open_items[0].item_id == "oe01"


def test_full_bank_round_trip_validates(tmp_path) -> None:
    bank = generate.generate_bank(FakeClient(), spec=BankSpec(), seed=2026)

    paths = write_bank(bank, public_dir=tmp_path / "public", private_dir=tmp_path / "keys")
    reloaded = read_bank(tmp_path / "keys")
    problems = validate.validate_bank(reloaded, public_payload=read_json(paths["public"]))

    assert problems == []
    table = validate.design_table(reloaded)
    assert table["n_closed"] == 252
    assert table["n_open"] == 20
    assert table["distinct_cross_pairs"] == 28
    assert table["by_cell"]["primary/likert5/positive"] == 88
    assert table["by_cell"]["primary/likert5/negative"] == 40
    assert table["by_cell"]["primary/binary/positive"] == 40
    assert table["by_cell"]["primary/binary/negative"] == 24
    assert json.dumps(table), "the design table must be JSON-serialisable"
    assert sum(table["items_per_topic_domain"].values()) == 252 + 20
    assert all(count >= 24 for count in table["items_touching_dimension"].values())


def test_public_file_carries_no_planted_design(tmp_path, reduced) -> None:
    bank = generate.generate_bank(FakeClient(), spec=reduced, seed=4)
    paths = write_bank(bank, public_dir=tmp_path / "public", private_dir=tmp_path / "keys")

    public_text = paths["public"].read_text(encoding="utf-8")
    for forbidden in PRIVATE_ONLY_KEYS:
        assert forbidden not in public_text, f"public file mentions {forbidden}"
    for number in ("0.7", "0.5", "0.25", "1.0", "-1.0", "-0.7", "-0.25"):
        assert number not in public_text, f"public file contains the loading value {number}"

    payload = json.loads(public_text)
    assert set(payload["items"][0]) == {"item_id", "text", "type", "options", "topic_domain"}
    assert set(payload["open_ended"][0]) == {"item_id", "text"}
    assert payload["items"][0]["options"][0] == {"value": 1, "label": "strongly disagree"}
    binary = next(row for row in payload["items"] if row["type"] == "binary")
    assert [option["label"] for option in binary["options"]] == ["yes", "no"]

    # The private side does hold all of it.
    private_text = paths["private"].read_text(encoding="utf-8")
    assert '"loadings"' in private_text
    assert '"negatively_keyed"' in private_text
    assert '"strength_class"' in private_text
    assert '"design_notes"' in private_text
    assert '"target_dims"' in paths["open_private"].read_text(encoding="utf-8")


def test_the_same_sentence_is_only_ever_accepted_once(reduced) -> None:
    client = TwinClient()
    bank = generate.generate_bank(client, spec=reduced, seed=9)

    assert len(client.calls) > 10, "the twin sentence must be offered in many separate calls"
    assert sum(1 for item in bank.closed + bank.open_items if item.text == TWIN) == 1
    keys = [normalize_text(item.text) for item in bank.closed + bank.open_items]
    assert len(keys) == len(set(keys))


def test_duplicate_drafts_are_dropped_and_redrafted(reduced) -> None:
    client = RepeatingClient()
    bank = generate.generate_bank(client, spec=reduced, seed=8)

    keys = [normalize_text(item.text) for item in bank.closed + bank.open_items]
    assert len(keys) == len(set(keys)), "a repeated draft made it into the bank"
    assert client.first_replies > 0
    assert any("RETRY" in call["user"] for call in client.calls)
    assert validate.validate_bank(bank) == []


def test_a_client_that_omits_topic_labels_still_fills_every_slot(reduced) -> None:
    bank = generate.generate_bank(FencedClient(), spec=reduced, seed=6)
    assert all(item.text for item in bank.closed)
    assert all(item.topic_domain in TOPIC_DOMAINS for item in bank.closed)
    assert validate.validate_bank(bank) == []


def test_a_client_that_never_answers_raises(reduced) -> None:
    class SilentClient(LLMClient):
        def generate(self, system, user, temperature=0.7, max_tokens=512):
            return LLMResponse(text="[]", input_tokens=0, output_tokens=0, model="fake")

    with pytest.raises(RuntimeError, match="drafting fell short"):
        generate.generate_bank(SilentClient(), spec=reduced, seed=1, max_retries=1)


# --------------------------------------------------------------------------
# The validator catches things
# --------------------------------------------------------------------------


def test_validator_catches_a_duplicate_text(reduced) -> None:
    bank = generate.generate_bank(FakeClient(), spec=reduced, seed=21)
    bank.closed[3].text = bank.closed[2].text
    assert any("duplicate item text" in p for p in validate.validate_bank(bank))


def test_validator_catches_wrong_counts_and_loadings(reduced) -> None:
    bank = generate.generate_bank(FakeClient(), spec=reduced, seed=22)
    bank.closed.pop()
    assert any("closed items" in p for p in validate.validate_bank(bank))

    bank = generate.generate_bank(FakeClient(), spec=reduced, seed=23)
    bank.closed[0].loadings = {"TRU": 0.4}
    assert any("primary loading magnitude" in p for p in validate.validate_bank(bank))

    bank = generate.generate_bank(FakeClient(), spec=reduced, seed=23)
    bank.closed[0].negatively_keyed = True
    assert any("keying flag" in p for p in validate.validate_bank(bank))


def test_validator_catches_a_leaky_public_file(reduced) -> None:
    bank = generate.generate_bank(FakeClient(), spec=reduced, seed=24)
    leaky = bank.public_payload()
    leaky["items"][0]["loadings"] = {"TRU": 1.0}
    assert any("planted-design keys" in p for p in validate.validate_bank(bank, public_payload=leaky))


def test_validator_catches_a_missing_frozen_zero_pair(reduced) -> None:
    bank = generate.generate_bank(FakeClient(), spec=reduced, seed=25)
    victim = next(item for item in bank.closed if tuple(sorted(item.dims)) == ("RSK", "TRU"))
    victim.loadings = {"SOC": 0.7, "TEC": 0.5}
    assert any("zero-correlation pair" in p for p in validate.validate_bank(bank))


def test_validator_catches_a_broken_id_sequence(reduced) -> None:
    bank = generate.generate_bank(FakeClient(), spec=reduced, seed=26)
    bank.closed[5].item_id = "q999"
    assert any("sequential" in p for p in validate.validate_bank(bank))


def test_validator_catches_a_distractor_with_a_loading(reduced) -> None:
    bank = generate.generate_bank(FakeClient(), spec=reduced, seed=27)
    distractor = next(item for item in bank.closed if item.strength_class == "none")
    distractor.loadings = {"ENV": 0.0, "SOC": 0.9}
    assert any("non-zero loading" in p for p in validate.validate_bank(bank))


# --------------------------------------------------------------------------
# Post-freeze amendments
# --------------------------------------------------------------------------


def an_amendment(item_id: str, **overrides) -> dict:
    entry = {
        "item_id": item_id,
        "field": "loadings.TRU",
        "old_value": 0.25,
        "new_value": -0.25,
        "date": "2026-07-31",
        "authorized_by": "owner (test)",
        "reason": "the verifier flagged this item before any fitting",
        "log_citation": "results/PROJECT_LOG.md, some entry heading",
    }
    entry.update(overrides)
    return entry


def test_a_written_up_amendment_validates_and_survives_a_round_trip(tmp_path, reduced) -> None:
    bank = generate.generate_bank(FakeClient(), spec=reduced, seed=28)
    weak = next(item for item in bank.closed if item.strength_class == "weak")
    bank.amendments = [an_amendment(weak.item_id)]

    paths = write_bank(bank, public_dir=tmp_path / "public", private_dir=tmp_path / "keys")
    reloaded = read_bank(tmp_path / "keys")

    assert validate.validate_bank(reloaded, public_payload=read_json(paths["public"])) == []
    assert reloaded.amendments == bank.amendments
    assert set(an_amendment(weak.item_id)) == set(AMENDMENT_KEYS)
    # the amendment log is design material: it never crosses to the public side
    assert "amendments" in PRIVATE_ONLY_KEYS
    assert "amendments" not in paths["public"].read_text(encoding="utf-8")
    assert '"amendments"' in paths["private"].read_text(encoding="utf-8")


def test_a_bank_with_no_amendments_writes_no_amendment_block(tmp_path, reduced) -> None:
    """The file a never-amended bank writes is exactly the file it always wrote."""
    bank = generate.generate_bank(FakeClient(), spec=reduced, seed=29)
    assert bank.amendments == []
    assert "amendments" not in bank.private_payload()

    write_bank(bank, public_dir=tmp_path / "public", private_dir=tmp_path / "keys")
    assert "amendments" not in (tmp_path / "keys" / PRIVATE_FILENAME).read_text(encoding="utf-8")


def test_validator_catches_an_amendment_that_is_not_written_up(reduced) -> None:
    bank = generate.generate_bank(FakeClient(), spec=reduced, seed=30)
    weak = next(item for item in bank.closed if item.strength_class == "weak")

    incomplete = an_amendment(weak.item_id)
    del incomplete["log_citation"]
    del incomplete["authorized_by"]
    bank.amendments = [incomplete]
    problems = validate.validate_bank(bank)
    assert any("missing field(s)" in p and "log_citation" in p for p in problems)

    bank.amendments = [an_amendment(weak.item_id, reason="   ")]
    assert any("empty field(s)" in p for p in validate.validate_bank(bank))

    bank.amendments = [an_amendment("q999")]
    assert any("not in the bank" in p for p in validate.validate_bank(bank))

    bank.amendments = [an_amendment(weak.item_id, new_value=0.25)]
    assert any("old and new value are the same" in p for p in validate.validate_bank(bank))


# --------------------------------------------------------------------------
# The command line
# --------------------------------------------------------------------------


def test_generate_then_validate_from_the_command_line(tmp_path, capsys) -> None:
    keys_dir = tmp_path / "keys"
    public_dir = tmp_path / "public"

    generated = generate.main(
        [
            "--out-truth", str(keys_dir),
            "--out-public", str(public_dir),
            "--seed", "77",
            "--reduced",
            "--quiet",
        ],
        client=FakeClient(),
    )
    assert generated == 0
    assert (public_dir / PUBLIC_FILENAME).is_file()
    assert (keys_dir / PRIVATE_FILENAME).is_file()
    assert (keys_dir / OPEN_PRIVATE_FILENAME).is_file()

    checked = validate.main(["--truth", str(keys_dir), "--public", str(public_dir), "--table"])
    assert checked == 0
    assert "closed items" in capsys.readouterr().out


def test_generate_refuses_to_run_without_output_directories() -> None:
    assert generate.main([], client=FakeClient()) == 2


def test_validate_exits_nonzero_on_a_broken_bank(tmp_path, reduced) -> None:
    bank = generate.generate_bank(FakeClient(), spec=reduced, seed=31)
    bank.closed[1].text = bank.closed[0].text
    write_bank(bank, public_dir=tmp_path / "public", private_dir=tmp_path / "keys")

    assert validate.main(["--truth", str(tmp_path / "keys")]) == 1


def test_output_directories_are_only_ever_command_line_arguments() -> None:
    """Nothing in the bank package knows where the planted design is kept."""
    args = generate.build_arg_parser().parse_args([])
    assert args.out_truth is None
    assert args.out_public is None

    for name in (PUBLIC_FILENAME, PRIVATE_FILENAME, OPEN_PRIVATE_FILENAME):
        assert "/" not in name and "\\" not in name
