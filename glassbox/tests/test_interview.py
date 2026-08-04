"""Tests for M2, the responder runtime.

Everything here is synthetic and offline: a small bank, three cards, no model
call anywhere. Four things are checked, in order of how badly they would hurt
if wrong:

  1. The prompt says the right thing -- the answer format matches the item
     type, the persona's wobble reaches the noise paragraph, and each round
     carries its own session line.
  2. The fixed retest selection is reproducible, stratified, and stays fixed
     once it is on disk.
  3. Nothing already answered is ever asked again.
  4. Answers are read strictly: the tolerable spellings normalize, anything
     ambiguous is refused, and re-parsing never duplicates a cell.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.bank.schema import DIMENSIONS, options_for
from src.interview import (
    assemble_transcripts,
    emit_answer_jobs,
    emit_open_jobs,
    parse_answers,
    parse_open_answers,
    responder_prompts,
)
from src.personas.card_prompts import build_noise_instruction

# --------------------------------------------------------------------------
# synthetic fixtures
# --------------------------------------------------------------------------

PRIMARY_PER_DIM = 4
N_CROSS = 8
N_WEAK = 4
N_DISTRACTOR = 4


def make_bank() -> list[dict]:
    """A small bank with every design cell present, as private item records."""
    items: list[dict] = []

    def add(loadings: dict[str, float], strength: str, item_type: str) -> None:
        index = len(items) + 1
        items.append(
            {
                "item_id": f"q{index:03d}",
                "text": f"Statement number {index}.",
                "type": item_type,
                "topic_domain": "money",
                "loadings": loadings,
                "strength_class": strength,
                "negatively_keyed": any(v < 0 for v in loadings.values()),
            }
        )

    for dim in DIMENSIONS:
        for position in range(PRIMARY_PER_DIM):
            sign = -1.0 if position % 2 else 1.0
            item_type = "likert5" if position < 2 else "binary"
            add({dim: sign * 1.0}, "strong", item_type)

    for position in range(N_CROSS):
        primary = DIMENSIONS[position % len(DIMENSIONS)]
        secondary = DIMENSIONS[(position + 3) % len(DIMENSIONS)]
        add({primary: 0.7, secondary: 0.5}, "strong", "likert5")

    for position in range(N_WEAK):
        add({DIMENSIONS[position]: 0.25}, "weak", "likert5")

    for _ in range(N_DISTRACTOR):
        add({}, "none", "binary")

    return items


def write_bank(tmp_path: Path) -> tuple[Path, Path, list[dict]]:
    """Write the public bank and the design file. Returns both paths."""
    items = make_bank()
    public = {
        "bank_version": 1,
        "n_closed": len(items),
        "items": [
            {
                "item_id": row["item_id"],
                "text": row["text"],
                "type": row["type"],
                "options": options_for(row["type"]),
                "topic_domain": row["topic_domain"],
            }
            for row in items
        ],
    }
    design = {"bank_version": 1, "n_closed": len(items), "items": items}

    public_path = tmp_path / "bank_items.json"
    design_path = tmp_path / "design" / "bank_truth.json"
    design_path.parent.mkdir(parents=True, exist_ok=True)
    public_path.write_text(json.dumps(public), encoding="utf-8")
    design_path.write_text(json.dumps(design), encoding="utf-8")
    return public_path, design_path, items


def write_personas(tmp_path: Path, n: int = 3, wobble: float = 0.2) -> tuple[Path, Path]:
    cards_dir = tmp_path / "cards"
    noise_dir = tmp_path / "noise"
    cards_dir.mkdir(parents=True, exist_ok=True)
    noise_dir.mkdir(parents=True, exist_ok=True)
    for index in range(1, n + 1):
        pid = f"p{index:04d}"
        card = {
            "pid": pid,
            "biography": f"Person {index} repairs bicycles in a small town and walks everywhere.",
            "quirks": [f"keeps receipts in a tin, batch {index}", "never answers the door after nine"],
            "speech_style": "Short sentences, dry, dwells on what things cost.",
        }
        (cards_dir / f"{pid}.json").write_text(json.dumps(card), encoding="utf-8")
        (noise_dir / f"{pid}.json").write_text(
            json.dumps({"pid": pid, "wobble": wobble}), encoding="utf-8"
        )
    return cards_dir, noise_dir


CARD = {
    "pid": "p0001",
    "biography": "Ana drives a delivery van and keeps every receipt in a shoebox.",
    "quirks": ["counts change twice", "walks the same route home"],
    "speech_style": "Blunt, short sentences.",
}

LIKERT_ITEM = {"item_id": "q001", "text": "I compare prices before I buy.", "type": "likert5"}
BINARY_ITEM = {"item_id": "q002", "text": "I have a savings account.", "type": "binary"}


# --------------------------------------------------------------------------
# 1. the prompt
# --------------------------------------------------------------------------


def test_format_instruction_matches_the_item_type() -> None:
    likert = responder_prompts.build_answer_prompt(CARD, 0.2, LIKERT_ITEM, "main-abc")
    binary = responder_prompts.build_answer_prompt(CARD, 0.2, BINARY_ITEM, "main-abc")

    assert "single digit 1-5" in likert["system"]
    assert "1=strongly disagree" in likert["system"]
    assert "Output ONLY the digit." in likert["system"]
    assert "yes or no" not in likert["system"]

    assert "Answer only yes or no." in binary["system"]
    assert "single digit" not in binary["system"]


def test_unknown_item_type_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown item type"):
        responder_prompts.format_instruction("slider")


def test_the_card_is_in_the_system_prompt() -> None:
    system = responder_prompts.build_system_prompt(CARD, 0.2, "likert5")
    assert CARD["biography"] in system
    assert "counts change twice" in system
    assert CARD["speech_style"] in system


def test_a_card_with_no_biography_is_refused() -> None:
    with pytest.raises(ValueError, match="no biography"):
        responder_prompts.build_system_prompt({"pid": "p0001"}, 0.2, "likert5")


@pytest.mark.parametrize("wobble", [0.05, 0.2, 0.45, 0.9])
def test_the_wobble_reaches_the_noise_paragraph(wobble: float) -> None:
    system = responder_prompts.build_system_prompt(CARD, wobble, "likert5")
    assert build_noise_instruction(wobble) in system


def test_different_wobbles_give_different_prompts() -> None:
    low = responder_prompts.build_system_prompt(CARD, 0.1, "likert5")
    high = responder_prompts.build_system_prompt(CARD, 0.4, "likert5")
    assert "1 in 10" in low
    assert "4 in 10" in high
    assert low != high


def test_the_session_salt_is_in_the_user_message() -> None:
    session = responder_prompts.session_id("p0001", "main", 7)
    prompt = responder_prompts.build_answer_prompt(CARD, 0.2, LIKERT_ITEM, session)
    assert prompt["user"].startswith(f"Session: {session}")
    assert LIKERT_ITEM["text"] in prompt["user"]


def test_the_retest_session_is_a_different_context() -> None:
    main = responder_prompts.session_id("p0001", "main", 7)
    retest = responder_prompts.session_id("p0001", "retest", 7)
    assert main != retest
    assert main == responder_prompts.session_id("p0001", "main", 7)
    assert main != responder_prompts.session_id("p0002", "main", 7)
    assert main != responder_prompts.session_id("p0001", "main", 8)


def test_an_unknown_round_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown round"):
        responder_prompts.session_id("p0001", "followup", 1)


# --------------------------------------------------------------------------
# 2. the fixed retest selection
# --------------------------------------------------------------------------


def selection_classes() -> tuple[list[str], dict[str, dict]]:
    items = make_bank()
    classes = emit_answer_jobs.classify_items({"items": items})
    return [row["item_id"] for row in items], classes


def test_item_classes_come_out_of_the_design() -> None:
    _, classes = selection_classes()
    kinds = [entry["kind"] for entry in classes.values()]
    assert kinds.count("primary") == PRIMARY_PER_DIM * len(DIMENSIONS)
    assert kinds.count("cross") == N_CROSS
    assert kinds.count("weak") == N_WEAK
    assert kinds.count("distractor") == N_DISTRACTOR


def test_the_retest_selection_is_seed_stable() -> None:
    pool, classes = selection_classes()
    first = emit_answer_jobs.select_retest_items(pool, classes, "p0001", seed=11)
    again = emit_answer_jobs.select_retest_items(pool, classes, "p0001", seed=11)
    other_persona = emit_answer_jobs.select_retest_items(pool, classes, "p0002", seed=11)
    other_seed = emit_answer_jobs.select_retest_items(pool, classes, "p0001", seed=12)

    assert first == again
    assert first != other_persona
    assert first != other_seed
    assert len(set(first)) == len(first) == emit_answer_jobs.RETEST_ITEMS


def test_the_retest_selection_is_stratified() -> None:
    pool, classes = selection_classes()
    chosen = emit_answer_jobs.select_retest_items(pool, classes, "p0007", seed=3)

    per_dim = {dim: 0 for dim in DIMENSIONS}
    per_class = {"primary": 0, "cross": 0, "weak": 0, "distractor": 0}
    for item_id in chosen:
        entry = classes[item_id]
        per_class[entry["kind"]] += 1
        if entry["kind"] == "primary":
            per_dim[entry["dims"][0]] += 1

    assert all(count >= emit_answer_jobs.RETEST_MIN_PER_DIM for count in per_dim.values())
    # 30 items: 16 primary (2 per dimension), then 14 spread over the other
    # three classes in proportion to their sizes (8 cross, 4 weak, 4 distractor).
    assert per_class["primary"] == 16
    assert per_class["cross"] == 7
    assert per_class["weak"] == 4
    assert per_class["distractor"] == 3
    assert sum(per_class.values()) == emit_answer_jobs.RETEST_ITEMS


def test_proportional_quotas_split_the_remainder() -> None:
    quotas = emit_answer_jobs.proportional_quotas({"cross": 30, "weak": 15, "distractor": 15}, 14)
    assert sum(quotas.values()) == 14
    assert quotas["cross"] == 7
    assert quotas["weak"] + quotas["distractor"] == 7
    assert abs(quotas["weak"] - quotas["distractor"]) <= 1


def test_a_thin_pool_still_fills_up_to_what_exists() -> None:
    pool, classes = selection_classes()
    small = pool[:20]
    chosen = emit_answer_jobs.select_retest_items(small, classes, "p0001", seed=1)
    assert len(chosen) == 20
    assert set(chosen) == set(small)


def test_the_selection_file_is_written_once_and_reused(tmp_path: Path) -> None:
    public_bank, design, _ = write_bank(tmp_path)
    cards_dir, noise_dir = write_personas(tmp_path, n=3)
    run_dir = tmp_path / "run"

    args = [
        "--cards-dir", str(cards_dir),
        "--noise-dir", str(noise_dir),
        "--public-bank", str(public_bank),
        "--bank-design", str(design),
        "--out", str(run_dir),
        "--round", "retest",
        "--session-seed", "5",
    ]
    assert emit_answer_jobs.main(args) == 0
    first = json.loads((run_dir / "retest_selection.json").read_text(encoding="utf-8"))

    # A different seed must not move a selection that is already fixed on disk.
    assert emit_answer_jobs.main(args[:-1] + ["99"]) == 0
    second = json.loads((run_dir / "retest_selection.json").read_text(encoding="utf-8"))
    assert first["selection"] == second["selection"]
    assert first["seed"] == second["seed"] == 5


def test_retest_needs_the_design_file(tmp_path: Path) -> None:
    public_bank, _, _ = write_bank(tmp_path)
    cards_dir, noise_dir = write_personas(tmp_path, n=1)
    with pytest.raises(SystemExit):
        emit_answer_jobs.main(
            [
                "--cards-dir", str(cards_dir),
                "--noise-dir", str(noise_dir),
                "--public-bank", str(public_bank),
                "--out", str(tmp_path / "run"),
                "--round", "retest",
            ]
        )


# --------------------------------------------------------------------------
# 3. never re-ask what is already answered
# --------------------------------------------------------------------------


def emit_main(tmp_path: Path, run_dir: Path, n_personas: int = 2) -> int:
    public_bank, _, _ = write_bank(tmp_path)
    cards_dir, noise_dir = write_personas(tmp_path, n=n_personas)
    return emit_answer_jobs.main(
        [
            "--cards-dir", str(cards_dir),
            "--noise-dir", str(noise_dir),
            "--public-bank", str(public_bank),
            "--out", str(run_dir),
        ]
    )


def prompt_rows(run_dir: Path) -> list[dict]:
    return emit_answer_jobs.read_jsonl(run_dir / "prompts.jsonl")


def test_a_fresh_run_emits_every_cell(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    emit_main(tmp_path, run_dir, n_personas=2)
    rows = prompt_rows(run_dir)

    n_items = len(make_bank())
    assert len(rows) == 2 * n_items
    assert {row["round"] for row in rows} == {"main"}
    assert {row["pid"] for row in rows} == {"p0001", "p0002"}
    assert all(row["system"] and row["user"] for row in rows)


def test_a_second_emit_after_a_full_answer_file_emits_nothing(tmp_path: Path, capsys) -> None:
    run_dir = tmp_path / "run"
    emit_main(tmp_path, run_dir, n_personas=2)
    rows = prompt_rows(run_dir)

    (run_dir / "answers.jsonl").write_text(
        "".join(
            json.dumps({"pid": row["pid"], "item_id": row["item_id"], "round": "main", "answer": 3})
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )

    emit_main(tmp_path, run_dir, n_personas=2)
    printed = capsys.readouterr().out
    assert prompt_rows(run_dir) == []
    assert f"emitted 0 prompts, skipped {len(rows)}" in printed


def test_a_second_emit_fills_only_the_gap(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    emit_main(tmp_path, run_dir, n_personas=2)
    rows = prompt_rows(run_dir)
    done = rows[: len(rows) // 2]

    (run_dir / "answers.jsonl").write_text(
        "".join(
            json.dumps({"pid": row["pid"], "item_id": row["item_id"], "round": "main", "answer": 3})
            + "\n"
            for row in done
        ),
        encoding="utf-8",
    )

    emit_main(tmp_path, run_dir, n_personas=2)
    left = prompt_rows(run_dir)
    assert len(left) == len(rows) - len(done)
    assert {(r["pid"], r["item_id"]) for r in left}.isdisjoint(
        {(r["pid"], r["item_id"]) for r in done}
    )


def test_the_item_file_restricts_the_round(tmp_path: Path) -> None:
    public_bank, _, items = write_bank(tmp_path)
    cards_dir, noise_dir = write_personas(tmp_path, n=2)
    wanted = [row["item_id"] for row in items[:5]]
    ids_file = tmp_path / "items.txt"
    ids_file.write_text("\n".join(wanted) + "\n# a comment\n", encoding="utf-8")

    run_dir = tmp_path / "run"
    assert emit_answer_jobs.main(
        [
            "--cards-dir", str(cards_dir),
            "--noise-dir", str(noise_dir),
            "--public-bank", str(public_bank),
            "--out", str(run_dir),
            "--items", str(ids_file),
            "--limit-personas", "1",
        ]
    ) == 0

    rows = prompt_rows(run_dir)
    assert [row["item_id"] for row in rows] == wanted
    assert {row["pid"] for row in rows} == {"p0001"}


def test_a_persona_without_a_wobble_is_skipped(tmp_path: Path) -> None:
    cards_dir, noise_dir = write_personas(tmp_path, n=3)
    (noise_dir / "p0002.json").unlink()
    people = emit_answer_jobs.load_personas(cards_dir, noise_dir)
    assert [pid for pid, _, _ in people] == ["p0001", "p0003"]


# --------------------------------------------------------------------------
# 4. reading the answers back
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("5", 5),
        (" 4 ", 4),
        ("5.", 5),
        ("\n2\n", 2),
        ("3.", 3),
        ("strongly agree", 5),
        ("Strongly Agree.", 5),
        ("strongly disagree", 1),
        ("disagree", 2),
        ("Agree", 4),
        ("neither agree nor disagree", 3),
        ("neutral", 3),
    ],
)
def test_likert_answers_normalize(raw: str, expected: int) -> None:
    answer, reason = parse_answers.normalize_answer(raw, "likert5")
    assert answer == expected, reason
    assert reason == ""


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("yes", "yes"),
        ("Yes,", "yes"),
        ("YES", "yes"),
        (" no. ", "no"),
        ("Nope", "no"),
        ("y", "yes"),
        ("N", "no"),
        ("Yes, I do that every week.", "yes"),
        ("Not really, no.", "no"),
    ],
)
def test_binary_answers_normalize(raw: str, expected: str) -> None:
    answer, reason = parse_answers.normalize_answer(raw, "binary")
    assert answer == expected, reason
    assert reason == ""


@pytest.mark.parametrize(
    "raw,item_type",
    [
        ("", "likert5"),
        ("hmm", "likert5"),
        ("maybe a 3 or a 4", "likert5"),
        ("a single digit 1-5", "likert5"),
        ("I agree and I disagree", "likert5"),
        ("7", "likert5"),
        ("yes", "likert5"),
        ("", "binary"),
        ("maybe", "binary"),
        ("yes and no", "binary"),
        ("4", "binary"),
        ("strongly agree", "binary"),
    ],
)
def test_unreadable_answers_are_rejected(raw: str, item_type: str) -> None:
    answer, reason = parse_answers.normalize_answer(raw, item_type)
    assert answer is None
    assert reason


def completions_file(tmp_path: Path, rows: list[dict], name: str = "completions.jsonl") -> Path:
    path = tmp_path / name
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_parsing_writes_answers_and_rejects(tmp_path: Path) -> None:
    public_bank, _, items = write_bank(tmp_path)
    likert = next(row["item_id"] for row in items if row["type"] == "likert5")
    binary = next(row["item_id"] for row in items if row["type"] == "binary")

    path = completions_file(
        tmp_path,
        [
            {"pid": "p0001", "item_id": likert, "round": "main", "completion": "4."},
            {"pid": "p0001", "item_id": binary, "round": "main", "completion": "Yes,"},
            {"pid": "p0002", "item_id": likert, "round": "main", "completion": "somewhere in between"},
            {"pid": "p0002", "item_id": "q999", "round": "main", "completion": "3"},
        ],
    )
    run_dir = tmp_path / "run"
    summary = parse_answers.parse_file(path, public_bank, run_dir)

    assert summary["kept"] == 2
    assert summary["rejected"] == 2
    answers = emit_answer_jobs.read_jsonl(run_dir / "answers.jsonl")
    assert {(row["pid"], row["item_id"], row["answer"]) for row in answers} == {
        ("p0001", likert, 4),
        ("p0001", binary, "yes"),
    }
    assert all(row["raw"] for row in answers)

    rejects = emit_answer_jobs.read_jsonl(run_dir / "answer_rejects.jsonl")
    reasons = {row["item_id"]: row["reason"] for row in rejects}
    assert "not in the public bank" in reasons["q999"]
    assert reasons[likert]


def test_reparsing_never_duplicates_a_cell(tmp_path: Path) -> None:
    public_bank, _, items = write_bank(tmp_path)
    likert = next(row["item_id"] for row in items if row["type"] == "likert5")
    path = completions_file(
        tmp_path,
        [
            {"pid": "p0001", "item_id": likert, "round": "main", "completion": "4"},
            {"pid": "p0001", "item_id": likert, "round": "retest", "completion": "5"},
            {"pid": "p0001", "item_id": likert, "round": "main", "completion": "1"},
        ],
    )
    run_dir = tmp_path / "run"

    first = parse_answers.parse_file(path, public_bank, run_dir)
    assert first["kept"] == 2
    assert first["duplicates"] == 1

    second = parse_answers.parse_file(path, public_bank, run_dir)
    assert second["kept"] == 0
    assert second["duplicates"] == 3

    answers = emit_answer_jobs.read_jsonl(run_dir / "answers.jsonl")
    assert len(answers) == 2
    assert {(row["item_id"], row["round"], row["answer"]) for row in answers} == {
        (likert, "main", 4),
        (likert, "retest", 5),
    }


def test_a_reject_leaves_the_queue_once_it_is_answered(tmp_path: Path) -> None:
    public_bank, _, items = write_bank(tmp_path)
    likert = next(row["item_id"] for row in items if row["type"] == "likert5")
    run_dir = tmp_path / "run"

    bad = completions_file(
        tmp_path,
        [{"pid": "p0001", "item_id": likert, "round": "main", "completion": "no idea"}],
        name="bad.jsonl",
    )
    parse_answers.parse_file(bad, public_bank, run_dir)
    assert len(emit_answer_jobs.read_jsonl(run_dir / "answer_rejects.jsonl")) == 1

    good = completions_file(
        tmp_path,
        [{"pid": "p0001", "item_id": likert, "round": "main", "completion": "2"}],
        name="good.jsonl",
    )
    summary = parse_answers.parse_file(good, public_bank, run_dir)
    assert summary["kept"] == 1
    assert summary["outstanding"] == 0
    assert emit_answer_jobs.read_jsonl(run_dir / "answer_rejects.jsonl") == []


def test_answers_close_the_loop_with_the_emitter(tmp_path: Path) -> None:
    """Emit, answer, parse, emit again: the second emit has nothing left to do."""
    public_bank, _, _ = write_bank(tmp_path)
    cards_dir, noise_dir = write_personas(tmp_path, n=1)
    run_dir = tmp_path / "run"
    args = [
        "--cards-dir", str(cards_dir),
        "--noise-dir", str(noise_dir),
        "--public-bank", str(public_bank),
        "--out", str(run_dir),
    ]
    emit_answer_jobs.main(args)
    rows = prompt_rows(run_dir)

    types = parse_answers.item_types(public_bank)
    completions = completions_file(
        tmp_path,
        [
            {
                "pid": row["pid"],
                "item_id": row["item_id"],
                "round": row["round"],
                "completion": "4" if types[row["item_id"]] == "likert5" else "yes",
            }
            for row in rows
        ],
    )
    parse_answers.parse_file(completions, public_bank, run_dir)

    emit_answer_jobs.main(args)
    assert prompt_rows(run_dir) == []


# --------------------------------------------------------------------------
# 5. the Stage 3 interview: rounds, open prompts, transcripts
# --------------------------------------------------------------------------

OPEN_ITEMS = [
    {"item_id": "oe01", "text": "Tell me about a time you dealt with a bank."},
    {"item_id": "oe15", "text": "Tell me about handing a job over to a new device."},
    {"item_id": "oe16", "text": "Tell me about a gathering you keep going back to."},
]

SCRIPTED_OPEN = ["oe01", "oe16", "oe15"]


def write_public_bank_with_open(tmp_path: Path, scripted_closed: list[str]) -> Path:
    """A public bank carrying both the closed items and the open prompts."""
    payload = {
        "bank_version": 1,
        "items": [
            {"item_id": item_id, "text": f"Statement {item_id}.", "type": "likert5",
             "options": options_for("likert5"), "topic_domain": "money"}
            for item_id in scripted_closed
        ],
        "open_ended": OPEN_ITEMS,
    }
    path = tmp_path / "bank_items.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def write_splits(tmp_path: Path, scripted_closed: list[str]) -> Path:
    path = tmp_path / "splits.json"
    path.write_text(
        json.dumps({"interview_closed": scripted_closed, "interview_open": SCRIPTED_OPEN}),
        encoding="utf-8",
    )
    return path


def write_profiles(tmp_path: Path, pids: list[str]) -> Path:
    profiles = tmp_path / "profiles"
    profiles.mkdir(parents=True, exist_ok=True)
    for index, pid in enumerate(pids, start=1):
        (profiles / f"{pid}.json").write_text(
            json.dumps({
                "pid": pid, "age": 30 + index, "occupation": "barista",
                "city_size": "small town", "household": "alone",
                "region_type": "urban",
            }),
            encoding="utf-8",
        )
    return profiles


# ---- the round tags ------------------------------------------------------


def test_the_interview_round_is_a_session_of_its_own() -> None:
    main = responder_prompts.session_id("p0001", "main", 548)
    retest = responder_prompts.session_id("p0001", "retest", 548)
    interview = responder_prompts.session_id("p0001", "interview1", 548)
    assert len({main, retest, interview}) == 3
    assert interview.startswith("interview1-")
    assert interview == responder_prompts.session_id("p0001", "interview1", 548)


def test_the_closed_emitter_refuses_an_interview_round(tmp_path: Path) -> None:
    """A closed interview answer is drawn from the sweep, never asked again."""
    public_bank, design, _ = write_bank(tmp_path)
    cards_dir, noise_dir = write_personas(tmp_path)
    with pytest.raises(SystemExit):
        emit_answer_jobs.main([
            "--cards-dir", str(cards_dir), "--noise-dir", str(noise_dir),
            "--public-bank", str(public_bank), "--out", str(tmp_path / "run"),
            "--round", "interview1",
        ])


def test_the_closed_rounds_are_exactly_two() -> None:
    assert responder_prompts.CLOSED_ROUNDS == ("main", "retest")
    assert "interview1" in responder_prompts.INTERVIEW_ROUNDS
    assert set(responder_prompts.ROUNDS) == {"main", "retest", "interview1"}


# ---- the open-ended prompts ---------------------------------------------


def test_the_open_instruction_replaces_the_one_token_one() -> None:
    system = responder_prompts.build_system_prompt(CARD, 0.2, "open")
    assert "your own words" in system
    assert "Output ONLY the digit." not in system
    assert "yes or no" not in system
    assert "no explanation, no extra words" not in system, (
        "the terse line cancels the open instruction above it"
    )
    assert CARD["biography"] in system
    assert build_noise_instruction(0.2) in system


def test_open_prompts_carry_the_card_the_question_and_the_session_line() -> None:
    session = responder_prompts.session_id("p0001", "interview1", 548)
    prompt = emit_open_jobs.build_open_prompt(CARD, 0.2, OPEN_ITEMS[0], session)
    assert prompt["user"].startswith(f"Session: {session}")
    assert OPEN_ITEMS[0]["text"] in prompt["user"]
    assert CARD["biography"] in prompt["system"]


def test_emit_open_writes_one_prompt_per_persona_and_question(tmp_path: Path) -> None:
    cards_dir, noise_dir = write_personas(tmp_path, n=4)
    public_bank = write_public_bank_with_open(tmp_path, ["q001"])
    splits = write_splits(tmp_path, ["q001"])
    run_dir = tmp_path / "run"

    assert emit_open_jobs.main([
        "--cards-dir", str(cards_dir), "--noise-dir", str(noise_dir),
        "--public-bank", str(public_bank), "--splits", str(splits),
        "--out", str(run_dir),
    ]) == 0

    rows = [json.loads(line) for line in
            (run_dir / "prompts_open.jsonl").read_text().splitlines()]
    assert len(rows) == 4 * 3
    assert all(set(row) == {"pid", "item_id", "round", "system", "user"} for row in rows)
    assert {row["round"] for row in rows} == {"interview1"}
    # scripted order, repeated per persona
    assert [row["item_id"] for row in rows[:3]] == SCRIPTED_OPEN
    assert len({row["pid"] for row in rows}) == 4

    manifest = json.loads((run_dir / "open_manifest.json").read_text())
    assert manifest["emitted"] == 12
    assert manifest["prompt_ids"] == SCRIPTED_OPEN
    assert manifest["session_seed"] == 548


def test_emit_open_skips_what_is_already_answered(tmp_path: Path) -> None:
    cards_dir, noise_dir = write_personas(tmp_path, n=2)
    public_bank = write_public_bank_with_open(tmp_path, ["q001"])
    splits = write_splits(tmp_path, ["q001"])
    run_dir = tmp_path / "run"
    args = ["--cards-dir", str(cards_dir), "--noise-dir", str(noise_dir),
            "--public-bank", str(public_bank), "--splits", str(splits),
            "--out", str(run_dir)]

    emit_open_jobs.main(args)
    (run_dir / "open_answers.jsonl").write_text(
        json.dumps({"pid": "p0001", "prompt_id": "oe01", "round": "interview1",
                    "answer": "It went badly and I never went back."}) + "\n",
        encoding="utf-8",
    )
    emit_open_jobs.main(args)

    rows = [json.loads(line) for line in
            (run_dir / "prompts_open.jsonl").read_text().splitlines()]
    assert len(rows) == 5
    assert ("p0001", "oe01") not in {(row["pid"], row["item_id"]) for row in rows}


def test_emit_open_refuses_a_prompt_that_is_not_in_the_public_bank(tmp_path: Path) -> None:
    cards_dir, noise_dir = write_personas(tmp_path, n=1)
    public_bank = write_public_bank_with_open(tmp_path, ["q001"])
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"interview_closed": ["q001"],
                                  "interview_open": ["oe01", "oe99"]}))
    with pytest.raises(ValueError, match="not in the public bank"):
        emit_open_jobs.main([
            "--cards-dir", str(cards_dir), "--noise-dir", str(noise_dir),
            "--public-bank", str(public_bank), "--splits", str(splits),
            "--out", str(tmp_path / "run"),
        ])


# ---- parsing the open answers -------------------------------------------


def test_open_answers_are_tidied() -> None:
    answer, reason = parse_open_answers.clean_answer(
        '  "The bank lost my form twice.\n\n\n  I gave up and walked in."  '
    )
    assert reason == ""
    assert answer == "The bank lost my form twice.\nI gave up and walked in."


def test_an_echoed_session_line_is_dropped() -> None:
    answer, _ = parse_open_answers.clean_answer(
        "Session: interview1-abc123\n\nThey kept me waiting for an hour, so I left."
    )
    assert "Session:" not in answer
    assert answer.startswith("They kept me waiting")


@pytest.mark.parametrize("raw", ["", "   ", "Sure.", "Okay, will do."])
def test_empty_and_too_short_open_answers_are_rejected(raw: str) -> None:
    answer, reason = parse_open_answers.clean_answer(raw)
    assert answer == ""
    assert reason


def test_parsing_open_answers_never_duplicates_a_cell(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    completions = tmp_path / "completions_open.jsonl"
    completions.write_text(
        json.dumps({"pid": "p0001", "item_id": "oe01", "round": "interview1",
                    "completion": "The queue was long and nobody knew the rules."}) + "\n",
        encoding="utf-8",
    )
    first = parse_open_answers.parse_file(completions, run_dir)
    second = parse_open_answers.parse_file(completions, run_dir)

    assert first["kept"] == 1 and second["kept"] == 0
    assert second["duplicates"] == 1
    rows = [json.loads(line) for line in
            (run_dir / "open_answers.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    assert set(rows[0]) == {"pid", "prompt_id", "round", "answer"}


def test_an_open_reject_leaves_the_queue_once_it_is_answered(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"pid": "p0001", "item_id": "oe01",
                               "round": "interview1", "completion": "Yes."}) + "\n")
    summary = parse_open_answers.parse_file(bad, run_dir)
    assert summary["rejected"] == 1 and summary["outstanding"] == 1

    good = tmp_path / "good.jsonl"
    good.write_text(json.dumps({
        "pid": "p0001", "item_id": "oe01", "round": "interview1",
        "completion": "I queued for an hour and then the office shut."}) + "\n")
    summary = parse_open_answers.parse_file(good, run_dir)
    assert summary["kept"] == 1
    assert (run_dir / "open_answer_rejects.jsonl").read_text().strip() == ""


# ---- assembling the transcript ------------------------------------------


SCRIPTED_CLOSED = ["q003", "q001", "q002"]


def transcript_fixture(tmp_path: Path, pids: list[str] | None = None) -> dict:
    pids = pids or ["p0001", "p0002"]
    public_bank = write_public_bank_with_open(tmp_path, SCRIPTED_CLOSED)
    splits = write_splits(tmp_path, SCRIPTED_CLOSED)
    profiles = write_profiles(tmp_path, pids)

    answers = tmp_path / "answers_interview1.jsonl"
    answers.write_text("".join(
        json.dumps({"pid": pid, "item_id": item_id, "round": "interview1",
                    "answer": 4}) + "\n"
        for pid in pids for item_id in SCRIPTED_CLOSED), encoding="utf-8")

    return {"public_bank": public_bank, "splits": splits, "profiles": profiles,
            "answers": answers, "pids": pids, "out": tmp_path / "transcripts.jsonl"}


def run_assembler(fixture: dict, extra: list[str] | None = None) -> list[dict]:
    assemble_transcripts.main([
        "--profiles-dir", str(fixture["profiles"]),
        "--public-bank", str(fixture["public_bank"]),
        "--splits", str(fixture["splits"]),
        "--answers", str(fixture["answers"]),
        "--out", str(fixture["out"]),
        *(extra or []),
    ])
    return [json.loads(line) for line in fixture["out"].read_text().splitlines()]


def test_the_transcript_has_the_stage_three_schema(tmp_path: Path) -> None:
    fixture = transcript_fixture(tmp_path)
    records = run_assembler(fixture)

    assert [r["pid"] for r in records] == fixture["pids"]
    record = records[0]
    assert set(record) == {"pid", "profile", "turns", "open"}
    assert set(record["profile"]) == {"pid", "age", "occupation", "city_size",
                                      "household", "region_type"}
    assert [t["n"] for t in record["turns"]] == [1, 2, 3]
    assert all(set(t) == {"n", "item_id", "question", "answer"} for t in record["turns"])
    assert record["open"] == []


def test_the_turns_follow_the_scripted_order_not_the_file_order(tmp_path: Path) -> None:
    fixture = transcript_fixture(tmp_path, pids=["p0001"])
    # rewrite the answers in a deliberately wrong order
    fixture["answers"].write_text("".join(
        json.dumps({"pid": "p0001", "item_id": item_id, "round": "interview1",
                    "answer": 4}) + "\n"
        for item_id in reversed(SCRIPTED_CLOSED)), encoding="utf-8")
    records = run_assembler(fixture)
    assert [t["item_id"] for t in records[0]["turns"]] == SCRIPTED_CLOSED


def test_the_transcript_carries_the_public_question_text(tmp_path: Path) -> None:
    fixture = transcript_fixture(tmp_path)
    records = run_assembler(fixture)
    assert records[0]["turns"][0]["question"] == f"Statement {SCRIPTED_CLOSED[0]}."


def test_the_open_block_fills_in_on_a_second_run(tmp_path: Path) -> None:
    fixture = transcript_fixture(tmp_path, pids=["p0001"])
    assert run_assembler(fixture)[0]["open"] == []

    open_answers = tmp_path / "open_answers.jsonl"
    open_answers.write_text("".join(
        json.dumps({"pid": "p0001", "prompt_id": prompt_id, "round": "interview1",
                    "answer": f"Something happened about {prompt_id} and I dealt with it."}) + "\n"
        for prompt_id in SCRIPTED_OPEN), encoding="utf-8")

    records = run_assembler(fixture, ["--open-answers", str(open_answers)])
    block = records[0]["open"]
    assert [entry["prompt_id"] for entry in block] == SCRIPTED_OPEN
    assert all(set(entry) == {"prompt_id", "question", "answer"} for entry in block)
    assert block[0]["question"] == OPEN_ITEMS[0]["text"]
    assert records[0]["turns"], "the closed part must survive the second run"


def test_a_leaking_open_answer_is_dropped_and_counted(tmp_path: Path) -> None:
    fixture = transcript_fixture(tmp_path, pids=["p0001"])
    open_answers = tmp_path / "open_answers.jsonl"
    open_answers.write_text("".join(json.dumps(row) + "\n" for row in [
        {"pid": "p0001", "prompt_id": "oe01", "round": "interview1",
         "answer": "My institution trust is low, so I never go to the bank."},
        {"pid": "p0001", "prompt_id": "oe16", "round": "interview1",
         "answer": "We eat at my sister's every Sunday and I would hate a change."},
    ]), encoding="utf-8")

    records = run_assembler(fixture, ["--open-answers", str(open_answers)])
    kept = records[0]["open"]
    assert [entry["prompt_id"] for entry in kept] == ["oe16"]
    assert "institution trust" not in json.dumps(records[0])

    report = json.loads(Path(str(fixture["out"]) + ".report.json").read_text())
    assert report["open_answers_kept"] == 1
    assert report["open_answers_rejected_for_leaks"] == 1
    assert report["open_leak_rejects"][0]["pid"] == "p0001"
    assert report["open_leak_rejects"][0]["prompt_id"] == "oe01"
    assert "answer" not in report["open_leak_rejects"][0], (
        "the rejected text must not be stored either"
    )


def test_an_unexpected_profile_key_stops_the_run(tmp_path: Path) -> None:
    fixture = transcript_fixture(tmp_path, pids=["p0001"])
    path = fixture["profiles"] / "p0001.json"
    record = json.loads(path.read_text())
    record["wobble"] = 0.31
    path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="allowlist"):
        run_assembler(fixture)


def test_a_persona_without_a_public_profile_is_skipped(tmp_path: Path) -> None:
    fixture = transcript_fixture(tmp_path, pids=["p0001", "p0002"])
    (fixture["profiles"] / "p0002.json").unlink()
    records = run_assembler(fixture)
    assert [r["pid"] for r in records] == ["p0001"]
    report = json.loads(Path(str(fixture["out"]) + ".report.json").read_text())
    assert report["personas_without_a_public_profile"] == ["p0002"]


def test_a_short_script_is_reported(tmp_path: Path) -> None:
    fixture = transcript_fixture(tmp_path, pids=["p0001"])
    rows = [json.loads(line) for line in fixture["answers"].read_text().splitlines()]
    fixture["answers"].write_text(
        "".join(json.dumps(row) + "\n" for row in rows[:2]), encoding="utf-8")
    records = run_assembler(fixture)
    assert [t["n"] for t in records[0]["turns"]] == [1, 2]
    report = json.loads(Path(str(fixture["out"]) + ".report.json").read_text())
    assert report["incomplete_personas"] == ["p0001"]
    assert report["turns_expected_per_persona"] == 3


def test_a_forbidden_field_is_refused_even_if_it_gets_that_far() -> None:
    """The last line of defence: nothing named like planted truth is written."""
    record = {"pid": "p0001", "profile": {"pid": "p0001"},
              "turns": [{"n": 1, "item_id": "q001", "question": "?", "answer": 4}],
              "open": [{"prompt_id": "oe01", "question": "?", "answer": "x",
                        "wobble": 0.3}]}
    assert assemble_transcripts.banned_keys_in(record) == {"wobble"}
    assert assemble_transcripts.banned_keys_in({"pid": "p1"}) == set()
