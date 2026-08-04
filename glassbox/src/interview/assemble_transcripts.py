"""Assemble the Stage 3 transcript the SYSTEM side is allowed to see.

    python -m src.interview.assemble_transcripts \\
        --profiles-dir <data/public/profiles> --public-bank <bank_items.json> \\
        --splits <splits_v1.json> --answers <answers_interview1.jsonl> \\
        [--open-answers <open_answers.jsonl>] \\
        --out <data/runs/interview_v1/transcripts.jsonl> [--report <file.json>]

One line per persona:

    {"pid": ...,
     "profile": {public profile},
     "turns": [{"n": 1..15, "item_id", "question", "answer"}, ...],
     "open":  [{"prompt_id", "question", "answer"}, ...]}

This file is the interviewer's, the person encoder's and the predictor's whole
view of a persona, so it is the single most dangerous artifact in the project.
Three things keep it honest:

  1. **It is built, never copied.** Every field is written out by name from an
     allowed source. Nothing is passed through from an input record, so a new
     field appearing upstream cannot ride along into a transcript.
  2. **Allowlists, checked here and again in ``tests/test_wall.py``.** A profile
     key that is not on the list stops the run rather than being quietly
     dropped: an unexpected key in a public profile means something changed
     that a person should look at.
  3. **The open answers are leak-checked at the crossing.** Free text is the
     one thing a persona generates in its own words while holding a card that
     states its traits as life details, so every open answer goes through the
     same vocabulary check that rejects a leaking persona card
     (``src.personas.ingest.leak_reasons``). A failing answer is reported and
     dropped -- never stored, not even quoted in the report.

Re-runnable. Run it now with only ``--answers`` for the closed part, and again
with ``--open-answers`` when the open-ended job comes back; the second run
rewrites the file with the open sections filled in.

WALL RULE: every path is a command-line argument. The output belongs on the
system side (``data/runs/<run>/transcripts.jsonl``), which is exactly why
everything above exists.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..personas.ingest import leak_reasons

#: Keys a public profile may carry into a transcript. Anything else stops the
#: run: the profile file is the system side's demographic view and nothing that
#: is not on this list has been cleared to be in it.
PROFILE_KEYS: tuple[str, ...] = (
    "pid",
    "age",
    "occupation",
    "city_size",
    "household",
    "region_type",
)

#: Field names that must never appear anywhere in a transcript, at any depth.
#: Belt and braces on top of building every record by hand.
BANNED_KEYS: frozenset[str] = frozenset(
    {
        "theta",
        "thetas",
        "wobble",
        "noise",
        "biography",
        "quirks",
        "speech_style",
        "card",
        "raw",
        "loadings",
        "target_dims",
        "strength_class",
        "negatively_keyed",
        "system",
        "prompt",
        "completion",
    }
)


def load_profiles(profiles_dir: str | Path) -> dict[str, dict[str, Any]]:
    """The public profiles, keyed by pid, with the allowlist enforced."""
    profiles: dict[str, dict[str, Any]] = {}
    for path in sorted(Path(profiles_dir).glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        extra = sorted(set(record) - set(PROFILE_KEYS))
        if extra:
            raise ValueError(
                f"{path} carries {extra}, which is not on the public-profile "
                "allowlist -- refusing to put it in a transcript"
            )
        profile = {key: record[key] for key in PROFILE_KEYS if key in record}
        profile["pid"] = str(record.get("pid", path.stem))
        profiles[profile["pid"]] = profile
    return profiles


def load_question_text(public_bank: str | Path) -> tuple[dict[str, str], dict[str, str]]:
    """(closed item text, open prompt text), both keyed by id, from the public bank."""
    payload = json.loads(Path(public_bank).read_text(encoding="utf-8"))
    closed = {str(row["item_id"]): str(row["text"]) for row in payload.get("items", [])}
    open_text = {
        str(row["item_id"]): str(row["text"]) for row in payload.get("open_ended", [])
    }
    return closed, open_text


def load_scripted(splits_path: str | Path) -> tuple[list[str], list[str]]:
    """The frozen interview: closed items in scripted order, then open prompts."""
    payload = json.loads(Path(splits_path).read_text(encoding="utf-8"))
    closed = [str(item_id) for item_id in payload.get("interview_closed") or []]
    open_ids = [str(item_id) for item_id in payload.get("interview_open") or []]
    if not closed:
        raise ValueError(
            f"{splits_path} has no interview_closed list -- this is not the "
            "frozen splits file"
        )
    return closed, open_ids


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        return []
    return [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def banned_keys_in(value: Any) -> set[str]:
    """Every forbidden field name anywhere inside a built record."""
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, inner in value.items():
            if str(key).lower() in BANNED_KEYS:
                found.add(str(key))
            found |= banned_keys_in(inner)
    elif isinstance(value, (list, tuple)):
        for inner in value:
            found |= banned_keys_in(inner)
    return found


def assemble(
    profiles: Mapping[str, Mapping[str, Any]],
    closed_text: Mapping[str, str],
    open_text: Mapping[str, str],
    scripted_closed: Sequence[str],
    scripted_open: Sequence[str],
    answers: Sequence[Mapping[str, Any]],
    open_answers: Sequence[Mapping[str, Any]] = (),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build one transcript per persona. Returns the records and a report."""
    missing_text = [i for i in scripted_closed if i not in closed_text]
    if missing_text:
        raise ValueError(
            f"{len(missing_text)} scripted item(s) have no public text, first: "
            f"{missing_text[0]}"
        )

    by_persona: dict[str, dict[str, Any]] = {}
    for row in answers:
        pid = str(row.get("pid"))
        by_persona.setdefault(pid, {})[str(row.get("item_id"))] = row.get("answer")

    open_by_persona: dict[str, dict[str, str]] = {}
    for row in open_answers:
        pid = str(row.get("pid"))
        prompt_id = str(row.get("prompt_id", row.get("item_id")))
        open_by_persona.setdefault(pid, {})[prompt_id] = str(row.get("answer", ""))

    records: list[dict[str, Any]] = []
    incomplete: list[str] = []
    no_profile: list[str] = []
    open_rejects: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    open_kept = 0
    total_turns = 0

    for pid in sorted(by_persona):
        profile = profiles.get(pid)
        if profile is None:
            no_profile.append(pid)
            continue

        given = by_persona[pid]
        turns: list[dict[str, Any]] = []
        for item_id in scripted_closed:
            if item_id not in given:
                continue
            turns.append(
                {
                    "n": len(turns) + 1,
                    "item_id": item_id,
                    "question": closed_text[item_id],
                    "answer": given[item_id],
                }
            )
        if len(turns) < len(scripted_closed):
            incomplete.append(pid)
        total_turns += len(turns)

        said = open_by_persona.get(pid, {})
        open_block: list[dict[str, Any]] = []
        for prompt_id in scripted_open:
            answer = said.get(prompt_id)
            if not answer:
                continue
            reasons = leak_reasons(answer)
            if reasons:
                open_rejects.append(
                    {"pid": pid, "prompt_id": prompt_id, "reasons": reasons}
                )
                reason_counts.update(reasons)
                continue
            open_kept += 1
            open_block.append(
                {
                    "prompt_id": prompt_id,
                    "question": open_text.get(prompt_id, ""),
                    "answer": answer,
                }
            )

        record = {
            "pid": pid,
            "profile": dict(profile),
            "turns": turns,
            "open": open_block,
        }
        offending = banned_keys_in(record)
        if offending:
            raise ValueError(
                f"transcript for {pid} carries forbidden field(s) "
                f"{sorted(offending)} -- refusing to write it"
            )
        records.append(record)

    report = {
        "n_personas": len(records),
        "turns_expected_per_persona": len(scripted_closed),
        "n_turns": total_turns,
        "incomplete_personas": incomplete,
        "personas_without_a_public_profile": no_profile,
        "open_prompts_scripted": list(scripted_open),
        "open_answers_seen": sum(len(v) for v in open_by_persona.values()),
        "open_answers_kept": open_kept,
        "open_answers_rejected_for_leaks": len(open_rejects),
        "open_leak_rejects": open_rejects,
        "open_leak_reason_counts": dict(reason_counts),
    }
    return records, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.interview.assemble_transcripts",
        description="Build the system-side Stage 3 transcripts.",
    )
    parser.add_argument(
        "--profiles-dir",
        type=Path,
        required=True,
        help="directory of public {pid}.json profiles",
    )
    parser.add_argument(
        "--public-bank",
        type=Path,
        required=True,
        help="the public bank file: item text and open-prompt text",
    )
    parser.add_argument(
        "--splits",
        type=Path,
        required=True,
        help="the frozen splits file: the scripted interview, in order",
    )
    parser.add_argument(
        "--answers",
        type=Path,
        required=True,
        help="the interview's closed answers, from src.interview.interview_answers",
    )
    parser.add_argument(
        "--open-answers",
        type=Path,
        default=None,
        help=(
            "the parsed open-ended answers, when the job has come back. Omit "
            "for the closed-only pass; re-run with it to fill the open part in."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="where to write transcripts.jsonl (system side)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="where to write the assembly report (default: <out>.report.json)",
    )
    args = parser.parse_args(argv)

    profiles = load_profiles(args.profiles_dir)
    if not profiles:
        parser.error(f"no public profiles in {args.profiles_dir}")

    closed_text, open_text = load_question_text(args.public_bank)
    scripted_closed, scripted_open = load_scripted(args.splits)

    answers = read_jsonl(args.answers)
    if not answers:
        parser.error(f"no interview answers in {args.answers}")

    open_answers = read_jsonl(args.open_answers) if args.open_answers else []

    records, report = assemble(
        profiles,
        closed_text,
        open_text,
        scripted_closed,
        scripted_open,
        answers,
        open_answers,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )

    report_path = args.report or Path(str(out) + ".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(
        f"assembled {report['n_personas']} transcripts, {report['n_turns']} turns "
        f"({report['turns_expected_per_persona']} expected per persona)"
    )
    if report["incomplete_personas"]:
        print(f"  INCOMPLETE: {len(report['incomplete_personas'])} personas short of a full script")
    if report["personas_without_a_public_profile"]:
        print(f"  SKIPPED: {len(report['personas_without_a_public_profile'])} personas have no public profile")
    print(
        f"  open answers: {report['open_answers_kept']} kept, "
        f"{report['open_answers_rejected_for_leaks']} rejected by the leak check"
    )
    print(f"  transcripts -> {out}")
    print(f"  report      -> {report_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
