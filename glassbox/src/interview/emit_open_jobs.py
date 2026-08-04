"""Write the batch file of open-ended interview prompts for a Leonardo job.

    python -m src.interview.emit_open_jobs \\
        --cards-dir <dir> --noise-dir <dir> --public-bank <bank_items.json> \\
        --splits <splits_v1.json> --out <run dir> \\
        [--round interview1] [--session-seed N] [--limit-personas N]

Output, inside the run directory:

    prompts_open.jsonl   one line per (persona, open prompt): {"pid",
                         "item_id", "round", "system", "user"}
    open_manifest.json   what was emitted and at which settings

Open-ended answers are the one part of a Stage 3 interview that IS a fresh
generation. The closed answers are drawn from the sweep's recorded material by
``src.interview.interview_answers`` because a recorded answer distribution
exists for every closed cell; no such thing exists for free text, so the
persona has to actually say it.

The prompts are the same shape as the closed ones (role lead, card, quirks,
speech style, wobble paragraph, then how to answer), with the open answer
instruction in place of the one-token one, and the same session-line
convention: the first line of the user message carries a tag built from the
round, so the interview sitting is a fresh context and not a cheap continuation
of a cached closed-item prefix.

NEVER RE-QUERY WHAT IS STORED. Any (pid, prompt_id, round) already in the run
directory's ``open_answers.jsonl`` is skipped, so re-running after a
half-finished job emits only the gap.

WALL RULE: the card directory and the noise directory arrive as command-line
arguments and no planted-truth path is written down here. The question text is
public -- the 20 open prompts are in the public bank's ``open_ended`` list. The
emitted prompts carry persona cards, so the run directory is a persona-side
artifact: keep it behind the Wall and never in git.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import FROZEN_SEED_BASE
from .emit_answer_jobs import load_personas, read_jsonl
from .responder_prompts import (
    INTERVIEW_ROUNDS,
    build_system_prompt,
    build_user_prompt,
    session_id,
)

PROMPTS_FILENAME = "prompts_open.jsonl"
ANSWERS_FILENAME = "open_answers.jsonl"
MANIFEST_FILENAME = "open_manifest.json"

#: The item type whose answer instruction the open prompts use.
OPEN_TYPE = "open"


def load_open_items(public_bank: str | Path) -> dict[str, dict[str, Any]]:
    """The public open-ended prompts, keyed by prompt id."""
    payload = json.loads(Path(public_bank).read_text(encoding="utf-8"))
    return {str(row["item_id"]): dict(row) for row in payload.get("open_ended", [])}


def interview_open_ids(splits_path: str | Path) -> list[str]:
    """The scripted open prompts, in the order the frozen splits file lists them."""
    payload = json.loads(Path(splits_path).read_text(encoding="utf-8"))
    ids = [str(item_id) for item_id in payload.get("interview_open") or []]
    if not ids:
        raise ValueError(
            f"{splits_path} has no interview_open list -- this is not the "
            "frozen splits file"
        )
    return ids


def answered_keys(run_dir: str | Path) -> set[tuple[str, str, str]]:
    """(pid, prompt_id, round) triples already answered in this run directory."""
    keys: set[tuple[str, str, str]] = set()
    for row in read_jsonl(Path(run_dir) / ANSWERS_FILENAME):
        prompt_id = row.get("prompt_id", row.get("item_id"))
        keys.add((str(row.get("pid")), str(prompt_id), str(row.get("round") or "")))
    return keys


def build_open_prompt(
    card: Mapping[str, Any],
    wobble: float,
    item: Mapping[str, Any],
    session: str,
) -> dict[str, str]:
    """The full prompt for one persona answering one open interview prompt."""
    return {
        "system": build_system_prompt(card, wobble, OPEN_TYPE),
        "user": build_user_prompt(item, session),
    }


def emit_jobs(
    people: Sequence[tuple[str, Mapping[str, Any], float]],
    items: Mapping[str, Mapping[str, Any]],
    prompt_ids: Sequence[str],
    run_dir: str | Path,
    round_name: str,
    session_seed: int,
) -> dict[str, Any]:
    """Write ``prompts_open.jsonl`` for everything not already answered."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    done = answered_keys(run_dir)

    missing = [prompt_id for prompt_id in prompt_ids if prompt_id not in items]
    if missing:
        raise ValueError(
            f"{len(missing)} scripted open prompt(s) are not in the public "
            f"bank, first: {missing[0]}"
        )

    lines: list[str] = []
    skipped = 0
    longest = 0

    for pid, card, wobble in people:
        session = session_id(pid, round_name, session_seed)
        for prompt_id in prompt_ids:
            if (pid, prompt_id, round_name) in done:
                skipped += 1
                continue
            prompt = build_open_prompt(card, wobble, items[prompt_id], session)
            longest = max(longest, len(prompt["system"]) + len(prompt["user"]))
            lines.append(
                json.dumps(
                    {
                        "pid": pid,
                        "item_id": prompt_id,
                        "round": round_name,
                        "system": prompt["system"],
                        "user": prompt["user"],
                    },
                    ensure_ascii=False,
                )
            )

    path = run_dir / PROMPTS_FILENAME
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return {
        "emitted": len(lines),
        "skipped": skipped,
        "already_answered": len(done),
        "n_personas": len(people),
        "n_prompts": len(prompt_ids),
        "prompt_ids": list(prompt_ids),
        "round": round_name,
        "session_seed": int(session_seed),
        "longest_prompt_chars": longest,
        "prompts_path": path,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.interview.emit_open_jobs",
        description="Build the batch of open-ended interview prompts.",
    )
    parser.add_argument(
        "--cards-dir",
        type=Path,
        required=True,
        help=(
            "directory of persona cards, one {pid}.json per persona. Planted "
            "truth: supplied per run, never hardcoded."
        ),
    )
    parser.add_argument(
        "--noise-dir",
        type=Path,
        required=True,
        help="directory of {pid}.json wobble records, supplied per run",
    )
    parser.add_argument(
        "--public-bank",
        type=Path,
        required=True,
        help="the public bank file: the open prompt ids and their public text",
    )
    parser.add_argument(
        "--splits",
        type=Path,
        required=True,
        help="the frozen splits file: interview_open, in scripted order",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="run directory; prompts_open.jsonl is written here",
    )
    parser.add_argument(
        "--round",
        choices=INTERVIEW_ROUNDS,
        default=INTERVIEW_ROUNDS[0],
        help="which interview sitting this is (default %(default)s)",
    )
    parser.add_argument(
        "--session-seed",
        type=int,
        default=FROZEN_SEED_BASE,
        help="seed for the session tags (default %(default)s)",
    )
    parser.add_argument(
        "--limit-personas",
        type=int,
        default=None,
        help="only the first N personas, by pid; for smoke tests",
    )
    args = parser.parse_args(argv)

    items = load_open_items(args.public_bank)
    if not items:
        parser.error(f"no open-ended prompts in {args.public_bank}")

    prompt_ids = interview_open_ids(args.splits)

    people = load_personas(args.cards_dir, args.noise_dir, args.limit_personas)
    if not people:
        parser.error(
            f"no persona has both a card in {args.cards_dir} and a wobble in "
            f"{args.noise_dir}"
        )

    summary = emit_jobs(
        people, items, prompt_ids, args.out, args.round, args.session_seed
    )

    manifest_path = Path(args.out) / MANIFEST_FILENAME
    manifest = {k: v for k, v in summary.items() if k != "prompts_path"}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(
        f"round {summary['round']}: emitted {summary['emitted']} open prompts, "
        f"skipped {summary['skipped']} already answered "
        f"({summary['n_personas']} personas x {summary['n_prompts']} prompts)"
    )
    print(f"  longest prompt: {summary['longest_prompt_chars']} characters")
    print(f"  prompts  -> {summary['prompts_path']}")
    print(f"  manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
