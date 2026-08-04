"""Turn raw responder completions into clean answers.

    python -m src.interview.parse_answers \\
        --completions <file.jsonl> --public-bank <bank_items.json> --out <run dir>

Input: one JSON object per line, ``{"pid", "item_id", "round", "completion"}``.
Output, in the run directory:

    answers.jsonl         {"pid", "item_id", "round", "answer", "raw"} where
                          ``answer`` is an int 1-5 for a Likert item and the
                          string "yes" or "no" for a binary one
    answer_rejects.jsonl  everything that could not be read, with a reason

Two rules the rest of the pipeline leans on:

  * **append, never duplicate.** A (pid, item_id, round) already in
    ``answers.jsonl`` is left alone, so re-parsing a completions file, or
    parsing a second chunk of the same job, adds only what is new.
  * **when in doubt, reject.** Whitespace, a trailing full stop, a comma after
    "Yes", a spelled-out "strongly agree" -- all fine. Two different answers in
    one completion, a digit for a yes/no item, or a bare "maybe" -- rejected,
    because a guessed answer is worse than a missing one. The rejects file is
    the re-ask queue: a row leaves it as soon as that cell has an answer.

The public bank is needed here because the item type decides what counts as a
valid answer: a digit for a binary item is not an answer, it is a confusion.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

ANSWERS_FILENAME = "answers.jsonl"
REJECTS_FILENAME = "answer_rejects.jsonl"

#: Spelled-out Likert answers. Longest phrase wins, so "strongly agree" is not
#: also read as "agree".
LIKERT_LABELS: dict[str, int] = {
    "strongly disagree": 1,
    "strongly agreed": 5,
    "strongly agree": 5,
    "neither agree nor disagree": 3,
    "neither agree or disagree": 3,
    "neutral": 3,
    "disagree": 2,
    "agree": 4,
}

YES_WORDS: tuple[str, ...] = ("yes", "yeah", "yep", "yup", "true", "correct")
NO_WORDS: tuple[str, ...] = ("no", "nope", "nah", "false", "incorrect")

#: A single 1-5 that is not part of a longer number. "5." is a five; "5.5" and
#: "15" are not answers.
_DIGIT = re.compile(r"(?<![\d.])([1-5])(?!\d)(?!\.\d)")


def _clean(raw: Any) -> str:
    """Lowercase the completion and collapse its whitespace."""
    return re.sub(r"\s+", " ", str(raw or "")).strip().lower()


def _bare(text: str) -> str:
    """The completion with punctuation and quotes stripped off both ends."""
    return text.strip(" \t\r\n.,;:!?'\"`*-_()[]")


def likert_candidates(text: str) -> set[int]:
    """Every distinct Likert value this completion could be read as."""
    values: set[int] = set()
    working = f" {text} "

    for phrase in sorted(LIKERT_LABELS, key=len, reverse=True):
        pattern = re.compile(rf"(?<![a-z]){re.escape(phrase)}(?![a-z])")
        while True:
            match = pattern.search(working)
            if match is None:
                break
            values.add(LIKERT_LABELS[phrase])
            # Blank out the span so a longer phrase is not counted twice by the
            # shorter phrase inside it.
            working = working[: match.start()] + " " * (match.end() - match.start()) + working[match.end():]

    for match in _DIGIT.finditer(working):
        values.add(int(match.group(1)))
    return values


def binary_candidates(text: str) -> set[str]:
    """Every distinct yes/no answer this completion could be read as."""
    values: set[str] = set()
    bare = _bare(text)
    if bare == "y":
        return {"yes"}
    if bare == "n":
        return {"no"}

    for word in YES_WORDS:
        if re.search(rf"(?<![a-z]){word}(?![a-z])", text):
            values.add("yes")
            break
    for word in NO_WORDS:
        if re.search(rf"(?<![a-z]){word}(?![a-z])", text):
            values.add("no")
            break
    return values


def normalize_answer(raw: Any, item_type: str) -> tuple[int | str | None, str]:
    """Read one completion as an answer of the given type.

    Returns ``(answer, reason)``. ``answer`` is ``None`` when the completion
    cannot be read, and ``reason`` says why -- empty when it can.
    """
    text = _clean(raw)
    if not text:
        return None, "empty completion"

    if item_type == "likert5":
        values = likert_candidates(text)
        if not values:
            return None, "no 1-5 answer in the completion"
        if len(values) > 1:
            return None, f"ambiguous: reads as {sorted(values)}"
        return values.pop(), ""

    if item_type == "binary":
        values = binary_candidates(text)
        if not values:
            return None, "no yes/no answer in the completion"
        if len(values) > 1:
            return None, "ambiguous: reads as both yes and no"
        return values.pop(), ""

    return None, f"unknown item type {item_type!r}"


# --------------------------------------------------------------------------
# files
# --------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("pid")), str(row.get("item_id")), str(row.get("round") or "main"))


def _append_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows)
    if body:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(body)


def item_types(public_bank: str | Path) -> dict[str, str]:
    """Item id -> item type, from the public bank."""
    payload = json.loads(Path(public_bank).read_text(encoding="utf-8"))
    return {str(row["item_id"]): str(row["type"]) for row in payload.get("items", [])}


def parse_file(
    completions_path: str | Path,
    public_bank: str | Path,
    run_dir: str | Path,
) -> dict[str, Any]:
    """Parse one completions file into the run directory. Returns a summary."""
    completions_path = Path(completions_path)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    types = item_types(public_bank)
    answers_path = run_dir / ANSWERS_FILENAME
    known = {_key(row) for row in _read_jsonl(answers_path)}

    kept: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    duplicates = 0
    seen: set[tuple[str, str, str]] = set()

    for lineno, line in enumerate(completions_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            rejects.append({"pid": None, "item_id": None, "round": None, "raw": line[:200],
                            "reason": f"line {lineno} is not JSON: {exc}"})
            continue

        key = _key(record)
        pid, item_id, round_name = key
        if key in known or key in seen:
            duplicates += 1
            continue

        row = {"pid": pid, "item_id": item_id, "round": round_name,
               "raw": str(record.get("completion", ""))}

        item_type = types.get(item_id)
        if item_type is None:
            rejects.append({**row, "reason": "item id is not in the public bank"})
            continue

        answer, reason = normalize_answer(record.get("completion", ""), item_type)
        if answer is None:
            rejects.append({**row, "reason": reason})
            continue

        seen.add(key)
        kept.append({"pid": pid, "item_id": item_id, "round": round_name,
                     "answer": answer, "raw": row["raw"]})

    _append_jsonl(answers_path, kept)

    # The rejects file is the re-ask queue, so it holds only cells that still
    # have no answer: old rows that have since been answered are dropped.
    answered = known | seen
    rejects_path = run_dir / REJECTS_FILENAME
    outstanding: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in _read_jsonl(rejects_path) + rejects:
        key = _key(row)
        if key not in answered:
            outstanding[key] = dict(row)
    rejects_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in outstanding.values()),
        encoding="utf-8",
    )

    return {
        "kept": len(kept),
        "rejected": len(rejects),
        "duplicates": duplicates,
        "outstanding": len(outstanding),
        "answers_path": answers_path,
        "rejects_path": rejects_path,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.interview.parse_answers",
        description="Normalize responder completions into answers.jsonl.",
    )
    parser.add_argument(
        "--completions",
        type=Path,
        required=True,
        help="JSONL of {'pid', 'item_id', 'round', 'completion'} lines from the job",
    )
    parser.add_argument(
        "--public-bank",
        type=Path,
        required=True,
        help="the public bank file: needed for each item's type",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="run directory; answers.jsonl is appended to and answer_rejects.jsonl rewritten",
    )
    args = parser.parse_args(argv)

    summary = parse_file(args.completions, args.public_bank, args.out)
    print(
        f"parsed {summary['kept']} new answers, rejected {summary['rejected']}, "
        f"skipped {summary['duplicates']} already on file"
    )
    print(f"  answers -> {summary['answers_path']}")
    if summary["outstanding"]:
        print(f"  {summary['outstanding']} cells still unanswered -> {summary['rejects_path']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
