"""Turn raw open-ended completions into clean interview answers.

    python -m src.interview.parse_open_answers \\
        --completions <file.jsonl> --out <run dir>

Input: one JSON object per line, ``{"pid", "item_id", "round", "completion"}``
-- the job's output, with the prompt keys passed through. Output, in the run
directory:

    open_answers.jsonl         {"pid", "prompt_id", "round", "answer"}
    open_answer_rejects.jsonl  everything unusable, with a reason

There is almost nothing to normalise in free text, so this does four things
only: strip whitespace and stray wrapping quotes, drop a session line if the
model echoed one back, refuse an answer that is empty or too short to carry
anything, and refuse a duplicate. Everything else the persona said is kept
verbatim, including bad grammar and changes of mind -- that is the data.

Same two rules as the closed-form parser: **append, never duplicate**, and
**when in doubt, reject**. The rejects file is the re-ask queue.

Note this does NOT check the answer for leaked planted truth. That check runs
where the text crosses the Wall, in ``src.interview.assemble_transcripts``, so
that nothing can reach a transcript without passing it.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

ANSWERS_FILENAME = "open_answers.jsonl"
REJECTS_FILENAME = "open_answer_rejects.jsonl"

#: Shorter than this and there is no answer in there, only an acknowledgement.
MIN_WORDS = 5

_SESSION_LINE = re.compile(r"^\s*session\s*:.*$", re.IGNORECASE | re.MULTILINE)


def clean_answer(raw: Any) -> tuple[str, str]:
    """Tidy one completion. Returns ``(answer, reason)``; reason empty is good."""
    text = str(raw or "")
    text = _SESSION_LINE.sub("", text)
    # Collapse runs of blank lines, trim every line, drop wrapping quotes.
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()

    if not text:
        return "", "empty completion"
    if len(text.split()) < MIN_WORDS:
        return "", f"only {len(text.split())} words, under the {MIN_WORDS} minimum"
    return text, ""


# --------------------------------------------------------------------------
# files
# --------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    prompt_id = row.get("prompt_id", row.get("item_id"))
    return (str(row.get("pid")), str(prompt_id), str(row.get("round") or ""))


def _append_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows)
    if body:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(body)


def parse_file(completions_path: str | Path, run_dir: str | Path) -> dict[str, Any]:
    """Parse one open-ended completions file into the run directory."""
    completions_path = Path(completions_path)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    answers_path = run_dir / ANSWERS_FILENAME
    known = {_key(row) for row in _read_jsonl(answers_path)}

    kept: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    duplicates = 0
    seen: set[tuple[str, str, str]] = set()

    for lineno, line in enumerate(
        completions_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            rejects.append(
                {
                    "pid": None,
                    "prompt_id": None,
                    "round": None,
                    "reason": f"line {lineno} is not JSON: {exc}",
                }
            )
            continue

        key = _key(record)
        pid, prompt_id, round_name = key
        if key in known or key in seen:
            duplicates += 1
            continue

        answer, reason = clean_answer(record.get("completion", ""))
        row = {"pid": pid, "prompt_id": prompt_id, "round": round_name}
        if not answer:
            rejects.append({**row, "reason": reason})
            continue

        seen.add(key)
        kept.append({**row, "answer": answer})

    _append_jsonl(answers_path, kept)

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
        prog="python -m src.interview.parse_open_answers",
        description="Normalize open-ended completions into open_answers.jsonl.",
    )
    parser.add_argument(
        "--completions",
        type=Path,
        required=True,
        help="JSONL of {'pid', 'item_id', 'round', 'completion'} lines from the job",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="run directory; open_answers.jsonl is appended to",
    )
    args = parser.parse_args(argv)

    summary = parse_file(args.completions, args.out)
    print(
        f"parsed {summary['kept']} new open answers, rejected "
        f"{summary['rejected']}, skipped {summary['duplicates']} already on file"
    )
    print(f"  answers -> {summary['answers_path']}")
    if summary["outstanding"]:
        print(f"  {summary['outstanding']} still unanswered -> {summary['rejects_path']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
