"""Turn pass-1 card completions into the pass-2 self-check batch.

    python -m src.personas.build_selfcheck_batch \\
        --pass1 <card_completions_v2_pass1.jsonl> \\
        --truth-dir <planted-truth dir> \\
        --out <selfcheck_prompts.jsonl> \\
        --retry-out <selfcheck_retry.jsonl>

Input is what the Leonardo chat driver wrote: one ``{"pid", "completion"}`` per
line. For every line that parsed into a card with a biography, this emits one
self-check prompt built from that draft and from the persona's own eight
guidance lines, rebuilt from the planted theta.

Lines that did not parse, or came back with no biography, are not prompts --
they are re-renders. They go to the retry file as ``{"pid", "reason"}`` and the
orchestrator sends that list back through pass 1.

WHY EVERY RECORD CARRIES ``"round": "selfcheck"``. The driver derives each
record's sampling seed from ``(seed-base, pid, item_id, round, temperature)``.
Without the round field the pass-2 record for a persona would have the same
identity as its pass-1 record, so the two would be handed the same random
numbers -- common random numbers, the exact bug that inflated the Stage 1 pilot
test-retest to 0.98. The field is one word and it makes pass 2 an independent
draw.

WALL RULE: the planted-truth directory arrives as ``--truth-dir`` and has no
default. Nothing here writes it down.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .card_prompts import build_selfcheck_prompt, trait_guidance
from .ingest import normalise_card, parse_completion

#: The value written into every emitted record's ``round`` field.
ROUND = "selfcheck"


def _read_json(path: Path) -> Any:
    """Parsed JSON, or ``None`` when the file is missing or unreadable."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_theta(truth_dir: Path, pid: str) -> dict[str, float] | None:
    """This persona's planted trait vector, or ``None`` if it is not there."""
    record = _read_json(Path(truth_dir) / "theta" / f"{pid}.json")
    if not isinstance(record, dict):
        return None
    theta = record.get("theta")
    return theta if isinstance(theta, dict) else None


def load_profile(truth_dir: Path, pid: str) -> dict[str, Any] | None:
    """The demographics stored with this persona's card prompt.

    Same source ``ingest`` uses, for the same reason: the fixed facts must come
    from inside the planted-truth tree, not from the public profiles.
    """
    record = _read_json(Path(truth_dir) / "card_prompts" / f"{pid}.json")
    if not isinstance(record, dict):
        return None
    profile = record.get("profile")
    return profile if isinstance(profile, dict) else None


def build_record(pid: str, completion: str, truth_dir: Path) -> dict[str, str]:
    """One self-check prompt record. Raises ``ValueError`` if it cannot be built.

    Every failure here means "render this persona's card again", so they all
    raise the same type and the caller routes them to the retry file with the
    message as the reason.
    """
    profile = load_profile(truth_dir, pid)
    if profile is None:
        raise ValueError("no prompt record for this pid in this batch")

    theta = load_theta(truth_dir, pid)
    if theta is None:
        raise ValueError("no planted theta for this pid in this batch")

    card = normalise_card(pid, parse_completion(completion), profile)
    if not card["biography"]:
        raise ValueError("no biography in the parsed JSON")

    try:
        lines = trait_guidance(theta)
    except KeyError as exc:
        raise ValueError(f"planted theta is missing dimension {exc}") from exc

    prompt = build_selfcheck_prompt(lines, card)
    return {
        "pid": pid,
        "round": ROUND,
        "system": prompt["system"],
        "user": prompt["user"],
    }


def build_batch(pass1_path: Path, truth_dir: Path) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """Read the pass-1 file. Returns ``(prompt records, retry rows)``.

    A pid that appears twice is built once: the driver would skip the duplicate
    anyway (same passthrough signature), and a duplicated prompt line is only a
    way to make the line counts lie.
    """
    prompts: list[dict[str, str]] = []
    retries: list[dict[str, Any]] = []
    seen: set[str] = set()

    lines = Path(pass1_path).read_text(encoding="utf-8").splitlines()
    for lineno, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            retries.append({"pid": None, "reason": f"line {lineno} is not JSON: {exc}"})
            continue

        pid = record.get("pid")
        if not pid:
            retries.append({"pid": None, "reason": f"line {lineno} has no pid"})
            continue
        pid = str(pid)
        if pid in seen:
            continue
        seen.add(pid)

        try:
            prompts.append(build_record(pid, record.get("completion", ""), truth_dir))
        except ValueError as exc:
            retries.append({"pid": pid, "reason": str(exc)})

    return prompts, retries


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Rewrite a file for this run; an empty run leaves an empty file behind."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(body, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.personas.build_selfcheck_batch",
        description="Build the pass-2 self-check prompts from pass-1 card completions.",
    )
    parser.add_argument(
        "--pass1",
        type=Path,
        required=True,
        help="JSONL of {'pid', 'completion'} lines from the pass-1 card render",
    )
    parser.add_argument(
        "--truth-dir",
        type=Path,
        required=True,
        help=(
            "the batch's planted-truth directory: theta/ and card_prompts/ are "
            "read from it. Supplied per run, never hardcoded."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="JSONL of {'pid', 'round', 'system', 'user'} for the pass-2 render",
    )
    parser.add_argument(
        "--retry-out",
        type=Path,
        required=True,
        help="JSONL of {'pid', 'reason'} for the personas that need pass 1 again",
    )
    args = parser.parse_args(argv)

    prompts, retries = build_batch(args.pass1, args.truth_dir)
    _write_jsonl(args.out, prompts)
    _write_jsonl(args.retry_out, retries)

    print(f"built {len(prompts)} self-check prompts -> {args.out}")
    print(f"{len(retries)} persona(s) need pass 1 again -> {args.retry_out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
