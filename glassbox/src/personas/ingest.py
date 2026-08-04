"""Read Gemma's card completions back in, check them, store the cards.

    python -m src.personas.ingest --completions <file.jsonl> --out-truth <dir>

Input: one JSON object per line, ``{"pid": ..., "completion": "<raw model text>"}``.
The raw text is whatever the model produced -- a clean JSON object, a fenced
block, or JSON with chat around it. All three are handled.

A card is **rejected** (written to ``card_rejects.jsonl``, no card file) when:
  * the completion has no readable JSON object, or no biography in it;
  * there is no prompt record for that pid in this batch;
  * the biography leaks -- it names a trait dimension or carries a number that
    looks like a trait value. A leaked card poisons the study, so this is the
    one content rule that rejects rather than warns.

A card is **kept with warnings** (``card_warnings.jsonl``) when the biography is
outside the 120-450 word band or does not visibly use the demographics it was
given. Both are quality signals, not leaks, and string matching on demographics
is fuzzy enough that rejecting on it would throw away good cards. The
orchestrator reads the counts and decides what to regenerate.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from .sampler import DIMENSIONS, DIMENSION_NAMES

WORD_COUNT_BAND = (120, 450)

#: Extra spellings of the dimension names that mean the same thing.
_EXTRA_TRAIT_PHRASES = (
    "institutional trust",
    "trust in institutions",
    "risk aversion",
    "risk tolerance",
    "price sensitive",
    "tech optimist",
    "local attachment",
    "trait value",
)

#: The dimension codes, matched case-sensitively so ordinary words are safe.
_CODE_PATTERN = re.compile(r"\b(" + "|".join(DIMENSIONS) + r")\b")

#: A signed number, or a bare decimal, small enough to be a trait value.
_SIGNED_NUMBER = re.compile(r"(?<![\w.])[+\-−–]\s?([0-2](?:\.\d+)?)(?![\d.])")
_BARE_DECIMAL = re.compile(r"(?<![\w.\-−–])([0-2]\.\d+)(?![\d.])")

#: Words that make a small decimal a measurement rather than a score.
_UNIT_WORDS = {
    "km", "kilometre", "kilometres", "kilometer", "kilometers", "mile", "miles",
    "m", "metre", "metres", "meter", "meters", "cm", "mm",
    "kg", "kilo", "kilos", "g", "gram", "grams", "tonne", "tonnes", "ton", "tons",
    "l", "litre", "litres", "liter", "liters", "ml",
    "second", "seconds", "minute", "minutes", "min", "hour", "hours", "hr", "hrs",
    "day", "days", "week", "weeks", "month", "months", "year", "years", "yr", "yrs",
    "percent", "%", "degree", "degrees", "celsius",
    "million", "billion", "thousand", "hundred",
    "euro", "euros", "dollar", "dollars", "pound", "pounds", "krona", "kronor",
    "times", "per", "acre", "acres", "hectare", "hectares", "points", "stars",
}

_CURRENCY_BEFORE = set("€$£")


def parse_completion(raw: str) -> dict[str, Any]:
    """Pull the JSON object out of a raw model completion.

    Tolerates code fences and prose before or after the object. Raises
    ``ValueError`` when nothing parseable is in there.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("empty completion")

    text = _strip_fences(raw)

    for start in (index for index, char in enumerate(text) if char == "{"):
        end = _matching_brace(text, start)
        if end is None:
            continue
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    raise ValueError("no JSON object found in the completion")


def _strip_fences(text: str) -> str:
    """Drop ``` fences, keeping what is inside the first fenced block."""
    fenced = re.search(r"```[a-zA-Z]*\s*\n(.*?)```", text, flags=re.DOTALL)
    if fenced:
        return fenced.group(1)
    return text.replace("```json", " ").replace("```", " ")


def _matching_brace(text: str, start: int) -> int | None:
    """Index of the ``}`` closing the ``{`` at ``start``, honouring strings."""
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    return None


def _normalise(text: str) -> str:
    """Lowercase, punctuation to spaces, whitespace collapsed."""
    return re.sub(r"[^a-z0-9%]+", " ", text.lower()).strip()


def leak_reasons(biography: str) -> list[str]:
    """Every reason this biography leaks planted truth. Empty means clean."""
    reasons: list[str] = []
    flat = _normalise(biography)
    padded = f" {flat} "

    for code, name in DIMENSION_NAMES.items():
        if f" {name} " in padded:
            reasons.append(f"names the {code} dimension ({name!r})")
    for phrase in _EXTRA_TRAIT_PHRASES:
        if f" {phrase} " in padded:
            reasons.append(f"uses trait wording {phrase!r}")
    for code in sorted(set(_CODE_PATTERN.findall(biography))):
        reasons.append(f"contains the dimension code {code!r}")
    reasons.extend(_numeric_leaks(biography))
    return reasons


def _numeric_leaks(text: str) -> list[str]:
    """Numbers that look like trait values. Ages, counts and measurements pass."""
    hits: list[str] = []
    for pattern in (_SIGNED_NUMBER, _BARE_DECIMAL):
        for match in pattern.finditer(text):
            token = match.group(0).strip()
            if _looks_like_measurement(text, match.start(), match.end()):
                continue
            hits.append(f"contains {token!r}, which reads as a trait value")
    return hits


def _looks_like_measurement(text: str, start: int, end: int) -> bool:
    """True when a unit or a currency symbol sits next to the number."""
    before = text[:start].rstrip()
    if before and before[-1] in _CURRENCY_BEFORE:
        return True
    following = _normalise(text[end : end + 24]).split()
    return bool(following) and following[0] in _UNIT_WORDS


def card_warnings(card: dict[str, Any], profile: dict[str, Any]) -> list[str]:
    """Soft-quality complaints: length band and demographics actually used."""
    warnings: list[str] = []

    words = len(str(card.get("biography", "")).split())
    low, high = WORD_COUNT_BAND
    if not low <= words <= high:
        warnings.append(f"biography is {words} words, outside the {low}-{high} band")

    if not card.get("quirks"):
        warnings.append("no quirks in the completion")
    if not card.get("speech_style"):
        warnings.append("no speech_style in the completion")

    flat = _normalise(str(card.get("biography", "")))
    age = profile.get("age")
    if age is not None and str(age) not in re.findall(r"\d+", flat):
        warnings.append(f"age {age} is not written as a number in the biography")

    for field in ("occupation", "city_size", "household", "region_type"):
        value = profile.get(field)
        if value and not _mentions(flat, str(value)):
            warnings.append(f"{field} {value!r} is not visible in the biography")

    return warnings


def _mentions(flat_text: str, value: str) -> bool:
    """Loose check that a demographic value made it into the text.

    Matches on the stem of each content word, so "driver" is found in "drives"
    and "teacher" in "teaches". Deliberately generous: this drives a warning,
    not a rejection.
    """
    words = [word for word in _normalise(value).split() if len(word) >= 4]
    if not words:
        return True
    return all((word[:5] if len(word) > 5 else word) in flat_text for word in words)


def normalise_card(pid: str, parsed: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    """Shape one parsed completion into the card record we store."""
    quirks = parsed.get("quirks") or []
    if isinstance(quirks, str):
        quirks = [quirks]
    card = {
        "pid": pid,
        "biography": str(parsed.get("biography", "")).strip(),
        "quirks": [str(item).strip() for item in quirks],
        "speech_style": str(parsed.get("speech_style", "")).strip(),
    }
    for field in ("age", "occupation", "city_size", "household", "region_type"):
        if field in profile:
            card[field] = profile[field]
    return card


def ingest_file(completions_path: Path, out_truth: Path) -> dict[str, Any]:
    """Ingest one completions file. Returns a summary of what happened."""
    completions_path = Path(completions_path)
    out_truth = Path(out_truth)
    cards_dir = out_truth / "cards"
    cards_dir.mkdir(parents=True, exist_ok=True)

    kept: list[str] = []
    rejects: list[dict[str, Any]] = []
    warned: list[dict[str, Any]] = []

    for lineno, line in enumerate(completions_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            rejects.append({"pid": None, "line": lineno, "reason": f"line is not JSON: {exc}"})
            continue

        pid = record.get("pid")
        if not pid:
            rejects.append({"pid": None, "line": lineno, "reason": "line has no pid"})
            continue

        profile = _load_profile(out_truth, str(pid))
        if profile is None:
            rejects.append({"pid": pid, "reason": "no prompt record for this pid in this batch"})
            continue

        try:
            parsed = parse_completion(record.get("completion", ""))
        except ValueError as exc:
            rejects.append({"pid": pid, "reason": str(exc)})
            continue

        card = normalise_card(str(pid), parsed, profile)
        if not card["biography"]:
            rejects.append({"pid": pid, "reason": "no biography in the parsed JSON"})
            continue

        leaks = leak_reasons(card["biography"])
        if leaks:
            rejects.append({"pid": pid, "reason": "leak: " + "; ".join(leaks)})
            continue

        warnings = card_warnings(card, profile)
        if warnings:
            warned.append({"pid": pid, "warnings": warnings})

        (cards_dir / f"{pid}.json").write_text(
            json.dumps(card, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        kept.append(str(pid))

    _write_jsonl(out_truth / "card_rejects.jsonl", rejects)
    _write_jsonl(out_truth / "card_warnings.jsonl", warned)

    return {
        "kept": len(kept),
        "rejected": len(rejects),
        "with_warnings": len(warned),
        "kept_pids": kept,
        "rejects": rejects,
        "warnings": warned,
    }


def _load_profile(out_truth: Path, pid: str) -> dict[str, Any] | None:
    """The demographics recorded with this persona's card prompt."""
    path = out_truth / "card_prompts" / f"{pid}.json"
    if not path.is_file():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    return record.get("profile") or {}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Rewrite a report file for this run; an empty run leaves an empty file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(body, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.personas.ingest",
        description="Parse Gemma card completions into stored persona cards.",
    )
    parser.add_argument(
        "--completions",
        type=Path,
        required=True,
        help="JSONL file of {'pid', 'completion'} lines from the card-writing job",
    )
    parser.add_argument(
        "--out-truth",
        type=Path,
        required=True,
        help=(
            "the batch's planted-truth directory: cards are written into its "
            "cards/ subdirectory. Supplied per run, never hardcoded."
        ),
    )
    args = parser.parse_args(argv)

    summary = ingest_file(args.completions, args.out_truth)
    print(
        f"kept {summary['kept']} cards, rejected {summary['rejected']}, "
        f"{summary['with_warnings']} kept with warnings"
    )
    if summary["rejected"]:
        print(f"  rejects listed in {args.out_truth / 'card_rejects.jsonl'}")
    if summary["with_warnings"]:
        print(f"  warnings listed in {args.out_truth / 'card_warnings.jsonl'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
