"""Gate 4 baseline 4: the system LLM, public profile only, no interview.

    # 1. write the prompt batch for the cluster driver
    python -m src.model.llm_profile_baseline emit \\
        --out data/runs/stage4_qwen_baseline/baseline_in.jsonl

    # 2. (Leonardo runs jobs/render_chat_batch.py over that file)

    # 3. turn the completions into one predicted distribution per cell
    python -m src.model.llm_profile_baseline parse \\
        --completions data/runs/stage4_qwen_baseline/baseline_out.jsonl \\
        --out data/runs/stage4_qwen_baseline/predictions.jsonl

What this is
------------
PREREGISTRATION section 4 lists four mandatory baselines. This is number four:
the frozen system-side model (Qwen3.6-27B on Leonardo) is shown a held-out
persona's **public profile** and one held-out probe item, and asked to predict
how that person would answer, as a probability distribution over the item's
answer options. No interview, no transcript, no encoder, no theta. It is the
"could a good LLM just guess this from demographics?" reference that the
end-to-end predictor has to beat.

Cells: the 100 held-out personas x the 50 held-out items of
``experiments/splits_v1.json`` = 5,000 cells. Three independently seeded
repeats per cell at temperature 0.7, so 15,000 generations, each a single short
line. The repeats exist because one sample of a distribution-shaped question is
noisy in a way that is not the model's opinion; averaging three surviving
repeats is the cheapest fix and needs no token distributions (which
``tests/test_wall.py`` rule (c) forbids this side of the Wall anyway).

What it is allowed to see
-------------------------
Exactly two public files: ``data/public/profiles/{pid}.json`` (age, occupation,
city size, household, region type) and ``data/public/bank_items.json`` (item
text and answer options). The profile fields are whitelisted in
:data:`PROFILE_FIELDS` so a future field cannot drift into a prompt unnoticed.
No persona card, no wobble, no trait value, no interview answer, no designed
loading, nothing from the planted-truth directory. That is why the emitted
batch and its completions live in ``data/runs/`` rather than behind the Wall.

Category order (the convention every consumer must share)
---------------------------------------------------------
``probs`` is in **category-index order**, the one ``src.model.mirt`` codes
answers in and the one ``src.eval.truth_distributions`` and ``src.eval.gate4``
read predictions in:

  * ``likert5`` -> answer values 1, 2, 3, 4, 5 (strongly disagree .. strongly
    agree).
  * ``binary``  -> **[no, yes]**, i.e. index 0 = no, index 1 = yes.

The prompt's wire format is a separate thing and lists the options the way the
public bank does ("yes:70 no:30"), because that is the order a reader of the
item sees. The parser matches by name, not by position, so the two orders never
have to agree; :func:`categories` is the output order and
:func:`prompt_categories` is the wire order.

Parsing
-------
A repeat is usable when its completion names every category exactly once with a
non-negative number and at least one number is positive; the vector is then
renormalized to sum 1. Anything else is a malformed repeat and is dropped. A
cell keeps the mean of its surviving repeats; a cell with no surviving repeat
falls back to the uniform distribution and is counted and named in the manifest.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]

BANK_PATH = REPO_ROOT / "data" / "public" / "bank_items.json"
PROFILES_DIR = REPO_ROOT / "data" / "public" / "profiles"
SPLITS_PATH = REPO_ROOT / "experiments" / "splits_v1.json"

#: Repeats per cell. Three independent draws, averaged.
DEFAULT_REPEATS = 3

#: Sampling temperature, the same 0.7 the other sampled-judgment jobs use.
DEFAULT_TEMPERATURE = 0.7

#: Round tag prefix. The driver seeds each record from (seed-base, pid,
#: item_id, round), so the repeat index has to live inside the round tag or
#: three repeats of one cell would share their random numbers.
ROUND_TAG = "qwen_baseline"

#: The only public-profile fields that may enter a prompt, and their label in
#: the prompt text. A field outside this map is never read.
PROFILE_FIELDS: dict[str, str] = {
    "age": "age",
    "occupation": "occupation",
    "city_size": "size of the place they live",
    "household": "household",
    "region_type": "area",
}

LIKERT = "likert5"
BINARY = "binary"

#: Accepted windows for the raw sum of a reply, as a fraction of the scale it
#: was written on: percentages summing near 100, or -- since a few models
#: answer in fractions however the format line is worded -- near 1. The vector
#: is renormalized either way, so the two readings give the identical
#: distribution and this is only a "did it answer the question at all" filter.
#: Wide on purpose: renormalizing a 98-summing line is not a parse failure.
SUM_SCALES: tuple[float, ...] = (100.0, 1.0)
SUM_TOLERANCE = 0.25

SCHEMA = "glassbox.stage4.llm_profile_baseline/1"


# ---------------------------------------------------------------------------
# prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a survey researcher who predicts how one specific person would "
    "answer a survey question. You are given a short public profile of the "
    "person and one survey statement. You reply with a probability "
    "distribution over the answer options, on a single line, in the exact "
    "format asked for -- no words, no explanation, no extra lines."
)


def prompt_categories(item: dict) -> list[str]:
    """The category names in the order the prompt shows them: the bank's order.

    1..5 for a Likert-5 item, [yes, no] for a binary one. This is the wire
    format only -- what the model is asked to write, not how ``probs`` is
    stored.
    """
    kind = item.get("type")
    if kind == LIKERT:
        return [str(option["value"]) for option in item["options"]]
    if kind == BINARY:
        return [str(option["label"]).strip().lower() for option in item["options"]]
    raise ValueError(f"item {item.get('item_id')}: unknown type {kind!r}")


def categories(item: dict) -> list[str]:
    """The category names in **category-index order** -- the output order.

    Likert-5 keeps 1..5; binary is [no, yes], because that is how
    ``src.model.mirt`` codes a binary answer and therefore how the truth store
    and the Gate 4 grader index the category axis.
    """
    kind = item.get("type")
    if kind == LIKERT:
        return prompt_categories(item)
    if kind == BINARY:
        return ["no", "yes"]
    raise ValueError(f"item {item.get('item_id')}: unknown type {kind!r}")


def answer_values(item: dict) -> list[int]:
    """The recorded answer value behind each output category, in the same order."""
    if item.get("type") == BINARY:
        return [0, 1]
    return [int(option["value"]) for option in item["options"]]


def options_line(item: dict) -> str:
    """One line naming the answer options, from the public bank."""
    if item.get("type") == BINARY:
        return "Answer options: yes, no"
    pairs = [
        f"{option['value']} = {option['label']}" for option in item["options"]
    ]
    return "Answer options: " + ", ".join(pairs)


def format_line(item: dict) -> str:
    """The one-line output template the completion has to match."""
    return " ".join(f"{name}:<pct>" for name in prompt_categories(item))


def profile_lines(profile: dict) -> str:
    """The public profile as prompt text -- whitelisted fields only."""
    lines = []
    for field, label in PROFILE_FIELDS.items():
        if field in profile:
            lines.append(f"{label}: {profile[field]}")
    return "\n".join(lines)


def build_prompt(profile: dict, item: dict) -> tuple[str, str]:
    """The (system, user) turns for one persona x one item."""
    user = (
        "PERSON -- this public profile is everything that is known about "
        "them:\n"
        f"{profile_lines(profile)}\n"
        "\n"
        "SURVEY STATEMENT:\n"
        f'"{item["text"]}"\n'
        f"{options_line(item)}\n"
        "\n"
        "QUESTION: If this person were asked this statement many times, how "
        "would their answers be spread over the options? Give the percentage "
        "chance of each option. The percentages must add up to 100.\n"
        "\n"
        "Reply with exactly one line in this format and nothing else:\n"
        f"{format_line(item)}\n"
        "\n"
        "Your answer (one line only):"
    )
    return SYSTEM_PROMPT, user


# ---------------------------------------------------------------------------
# what to emit
# ---------------------------------------------------------------------------


def load_bank() -> dict[str, dict]:
    """The 252 closed items, keyed by item id."""
    payload = json.loads(BANK_PATH.read_text(encoding="utf-8"))
    return {str(item["item_id"]): dict(item) for item in payload["items"]}


def load_profile(pid: str, profiles_dir: Path = PROFILES_DIR) -> dict:
    """One public profile, reduced to the whitelisted fields."""
    payload = json.loads((profiles_dir / f"{pid}.json").read_text(encoding="utf-8"))
    return {field: payload[field] for field in PROFILE_FIELDS if field in payload}


def load_splits(path: Path = SPLITS_PATH) -> tuple[list[str], list[str]]:
    """The frozen held-out persona ids and held-out item ids, sorted."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    personas = sorted(str(pid) for pid in payload["persona_holdout"])
    items = sorted(str(iid) for iid in payload["item_holdout"])
    if not personas or not items:
        raise ValueError(f"{path} has no held-out sets -- not the frozen splits file")
    return personas, items


def iter_records(
    persona_ids: Sequence[str],
    item_ids: Sequence[str],
    bank: dict[str, dict],
    profiles_dir: Path = PROFILES_DIR,
    repeats: int = DEFAULT_REPEATS,
) -> Iterable[dict]:
    """One record per (persona, item, repeat)."""
    for pid in persona_ids:
        profile = load_profile(pid, profiles_dir)
        for item_id in item_ids:
            item = bank[item_id]
            system, user = build_prompt(profile, item)
            for rep in range(repeats):
                yield {
                    "pid": pid,
                    "item_id": item_id,
                    "rep": rep,
                    "round": f"{ROUND_TAG}|r{rep}",
                    "system": system,
                    "user": user,
                }


def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


# ---------------------------------------------------------------------------
# parsing what came back
# ---------------------------------------------------------------------------

_PAIR = re.compile(r"([A-Za-z0-9]+)\s*[:=]\s*([0-9]*\.?[0-9]+)\s*%?")


def parse_distribution(text: str, names: Sequence[str]) -> list[float] | None:
    """The distribution a completion meant, renormalized, or ``None``.

    Usable means: every category named exactly once, every number
    non-negative, at least one positive, and the raw numbers summing near one
    of the :data:`SUM_SCALES`. Extra keys the item does not have make the line
    unusable -- a reply that answers a different question is not a distribution
    over this item.
    """
    found: dict[str, float] = {}
    for key, value in _PAIR.findall(text or ""):
        key = key.strip().lower()
        if key in found:  # a category named twice: ambiguous, drop the line
            return None
        found[key] = float(value)
    wanted = [name.lower() for name in names]
    if sorted(found) != sorted(wanted):
        return None
    values = [found[name] for name in wanted]
    if any(v < 0 for v in values):
        return None
    total = sum(values)
    if total <= 0:
        return None
    if not any(abs(total / scale - 1.0) <= SUM_TOLERANCE for scale in SUM_SCALES):
        return None
    return [v / total for v in values]


def read_completions(path: Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def entropy(probs: Sequence[float]) -> float:
    """Shannon entropy in nats; 0 log 0 counted as 0."""
    return -sum(p * math.log(p) for p in probs if p > 0)


def average(vectors: Sequence[Sequence[float]]) -> list[float]:
    """Mean of the surviving repeats, renormalized against float drift."""
    n = len(vectors)
    summed = [sum(vec[i] for vec in vectors) / n for i in range(len(vectors[0]))]
    total = sum(summed)
    return [value / total for value in summed]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cmd_emit(args: argparse.Namespace) -> None:
    persona_ids, item_ids = load_splits(Path(args.splits))
    bank = load_bank()
    missing = [iid for iid in item_ids if iid not in bank]
    if missing:
        raise SystemExit(f"{len(missing)} held-out item(s) not in the bank: {missing[:3]}")

    records = list(
        iter_records(persona_ids, item_ids, bank, Path(args.profiles), args.repeats)
    )
    out = Path(args.out)
    written = write_jsonl(out, records)

    kinds = {iid: bank[iid]["type"] for iid in item_ids}
    manifest = {
        "schema": SCHEMA,
        "out_file": str(out),
        "records": written,
        "personas": len(persona_ids),
        "items": len(item_ids),
        "repeats": args.repeats,
        "cells": len(persona_ids) * len(item_ids),
        "likert_items": sum(1 for k in kinds.values() if k == LIKERT),
        "binary_items": sum(1 for k in kinds.values() if k == BINARY),
        "round_tag": ROUND_TAG,
        "temperature": DEFAULT_TEMPERATURE,
        "profile_fields": list(PROFILE_FIELDS),
        "longest_prompt_chars": max(
            len(r["system"]) + len(r["user"]) for r in records
        ),
        "wire_format_order": {"likert5": [1, 2, 3, 4, 5], "binary": ["yes", "no"]},
        "output_category_order": {"likert5": [1, 2, 3, 4, 5], "binary": ["no", "yes"]},
    }
    out.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


def cmd_parse(args: argparse.Namespace) -> None:
    persona_ids, item_ids = load_splits(Path(args.splits))
    bank = load_bank()
    wanted = {(pid, iid) for pid in persona_ids for iid in item_ids}

    by_cell: dict[tuple[str, str], list[list[float]]] = {cell: [] for cell in wanted}
    seen = 0
    failures = 0
    unknown = 0
    for row in read_completions(Path(args.completions)):
        cell = (str(row.get("pid")), str(row.get("item_id")))
        if cell not in by_cell:
            unknown += 1
            continue
        seen += 1
        names = categories(bank[cell[1]])
        probs = parse_distribution(row.get("completion", ""), names)
        if probs is None:
            failures += 1
            continue
        by_cell[cell].append(probs)

    fallbacks: list[dict[str, str]] = []
    repeat_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    lines: list[dict[str, Any]] = []
    entropies: list[float] = []
    for pid in persona_ids:
        for item_id in item_ids:
            kept = by_cell[(pid, item_id)]
            repeat_counts[len(kept)] = repeat_counts.get(len(kept), 0) + 1
            if kept:
                probs = average(kept)
            else:
                k = len(categories(bank[item_id]))
                probs = [1.0 / k] * k
                fallbacks.append({"persona_id": pid, "item_id": item_id})
            entropies.append(entropy(probs))
            lines.append(
                {"persona_id": pid, "item_id": item_id, "probs": probs}
            )

    out = Path(args.out)
    write_jsonl(out, lines)

    manifest = {
        "schema": SCHEMA,
        "out_file": str(out),
        "completions_file": str(args.completions),
        "cells": len(lines),
        "completions_read": seen,
        "completions_off_split": unknown,
        "malformed_repeats": failures,
        "repeats_kept_histogram": {str(k): v for k, v in sorted(repeat_counts.items())},
        "uniform_fallback_cells": len(fallbacks),
        "fallback_list": fallbacks,
        "mean_entropy_nats": sum(entropies) / len(entropies) if entropies else None,
        "category_order": {
            "likert5": [1, 2, 3, 4, 5],
            "binary": ["no", "yes"],
            "note": (
                "category-index order, the one src.model.mirt codes answers in "
                "and src.eval.gate4 reads predictions in"
            ),
        },
    }
    out.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    summary = {k: v for k, v in manifest.items() if k != "fallback_list"}
    print(json.dumps(summary, indent=2))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Gate 4 baseline 4: LLM with the public profile, no interview."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    emit = sub.add_parser("emit", help="Write the prediction prompt batch as JSONL.")
    emit.add_argument("--out", required=True)
    emit.add_argument("--splits", default=str(SPLITS_PATH))
    emit.add_argument("--profiles", default=str(PROFILES_DIR))
    emit.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    emit.set_defaults(func=cmd_emit)

    parse = sub.add_parser("parse", help="Average the repeats into one row per cell.")
    parse.add_argument("--completions", required=True)
    parse.add_argument("--out", required=True)
    parse.add_argument("--splits", default=str(SPLITS_PATH))
    parse.set_defaults(func=cmd_parse)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
