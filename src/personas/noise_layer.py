"""Mechanical answer noise, driven by the responder's own answer distribution.

    python -m src.personas.noise_layer \\
        --answers <answers.jsonl> --logprobs <completions.jsonl> \\
        --noise-dir <dir> --public-bank <bank_items.json> \\
        --out <noised answers.jsonl> --stats <stats.json> \\
        --a 0.15 --b 0.6 --t-noise 2.0 --seed-base 548

PREREGISTRATION.md section 2 gives every persona a wobble: "how often it flips
or shifts answers on items it is lukewarm about". The responder was supposed to
produce that itself, from a paragraph in its system prompt telling it how steady
it is. It does not: measured test-retest agreement came out at 0.978 and did not
move when the sampling temperature was raised from 0.7 to 1.3, because the
model's answer-token distribution is extremely peaked. The prompt-side
instruction is dead. This module implements the frozen definition literally
instead, as a layer applied after the answers are parsed.

The layer is not arbitrary jitter. It reads the responder's own top-20 logprobs
for the answer token, so "lukewarm" means what the responder actually found hard:

  1. Build the distribution over the valid answers for that item -- "1".."5" for
     a Likert item, the yes and no spellings for a binary one -- from the
     recorded logprobs, renormalised over just those. Its largest probability is
     the persona's conviction on that item.
  2. A noise event fires with probability ``wobble * (a + b * (1 - conviction))``,
     capped at 0.95. The ``a`` term is a flat slip rate, because people
     misclick and misread even on questions they are sure about; the ``b`` term
     is the ambivalence boost, which is what makes a lukewarm item wobble more
     than a convicted one.
  3. On an event the answer is re-drawn from the same distribution flattened by
     ``T_noise``. The original answer can come back up -- it usually should,
     since it was the likeliest -- and that is correct: a slip is a re-draw, not
     a forced change.

Every cell has its own seed, from the record's identity and ``--seed-base``, so
the layer is reproducible and the main round and the retest round wobble
independently of each other.

CONSISTENCY RULE: from Stage 3 onward every closed-form answer a persona gives
in an interview must come out of this layer, at the parameters frozen in
PREREGISTRATION.md section 5, addendum -- a = 1.2, b = 4.0, T_noise = 16,
seed-base 548, seeded per record. No interview path may bypass it or re-run it
with different numbers, because the true answer distribution that Stage 4 is
graded against is defined as repeated seeded applications of this same layer.
The full statement of the rule is in ``src/interview/__init__.py``.

WALL RULE: this is persona-side code. The wobble directory, the answers and the
item design all arrive as command-line arguments and none of them is written
down here. The output is a persona-side artifact: keep it with the rest of the
batch, never in a system-side folder and never in git.

Output schema is exactly the input's, so ``src.eval.population_qa`` reads a
noised answers file without knowing anything happened. Everything diagnostic --
event rates per persona and per item class, the conviction spread -- goes to the
separate stats file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

ANSWERS_FILENAME = "answers.jsonl"

#: How far below the weakest recorded logprob a valid answer sits when it did
#: not make the top-20 at all. Two nats: clearly unlikely, never impossible.
FLOOR_MARGIN = 2.0

#: A noise event never becomes a certainty, however high the wobble.
MAX_EVENT_PROBABILITY = 0.95

#: The Likert answers, as the token text the responder would emit.
LIKERT_ANSWERS: tuple[int, ...] = (1, 2, 3, 4, 5)

#: The exact token texts that count as a Likert answer. Spelled out rather than
#: tested with ``str.isdigit()``: the responder's top-20 really does contain
#: things like the subscript "₃", which is a digit by Unicode's reckoning
#: but is not the answer "3" and is not even parseable by ``int()``.
LIKERT_TOKENS: dict[str, int] = {str(value): value for value in LIKERT_ANSWERS}

#: Spellings of a binary answer that a first token can carry. Compared after
#: stripping surrounding whitespace and lowercasing, so " Yes" counts as "yes".
YES_TOKENS: frozenset[str] = frozenset({"yes", "y", "yeah", "yep", "yup", "true", "correct"})
NO_TOKENS: frozenset[str] = frozenset({"no", "n", "nope", "nah", "false", "incorrect"})


@dataclass(frozen=True)
class NoiseConfig:
    """The three tunable numbers of the layer.

    ``a``       flat slip rate: the share of a persona's wobble that fires even
                on an item it is completely sure about.
    ``b``       ambivalence boost: how much more often an item fires as the
                responder's conviction on it falls.
    ``t_noise`` how flat the re-draw distribution is. 1.0 re-draws from the
                responder's own probabilities; larger values flatten towards
                uniform, so a slip is more likely to land somewhere else.
    """

    a: float = 0.15
    b: float = 0.6
    t_noise: float = 2.0

    def __post_init__(self) -> None:
        if self.a < 0 or self.b < 0:
            raise ValueError(f"a and b must be non-negative, got a={self.a}, b={self.b}")
        if self.t_noise <= 0:
            raise ValueError(f"t_noise must be positive, got {self.t_noise}")

    def as_dict(self) -> dict[str, float]:
        return {"a": float(self.a), "b": float(self.b), "t_noise": float(self.t_noise)}


# --------------------------------------------------------------------------
# the answer distribution
# --------------------------------------------------------------------------


def normalize_token(text: Any) -> str:
    """A recorded token reduced to the form the answer sets are written in."""
    return str(text).strip().lower()


def answer_of_token(text: Any, item_type: str) -> int | str | None:
    """Which valid answer this token is, or ``None`` if it is not one.

    Likert tokens are the plain ASCII digits 1-5; binary tokens are any
    spelling in :data:`YES_TOKENS` or :data:`NO_TOKENS`. Leading spaces and
    capitals are ignored, which is what makes " Yes" and "yes" the same answer.
    """
    token = normalize_token(text)
    if item_type == "likert5":
        return LIKERT_TOKENS.get(token)
    if item_type == "binary":
        if token in YES_TOKENS:
            return "yes"
        if token in NO_TOKENS:
            return "no"
        return None
    raise ValueError(f"unknown item type {item_type!r}")


def valid_answers(item_type: str) -> tuple[int | str, ...]:
    """Every answer an item of this type can take, in a fixed order."""
    if item_type == "likert5":
        return LIKERT_ANSWERS
    if item_type == "binary":
        return ("no", "yes")
    raise ValueError(f"unknown item type {item_type!r}")


def answer_distribution(
    logprobs: Mapping[str, float] | None,
    item_type: str,
) -> dict[int | str, float] | None:
    """Probability of each valid answer, from one cell's recorded logprobs.

    Tokens that spell the same answer are added together, because they are
    mutually exclusive ways of saying it. A valid answer that did not appear in
    the top-20 is floored at ``min(recorded) - FLOOR_MARGIN`` rather than zero,
    so no answer is ever impossible to re-draw. The result is renormalised over
    the valid answers alone.

    ``None`` when there is nothing to build a distribution from.
    """
    if not logprobs:
        return None

    values = [float(v) for v in logprobs.values()]
    if not values:
        return None
    floor = min(values) - FLOOR_MARGIN

    # log-sum-exp per answer, so variant spellings add as probabilities.
    per_answer: dict[int | str, list[float]] = {a: [] for a in valid_answers(item_type)}
    for token, logprob in logprobs.items():
        answer = answer_of_token(token, item_type)
        if answer is not None:
            per_answer[answer].append(float(logprob))

    logits: dict[int | str, float] = {}
    for answer, found in per_answer.items():
        if not found:
            logits[answer] = floor
            continue
        top = max(found)
        logits[answer] = top + math.log(sum(math.exp(v - top) for v in found))

    biggest = max(logits.values())
    weights = {a: math.exp(v - biggest) for a, v in logits.items()}
    total = sum(weights.values())
    if total <= 0:  # pragma: no cover -- max weight is 1.0 by construction
        return None
    return {a: w / total for a, w in weights.items()}


def conviction(distribution: Mapping[int | str, float]) -> float:
    """How sure the responder was: the largest probability in the distribution."""
    return float(max(distribution.values()))


# --------------------------------------------------------------------------
# the event and the re-draw
# --------------------------------------------------------------------------


def event_probability(wobble: float, p_max: float, config: NoiseConfig) -> float:
    """How often this cell wobbles: ``wobble * (a + b * (1 - conviction))``, capped."""
    raw = float(wobble) * (config.a + config.b * (1.0 - float(p_max)))
    return float(min(max(raw, 0.0), MAX_EVENT_PROBABILITY))


def flattened(distribution: Mapping[int | str, float], t_noise: float) -> dict[int | str, float]:
    """The re-draw distribution: ``p ** (1 / t_noise)``, renormalised.

    ``t_noise = 1`` is the responder's own distribution. Larger values pull it
    towards uniform, which makes a slip more likely to land on a different
    answer instead of re-drawing the same one.
    """
    logits = {
        a: (math.log(p) / t_noise if p > 0 else -math.inf)
        for a, p in distribution.items()
    }
    biggest = max(logits.values())
    weights = {a: math.exp(v - biggest) for a, v in logits.items()}
    total = sum(weights.values())
    return {a: w / total for a, w in weights.items()}


def cell_seed(seed_base: int, pid: str, item_id: str, round_name: str) -> int:
    """A seed of this cell's own: reproducible, and independent across rounds."""
    material = f"{seed_base}|{pid}|{item_id}|{round_name}"
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def draw(distribution: Mapping[int | str, float], rng: np.random.Generator) -> int | str:
    """One answer from a distribution, by inverse CDF over a fixed key order."""
    answers = sorted(distribution)
    cumulative = 0.0
    point = float(rng.random())
    for answer in answers:
        cumulative += distribution[answer]
        if point < cumulative:
            return answer
    return answers[-1]


# --------------------------------------------------------------------------
# files
# --------------------------------------------------------------------------


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def cell_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("pid")), str(row.get("item_id")), str(row.get("round") or "main"))


def item_types(public_bank: str | Path) -> dict[str, str]:
    """Item id -> item type, from the public bank."""
    payload = json.loads(Path(public_bank).read_text(encoding="utf-8"))
    return {str(row["item_id"]): str(row["type"]) for row in payload.get("items", [])}


def load_wobbles(noise_dir: str | Path) -> dict[str, float]:
    """pid -> wobble, from a directory of ``{pid}.json`` records."""
    wobbles: dict[str, float] = {}
    for path in sorted(Path(noise_dir).glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        value = record.get("wobble") if isinstance(record, Mapping) else record
        if value is not None:
            wobbles[path.stem] = float(value)
    return wobbles


def logprob_index(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str], dict[str, float]]:
    """(pid, item_id, round) -> recorded logprobs, from a completions file."""
    found: dict[tuple[str, str, str], dict[str, float]] = {}
    for row in rows:
        recorded = row.get("logprobs")
        if recorded:
            found[cell_key(row)] = {str(k): float(v) for k, v in recorded.items()}
    return found


def design_classes(bank_design: Mapping[str, Any]) -> dict[str, str]:
    """Item id -> design class, for the by-class breakdown in the stats file.

    Four classes, matching how the bank was generated: ``primary`` (one
    dimension at full strength), ``cross`` (two dimensions), ``weak``
    (deliberately mushy) and ``distractor`` (no designed loading). Used for
    reporting only -- it never changes an answer.
    """
    classes: dict[str, str] = {}
    for row in bank_design.get("items", []):
        loadings = {str(k): float(v) for k, v in (row.get("loadings") or {}).items()}
        live = [dim for dim, value in loadings.items() if value != 0.0]
        strength = str(row.get("strength_class", "strong"))
        if strength == "none" or not live:
            kind = "distractor"
        elif strength == "weak":
            kind = "weak"
        elif len(live) > 1:
            kind = "cross"
        else:
            kind = "primary"
        classes[str(row["item_id"])] = kind
    return classes


# --------------------------------------------------------------------------
# applying the layer
# --------------------------------------------------------------------------


def _percentiles(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {}
    array = np.asarray(values, dtype=float)
    return {
        str(step): round(float(np.percentile(array, step)), 6)
        for step in range(0, 101, 10)
    }


def apply_noise(
    answers: Sequence[Mapping[str, Any]],
    logprobs_at: Mapping[tuple[str, str, str], Mapping[str, float]],
    wobbles: Mapping[str, float],
    types: Mapping[str, str],
    config: NoiseConfig,
    seed_base: int,
    classes: Mapping[str, str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Noise every answer that has a distribution behind it.

    Returns the new rows -- same keys as the input, minus any inline logprobs --
    and a stats block. A cell with no recorded logprobs, no wobble on file or no
    item type is passed through untouched and counted; the layer never guesses.
    """
    out: list[dict[str, Any]] = []
    per_persona: dict[str, dict[str, Any]] = {}
    per_class: dict[str, dict[str, int]] = {}
    per_type: dict[str, dict[str, int]] = {}
    convictions: list[float] = []
    event_rates: list[float] = []
    skipped = {"no_logprobs": 0, "no_wobble": 0, "unknown_item": 0, "no_distribution": 0}
    events = 0
    changed = 0

    def bucket(store: dict[str, dict[str, int]], name: str) -> dict[str, int]:
        return store.setdefault(name, {"cells": 0, "events": 0, "changed": 0})

    for row in answers:
        pid, item_id, round_name = cell_key(row)
        record = {k: v for k, v in row.items() if k != "logprobs"}

        item_type = types.get(item_id)
        recorded = row.get("logprobs") or logprobs_at.get((pid, item_id, round_name))
        wobble = wobbles.get(pid)

        if item_type is None:
            skipped["unknown_item"] += 1
            out.append(record)
            continue
        if not recorded:
            skipped["no_logprobs"] += 1
            out.append(record)
            continue
        if wobble is None:
            skipped["no_wobble"] += 1
            out.append(record)
            continue

        distribution = answer_distribution(recorded, item_type)
        if distribution is None:
            skipped["no_distribution"] += 1
            out.append(record)
            continue

        p_max = conviction(distribution)
        probability = event_probability(wobble, p_max, config)
        convictions.append(p_max)
        event_rates.append(probability)

        persona = per_persona.setdefault(
            pid, {"wobble": float(wobble), "cells": 0, "events": 0, "changed": 0}
        )
        persona["cells"] += 1
        type_bucket = bucket(per_type, item_type)
        type_bucket["cells"] += 1
        class_bucket = bucket(per_class, (classes or {}).get(item_id, "unclassified"))
        class_bucket["cells"] += 1

        rng = np.random.default_rng(cell_seed(seed_base, pid, item_id, round_name))
        if float(rng.random()) < probability:
            events += 1
            persona["events"] += 1
            type_bucket["events"] += 1
            class_bucket["events"] += 1
            new_answer = draw(flattened(distribution, config.t_noise), rng)
            if new_answer != record.get("answer"):
                changed += 1
                persona["changed"] += 1
                type_bucket["changed"] += 1
                class_bucket["changed"] += 1
            record["answer"] = new_answer

        out.append(record)

    def rate(part: int, whole: int) -> float | None:
        return round(part / whole, 6) if whole else None

    for store in (per_type, per_class):
        for name, counts in store.items():
            counts["event_rate"] = rate(counts["events"], counts["cells"])
            counts["change_rate"] = rate(counts["changed"], counts["cells"])
    for pid, counts in per_persona.items():
        counts["event_rate"] = rate(counts["events"], counts["cells"])
        counts["change_rate"] = rate(counts["changed"], counts["cells"])

    noised = len(convictions)
    stats: dict[str, Any] = {
        "config": config.as_dict(),
        "seed_base": int(seed_base),
        "n_answers": len(answers),
        "n_noise_eligible": noised,
        "n_events": events,
        "n_changed": changed,
        "event_rate": rate(events, noised),
        "change_rate": rate(changed, noised),
        "skipped": skipped,
        "conviction_percentiles": _percentiles(convictions),
        "mean_conviction": round(float(np.mean(convictions)), 6) if convictions else None,
        "mean_event_probability": round(float(np.mean(event_rates)), 6) if event_rates else None,
        "by_item_type": per_type,
        "by_design_class": per_class,
        "by_persona": {pid: per_persona[pid] for pid in sorted(per_persona)},
    }
    return out, stats


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.personas.noise_layer",
        description="Add the planted wobble to parsed answers, mechanically.",
    )
    parser.add_argument("--answers", type=Path, required=True,
                        help="parsed answers JSONL; may carry inline 'logprobs'")
    parser.add_argument("--logprobs", type=Path, default=None,
                        help=("JSONL carrying pid/item_id/round/logprobs -- normally "
                              "the completions file the answers were parsed from. "
                              "Needed unless the answers already carry them inline."))
    parser.add_argument("--noise-dir", type=Path, required=True,
                        help="directory of {pid}.json wobble records, supplied per run")
    parser.add_argument("--public-bank", type=Path, required=True,
                        help="the public bank file: needed for each item's type")
    parser.add_argument("--bank-design", type=Path, default=None,
                        help=("the bank's design file, for the by-design-class "
                              "breakdown in the stats. Reporting only; it never "
                              "changes an answer."))
    parser.add_argument("--out", type=Path, required=True,
                        help="where to write the noised answers JSONL")
    parser.add_argument("--stats", type=Path, default=None,
                        help="where to write the stats JSON (default: <out>.stats.json)")
    parser.add_argument("--a", type=float, default=NoiseConfig.a,
                        help="flat slip rate (default %(default)s)")
    parser.add_argument("--b", type=float, default=NoiseConfig.b,
                        help="ambivalence boost (default %(default)s)")
    parser.add_argument("--t-noise", type=float, default=NoiseConfig.t_noise,
                        help="re-draw flattening temperature (default %(default)s)")
    parser.add_argument("--seed-base", type=int, default=0,
                        help="base for the per-cell seeds (default %(default)s)")
    args = parser.parse_args(argv)

    answers = read_jsonl(args.answers)
    if not answers:
        parser.error(f"no answers in {args.answers}")

    logprobs_at = logprob_index(read_jsonl(args.logprobs)) if args.logprobs else {}
    inline = any(row.get("logprobs") for row in answers)
    if not logprobs_at and not inline:
        parser.error(
            "no logprobs anywhere: the answers carry none inline and --logprobs "
            "was not given. The layer needs the responder's answer distribution."
        )

    wobbles = load_wobbles(args.noise_dir)
    if not wobbles:
        parser.error(f"no wobble records in {args.noise_dir}")

    types = item_types(args.public_bank)
    classes = (
        design_classes(json.loads(args.bank_design.read_text(encoding="utf-8")))
        if args.bank_design
        else None
    )

    config = NoiseConfig(a=args.a, b=args.b, t_noise=args.t_noise)
    rows, stats = apply_noise(
        answers, logprobs_at, wobbles, types, config, args.seed_base, classes
    )

    write_jsonl(args.out, rows)
    stats_path = args.stats or Path(str(args.out) + ".stats.json")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    print(
        f"noised {stats['n_noise_eligible']} of {stats['n_answers']} answers: "
        f"{stats['n_events']} events ({stats['event_rate']}), "
        f"{stats['n_changed']} actually changed ({stats['change_rate']})"
    )
    if any(stats["skipped"].values()):
        print(f"  passed through untouched: {stats['skipped']}")
    print(f"  answers -> {args.out}")
    print(f"  stats   -> {stats_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
