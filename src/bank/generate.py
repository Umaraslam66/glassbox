"""Draft the question bank with an LLM, against a design fixed in code.

The design comes first and the model only fills in wording. Every item exists
because a *slot* exists: a slot says which trait(s) the item must load on, with
what sign and strength, whether it is Likert or binary, and what topic it must
be about. Slots are built from ``BankSpec`` plus a seeded random number
generator; the model is then asked to write text for each slot. That ordering is
what makes the planted loadings real planted truth rather than a post-hoc guess.

Four kinds of closed item (full design in brackets):

  primary      [192]  one dimension, loading +/-1.0. 24 per dimension:
                      16 Likert + 8 binary, one third of each group negatively
                      keyed (agreeing signals a LOW trait value).
  cross        [30]   two dimensions, 0.7 on the main one and 0.5 on the second,
                      signs varied. Covers all 28 unordered dimension pairs,
                      including the two the population correlation matrix sets
                      to exactly zero (TRU-RSK, ENV-RSK).
  weak         [15]   one dimension, magnitude 0.25, deliberately mushy wording.
                      Stage 2 must recover these as low-discrimination items.
  distractor   [15]   no loadings at all. Pure taste questions.

Plus 20 open-ended interview prompts, each tagged with one or two dimensions.

WALL: the private output directory is only ever a command-line argument. No
path to it is written anywhere in this file.

Run:
    python -m src.bank.generate --out-truth <dir> --out-public <dir> [--seed N]
    python -m src.bank.generate --dry-run          # prints the prompts, no calls
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.llm_client import LLMClient, get_client, load_dotenv_if_present

from src.bank.schema import (
    DIMENSION_HIGH,
    DIMENSION_LABELS,
    DIMENSION_LOW,
    DIMENSIONS,
    TOPIC_DOMAINS,
    Bank,
    BankSpec,
    ClosedItem,
    OpenItem,
    closed_item_id,
    normalize_text,
    open_item_id,
    write_bank,
)

# --------------------------------------------------------------------------
# Which topics each dimension gets asked about
# --------------------------------------------------------------------------

#: Six topic domains per dimension, most natural first. A dimension's primary
#: items are dealt evenly across its six, so no trait is measured only through
#: one kind of situation -- which is what makes the later near / same-domain /
#: far probe split meaningful. Every domain is shared by at least three
#: dimensions, so no domain ends up too thin to stratify on.
DIMENSION_DOMAINS: dict[str, tuple[str, ...]] = {
    "TRU": ("public-services", "health", "media", "money", "neighborhood", "work"),
    "RSK": ("money", "work", "travel", "health", "technology", "social-life"),
    "ENV": ("environment", "shopping", "travel", "neighborhood", "media", "money"),
    "PRC": ("money", "shopping", "travel", "technology", "health", "public-services"),
    "TRD": ("family-traditions", "social-life", "media", "work", "neighborhood", "health"),
    "SOC": ("social-life", "neighborhood", "family-traditions", "work", "travel", "public-services"),
    "TEC": ("technology", "media", "work", "shopping", "environment", "public-services"),
    "LOC": ("neighborhood", "travel", "work", "social-life", "family-traditions", "environment"),
}

#: The two pairs the frozen population correlation matrix sets to exactly zero.
#: Cross-loading items on these pairs matter most: recovery there cannot lean on
#: a correlation crutch.
REQUIRED_CROSS_PAIRS: tuple[tuple[str, str], ...] = (("TRU", "RSK"), ("ENV", "RSK"))

#: Open-ended prompts: eight single-dimension, then twelve two-dimension.
OPEN_SLOT_DIMS: tuple[tuple[str, ...], ...] = (
    ("TRU",),
    ("RSK",),
    ("ENV",),
    ("PRC",),
    ("TRD",),
    ("SOC",),
    ("TEC",),
    ("LOC",),
    ("TRU", "ENV"),
    ("RSK", "PRC"),
    ("TRD", "LOC"),
    ("SOC", "LOC"),
    ("TEC", "TRU"),
    ("ENV", "PRC"),
    ("RSK", "TEC"),
    ("TRD", "SOC"),
    ("PRC", "SOC"),
    ("ENV", "LOC"),
    ("TRU", "RSK"),
    ("TRD", "TEC"),
)


def _round_half_up(value: float) -> int:
    return int(value + 0.5)


def _two_thirds(n: int) -> int:
    """How many of ``n`` items are Likert, keeping the bank's 2:1 Likert:binary mix."""
    return _round_half_up(2 * n / 3)


def _one_third(n: int) -> int:
    """How many of ``n`` items are negatively keyed."""
    return _round_half_up(n / 3)


# --------------------------------------------------------------------------
# Slots and drafting cells
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DraftCell:
    """Everything about a slot except which topic it is about.

    Slots sharing a cell share a drafting prompt, so the model writes a batch of
    items with the same trait, sign, type and strength in one call.
    """

    kind: str
    dims: tuple[str, ...]
    loadings: tuple[tuple[str, float], ...]
    item_type: str
    strength_class: str
    negatively_keyed: bool
    design_notes: str

    def loading_dict(self) -> dict[str, float]:
        return {dim: value for dim, value in self.loadings}


@dataclass(frozen=True)
class Slot:
    """One item to be written: its design cell plus its topic domain."""

    cell: DraftCell
    topic_domain: str


def deal_domains(rng: random.Random, palette: Sequence[str], count: int) -> list[str]:
    """Deal ``count`` topics from ``palette`` as evenly as the count allows.

    Each full pass through the palette is shuffled separately, so with 24 items
    over 6 topics every topic appears exactly 4 times in a seed-dependent order.
    """
    pool: list[str] = []
    while len(pool) < count:
        chunk = list(palette)
        rng.shuffle(chunk)
        pool.extend(chunk)
    return pool[:count]


def cross_pair_order() -> list[tuple[str, str]]:
    """All 28 unordered dimension pairs, the two frozen-zero pairs first."""
    ordered: list[tuple[str, str]] = list(REQUIRED_CROSS_PAIRS)
    already = {frozenset(pair) for pair in ordered}
    for index, first in enumerate(DIMENSIONS):
        for second in DIMENSIONS[index + 1:]:
            if frozenset((first, second)) in already:
                continue
            ordered.append((first, second))
    return ordered


def _primary_cell(dim: str, item_type: str, negative: bool, spec: BankSpec) -> DraftCell:
    sign = -1.0 if negative else 1.0
    direction = "low" if negative else "high"
    return DraftCell(
        kind="primary",
        dims=(dim,),
        loadings=((dim, sign * spec.strong_loading),),
        item_type=item_type,
        strength_class="strong",
        negatively_keyed=negative,
        design_notes=(
            f"primary item on {dim}; agreeing signals a {direction} value on "
            f"{DIMENSION_LABELS[dim]}"
        ),
    )


def _cross_cell(
    primary: str,
    secondary: str,
    sign_primary: float,
    sign_secondary: float,
    item_type: str,
    spec: BankSpec,
) -> DraftCell:
    return DraftCell(
        kind="cross",
        dims=(primary, secondary),
        loadings=(
            (primary, sign_primary * spec.cross_primary_loading),
            (secondary, sign_secondary * spec.cross_secondary_loading),
        ),
        item_type=item_type,
        strength_class="strong",
        negatively_keyed=sign_primary < 0,
        design_notes=(
            f"cross-loading item: {primary} (main) with {secondary} (secondary); "
            f"tests whether fitting separates two traits inside one statement"
        ),
    )


def _weak_cell(dim: str, item_type: str, negative: bool, spec: BankSpec) -> DraftCell:
    sign = -1.0 if negative else 1.0
    return DraftCell(
        kind="weak",
        dims=(dim,),
        loadings=((dim, sign * spec.weak_loading),),
        item_type=item_type,
        strength_class="weak",
        negatively_keyed=negative,
        design_notes=(
            f"designed-weak item on {dim}; mushy wording on purpose, so fitting "
            f"must recover a LOW discrimination for it"
        ),
    )


def _distractor_cell(item_type: str) -> DraftCell:
    return DraftCell(
        kind="distractor",
        dims=(),
        loadings=(),
        item_type=item_type,
        strength_class="none",
        negatively_keyed=False,
        design_notes="distractor: no designed loading on any dimension",
    )


def _open_cell(dims: tuple[str, ...]) -> DraftCell:
    labels = ", ".join(DIMENSION_LABELS[d] for d in dims)
    return DraftCell(
        kind="open",
        dims=dims,
        loadings=(),
        item_type="open",
        strength_class="none",
        negatively_keyed=False,
        design_notes=f"open-ended interview prompt aimed at {labels}",
    )


def build_closed_slots(spec: BankSpec, rng: random.Random) -> list[Slot]:
    """Every closed-item slot, in the order the items will be numbered."""
    slots: list[Slot] = []

    # --- primary: 24 per dimension in the full design -------------------
    for dim in DIMENSIONS:
        palette = DIMENSION_DOMAINS[dim]
        domains = deal_domains(rng, palette, spec.per_dim)
        cursor = 0
        groups = (
            ("likert5", spec.likert_per_dim, spec.neg_likert_per_dim),
            ("binary", spec.binary_per_dim, spec.neg_binary_per_dim),
        )
        for item_type, total, negatives in groups:
            for position in range(total):
                negative = position >= total - negatives
                slots.append(
                    Slot(
                        cell=_primary_cell(dim, item_type, negative, spec),
                        topic_domain=domains[cursor],
                    )
                )
                cursor += 1

    # --- cross-loading --------------------------------------------------
    pairs = cross_pair_order()
    for index in range(spec.n_cross):
        if index < len(pairs):
            primary, secondary = pairs[index]
        else:
            # Extra slots revisit the frozen-zero pairs with the roles swapped,
            # so those two pairs get an item from each side.
            swapped = REQUIRED_CROSS_PAIRS[(index - len(pairs)) % len(REQUIRED_CROSS_PAIRS)]
            primary, secondary = swapped[1], swapped[0]
        sign_primary = 1.0 if index % 4 < 2 else -1.0
        sign_secondary = 1.0 if index % 2 == 0 else -1.0
        item_type = "binary" if index % 3 == 2 else "likert5"
        palette = tuple(
            dict.fromkeys(DIMENSION_DOMAINS[primary] + DIMENSION_DOMAINS[secondary])
        )
        slots.append(
            Slot(
                cell=_cross_cell(primary, secondary, sign_primary, sign_secondary, item_type, spec),
                topic_domain=deal_domains(rng, palette, 1)[0],
            )
        )

    # --- designed-weak --------------------------------------------------
    weak_likert = _two_thirds(spec.n_weak)
    weak_plan = [("likert5", weak_likert), ("binary", spec.n_weak - weak_likert)]
    weak_index = 0
    for item_type, total in weak_plan:
        negatives = _one_third(total)
        for position in range(total):
            dim = DIMENSIONS[weak_index % len(DIMENSIONS)]
            negative = position >= total - negatives
            slots.append(
                Slot(
                    cell=_weak_cell(dim, item_type, negative, spec),
                    topic_domain=deal_domains(rng, DIMENSION_DOMAINS[dim], 1)[0],
                )
            )
            weak_index += 1

    # --- distractors ----------------------------------------------------
    distractor_likert = _two_thirds(spec.n_distractor)
    distractor_domains = deal_domains(rng, TOPIC_DOMAINS, spec.n_distractor)
    for position in range(spec.n_distractor):
        item_type = "likert5" if position < distractor_likert else "binary"
        slots.append(
            Slot(cell=_distractor_cell(item_type), topic_domain=distractor_domains[position])
        )

    return slots


def build_open_slots(spec: BankSpec, rng: random.Random) -> list[Slot]:
    """Every open-ended prompt slot, in numbering order."""
    slots: list[Slot] = []
    for index in range(spec.n_open):
        dims = OPEN_SLOT_DIMS[index % len(OPEN_SLOT_DIMS)]
        palette = tuple(dict.fromkeys(sum((DIMENSION_DOMAINS[d] for d in dims), ())))
        slots.append(
            Slot(cell=_open_cell(dims), topic_domain=deal_domains(rng, palette, 1)[0])
        )
    return slots


def build_slots(spec: BankSpec, seed: int) -> tuple[list[Slot], list[Slot]]:
    """Closed and open slots for one seed. Same seed, same slots, always."""
    rng = random.Random(seed)
    closed = build_closed_slots(spec, rng)
    open_slots = build_open_slots(spec, rng)
    return closed, open_slots


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You draft items for a research questionnaire. You follow the writing rules "
    "exactly and you reply with strict JSON only -- no commentary, no markdown "
    "fences, no numbering outside the JSON."
)

_SHARED_RULES = (
    "- First person, present tense, about the respondent's own life. Example of "
    'the register: "I generally trust what city officials tell us".',
    "- Everyday and concrete. Name a real situation, not an abstraction.",
    "- No jargon and no trait words. Never write words like trust, risk, "
    "environmental, sustainable, price-conscious, traditional, sociable, "
    "tech-savvy, local identity, or any label for the thing being measured.",
    "- No double negatives. No questions -- these are statements.",
    "- Every statement must describe a different situation from the others.",
)

_ANSWER_RULES = {
    "likert5": (
        "- The respondent answers on a 1-5 scale from 'strongly disagree' to "
        "'strongly agree'. Write something a person can genuinely land anywhere "
        "on."
    ),
    "binary": (
        "- The respondent answers only 'yes' or 'no'. Phrase the statement so a "
        "flat yes or no is a natural answer."
    ),
}


def _trait_block(dims: Sequence[str]) -> str:
    lines = []
    for dim in dims:
        lines.append(
            f"{DIMENSION_LABELS[dim]}\n"
            f"  someone LOW on it: {DIMENSION_LOW[dim]}\n"
            f"  someone HIGH on it: {DIMENSION_HIGH[dim]}"
        )
    return "\n".join(lines)


def _topic_block(domains: Sequence[str]) -> str:
    return "\n".join(f"{position}. {domain}" for position, domain in enumerate(domains, start=1))


def _output_block(count: int, field_name: str) -> str:
    return (
        f"OUTPUT\n"
        f"A JSON array of exactly {count} objects, in the same order as the topic "
        f'list, each of the form {{"topic_domain": "<topic from the list>", '
        f'"{field_name}": "<the text>"}}. Return the array and nothing else.'
    )


def _avoid_block(avoid: Sequence[str], retry: bool = False) -> str:
    blocks = []
    if retry:
        blocks.append(
            "\nRETRY -- the previous reply could not be used: it was empty, "
            "malformed, or repeated wording already in the bank. Write different "
            "statements this time.\n"
        )
    if avoid:
        listed = "\n".join(f"- {text}" for text in avoid[:40])
        blocks.append(
            "\nALREADY WRITTEN -- do not repeat, reorder or paraphrase any of these:\n"
            f"{listed}\n"
        )
    return "".join(blocks)


def build_prompt(
    cell: DraftCell,
    domains: Sequence[str],
    avoid: Sequence[str] = (),
    retry: bool = False,
) -> tuple[str, str]:
    """The (system, user) prompt that drafts one chunk of one design cell."""
    count = len(domains)
    if cell.kind == "open":
        return SYSTEM_PROMPT, _open_prompt(cell, domains, avoid, retry)
    if cell.kind == "distractor":
        return SYSTEM_PROMPT, _distractor_prompt(cell, domains, avoid, retry)

    direction = "LOW" if cell.negatively_keyed else "HIGH"
    parts = [
        f"TASK\nWrite {count} survey statements for a questionnaire.",
        f"TRAIT BEING MEASURED (never name it in the statement)\n{_trait_block(cell.dims[:1])}",
    ]

    if cell.kind == "cross":
        secondary = cell.dims[1]
        loadings = cell.loading_dict()
        secondary_direction = "LOW" if loadings[secondary] < 0 else "HIGH"
        parts.append(
            "SECOND TRAIT, more weakly involved (never name it either)\n"
            f"{_trait_block((secondary,))}"
        )
        parts.append(
            "DIRECTION\n"
            f"A respondent who AGREES is {direction} on the first trait and "
            f"{secondary_direction} on the second. The first trait must be the "
            "stronger pull. Write a situation where both genuinely matter at "
            "once, not two ideas stapled together."
        )
    else:
        parts.append(
            f"DIRECTION\nA respondent who AGREES with the statement is {direction} "
            "on that trait."
        )

    rules = list(_SHARED_RULES)
    rules.append(_ANSWER_RULES[cell.item_type])
    if cell.strength_class == "weak":
        rules.insert(
            0,
            "- THIS ITEM IS DELIBERATELY WEAK. Make it mushy and low-information: "
            "double-barreled (two different things joined by 'and' or 'or'), and "
            "hedged with a vague quantifier such as 'sometimes', 'in general', "
            "'a bit', 'more or less'. A respondent should be able to agree with "
            "it for several unrelated reasons.",
        )
    else:
        rules.insert(
            0,
            "- One clause only. No 'and', no 'but', no lists, no hedging words.",
        )
    parts.append("WRITING RULES\n" + "\n".join(rules))
    parts.append(
        "TOPICS -- write exactly one statement per line, in this order:\n"
        + _topic_block(domains)
    )
    parts.append(_output_block(count, "text").rstrip())
    return SYSTEM_PROMPT, "\n\n".join(parts) + _avoid_block(avoid, retry)


def _distractor_prompt(
    cell: DraftCell, domains: Sequence[str], avoid: Sequence[str], retry: bool = False
) -> str:
    trait_areas = ", ".join(DIMENSION_LABELS[dim] for dim in DIMENSIONS)
    rules = [
        "- These are DISTRACTORS. They must carry no information about any of "
        f"these: {trait_areas}. If knowing the answer would let someone guess "
        "any of those, the item is wrong.",
        "- Pure taste or habit, with no moral, financial or social stake. "
        "Favourite season, cats or dogs, whether the person likes the smell of "
        "rain, whether they salt their food before tasting it.",
        "- First person, present tense, one clause.",
        _ANSWER_RULES[cell.item_type],
        "- Every statement must be about a different thing.",
    ]
    parts = [
        f"TASK\nWrite {len(domains)} filler survey statements.",
        "WRITING RULES\n" + "\n".join(rules),
        (
            "TOPICS -- loosely, one statement per line, in this order. Stay on the "
            "topic but keep it a matter of pure taste:\n" + _topic_block(domains)
        ),
        _output_block(len(domains), "text").rstrip(),
    ]
    return "\n\n".join(parts) + _avoid_block(avoid, retry)


def _open_prompt(
    cell: DraftCell, domains: Sequence[str], avoid: Sequence[str], retry: bool = False
) -> str:
    rules = [
        "- Second person, spoken out loud, warm and neutral.",
        "- Invites a story or a concrete example. Never answerable with yes or no.",
        '- Register: "Tell me about a time a public service let you down."',
        "- Not leading: it must not hint at which answer is interesting.",
        "- No trait words and no jargon.",
        "- One or two sentences, at most.",
    ]
    parts = [
        f"TASK\nWrite {len(domains)} conversational interview questions.",
        (
            "WHAT THE ANSWER SHOULD REVEAL (never name this in the question)\n"
            + _trait_block(cell.dims)
        ),
        "WRITING RULES\n" + "\n".join(rules),
        "TOPICS -- one question per line, in this order:\n" + _topic_block(domains),
        _output_block(len(domains), "text").rstrip(),
    ]
    return "\n\n".join(parts) + _avoid_block(avoid, retry)


# --------------------------------------------------------------------------
# Parsing whatever the model actually sends back
# --------------------------------------------------------------------------

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
_TEXT_KEYS = ("text", "item", "statement", "question", "prompt")
_DOMAIN_KEYS = ("topic_domain", "domain", "topic")


def _strip_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = _FENCE.sub("", text)
    return text.strip()


def _normalize_domain(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip().lower().replace("_", "-").replace(" ", "-")
    return cleaned if cleaned in TOPIC_DOMAINS else None


def _as_list(data: Any) -> list[Any] | None:
    if isinstance(data, list):
        return data
    if isinstance(data, Mapping):
        for key in ("items", "statements", "questions", "results", "data"):
            value = data.get(key)
            if isinstance(value, list):
                return value
    return None


def _load_json_list(text: str) -> list[Any] | None:
    try:
        return _as_list(json.loads(text))
    except (ValueError, TypeError):
        pass
    start, end = text.find("["), text.rfind("]")
    if 0 <= start < end:
        try:
            return _as_list(json.loads(text[start:end + 1]))
        except (ValueError, TypeError):
            pass
    return None


def parse_candidates(raw: str) -> list[dict[str, Any]]:
    """Pull ``{"topic_domain", "text"}`` candidates out of a model reply.

    Tolerates markdown fences, an object wrapper around the array, bare strings
    instead of objects, and -- last resort -- a plain bulleted list.
    """
    text = _strip_fences(raw or "")
    entries = _load_json_list(text)

    if entries is None:
        entries = []
        for line in text.splitlines():
            stripped = _BULLET.sub("", line).strip().strip(",").strip('"').strip()
            if len(stripped.split()) >= 3:
                entries.append(stripped)

    candidates: list[dict[str, Any]] = []
    for entry in entries:
        if isinstance(entry, str):
            body, domain = entry, None
        elif isinstance(entry, Mapping):
            body = next(
                (str(entry[key]) for key in _TEXT_KEYS if isinstance(entry.get(key), str)),
                "",
            )
            domain = next(
                (_normalize_domain(entry.get(key)) for key in _DOMAIN_KEYS if key in entry),
                None,
            )
        else:
            continue
        body = body.strip().strip('"').strip()
        if body:
            candidates.append({"topic_domain": domain, "text": body})
    return candidates


# --------------------------------------------------------------------------
# Drafting
# --------------------------------------------------------------------------


def group_slots(slots: Sequence[Slot], max_per_call: int) -> list[tuple[DraftCell, list[int]]]:
    """Slot indices grouped by design cell, split into chunks of one call each."""
    by_cell: dict[DraftCell, list[int]] = {}
    for index, slot in enumerate(slots):
        by_cell.setdefault(slot.cell, []).append(index)

    chunks: list[tuple[DraftCell, list[int]]] = []
    for cell, indices in by_cell.items():
        for start in range(0, len(indices), max_per_call):
            chunks.append((cell, indices[start:start + max_per_call]))
    return chunks


def dry_run_prompts(
    spec: BankSpec | None = None,
    seed: int = 0,
    max_per_call: int = 8,
    overdraft: int = 2,
) -> list[dict[str, Any]]:
    """Every drafting prompt this run would send, without sending any of them."""
    spec = spec or BankSpec()
    closed_slots, open_slots = build_slots(spec, seed)
    prompts: list[dict[str, Any]] = []
    for slots in (closed_slots, open_slots):
        for cell, indices in group_slots(slots, max_per_call):
            domains = [slots[i].topic_domain for i in indices]
            padded = _pad_domains(domains, overdraft)
            system, user = build_prompt(cell, padded)
            prompts.append(
                {
                    "kind": cell.kind,
                    "dims": list(cell.dims),
                    "loadings": cell.loading_dict(),
                    "type": cell.item_type,
                    "strength_class": cell.strength_class,
                    "negatively_keyed": cell.negatively_keyed,
                    "n_slots": len(indices),
                    "n_requested": len(padded),
                    "topic_domains": domains,
                    "system": system,
                    "user": user,
                }
            )
    return prompts


def _pad_domains(domains: Sequence[str], overdraft: int) -> list[str]:
    """Ask for a few spare items so a near-duplicate can be dropped.

    Capped at half the chunk, so a one-slot cell asks for two items rather than
    three: the spares are dedupe headroom, not a reason to triple the bill.
    """
    spare = min(max(0, overdraft), max(1, -(-len(domains) // 2)))
    return list(domains) + [domains[i % len(domains)] for i in range(spare)]


def draft_texts(
    client: LLMClient,
    slots: Sequence[Slot],
    *,
    temperature: float = 0.9,
    max_tokens: int = 1400,
    max_per_call: int = 8,
    overdraft: int = 2,
    max_retries: int = 2,
    seen: set[str] | None = None,
    verbose: bool = False,
) -> list[str]:
    """One item text per slot, deduplicated across the whole run."""
    seen = seen if seen is not None else set()
    texts: list[str | None] = [None] * len(slots)

    for cell, indices in group_slots(slots, max_per_call):
        pending = list(indices)
        pool: list[dict[str, Any]] = []
        accepted: list[str] = []
        attempt = 0

        while pending and attempt <= max_retries:
            wanted = [slots[i].topic_domain for i in pending]
            system, user = build_prompt(
                cell, _pad_domains(wanted, overdraft), accepted, retry=attempt > 0
            )
            response = client.generate(
                system=system, user=user, temperature=temperature, max_tokens=max_tokens
            )
            for candidate in parse_candidates(response.text):
                key = normalize_text(candidate["text"])
                if not key or key in seen:
                    continue
                seen.add(key)
                pool.append(candidate)

            # Prefer a candidate the model tagged with the topic we asked for;
            # the slot's topic domain is authoritative either way.
            for index in list(pending):
                wanted_domain = slots[index].topic_domain
                match = next((c for c in pool if c["topic_domain"] == wanted_domain), None)
                if match is None:
                    continue
                pool.remove(match)
                texts[index] = match["text"]
                accepted.append(match["text"])
                pending.remove(index)

            for index in list(pending):
                if not pool:
                    break
                candidate = pool.pop(0)
                texts[index] = candidate["text"]
                accepted.append(candidate["text"])
                pending.remove(index)

            attempt += 1

        if pending:
            raise RuntimeError(
                f"drafting fell short for the {cell.kind} cell "
                f"{'/'.join(cell.dims) or 'no-dimension'} ({cell.item_type}): "
                f"{len(pending)} of {len(indices)} slots still empty after "
                f"{attempt} call(s). Raise --max-retries or loosen the prompt."
            )
        if verbose:
            print(
                f"  {cell.kind:<10} {'/'.join(cell.dims) or '-':<9} "
                f"{cell.item_type:<8} {len(indices):>3} items",
                file=sys.stderr,
            )

    return [text or "" for text in texts]


def generate_bank(
    client: LLMClient,
    spec: BankSpec | None = None,
    seed: int = 0,
    *,
    temperature: float = 0.9,
    max_tokens: int = 1400,
    max_per_call: int = 8,
    overdraft: int = 2,
    max_retries: int = 2,
    generator: str = "",
    verbose: bool = False,
) -> Bank:
    """Build every slot, have the model write them, and assemble the bank."""
    spec = spec or BankSpec()
    closed_slots, open_slots = build_slots(spec, seed)

    seen: set[str] = set()
    shared = dict(
        temperature=temperature,
        max_tokens=max_tokens,
        max_per_call=max_per_call,
        overdraft=overdraft,
        max_retries=max_retries,
        seen=seen,
        verbose=verbose,
    )
    closed_texts = draft_texts(client, closed_slots, **shared)
    open_texts = draft_texts(client, open_slots, **shared)

    closed_items = [
        ClosedItem(
            item_id=closed_item_id(index),
            text=text,
            item_type=slot.cell.item_type,
            topic_domain=slot.topic_domain,
            loadings=slot.cell.loading_dict(),
            strength_class=slot.cell.strength_class,
            negatively_keyed=slot.cell.negatively_keyed,
            design_notes=slot.cell.design_notes,
        )
        for index, (slot, text) in enumerate(zip(closed_slots, closed_texts))
    ]
    open_the_items = [
        OpenItem(
            item_id=open_item_id(index),
            text=text,
            target_dims=slot.cell.dims,
            topic_domain=slot.topic_domain,
        )
        for index, (slot, text) in enumerate(zip(open_slots, open_texts))
    ]

    return Bank(
        closed=closed_items,
        open_items=open_the_items,
        spec=spec,
        seed=seed,
        generator=generator or type(client).__name__,
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.bank.generate",
        description="Draft the question bank: public item file plus the private design file.",
    )
    parser.add_argument(
        "--out-truth",
        metavar="DIR",
        help=(
            "directory for the private design files (loadings, keying, seed). "
            "Required unless --dry-run. Always passed in; never hardcoded."
        ),
    )
    parser.add_argument(
        "--out-public",
        metavar="DIR",
        help="directory for the item file the system side may read. Required unless --dry-run.",
    )
    parser.add_argument("--seed", type=int, default=0, help="seed for slot to topic assignment")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the drafting prompts as JSON and exit without calling any model",
    )
    parser.add_argument(
        "--prompts-out", metavar="FILE", help="with --dry-run, write the prompts here instead of stdout"
    )
    parser.add_argument("--client", default="gemini", help="LLM client name (gemini, vllm)")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=1400)
    parser.add_argument("--max-per-call", type=int, default=8, help="items requested per call")
    parser.add_argument("--overdraft", type=int, default=2, help="spare items requested per call")
    parser.add_argument("--max-retries", type=int, default=2, help="extra calls when a cell falls short")
    parser.add_argument(
        "--reduced", action="store_true", help="tiny bank with every design cell (smoke runs)"
    )
    parser.add_argument("--quiet", action="store_true", help="no per-cell progress on stderr")
    return parser


def main(argv: Sequence[str] | None = None, client: LLMClient | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    spec = BankSpec.reduced() if args.reduced else BankSpec()

    if args.dry_run:
        prompts = dry_run_prompts(
            spec=spec, seed=args.seed, max_per_call=args.max_per_call, overdraft=args.overdraft
        )
        rendered = json.dumps(prompts, indent=2, ensure_ascii=False)
        if args.prompts_out:
            path = Path(args.prompts_out)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(rendered + "\n", encoding="utf-8")
            print(f"{len(prompts)} drafting prompts written to {path}")
        else:
            print(rendered)
        return 0

    missing = [
        name
        for name, value in (("--out-truth", args.out_truth), ("--out-public", args.out_public))
        if not value
    ]
    if missing:
        print(f"error: {' and '.join(missing)} are required unless --dry-run", file=sys.stderr)
        return 2

    if client is None:
        load_dotenv_if_present()
        client = get_client(args.client)

    bank = generate_bank(
        client,
        spec=spec,
        seed=args.seed,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_per_call=args.max_per_call,
        overdraft=args.overdraft,
        max_retries=args.max_retries,
        generator=f"{type(client).__name__}:{os.environ.get('MODEL_NAME', 'unknown')}",
        verbose=not args.quiet,
    )
    written = write_bank(bank, public_dir=args.out_public, private_dir=args.out_truth)
    print(
        f"{len(bank.closed)} closed items and {len(bank.open_items)} open prompts "
        f"(seed {bank.seed})"
    )
    for label, path in written.items():
        print(f"  {label:<13} {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
