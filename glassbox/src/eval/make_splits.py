"""Build the frozen train / held-out splits (PREREGISTRATION.md section 5).

    python -m src.eval.make_splits --truth <planted dir> --out experiments/splits_v1.json

Grader-side code: it lives in ``src/eval/``, so it may read the planted design.
The directory holding that design is a command-line argument; no path to it is
spelled out here.

What this decides, once, from one seeded random stream:

1. **Held-out personas** -- 100 of the batch's 500, drawn uniformly. The
   personas are independent draws from one population, so there is nothing to
   stratify on.
2. **Held-out items** -- 50 of the 252 closed items, drawn uniformly. Open
   prompts are not split; all of them stay available.
3. **The scripted interview set** -- 15 closed items and 3 open prompts, drawn
   from the *training* items only. Frozen now because the probe strata below
   are defined relative to it. Used from Stage 3 on.
4. **Probe strata** -- each held-out item labelled near / same-domain / far by
   how close it sits to the interview set.

The draw order is part of the definition and never changes: personas, items,
interview closed set, interview open set, scripted order.

THE MANIFEST IS COMMITTED TO GIT, so it carries identifiers and counts and
nothing else. No loadings, no trait values, no item text, no domain names, and
above all no dimension codes -- naming the trait an interview item was built to
measure would put a piece of the answer key in a public file. Everything
trait-shaped is derived at use time, behind the Wall, from the planted design.
The interview items are stored in their shuffled scripted order for the same
reason: their selection order groups them by dimension, and that grouping is
itself a hint.

Selection rules, exactly as frozen by the orchestrator:

  * every dimension is covered by primary items whose designed loading has
    magnitude 1.0; seven dimensions get two items, one gets a single item and
    the seed picks which;
  * the 15 items span at most 7 of the 12 topic domains, so at least five
    domains stay untouched and the "far" probe stratum is not empty;
  * at least 4 binary and at least 8 Likert items;
  * the three open prompts have three different primary target dimensions.

The set is found by seeded rejection sampling: draw a candidate set, keep it if
every rule holds, otherwise draw again. Deterministic given the seed.

Probe strata (deterministic rule, computed here and stored per item):

  near         the item's topic domain is used by the interview's closed items
               AND the item's primary designed dimension matches the primary
               dimension of at least one closed interview item in that same
               domain;
  same-domain  the domain is touched by the interview -- by a closed item or by
               one of the three open prompts -- but the item is not near;
  far          the domain is touched by nothing in the interview: no closed
               item and no open prompt.

"Far" means untouched by the *whole* interview. An open prompt about money
makes money a domain the person has already talked about, so a held-out money
item is not a cold-start probe even when no closed question went near it. The
near rule stays defined against the closed items alone: open prompts carry
target dimensions but no designed loadings, so there is nothing to match a
held-out item's primary dimension against with the same meaning.

An item's *primary designed dimension* is the dimension carrying the largest
absolute designed loading: the single dimension of a primary or designed-weak
item, the 0.7 side of a cross-loading item. Distractors have no loadings, so
they have no primary dimension and can never be "near".
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..bank.schema import Bank, ClosedItem, DIMENSIONS, OpenItem, read_bank
from ..personas.sampler import persona_id
from . import truth_store
from .population_qa import truth_dir as truth_dir_env

#: Version tag written into the manifest.
SPLITS_SCHEMA = "glassbox.splits/1"

#: The one seed behind every choice in this file (frozen by the orchestrator).
SPLIT_SEED = 2026

#: How many personas and how many closed items are held out (20% of each).
N_PERSONA_HOLDOUT = 100
N_ITEM_HOLDOUT = 50

#: The scripted interview: 15 closed questions plus 3 open prompts.
N_INTERVIEW_CLOSED = 15
N_INTERVIEW_OPEN = 3

#: Interview constraints.
MAX_INTERVIEW_DOMAINS = 7
MIN_INTERVIEW_BINARY = 4
MIN_INTERVIEW_LIKERT = 8

#: Give up rather than spin forever if the constraints turn out unsatisfiable.
MAX_DRAW_ATTEMPTS = 200_000

#: Written by the persona factory next to the planted trait vectors.
BATCH_MANIFEST_FILENAME = "batch_manifest.json"

#: The three probe strata.
NEAR = "near"
SAME_DOMAIN = "same-domain"
FAR = "far"
STRATA = (NEAR, SAME_DOMAIN, FAR)

#: Item design classes, derived from the planted design.
PRIMARY = "primary"
CROSS = "cross"
WEAK = "weak"
DISTRACTOR = "distractor"
ITEM_CLASSES = (PRIMARY, CROSS, WEAK, DISTRACTOR)


# --------------------------------------------------------------------------
# Reading the inputs
# --------------------------------------------------------------------------


def read_persona_ids(planted_dir: str | Path) -> list[str]:
    """The batch's persona ids, in order.

    The batch manifest says how many personas were minted; the ids are that
    count in the factory's own format. When the planted trait vectors are on
    disk their ids are checked against that list, so a half-written batch is
    caught here instead of three stages later.
    """
    planted_dir = Path(planted_dir)
    manifest_path = planted_dir / BATCH_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no batch manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    n_personas = int(manifest["n"])
    if n_personas <= 0:
        raise ValueError(f"batch manifest reports {n_personas} personas")

    ids = [persona_id(i) for i in range(1, n_personas + 1)]

    with truth_dir_env(planted_dir):
        on_disk = truth_store.list_persona_ids()
    if on_disk and sorted(on_disk) != ids:
        raise ValueError(
            f"batch manifest says {n_personas} personas but the planted "
            f"directory holds {len(on_disk)} with different ids"
        )
    return ids


def load_bank(planted_dir: str | Path) -> Bank:
    """The full bank, closed items and open prompts, with the planted design."""
    return read_bank(planted_dir)


# --------------------------------------------------------------------------
# Reading the planted design of one item
# --------------------------------------------------------------------------


def item_class(item: ClosedItem) -> str:
    """Which design cell an item came from."""
    nonzero = {d: v for d, v in item.loadings.items() if v}
    if item.strength_class == "none" or not nonzero:
        return DISTRACTOR
    if item.strength_class == "weak":
        return WEAK
    if len(nonzero) > 1:
        return CROSS
    return PRIMARY


def primary_dimension(item: ClosedItem) -> str | None:
    """The dimension with the largest designed loading, or None for distractors.

    Ties break on the frozen dimension order, so the answer never depends on
    dictionary ordering. The real bank has no ties.
    """
    nonzero = [(d, v) for d, v in item.loadings.items() if v]
    if not nonzero:
        return None
    return max(nonzero, key=lambda kv: (abs(kv[1]), -DIMENSIONS.index(kv[0])))[0]


def is_interview_candidate(item: ClosedItem) -> bool:
    """True for primary items whose designed loading has magnitude 1.0."""
    if item_class(item) != PRIMARY:
        return False
    return abs(next(iter(item.loadings.values()))) == 1.0


def open_primary_dimension(item: OpenItem) -> str | None:
    """An open prompt's first target dimension, which is its main one."""
    return item.target_dims[0] if item.target_dims else None


# --------------------------------------------------------------------------
# The five draws
# --------------------------------------------------------------------------


def draw_holdout(rng: np.random.Generator, ids: Sequence[str], k: int) -> list[str]:
    """A uniform sample of ``k`` ids without replacement, returned sorted."""
    if k > len(ids):
        raise ValueError(f"cannot hold out {k} of {len(ids)}")
    picked = rng.choice(len(ids), size=k, replace=False)
    return sorted(ids[int(i)] for i in picked)


def _candidate_pools(train_items: Sequence[ClosedItem]) -> dict[str, list[ClosedItem]]:
    """Interview candidates per dimension, in bank order."""
    pools: dict[str, list[ClosedItem]] = {d: [] for d in DIMENSIONS}
    for item in train_items:
        if is_interview_candidate(item):
            pools[primary_dimension(item)].append(item)
    return pools


def draw_interview_closed(
    rng: np.random.Generator,
    train_items: Sequence[ClosedItem],
) -> tuple[list[ClosedItem], str, int]:
    """Seeded rejection sampling for the 15 scripted closed questions.

    Returns the chosen items in dimension order, which dimension got only one
    item, and how many draws it took. The caller shuffles before recording.
    """
    pools = _candidate_pools(train_items)
    short = {d: len(p) for d, p in pools.items() if len(p) < 2}
    if short:
        raise ValueError(f"too few interview candidates left after the item split: {short}")

    for attempt in range(1, MAX_DRAW_ATTEMPTS + 1):
        solo_index = int(rng.integers(0, len(DIMENSIONS)))
        chosen: list[ClosedItem] = []
        for index, dimension in enumerate(DIMENSIONS):
            pool = pools[dimension]
            wanted = 1 if index == solo_index else 2
            picked = rng.choice(len(pool), size=wanted, replace=False)
            chosen.extend(pool[int(i)] for i in picked)

        domains = {item.topic_domain for item in chosen}
        n_binary = sum(1 for item in chosen if item.item_type == "binary")
        n_likert = len(chosen) - n_binary
        if (
            len(domains) <= MAX_INTERVIEW_DOMAINS
            and n_binary >= MIN_INTERVIEW_BINARY
            and n_likert >= MIN_INTERVIEW_LIKERT
        ):
            return chosen, DIMENSIONS[solo_index], attempt

    raise RuntimeError(
        f"no interview set met the constraints in {MAX_DRAW_ATTEMPTS} draws -- "
        "the constraints and the bank disagree"
    )


def draw_interview_open(
    rng: np.random.Generator,
    open_items: Sequence[OpenItem],
) -> tuple[list[OpenItem], int]:
    """Three open prompts with three different primary target dimensions."""
    if len(open_items) < N_INTERVIEW_OPEN:
        raise ValueError(f"only {len(open_items)} open prompts to choose from")

    for attempt in range(1, MAX_DRAW_ATTEMPTS + 1):
        picked = rng.choice(len(open_items), size=N_INTERVIEW_OPEN, replace=False)
        chosen = [open_items[int(i)] for i in picked]
        dims = [open_primary_dimension(item) for item in chosen]
        if None not in dims and len(set(dims)) == N_INTERVIEW_OPEN:
            return chosen, attempt

    raise RuntimeError(
        f"no set of {N_INTERVIEW_OPEN} open prompts covered three different "
        f"dimensions in {MAX_DRAW_ATTEMPTS} draws"
    )


# --------------------------------------------------------------------------
# Probe strata
# --------------------------------------------------------------------------


def interview_domain_map(interview: Sequence[ClosedItem]) -> dict[str, set[str]]:
    """Per topic domain used by the closed interview items, the dimensions in it."""
    by_domain: dict[str, set[str]] = defaultdict(set)
    for item in interview:
        dimension = primary_dimension(item)
        if dimension is not None:
            by_domain[item.topic_domain].add(dimension)
    return dict(by_domain)


def touched_domains(
    domain_map: Mapping[str, set[str]],
    open_items: Sequence[OpenItem],
) -> set[str]:
    """Every domain the interview touches: closed questions and open prompts."""
    return set(domain_map) | {item.topic_domain for item in open_items if item.topic_domain}


def stratify(
    item: ClosedItem,
    domain_map: Mapping[str, set[str]],
    touched: set[str],
) -> str:
    """near / same-domain / far for one held-out item.

    ``domain_map`` covers the closed interview items only -- that is what the
    near rule matches dimensions against. ``touched`` covers the whole
    interview, closed and open, and is what separates far from same-domain.
    """
    if item.topic_domain not in touched:
        return FAR
    dimension = primary_dimension(item)
    in_domain = domain_map.get(item.topic_domain, set())
    if dimension is not None and dimension in in_domain:
        return NEAR
    return SAME_DOMAIN


# --------------------------------------------------------------------------
# Putting it together
# --------------------------------------------------------------------------


def build_splits(
    bank: Bank,
    persona_ids: Sequence[str],
    *,
    seed: int = SPLIT_SEED,
    created: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Make the splits.

    Returns ``(manifest, detail)``. The manifest is the committed file: ids,
    labels and counts. The detail is grader-side scratch -- dimensions,
    domains, design classes -- used for the report printed to the terminal and
    never written to disk.
    """
    if created is None:
        created = datetime.now(timezone.utc).date().isoformat()

    rng = np.random.default_rng(seed)
    closed = list(bank.closed)
    closed_ids = [item.item_id for item in closed]

    # (i) personas, (ii) items.
    persona_holdout = draw_holdout(rng, list(persona_ids), N_PERSONA_HOLDOUT)
    item_holdout = draw_holdout(rng, closed_ids, N_ITEM_HOLDOUT)
    held_out_items = set(item_holdout)
    train_items = [item for item in closed if item.item_id not in held_out_items]

    # (iii) interview closed set, (iv) interview open set.
    interview_items, solo_dimension, closed_attempts = draw_interview_closed(rng, train_items)
    interview_open, open_attempts = draw_interview_open(rng, list(bank.open_items))

    # (v) scripted order.
    order = rng.permutation(len(interview_items))
    scripted = [interview_items[int(i)] for i in order]

    # Probe strata over the held-out items. "Far" is measured against the whole
    # interview -- closed questions and open prompts; "near" against the closed
    # questions, which are the only ones with designed loadings to match.
    domain_map = interview_domain_map(interview_items)
    touched = touched_domains(domain_map, interview_open)
    by_id = {item.item_id: item for item in closed}
    strata = {
        item_id: stratify(by_id[item_id], domain_map, touched) for item_id in item_holdout
    }
    stratum_counts = {label: sum(1 for s in strata.values() if s == label) for label in STRATA}

    per_dimension = Counter(primary_dimension(item) for item in interview_items)
    n_binary = sum(1 for item in interview_items if item.item_type == "binary")
    open_dims = [open_primary_dimension(item) for item in interview_open]

    constraint_report = {
        "personas_total": len(persona_ids),
        "personas_held_out": len(persona_holdout),
        "personas_train": len(persona_ids) - len(persona_holdout),
        "closed_items_total": len(closed),
        "closed_items_held_out": len(item_holdout),
        "closed_items_train": len(train_items),
        "open_prompts_total": len(bank.open_items),
        "open_prompts_split": 0,
        "interview_closed_size": len(interview_items),
        "interview_from_training_items_only": True,
        "dimensions_total": len(DIMENSIONS),
        "dimensions_covered": len(per_dimension),
        "dimensions_with_two_items": sum(1 for c in per_dimension.values() if c == 2),
        "dimensions_with_one_item": sum(1 for c in per_dimension.values() if c == 1),
        "all_interview_items_primary_unit_loading": all(
            is_interview_candidate(item) for item in interview_items
        ),
        "interview_domains_used": len(domain_map),
        "interview_domains_allowed": MAX_INTERVIEW_DOMAINS,
        "domains_uncovered_by_closed_set": len(_all_domains(closed)) - len(domain_map),
        "interview_domains_touched_with_open": len(touched),
        "domains_untouched_by_interview": len(_all_domains(closed) - touched),
        "interview_binary_items": n_binary,
        "interview_binary_required": MIN_INTERVIEW_BINARY,
        "interview_likert_items": len(interview_items) - n_binary,
        "interview_likert_required": MIN_INTERVIEW_LIKERT,
        "closed_draw_attempts": closed_attempts,
        "interview_open_size": len(interview_open),
        "open_distinct_target_dimensions": len(set(open_dims)),
        "open_draw_attempts": open_attempts,
        "scripted_order_is_permutation": sorted(item.item_id for item in scripted)
        == sorted(item.item_id for item in interview_items),
        "held_out_items_stratified": len(strata),
    }

    manifest = {
        "schema": SPLITS_SCHEMA,
        "seed": int(seed),
        "created": created,
        "persona_holdout": persona_holdout,
        "item_holdout": item_holdout,
        "interview_closed": [item.item_id for item in scripted],
        "interview_open": [item.item_id for item in interview_open],
        "strata": strata,
        "stratum_counts": stratum_counts,
        "constraint_report": constraint_report,
    }

    detail = {
        "solo_dimension": solo_dimension,
        "interview": [
            {
                "item_id": item.item_id,
                "dimension": primary_dimension(item),
                "type": item.item_type,
                "topic_domain": item.topic_domain,
            }
            for item in scripted
        ],
        "interview_selection_order": [item.item_id for item in interview_items],
        "interview_domains": sorted(domain_map),
        "uncovered_by_closed_set": sorted(_all_domains(closed) - set(domain_map)),
        "open_domains": sorted({item.topic_domain for item in interview_open}),
        "far_domains": sorted(_all_domains(closed) - touched),
        "interview_open": [
            {
                "item_id": item.item_id,
                "dimension": open_primary_dimension(item),
                "target_dims": list(item.target_dims),
                "topic_domain": item.topic_domain,
            }
            for item in interview_open
        ],
        "holdout_by_class_and_stratum": _class_stratum_table(
            [by_id[item_id] for item_id in item_holdout], strata
        ),
        "holdout_by_dimension": _dimension_table(
            [by_id[item_id] for item_id in item_holdout], strata
        ),
        "holdout_domains": sorted({by_id[i].topic_domain for i in item_holdout}),
    }
    return manifest, detail


def _all_domains(items: Iterable[ClosedItem]) -> set[str]:
    return {item.topic_domain for item in items}


def _class_stratum_table(
    items: Sequence[ClosedItem],
    strata: Mapping[str, str],
) -> dict[str, dict[str, int]]:
    """Held-out item counts: design class x probe stratum."""
    table = {cls: {label: 0 for label in STRATA} for cls in ITEM_CLASSES}
    for item in items:
        table[item_class(item)][strata[item.item_id]] += 1
    return table


def _dimension_table(
    items: Sequence[ClosedItem],
    strata: Mapping[str, str],
) -> dict[str, dict[str, int]]:
    """Held-out item counts per primary dimension, split by stratum."""
    table = {d: {label: 0 for label in STRATA} for d in DIMENSIONS}
    table["(none)"] = {label: 0 for label in STRATA}
    for item in items:
        dimension = primary_dimension(item) or "(none)"
        table[dimension][strata[item.item_id]] += 1
    return table


def render_report(manifest: Mapping[str, Any], detail: Mapping[str, Any]) -> str:
    """The grader-side terminal report. Names dimensions; never written to disk."""
    lines: list[str] = []
    report = manifest["constraint_report"]
    lines.append(
        f"seed {manifest['seed']}   personas held out "
        f"{report['personas_held_out']}/{report['personas_total']}   "
        f"closed items held out {report['closed_items_held_out']}/"
        f"{report['closed_items_total']}"
    )
    lines.append("")
    lines.append(f"scripted interview ({report['interview_closed_size']} closed, in asked order)")
    for position, row in enumerate(detail["interview"], start=1):
        lines.append(
            f"  {position:2d}. {row['item_id']}  {row['dimension']}  "
            f"{row['type']:<7}  {row['topic_domain']}"
        )
    lines.append(
        f"  one item on {detail['solo_dimension']}, two on each of the other seven; "
        f"{report['interview_binary_items']} binary / "
        f"{report['interview_likert_items']} likert; "
        f"{report['closed_draw_attempts']} draws to satisfy the constraints"
    )
    lines.append(f"  domains used ({len(detail['interview_domains'])}): "
                 + ", ".join(detail["interview_domains"]))
    lines.append(f"  domains no closed question touches ({len(detail['uncovered_by_closed_set'])}): "
                 + ", ".join(detail["uncovered_by_closed_set"]))
    lines.append("")
    lines.append("open prompts")
    for row in detail["interview_open"]:
        lines.append(
            f"  {row['item_id']}  {row['dimension']}  "
            f"(targets {', '.join(row['target_dims'])})  {row['topic_domain']}"
        )
    lines.append(f"  domains they add ({len(detail['open_domains'])}): "
                 + ", ".join(detail["open_domains"]))
    lines.append("")
    counts = manifest["stratum_counts"]
    lines.append(
        "probe strata over the held-out items: "
        + "   ".join(f"{label} {counts[label]}" for label in STRATA)
    )
    lines.append(
        f"  far = the {len(detail['far_domains'])} domains the whole interview leaves "
        "untouched: " + ", ".join(detail["far_domains"])
    )
    lines.append("")
    lines.append("held-out composition (design class x stratum)")
    lines.append("  " + "class".ljust(12) + "".join(label.rjust(14) for label in STRATA) + "total".rjust(9))
    for cls, row in detail["holdout_by_class_and_stratum"].items():
        total = sum(row.values())
        lines.append(
            "  " + cls.ljust(12)
            + "".join(str(row[label]).rjust(14) for label in STRATA)
            + str(total).rjust(9)
        )
    lines.append("")
    lines.append("held-out composition (primary dimension x stratum)")
    lines.append("  " + "dim".ljust(12) + "".join(label.rjust(14) for label in STRATA) + "total".rjust(9))
    for dimension, row in detail["holdout_by_dimension"].items():
        total = sum(row.values())
        if total == 0:
            continue
        lines.append(
            "  " + dimension.ljust(12)
            + "".join(str(row[label]).rjust(14) for label in STRATA)
            + str(total).rjust(9)
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.eval.make_splits",
        description="Freeze the train / held-out splits and the scripted interview set.",
    )
    parser.add_argument("--truth", type=Path, required=True,
                        help="the batch's planted directory: bank design and batch manifest")
    parser.add_argument("--out", type=Path, required=True,
                        help="where to write the splits manifest")
    parser.add_argument("--seed", type=int, default=SPLIT_SEED,
                        help=f"split seed (frozen at {SPLIT_SEED})")
    parser.add_argument("--created", default=None,
                        help="date stamp for the manifest (default: today, UTC)")
    args = parser.parse_args(argv)

    bank = load_bank(args.truth)
    persona_ids = read_persona_ids(args.truth)
    manifest, detail = build_splits(
        bank, persona_ids, seed=args.seed, created=args.created
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(render_report(manifest, detail))
    print("")
    print(f"splits -> {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
