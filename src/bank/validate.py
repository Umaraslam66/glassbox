"""Bank-level checks, run after generation and before the bank is used.

Nothing here calls a model. It reads the files ``generate.py`` wrote and
answers one question: is this bank the bank the design asked for?

  * every design cell has exactly the number of items the spec says;
  * loading magnitudes and signs match the cell (primary +/-1.0, cross 0.7/0.5,
    designed-weak 0.25, distractors exactly zero);
  * the cross-loading items cover enough distinct dimension pairs, including
    the two the population correlation matrix sets to zero;
  * no duplicate item text anywhere, closed or open;
  * every dimension is touched by at least a full dimension's worth of items;
  * item ids are unique and sequential;
  * every post-freeze amendment is fully written up -- which item, which field,
    the old and new value, the date, who authorised it and the log entry that
    justifies it -- and names an item that exists;
  * the public file carries none of the planted design -- no loadings, no
    strength class, no keying flag, no target dimensions, no amendment log.

Run:
    python -m src.bank.validate --truth <dir> --public <dir>
Exit code 0 means the bank is usable; 1 means it is not.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.bank.schema import (
    AMENDMENT_KEYS,
    DIMENSIONS,
    PRIVATE_ONLY_KEYS,
    PUBLIC_FILENAME,
    TOPIC_DOMAINS,
    Bank,
    BankSpec,
    ClosedItem,
    closed_item_id,
    collect_keys,
    dimension_pairs,
    normalize_text,
    open_item_id,
    read_bank,
    read_json,
    unique_texts,
)

TOLERANCE = 1e-9

#: The two dimension pairs cross-loading items must cover, whatever else they do.
REQUIRED_PAIRS: tuple[tuple[str, ...], ...] = (
    dimension_pairs(("TRU", "RSK")),
    dimension_pairs(("ENV", "RSK")),
)


def item_kind(item: ClosedItem, spec: BankSpec) -> str:
    """Which design cell an item claims to belong to, read off its loadings."""
    if item.strength_class == "none":
        return "distractor"
    if item.strength_class == "weak":
        return "weak"
    return {1: "primary", 2: "cross"}.get(len(item.dims), "malformed")


def design_table(bank: Bank, spec: BankSpec | None = None) -> dict[str, Any]:
    """Counts by design cell -- the table to eyeball after a generation run."""
    spec = spec or bank.spec
    by_cell: Counter[str] = Counter()
    by_dimension: Counter[str] = Counter()
    by_domain: Counter[str] = Counter()
    pairs: set[tuple[str, ...]] = set()

    for item in bank.closed:
        kind = item_kind(item, spec)
        keying = "negative" if item.negatively_keyed else "positive"
        by_cell[f"{kind}/{item.item_type}/{keying}"] += 1
        by_domain[item.topic_domain] += 1
        for dim in item.dims:
            by_dimension[dim] += 1
        if kind == "cross":
            pairs.add(dimension_pairs(item.dims))

    for item in bank.open_items:
        by_domain[item.topic_domain] += 1
        for dim in item.target_dims:
            by_dimension[dim] += 1

    return {
        "n_closed": len(bank.closed),
        "n_open": len(bank.open_items),
        "by_cell": dict(sorted(by_cell.items())),
        "items_touching_dimension": {dim: by_dimension.get(dim, 0) for dim in DIMENSIONS},
        "items_per_topic_domain": {d: by_domain.get(d, 0) for d in TOPIC_DOMAINS},
        "distinct_cross_pairs": len(pairs),
        "cross_pairs": sorted("-".join(pair) for pair in pairs),
    }


def format_design_table(table: Mapping[str, Any]) -> str:
    """The design table as plain text."""
    lines = [
        f"closed items: {table['n_closed']}    open prompts: {table['n_open']}",
        "",
        f"{'kind':<11}{'type':<9}{'keying':<10}count",
    ]
    for cell, count in table["by_cell"].items():
        kind, item_type, keying = cell.split("/")
        lines.append(f"{kind:<11}{item_type:<9}{keying:<10}{count}")
    lines.append("")
    lines.append("items touching each dimension:")
    lines.append(
        "  " + "  ".join(f"{dim}={n}" for dim, n in table["items_touching_dimension"].items())
    )
    lines.append("items per topic domain:")
    lines.append(
        "  " + "  ".join(f"{d}={n}" for d, n in table["items_per_topic_domain"].items())
    )
    lines.append(f"distinct cross-loading dimension pairs: {table['distinct_cross_pairs']}")
    return "\n".join(lines)


def _magnitudes(item: ClosedItem) -> list[float]:
    return sorted(abs(value) for value in item.loadings.values() if value != 0.0)


def _check_ids(bank: Bank, problems: list[str]) -> None:
    expected = [closed_item_id(i) for i in range(len(bank.closed))]
    actual = [item.item_id for item in bank.closed]
    if actual != expected:
        duplicates = [i for i, n in Counter(actual).items() if n > 1]
        if duplicates:
            problems.append(f"duplicate closed item ids: {sorted(duplicates)[:5]}")
        else:
            first = next((a for a, e in zip(actual, expected) if a != e), "?")
            problems.append(f"closed item ids are not sequential q001.. (first bad id: {first})")

    expected_open = [open_item_id(i) for i in range(len(bank.open_items))]
    actual_open = [item.item_id for item in bank.open_items]
    if actual_open != expected_open:
        problems.append("open item ids are not sequential oe01..")


def _check_counts(bank: Bank, spec: BankSpec, problems: list[str]) -> None:
    if len(bank.closed) != spec.n_closed:
        problems.append(f"expected {spec.n_closed} closed items, found {len(bank.closed)}")
    if len(bank.open_items) != spec.n_open:
        problems.append(f"expected {spec.n_open} open prompts, found {len(bank.open_items)}")

    kinds = Counter(item_kind(item, spec) for item in bank.closed)
    for kind, expected in (
        ("primary", spec.n_primary),
        ("cross", spec.n_cross),
        ("weak", spec.n_weak),
        ("distractor", spec.n_distractor),
    ):
        if kinds.get(kind, 0) != expected:
            problems.append(f"expected {expected} {kind} items, found {kinds.get(kind, 0)}")
    if kinds.get("malformed"):
        problems.append(f"{kinds['malformed']} items load on zero or more than two dimensions")


def _check_primary(bank: Bank, spec: BankSpec, problems: list[str]) -> None:
    per_dim: dict[str, Counter[tuple[str, bool]]] = {dim: Counter() for dim in DIMENSIONS}

    for item in bank.closed:
        if item_kind(item, spec) != "primary":
            continue
        dim = item.dims[0]
        per_dim[dim][(item.item_type, item.negatively_keyed)] += 1
        loading = item.loadings[dim]
        if abs(abs(loading) - spec.strong_loading) > TOLERANCE:
            problems.append(
                f"{item.item_id}: primary loading magnitude {abs(loading)} "
                f"(expected {spec.strong_loading})"
            )
        if (loading < 0) != item.negatively_keyed:
            problems.append(f"{item.item_id}: keying flag disagrees with the loading sign")

    for dim in DIMENSIONS:
        counts = per_dim[dim]
        wanted = {
            ("likert5", False): spec.likert_per_dim - spec.neg_likert_per_dim,
            ("likert5", True): spec.neg_likert_per_dim,
            ("binary", False): spec.binary_per_dim - spec.neg_binary_per_dim,
            ("binary", True): spec.neg_binary_per_dim,
        }
        for key, expected in wanted.items():
            found = counts.get(key, 0)
            if found != expected:
                keying = "negatively keyed" if key[1] else "positively keyed"
                problems.append(
                    f"{dim}: expected {expected} {keying} {key[0]} primary items, found {found}"
                )


def _check_cross(bank: Bank, spec: BankSpec, problems: list[str]) -> None:
    pairs: set[tuple[str, ...]] = set()
    expected_magnitudes = sorted((spec.cross_secondary_loading, spec.cross_primary_loading))

    for item in bank.closed:
        if item_kind(item, spec) != "cross":
            continue
        pairs.add(dimension_pairs(item.dims))
        found = _magnitudes(item)
        if len(found) != 2 or any(
            abs(a - b) > TOLERANCE for a, b in zip(found, expected_magnitudes)
        ):
            problems.append(
                f"{item.item_id}: cross loadings {found} (expected {expected_magnitudes})"
            )
            continue
        main_dim = max(item.dims, key=lambda d: abs(item.loadings[d]))
        if (item.loadings[main_dim] < 0) != item.negatively_keyed:
            problems.append(
                f"{item.item_id}: keying flag disagrees with the main loading's sign"
            )

    if spec.n_cross:
        needed = min(spec.min_cross_pairs, spec.n_cross)
        if len(pairs) < needed:
            problems.append(
                f"cross-loading items cover {len(pairs)} distinct dimension pairs, need {needed}"
            )
        for required in REQUIRED_PAIRS:
            if required not in pairs:
                problems.append(
                    f"no cross-loading item on the frozen zero-correlation pair "
                    f"{'-'.join(required)}"
                )


def _check_weak_and_distractors(bank: Bank, spec: BankSpec, problems: list[str]) -> None:
    for item in bank.closed:
        kind = item_kind(item, spec)
        if kind == "weak":
            if len(item.dims) != 1:
                problems.append(f"{item.item_id}: weak item touches {len(item.dims)} dimensions")
                continue
            magnitude = abs(item.loadings[item.dims[0]])
            if abs(magnitude - spec.weak_loading) > TOLERANCE:
                problems.append(
                    f"{item.item_id}: weak loading magnitude {magnitude} "
                    f"(expected {spec.weak_loading})"
                )
        elif kind == "distractor":
            if any(value != 0.0 for value in item.loading_vector()):
                problems.append(f"{item.item_id}: distractor has a non-zero loading")
            if item.negatively_keyed:
                problems.append(f"{item.item_id}: distractor is marked negatively keyed")


def _check_texts_and_domains(bank: Bank, spec: BankSpec, problems: list[str]) -> None:
    all_items: list[Any] = list(bank.closed) + list(bank.open_items)

    for item in all_items:
        if not item.text.strip():
            problems.append(f"{item.item_id}: empty text")

    for key, ids in unique_texts(all_items).items():
        if len(ids) > 1:
            problems.append(f"duplicate item text shared by {ids}: {key[:60]!r}")

    for item in all_items:
        if item.topic_domain not in TOPIC_DOMAINS:
            problems.append(f"{item.item_id}: unknown topic domain {item.topic_domain!r}")

    if spec.require_all_domains:
        used = {item.topic_domain for item in all_items}
        missing = [domain for domain in TOPIC_DOMAINS if domain not in used]
        if missing:
            problems.append(f"topic domains never used: {missing}")


def _check_dimension_coverage(bank: Bank, spec: BankSpec, problems: list[str]) -> None:
    touched: Counter[str] = Counter()
    for item in bank.closed:
        for dim in item.dims:
            touched[dim] += 1
    for dim in DIMENSIONS:
        if touched[dim] < spec.per_dim:
            problems.append(
                f"{dim}: only {touched[dim]} closed items touch it, need at least {spec.per_dim}"
            )


def _check_open_items(bank: Bank, spec: BankSpec, problems: list[str]) -> None:
    covered: Counter[str] = Counter()
    for item in bank.open_items:
        if not 1 <= len(item.target_dims) <= 2:
            problems.append(
                f"{item.item_id}: {len(item.target_dims)} target dimensions (expected 1 or 2)"
            )
        for dim in item.target_dims:
            covered[dim] += 1
    if spec.n_open:
        missing = [dim for dim in DIMENSIONS if covered[dim] == 0]
        if missing:
            problems.append(f"no open-ended prompt targets: {missing}")


def _check_amendments(bank: Bank, problems: list[str]) -> None:
    """A post-freeze edit is only legitimate if it is written up in full.

    The bank is final once planted, so anything changed afterwards has to say
    what it changed, to which item, when, on whose authority and against which
    entry in the project log. This checks the paperwork is there; it does not
    and cannot judge whether the reason is a good one.
    """
    known_ids = {item.item_id for item in bank.closed}
    for position, entry in enumerate(bank.amendments):
        label = f"amendment {position + 1}"
        if not isinstance(entry, Mapping):
            problems.append(f"{label}: not a mapping")
            continue
        missing = [key for key in AMENDMENT_KEYS if key not in entry]
        if missing:
            problems.append(f"{label}: missing field(s) {missing}")
        blank = [
            key
            for key in ("item_id", "field", "date", "authorized_by", "reason", "log_citation")
            if key in entry and not str(entry[key]).strip()
        ]
        if blank:
            problems.append(f"{label}: empty field(s) {blank}")
        item_id = entry.get("item_id")
        if item_id is not None and item_id not in known_ids:
            problems.append(f"{label}: names item {item_id!r}, which is not in the bank")
        if "old_value" in entry and "new_value" in entry and entry["old_value"] == entry["new_value"]:
            problems.append(f"{label}: old and new value are the same")


def check_public_payload(
    public_payload: Mapping[str, Any], bank: Bank, problems: list[str]
) -> None:
    """The public file must carry the wording and nothing about the design."""
    leaked = sorted(collect_keys(public_payload) & set(PRIVATE_ONLY_KEYS))
    if leaked:
        problems.append(f"public file exposes planted-design keys: {leaked}")

    rendered = json.dumps(public_payload)
    for key in PRIVATE_ONLY_KEYS:
        if f'"{key}"' in rendered:
            problems.append(f"public file mentions {key!r}")

    public_items = public_payload.get("items") or []
    if len(public_items) != len(bank.closed):
        problems.append(
            f"public file has {len(public_items)} closed items, private file has {len(bank.closed)}"
        )
    public_open = public_payload.get("open_ended") or []
    if len(public_open) != len(bank.open_items):
        problems.append(
            f"public file has {len(public_open)} open prompts, "
            f"private file has {len(bank.open_items)}"
        )

    private_by_id = {item.item_id: item for item in bank.closed}
    for row in public_items:
        item = private_by_id.get(row.get("item_id"))
        if item is None:
            problems.append(f"public item {row.get('item_id')!r} is not in the private file")
            continue
        if normalize_text(row.get("text", "")) != normalize_text(item.text):
            problems.append(f"{item.item_id}: public and private text differ")
        if row.get("type") != item.item_type:
            problems.append(f"{item.item_id}: public and private item type differ")
        if row.get("topic_domain") != item.topic_domain:
            problems.append(f"{item.item_id}: public and private topic domain differ")
        if not row.get("options"):
            problems.append(f"{item.item_id}: public item has no answer options")

    private_open_by_id = {item.item_id: item for item in bank.open_items}
    for row in public_open:
        item = private_open_by_id.get(row.get("item_id"))
        if item is None:
            problems.append(f"public open prompt {row.get('item_id')!r} is not in the private file")
        elif normalize_text(row.get("text", "")) != normalize_text(item.text):
            problems.append(f"{item.item_id}: public and private text differ")


def validate_bank(
    bank: Bank,
    spec: BankSpec | None = None,
    public_payload: Mapping[str, Any] | None = None,
) -> list[str]:
    """Every way this bank fails its design. Empty list means it passes."""
    spec = spec if spec is not None else bank.spec
    problems: list[str] = []
    _check_ids(bank, problems)
    _check_counts(bank, spec, problems)
    _check_primary(bank, spec, problems)
    _check_cross(bank, spec, problems)
    _check_weak_and_distractors(bank, spec, problems)
    _check_texts_and_domains(bank, spec, problems)
    _check_dimension_coverage(bank, spec, problems)
    _check_open_items(bank, spec, problems)
    _check_amendments(bank, problems)
    if public_payload is not None:
        check_public_payload(public_payload, bank, problems)
    return problems


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.bank.validate",
        description="Check a generated question bank against its design.",
    )
    parser.add_argument(
        "--truth",
        required=True,
        metavar="DIR",
        help="directory holding the private design files. Always passed in; never hardcoded.",
    )
    parser.add_argument(
        "--public", metavar="DIR", help="directory holding the public item file, if it should be checked"
    )
    parser.add_argument("--table", action="store_true", help="print the design table as well")
    parser.add_argument("--json", action="store_true", help="print the design table as JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    bank = read_bank(args.truth)
    public_payload = read_json(Path(args.public) / PUBLIC_FILENAME) if args.public else None

    problems = validate_bank(bank, public_payload=public_payload)
    table = design_table(bank)

    if args.json:
        print(json.dumps({"problems": problems, "table": table}, indent=2))
    elif args.table:
        print(format_design_table(table))
        print()

    if problems:
        print(f"FAIL: {len(problems)} problem(s)", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print(f"OK: {len(bank.closed)} closed items and {len(bank.open_items)} open prompts")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
