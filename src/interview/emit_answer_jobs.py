"""Write the batch file of answer prompts for a Leonardo job.

    python -m src.interview.emit_answer_jobs \\
        --cards-dir <dir> --noise-dir <dir> --public-bank <bank_items.json> \\
        --out <run dir> [--items <ids file>] [--round main|retest] \\
        [--session-seed N] [--limit-personas N] [--bank-design <file>]

Output, all inside the run directory:

    prompts.jsonl          one line per (persona, item): {"pid", "item_id",
                           "round", "system", "user"}
    retest_selection.json  retest rounds only: which 30 items each persona is
                           re-asked. Written once and reused ever after.

NEVER RE-QUERY WHAT IS STORED. Before emitting anything the run directory's
``answers.jsonl`` is read and every (pid, item_id, round) already in it is
skipped. Re-running after a half-finished job emits only the gap; re-running
after a complete job emits nothing.

WALL RULE: the card directory, the noise directory and the item-design file all
arrive as command-line arguments and have no defaults. No planted-truth path is
written down in this file. The run directory holds persona cards inside the
prompts, so it is a persona-side artifact: keep it behind the Wall with the
rest of the batch, never in a system-side folder and never in git.

The retest selection needs to know which dimension an item was designed to
measure, and that lives in the bank's design file, not in the public bank. So
``--round retest`` requires ``--bank-design``. The design is used only to
choose which items to re-ask -- it never reaches a prompt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..bank.schema import DIMENSIONS
from .responder_prompts import CLOSED_ROUNDS, build_answer_prompt, session_id

#: Items re-asked per persona in the retest round. Frozen in
#: PREREGISTRATION.md section 5: "re-ask 30 fixed items per persona".
RETEST_ITEMS = 30

#: Primary items guaranteed per dimension in a retest selection.
RETEST_MIN_PER_DIM = 2

#: The non-primary design classes the remainder is spread over, in the order
#: used to break ties when the proportional split lands on a fraction.
FILL_CLASSES: tuple[str, ...] = ("cross", "weak", "distractor")

PROMPTS_FILENAME = "prompts.jsonl"
ANSWERS_FILENAME = "answers.jsonl"
SELECTION_FILENAME = "retest_selection.json"


# --------------------------------------------------------------------------
# reading the pieces
# --------------------------------------------------------------------------


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Every JSON object in a JSONL file; blank lines skipped."""
    path = Path(path)
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_public_items(public_bank: str | Path) -> dict[str, dict[str, Any]]:
    """The public item records, keyed by item id, in bank order."""
    payload = json.loads(Path(public_bank).read_text(encoding="utf-8"))
    return {str(row["item_id"]): dict(row) for row in payload.get("items", [])}


def load_personas(
    cards_dir: str | Path,
    noise_dir: str | Path,
    limit: int | None = None,
) -> list[tuple[str, dict[str, Any], float]]:
    """Personas that have both a card and a wobble on file, sorted by pid.

    A persona missing either one is skipped: without a card there is nobody to
    role-play, and without a wobble the noise paragraph cannot be built.
    """
    cards_dir = Path(cards_dir)
    noise_dir = Path(noise_dir)

    people: list[tuple[str, dict[str, Any], float]] = []
    for card_path in sorted(cards_dir.glob("*.json")):
        pid = card_path.stem
        noise_path = noise_dir / f"{pid}.json"
        if not noise_path.is_file():
            continue
        card = json.loads(card_path.read_text(encoding="utf-8"))
        noise = json.loads(noise_path.read_text(encoding="utf-8"))
        wobble = noise.get("wobble") if isinstance(noise, Mapping) else noise
        if wobble is None:
            continue
        people.append((pid, card, float(wobble)))

    if limit is not None:
        people = people[:limit]
    return people


def read_item_ids(path: str | Path) -> list[str]:
    """Item ids from a plain text file, one per line; ``#`` comments allowed."""
    ids: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        token = line.split("#", 1)[0].strip()
        if token:
            ids.append(token)
    return ids


def answered_keys(run_dir: str | Path) -> set[tuple[str, str, str]]:
    """(pid, item_id, round) triples already answered in this run directory."""
    keys: set[tuple[str, str, str]] = set()
    for row in read_jsonl(Path(run_dir) / ANSWERS_FILENAME):
        keys.add((str(row.get("pid")), str(row.get("item_id")), str(row.get("round") or "main")))
    return keys


# --------------------------------------------------------------------------
# the retest selection
# --------------------------------------------------------------------------


def classify_items(bank_design: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Design class and dimensions per item, from the bank's design file.

    Four classes, matching how the bank is generated: ``primary`` (one
    dimension at full strength), ``cross`` (two dimensions, 0.7 and 0.5),
    ``weak`` (one dimension, deliberately mushy) and ``distractor`` (no
    designed loading at all).
    """
    classes: dict[str, dict[str, Any]] = {}
    for row in bank_design.get("items", []):
        loadings = {str(k): float(v) for k, v in (row.get("loadings") or {}).items()}
        dims = tuple(d for d in DIMENSIONS if loadings.get(d, 0.0) != 0.0)
        strength = str(row.get("strength_class", "strong"))

        if strength == "none" or not dims:
            kind = "distractor"
        elif strength == "weak":
            kind = "weak"
        elif len(dims) > 1:
            kind = "cross"
        else:
            kind = "primary"

        classes[str(row["item_id"])] = {"kind": kind, "dims": dims, "loadings": loadings}
    return classes


def _persona_rng(pid: str, seed: int) -> np.random.Generator:
    """A generator that depends on the persona and the seed, and nothing else."""
    digest = hashlib.sha256(f"{seed}|{pid}".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def _take(rng: np.random.Generator, options: Sequence[str], count: int) -> list[str]:
    """``count`` distinct picks from ``options``, or all of them if fewer."""
    if count <= 0 or not options:
        return []
    if count >= len(options):
        return list(options)
    picked = rng.choice(np.array(options, dtype=object), size=count, replace=False)
    return [str(item) for item in picked]


def proportional_quotas(sizes: Mapping[str, int], total: int) -> dict[str, int]:
    """Split ``total`` across classes in proportion to how many items each has.

    Largest-remainder rounding, ties broken by :data:`FILL_CLASSES` order, and
    no class is ever given more items than it actually has. Whatever cannot be
    placed comes back as a shortfall for the caller to fill elsewhere.
    """
    available = {name: max(0, int(sizes.get(name, 0))) for name in FILL_CLASSES}
    pool = sum(available.values())
    if total <= 0 or pool == 0:
        return {name: 0 for name in FILL_CLASSES}

    total = min(total, pool)
    exact = {name: total * available[name] / pool for name in FILL_CLASSES}
    quotas = {name: int(exact[name]) for name in FILL_CLASSES}

    leftover = total - sum(quotas.values())
    order = sorted(
        FILL_CLASSES,
        key=lambda name: (-(exact[name] - quotas[name]), FILL_CLASSES.index(name)),
    )
    index = 0
    while leftover > 0:
        name = order[index % len(order)]
        if quotas[name] < available[name]:
            quotas[name] += 1
            leftover -= 1
        index += 1
        if index > 4 * len(order) + total:  # pragma: no cover -- pool >= total
            break
    return quotas


def select_retest_items(
    pool: Sequence[str],
    classes: Mapping[str, Mapping[str, Any]],
    pid: str,
    seed: int,
    n_items: int = RETEST_ITEMS,
) -> list[str]:
    """The items one persona is re-asked, stratified and seeded.

    At least :data:`RETEST_MIN_PER_DIM` primary items from each of the 8
    dimensions, then the remainder spread over the cross / weak / distractor
    classes in proportion to how many of each the bank holds. If a stratum is
    too thin to fill its quota (a reduced bank, or a restricted item pool), the
    gap is filled from whatever is left, so the count still comes out at
    ``n_items`` whenever the pool is big enough.
    """
    rng = _persona_rng(pid, seed)
    in_pool = [item_id for item_id in pool if item_id in classes]

    primary_by_dim: dict[str, list[str]] = {dim: [] for dim in DIMENSIONS}
    by_class: dict[str, list[str]] = {name: [] for name in FILL_CLASSES}
    for item_id in in_pool:
        entry = classes[item_id]
        if entry["kind"] == "primary":
            dims = entry["dims"]
            if dims:
                primary_by_dim[dims[0]].append(item_id)
        else:
            by_class[str(entry["kind"])].append(item_id)

    chosen: list[str] = []
    taken: set[str] = set()

    for dim in DIMENSIONS:
        picks = _take(rng, primary_by_dim[dim], RETEST_MIN_PER_DIM)
        chosen.extend(picks)
        taken.update(picks)

    remainder = n_items - len(chosen)
    quotas = proportional_quotas({name: len(by_class[name]) for name in FILL_CLASSES}, remainder)
    for name in FILL_CLASSES:
        picks = _take(rng, [i for i in by_class[name] if i not in taken], quotas[name])
        chosen.extend(picks)
        taken.update(picks)

    if len(chosen) < n_items:
        rest = [item_id for item_id in in_pool if item_id not in taken]
        picks = _take(rng, rest, n_items - len(chosen))
        chosen.extend(picks)
        taken.update(picks)

    order = {item_id: index for index, item_id in enumerate(in_pool)}
    return sorted(chosen[:n_items], key=lambda item_id: order[item_id])


def load_or_build_selection(
    run_dir: str | Path,
    pids: Sequence[str],
    pool: Sequence[str],
    classes: Mapping[str, Mapping[str, Any]],
    seed: int,
    n_items: int = RETEST_ITEMS,
) -> dict[str, list[str]]:
    """The retest selection for this run: read it if it exists, else write it.

    "Fixed items" in the pre-registration means fixed on disk. Once
    ``retest_selection.json`` is there it is authoritative, even if the pool or
    the seed changes; personas added later are appended to it.
    """
    run_dir = Path(run_dir)
    path = run_dir / SELECTION_FILENAME

    stored: dict[str, Any] = {}
    if path.is_file():
        stored = json.loads(path.read_text(encoding="utf-8"))
    selection: dict[str, list[str]] = {
        str(pid): [str(item) for item in items]
        for pid, items in (stored.get("selection") or {}).items()
    }

    added = False
    for pid in pids:
        if pid not in selection:
            selection[pid] = select_retest_items(pool, classes, pid, seed, n_items)
            added = True

    if added or not path.is_file():
        run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "seed": int(stored.get("seed", seed)),
            "n_items": int(stored.get("n_items", n_items)),
            "selection": {pid: selection[pid] for pid in sorted(selection)},
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return selection


# --------------------------------------------------------------------------
# emitting
# --------------------------------------------------------------------------


def emit_jobs(
    people: Sequence[tuple[str, Mapping[str, Any], float]],
    items: Mapping[str, Mapping[str, Any]],
    per_persona: Mapping[str, Sequence[str]],
    run_dir: str | Path,
    round_name: str,
    session_seed: int,
) -> dict[str, Any]:
    """Write ``prompts.jsonl`` for everything not already answered."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    done = answered_keys(run_dir)

    lines: list[str] = []
    skipped = 0
    seen: set[tuple[str, str, str]] = set()

    for pid, card, wobble in people:
        session = session_id(pid, round_name, session_seed)
        for item_id in per_persona.get(pid, ()):
            key = (pid, item_id, round_name)
            if key in done or key in seen:
                skipped += 1
                continue
            if item_id not in items:
                raise ValueError(
                    f"item {item_id!r} is asked of persona {pid!r} but is not in "
                    "the public bank -- the run directory and the bank do not match"
                )
            seen.add(key)
            prompt = build_answer_prompt(card, wobble, items[item_id], session)
            lines.append(
                json.dumps(
                    {
                        "pid": pid,
                        "item_id": item_id,
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
        "prompts_path": path,
        "n_personas": len(people),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.interview.emit_answer_jobs",
        description="Build the batch of answer prompts for one responder round.",
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
        help="the public bank file (bank_items.json): item ids, texts and types",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="run directory; prompts.jsonl is written here and answers.jsonl is read from here",
    )
    parser.add_argument(
        "--items",
        type=Path,
        default=None,
        help="optional file of item ids (one per line) to restrict the round to",
    )
    parser.add_argument(
        "--round",
        choices=CLOSED_ROUNDS,
        default="main",
        help=(
            "main pass over the bank, or the fixed retest re-ask (default "
            "main). These are the only two: an interview round is never "
            "emitted here, because a closed interview answer is drawn from "
            "what these rounds recorded, not asked again (consistency rule)."
        ),
    )
    parser.add_argument(
        "--session-seed",
        type=int,
        default=0,
        help="seed for the session tags and the retest draw (default 0)",
    )
    parser.add_argument(
        "--limit-personas",
        type=int,
        default=None,
        help="only the first N personas, by pid; for smoke tests",
    )
    parser.add_argument(
        "--bank-design",
        type=Path,
        default=None,
        help=(
            "the bank's design file (bank_truth.json). Required for --round "
            "retest, which stratifies the re-asked items by designed "
            "dimension. Used only to pick items; never enters a prompt."
        ),
    )
    args = parser.parse_args(argv)

    items = load_public_items(args.public_bank)
    if not items:
        parser.error(f"no items in {args.public_bank}")

    pool = list(items)
    if args.items is not None:
        wanted = read_item_ids(args.items)
        missing = [item_id for item_id in wanted if item_id not in items]
        if missing:
            parser.error(f"{len(missing)} item ids are not in the bank, first: {missing[0]}")
        pool = wanted

    people = load_personas(args.cards_dir, args.noise_dir, args.limit_personas)
    if not people:
        parser.error(f"no persona has both a card in {args.cards_dir} and a wobble in {args.noise_dir}")
    pids = [pid for pid, _, _ in people]

    if args.round == "retest":
        if args.bank_design is None:
            parser.error(
                "--round retest needs --bank-design: the 30 re-asked items are "
                "stratified by the dimension each item was designed to measure, "
                "which is not in the public bank"
            )
        classes = classify_items(json.loads(Path(args.bank_design).read_text(encoding="utf-8")))
        per_persona: dict[str, Sequence[str]] = load_or_build_selection(
            args.out, pids, pool, classes, args.session_seed
        )
    else:
        per_persona = {pid: pool for pid in pids}

    summary = emit_jobs(people, items, per_persona, args.out, args.round, args.session_seed)

    print(
        f"round {args.round}: emitted {summary['emitted']} prompts, "
        f"skipped {summary['skipped']} already answered "
        f"({summary['n_personas']} personas, {len(pool)} items in the pool)"
    )
    print(f"  prompts -> {summary['prompts_path']}")
    if args.round == "retest":
        print(f"  fixed retest selection -> {Path(args.out) / SELECTION_FILENAME}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
