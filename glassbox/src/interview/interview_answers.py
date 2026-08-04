"""Draw the closed-form answers a persona gives in the scripted interview.

    python -m src.interview.interview_answers \\
        --answers <sweep run>/answers.jsonl \\
        --recorded <sweep run>/completions.jsonl \\
        --noise-dir <dir> --public-bank <bank_items.json> \\
        --splits <splits_v1.json> --out <sweep run>/answers_interview1.jsonl

THE INTERVIEW NEVER RE-QUERIES THE RESPONDER ON A CLOSED ITEM. This is the
consistency rule (``src/interview/__init__.py``) in operation. The interview
answer for one persona on one item is a fresh, independently seeded application
of the frozen noise layer to what that persona already recorded for that item
in the full sweep: same recorded material, same frozen settings, new seed. So an
interview answer can differ from the sweep answer -- people do answer the same
question differently on a different day, which is the whole point of the wobble
-- but it can never come from outside the persona's own answer distribution.

The new seed comes from the round tag. ``src.personas.noise_layer.cell_seed``
folds ``(seed base, pid, item_id, round)`` into every cell's seed, which is how
the main pass and the retest already wobble independently of each other. The
interview round tag is one more value in that same slot.

The settings are not options. a, b, T_noise and the seed base are the frozen
constants in ``src/interview/__init__.py`` and this module has no flags for
them, on purpose: an interview that could be run at other settings would
produce answers from a different world than the one Stage 4 grades against.

The interview is the 15 items in ``interview_closed`` from the frozen splits
file, in the order they are listed there. That order is the scripted order.

WALL RULE: every path is a command-line argument with no default, so nothing
here writes down where the planted truth lives. The recorded material is read
by the persona-side layer, which is handed the path and hands answers back. The
output is a persona-side artifact -- keep it in the run directory behind the
Wall; ``src.interview.assemble_transcripts`` is what carries answers across.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..personas import noise_layer
from . import FROZEN_NOISE_A, FROZEN_NOISE_B, FROZEN_NOISE_T, FROZEN_SEED_BASE
from .responder_prompts import CLOSED_ROUNDS, INTERVIEW_ROUNDS

#: The frozen layer, assembled once from the constants the package publishes.
FROZEN_CONFIG = noise_layer.NoiseConfig(
    a=FROZEN_NOISE_A, b=FROZEN_NOISE_B, t_noise=FROZEN_NOISE_T
)


def interview_items(splits_path: str | Path) -> list[str]:
    """The scripted closed items, in the order the frozen splits file lists them."""
    payload = json.loads(Path(splits_path).read_text(encoding="utf-8"))
    items = [str(item_id) for item_id in payload.get("interview_closed") or []]
    if not items:
        raise ValueError(
            f"{splits_path} has no interview_closed list -- this is not the "
            "frozen splits file"
        )
    return items


def draw(
    answers_path: str | Path,
    recorded_path: str | Path,
    noise_dir: str | Path,
    public_bank: str | Path,
    items: list[str],
    round_name: str,
    source_round: str = "main",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Draw one interview round for every persona in the answers file."""
    types = noise_layer.item_types(public_bank)
    unknown = [item_id for item_id in items if item_id not in types]
    if unknown:
        raise ValueError(
            f"{len(unknown)} scripted item(s) are not in the public bank, "
            f"first: {unknown[0]}"
        )

    wobbles = noise_layer.load_wobbles(noise_dir)
    if not wobbles:
        raise ValueError(f"no wobble records in {noise_dir}")

    wanted = set(items)
    answers = [
        row
        for row in noise_layer.read_jsonl(answers_path)
        if str(row.get("item_id")) in wanted
        and str(row.get("round") or "main") == source_round
    ]
    if not answers:
        raise ValueError(
            f"{answers_path} holds no {source_round} answers on the 15 scripted "
            "items -- wrong run directory?"
        )

    recorded = noise_layer.recorded_distributions(
        recorded_path, item_ids=items, rounds=(source_round,)
    )

    return noise_layer.draw_round(
        answers,
        recorded,
        wobbles,
        types,
        FROZEN_CONFIG,
        FROZEN_SEED_BASE,
        round_name=round_name,
        source_round=source_round,
        item_ids=items,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.interview.interview_answers",
        description=(
            "Draw the scripted interview's closed answers from what the sweep "
            "already recorded. Never re-queries the responder."
        ),
        # No abbreviations: "--a 1.5" must be an error, not a silent prefix
        # match on "--answers". The frozen settings have no flags and an
        # attempt to give them one has to fail loudly.
        allow_abbrev=False,
    )
    parser.add_argument(
        "--answers",
        type=Path,
        required=True,
        help="the sweep's parsed answers JSONL, supplied per run",
    )
    parser.add_argument(
        "--recorded",
        type=Path,
        required=True,
        help=(
            "the responder completions file those answers were parsed from: it "
            "carries the recorded material the frozen layer re-draws from"
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
        help="the public bank file: needed for each item's type",
    )
    parser.add_argument(
        "--splits",
        type=Path,
        required=True,
        help="the frozen splits file: interview_closed, in scripted order",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="where to write the interview answers JSONL (persona side)",
    )
    parser.add_argument(
        "--stats",
        type=Path,
        default=None,
        help="where to write the stats JSON (default: <out>.stats.json)",
    )
    parser.add_argument(
        "--round",
        choices=INTERVIEW_ROUNDS,
        default=INTERVIEW_ROUNDS[0],
        help="which interview sitting this is (default %(default)s)",
    )
    parser.add_argument(
        "--source-round",
        choices=CLOSED_ROUNDS,
        default="main",
        help="which recorded round to draw from (default %(default)s)",
    )
    args = parser.parse_args(argv)

    items = interview_items(args.splits)
    rows, stats = draw(
        args.answers,
        args.recorded,
        args.noise_dir,
        args.public_bank,
        items,
        args.round,
        args.source_round,
    )
    stats["items"] = items

    noise_layer.write_jsonl(args.out, rows)
    stats_path = args.stats or Path(str(args.out) + ".stats.json")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    personas = len({str(row.get("pid")) for row in rows})
    print(
        f"round {args.round}: drew {len(rows)} answers "
        f"({personas} personas x {len(items)} scripted items) from the "
        f"{args.source_round} round"
    )
    print(
        f"  {stats['n_events']} noise events ({stats['event_rate']}), "
        f"{stats['n_changed']} answers actually changed ({stats['change_rate']})"
    )
    if any(stats["skipped"].values()):
        print(f"  passed through untouched: {stats['skipped']}")
    print(f"  answers -> {args.out}")
    print(f"  stats   -> {stats_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
