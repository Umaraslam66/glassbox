"""The true answer distribution of every held-out cell, as Stage 4 grades it.

    python -m src.eval.truth_distributions \\
        --answers <sweep run>/answers.jsonl \\
        --recorded <sweep run>/completions.jsonl \\
        --noise-dir <wobble dir> \\
        --public-bank data/public/bank_items.json \\
        --splits experiments/splits_v1.json \\
        --out <truth runs dir>/truth50

PREREGISTRATION.md section 5 says prediction quality is measured "against the
persona's true answer distribution, estimated from 50 samples per item at fixed
temperature", and the Gate-1 addendum makes that operational: the estimate is
50 **independent seeded applications of the frozen noise layer** to the
responder's recorded answer distribution, at a = 1.2, b = 4.0, T_noise = 16,
seed-base 548, responder temperature 0.7. This module is that sentence in code.

Independence, and how it is obtained
------------------------------------
``src.personas.noise_layer.cell_seed`` folds ``(seed base, pid, item_id, round)``
into every cell's random seed. Two rounds with different names therefore wobble
independently of each other -- that is the same mechanism the main pass, the
retest and the interview sitting already rely on. The 50 rounds here are named
``truth50_s01`` .. ``truth50_s50``, which collide with none of ``main``,
``retest`` or ``interview1``, so all 53 sittings of a cell are independent draws
from the one distribution.

Nothing is re-queried. The responder answered once, in the full sweep; every
sitting is a fresh application of the layer to what it recorded then.

Also written: the analytic distribution
---------------------------------------
The layer is simple enough to integrate in closed form. With the responder's
recorded answer ``a0``, its conviction ``p_max``, the persona's wobble ``w`` and
the flattened re-draw distribution ``f``:

    event probability  q = min(w * (a + b * (1 - p_max)), 0.95)
    P(answer = c)      = (1 - q) * [c == a0]  +  q * f_c

because a cell either passes through untouched or is re-drawn from ``f``. That
exact distribution is stored beside the 50-sample counts as ``expected.npy``. It
is **grader-side only** -- it is the oracle a perfect predictor would produce,
and it is what the Gate 4 report uses to separate "the model is wrong" from
"50 samples are a noisy yardstick". It never crosses to the system side.

WALL RULE. This module lives in ``src/eval/``, so it may read the recorded
answer distributions. Every path is a command-line argument with no default, so
the planted-truth directory is not written down anywhere in the source. The
output belongs behind the Wall with the rest of the run material: pass an
``--out`` inside the truth store's ``runs/`` folder, never ``results/`` and
never ``data/runs/``. :func:`assert_output_is_behind_the_wall` refuses the two
system-side locations outright.

Determinism. Rerunning with the same inputs reproduces byte-identical output:
``.npy`` carries no timestamp, and the manifest carries no clock reading.
``tests/test_truth_distributions.py`` checks that by hashing the files twice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..interview import (
    FROZEN_NOISE_A,
    FROZEN_NOISE_B,
    FROZEN_NOISE_T,
    FROZEN_RESPONDER_TEMPERATURE,
    FROZEN_SEED_BASE,
)
from ..personas import noise_layer

#: The frozen layer, assembled from the constants the interview package
#: publishes. Not options: the grader and the interview have to describe one
#: world (``src/interview/__init__.py``, the consistency rule).
FROZEN_CONFIG = noise_layer.NoiseConfig(
    a=FROZEN_NOISE_A, b=FROZEN_NOISE_B, t_noise=FROZEN_NOISE_T
)

#: How many independent sittings make up one "true answer distribution".
#: PREREGISTRATION.md section 5: "estimated from 50 samples per item".
N_SAMPLES = 50

#: Round-name prefix for those sittings. Chosen so it collides with no round
#: that has already been drawn (``main``, ``retest``, ``interview1``).
ROUND_PREFIX = "truth50_s"

#: Widest category count in the bank (Likert-5). Binary items use columns 0/1.
MAX_CATEGORIES = 5

SCHEMA = "glassbox.stage4.truth50/1"


def round_names(n_samples: int = N_SAMPLES) -> list[str]:
    """``truth50_s01`` .. ``truth50_s50``, in order."""
    return [f"{ROUND_PREFIX}{i:02d}" for i in range(1, n_samples + 1)]


# --------------------------------------------------------------------------
# coding
# --------------------------------------------------------------------------


def category_values(item_type: str) -> tuple[int | str, ...]:
    """The item's answers in category-index order -- the same order MIRT codes.

    ``src.model.mirt._code_answer`` maps a Likert answer ``v`` to ``v - 1`` and
    a binary answer to 0 for no / 1 for yes.
    ``src.personas.noise_layer.valid_answers`` already lists them that way, so
    the two sides of the Wall agree on what "category 3" means without either
    importing the other.
    """
    return noise_layer.valid_answers(item_type)


def n_categories(item_type: str) -> int:
    return len(category_values(item_type))


def code_of(answer: Any, item_type: str) -> int:
    """Category index of one recorded answer, or -1 when it is not a valid one."""
    values = category_values(item_type)
    if item_type == "likert5":
        try:
            value: Any = int(answer)
        except (TypeError, ValueError):
            return -1
    else:
        value = str(answer).strip().lower()
    return values.index(value) if value in values else -1


# --------------------------------------------------------------------------
# the analytic distribution
# --------------------------------------------------------------------------


def expected_distribution(
    recorded: Mapping[str, float] | None,
    pre_noise_answer: Any,
    wobble: float,
    item_type: str,
    config: noise_layer.NoiseConfig = FROZEN_CONFIG,
) -> np.ndarray | None:
    """The exact post-layer distribution of one cell, as a length-5 row.

    ``(1 - q) * onehot(pre-noise answer) + q * flattened`` -- see the module
    docstring. ``None`` when the cell has nothing to build a distribution from,
    which is exactly when the layer would have passed it through untouched.
    """
    distribution = noise_layer.answer_distribution(recorded, item_type)
    if distribution is None:
        return None
    values = category_values(item_type)
    probability = noise_layer.event_probability(
        wobble, noise_layer.conviction(distribution), config
    )
    flat = noise_layer.flattened(distribution, config.t_noise)

    row = np.zeros(MAX_CATEGORIES, dtype=float)
    for index, value in enumerate(values):
        row[index] = probability * float(flat[value])
    code = code_of(pre_noise_answer, item_type)
    if code < 0:
        return None
    row[code] += 1.0 - probability
    return row


# --------------------------------------------------------------------------
# drawing the 50 sittings
# --------------------------------------------------------------------------


def draw_sittings(
    answers: Sequence[Mapping[str, Any]],
    recorded: Mapping[tuple[str, str, str], Mapping[str, float]],
    wobbles: Mapping[str, float],
    types: Mapping[str, str],
    persona_ids: Sequence[str],
    item_ids: Sequence[str],
    *,
    n_samples: int = N_SAMPLES,
    source_round: str = "main",
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """``(draws, per_round_stats)`` with ``draws`` shaped (sittings, people, items).

    Each sitting is one call to :func:`src.personas.noise_layer.draw_round` with
    its own round name, which is the only thing that differs between them.
    """
    prow = {pid: i for i, pid in enumerate(persona_ids)}
    icol = {iid: j for j, iid in enumerate(item_ids)}
    draws = np.full((n_samples, len(persona_ids), len(item_ids)), -1, dtype=np.int8)
    stats: list[dict[str, Any]] = []

    for sitting, name in enumerate(round_names(n_samples)):
        rows, block = noise_layer.draw_round(
            answers,
            recorded,
            wobbles,
            types,
            FROZEN_CONFIG,
            FROZEN_SEED_BASE,
            round_name=name,
            source_round=source_round,
            item_ids=list(item_ids),
        )
        for row in rows:
            pid = str(row.get("pid"))
            item_id = str(row.get("item_id"))
            if pid not in prow or item_id not in icol:
                continue
            draws[sitting, prow[pid], icol[item_id]] = code_of(
                row.get("answer"), types[item_id]
            )
        stats.append(
            {
                "round": name,
                "n_answers": block["n_answers"],
                "n_events": block["n_events"],
                "n_changed": block["n_changed"],
                "event_rate": block["event_rate"],
                "change_rate": block["change_rate"],
                "skipped": block["skipped"],
            }
        )
    return draws, stats


def counts_from_draws(draws: np.ndarray, n_people: int, n_items: int) -> np.ndarray:
    """(people, items, 5) counts of each category across the sittings."""
    counts = np.zeros((n_people, n_items, MAX_CATEGORIES), dtype=np.int32)
    for category in range(MAX_CATEGORIES):
        counts[:, :, category] = np.sum(draws == category, axis=0)
    return counts


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------


def assert_output_is_behind_the_wall(out_dir: Path) -> None:
    """Refuse to write the answer key anywhere the system side can read it."""
    parts = [part.lower() for part in Path(out_dir).resolve().parts]
    for forbidden in (("results",), ("data", "runs")):
        width = len(forbidden)
        if any(tuple(parts[i : i + width]) == forbidden for i in range(len(parts))):
            raise ValueError(
                f"--out {out_dir} is on the system side of the Wall. The "
                "50-sample true distributions are the answer key: they belong "
                "in the planted-truth store's runs/ folder."
            )


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(
    answers_path: Path,
    recorded_path: Path,
    noise_dir: Path,
    public_bank: Path,
    splits_path: Path,
    out_dir: Path,
    *,
    n_samples: int = N_SAMPLES,
    source_round: str = "main",
    verbose: bool = True,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    assert_output_is_behind_the_wall(out_dir)

    splits = json.loads(Path(splits_path).read_text(encoding="utf-8"))
    persona_ids = [str(p) for p in splits["persona_holdout"]]
    item_ids = [str(i) for i in splits["item_holdout"]]
    strata = {str(k): str(v) for k, v in (splits.get("strata") or {}).items()}

    types = noise_layer.item_types(public_bank)
    missing = [i for i in item_ids if i not in types]
    if missing:
        raise ValueError(f"{len(missing)} probe item(s) not in the public bank: {missing[:3]}")

    wobbles = noise_layer.load_wobbles(noise_dir)
    if not wobbles:
        raise ValueError(f"no wobble records in {noise_dir}")

    wanted_people = set(persona_ids)
    wanted_items = set(item_ids)
    answers = [
        row
        for row in noise_layer.read_jsonl(answers_path)
        if str(row.get("pid")) in wanted_people
        and str(row.get("item_id")) in wanted_items
        and str(row.get("round") or "main") == source_round
    ]
    expected_cells = len(persona_ids) * len(item_ids)
    if len(answers) != expected_cells:
        raise ValueError(
            f"{answers_path} holds {len(answers)} of the expected {expected_cells} "
            f"{source_round}-round cells for the held-out personas on the probe items"
        )

    recorded = noise_layer.recorded_distributions(
        recorded_path, item_ids=item_ids, rounds=(source_round,)
    )

    if verbose:
        print(
            f"{len(persona_ids)} held-out personas x {len(item_ids)} probe items "
            f"= {expected_cells:,} cells, {n_samples} independent sittings each"
        )

    draws, per_round = draw_sittings(
        answers,
        recorded,
        wobbles,
        types,
        persona_ids,
        item_ids,
        n_samples=n_samples,
        source_round=source_round,
    )
    if int((draws < 0).sum()):
        raise ValueError(
            f"{int((draws < 0).sum())} drawn cells did not code to a valid answer"
        )
    counts = counts_from_draws(draws, len(persona_ids), len(item_ids))

    # ---- the analytic distribution, for the grader's oracle ceiling --------
    by_cell = {(str(r.get("pid")), str(r.get("item_id"))): r for r in answers}
    expected = np.zeros((len(persona_ids), len(item_ids), MAX_CATEGORIES), dtype=float)
    n_analytic_missing = 0
    for row, pid in enumerate(persona_ids):
        for col, item_id in enumerate(item_ids):
            record = by_cell[(pid, item_id)]
            distribution = expected_distribution(
                record.get("logprobs") or recorded.get((pid, item_id, source_round)),
                record.get("answer"),
                wobbles[pid],
                types[item_id],
            )
            if distribution is None:
                n_analytic_missing += 1
                observed = counts[row, col].astype(float)
                distribution = observed / max(observed.sum(), 1.0)
            expected[row, col] = distribution

    # ---- write ------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    kinds = [types[i] for i in item_ids]
    paths = {
        "draws": out_dir / "draws.npy",
        "counts": out_dir / "counts.npy",
        "expected": out_dir / "expected.npy",
    }
    np.save(paths["draws"], draws)
    np.save(paths["counts"], counts)
    np.save(paths["expected"], expected)

    empirical = counts / float(n_samples)
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "what": (
            "the true answer distribution of every held-out persona x probe item "
            "cell, as PREREGISTRATION.md section 5 defines it: 50 independent "
            "seeded applications of the frozen noise layer to the responder's "
            "recorded answer distribution"
        ),
        "frozen_config": FROZEN_CONFIG.as_dict()
        | {
            "seed_base": FROZEN_SEED_BASE,
            "responder_temperature": FROZEN_RESPONDER_TEMPERATURE,
        },
        "n_samples": n_samples,
        "rounds": round_names(n_samples),
        "source_round": source_round,
        "independence": (
            "cell_seed folds the round name into every cell's seed, so these 50 "
            "sittings are independent of each other and of main / retest / "
            "interview1"
        ),
        "persona_ids": persona_ids,
        "item_ids": item_ids,
        "item_kinds": kinds,
        "n_categories": [n_categories(k) for k in kinds],
        "strata": [strata.get(i, "unknown") for i in item_ids],
        "layout": {
            "draws.npy": "int8 (sittings, personas, items) category index",
            "counts.npy": "int32 (personas, items, 5) count of each category",
            "expected.npy": (
                "float64 (personas, items, 5) the analytic post-layer "
                "distribution -- GRADER SIDE ONLY, the oracle a perfect "
                "predictor would emit"
            ),
        },
        "inputs": {
            "answers": str(answers_path),
            "recorded": str(recorded_path),
            "noise_dir": str(noise_dir),
            "public_bank": str(public_bank),
            "splits": str(splits_path),
        },
        "checksums": {name: sha256(path) for name, path in paths.items()},
        "summary": {
            "n_cells": expected_cells,
            "n_draws": int(draws.size),
            "mean_event_rate": float(
                np.mean([row["event_rate"] for row in per_round if row["event_rate"]])
            ),
            "mean_change_rate": float(
                np.mean([row["change_rate"] for row in per_round if row["change_rate"]])
            ),
            "cells_with_a_degenerate_empirical_distribution": int(
                np.sum(empirical.max(axis=2) >= 1.0)
            ),
            "mean_empirical_max_probability": float(np.mean(empirical.max(axis=2))),
            "mean_total_variation_empirical_vs_analytic": float(
                np.mean(0.5 * np.abs(empirical - expected).sum(axis=2))
            ),
            "cells_without_an_analytic_distribution": n_analytic_missing,
            "analytic_noise_floor_brier": float(
                np.mean(
                    np.sum(expected * (1.0 - expected), axis=2)
                    / (np.asarray([n_categories(k) for k in kinds], dtype=float)[None, :]
                       * n_samples)
                )
            ),
            "analytic_noise_floor_note": (
                "mean over cells of sum_c p(1-p) / (categories * 50): what a "
                "PERFECT predictor would still score on the Brier defined as the "
                "mean over the item's own categories, purely because the yardstick "
                "is 50 samples rather than the exact distribution"
            ),
        },
        "per_round": per_round,
        "determinism": (
            "no clock reading is recorded anywhere in this file, and .npy has no "
            "timestamp, so a rerun on the same inputs is byte-identical"
        ),
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    if verbose:
        print(
            f"  event rate {manifest['summary']['mean_event_rate']:.4f}, "
            f"change rate {manifest['summary']['mean_change_rate']:.4f}"
        )
        print(
            "  mean total variation between the 50-sample estimate and the exact "
            f"distribution: {manifest['summary']['mean_total_variation_empirical_vs_analytic']:.4f}"
        )
        for name, path in paths.items():
            print(f"  {name:9s} -> {path}")
        print(f"  manifest  -> {manifest_path}")
    return manifest


# --------------------------------------------------------------------------
# reading back
# --------------------------------------------------------------------------


class Truth50:
    """The stored 50-sample distributions, loaded. Grader-side."""

    def __init__(self, directory: str | Path) -> None:
        directory = Path(directory)
        self.directory = directory
        self.manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        self.draws = np.load(directory / "draws.npy")
        self.counts = np.load(directory / "counts.npy")
        self.expected = np.load(directory / "expected.npy")
        self.persona_ids = [str(p) for p in self.manifest["persona_ids"]]
        self.item_ids = [str(i) for i in self.manifest["item_ids"]]
        self.item_kinds = [str(k) for k in self.manifest["item_kinds"]]
        self.n_categories = np.asarray(self.manifest["n_categories"], dtype=int)
        self.strata = [str(s) for s in self.manifest["strata"]]

    @property
    def n_samples(self) -> int:
        return int(self.manifest["n_samples"])

    @property
    def empirical(self) -> np.ndarray:
        """(personas, items, 5) empirical frequencies, counts / 50."""
        return self.counts.astype(float) / float(self.n_samples)

    def category_mask(self) -> np.ndarray:
        """(items, 5) boolean: which category columns this item actually has."""
        mask = np.zeros((len(self.item_ids), MAX_CATEGORIES), dtype=bool)
        for col, k in enumerate(self.n_categories):
            mask[col, : int(k)] = True
        return mask

    def verify_checksums(self) -> dict[str, bool]:
        return {
            name: sha256(self.directory / f"{name}.npy") == digest
            for name, digest in self.manifest["checksums"].items()
        }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.eval.truth_distributions",
        description=(
            "Draw the 50 independent seeded sittings that define every held-out "
            "cell's true answer distribution. Never re-queries the responder."
        ),
        # No abbreviations, for the same reason src.interview.interview_answers
        # forbids them: the frozen settings have no flags and an attempt to give
        # them one must fail loudly rather than prefix-match something else.
        allow_abbrev=False,
    )
    parser.add_argument("--answers", type=Path, required=True,
                        help="the sweep's parsed answers JSONL, supplied per run")
    parser.add_argument("--recorded", type=Path, required=True,
                        help="the completions file those answers were parsed from")
    parser.add_argument("--noise-dir", type=Path, required=True,
                        help="directory of {pid}.json wobble records")
    parser.add_argument("--public-bank", type=Path, required=True,
                        help="the public bank file: needed for each item's type")
    parser.add_argument("--splits", type=Path, required=True,
                        help="the frozen splits file: held-out personas and probe items")
    parser.add_argument("--out", type=Path, required=True,
                        help="output directory, behind the Wall")
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES,
                        help=argparse.SUPPRESS)
    parser.add_argument("--source-round", default="main")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if args.n_samples != N_SAMPLES:
        raise SystemExit(
            f"--n-samples is frozen at {N_SAMPLES} (PREREGISTRATION.md section 5)"
        )

    run(
        args.answers,
        args.recorded,
        args.noise_dir,
        args.public_bank,
        args.splits,
        args.out,
        n_samples=args.n_samples,
        source_round=args.source_round,
        verbose=not args.quiet,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
