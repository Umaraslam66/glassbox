"""Stage 5 episodes: one sitting of an adaptive interview, per persona.

    python -m src.interview.episodes \\
        --answers <sweep run>/answers.jsonl \\
        --recorded <sweep run>/completions.jsonl \\
        --noise-dir <dir> --public-bank <bank_items.json> \\
        --fit results/stage2_v2_fit.npz \\
        --strategies random,heuristic,fixed --replicates 51 \\
        --out data/runs/stage5/episode_codes.npz

THE EPISODE NEVER RE-QUERIES THE RESPONDER. Same consistency rule as Stage 3
(``src/interview/__init__.py``), same machinery: an episode's answer to a closed
item is a fresh, independently seeded application of the frozen noise layer
(a = 1.2, b = 4.0, T_noise = 16, seed base 548) to what that persona already
recorded for that item in the v2 full sweep. Stage 3 did this once, with the
round tag ``interview1``. Stage 5 does it once per episode, with the round tag
``ep_<strategy>_<replicate>``, which is the same slot in the same per-record
seed hash (:func:`src.personas.noise_layer.cell_seed`). So every episode is an
independent sitting of the same person, reproducible from its tag, and no
episode can produce an answer from outside that person's own distribution.

Why the answers can be drawn before the questions are chosen
-----------------------------------------------------------
A cell's draw depends on ``(seed base, persona, item, round)`` and on the
recorded material -- not on what else was asked, or when. So drawing the whole
candidate bank for a round up front gives exactly the answers an
asked-one-at-a-time episode would have got, and costs one pass over the
recorded material instead of one per question. The simulator then keeps that
matrix **private** and hands out one cell at a time
(:class:`AnswerBank`), which is what makes the strategy's ignorance structural:
there is no public route from a strategy to an answer it did not ask for.

Reading this the other way round: the episode is a replay, not a new
conversation. That is the whole reason Stage 5 costs no GPU.

WALL. Every path is a command-line argument with no default, exactly as in
``src.interview.interview_answers``: nothing here writes down where the planted
material lives. The recorded material is handed to
:mod:`src.personas.noise_layer` by path and drawn answers come back; the
distributions themselves stay on the persona side, and the system-side objects
in this module (:class:`AnswerBank`, :class:`EpisodeSimulator`) carry coded
answers and nothing else.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from ..model import mirt
from ..model.sequential_posterior import UNASKED, SequentialPosterior
from ..personas import noise_layer
from . import FROZEN_NOISE_A, FROZEN_NOISE_B, FROZEN_NOISE_T, FROZEN_SEED_BASE
from .strategies import Strategy

#: The frozen layer, assembled from the constants the package publishes. Same
#: object Stage 3's scripted interview used; there are no flags for it.
FROZEN_CONFIG = noise_layer.NoiseConfig(
    a=FROZEN_NOISE_A, b=FROZEN_NOISE_B, t_noise=FROZEN_NOISE_T
)

#: Prefix of every Stage 5 round tag. Kept distinct from ``main``, ``retest``
#: and ``interview1`` so an episode's draws are independent of all of them.
ROUND_PREFIX = "ep"


def round_tag(strategy: str, replicate: int) -> str:
    """The round name for one episode. Reproducible, and unique per sitting."""
    return f"{ROUND_PREFIX}_{strategy}_{int(replicate):03d}"


def episode_seed(strategy: str, replicate: int, base: int = FROZEN_SEED_BASE) -> int:
    """The seed a strategy's own randomness uses for this episode.

    Folded from the same material as the round tag, so an episode is one thing:
    the same tag gives the same answers and the same random ordering.
    """
    return int(noise_layer.cell_seed(base, strategy, "strategy", round_tag(strategy, replicate)) % (2**63))


# --------------------------------------------------------------------------
# persona side: drawing a round
# --------------------------------------------------------------------------


class EpisodeSource:
    """The persona side of an episode: draws a round of answers on demand.

    Loads the sweep's recorded material once -- it is tens of megabytes and 150
    rounds would otherwise read it 150 times -- and then produces one coded
    answer matrix per round tag. Nothing about the strategy reaches this class;
    it does not know or care which items will end up being asked.
    """

    def __init__(
        self,
        answers_path: str | Path,
        recorded_path: str | Path,
        noise_dir: str | Path,
        public_bank: str | Path,
        item_ids: Sequence[str],
        persona_ids: Sequence[str],
        *,
        source_round: str = "main",
    ) -> None:
        self.item_ids = [str(i) for i in item_ids]
        self.persona_ids = [str(p) for p in persona_ids]
        self.source_round = str(source_round)

        self._types = noise_layer.item_types(public_bank)
        unknown = [i for i in self.item_ids if i not in self._types]
        if unknown:
            raise ValueError(
                f"{len(unknown)} candidate item(s) are not in the public bank, "
                f"first: {unknown[0]}"
            )
        self.kinds = [self._types[i] for i in self.item_ids]

        self._wobbles = noise_layer.load_wobbles(noise_dir)
        if not self._wobbles:
            raise ValueError(f"no wobble records in {noise_dir}")

        wanted_items = set(self.item_ids)
        wanted_pids = set(self.persona_ids)
        self._answers = [
            row
            for row in noise_layer.read_jsonl(answers_path)
            if str(row.get("item_id")) in wanted_items
            and str(row.get("pid")) in wanted_pids
            and str(row.get("round") or "main") == self.source_round
        ]
        if not self._answers:
            raise ValueError(
                f"{answers_path} holds no {self.source_round} answers on the "
                f"{len(self.item_ids)} candidate items -- wrong run directory?"
            )
        self._recorded = noise_layer.recorded_distributions(
            recorded_path, item_ids=self.item_ids, rounds=(self.source_round,)
        )

        self._row_of = {pid: r for r, pid in enumerate(self.persona_ids)}
        self._col_of = {iid: c for c, iid in enumerate(self.item_ids)}

    @property
    def n_personas(self) -> int:
        return len(self.persona_ids)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    def round_codes(self, round_name: str) -> tuple[np.ndarray, dict[str, Any]]:
        """One episode's coded answers, ``(n_personas, n_items)``, plus counts.

        Every cell is filled: the draw is per cell, so the whole bank is drawn
        and the simulator decides which cells anyone ever sees. A cell the
        layer could not draw (no recorded material) stays at ``-1`` and is
        counted in the summary; a strategy that picks it would be scoring an
        unanswered question, so the simulator refuses.
        """
        rows, stats = noise_layer.draw_round(
            self._answers,
            self._recorded,
            self._wobbles,
            self._types,
            FROZEN_CONFIG,
            FROZEN_SEED_BASE,
            round_name=str(round_name),
            source_round=self.source_round,
            item_ids=self.item_ids,
        )
        codes = np.full((self.n_personas, self.n_items), UNASKED, dtype=np.int8)
        for row in rows:
            r = self._row_of.get(str(row.get("pid")))
            c = self._col_of.get(str(row.get("item_id")))
            if r is None or c is None:
                continue
            codes[r, c] = mirt._code_answer(row["answer"], self.kinds[c])
        summary = {
            "round": str(round_name),
            "source_round": self.source_round,
            "n_cells": int(codes.size),
            "n_missing": int((codes < 0).sum()),
            "n_events": int(stats["n_events"]),
            "n_changed": int(stats["n_changed"]),
            "event_rate": stats["event_rate"],
            "change_rate": stats["change_rate"],
        }
        return codes, summary


# --------------------------------------------------------------------------
# system side: the private answer bank
# --------------------------------------------------------------------------


class AnswerBank:
    """The episode's answers, kept private and handed out one cell at a time.

    The whole point of this class is what it does *not* offer: there is no
    public attribute or method that returns the answer matrix, a column of it,
    or anything about a cell that was not asked for. :meth:`reveal` is the only
    way in, and it takes the items a strategy actually chose.
    """

    __slots__ = ("__codes", "n_personas", "n_items", "round_name")

    def __init__(self, codes: np.ndarray, round_name: str = "") -> None:
        codes = np.asarray(codes, dtype=np.int16)
        if codes.ndim != 2:
            raise ValueError(f"codes must be (personas, items), got {codes.shape}")
        # name-mangled: reachable from inside this class, awkward from outside,
        # and invisible to anything that walks the public surface.
        self.__codes = codes
        self.n_personas = int(codes.shape[0])
        self.n_items = int(codes.shape[1])
        self.round_name = str(round_name)

    def reveal(self, columns: np.ndarray) -> np.ndarray:
        """The answer each persona gives to the item that persona was just asked."""
        columns = np.asarray(columns, dtype=np.int64).reshape(-1)
        if columns.shape != (self.n_personas,):
            raise ValueError(
                f"expected one item per persona ({self.n_personas}), got {columns.shape}"
            )
        if columns.min() < 0 or columns.max() >= self.n_items:
            raise ValueError("item column out of range")
        answers = self.__codes[np.arange(self.n_personas), columns]
        if np.any(answers < 0):
            missing = int(np.flatnonzero(answers < 0)[0])
            raise ValueError(
                f"persona row {missing} has no recorded answer for item column "
                f"{int(columns[missing])} in round {self.round_name!r}"
            )
        return answers.astype(np.int64)

    def full_matrix_for_scoring(self) -> np.ndarray:
        """Every answer at once -- the full-bank reference arm, not an episode.

        Named at length on purpose. This is the "what if you asked all of them"
        denominator of the fraction-of-full-matrix measure; it is never handed
        to a strategy, and no episode calls it.
        """
        return np.array(self.__codes, copy=True)


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EpisodeResult:
    """What an episode did, in system-side terms only."""

    strategy: str
    round_name: str
    n_steps: int
    capped_at: int | None
    chosen: np.ndarray  # (n_steps, n_personas) item columns, in order asked


class EpisodeSimulator:
    """Ask, answer, update -- until the question budget or the strategy runs out.

    ``on_step`` is called after every update with ``(step, posterior)`` so a
    caller can record whatever it is entitled to record. The Gate 5 grader is
    the caller that records truth-side numbers, and it lives in ``src/eval``.
    """

    def __init__(
        self,
        bank: AnswerBank,
        posterior: SequentialPosterior,
        strategy: Strategy,
    ) -> None:
        if bank.n_personas != posterior.n_personas:
            raise ValueError(
                f"bank has {bank.n_personas} personas, posterior has "
                f"{posterior.n_personas}"
            )
        if bank.n_items != posterior.n_items:
            raise ValueError(
                f"bank has {bank.n_items} items, posterior has {posterior.n_items}"
            )
        self._bank = bank
        self.posterior = posterior
        self.strategy = strategy

    def run(
        self,
        n_steps: int,
        *,
        on_step: Callable[[int, SequentialPosterior], None] | None = None,
    ) -> EpisodeResult:
        cap = self.strategy.max_steps()
        budget = int(n_steps) if cap is None else min(int(n_steps), int(cap))
        budget = min(budget, self.posterior.n_items)
        chosen = np.zeros((budget, self.posterior.n_personas), dtype=np.int64)

        self.posterior.reset()
        self.strategy.reset()
        for step in range(budget):
            columns = np.asarray(self.strategy.select(self.posterior.state()), dtype=np.int64)
            answers = self._bank.reveal(columns)
            self.strategy.observe(columns, answers)
            self.posterior.observe(columns, answers)
            chosen[step] = columns
            if on_step is not None:
                on_step(step + 1, self.posterior)
        return EpisodeResult(
            strategy=getattr(self.strategy, "name", "strategy"),
            round_name=self._bank.round_name,
            n_steps=budget,
            capped_at=None if cap is None or cap > n_steps else int(cap),
            chosen=chosen,
        )


# --------------------------------------------------------------------------
# CLI: draw and cache every episode's answers once
# --------------------------------------------------------------------------


def draw_rounds(
    source: EpisodeSource, rounds: Iterable[str], *, verbose: bool = True
) -> tuple[np.ndarray, list[str], list[dict[str, Any]]]:
    """Draw a list of rounds and stack them, ``(n_rounds, personas, items)``."""
    names = [str(r) for r in rounds]
    stack = np.zeros(
        (len(names), source.n_personas, source.n_items), dtype=np.int8
    )
    summaries = []
    for k, name in enumerate(names):
        codes, summary = source.round_codes(name)
        stack[k] = codes
        summaries.append(summary)
        if verbose:
            print(
                f"  {name}: event rate {summary['event_rate']}, "
                f"changed {summary['change_rate']}, missing {summary['n_missing']}"
            )
    return stack, names, summaries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.interview.episodes",
        description=(
            "Draw and cache one round of closed answers per Stage 5 episode. "
            "Never re-queries the responder."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--recorded", type=Path, required=True)
    parser.add_argument("--noise-dir", type=Path, required=True)
    parser.add_argument("--public-bank", type=Path, required=True)
    parser.add_argument(
        "--fit",
        type=Path,
        required=True,
        help="the Stage 2 fit archive: its training items are the candidate bank",
    )
    parser.add_argument("--strategies", default="random,heuristic,fixed")
    parser.add_argument("--replicates", type=int, default=51)
    parser.add_argument("--source-round", default="main")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    fit = np.load(args.fit, allow_pickle=False)
    item_ids = [str(i) for i in fit["item_train_ids"]]
    persona_ids = [str(p) for p in fit["persona_train_ids"]] + [
        str(p) for p in fit["persona_holdout_ids"]
    ]

    source = EpisodeSource(
        args.answers,
        args.recorded,
        args.noise_dir,
        args.public_bank,
        item_ids,
        persona_ids,
        source_round=args.source_round,
    )
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    rounds = [
        round_tag(name, rep)
        for name in strategies
        for rep in range(int(args.replicates))
    ]
    print(
        f"drawing {len(rounds)} episodes x {source.n_personas} personas x "
        f"{source.n_items} items from the {args.source_round} round"
    )
    stack, names, summaries = draw_rounds(source, rounds, verbose=False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        codes=stack,
        rounds=np.array(names),
        persona_ids=np.array(persona_ids),
        item_ids=np.array(item_ids),
    )
    manifest = {
        "rounds": names,
        "n_personas": source.n_personas,
        "n_items": source.n_items,
        "source_round": args.source_round,
        "noise_config": FROZEN_CONFIG.as_dict(),
        "seed_base": FROZEN_SEED_BASE,
        "per_round": [
            {k: v for k, v in s.items() if k != "by_persona"} for s in summaries
        ],
    }
    manifest_path = Path(str(args.out) + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"  codes    -> {args.out}")
    print(f"  manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
