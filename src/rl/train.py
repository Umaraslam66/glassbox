"""The Stage 5 RL training entry point. One command, resume-safe.

    python -m src.rl.train \\
        --config experiments/stage5_rl.json --profile confirmatory \\
        --answers <rl batch sweep>/answers.jsonl \\
        --recorded <rl batch sweep>/completions.jsonl \\
        --noise-dir <rl batch noise dir> \\
        --public-bank data/public/bank_items.json \\
        --fit results/stage2_v2_fit.npz \\
        --persona-source data/runs/rl_batch/answers_noised.jsonl \\
        --episodes data/runs/stage5_rl/episode_codes.npz \\
        --out-dir data/runs/stage5_rl/confirmatory \\
        --results results --prefix stage5_rl

Run it again after an interruption with the same arguments and it picks up from
the last checkpoint: the episode cache is only drawn if the file is not already
there, and training restarts at the iteration the checkpoint records with the
draws an uninterrupted run would have made.

What it does, in order
----------------------
1. **Draws the episodes once.** Every training sitting is a fresh, independently
   seeded application of the frozen noise layer to the batch's recorded answer
   material, under round tags ``ep_rl_<k>`` -- the same machinery, the same
   frozen constants and the same consistency rule Stage 3's scripted interview
   and the Stage 5 non-RL episodes used (:mod:`src.interview.episodes`). The
   responder is never re-queried, so training costs no GPU. The drawn matrix is
   cached to ``--episodes`` and reused on every later run.

   The ``--answers`` file is the sweep's own answer file, not a noised one: the
   layer is applied to the recorded material exactly once per episode. Handing
   it an already-noised file would noise it twice and leave the training
   sittings outside the frozen layer.

2. **Picks the training personas.** Either every persona in the batch
   (``--persona-source``, the usual case for the RL batch, whose personas are
   all training personas) or the fit's own training block (``--persona-block
   train``, used by the development smoke run on the original population).
   Whichever route, the held-out personas of the original fit are refused: the
   confirmatory evaluation set never appears in training, and the check is on
   the persona ids themselves rather than on a convention.

3. **Trains** with :func:`src.rl.trainer.train` on the blur reward -- declared
   posterior-variance reduction minus a per-question cost (owner RULING 2).

4. **Writes** the training curves, the item-choice histogram and the config into
   ``results/``, and the checkpoints into ``--out-dir``.

WALL. Every path is a command-line argument with no default, exactly as in
:mod:`src.interview.episodes`: nothing here writes down where the planted
material lives. The recorded material is handed to the noise layer by path and
coded answers come back.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..interview.episodes import EpisodeSource, draw_rounds, round_tag
from ..model.person_encoder import item_params_from_fit
from .policy import BLUR_REWARD, CONFIRMATORY_ARM
from .trainer import EpisodeCache, TrainConfig, draw_training_curves, train

__all__ = ["TRAINING_STRATEGY_TAG", "load_profile", "main", "persona_ids_from_jsonl"]

#: The strategy name the training rounds are tagged with. Kept distinct from
#: ``random``, ``fixed``, ``heuristic`` and ``policy`` so a training sitting can
#: never be confused with -- or reused as -- an evaluation sitting.
TRAINING_STRATEGY_TAG = "rl"


def load_profile(path: str | Path, profile: str) -> tuple[TrainConfig, dict[str, Any]]:
    """Read one named profile out of the experiments config file."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    profiles = document.get("profiles") or {}
    if profile not in profiles:
        raise ValueError(
            f"{path} has no profile {profile!r}; known: {sorted(profiles)}"
        )
    block = dict(profiles[profile])
    run = {
        "n_rounds": int(block.pop("n_rounds", 24)),
        "notes": block.pop("notes", ""),
        "profile": profile,
        "config_file": str(path),
    }
    return TrainConfig.from_dict(block), run


def persona_ids_from_jsonl(path: str | Path) -> list[str]:
    """Distinct ``pid`` values in a JSONL answer file, in first-seen order."""
    seen: dict[str, None] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            pid = json.loads(line).get("pid")
            if pid is not None:
                seen.setdefault(str(pid), None)
    return list(seen)


def refuse_holdout(persona_ids: Sequence[str], fit: Any) -> None:
    """Hard stop if any confirmatory evaluation persona is about to be trained on."""
    holdout = {str(p) for p in fit["persona_holdout_ids"]}
    overlap = sorted(set(map(str, persona_ids)) & holdout)
    if overlap:
        raise ValueError(
            f"{len(overlap)} of the requested training personas are in the "
            f"confirmatory held-out set (first: {overlap[0]}). The RL policy "
            "never trains on the evaluation personas."
        )


def ensure_episode_cache(
    cache_path: Path,
    *,
    answers: Path,
    recorded: Path,
    noise_dir: Path,
    public_bank: Path,
    item_ids: Sequence[str],
    persona_ids: Sequence[str],
    n_rounds: int,
    source_round: str = "main",
    verbose: bool = True,
) -> EpisodeCache:
    """Draw the training sittings if they are not already on disk."""
    if cache_path.is_file():
        if verbose:
            print(f"episode cache already drawn: {cache_path}")
        return EpisodeCache(cache_path)

    source = EpisodeSource(
        answers, recorded, noise_dir, public_bank, item_ids, persona_ids,
        source_round=source_round,
    )
    rounds = [round_tag(TRAINING_STRATEGY_TAG, k) for k in range(int(n_rounds))]
    if verbose:
        print(
            f"drawing {len(rounds)} training sittings x {source.n_personas} personas "
            f"x {source.n_items} closed items through the frozen noise layer"
        )
    stack, names, summaries = draw_rounds(source, rounds, verbose=verbose)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        codes=stack,
        rounds=np.array(names),
        persona_ids=np.array([str(p) for p in persona_ids]),
        item_ids=np.array([str(i) for i in item_ids]),
    )
    Path(str(cache_path) + ".manifest.json").write_text(
        json.dumps(
            {
                "rounds": names,
                "n_personas": source.n_personas,
                "n_items": source.n_items,
                "source_round": source_round,
                "per_round": [
                    {k: v for k, v in s.items() if k != "by_persona"} for s in summaries
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return EpisodeCache(cache_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.rl.train",
        description=(
            "Train the Stage 5 RL interviewer on the blur reward (owner RULING 2). "
            "Resume-safe: run the same command again to continue."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", default="confirmatory")
    parser.add_argument("--fit", type=Path, required=True,
                        help="the Stage 2 fit: its training items are the candidate bank "
                             "and its held-out personas are refused as training personas")
    parser.add_argument("--episodes", type=Path, required=True,
                        help="where the drawn training sittings are cached")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="checkpoints and the resume file")
    parser.add_argument("--results", type=Path, required=True,
                        help="directory for the training report and plot")
    parser.add_argument("--prefix", default="stage5_rl")
    # only needed when the episode cache has to be drawn
    parser.add_argument("--answers", type=Path, default=None)
    parser.add_argument("--recorded", type=Path, default=None)
    parser.add_argument("--noise-dir", type=Path, default=None)
    parser.add_argument("--public-bank", type=Path, default=None)
    parser.add_argument("--source-round", default="main")
    parser.add_argument("--persona-source", type=Path, default=None,
                        help="JSONL with pid fields naming the batch's personas")
    parser.add_argument("--persona-block", default="train", choices=("train", "all"),
                        help="used when --persona-source is absent: which block of "
                             "the fit's personas to train on (never the held-out set)")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    config, run = load_profile(args.config, args.profile)
    fit = np.load(args.fit, allow_pickle=False)
    params, item_ids = item_params_from_fit(fit, block="train")

    if args.persona_source is not None:
        persona_ids = persona_ids_from_jsonl(args.persona_source)
    elif args.persona_block == "train":
        persona_ids = [str(p) for p in fit["persona_train_ids"]]
    else:
        persona_ids = [str(p) for p in fit["persona_train_ids"]] + [
            str(p) for p in fit["persona_holdout_ids"]
        ]
    refuse_holdout(persona_ids, fit)

    if args.episodes.is_file():
        cache = EpisodeCache(args.episodes)
    else:
        missing = [
            name
            for name, value in (
                ("--answers", args.answers),
                ("--recorded", args.recorded),
                ("--noise-dir", args.noise_dir),
                ("--public-bank", args.public_bank),
            )
            if value is None
        ]
        if missing:
            raise SystemExit(
                f"{args.episodes} does not exist yet, so the sittings have to be "
                f"drawn; missing arguments: {', '.join(missing)}"
            )
        cache = ensure_episode_cache(
            args.episodes,
            answers=args.answers,
            recorded=args.recorded,
            noise_dir=args.noise_dir,
            public_bank=args.public_bank,
            item_ids=item_ids,
            persona_ids=persona_ids,
            n_rounds=run["n_rounds"],
            source_round=args.source_round,
            verbose=not args.quiet,
        )

    rows = cache.rows_for(persona_ids)
    refuse_holdout([cache.persona_ids[int(r)] for r in rows], fit)

    if not args.quiet:
        print(
            f"training on {rows.size} personas, {cache.n_rounds} cached sittings, "
            f"{cache.n_items} candidate items, horizon {config.horizon}, "
            f"{config.iterations} iterations x {config.batch_personas} episodes"
        )

    report = train(
        cache,
        params,
        config,
        out_dir=args.out_dir,
        persona_rows=rows,
        arm=CONFIRMATORY_ARM,
        reward_name=BLUR_REWARD,
        resume=not args.no_resume,
        verbose=not args.quiet,
    )
    report["run"] = run
    report["inputs"] = {
        "fit": str(args.fit),
        "episodes": str(args.episodes),
        "out_dir": str(args.out_dir),
        "persona_source": str(args.persona_source) if args.persona_source else None,
        "persona_block": args.persona_block if args.persona_source is None else None,
    }

    args.results.mkdir(parents=True, exist_ok=True)
    plot = draw_training_curves(report, args.results / f"{args.prefix}_training.png")
    report["artifacts"]["plot"] = str(plot)
    json_path = args.results / f"{args.prefix}_training.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if not args.quiet:
        first = report["history"][0]
        last = report["history"][-1]
        print()
        print(
            f"return {first['episode_return']:.4f} -> {last['episode_return']:.4f}, "
            f"entropy {first['entropy']:.3f} -> {last['entropy']:.3f}, "
            f"{int(np.count_nonzero(report['item_choice_counts']))} of "
            f"{cache.n_items} items ever chosen"
        )
        print(f"  report -> {json_path}")
        print(f"  plot   -> {plot}")
        print(f"  policy -> {report['artifacts']['final_policy']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
