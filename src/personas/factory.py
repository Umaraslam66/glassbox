"""Mint a batch of personas: sample, split, write to disk.

    python -m src.personas.factory --n 500 --seed 42 \\
        --out-truth <planted-truth dir> --out-public <public dir>

Two output trees, and the split between them is the Wall:

  planted-truth dir           public dir
    theta/{pid}.json            profiles/{pid}.json
    noise/{pid}.json
    card_prompts/{pid}.json
    card_prompts.jsonl
    batch_manifest.json

WALL RULE: the location of the planted-truth tree is never written down in
source. It arrives as a command-line argument, and nothing here has a default
for it. ``tests/test_wall.py`` fails the build if any module outside
``src/eval/`` so much as mentions the path.

The public profile carries demographics and nothing else -- no trait value, no
trait name, no wobble. ``tests/test_personas.py`` reads every public file back
and checks that.

``card_prompts.jsonl`` is the batch file a Leonardo job renders with Gemma: one
line per persona, ``{"pid", "system", "user"}``. Its completions come back
through ``src.personas.ingest``.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from .card_prompts import build_card_prompt
from .sampler import (
    DEFAULT_WOBBLE_RANGE,
    DIMENSIONS,
    SAMPLER_VERSION,
    THETA_CLIP,
    Persona,
    sample_population,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_batch(
    people: list[Persona],
    seed: int,
    out_truth: Path,
    out_public: Path,
    wobble_range: tuple[float, float],
) -> dict[str, object]:
    """Write one sampled batch to the two trees. Returns the manifest."""
    out_truth = Path(out_truth)
    out_public = Path(out_public)

    prompt_lines: list[str] = []
    for person in people:
        prompt = build_card_prompt(person)

        _write_json(
            out_truth / "theta" / f"{person.pid}.json",
            {"pid": person.pid, "theta": person.theta, "batch_seed": seed},
        )
        _write_json(
            out_truth / "noise" / f"{person.pid}.json",
            {"pid": person.pid, "wobble": person.wobble},
        )
        # The prompt record carries the public profile too: ingest needs it to
        # check that the written card actually used the demographics it was
        # given, and it must do that without reaching outside this tree.
        _write_json(
            out_truth / "card_prompts" / f"{person.pid}.json",
            {
                "pid": person.pid,
                "system": prompt["system"],
                "user": prompt["user"],
                "profile": person.public_profile(),
            },
        )
        _write_json(out_public / "profiles" / f"{person.pid}.json", person.public_profile())

        prompt_lines.append(
            json.dumps(
                {"pid": person.pid, "system": prompt["system"], "user": prompt["user"]},
                ensure_ascii=False,
            )
        )

    (out_truth / "card_prompts.jsonl").write_text(
        "\n".join(prompt_lines) + "\n", encoding="utf-8"
    )

    manifest = {
        "n": len(people),
        "seed": seed,
        "wobble_range": [wobble_range[0], wobble_range[1]],
        "sampler_version": SAMPLER_VERSION,
        "dimensions": list(DIMENSIONS),
        "theta_clip": THETA_CLIP,
        "date": date.today().isoformat(),
    }
    _write_json(out_truth / "batch_manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.personas.factory",
        description="Sample a persona population and write the two output trees.",
    )
    parser.add_argument("--n", type=int, required=True, help="how many personas to mint")
    parser.add_argument("--seed", type=int, required=True, help="batch seed; recorded in the manifest")
    parser.add_argument(
        "--out-truth",
        type=Path,
        required=True,
        help=(
            "directory for the planted truth (theta, noise, card prompts). No "
            "default on purpose: this path is supplied per run and never lives "
            "in source. Keep it outside version control."
        ),
    )
    parser.add_argument(
        "--out-public",
        type=Path,
        required=True,
        help="directory for the public profiles, the only persona data the system side may read",
    )
    parser.add_argument(
        "--wobble-min",
        type=float,
        default=DEFAULT_WOBBLE_RANGE[0],
        help=f"low end of the response-noise range (default {DEFAULT_WOBBLE_RANGE[0]})",
    )
    parser.add_argument(
        "--wobble-max",
        type=float,
        default=DEFAULT_WOBBLE_RANGE[1],
        help=f"high end of the response-noise range (default {DEFAULT_WOBBLE_RANGE[1]})",
    )
    args = parser.parse_args(argv)

    wobble_range = (args.wobble_min, args.wobble_max)
    people = sample_population(args.n, args.seed, wobble_range=wobble_range)
    manifest = write_batch(people, args.seed, args.out_truth, args.out_public, wobble_range)

    print(
        f"minted {manifest['n']} personas from seed {manifest['seed']} "
        f"(sampler {manifest['sampler_version']}, wobble "
        f"{wobble_range[0]}-{wobble_range[1]})"
    )
    print(f"  planted truth -> {args.out_truth}")
    print(f"  public profiles -> {args.out_public / 'profiles'}")
    print(f"  card prompts for Gemma -> {args.out_truth / 'card_prompts.jsonl'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
