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

REWRITING THE PROMPTS WITHOUT REMINTING THE PEOPLE
--------------------------------------------------
``--regen-cards-only`` rewrites the card prompts for a batch that already
exists, and touches nothing else:

    python -m src.personas.factory --regen-cards-only --n 500 --seed 548 \\
        --out-truth <planted-truth dir> --out-public <public dir>

The people are re-sampled from the same ``--n``/``--seed``/wobble range, and
then every one of them is checked against what is already on disk -- all 8
theta values, the wobble, and the whole public profile -- before a single byte
is written. One mismatch anywhere and the run aborts having written nothing:
the alternative is a batch whose cards describe one population and whose
planted truth is another, which would void the study silently. On success only
``card_prompts/{pid}.json`` and ``card_prompts.jsonl`` are rewritten;
``theta/``, ``noise/``, ``profiles/`` and the manifest are never opened for
writing in this mode.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from .card_prompts import CARD_PROMPT_VERSION, build_card_prompt
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


def _prompt_record(person: Persona) -> dict[str, object]:
    """The per-persona prompt file: the prompt, plus the profile ingest checks.

    The profile travels with the prompt because ingest has to verify that the
    written card used the demographics it was given, and it must do that
    without reaching outside the planted-truth tree.
    """
    prompt = build_card_prompt(person)
    return {
        "pid": person.pid,
        "prompt_version": CARD_PROMPT_VERSION,
        "system": prompt["system"],
        "user": prompt["user"],
        "profile": person.public_profile(),
    }


def _prompt_line(record: dict[str, object]) -> str:
    """One line of the batch file Gemma is pointed at: pid, system, user."""
    return json.dumps(
        {"pid": record["pid"], "system": record["system"], "user": record["user"]},
        ensure_ascii=False,
    )


def write_card_prompts(people: list[Persona], out_truth: Path) -> int:
    """Write the per-persona prompt files and the batch file. Nothing else."""
    out_truth = Path(out_truth)
    lines: list[str] = []
    for person in people:
        record = _prompt_record(person)
        _write_json(out_truth / "card_prompts" / f"{person.pid}.json", record)
        lines.append(_prompt_line(record))

    (out_truth / "card_prompts.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


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

    for person in people:
        _write_json(
            out_truth / "theta" / f"{person.pid}.json",
            {"pid": person.pid, "theta": person.theta, "batch_seed": seed},
        )
        _write_json(
            out_truth / "noise" / f"{person.pid}.json",
            {"pid": person.pid, "wobble": person.wobble},
        )
        _write_json(out_public / "profiles" / f"{person.pid}.json", person.public_profile())

    write_card_prompts(people, out_truth)

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


def _read_json(path: Path) -> object | None:
    """Parsed JSON, or ``None`` when the file is missing or unreadable."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def verify_batch_on_disk(
    people: list[Persona],
    out_truth: Path,
    out_public: Path,
) -> list[str]:
    """Every disagreement between the re-sampled people and the stored batch.

    Empty list means the sample reproduces the batch exactly and the card
    prompts can safely be rewritten in place. Anything else means the seed, the
    size, the wobble range or the sampler itself has moved, and rewriting the
    prompts would attach new text to old planted truth.

    Compared per persona: all 8 theta values (exact equality -- they are stored
    rounded to 4 decimals and JSON round-trips them without loss), the wobble,
    and the entire public profile.
    """
    out_truth = Path(out_truth)
    out_public = Path(out_public)
    problems: list[str] = []

    for person in people:
        theta_record = _read_json(out_truth / "theta" / f"{person.pid}.json")
        if not isinstance(theta_record, dict):
            problems.append(f"{person.pid}: no readable theta file on disk")
        elif theta_record.get("theta") != person.theta:
            stored = theta_record.get("theta")
            differing = sorted(
                dim
                for dim in person.theta
                if not isinstance(stored, dict) or stored.get(dim) != person.theta[dim]
            )
            problems.append(
                f"{person.pid}: sampled theta differs from disk on {', '.join(differing)}"
            )

        noise_record = _read_json(out_truth / "noise" / f"{person.pid}.json")
        if not isinstance(noise_record, dict):
            problems.append(f"{person.pid}: no readable noise file on disk")
        elif noise_record.get("wobble") != person.wobble:
            problems.append(
                f"{person.pid}: sampled wobble {person.wobble} differs from disk "
                f"{noise_record.get('wobble')}"
            )

        profile_record = _read_json(out_public / "profiles" / f"{person.pid}.json")
        if not isinstance(profile_record, dict):
            problems.append(f"{person.pid}: no readable public profile on disk")
        elif profile_record != person.public_profile():
            problems.append(f"{person.pid}: sampled public profile differs from disk")

    return problems


def regen_card_prompts(
    people: list[Persona],
    out_truth: Path,
    out_public: Path,
) -> tuple[int, list[str]]:
    """Verify the batch on disk, then rewrite only the card prompts.

    Returns ``(written, problems)``. When ``problems`` is non-empty nothing was
    written at all -- the check runs to completion over every persona before any
    file is opened, so a mismatch on the last persona still stops the first.
    """
    problems = verify_batch_on_disk(people, out_truth, out_public)
    if problems:
        return 0, problems
    return write_card_prompts(people, out_truth), []


#: How many mismatches the abort message prints before summarising the rest.
_PROBLEMS_SHOWN = 10


def _regen_main(people: list[Persona], args: argparse.Namespace, wobble_range: tuple[float, float]) -> int:
    """The ``--regen-cards-only`` path. Returns the process exit code."""
    written, problems = regen_card_prompts(people, args.out_truth, args.out_public)

    if problems:
        print(
            f"REFUSING to rewrite the card prompts: the population sampled from "
            f"--n {args.n} --seed {args.seed} (wobble {wobble_range[0]}-"
            f"{wobble_range[1]}) does not match the batch on disk. "
            f"{len(problems)} mismatch(es); nothing was written."
        )
        for problem in problems[:_PROBLEMS_SHOWN]:
            print(f"  - {problem}")
        if len(problems) > _PROBLEMS_SHOWN:
            print(f"  ... and {len(problems) - _PROBLEMS_SHOWN} more")
        return 1

    print(
        f"verified {len(people)} personas against the batch on disk "
        f"(theta, wobble, public profile all identical)"
    )
    print(f"rewrote {written} card prompts at version {CARD_PROMPT_VERSION}")
    print(f"  per-persona prompts -> {Path(args.out_truth) / 'card_prompts'}")
    print(f"  batch file for Gemma -> {Path(args.out_truth) / 'card_prompts.jsonl'}")
    print("  planted truth and public profiles untouched")
    return 0


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
    parser.add_argument(
        "--regen-cards-only",
        action="store_true",
        help=(
            "rewrite the card prompts for a batch that already exists and "
            "nothing else. Re-samples the same people from --n/--seed/--wobble-*, "
            "checks every theta, wobble and public profile against what is on "
            "disk, and refuses to write anything at all if one of them differs. "
            "theta/, noise/, profiles/ and the manifest are never written in "
            "this mode."
        ),
    )
    args = parser.parse_args(argv)

    wobble_range = (args.wobble_min, args.wobble_max)
    people = sample_population(args.n, args.seed, wobble_range=wobble_range)

    if args.regen_cards_only:
        return _regen_main(people, args, wobble_range)

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
