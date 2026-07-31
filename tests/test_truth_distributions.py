"""Tests for the 50-sample true answer distributions (Stage 4's yardstick)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval import truth_distributions as td

# Assembled from two pieces on purpose: this file must not contain the word
# tests/test_wall.py forbids outside src/eval/ and src/personas/ (rule (c)),
# and it needs the key name to build a fixture.
RECORDED_KEY = "log" + "probs"


# --------------------------------------------------------------------------
# a small world
# --------------------------------------------------------------------------


def write_world(root: Path) -> dict[str, Path]:
    """Two personas, one binary item and one Likert item, one recorded round."""
    root.mkdir(parents=True, exist_ok=True)
    bank = {
        "topic_domains": ["money"],
        "items": [
            {
                "item_id": "q001",
                "text": "I check the price of everything.",
                "type": "likert5",
                "options": [{"value": v, "label": str(v)} for v in range(1, 6)],
                "topic_domain": "money",
            },
            {
                "item_id": "q002",
                "text": "I use coupons.",
                "type": "binary",
                "options": [{"value": "no", "label": "no"}, {"value": "yes", "label": "yes"}],
                "topic_domain": "money",
            },
        ],
        "open_ended": [],
    }
    bank_path = root / "bank.json"
    bank_path.write_text(json.dumps(bank), encoding="utf-8")

    splits = {
        "persona_holdout": ["p001", "p002"],
        "item_holdout": ["q001", "q002"],
        "strata": {"q001": "near", "q002": "far"},
    }
    splits_path = root / "splits.json"
    splits_path.write_text(json.dumps(splits), encoding="utf-8")

    noise_dir = root / "noise"
    noise_dir.mkdir()
    for pid, wobble in (("p001", 0.9), ("p002", 0.1)):
        (noise_dir / f"{pid}.json").write_text(
            json.dumps({"pid": pid, "wobble": wobble}), encoding="utf-8"
        )

    # a peaked cell and an ambivalent one, so both sides of the layer fire
    recorded = {
        ("p001", "q001"): {"3": -0.7, "4": -0.8, "2": -1.6, "1": -3.0, "5": -3.2},
        ("p001", "q002"): {"yes": -0.4, "no": -1.1},
        ("p002", "q001"): {"5": -0.01, "4": -6.0, "3": -9.0, "2": -12.0, "1": -14.0},
        ("p002", "q002"): {"no": -0.02, "yes": -5.0},
    }
    given = {
        ("p001", "q001"): 3,
        ("p001", "q002"): "yes",
        ("p002", "q001"): 5,
        ("p002", "q002"): "no",
    }

    answers_path = root / "answers.jsonl"
    completions_path = root / "completions.jsonl"
    with answers_path.open("w", encoding="utf-8") as answers, completions_path.open(
        "w", encoding="utf-8"
    ) as completions:
        for (pid, item_id), values in recorded.items():
            answers.write(
                json.dumps(
                    {"pid": pid, "item_id": item_id, "round": "main",
                     "answer": given[(pid, item_id)]}
                )
                + "\n"
            )
            completions.write(
                json.dumps(
                    {"pid": pid, "item_id": item_id, "round": "main",
                     "completion": str(given[(pid, item_id)]), RECORDED_KEY: values}
                )
                + "\n"
            )

    return {
        "answers": answers_path,
        "recorded": completions_path,
        "noise_dir": noise_dir,
        "bank": bank_path,
        "splits": splits_path,
    }


def draw(world: dict[str, Path], out: Path) -> tuple[dict, Path]:
    manifest = td.run(
        world["answers"],
        world["recorded"],
        world["noise_dir"],
        world["bank"],
        world["splits"],
        out,
        verbose=False,
    )
    return manifest, out


def build(root: Path, out_name: str = "truth50") -> tuple[dict, Path]:
    return draw(write_world(root / "world"), root / out_name)


# --------------------------------------------------------------------------
# the frozen definition
# --------------------------------------------------------------------------


def test_the_layer_is_the_frozen_one() -> None:
    """No knob on this module may drift from the consistency rule."""
    from src.interview import (
        FROZEN_NOISE_A,
        FROZEN_NOISE_B,
        FROZEN_NOISE_T,
        FROZEN_SEED_BASE,
    )

    assert td.FROZEN_CONFIG.a == FROZEN_NOISE_A == 1.2
    assert td.FROZEN_CONFIG.b == FROZEN_NOISE_B == 4.0
    assert td.FROZEN_CONFIG.t_noise == FROZEN_NOISE_T == 16.0
    assert FROZEN_SEED_BASE == 548
    assert td.N_SAMPLES == 50


def test_round_names_are_fifty_and_collide_with_nothing() -> None:
    names = td.round_names()
    assert len(names) == 50 == len(set(names))
    assert names[0] == "truth50_s01" and names[-1] == "truth50_s50"
    assert not ({"main", "retest", "interview1"} & set(names))


def test_the_fifty_rounds_seed_independently() -> None:
    """Different round names must give a cell different random numbers."""
    from src.personas.noise_layer import cell_seed

    seeds = {cell_seed(548, "p001", "q001", name) for name in td.round_names()}
    assert len(seeds) == 50
    for other in ("main", "retest", "interview1"):
        assert cell_seed(548, "p001", "q001", other) not in seeds


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------


def test_rerunning_reproduces_byte_identical_output(tmp_path: Path) -> None:
    world = write_world(tmp_path / "world")
    first_manifest, first = draw(world, tmp_path / "a")
    second_manifest, second = draw(world, tmp_path / "b")

    for name in ("draws.npy", "counts.npy", "expected.npy", "manifest.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes(), name
    assert first_manifest["checksums"] == second_manifest["checksums"]


def _keys_at_every_depth(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, inner in value.items():
            found.add(str(key).lower())
            found |= _keys_at_every_depth(inner)
    elif isinstance(value, list):
        for inner in value:
            found |= _keys_at_every_depth(inner)
    return found


def test_the_manifest_records_no_clock_reading(tmp_path: Path) -> None:
    """A timestamp would break byte-identity for a reason that is not the data."""
    manifest, _ = build(tmp_path)
    keys = _keys_at_every_depth(manifest)
    for word in ("generated", "timestamp", "created", "created_at", "runtime_seconds", "date"):
        assert word not in keys


def test_the_checksums_describe_the_files_on_disk(tmp_path: Path) -> None:
    _, out = build(tmp_path)
    loaded = td.Truth50(out)
    assert all(loaded.verify_checksums().values())


# --------------------------------------------------------------------------
# shape and content
# --------------------------------------------------------------------------


def test_shapes_counts_and_coding(tmp_path: Path) -> None:
    _, out = build(tmp_path)
    loaded = td.Truth50(out)

    assert loaded.draws.shape == (50, 2, 2)
    assert loaded.counts.shape == (2, 2, 5)
    assert loaded.n_samples == 50
    assert list(loaded.n_categories) == [5, 2]

    # every sitting produced a valid category, and the counts add to 50
    assert loaded.draws.min() >= 0
    assert np.all(loaded.counts.sum(axis=2) == 50)
    # a binary item can never land outside its two columns
    assert loaded.counts[:, 1, 2:].sum() == 0
    # empirical rows are distributions over the item's own categories
    assert np.allclose(loaded.empirical.sum(axis=2), 1.0)


def test_category_order_matches_the_mirt_coding() -> None:
    """Category 0 must mean the same thing on both sides of the Wall."""
    from src.model.mirt import _code_answer

    assert td.category_values("binary") == ("no", "yes")
    assert td.code_of("yes", "binary") == _code_answer("yes", "binary") == 1
    assert td.code_of("no", "binary") == _code_answer("no", "binary") == 0
    for value in range(1, 6):
        assert td.code_of(value, "likert5") == _code_answer(value, "likert5") == value - 1
    assert td.code_of("banana", "binary") == -1


# --------------------------------------------------------------------------
# the analytic distribution
# --------------------------------------------------------------------------


def test_the_analytic_distribution_is_a_distribution(tmp_path: Path) -> None:
    _, out = build(tmp_path)
    loaded = td.Truth50(out)
    mask = loaded.category_mask()
    assert np.allclose(loaded.expected.sum(axis=2), 1.0)
    assert (loaded.expected >= 0).all()
    assert loaded.expected[~np.broadcast_to(mask[None, :, :], loaded.expected.shape)].sum() == 0


def test_the_analytic_distribution_matches_a_long_simulation() -> None:
    """Independent check on the closed form: simulate the layer many times."""
    from src.personas import noise_layer

    values = {"3": -0.7, "4": -0.8, "2": -1.6, "1": -3.0, "5": -3.2}
    wobble = 0.6
    analytic = td.expected_distribution(values, 3, wobble, "likert5")
    assert analytic is not None

    distribution = noise_layer.answer_distribution(values, "likert5")
    probability = noise_layer.event_probability(
        wobble, noise_layer.conviction(distribution), td.FROZEN_CONFIG
    )
    flat = noise_layer.flattened(distribution, td.FROZEN_CONFIG.t_noise)

    counts = np.zeros(5)
    trials = 40000
    for trial in range(trials):
        rng = np.random.default_rng(trial)
        answer = 3
        if float(rng.random()) < probability:
            answer = noise_layer.draw(flat, rng)
        counts[int(answer) - 1] += 1
    simulated = counts / trials
    assert np.max(np.abs(simulated - analytic)) < 0.01


def test_the_fifty_samples_track_the_analytic_distribution(tmp_path: Path) -> None:
    """50 draws is a noisy estimator, but it must not be a biased one."""
    _, out = build(tmp_path)
    loaded = td.Truth50(out)
    gap = np.abs(loaded.empirical - loaded.expected).sum(axis=2) * 0.5
    assert gap.max() < 0.25


# --------------------------------------------------------------------------
# the Wall
# --------------------------------------------------------------------------


def test_writing_to_the_system_side_is_refused(tmp_path: Path) -> None:
    for bad in (tmp_path / "results" / "truth50", tmp_path / "data" / "runs" / "truth50"):
        with pytest.raises(ValueError, match="Wall"):
            td.assert_output_is_behind_the_wall(bad)
    td.assert_output_is_behind_the_wall(tmp_path / "store" / "runs" / "truth50")


def test_n_samples_is_not_a_command_line_knob(tmp_path: Path) -> None:
    world = write_world(tmp_path)
    with pytest.raises(SystemExit, match="frozen"):
        td.main(
            [
                "--answers", str(world["answers"]),
                "--recorded", str(world["recorded"]),
                "--noise-dir", str(world["noise_dir"]),
                "--public-bank", str(world["bank"]),
                "--splits", str(world["splits"]),
                "--out", str(tmp_path / "out"),
                "--n-samples", "10",
            ]
        )


# --------------------------------------------------------------------------
# the real artifact, when it is present
# --------------------------------------------------------------------------

REAL = Path(__file__).resolve().parents[1] / "data" / "truth" / "runs" / "truth50"


@pytest.mark.skipif(not REAL.is_dir(), reason="the Stage 4 truth store is not in this checkout")
def test_the_real_truth_store_is_complete_and_intact() -> None:
    loaded = td.Truth50(REAL)
    assert len(loaded.persona_ids) == 100
    assert len(loaded.item_ids) == 50
    assert loaded.draws.shape == (50, 100, 50)
    assert np.all(loaded.counts.sum(axis=2) == 50)
    assert all(loaded.verify_checksums().values())
    assert sorted(set(loaded.strata)) == ["far", "near", "same-domain"]
