"""Tests for M0, the persona factory.

Four things are checked here, in order of how badly they would hurt if wrong:

  1. The planted population matches the frozen pre-registration -- the exact
     correlation matrix, and a large sample that actually comes out with it.
  2. The batch is reproducible from its seed, bit for bit.
  3. Nothing about the planted truth reaches a public file.
  4. Card completions come back in cleanly, and a leaking card is refused.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.personas import card_prompts, factory, ingest, sampler

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# 1. the frozen population
# --------------------------------------------------------------------------


def matrix_from_preregistration() -> np.ndarray:
    """Read the correlation matrix straight out of the frozen document."""
    text = (REPO_ROOT / "PREREGISTRATION.md").read_text(encoding="utf-8")
    rows = []
    for dim in sampler.DIMENSIONS:
        found = re.findall(rf"^{dim}((?:\s+[+-]\d\.\d\d){{8}})\s*$", text, flags=re.MULTILINE)
        assert len(found) == 1, f"expected exactly one {dim} row in the document, got {len(found)}"
        rows.append([float(cell) for cell in found[0].split()])
    return np.array(rows)


def test_matrix_matches_the_frozen_document() -> None:
    assert np.array_equal(sampler.correlation_matrix(), matrix_from_preregistration())


def test_matrix_is_a_correlation_matrix() -> None:
    sigma = sampler.correlation_matrix()
    assert sigma.shape == (8, 8)
    assert np.array_equal(sigma, sigma.T), "not symmetric"
    assert np.array_equal(np.diag(sigma), np.ones(8)), "diagonal is not all ones"
    assert np.min(np.linalg.eigvalsh(sigma)) > 0, "not positive definite"
    assert np.max(np.abs(sigma - np.eye(8))) <= 0.40, "some |r| exceeds the frozen 0.40 cap"


def test_dimension_names_cover_every_dimension() -> None:
    assert tuple(sampler.DIMENSION_NAMES) == sampler.DIMENSIONS


# --------------------------------------------------------------------------
# 2. reproducibility
# --------------------------------------------------------------------------


def test_same_seed_gives_the_same_people() -> None:
    first = sampler.sample_population(50, 7)
    second = sampler.sample_population(50, 7)
    assert [p.as_dict() for p in first] == [p.as_dict() for p in second]


def test_different_seeds_give_different_people() -> None:
    assert sampler.sample_population(20, 7) != sampler.sample_population(20, 8)


def test_a_bigger_batch_starts_with_the_same_people() -> None:
    """Draws are per persona, so a batch can be extended without moving anyone."""
    small = sampler.sample_population(10, 3)
    large = sampler.sample_population(25, 3)
    assert [p.as_dict() for p in large[:10]] == [p.as_dict() for p in small]


def test_ids_are_zero_padded_from_one() -> None:
    people = sampler.sample_population(3, 1)
    assert [p.pid for p in people] == ["p0001", "p0002", "p0003"]


def test_wobble_range_is_a_knob_not_a_constant() -> None:
    people = sampler.sample_population(200, 11, wobble_range=(0.05, 0.20))
    values = [p.wobble for p in people]
    assert min(values) >= 0.05 and max(values) <= 0.20
    assert all(round(v, 2) == v for v in values), "wobble must be rounded to 2 decimals"

    default = sampler.sample_population(200, 11)
    assert max(p.wobble for p in default) > 0.20, "the default range should be wider"


@pytest.mark.parametrize("bad", [(0.5, 0.5), (0.5, 0.2), (-0.1, 0.3), (0.2, 1.5)])
def test_bad_wobble_range_is_refused(bad) -> None:
    with pytest.raises(ValueError):
        sampler.sample_population(5, 1, wobble_range=bad)


# --------------------------------------------------------------------------
# 3. the population that actually comes out
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def big_sample() -> np.ndarray:
    return sampler.theta_matrix(sampler.sample_population(100_000, 20260730))


def test_empirical_correlations_match_the_planted_ones(big_sample) -> None:
    empirical = np.corrcoef(big_sample.T)
    deviation = np.abs(empirical - sampler.correlation_matrix())
    worst = np.unravel_index(np.argmax(deviation), deviation.shape)
    assert deviation.max() <= 0.03, (
        f"worst cell {sampler.DIMENSIONS[worst[0]]}-{sampler.DIMENSIONS[worst[1]]} "
        f"off by {deviation.max():.4f}"
    )


def test_marginals_are_standard_normal_after_clipping(big_sample) -> None:
    assert np.abs(big_sample.mean(axis=0)).max() < 0.02
    # Clipping at 2 sigma shaves a little variance off; it should be small.
    assert np.all(big_sample.std(axis=0) > 0.93)


def test_clipping_stays_rare(big_sample) -> None:
    clipped = np.mean(np.abs(big_sample) >= sampler.THETA_CLIP, axis=0)
    assert clipped.max() < 0.06, f"clip fractions {np.round(clipped, 4).tolist()}"


def test_demographics_are_diverse() -> None:
    people = sampler.sample_population(500, 42)
    assert len({p.occupation for p in people}) >= 40
    assert {p.city_size for p in people} == set(sampler.CITY_SIZES)
    assert {p.region_type for p in people} == set(sampler.REGION_TYPES)
    assert {p.household for p in people} == set(sampler.HOUSEHOLDS)
    assert min(p.age for p in people) >= 18 and max(p.age for p in people) <= 79


def test_occupation_list_is_long_and_unique() -> None:
    assert len(sampler.OCCUPATIONS) >= 60
    assert len(set(sampler.OCCUPATIONS)) == len(sampler.OCCUPATIONS)


def test_age_gates_hold() -> None:
    """Nobody is a retired 20-year-old or a student pensioner."""
    for person in sampler.sample_population(1000, 5):
        for field, limits in (
            (person.occupation, sampler._OCCUPATION_AGE_LIMITS),
            (person.household, sampler._HOUSEHOLD_AGE_LIMITS),
        ):
            if field in limits:
                low, high = limits[field]
                assert low <= person.age <= high, f"{field} at age {person.age}"


def test_village_is_never_urban() -> None:
    for person in sampler.sample_population(1000, 6):
        if person.city_size == "village":
            assert person.region_type != "urban"


# --------------------------------------------------------------------------
# 4. the card prompt
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        (1.4, "strong"),
        (-1.2, "strong"),
        (0.9, "clear"),
        (-0.5, "clear"),
        (0.49, "lean"),
        (-0.3, "lean"),
        (0.15, "lean"),
        (0.14, "neutral"),
        (-0.1, "neutral"),
        (0.0, "neutral"),
    ],
)
def test_strength_buckets(value, expected) -> None:
    assert card_prompts.strength_bucket(value) == expected


def test_guidance_mirrors_for_negative_values() -> None:
    high = card_prompts.trait_guidance_line("RSK", 1.5)
    low = card_prompts.trait_guidance_line("RSK", -1.5)
    assert high.startswith("Very strongly") and low.startswith("Very strongly")
    assert card_prompts.TRAIT_POLES["RSK"]["high"] in high
    assert card_prompts.TRAIT_POLES["RSK"]["low"] in low
    assert "ambivalence" in card_prompts.trait_guidance_line("RSK", 0.1)


def test_a_small_lean_still_has_a_direction() -> None:
    """+0.3 and -0.3 must not read the same: a lean is signal, not noise."""
    up = card_prompts.trait_guidance_line("RSK", 0.3)
    down = card_prompts.trait_guidance_line("RSK", -0.3)

    assert up != down
    assert "leans one way" in up and "leans one way" in down
    # The side named first -- the one the details should tilt toward -- flips.
    assert up.index(card_prompts.TRAIT_POLES["RSK"]["high"]) < up.index(
        card_prompts.TRAIT_POLES["RSK"]["low"]
    )
    assert down.index(card_prompts.TRAIT_POLES["RSK"]["low"]) < down.index(
        card_prompts.TRAIT_POLES["RSK"]["high"]
    )


def test_a_genuinely_neutral_trait_reads_the_same_both_ways() -> None:
    """Below 0.15 there is no lean to state, so the wording is bidirectional."""
    up = card_prompts.trait_guidance_line("RSK", 0.1)
    down = card_prompts.trait_guidance_line("RSK", -0.1)

    assert up == down
    assert "no lean either way" in up


def test_prompt_text_changes_with_the_sign_of_a_small_trait() -> None:
    """The whole prompt, not just the line, differs for +0.3 and -0.3."""
    person = sampler.sample_population(1, 42)[0]
    up = replace(person, theta={**person.theta, "RSK": 0.3})
    down = replace(person, theta={**person.theta, "RSK": -0.3})

    assert card_prompts.build_card_prompt(up)["user"] != card_prompts.build_card_prompt(down)["user"]


def test_card_prompt_carries_the_demographics_and_all_eight_sides() -> None:
    person = sampler.sample_population(1, 42)[0]
    prompt = card_prompts.build_card_prompt(person)

    assert set(prompt) == {"system", "user"}
    for value in (str(person.age), person.occupation, person.city_size, person.household, person.region_type):
        assert value in prompt["user"], f"{value!r} missing from the prompt"
    assert len(card_prompts.trait_guidance(person.theta)) == 8
    for index in range(1, 9):
        assert f"\n{index}. " in prompt["user"]
    assert "JSON" in prompt["user"] and "biography" in prompt["user"]


def test_card_prompt_never_shows_a_trait_name_or_a_number() -> None:
    """The prompt is truth-adjacent, but it hands the model nothing to copy."""
    for person in sampler.sample_population(30, 99):
        user = card_prompts.build_card_prompt(person)["user"]
        assert ingest.leak_reasons(user) == []


def test_noise_instruction_scales_with_wobble() -> None:
    quiet = card_prompts.build_noise_instruction(0.10)
    loud = card_prompts.build_noise_instruction(0.45)
    assert "1 in 10" in quiet and "4 in 10" in loud
    assert "real person" in quiet
    with pytest.raises(ValueError):
        card_prompts.build_noise_instruction(1.4)


# --------------------------------------------------------------------------
# 5. the batch on disk, and the Wall between the two trees
# --------------------------------------------------------------------------


@pytest.fixture()
def batch(tmp_path):
    """A small batch written to disk, exactly as the CLI would write it."""
    hidden = tmp_path / "hidden"
    public = tmp_path / "public"
    factory.main(
        [
            "--n", "12",
            "--seed", "42",
            "--out-truth", str(hidden),
            "--out-public", str(public),
        ]
    )
    people = sampler.sample_population(12, 42)
    return hidden, public, people


def test_factory_writes_every_file(batch) -> None:
    hidden, public, people = batch
    for person in people:
        assert (hidden / "theta" / f"{person.pid}.json").is_file()
        assert (hidden / "noise" / f"{person.pid}.json").is_file()
        assert (hidden / "card_prompts" / f"{person.pid}.json").is_file()
        assert (public / "profiles" / f"{person.pid}.json").is_file()

    theta = json.loads((hidden / "theta" / "p0001.json").read_text(encoding="utf-8"))
    assert theta == {"pid": "p0001", "theta": people[0].theta, "batch_seed": 42}
    noise = json.loads((hidden / "noise" / "p0001.json").read_text(encoding="utf-8"))
    assert noise == {"pid": "p0001", "wobble": people[0].wobble}

    manifest = json.loads((hidden / "batch_manifest.json").read_text(encoding="utf-8"))
    assert manifest["n"] == 12
    assert manifest["seed"] == 42
    assert manifest["wobble_range"] == list(sampler.DEFAULT_WOBBLE_RANGE)
    assert manifest["sampler_version"] == sampler.SAMPLER_VERSION
    assert manifest["date"]

    lines = (hidden / "card_prompts.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 12
    first = json.loads(lines[0])
    assert set(first) == {"pid", "system", "user"}
    assert first["pid"] == "p0001"


def test_public_profiles_carry_demographics_only(batch) -> None:
    hidden, public, people = batch
    for person in people:
        profile = json.loads((public / "profiles" / f"{person.pid}.json").read_text(encoding="utf-8"))
        assert profile == {
            "pid": person.pid,
            "age": person.age,
            "occupation": person.occupation,
            "city_size": person.city_size,
            "household": person.household,
            "region_type": person.region_type,
        }


def test_no_planted_truth_leaks_into_any_public_file(batch) -> None:
    hidden, public, people = batch
    by_pid = {person.pid: person for person in people}

    for path in sorted(public.rglob("*")):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        lowered = text.lower()

        for word in ("theta", "wobble", "noise", "trait"):
            assert word not in lowered, f"{path.name} mentions {word!r}"
        for code, name in sampler.DIMENSION_NAMES.items():
            assert code not in text, f"{path.name} mentions the code {code}"
            assert name not in lowered, f"{path.name} mentions {name!r}"

        person = by_pid[path.stem]
        for value in person.theta.values():
            for rendering in (str(value), f"{value:.2f}", f"{value:.1f}"):
                assert rendering not in text, f"{path.name} contains a trait value"
        assert str(person.wobble) not in text


def test_the_factory_has_no_default_for_the_hidden_tree(tmp_path) -> None:
    """The path must come from the command line; there is nowhere else to get it."""
    with pytest.raises(SystemExit):
        factory.main(["--n", "2", "--seed", "1", "--out-public", str(tmp_path / "public")])


# --------------------------------------------------------------------------
# 6. reading Gemma's cards back in
# --------------------------------------------------------------------------

GOOD_BIOGRAPHY = (
    "Marek is 43 and has been a school administrator for eleven years, most of "
    "them in the same mid-size city where he grew up. He walks the same suburban "
    "route to work every morning, greeting the baker by name. He lives with "
    "partner and kids in a flat his father helped him buy, and he has never "
    "seriously considered moving away. When the municipality resurfaced his "
    "street he wrote a short thank-you note to the office, which his colleagues "
    "found funny. He keeps his savings in a plain account because the stock "
    "market feels like a casino to him, and he still prints his tax forms "
    "instead of filing them online. On Sundays he hosts a card game that has "
    "run for a decade, and he checks the price of coffee across three shops "
    "before buying a bag. He sorts his recycling carefully but grumbles about "
    "the new bin rules, which he thinks were designed by someone who has never "
    "carried a bin down four flights of stairs. His mother still cooks Sunday "
    "lunch and he would not dream of missing it. Two years ago the school asked "
    "him to run a new attendance system; he insisted on keeping the paper "
    "register alongside it for a full year before he trusted the software."
)


def _completions_file(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    return path


def _card_json(biography: str) -> str:
    return json.dumps(
        {
            "biography": biography,
            "quirks": ["counts change twice", "keeps a paper diary"],
            "speech_style": "slow, concrete, circles back to the same three stories",
        }
    )


@pytest.fixture()
def hidden_batch(tmp_path):
    """A batch on disk whose demographics match GOOD_BIOGRAPHY for p0001."""
    hidden = tmp_path / "hidden"
    public = tmp_path / "public"
    people = sampler.sample_population(4, 42)
    factory.write_batch(people, 42, hidden, public, sampler.DEFAULT_WOBBLE_RANGE)

    # Rewrite p0001's prompt record with the demographics the sample biography
    # above actually describes, so the soft demographic check has something
    # honest to check against.
    record_path = hidden / "card_prompts" / "p0001.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["profile"] = {
        "pid": "p0001",
        "age": 43,
        "occupation": "school administrator",
        "city_size": "mid-size city",
        "household": "with partner and kids",
        "region_type": "suburban",
    }
    record_path.write_text(json.dumps(record), encoding="utf-8")
    return hidden


def test_plain_json_completion_round_trips(hidden_batch, tmp_path) -> None:
    path = _completions_file(
        tmp_path / "c.jsonl", [{"pid": "p0001", "completion": _card_json(GOOD_BIOGRAPHY)}]
    )
    summary = ingest.ingest_file(path, hidden_batch)

    assert summary["kept"] == 1 and summary["rejected"] == 0
    assert summary["warnings"] == [], summary["warnings"]

    card = json.loads((hidden_batch / "cards" / "p0001.json").read_text(encoding="utf-8"))
    assert card["pid"] == "p0001"
    assert card["biography"] == GOOD_BIOGRAPHY
    assert len(card["quirks"]) == 2
    assert card["speech_style"].startswith("slow")
    assert card["age"] == 43 and card["occupation"] == "school administrator"


def test_fenced_completion_round_trips(hidden_batch, tmp_path) -> None:
    raw = "```json\n" + _card_json(GOOD_BIOGRAPHY) + "\n```"
    path = _completions_file(tmp_path / "c.jsonl", [{"pid": "p0001", "completion": raw}])
    summary = ingest.ingest_file(path, hidden_batch)
    assert summary["kept"] == 1, summary["rejects"]
    assert (hidden_batch / "cards" / "p0001.json").is_file()


def test_prose_wrapped_completion_round_trips(hidden_batch, tmp_path) -> None:
    raw = (
        "Sure! Here is the persona card you asked for.\n\n"
        + _card_json(GOOD_BIOGRAPHY)
        + "\n\nLet me know if you would like a different tone."
    )
    path = _completions_file(tmp_path / "c.jsonl", [{"pid": "p0001", "completion": raw}])
    summary = ingest.ingest_file(path, hidden_batch)
    assert summary["kept"] == 1, summary["rejects"]


def test_a_card_that_names_a_trait_is_rejected(hidden_batch, tmp_path) -> None:
    leaking = GOOD_BIOGRAPHY + " His risk appetite is low."
    path = _completions_file(
        tmp_path / "c.jsonl", [{"pid": "p0001", "completion": _card_json(leaking)}]
    )
    summary = ingest.ingest_file(path, hidden_batch)

    assert summary["kept"] == 0 and summary["rejected"] == 1
    assert "risk appetite" in summary["rejects"][0]["reason"]
    assert not (hidden_batch / "cards" / "p0001.json").exists()
    rejects = (hidden_batch / "card_rejects.jsonl").read_text(encoding="utf-8").strip()
    assert '"p0001"' in rejects


@pytest.mark.parametrize(
    "snippet",
    [
        " He scores +1.4 on that.",
        " Tech optimism: -0.9 for him.",
        " His TRU is high.",
        " Institution trust runs deep in him.",
        " He is high on locality attachment.",
    ],
)
def test_other_leaks_are_caught(snippet) -> None:
    assert ingest.leak_reasons(GOOD_BIOGRAPHY + snippet)


@pytest.mark.parametrize(
    "snippet",
    [
        " He is 43 and has 2 children.",
        " He cycles 1.5 km to the shop.",
        " He paid 1.2 million for the workshop.",
        " It was -2 degrees that morning.",
        " He waited 1-2 years before replacing it.",
    ],
)
def test_ordinary_numbers_are_not_leaks(snippet) -> None:
    assert ingest.leak_reasons(GOOD_BIOGRAPHY + snippet) == []


def test_unparseable_completion_is_rejected(hidden_batch, tmp_path) -> None:
    path = _completions_file(
        tmp_path / "c.jsonl",
        [
            {"pid": "p0001", "completion": "I cannot help with that request."},
            {"pid": "p0002", "completion": ""},
            {"pid": "p9999", "completion": _card_json(GOOD_BIOGRAPHY)},
        ],
    )
    summary = ingest.ingest_file(path, hidden_batch)

    assert summary["kept"] == 0 and summary["rejected"] == 3
    reasons = {row["pid"]: row["reason"] for row in summary["rejects"]}
    assert "no JSON object" in reasons["p0001"]
    assert "empty completion" in reasons["p0002"]
    assert "no prompt record" in reasons["p9999"]


def test_short_biography_is_a_warning_not_a_rejection(hidden_batch, tmp_path) -> None:
    short = "Marek is 43, a school administrator in a mid-size city. He lives with partner and kids in a suburban street."
    path = _completions_file(
        tmp_path / "c.jsonl", [{"pid": "p0001", "completion": _card_json(short)}]
    )
    summary = ingest.ingest_file(path, hidden_batch)

    assert summary["kept"] == 1 and summary["with_warnings"] == 1
    assert any("words" in w for w in summary["warnings"][0]["warnings"])
    assert (hidden_batch / "cards" / "p0001.json").is_file()


def test_missing_demographics_warn(hidden_batch, tmp_path) -> None:
    vague = (
        "He is in his forties and works at a school of some kind. " * 3
        + "He walks a lot and keeps to a routine he has held for years. " * 12
    )
    path = _completions_file(
        tmp_path / "c.jsonl", [{"pid": "p0001", "completion": _card_json(vague)}]
    )
    summary = ingest.ingest_file(path, hidden_batch)

    assert summary["kept"] == 1
    warnings = " ".join(summary["warnings"][0]["warnings"])
    assert "age 43" in warnings and "occupation" in warnings


def test_ingest_cli_runs(hidden_batch, tmp_path, capsys) -> None:
    path = _completions_file(
        tmp_path / "c.jsonl", [{"pid": "p0001", "completion": _card_json(GOOD_BIOGRAPHY)}]
    )
    assert ingest.main(["--completions", str(path), "--out-truth", str(hidden_batch)]) == 0
    assert "kept 1 cards" in capsys.readouterr().out
