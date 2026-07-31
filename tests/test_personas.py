"""Tests for M0, the persona factory.

Six things are checked here, in order of how badly they would hurt if wrong:

  1. The planted population matches the frozen pre-registration -- the exact
     correlation matrix, and a large sample that actually comes out with it.
  2. The batch is reproducible from its seed, bit for bit.
  3. Nothing about the planted truth reaches a public file.
  4. Card completions come back in cleanly, and a leaking card is refused.
  5. Both card-writing passes hand the model no label it could echo back.
  6. Rewriting the card prompts cannot touch the planted truth, and refuses to
     run at all when the sampled population no longer matches what is on disk.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.personas import build_selfcheck_batch, card_prompts, factory, ingest, sampler

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


# --------------------------------------------------------------------------
# 7. the two-pass card prompts (PRD section 10 fallback)
# --------------------------------------------------------------------------

SYNTHETIC_CARD = {
    "pid": "p0001",
    "biography": GOOD_BIOGRAPHY,
    "quirks": ["counts change twice", "keeps a paper diary"],
    "speech_style": "slow, concrete, circles back to the same three stories",
    "age": 43,
    "occupation": "school administrator",
    "city_size": "mid-size city",
    "household": "with partner and kids",
    "region_type": "suburban",
}


def assert_no_label_the_model_could_echo(prompt: dict, where: str) -> None:
    """Neither turn of a prompt may carry a dimension code, name or trait phrase.

    Models reuse the words they are given, so a banned phrase written into the
    prompt -- even inside its own ban -- is a phrase the card can come back
    with, and ingest would then reject that card as a leak.
    """
    for turn in ("system", "user"):
        text = prompt[turn]
        assert ingest.leak_reasons(text) == [], f"{where}/{turn}: {ingest.leak_reasons(text)}"

        lowered = text.lower()
        for code, name in sampler.DIMENSION_NAMES.items():
            assert not re.search(rf"\b{code}\b", text), f"{where}/{turn} names the code {code}"
            assert name not in lowered, f"{where}/{turn} names {name!r}"
        for phrase in ingest._EXTRA_TRAIT_PHRASES:
            assert phrase not in lowered, f"{where}/{turn} uses trait wording {phrase!r}"


def test_card_prompt_tells_the_writer_the_sides_are_independent() -> None:
    person = sampler.sample_population(1, 42)[0]
    user = card_prompts.build_card_prompt(person)["user"]

    assert card_prompts.INDEPENDENCE_BLOCK in user
    # The block sits after the eight numbered lines and before the rules.
    assert user.index("\n8. ") < user.index(card_prompts.INDEPENDENCE_BLOCK) < user.index("\nRULES")


def test_the_independence_block_de_entangles_the_pair_stage_2_caught() -> None:
    """The two examples that split dealing-with-officials from money habits.

    Both directions are shown on purpose: one example each way, so neither
    combination can be read as the normal one.
    """
    block = card_prompts.INDEPENDENCE_BLOCK.lower()

    trusting_and_reckless = block.index("without a second thought")
    assert "savings into a cousin" in block[trusting_and_reckless:]

    suspicious_and_careful = block.index("are lying to them")
    assert "insured account" in block[suspicious_and_careful:]

    # And at least one example about a different pair entirely.
    assert "same street since birth" in block and "folding phone" in block


def test_the_independence_block_shows_behaviour_only() -> None:
    """No label and no number, or the writer has something to copy."""
    assert ingest.leak_reasons(card_prompts.INDEPENDENCE_BLOCK) == []
    for word in ("trait", "dimension", "score", "scale"):
        assert word not in card_prompts.INDEPENDENCE_BLOCK.lower()


def test_card_prompt_bans_trait_vocabulary() -> None:
    person = sampler.sample_population(1, 42)[0]
    user = card_prompts.build_card_prompt(person)["user"]

    assert "character-labelling vocabulary" in user
    assert "a cautious nature" in user
    assert user.index("character-labelling vocabulary") > user.index("\nRULES")


def test_selfcheck_prompt_carries_the_guidance_the_draft_and_the_fixed_facts() -> None:
    person = sampler.sample_population(1, 42)[0]
    lines = card_prompts.trait_guidance(person.theta)
    prompt = card_prompts.build_selfcheck_prompt(lines, SYNTHETIC_CARD)

    assert set(prompt) == {"system", "user"}
    user = prompt["user"]

    for index, line in enumerate(lines, start=1):
        assert f"{index}. {line}" in user, f"guidance line {index} missing"

    assert SYNTHETIC_CARD["biography"] in user
    assert SYNTHETIC_CARD["speech_style"] in user
    for quirk in SYNTHETIC_CARD["quirks"]:
        assert quirk in user

    for field in card_prompts.FIXED_FACT_FIELDS:
        assert str(SYNTHETIC_CARD[field]) in user, f"fixed fact {field} missing"


def test_selfcheck_prompt_asks_for_the_checks_and_the_whole_card() -> None:
    lines = card_prompts.trait_guidance(sampler.sample_population(1, 42)[0].theta)
    user = card_prompts.build_selfcheck_prompt(lines, SYNTHETIC_CARD)["user"]
    # The instructions are hard-wrapped, so phrases are matched on the text
    # with its line breaks collapsed.
    flat = " ".join(user.split())

    # The checking itself never reaches the output.
    assert "none of it appears in" in flat
    # (a) per-side strength and direction, through details of that side alone.
    assert "in the direction the line states" in flat
    assert "as strongly as the line states" in flat
    assert "belong to that side alone" in flat
    # (b) cross-bleed, named behaviourally rather than by dimension.
    assert "bleed between sides" in flat
    assert "savings account and insure everything" in flat
    assert "only from its own numbered line" in flat
    # (c) rewrite, keeping the fixed facts and every original rule.
    assert "Keep the fixed facts word for word" in flat
    assert "200 to 300 words, third person" in flat
    assert "never name a personality trait" in flat
    assert "character-labelling vocabulary" in flat
    assert "no real brands, no real public figures" in flat
    # (d) the same JSON schema as pass 1, always complete, no commentary.
    assert '{"biography"' in user and '"quirks"' in user and '"speech_style"' in user
    assert "every time, even if you changed nothing" in flat
    assert "No notes, no explanation" in flat


def test_selfcheck_prompt_refuses_a_card_it_cannot_preserve() -> None:
    lines = card_prompts.trait_guidance(sampler.sample_population(1, 42)[0].theta)

    with pytest.raises(ValueError, match="no biography"):
        card_prompts.build_selfcheck_prompt(lines, {**SYNTHETIC_CARD, "biography": "  "})

    with pytest.raises(ValueError, match="missing fixed facts"):
        card_prompts.build_selfcheck_prompt(
            lines, {k: v for k, v in SYNTHETIC_CARD.items() if k != "occupation"}
        )


def test_neither_pass_hands_the_model_a_label_to_echo() -> None:
    """The one rule both prompts live under, checked on a spread of personas."""
    for person in sampler.sample_population(30, 99):
        assert_no_label_the_model_could_echo(card_prompts.build_card_prompt(person), "pass1")

        lines = card_prompts.trait_guidance(person.theta)
        card = {**SYNTHETIC_CARD, "pid": person.pid}
        assert_no_label_the_model_could_echo(
            card_prompts.build_selfcheck_prompt(lines, card), "pass2"
        )


# --------------------------------------------------------------------------
# 8. rewriting the card prompts without reminting the people
# --------------------------------------------------------------------------

SENTINEL = "THIS FILE MUST NOT SURVIVE A SUCCESSFUL REGEN\n"


def _snapshot(root: Path) -> dict[str, bytes]:
    """Every file under a tree, by relative path, as raw bytes."""
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _stale_the_prompts(hidden: Path) -> None:
    """Overwrite the prompt files, so "was it rewritten?" has a real answer."""
    (hidden / "card_prompts.jsonl").write_text(SENTINEL, encoding="utf-8")
    (hidden / "card_prompts" / "p0001.json").write_text(SENTINEL, encoding="utf-8")


@pytest.fixture()
def minted(tmp_path):
    """A small batch on disk, minted the ordinary way."""
    hidden = tmp_path / "hidden"
    public = tmp_path / "public"
    assert (
        factory.main(
            ["--n", "6", "--seed", "42", "--out-truth", str(hidden), "--out-public", str(public)]
        )
        == 0
    )
    return hidden, public


def _regen(hidden: Path, public: Path, seed: str = "42") -> int:
    return factory.main(
        [
            "--regen-cards-only",
            "--n", "6",
            "--seed", seed,
            "--out-truth", str(hidden),
            "--out-public", str(public),
        ]
    )


def test_regen_rewrites_the_prompts_and_nothing_else(minted) -> None:
    hidden, public = minted
    before_hidden, before_public = _snapshot(hidden), _snapshot(public)

    _stale_the_prompts(hidden)
    assert _regen(hidden, public) == 0

    # The prompts came back, byte for byte -- the same seed writes the same
    # text -- and every other file is untouched.
    assert _snapshot(hidden) == before_hidden
    assert _snapshot(public) == before_public

    lines = (hidden / "card_prompts.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 6
    first = json.loads(lines[0])
    assert set(first) == {"pid", "system", "user"}
    assert card_prompts.INDEPENDENCE_BLOCK in first["user"]

    record = json.loads((hidden / "card_prompts" / "p0001.json").read_text(encoding="utf-8"))
    assert record["prompt_version"] == card_prompts.CARD_PROMPT_VERSION
    assert record["profile"]["pid"] == "p0001"


def _perturb_theta(hidden: Path, public: Path) -> None:
    path = hidden / "theta" / "p0003.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["theta"]["TRU"] = round(record["theta"]["TRU"] + 0.5, 4)
    path.write_text(json.dumps(record), encoding="utf-8")


def _perturb_wobble(hidden: Path, public: Path) -> None:
    path = hidden / "noise" / "p0002.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["wobble"] = round(record["wobble"] + 0.05, 2)
    path.write_text(json.dumps(record), encoding="utf-8")


def _perturb_profile(hidden: Path, public: Path) -> None:
    path = public / "profiles" / "p0004.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["age"] = int(record["age"]) + 1
    path.write_text(json.dumps(record), encoding="utf-8")


@pytest.mark.parametrize(
    "perturb,expected",
    [
        (_perturb_theta, "p0003: sampled theta differs from disk on TRU"),
        (_perturb_wobble, "p0002: sampled wobble"),
        (_perturb_profile, "p0004: sampled public profile differs"),
    ],
)
def test_regen_writes_nothing_when_the_batch_on_disk_has_moved(
    minted, capsys, perturb, expected
) -> None:
    """One mismatch anywhere and the whole run aborts, having written nothing."""
    hidden, public = minted
    _stale_the_prompts(hidden)
    perturb(hidden, public)
    before_hidden, before_public = _snapshot(hidden), _snapshot(public)

    assert _regen(hidden, public) == 1

    output = capsys.readouterr().out
    assert "REFUSING" in output and expected in output
    assert _snapshot(hidden) == before_hidden, "regen wrote something after refusing"
    assert _snapshot(public) == before_public
    assert (hidden / "card_prompts.jsonl").read_text(encoding="utf-8") == SENTINEL


def test_regen_refuses_a_seed_that_mints_different_people(minted, capsys) -> None:
    hidden, public = minted
    _stale_the_prompts(hidden)
    before = _snapshot(hidden)

    assert _regen(hidden, public, seed="43") == 1
    assert "REFUSING" in capsys.readouterr().out
    assert _snapshot(hidden) == before


def test_regen_reports_every_mismatch_it_found(minted, capsys) -> None:
    hidden, public = minted
    for pid in ("p0001", "p0002", "p0003", "p0004", "p0005", "p0006"):
        path = hidden / "theta" / f"{pid}.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record["theta"]["ENV"] = 1.9999
        path.write_text(json.dumps(record), encoding="utf-8")

    assert _regen(hidden, public) == 1
    assert "6 mismatch(es)" in capsys.readouterr().out


def test_verify_reports_a_missing_file_rather_than_crashing(minted) -> None:
    hidden, public = minted
    (hidden / "theta" / "p0002.json").unlink()
    people = sampler.sample_population(6, 42)

    problems = factory.verify_batch_on_disk(people, hidden, public)
    assert problems == ["p0002: no readable theta file on disk"]


# --------------------------------------------------------------------------
# 9. building the pass-2 self-check batch
# --------------------------------------------------------------------------


def _empty_card_json() -> str:
    return json.dumps({"biography": "   ", "quirks": [], "speech_style": ""})


@pytest.fixture()
def pass1(minted, tmp_path):
    """A pass-1 completions file with one of every outcome."""
    hidden, _ = minted
    path = _completions_file(
        tmp_path / "pass1.jsonl",
        [
            {"pid": "p0001", "completion": _card_json(GOOD_BIOGRAPHY)},
            {"pid": "p0002", "completion": "I cannot help with that request."},
            {"pid": "p0003", "completion": _empty_card_json()},
            {"pid": "p9999", "completion": _card_json(GOOD_BIOGRAPHY)},
        ],
    )
    return hidden, path


def test_selfcheck_batch_routes_good_lines_and_bad_lines_apart(pass1, tmp_path) -> None:
    hidden, path = pass1
    prompts, retries = build_selfcheck_batch.build_batch(path, hidden)

    assert [record["pid"] for record in prompts] == ["p0001"]
    record = prompts[0]
    assert set(record) == {"pid", "round", "system", "user"}
    assert record["round"] == "selfcheck"

    reasons = {row["pid"]: row["reason"] for row in retries}
    assert set(reasons) == {"p0002", "p0003", "p9999"}
    assert "no JSON object" in reasons["p0002"]
    assert "no biography" in reasons["p0003"]
    assert "no prompt record" in reasons["p9999"]


def test_selfcheck_batch_builds_the_prompt_from_the_planted_theta(pass1) -> None:
    hidden, path = pass1
    prompts, _ = build_selfcheck_batch.build_batch(path, hidden)
    user = prompts[0]["user"]

    theta = json.loads((hidden / "theta" / "p0001.json").read_text(encoding="utf-8"))["theta"]
    for index, line in enumerate(card_prompts.trait_guidance(theta), start=1):
        assert f"{index}. {line}" in user

    assert GOOD_BIOGRAPHY in user
    assert_no_label_the_model_could_echo(prompts[0], "batch")


def test_selfcheck_batch_skips_a_repeated_pid(minted, tmp_path) -> None:
    hidden, _ = minted
    path = _completions_file(
        tmp_path / "pass1.jsonl",
        [
            {"pid": "p0001", "completion": _card_json(GOOD_BIOGRAPHY)},
            {"pid": "p0001", "completion": _card_json(GOOD_BIOGRAPHY)},
        ],
    )
    prompts, retries = build_selfcheck_batch.build_batch(path, hidden)
    assert len(prompts) == 1 and retries == []


def test_selfcheck_batch_survives_a_broken_line(minted, tmp_path) -> None:
    hidden, _ = minted
    path = tmp_path / "pass1.jsonl"
    path.write_text(
        "{not json at all\n"
        + json.dumps({"completion": _card_json(GOOD_BIOGRAPHY)})
        + "\n\n"
        + json.dumps({"pid": "p0001", "completion": _card_json(GOOD_BIOGRAPHY)})
        + "\n",
        encoding="utf-8",
    )
    prompts, retries = build_selfcheck_batch.build_batch(path, hidden)

    assert [record["pid"] for record in prompts] == ["p0001"]
    assert [row["pid"] for row in retries] == [None, None]
    assert "not JSON" in retries[0]["reason"] and "no pid" in retries[1]["reason"]


def test_selfcheck_batch_cli_writes_both_files(pass1, tmp_path, capsys) -> None:
    hidden, path = pass1
    out = tmp_path / "selfcheck_prompts.jsonl"
    retry = tmp_path / "selfcheck_retry.jsonl"

    assert (
        build_selfcheck_batch.main(
            [
                "--pass1", str(path),
                "--truth-dir", str(hidden),
                "--out", str(out),
                "--retry-out", str(retry),
            ]
        )
        == 0
    )

    written = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(written) == 1 and written[0]["round"] == "selfcheck"
    assert len(retry.read_text(encoding="utf-8").strip().splitlines()) == 3
    assert "built 1 self-check prompts" in capsys.readouterr().out
