"""Tests for the Gate 1 population QA metrics (module M4 subset).

Every population here is built by hand so the right answer is known before the
code runs:

  * two identical personas must come out at 100% agreement and must fail the
    diversity bar;
  * answers driven by exactly two factors must give a participation ratio of 2;
  * answers built as sign(loading) x theta must give an obedience correlation
    of ~1, and building them with the keying reversed must flip the sign --
    that is the test that catches a dropped sign-flip;
  * a retest constructed at 80% agreement must be read as 80%, with the binary
    and Likert halves reported separately against their chance levels.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.bank.schema import DIMENSIONS, options_for
from src.eval import population_qa

# --------------------------------------------------------------------------
# helpers for building synthetic populations
# --------------------------------------------------------------------------


def pid_of(index: int) -> str:
    return f"p{index:04d}"


def answer_rows(matrix: dict[str, dict[str, object]], round_name: str = "main") -> list[dict]:
    """{pid: {item_id: answer}} -> answer records."""
    return [
        {"pid": pid, "item_id": item_id, "round": round_name, "answer": answer, "raw": str(answer)}
        for pid, row in matrix.items()
        for item_id, answer in row.items()
    ]


def likert_types(item_ids) -> dict[str, str]:
    return {item_id: "likert5" for item_id in item_ids}


def build_design(items) -> dict[str, dict[str, float]]:
    return {row["item_id"]: dict(row["loadings"]) for row in items}


def build_types(items) -> dict[str, str]:
    return {row["item_id"]: row["type"] for row in items}


def item_plan() -> list[dict]:
    """48 items: per dimension, 4 Likert and 2 binary, half negatively keyed."""
    items: list[dict] = []
    for dim in DIMENSIONS:
        for position in range(6):
            sign = 1.0 if position % 2 == 0 else -1.0
            item_type = "likert5" if position < 4 else "binary"
            items.append(
                {
                    "item_id": f"q{len(items) + 1:03d}",
                    "text": f"Statement {len(items) + 1}.",
                    "type": item_type,
                    "topic_domain": "money",
                    "loadings": {dim: sign},
                    "strength_class": "strong",
                }
            )
    return items


def obedient_population(
    n_personas: int = 40,
    seed: int = 7,
    keying: float = 1.0,
    noise: float = 0.1,
) -> tuple[list[dict], dict[str, dict[str, float]], dict[str, dict[str, float]], dict[str, str], list[dict]]:
    """Answers built straight from theta, so obedience is known to be near 1.

    ``keying = -1`` builds the same population with the sign convention
    reversed: the answers still come from theta, but agreeing now means a *low*
    trait value. Grading it against the real design must flip every
    correlation.
    """
    rng = np.random.default_rng(seed)
    items = item_plan()
    thetas = {
        pid_of(index): {dim: float(rng.uniform(-2.0, 2.0)) for dim in DIMENSIONS}
        for index in range(1, n_personas + 1)
    }

    matrix: dict[str, dict[str, object]] = {}
    for pid, theta in thetas.items():
        row: dict[str, object] = {}
        for item in items:
            dim, loading = next(iter(item["loadings"].items()))
            value = keying * loading * theta[dim]
            if item["type"] == "likert5":
                answer = int(np.clip(round(3 + value), 1, 5))
                if rng.random() < noise:
                    answer = int(np.clip(answer + rng.choice([-1, 1]), 1, 5))
                row[item["item_id"]] = answer
            else:
                answer = "yes" if value > 0 else "no"
                if rng.random() < noise:
                    answer = "no" if answer == "yes" else "yes"
                row[item["item_id"]] = answer
        matrix[pid] = row

    return answer_rows(matrix), build_design(items), thetas, build_types(items), items


# --------------------------------------------------------------------------
# 1. diversity: pairwise agreement
# --------------------------------------------------------------------------


def test_identical_personas_agree_completely_and_fail_the_bar() -> None:
    twin = {f"q{i:03d}": (i % 5) + 1 for i in range(1, 11)}
    different = {f"q{i:03d}": ((i + 2) % 5) + 1 for i in range(1, 11)}
    answers = answer_rows({"p0001": twin, "p0002": dict(twin), "p0003": different})

    report = population_qa.run_qa(answers, likert_types(twin), {}, {})
    diversity = report["diversity"]

    assert diversity["max_pair_agreement"] == 1.0
    assert report["verdict"]["bars"]["diversity_a_no_pair_above_95"]["pass"] is False
    assert report["verdict"]["all_pass"] is False


def test_personas_that_never_agree_score_zero() -> None:
    first = {f"q{i:03d}": 1 for i in range(1, 11)}
    second = {f"q{i:03d}": 5 for i in range(1, 11)}
    answers = answer_rows({"p0001": first, "p0002": second})

    report = population_qa.run_qa(answers, likert_types(first), {}, {})
    assert report["diversity"]["max_pair_agreement"] == 0.0
    assert report["verdict"]["bars"]["diversity_a_no_pair_above_95"]["pass"] is True


def test_agreement_is_exact_match_on_likert_not_within_one() -> None:
    first = {"q001": 3, "q002": 3, "q003": 3, "q004": 3}
    second = {"q001": 3, "q002": 4, "q003": 4, "q004": 4}
    answers = answer_rows({"p0001": first, "p0002": second})

    report = population_qa.run_qa(answers, likert_types(first), {}, {})
    assert report["diversity"]["max_pair_agreement"] == 0.25


def test_binary_agreement_counts_the_same_way() -> None:
    first = {"q001": "yes", "q002": "yes", "q003": "no", "q004": "no"}
    second = {"q001": "yes", "q002": "no", "q003": "no", "q004": "yes"}
    answers = answer_rows({"p0001": first, "p0002": second})
    types = {item_id: "binary" for item_id in first}

    report = population_qa.run_qa(answers, types, {}, {})
    assert report["diversity"]["max_pair_agreement"] == 0.5


def test_the_agreement_histogram_holds_every_pair() -> None:
    answers, design, thetas, types, _ = obedient_population(n_personas=12)
    report = population_qa.run_qa(answers, types, design, thetas)
    diversity = report["diversity"]

    assert diversity["n_pairs"] == 12 * 11 // 2
    assert sum(diversity["histogram"]["counts"]) == diversity["n_pairs"]
    assert len(diversity["histogram"]["bin_edges"]) == len(diversity["histogram"]["counts"]) + 1
    assert diversity["min_pair_agreement"] <= diversity["median_pair_agreement"]
    assert diversity["median_pair_agreement"] <= diversity["max_pair_agreement"]


# --------------------------------------------------------------------------
# 2. diversity: effective dimensionality
# --------------------------------------------------------------------------


def two_factor_answers(n_repeats: int = 4, n_items: int = 12, flip_every: int = 0) -> tuple[list[dict], dict[str, str]]:
    """Answers driven by exactly two orthogonal factors.

    Four persona types cover the four sign combinations of the two factors, so
    the factors are exactly uncorrelated. Even items follow factor one, odd
    items factor two. The item correlation matrix is then two perfect blocks,
    whose eigenvalues are (n_items / 2, n_items / 2) -- a participation ratio
    of exactly 2.
    """
    types_of_person = [(+1, +1), (+1, -1), (-1, +1), (-1, -1)]
    matrix: dict[str, dict[str, object]] = {}
    counter = 0
    for repeat in range(n_repeats):
        for first, second in types_of_person:
            counter += 1
            row: dict[str, object] = {}
            for index in range(n_items):
                factor = first if index % 2 == 0 else second
                answer = 4 if factor > 0 else 2
                if flip_every and (index + counter) % flip_every == 0:
                    answer = 3
                row[f"q{index + 1:03d}"] = answer
            matrix[pid_of(counter)] = row

    item_ids = [f"q{index + 1:03d}" for index in range(n_items)]
    return answer_rows(matrix), likert_types(item_ids)


def test_a_two_factor_population_has_effective_dimensionality_two() -> None:
    answers, types = two_factor_answers()
    report = population_qa.run_qa(answers, types, {}, {})

    assert report["diversity"]["effective_dimensionality"] == pytest.approx(2.0, abs=0.01)
    spectrum = report["diversity"]["eigenvalues"]
    assert spectrum == sorted(spectrum, reverse=True)
    assert spectrum[0] == pytest.approx(6.0, abs=1e-6)
    assert spectrum[1] == pytest.approx(6.0, abs=1e-6)
    assert spectrum[2] == pytest.approx(0.0, abs=1e-9)
    assert report["verdict"]["bars"]["diversity_c_effective_dimensionality"]["pass"] is False


def test_noise_pushes_effective_dimensionality_above_two() -> None:
    answers, types = two_factor_answers(flip_every=7)
    report = population_qa.run_qa(answers, types, {}, {})
    ratio = report["diversity"]["effective_dimensionality"]
    assert 2.0 < ratio < 5.0


def test_independent_items_give_dimensionality_near_the_item_count() -> None:
    rng = np.random.default_rng(0)
    n_items, n_personas = 8, 400
    matrix = {
        pid_of(index): {f"q{item + 1:03d}": int(rng.integers(1, 6)) for item in range(n_items)}
        for index in range(1, n_personas + 1)
    }
    answers = answer_rows(matrix)
    report = population_qa.run_qa(answers, likert_types([f"q{i + 1:03d}" for i in range(n_items)]), {}, {})

    assert report["diversity"]["effective_dimensionality"] == pytest.approx(n_items, rel=0.05)
    assert report["verdict"]["bars"]["diversity_c_effective_dimensionality"]["pass"] is True


def test_items_with_no_variance_are_dropped_not_crashed_on() -> None:
    matrix = {
        "p0001": {"q001": 1, "q002": 3, "q003": 5},
        "p0002": {"q001": 5, "q002": 3, "q003": 1},
        "p0003": {"q001": 3, "q002": 3, "q003": 2},
    }
    report = population_qa.run_qa(answer_rows(matrix), likert_types(["q001", "q002", "q003"]), {}, {})
    diversity = report["diversity"]

    assert diversity["items_dropped_no_variance"] == ["q002"]
    assert diversity["n_items_in_eigendecomposition"] == 2


# --------------------------------------------------------------------------
# 3. trait obedience
# --------------------------------------------------------------------------


def test_answers_built_from_theta_are_obedient() -> None:
    answers, design, thetas, types, _ = obedient_population(noise=0.0)
    report = population_qa.run_qa(answers, types, design, thetas)
    obedience = report["obedience"]

    assert set(obedience["per_dimension"]) == set(DIMENSIONS)
    assert all(value > 0.95 for value in obedience["per_dimension"].values())
    assert obedience["median"] > 0.95
    assert report["verdict"]["bars"]["trait_obedience_median"]["pass"] is True


def test_obedience_survives_a_little_response_noise() -> None:
    answers, design, thetas, types, _ = obedient_population(noise=0.15)
    report = population_qa.run_qa(answers, types, design, thetas)
    assert all(value > 0.85 for value in report["obedience"]["per_dimension"].values())
    assert report["verdict"]["bars"]["trait_obedience_median"]["pass"] is True


def test_reversing_the_keying_flips_every_correlation() -> None:
    """The sign-flip test. Answers built against the opposite keying must score
    about -1, not about +1: that only happens if negatively-loading items are
    genuinely flipped before averaging."""
    answers, design, thetas, types, _ = obedient_population(keying=-1.0, noise=0.0)
    report = population_qa.run_qa(answers, types, design, thetas)
    obedience = report["obedience"]

    assert all(value < -0.95 for value in obedience["per_dimension"].values())
    assert obedience["median"] < -0.95
    assert report["verdict"]["bars"]["trait_obedience_median"]["pass"] is False


def test_shuffled_theta_kills_obedience() -> None:
    answers, design, thetas, types, _ = obedient_population(noise=0.0)
    pids = sorted(thetas)
    rotated = {pid: thetas[pids[(index + 1) % len(pids)]] for index, pid in enumerate(pids)}

    report = population_qa.run_qa(answers, types, design, rotated)
    assert abs(report["obedience"]["median"]) < 0.5
    assert report["verdict"]["bars"]["trait_obedience_median"]["pass"] is False


def test_cross_loading_items_count_towards_both_dimensions() -> None:
    design = {
        "q001": {"TRU": 1.0},
        "q002": {"TRU": 0.7, "RSK": 0.5},
        "q003": {"TRU": 0.25},
        "q004": {},
        "q005": {"RSK": -0.5},
    }
    on_tru = population_qa.items_loading_on(design, "TRU")
    on_rsk = population_qa.items_loading_on(design, "RSK")

    assert sorted(on_tru) == ["q001", "q002"]
    assert sorted(on_rsk) == ["q002", "q005"]
    assert on_rsk["q005"] == -0.5


def test_a_dimension_with_no_items_is_reported_as_missing() -> None:
    matrix = {pid_of(i): {"q001": (i % 5) + 1, "q002": ((i + 1) % 5) + 1} for i in range(1, 11)}
    thetas = {pid_of(i): {dim: float(i) for dim in DIMENSIONS} for i in range(1, 11)}
    design = {"q001": {"TRU": 1.0}, "q002": {"TRU": -1.0}}

    report = population_qa.run_qa(answer_rows(matrix), likert_types(["q001", "q002"]), design, thetas)
    obedience = report["obedience"]

    assert obedience["per_dimension"]["TRU"] is not None
    assert obedience["per_dimension"]["RSK"] is None
    assert obedience["n_dimensions_measured"] == 1


# --------------------------------------------------------------------------
# 4. test-retest
# --------------------------------------------------------------------------


def retest_at(binary_hits: int, likert_hits: int) -> tuple[list[dict], list[dict], dict[str, str]]:
    """10 personas x (5 binary + 5 Likert), with a chosen number of agreements."""
    binary_ids = [f"b{index:02d}" for index in range(5)]
    likert_ids = [f"l{index:02d}" for index in range(5)]
    types = {item_id: "binary" for item_id in binary_ids}
    types.update({item_id: "likert5" for item_id in likert_ids})

    main: dict[str, dict[str, object]] = {}
    retest: dict[str, dict[str, object]] = {}
    binary_left, likert_left = binary_hits, likert_hits

    for index in range(1, 11):
        pid = pid_of(index)
        main[pid] = {}
        retest[pid] = {}
        for item_id in binary_ids:
            main[pid][item_id] = "yes"
            if binary_left > 0:
                retest[pid][item_id] = "yes"
                binary_left -= 1
            else:
                retest[pid][item_id] = "no"
        for item_id in likert_ids:
            main[pid][item_id] = 3
            if likert_left > 0:
                # within +-1 counts as agreement on Likert
                retest[pid][item_id] = 4
                likert_left -= 1
            else:
                retest[pid][item_id] = 1

    return answer_rows(main), answer_rows(retest, "retest"), types


def test_a_retest_built_at_80_percent_reads_as_80_percent() -> None:
    main, retest, types = retest_at(binary_hits=45, likert_hits=35)
    report = population_qa.run_qa(main, types, {}, {}, retest_answers=retest)
    block = report["retest"]

    assert block["n_reasked"] == 100
    assert block["pooled_agreement"] == pytest.approx(0.80)
    assert block["binary"]["agreement"] == pytest.approx(0.90)
    assert block["likert5"]["agreement"] == pytest.approx(0.70)
    assert block["binary"]["n"] == block["likert5"]["n"] == 50
    assert report["verdict"]["bars"]["retest_pooled_band"]["pass"] is True


def test_the_chance_levels_are_the_pre_registered_ones() -> None:
    main, retest, types = retest_at(binary_hits=45, likert_hits=35)
    report = population_qa.run_qa(main, types, {}, {}, retest_answers=retest)

    assert report["retest"]["binary"]["chance"] == 0.5
    assert report["retest"]["likert5"]["chance"] == pytest.approx(13 / 25)
    assert report["retest"]["likert5"]["rule"] == "within +-1"
    assert report["retest"]["binary"]["rule"] == "exact match"


@pytest.mark.parametrize(
    "binary_hits,likert_hits,expected,passes",
    [
        (50, 50, 1.0, False),   # too consistent: a survey machine
        (25, 25, 0.5, False),   # too noisy: coin flipping
        (45, 35, 0.80, True),
        (35, 35, 0.70, True),   # the low edge of the band is inside it
        (45, 45, 0.90, True),   # so is the high edge
    ],
)
def test_the_retest_band_is_inclusive(binary_hits, likert_hits, expected, passes) -> None:
    main, retest, types = retest_at(binary_hits, likert_hits)
    report = population_qa.run_qa(main, types, {}, {}, retest_answers=retest)

    assert report["retest"]["pooled_agreement"] == pytest.approx(expected)
    assert report["verdict"]["bars"]["retest_pooled_band"]["pass"] is passes


def test_per_persona_retest_rates_are_reported() -> None:
    main, retest, types = retest_at(binary_hits=45, likert_hits=35)
    report = population_qa.run_qa(main, types, {}, {}, retest_answers=retest)
    block = report["retest"]

    assert len(block["per_persona"]) == 10
    assert sum(block["per_persona_histogram"]["counts"]) == 10
    assert block["per_persona_median"] is not None
    assert np.mean(list(block["per_persona"].values())) == pytest.approx(0.80)


def test_a_likert_answer_two_steps_away_is_a_disagreement() -> None:
    assert population_qa.agrees(3, 4, "likert5") is True
    assert population_qa.agrees(3, 2, "likert5") is True
    assert population_qa.agrees(3, 5, "likert5") is False
    assert population_qa.agrees("yes", "yes", "binary") is True
    assert population_qa.agrees("yes", "no", "binary") is False


def test_a_retest_answer_with_no_main_answer_is_not_counted() -> None:
    main = answer_rows({"p0001": {"q001": 3}})
    retest = answer_rows({"p0001": {"q001": 3, "q002": 4}}, "retest")
    report = population_qa.run_qa(main, likert_types(["q001", "q002"]), {}, {}, retest_answers=retest)

    assert report["retest"]["n_reasked"] == 1
    assert report["retest"]["n_unmatched"] == 1


def test_without_a_retest_the_bar_is_not_evaluated() -> None:
    answers, design, thetas, types, _ = obedient_population(n_personas=12)
    report = population_qa.run_qa(answers, types, design, thetas)

    assert report["retest"] is None
    bar = report["verdict"]["bars"]["retest_pooled_band"]
    assert bar["pass"] is None
    assert bar["evaluated"] is False
    assert report["verdict"]["all_pass"] is False


# --------------------------------------------------------------------------
# 5. the whole gate, and the CLI
# --------------------------------------------------------------------------


def test_a_healthy_population_passes_every_gate_1_bar() -> None:
    answers, design, thetas, types, _ = obedient_population(n_personas=30, noise=0.1)
    main, retest, retest_types = retest_at(binary_hits=45, likert_hits=35)
    combined_types = {**types, **retest_types}

    report = population_qa.run_qa(answers + main, combined_types, design, thetas, retest_answers=retest)
    bars = report["verdict"]["bars"]

    assert bars["diversity_a_no_pair_above_95"]["pass"] is True
    assert bars["diversity_b_median_agreement"]["pass"] is True
    assert bars["diversity_c_effective_dimensionality"]["pass"] is True
    assert bars["trait_obedience_median"]["pass"] is True
    assert bars["retest_pooled_band"]["pass"] is True
    assert report["verdict"]["all_pass"] is True


def write_run(tmp_path: Path, noise: float = 0.1) -> tuple[Path, Path, Path, Path]:
    """A whole synthetic run on disk: answers, retest, public bank, planted truth."""
    answers, design, thetas, types, items = obedient_population(n_personas=25, noise=noise)
    main_retest, retest, retest_types = retest_at(binary_hits=45, likert_hits=35)

    truth = tmp_path / "planted"
    (truth / "theta").mkdir(parents=True, exist_ok=True)
    for pid, theta in thetas.items():
        (truth / "theta" / f"{pid}.json").write_text(
            json.dumps({"pid": pid, "theta": theta, "batch_seed": 7}), encoding="utf-8"
        )
    (truth / "bank_truth.json").write_text(
        json.dumps({"bank_version": 1, "items": items}), encoding="utf-8"
    )

    extra = [
        {"item_id": item_id, "text": item_id, "type": item_type, "topic_domain": "money"}
        for item_id, item_type in sorted(retest_types.items())
    ]
    public = {
        "bank_version": 1,
        "items": [
            {
                "item_id": row["item_id"],
                "text": row["text"],
                "type": row["type"],
                "options": options_for(row["type"]),
                "topic_domain": row["topic_domain"],
            }
            for row in items + extra
        ],
    }
    public_path = tmp_path / "bank_items.json"
    public_path.write_text(json.dumps(public), encoding="utf-8")

    answers_path = tmp_path / "answers.jsonl"
    answers_path.write_text(
        "".join(json.dumps(row) + "\n" for row in answers + main_retest), encoding="utf-8"
    )
    retest_path = tmp_path / "retest.jsonl"
    retest_path.write_text("".join(json.dumps(row) + "\n" for row in retest), encoding="utf-8")

    return answers_path, retest_path, public_path, truth


def test_the_cli_writes_results_and_prints_a_table(tmp_path: Path, capsys) -> None:
    answers, retest, public_bank, truth = write_run(tmp_path)
    out = tmp_path / "results" / "population_qa.json"

    code = population_qa.main(
        [
            "--answers", str(answers),
            "--truth", str(truth),
            "--public-bank", str(public_bank),
            "--out", str(out),
            "--retest-answers", str(retest),
        ]
    )
    assert code == 0

    report = json.loads(out.read_text(encoding="utf-8"))
    # The 10 retest personas are the first 10 of the 25, so the run has 25 people
    # and 58 items (48 trait items plus the 10 re-asked ones).
    assert report["n_personas"] == 25
    assert report["n_items"] == 58
    assert report["obedience"]["median"] > 0.85
    assert report["retest"]["pooled_agreement"] == pytest.approx(0.80)
    assert report["verdict"]["all_pass"] is True
    assert report["diversity"]["eigenvalues"]

    printed = capsys.readouterr().out
    assert "DIVERSITY" in printed
    assert "TRAIT OBEDIENCE" in printed
    assert "TEST-RETEST" in printed
    assert "gate 1: PASS" in printed


def test_the_cli_leaves_the_environment_as_it_found_it(tmp_path: Path) -> None:
    import os

    before = dict(os.environ)
    answers, retest, public_bank, truth = write_run(tmp_path)
    population_qa.main(
        [
            "--answers", str(answers),
            "--truth", str(truth),
            "--public-bank", str(public_bank),
            "--out", str(tmp_path / "results.json"),
            "--retest-answers", str(retest),
        ]
    )
    assert dict(os.environ) == before


def test_a_persona_with_no_planted_theta_is_skipped_not_crashed_on(tmp_path: Path) -> None:
    _, _, _, truth = write_run(tmp_path)
    found = population_qa.load_thetas(truth, ["p0001", "p9999"])
    assert "p0001" in found
    assert "p9999" not in found
    assert set(found["p0001"]) == set(DIMENSIONS)
