"""Tests for the mechanical noise layer.

The layer decides two things per answer -- whether it wobbles, and what it
wobbles to -- so the tests are built around those two decisions: the
distribution it reads out of the responder's logprobs, the event probability it
computes from that, the re-draw, and the reproducibility that makes the whole
thing a seeded transform rather than a random one.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.personas import noise_layer as nl


# --------------------------------------------------------------------------
# reading the answer distribution
# --------------------------------------------------------------------------


def test_answer_of_token_reads_likert_digits():
    assert nl.answer_of_token("3", "likert5") == 3
    assert nl.answer_of_token(" 5", "likert5") == 5
    assert nl.answer_of_token("1", "likert5") == 1


@pytest.mark.parametrize("token", ["6", "0", "x", "12", "", "3.5"])
def test_answer_of_token_rejects_non_answers(token):
    assert nl.answer_of_token(token, "likert5") is None


@pytest.mark.parametrize("token", ["₃", "٣", "②", "³", "３"])
def test_answer_of_token_rejects_lookalike_digits(token):
    """Regression: the responder's real top-20 contains the subscript three.

    Unicode calls it a digit, so ``str.isdigit()`` waves it through, and then
    ``int()`` raises on it. Only the plain ASCII digits are answers.
    """
    assert nl.answer_of_token(token, "likert5") is None


@pytest.mark.parametrize(
    "token,expected",
    [("yes", "yes"), (" Yes", "yes"), ("YES", "yes"), ("no", "no"), (" No", "no"), ("Nope", "no")],
)
def test_answer_of_token_reads_binary_spellings(token, expected):
    assert nl.answer_of_token(token, "binary") == expected


def test_answer_of_token_rejects_a_digit_on_a_binary_item():
    assert nl.answer_of_token("3", "binary") is None


def test_answer_of_token_rejects_an_unknown_type():
    with pytest.raises(ValueError):
        nl.answer_of_token("yes", "ranking")


def test_distribution_is_normalised_over_the_valid_answers_only():
    logprobs = {"1": -0.1, "2": -3.0, "junk": -0.05, "\n": -0.2}
    dist = nl.answer_distribution(logprobs, "likert5")
    assert set(dist) == {1, 2, 3, 4, 5}
    assert math.isclose(sum(dist.values()), 1.0, rel_tol=1e-12)
    # "junk" carried the most mass of all but is not an answer, so it is gone.
    assert dist[1] == max(dist.values())


def test_distribution_floors_answers_that_missed_the_top_n():
    logprobs = {"1": -0.1, "2": -1.0}
    dist = nl.answer_distribution(logprobs, "likert5")
    floored = {dist[a] for a in (3, 4, 5)}
    assert len(floored) == 1, "every unseen answer should sit at the same floor"
    assert 0 < floored.pop() < dist[2], "the floor is unlikely but never impossible"


def test_floor_is_two_nats_below_the_weakest_recorded_token():
    logprobs = {"1": -0.5, "2": -4.0}
    dist = nl.answer_distribution(logprobs, "likert5")
    # ratio of an unseen answer to the weakest seen one is exp(-FLOOR_MARGIN)
    assert math.isclose(dist[3] / dist[2], math.exp(-nl.FLOOR_MARGIN), rel_tol=1e-9)


def test_binary_variants_add_up_as_probabilities():
    #  yes split over two spellings, each half of a single-token yes
    both = nl.answer_distribution({"yes": math.log(0.3), "Yes": math.log(0.3),
                                   "no": math.log(0.4)}, "binary")
    single = nl.answer_distribution({"yes": math.log(0.6), "no": math.log(0.4)}, "binary")
    assert math.isclose(both["yes"], single["yes"], rel_tol=1e-9)
    assert math.isclose(both["no"], single["no"], rel_tol=1e-9)


def test_distribution_survives_a_real_top_twenty():
    """A verbatim top-20 from the pilot run, lookalike digits and all."""
    logprobs = {"2": -0.000431, "1": -7.750431, "3": -15.500431, "//": -17.312931,
                " ": -17.625431, "_": -18.937931, "4": -19.437931, "{": -19.437931,
                "\n": -19.594181, "(": -19.750431, " uma": -19.875431, "[": -21.000431,
                " ₨": -21.062931, "”（": -21.562931, " Dietary": -21.719181,
                "\n\n": -21.875431, "#": -22.062931, "C": -22.094181,
                " tumorigen": -22.125431, "₃": -22.125431}
    dist = nl.answer_distribution(logprobs, "likert5")
    assert set(dist) == {1, 2, 3, 4, 5}
    assert math.isclose(sum(dist.values()), 1.0, rel_tol=1e-12)
    assert dist[2] > 0.99, "the sampled answer should carry nearly all the mass"
    assert dist[5] > 0, "an answer outside the top-20 still gets the floor"


def test_distribution_is_none_without_logprobs():
    assert nl.answer_distribution(None, "likert5") is None
    assert nl.answer_distribution({}, "likert5") is None


def test_conviction_is_the_largest_probability():
    assert nl.conviction({1: 0.7, 2: 0.2, 3: 0.1}) == 0.7


# --------------------------------------------------------------------------
# the event probability
# --------------------------------------------------------------------------


def test_event_probability_follows_the_formula():
    config = nl.NoiseConfig(a=0.1, b=0.8, t_noise=2.0)
    assert math.isclose(nl.event_probability(0.5, 0.75, config), 0.5 * (0.1 + 0.8 * 0.25))


def test_a_certain_answer_still_slips_at_the_base_rate():
    config = nl.NoiseConfig(a=0.2, b=0.9, t_noise=2.0)
    assert math.isclose(nl.event_probability(0.4, 1.0, config), 0.4 * 0.2)


def test_ambivalence_raises_the_event_probability():
    config = nl.NoiseConfig(a=0.1, b=0.8, t_noise=2.0)
    sure = nl.event_probability(0.3, 0.95, config)
    torn = nl.event_probability(0.3, 0.35, config)
    assert torn > sure


def test_bigger_wobble_means_more_events():
    config = nl.NoiseConfig(a=0.1, b=0.8, t_noise=2.0)
    assert nl.event_probability(0.45, 0.6, config) > nl.event_probability(0.10, 0.6, config)


def test_event_probability_is_capped():
    config = nl.NoiseConfig(a=5.0, b=5.0, t_noise=2.0)
    assert nl.event_probability(1.0, 0.0, config) == nl.MAX_EVENT_PROBABILITY


def test_config_rejects_nonsense():
    with pytest.raises(ValueError):
        nl.NoiseConfig(a=-0.1)
    with pytest.raises(ValueError):
        nl.NoiseConfig(t_noise=0.0)


# --------------------------------------------------------------------------
# the re-draw
# --------------------------------------------------------------------------


def test_flattening_at_one_is_the_identity():
    dist = {1: 0.7, 2: 0.2, 3: 0.1}
    flat = nl.flattened(dist, 1.0)
    for answer, value in dist.items():
        assert math.isclose(flat[answer], value, rel_tol=1e-12)


def test_flattening_moves_towards_uniform():
    dist = {1: 0.9, 2: 0.07, 3: 0.03}
    mild = nl.flattened(dist, 2.0)
    strong = nl.flattened(dist, 8.0)
    assert dist[1] > mild[1] > strong[1]
    assert strong[3] > mild[3] > dist[3]
    assert math.isclose(sum(strong.values()), 1.0, rel_tol=1e-12)


def test_flattening_keeps_the_ordering():
    flat = nl.flattened({1: 0.5, 2: 0.3, 3: 0.2}, 3.0)
    assert flat[1] > flat[2] > flat[3]


def test_draw_respects_the_probabilities():
    rng = np.random.default_rng(0)
    counts = {1: 0, 2: 0}
    for _ in range(4000):
        counts[nl.draw({1: 0.25, 2: 0.75}, rng)] += 1
    assert 0.70 < counts[2] / 4000 < 0.80


def test_draw_can_return_the_original_answer():
    rng = np.random.default_rng(1)
    seen = {nl.draw({1: 0.5, 2: 0.5}, rng) for _ in range(50)}
    assert seen == {1, 2}, "a re-draw must be able to land back on either answer"


# --------------------------------------------------------------------------
# seeding
# --------------------------------------------------------------------------


def test_cell_seed_is_stable():
    assert nl.cell_seed(548, "p0001", "q010", "main") == nl.cell_seed(548, "p0001", "q010", "main")


def test_cell_seed_separates_the_rounds():
    assert nl.cell_seed(548, "p0001", "q010", "main") != nl.cell_seed(548, "p0001", "q010", "retest")


def test_cell_seed_separates_cells_and_seed_bases():
    assert nl.cell_seed(548, "p0001", "q010", "main") != nl.cell_seed(548, "p0002", "q010", "main")
    assert nl.cell_seed(548, "p0001", "q010", "main") != nl.cell_seed(548, "p0001", "q011", "main")
    assert nl.cell_seed(548, "p0001", "q010", "main") != nl.cell_seed(1, "p0001", "q010", "main")


# --------------------------------------------------------------------------
# applying the layer
# --------------------------------------------------------------------------


def _cell(pid, item_id, round_name, answer, logprobs=None):
    row = {"pid": pid, "item_id": item_id, "round": round_name, "answer": answer, "raw": str(answer)}
    if logprobs is not None:
        row["logprobs"] = logprobs
    return row


CONFIDENT = {"4": -0.02, "3": -4.5, "5": -5.0}
TORN = {"2": math.log(0.34), "3": math.log(0.33), "4": math.log(0.33)}


def test_layer_keeps_the_schema_and_drops_inline_logprobs():
    rows = [_cell("p1", "q1", "main", 4, CONFIDENT)]
    out, _ = nl.apply_noise(rows, {}, {"p1": 0.3}, {"q1": "likert5"},
                            nl.NoiseConfig(), seed_base=548)
    assert set(out[0]) == {"pid", "item_id", "round", "answer"}  # "raw" is truth-side, dropped
    assert out[0]["answer"] in nl.LIKERT_ANSWERS


def test_layer_is_reproducible():
    rows = [_cell("p1", f"q{i}", "main", 4, TORN) for i in range(40)]
    config = nl.NoiseConfig(a=0.3, b=0.9, t_noise=3.0)
    first, _ = nl.apply_noise(rows, {}, {"p1": 0.4}, {f"q{i}": "likert5" for i in range(40)},
                              config, seed_base=548)
    second, _ = nl.apply_noise(rows, {}, {"p1": 0.4}, {f"q{i}": "likert5" for i in range(40)},
                               config, seed_base=548)
    assert [r["answer"] for r in first] == [r["answer"] for r in second]


def test_a_different_seed_base_gives_a_different_result():
    rows = [_cell("p1", f"q{i}", "main", 4, TORN) for i in range(60)]
    types = {f"q{i}": "likert5" for i in range(60)}
    config = nl.NoiseConfig(a=0.5, b=0.9, t_noise=4.0)
    a, _ = nl.apply_noise(rows, {}, {"p1": 0.45}, types, config, seed_base=1)
    b, _ = nl.apply_noise(rows, {}, {"p1": 0.45}, types, config, seed_base=2)
    assert [r["answer"] for r in a] != [r["answer"] for r in b]


def test_rounds_wobble_independently():
    """The same cell in the two rounds must not share its coin flip."""
    types = {f"q{i}": "likert5" for i in range(120)}
    rows = ([_cell("p1", f"q{i}", "main", 4, TORN) for i in range(120)]
            + [_cell("p1", f"q{i}", "retest", 4, TORN) for i in range(120)])
    config = nl.NoiseConfig(a=0.4, b=0.9, t_noise=4.0)
    out, _ = nl.apply_noise(rows, {}, {"p1": 0.45}, types, config, seed_base=548)
    main = [r["answer"] for r in out if r["round"] == "main"]
    retest = [r["answer"] for r in out if r["round"] == "retest"]
    disagreements = sum(1 for x, y in zip(main, retest) if x != y)
    assert disagreements > 10, "the two rounds are moving together -- seeds are coupled"


def test_zero_wobble_changes_nothing():
    rows = [_cell("p1", f"q{i}", "main", 4, TORN) for i in range(100)]
    out, stats = nl.apply_noise(rows, {}, {"p1": 0.0}, {f"q{i}": "likert5" for i in range(100)},
                                nl.NoiseConfig(a=0.4, b=1.0, t_noise=4.0), seed_base=548)
    assert all(r["answer"] == 4 for r in out)
    assert stats["n_events"] == 0


def test_more_wobble_produces_more_events():
    types = {f"q{i}": "likert5" for i in range(300)}
    config = nl.NoiseConfig(a=0.2, b=0.8, t_noise=3.0)
    steady = [_cell("p1", f"q{i}", "main", 4, TORN) for i in range(300)]
    jumpy = [_cell("p2", f"q{i}", "main", 4, TORN) for i in range(300)]
    _, low = nl.apply_noise(steady, {}, {"p1": 0.10}, types, config, seed_base=548)
    _, high = nl.apply_noise(jumpy, {}, {"p2": 0.45}, types, config, seed_base=548)
    assert high["event_rate"] > low["event_rate"] * 2


def test_ambivalent_items_wobble_more_than_convicted_ones():
    types = {f"q{i}": "likert5" for i in range(400)}
    config = nl.NoiseConfig(a=0.1, b=0.9, t_noise=3.0)
    sure = [_cell("p1", f"q{i}", "main", 4, CONFIDENT) for i in range(400)]
    torn = [_cell("p1", f"q{i}", "main", 3, TORN) for i in range(400)]
    _, sure_stats = nl.apply_noise(sure, {}, {"p1": 0.4}, types, config, seed_base=7)
    _, torn_stats = nl.apply_noise(torn, {}, {"p1": 0.4}, types, config, seed_base=7)
    assert torn_stats["event_rate"] > sure_stats["event_rate"]


def test_logprobs_can_arrive_from_a_separate_file():
    rows = [_cell("p1", "q1", "main", 4)]
    index = {("p1", "q1", "main"): CONFIDENT}
    _, stats = nl.apply_noise(rows, index, {"p1": 0.3}, {"q1": "likert5"},
                              nl.NoiseConfig(), seed_base=548)
    assert stats["n_noise_eligible"] == 1
    assert stats["skipped"]["no_logprobs"] == 0


def test_cells_without_the_pieces_pass_through_untouched():
    rows = [
        _cell("p1", "q1", "main", 4),                     # no logprobs anywhere
        _cell("p1", "q2", "main", 4, CONFIDENT),          # item not in the bank
        _cell("ghost", "q1", "main", 4, CONFIDENT),       # no wobble on file
    ]
    out, stats = nl.apply_noise(rows, {}, {"p1": 0.3}, {"q1": "likert5"},
                                nl.NoiseConfig(), seed_base=548)
    assert [r["answer"] for r in out] == [4, 4, 4]
    assert stats["skipped"] == {"no_logprobs": 1, "no_wobble": 1,
                                "unknown_item": 1, "no_distribution": 0}
    assert stats["n_noise_eligible"] == 0


def test_binary_answers_stay_yes_or_no():
    logprobs = {"Yes": math.log(0.55), "No": math.log(0.45)}
    rows = [_cell("p1", f"q{i}", "main", "yes", logprobs) for i in range(60)]
    out, _ = nl.apply_noise(rows, {}, {"p1": 0.45},
                            {f"q{i}": "binary" for i in range(60)},
                            nl.NoiseConfig(a=0.4, b=1.0, t_noise=4.0), seed_base=548)
    assert {r["answer"] for r in out} <= {"yes", "no"}
    assert len({r["answer"] for r in out}) == 2, "a coin-flip binary item should move"


def test_stats_break_down_by_persona_type_and_class():
    types = {"q1": "likert5", "q2": "binary"}
    classes = {"q1": "primary", "q2": "distractor"}
    rows = [_cell("p1", "q1", "main", 4, TORN),
            _cell("p2", "q2", "main", "yes", {"yes": math.log(0.5), "no": math.log(0.5)})]
    _, stats = nl.apply_noise(rows, {}, {"p1": 0.3, "p2": 0.4}, types,
                              nl.NoiseConfig(), seed_base=548, classes=classes)
    assert stats["by_persona"]["p1"]["wobble"] == 0.3
    assert stats["by_persona"]["p1"]["cells"] == 1
    assert set(stats["by_item_type"]) == {"likert5", "binary"}
    assert set(stats["by_design_class"]) == {"primary", "distractor"}
    assert stats["conviction_percentiles"]["0"] <= stats["conviction_percentiles"]["100"]


def test_stats_record_the_config_and_seed():
    rows = [_cell("p1", "q1", "main", 4, CONFIDENT)]
    config = nl.NoiseConfig(a=0.11, b=0.22, t_noise=3.3)
    _, stats = nl.apply_noise(rows, {}, {"p1": 0.3}, {"q1": "likert5"}, config, seed_base=99)
    assert stats["config"] == {"a": 0.11, "b": 0.22, "t_noise": 3.3}
    assert stats["seed_base"] == 99


# --------------------------------------------------------------------------
# helpers and the cli
# --------------------------------------------------------------------------


def test_load_wobbles_reads_a_directory(tmp_path):
    (tmp_path / "p0001.json").write_text(json.dumps({"pid": "p0001", "wobble": 0.17}))
    (tmp_path / "p0002.json").write_text(json.dumps({"pid": "p0002", "wobble": 0.4}))
    assert nl.load_wobbles(tmp_path) == {"p0001": 0.17, "p0002": 0.4}


def test_design_classes_sorts_the_four_kinds():
    design = {"items": [
        {"item_id": "q1", "loadings": {"TRU": 1.0}, "strength_class": "strong"},
        {"item_id": "q2", "loadings": {"TRU": 0.7, "RSK": 0.5}, "strength_class": "strong"},
        {"item_id": "q3", "loadings": {"TRU": 0.3}, "strength_class": "weak"},
        {"item_id": "q4", "loadings": {}, "strength_class": "none"},
    ]}
    assert nl.design_classes(design) == {
        "q1": "primary", "q2": "cross", "q3": "weak", "q4": "distractor"}


def _write_batch(tmp_path):
    noise_dir = tmp_path / "wobbles"
    noise_dir.mkdir()
    (noise_dir / "p1.json").write_text(json.dumps({"wobble": 0.45}))
    bank = tmp_path / "bank_items.json"
    bank.write_text(json.dumps({"items": [
        {"item_id": f"q{i}", "text": "t", "type": "likert5"} for i in range(50)]}))
    answers = tmp_path / "answers.jsonl"
    answers.write_text("".join(
        json.dumps(_cell("p1", f"q{i}", "main", 3, TORN)) + "\n" for i in range(50)))
    return noise_dir, bank, answers


def test_cli_writes_answers_and_stats(tmp_path, capsys):
    noise_dir, bank, answers = _write_batch(tmp_path)
    out = tmp_path / "noised.jsonl"
    stats_path = tmp_path / "stats.json"
    rc = nl.main([
        "--answers", str(answers), "--noise-dir", str(noise_dir),
        "--public-bank", str(bank), "--out", str(out), "--stats", str(stats_path),
        "--a", "0.3", "--b", "0.9", "--t-noise", "3", "--seed-base", "548",
    ])
    assert rc == 0
    rows = nl.read_jsonl(out)
    assert len(rows) == 50
    assert all(set(r) == {"pid", "item_id", "round", "answer"} for r in rows)  # no "raw" on the system side
    stats = json.loads(stats_path.read_text())
    assert stats["config"] == {"a": 0.3, "b": 0.9, "t_noise": 3.0}
    assert stats["n_noise_eligible"] == 50
    assert stats["n_events"] > 0
    assert "noised 50 of 50" in capsys.readouterr().out


def test_cli_defaults_the_stats_path(tmp_path):
    noise_dir, bank, answers = _write_batch(tmp_path)
    out = tmp_path / "noised.jsonl"
    nl.main(["--answers", str(answers), "--noise-dir", str(noise_dir),
             "--public-bank", str(bank), "--out", str(out)])
    assert Path(str(out) + ".stats.json").is_file()


def test_cli_refuses_to_run_without_logprobs(tmp_path):
    noise_dir, bank, _ = _write_batch(tmp_path)
    bare = tmp_path / "bare.jsonl"
    bare.write_text(json.dumps(_cell("p1", "q0", "main", 3)) + "\n")
    with pytest.raises(SystemExit):
        nl.main(["--answers", str(bare), "--noise-dir", str(noise_dir),
                 "--public-bank", str(bank), "--out", str(tmp_path / "x.jsonl")])


def test_cli_refuses_an_empty_answers_file(tmp_path):
    noise_dir, bank, _ = _write_batch(tmp_path)
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    with pytest.raises(SystemExit):
        nl.main(["--answers", str(empty), "--noise-dir", str(noise_dir),
                 "--public-bank", str(bank), "--out", str(tmp_path / "x.jsonl")])


def test_noised_output_is_readable_by_the_population_qa_matrix_builder(tmp_path):
    """The whole point of the schema rule: the grader must not need changing."""
    from src.eval.population_qa import build_matrix

    noise_dir, bank, answers = _write_batch(tmp_path)
    out = tmp_path / "noised.jsonl"
    nl.main(["--answers", str(answers), "--noise-dir", str(noise_dir),
             "--public-bank", str(bank), "--out", str(out), "--seed-base", "548"])
    rows = nl.read_jsonl(out)
    types = {f"q{i}": "likert5" for i in range(50)}
    pids, item_ids, matrix = build_matrix(rows, types, "main")
    assert pids == ["p1"]
    assert len(item_ids) == 50
    assert not np.isnan(matrix).any()

# --------------------------------------------------------------------------
# the interview round (Stage 3 consistency rule)
# --------------------------------------------------------------------------
#
# These live here rather than in tests/test_interview.py because the thing
# under test is one seeded application of the noise layer, and the fixtures
# have to carry the responder's recorded answer distributions -- which is
# exactly what the Wall keeps out of the interview package and its tests.


def _distributions(pids, item_ids, round_name, logprobs):
    return {(pid, item_id, round_name): logprobs
            for pid in pids for item_id in item_ids}


def test_recorded_distributions_filters_by_item_and_round(tmp_path):
    path = tmp_path / "completions.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in [
        {"pid": "p1", "item_id": "q1", "round": "main", "logprobs": CONFIDENT},
        {"pid": "p1", "item_id": "q2", "round": "main", "logprobs": TORN},
        {"pid": "p1", "item_id": "q1", "round": "retest", "logprobs": TORN},
    ]))
    found = nl.recorded_distributions(path, item_ids=["q1"], rounds=["main"])
    assert set(found) == {("p1", "q1", "main")}
    assert found[("p1", "q1", "main")] == CONFIDENT

    assert len(nl.recorded_distributions(path)) == 3
    assert nl.recorded_distributions(tmp_path / "nope.jsonl") == {}


def test_recorded_distributions_skips_records_with_nothing_recorded(tmp_path):
    path = tmp_path / "completions.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in [
        {"pid": "p1", "item_id": "q1", "round": "main", "completion": "4"},
        {"pid": "p2", "item_id": "q1", "round": "main", "logprobs": CONFIDENT},
    ]))
    assert set(nl.recorded_distributions(path)) == {("p2", "q1", "main")}


def test_draw_round_relabels_the_round_and_drops_truth_side_fields():
    rows = [_cell("p1", "q1", "main", 4)]
    out, stats = nl.draw_round(
        rows, _distributions(["p1"], ["q1"], "main", CONFIDENT),
        {"p1": 0.3}, {"q1": "likert5"}, nl.NoiseConfig(), 548,
        round_name="interview1",
    )
    assert set(out[0]) == {"pid", "item_id", "round", "answer"}
    assert out[0]["round"] == "interview1"
    assert stats["round"] == "interview1"
    assert stats["source_round"] == "main"


def test_draw_round_is_reproducible():
    items = [f"q{i}" for i in range(60)]
    rows = [_cell("p1", item_id, "main", 4) for item_id in items]
    index = _distributions(["p1"], items, "main", TORN)
    types = {item_id: "likert5" for item_id in items}
    kwargs = dict(round_name="interview1")

    first, _ = nl.draw_round(rows, index, {"p1": 0.4}, types,
                             nl.NoiseConfig(a=1.2, b=4.0, t_noise=16.0), 548, **kwargs)
    second, _ = nl.draw_round(rows, index, {"p1": 0.4}, types,
                              nl.NoiseConfig(a=1.2, b=4.0, t_noise=16.0), 548, **kwargs)
    assert [r["answer"] for r in first] == [r["answer"] for r in second]


def test_the_interview_round_is_a_different_draw_from_main_and_retest():
    """A fresh sitting: same recorded material, a seed nobody else used."""
    items = [f"q{i}" for i in range(150)]
    rows = [_cell("p1", item_id, "main", 4) for item_id in items]
    index = _distributions(["p1"], items, "main", TORN)
    types = {item_id: "likert5" for item_id in items}
    config = nl.NoiseConfig(a=1.2, b=4.0, t_noise=16.0)

    def answers(round_name):
        out, _ = nl.draw_round(rows, index, {"p1": 0.45}, types, config, 548,
                               round_name=round_name)
        return [r["answer"] for r in out]

    interview = answers("interview1")
    main = answers("main")
    retest = answers("retest")

    assert sum(1 for a, b in zip(interview, main) if a != b) > 10, \
        "the interview is tracking the main round -- the seeds are coupled"
    assert sum(1 for a, b in zip(interview, retest) if a != b) > 10, \
        "the interview is tracking the retest -- the seeds are coupled"


def test_drawing_the_main_round_again_reproduces_the_main_answers():
    """The round tag is the only new thing: same tag, same answers."""
    items = [f"q{i}" for i in range(80)]
    rows = [_cell("p1", item_id, "main", 4, TORN) for item_id in items]
    types = {item_id: "likert5" for item_id in items}
    config = nl.NoiseConfig(a=1.2, b=4.0, t_noise=16.0)

    direct, _ = nl.apply_noise(rows, {}, {"p1": 0.45}, types, config, 548)
    redrawn, _ = nl.draw_round(rows, {}, {"p1": 0.45}, types, config, 548,
                               round_name="main")
    assert [r["answer"] for r in direct] == [r["answer"] for r in redrawn]


def test_draw_round_keeps_the_scripted_item_order():
    scripted = ["q9", "q2", "q5"]
    rows = [_cell("p2", "q5", "main", 4), _cell("p1", "q2", "main", 4),
            _cell("p1", "q9", "main", 4), _cell("p1", "q5", "main", 4),
            _cell("p1", "q7", "main", 4),  # not scripted, must be dropped
            _cell("p2", "q9", "main", 4), _cell("p2", "q2", "main", 4)]
    index = _distributions(["p1", "p2"], scripted, "main", CONFIDENT)
    types = {item_id: "likert5" for item_id in scripted + ["q7"]}
    out, stats = nl.draw_round(rows, index, {"p1": 0.2, "p2": 0.2}, types,
                               nl.NoiseConfig(), 548, round_name="interview1",
                               item_ids=scripted)
    assert [(r["pid"], r["item_id"]) for r in out] == [
        ("p1", "q9"), ("p1", "q2"), ("p1", "q5"),
        ("p2", "q9"), ("p2", "q2"), ("p2", "q5"),
    ]
    assert stats["n_items"] == 3


def test_draw_round_only_reads_the_source_round():
    rows = [_cell("p1", "q1", "main", 4), _cell("p1", "q1", "retest", 2)]
    out, _ = nl.draw_round(rows, _distributions(["p1"], ["q1"], "main", CONFIDENT),
                           {"p1": 0.0}, {"q1": "likert5"}, nl.NoiseConfig(), 548,
                           round_name="interview1", source_round="main")
    assert [(r["item_id"], r["answer"]) for r in out] == [("q1", 4)]


# --------------------------------------------------------------------------
# the interview-answer CLI
# --------------------------------------------------------------------------


def _interview_fixture(tmp_path, scripted=("q3", "q1", "q2"), n_personas=3):
    """Everything the interview CLI needs, written to disk."""
    pids = [f"p{i:04d}" for i in range(1, n_personas + 1)]
    items = ["q1", "q2", "q3", "q4"]

    bank = {"items": [{"item_id": i, "text": f"Statement {i}.", "type": "likert5"}
                      for i in items]}
    (tmp_path / "bank_items.json").write_text(json.dumps(bank))

    noise_dir = tmp_path / "noise"
    noise_dir.mkdir()
    for pid in pids:
        (noise_dir / f"{pid}.json").write_text(json.dumps({"pid": pid, "wobble": 0.45}))

    answers = tmp_path / "answers.jsonl"
    answers.write_text("".join(
        json.dumps({"pid": pid, "item_id": item_id, "round": "main",
                    "answer": 4, "raw": "4"}) + "\n"
        for pid in pids for item_id in items))

    completions = tmp_path / "completions.jsonl"
    completions.write_text("".join(
        json.dumps({"pid": pid, "item_id": item_id, "round": "main",
                    "completion": "4", "logprobs": TORN}) + "\n"
        for pid in pids for item_id in items))

    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"interview_closed": list(scripted),
                                  "interview_open": ["oe01"]}))
    return {"bank": tmp_path / "bank_items.json", "noise": noise_dir,
            "answers": answers, "completions": completions, "splits": splits,
            "pids": pids, "scripted": list(scripted)}


def _run_interview_cli(fixture, out, extra=()):
    from src.interview import interview_answers

    return interview_answers.main([
        "--answers", str(fixture["answers"]),
        "--recorded", str(fixture["completions"]),
        "--noise-dir", str(fixture["noise"]),
        "--public-bank", str(fixture["bank"]),
        "--splits", str(fixture["splits"]),
        "--out", str(out),
        *extra,
    ])


def test_interview_cli_writes_the_scripted_answers(tmp_path, capsys):
    fixture = _interview_fixture(tmp_path)
    out = tmp_path / "answers_interview1.jsonl"

    assert _run_interview_cli(fixture, out) == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]

    assert len(rows) == len(fixture["pids"]) * len(fixture["scripted"])
    assert {r["round"] for r in rows} == {"interview1"}
    assert all(set(r) == {"pid", "item_id", "round", "answer"} for r in rows)
    # scripted order, per persona, and nothing outside the scripted set
    assert [r["item_id"] for r in rows[:3]] == fixture["scripted"]
    assert capsys.readouterr().out.count("interview1") >= 1


def test_interview_cli_uses_the_frozen_settings(tmp_path):
    fixture = _interview_fixture(tmp_path)
    out = tmp_path / "answers_interview1.jsonl"
    _run_interview_cli(fixture, out)

    stats = json.loads((tmp_path / "answers_interview1.jsonl.stats.json").read_text())
    assert stats["config"] == {"a": 1.2, "b": 4.0, "t_noise": 16.0}
    assert stats["seed_base"] == 548
    assert stats["round"] == "interview1"
    assert stats["items"] == fixture["scripted"]


def test_interview_cli_has_no_way_to_change_the_frozen_settings(tmp_path):
    """The consistency rule, enforced by there being no flag for it."""
    fixture = _interview_fixture(tmp_path)
    out = tmp_path / "answers_interview1.jsonl"
    for flag in ("--a", "--b", "--t-noise", "--seed-base"):
        with pytest.raises(SystemExit):
            _run_interview_cli(fixture, out, extra=[flag, "1"])


def test_interview_cli_is_deterministic(tmp_path):
    fixture = _interview_fixture(tmp_path, n_personas=20)
    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    _run_interview_cli(fixture, first)
    _run_interview_cli(fixture, second)
    assert first.read_text() == second.read_text()


def test_interview_cli_refuses_a_splits_file_without_the_interview_set(tmp_path):
    from src.interview import interview_answers

    fixture = _interview_fixture(tmp_path)
    bad = tmp_path / "not_splits.json"
    bad.write_text(json.dumps({"persona_holdout": ["p0001"]}))
    with pytest.raises(ValueError, match="interview_closed"):
        interview_answers.interview_items(bad)


def test_interview_cli_refuses_an_item_that_is_not_in_the_bank(tmp_path):
    from src.interview import interview_answers

    fixture = _interview_fixture(tmp_path, scripted=("q1", "q99"))
    with pytest.raises(ValueError, match="not in the public bank"):
        interview_answers.draw(
            fixture["answers"], fixture["completions"], fixture["noise"],
            fixture["bank"], ["q1", "q99"], "interview1")
