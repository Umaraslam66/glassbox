"""Tests for the Stage 5 episode simulator and the strategies it runs.

Two things are being defended here.

*Reproducibility*: an episode is identified by its round tag, and the same tag
has to give the same sitting -- same answers, same random ordering -- or a
replicate is not a replicate.

*Information isolation*: a strategy may see its own state, the item it picked,
and the answer that came back. Not the persona's answer distribution, not an
answer it did not ask for, not planted truth. Those tests are written against
the public surface of the objects rather than against a convention, because a
convention is not a wall.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.interview import episodes as ep
from src.interview import strategies as st
from src.model import mirt
from src.model.sequential_posterior import SequentialPosterior

REPO_ROOT = Path(__file__).resolve().parents[1]


def toy_params(seed: int = 0, n_items: int = 6, dims: int = 3) -> mirt.ItemParams:
    rng = np.random.default_rng(seed)
    kinds = [mirt.LIKERT if j % 2 == 0 else mirt.BINARY for j in range(n_items)]
    cut_raw = np.zeros((n_items, mirt.LIKERT_CUTS))
    cut_raw[:, 0] = rng.normal(0.0, 0.6, size=n_items)
    cut_raw[:, 1:] = rng.normal(0.0, 0.3, size=(n_items, mirt.LIKERT_CUTS - 1))
    return mirt.ItemParams(
        loadings=rng.normal(0.0, 0.9, size=(n_items, dims)),
        cut_raw=cut_raw,
        kinds=kinds,
    )


def toy_codes(params: mirt.ItemParams, n_personas: int = 4, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    codes = np.zeros((n_personas, len(params.kinds)), dtype=np.int16)
    for j, kind in enumerate(params.kinds):
        top = mirt.LIKERT_CATEGORIES if kind == mirt.LIKERT else 2
        codes[:, j] = rng.integers(0, top, size=n_personas)
    return codes


# --------------------------------------------------------------------------
# round tags and seeds
# --------------------------------------------------------------------------


def test_round_tags_are_unique_per_sitting():
    tags = {
        ep.round_tag(name, rep)
        for name in ("random", "heuristic", "fixed")
        for rep in range(51)
    }
    assert len(tags) == 3 * 51
    assert ep.round_tag("random", 7) == "ep_random_007"


def test_round_tags_never_collide_with_the_earlier_rounds():
    """The sweep, the retest and the Stage 3 interview must keep their own noise."""
    from src.interview.responder_prompts import ROUNDS

    for name in ("random", "heuristic", "fixed"):
        for rep in range(51):
            assert ep.round_tag(name, rep) not in ROUNDS


def test_episode_seeds_are_deterministic_and_distinct():
    first = ep.episode_seed("random", 3)
    assert first == ep.episode_seed("random", 3)
    assert first != ep.episode_seed("random", 4)
    assert first != ep.episode_seed("heuristic", 3)


def test_the_same_seed_gives_the_same_random_order():
    left = st.RandomOrder(5, 12, ep.episode_seed("random", 2))
    right = st.RandomOrder(5, 12, ep.episode_seed("random", 2))
    other = st.RandomOrder(5, 12, ep.episode_seed("random", 3))
    assert np.array_equal(left._order, right._order)
    assert not np.array_equal(left._order, other._order)


# --------------------------------------------------------------------------
# the answer bank hands out one cell at a time
# --------------------------------------------------------------------------


def test_the_bank_reveals_only_the_cells_it_is_asked_for():
    codes = np.arange(12).reshape(3, 4) % 5
    bank = ep.AnswerBank(codes, "ep_test_000")
    revealed = bank.reveal(np.array([0, 2, 3]))
    assert revealed.tolist() == [codes[0, 0], codes[1, 2], codes[2, 3]]


def test_the_bank_has_no_public_route_to_the_whole_matrix():
    """The isolation test. A strategy holds no reference to the matrix.

    ``full_matrix_for_scoring`` exists for the full-bank reference arm in the
    grader and is deliberately named so that using it by accident is hard; it is
    excluded here and its single caller is checked below.
    """
    bank = ep.AnswerBank(np.zeros((2, 3), dtype=np.int16), "ep_test_000")
    public = [
        name
        for name in dir(bank)
        if not name.startswith("_") and name != "full_matrix_for_scoring"
    ]
    assert sorted(public) == ["n_items", "n_personas", "reveal", "round_name"]
    for name in public:
        value = getattr(bank, name)
        assert not isinstance(value, np.ndarray), f"{name} exposes an array"
    assert not hasattr(bank, "__dict__")  # __slots__: nothing can be attached later


def test_only_the_grader_reads_the_full_matrix():
    """``full_matrix_for_scoring`` is called from src/eval and nowhere else."""
    callers = []
    for path in (REPO_ROOT / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "full_matrix_for_scoring" in text and path.name != "episodes.py":
            callers.append(path.relative_to(REPO_ROOT).parts)
    assert callers, "the reference arm should call it from somewhere"
    for parts in callers:
        assert parts[:2] == ("src", "eval"), f"{parts} reads the whole answer matrix"


def test_the_bank_refuses_a_cell_with_no_recorded_answer():
    codes = np.array([[0, -1], [1, 1]], dtype=np.int16)
    bank = ep.AnswerBank(codes, "ep_test_000")
    with pytest.raises(ValueError, match="no recorded answer"):
        bank.reveal(np.array([1, 0]))


# --------------------------------------------------------------------------
# what a strategy is allowed to see
# --------------------------------------------------------------------------


def test_a_strategy_is_handed_the_state_and_nothing_else():
    """Record every argument the strategy is called with, and inspect it."""
    params = toy_params(n_items=5)
    codes = toy_codes(params, n_personas=3)
    seen: list[object] = []

    class Spy(st._Base):
        name = "spy"

        def select(self, state):
            seen.append(state)
            free = np.argmax(~np.asarray(state.answered), axis=1)
            return self._check(free, state)

        def observe(self, columns, answers):
            seen.append(("observe", np.asarray(columns), np.asarray(answers)))

    posterior = SequentialPosterior(params, 3)
    simulator = ep.EpisodeSimulator(ep.AnswerBank(codes, "ep_spy_000"), posterior, Spy(3, 5))
    simulator.run(3)

    states = [s for s in seen if not isinstance(s, tuple)]
    assert states
    for state in states:
        fields = set(vars(state))
        assert fields == {"theta", "cov", "blur", "answered", "n_asked", "step"}
    observations = [s for s in seen if isinstance(s, tuple)]
    assert len(observations) == 3
    for _, columns, answers in observations:
        assert columns.shape == (3,)
        assert answers.shape == (3,)


def test_the_simulator_exposes_no_answers_publicly():
    params = toy_params(n_items=5)
    codes = toy_codes(params, n_personas=3)
    posterior = SequentialPosterior(params, 3)
    strategy = st.RandomOrder(3, 5, 0)
    simulator = ep.EpisodeSimulator(ep.AnswerBank(codes, "ep_x_000"), posterior, strategy)
    public = {name for name in vars(simulator) if not name.startswith("_")}
    assert public == {"posterior", "strategy"}


def test_the_strategy_module_never_touches_answers_or_distributions():
    """Static check on the parsed code -- prose may discuss the Wall, code may not cross it."""
    import ast

    tree = ast.parse(
        (REPO_ROOT / "src" / "interview" / "strategies.py").read_text(encoding="utf-8")
    )
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
            imported.extend(f"{node.module or ''}.{alias.name}" for alias in node.names)
    for dotted in imported:
        assert "noise_layer" not in dotted, f"strategies.py imports {dotted}"
        assert "episodes" not in dotted, f"strategies.py imports {dotted}"
        assert "truth" not in dotted, f"strategies.py imports {dotted}"

    forbidden = {"reveal", "full_matrix_for_scoring", "recorded_distributions"}
    called = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    } | {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    }
    assert not (called & forbidden), f"strategies.py reaches for {called & forbidden}"


# --------------------------------------------------------------------------
# the strategies themselves
# --------------------------------------------------------------------------


def test_random_order_asks_every_item_once():
    params = toy_params(n_items=6)
    codes = toy_codes(params, n_personas=4)
    posterior = SequentialPosterior(params, 4)
    strategy = st.RandomOrder(4, 6, 99)
    result = ep.EpisodeSimulator(ep.AnswerBank(codes, "r"), posterior, strategy).run(6)
    assert result.n_steps == 6
    for row in range(4):
        assert sorted(result.chosen[:, row].tolist()) == list(range(6))
    assert posterior.n_asked.tolist() == [6, 6, 6, 6]


def test_fixed_order_follows_the_script_and_stops():
    params = toy_params(n_items=6)
    codes = toy_codes(params, n_personas=3)
    posterior = SequentialPosterior(params, 3)
    strategy = st.FixedOrder(3, 6, [4, 1, 0])
    result = ep.EpisodeSimulator(ep.AnswerBank(codes, "f"), posterior, strategy).run(50)
    assert result.n_steps == 3
    assert result.capped_at == 3
    assert result.chosen[:, 0].tolist() == [4, 1, 0]
    assert result.chosen[:, 2].tolist() == [4, 1, 0]


def test_fixed_order_rejects_a_repeated_script():
    with pytest.raises(ValueError, match="repeats"):
        st.FixedOrder(2, 5, [1, 1])


def test_the_heuristic_never_repeats_an_item():
    params = toy_params(seed=2, n_items=7)
    codes = toy_codes(params, n_personas=5, seed=3)
    posterior = SequentialPosterior(params, 5)
    strategy = st.InfoGain(5, params)
    result = ep.EpisodeSimulator(ep.AnswerBank(codes, "h"), posterior, strategy).run(7)
    for row in range(5):
        assert sorted(result.chosen[:, row].tolist()) == list(range(7))


def test_the_heuristic_is_deterministic():
    params = toy_params(seed=4, n_items=6)
    codes = toy_codes(params, n_personas=4, seed=5)
    runs = []
    for _ in range(2):
        posterior = SequentialPosterior(params, 4)
        runs.append(
            ep.EpisodeSimulator(
                ep.AnswerBank(codes, "h"), posterior, st.InfoGain(4, params)
            ).run(5).chosen
        )
    assert np.array_equal(runs[0], runs[1])


def test_the_heuristic_beats_a_random_order_on_its_own_objective():
    """It should shrink the posterior faster than an arbitrary order does."""
    params = toy_params(seed=6, n_items=10, dims=3)
    codes = toy_codes(params, n_personas=8, seed=7)
    finals = {}
    for name, strategy in (
        ("heuristic", st.InfoGain(8, params)),
        ("random", st.RandomOrder(8, 10, 12345)),
    ):
        posterior = SequentialPosterior(params, 8)
        ep.EpisodeSimulator(ep.AnswerBank(codes, name), posterior, strategy).run(4)
        finals[name] = float(np.mean(np.trace(posterior.cov, axis1=1, axis2=2)))
    assert finals["heuristic"] < finals["random"]


def test_a_strategy_that_repeats_an_item_is_refused():
    """Two guards, both live: the strategy's own, and the posterior's behind it."""
    params = toy_params(n_items=4)
    codes = toy_codes(params, n_personas=2)

    class Stubborn(st._Base):
        name = "stubborn"

        def select(self, state):
            return self._check(np.zeros(self.n_personas, dtype=np.int64), state)

    class Sneaky(st._Base):
        name = "sneaky"

        def select(self, state):
            return np.zeros(self.n_personas, dtype=np.int64)

    for strategy, message in ((Stubborn(2, 4), "already asked"), (Sneaky(2, 4), "twice")):
        posterior = SequentialPosterior(params, 2)
        simulator = ep.EpisodeSimulator(ep.AnswerBank(codes, "s"), posterior, strategy)
        with pytest.raises(ValueError, match=message):
            simulator.run(2)


def test_the_policy_seam_runs_like_any_other_strategy():
    params = toy_params(seed=8, n_items=5)
    codes = toy_codes(params, n_personas=3, seed=9)
    seen: list[int] = []

    def policy(state):
        seen.append(state.step)
        return np.argmax(~np.asarray(state.answered), axis=1)

    told: list[tuple] = []
    strategy = st.PolicyStrategy(
        3, 5, policy, name="stub_policy", on_observe=lambda c, a: told.append((c, a))
    )
    posterior = SequentialPosterior(params, 3)
    result = ep.EpisodeSimulator(ep.AnswerBank(codes, "p"), posterior, strategy).run(3)
    assert result.strategy == "stub_policy"
    assert seen == [0, 1, 2]
    assert len(told) == 3


def test_build_makes_the_three_non_rl_strategies():
    params = toy_params(n_items=5)
    assert st.build("random", n_personas=2, params=params, seed=1).name == "random"
    assert st.build("heuristic", n_personas=2, params=params).name == "heuristic"
    fixed = st.build("fixed", n_personas=2, params=params, scripted_columns=[0, 1])
    assert fixed.max_steps() == 2
    with pytest.raises(ValueError, match="unknown strategy"):
        st.build("magic", n_personas=2, params=params)


# --------------------------------------------------------------------------
# the simulator's own contract
# --------------------------------------------------------------------------


def test_the_simulator_records_every_step_once():
    params = toy_params(n_items=6)
    codes = toy_codes(params, n_personas=3)
    posterior = SequentialPosterior(params, 3)
    steps: list[int] = []
    ep.EpisodeSimulator(
        ep.AnswerBank(codes, "c"), posterior, st.RandomOrder(3, 6, 1)
    ).run(4, on_step=lambda step, state: steps.append(step))
    assert steps == [1, 2, 3, 4]


def test_the_simulator_checks_the_bank_and_the_posterior_agree():
    params = toy_params(n_items=6)
    posterior = SequentialPosterior(params, 3)
    with pytest.raises(ValueError, match="personas"):
        ep.EpisodeSimulator(
            ep.AnswerBank(np.zeros((2, 6), dtype=np.int16)), posterior,
            st.RandomOrder(3, 6, 1),
        )
    with pytest.raises(ValueError, match="items"):
        ep.EpisodeSimulator(
            ep.AnswerBank(np.zeros((3, 5), dtype=np.int16)), posterior,
            st.RandomOrder(3, 6, 1),
        )


def test_the_budget_never_exceeds_the_bank():
    params = toy_params(n_items=4)
    codes = toy_codes(params, n_personas=2)
    posterior = SequentialPosterior(params, 2)
    result = ep.EpisodeSimulator(
        ep.AnswerBank(codes, "b"), posterior, st.RandomOrder(2, 4, 1)
    ).run(99)
    assert result.n_steps == 4


def test_the_episode_module_states_the_frozen_noise_constants():
    """The consistency rule has no flags: the constants come from the package."""
    assert ep.FROZEN_CONFIG.a == 1.2
    assert ep.FROZEN_CONFIG.b == 4.0
    assert ep.FROZEN_CONFIG.t_noise == 16.0
    text = (REPO_ROOT / "src" / "interview" / "episodes.py").read_text(encoding="utf-8")
    assert "--a " not in text and "add_argument(\"--noise-a" not in text
