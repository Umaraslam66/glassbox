"""Tests for the Stage 5 RL interviewer.

Three of these are about the research contract rather than the arithmetic:

* ``src/rl/`` cannot reach planted truth -- checked by parsing every file, not by
  reading the docstrings;
* the reward is a function of the posterior and nothing else;
* the exploratory oracle-reward weights cannot be loaded by the confirmatory
  grading path.

The rest are the ordinary correctness checks a hand-written policy gradient
needs: the mask, the gradient against finite differences, and determinism.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from src.eval import gate5
from src.model import mirt
from src.model.sequential_posterior import InterviewState, SequentialPosterior
from src.rl import policy as policy_module
from src.rl import reward as reward_module
from src.rl import trainer as trainer_module
from src.rl.policy import (
    BLUR_REWARD,
    CONFIRMATORY_ARM,
    ORACLE_ARM,
    ORACLE_REWARD,
    MLPPolicy,
    load_confirmatory,
    masked_log_softmax,
    sample_actions,
)
from src.rl.reward import RewardContext, blur_reward, posterior_trace
from src.rl.trainer import EpisodeCache, TrainConfig, iteration_rng, returns_to_go, train

REPO_ROOT = Path(__file__).resolve().parents[1]
RL_DIR = REPO_ROOT / "src" / "rl"
DIMS = 3
N_ITEMS = 7


# --------------------------------------------------------------------------
# fabricated world
# --------------------------------------------------------------------------


def fake_params(n_items: int = N_ITEMS, dims: int = DIMS, seed: int = 0) -> mirt.ItemParams:
    rng = np.random.default_rng(seed)
    loadings = rng.normal(0.0, 0.8, size=(n_items, dims))
    cut_raw = np.tile(np.array([-1.2, 0.2, 0.2, 0.2]), (n_items, 1))
    return mirt.ItemParams(loadings=loadings, cut_raw=cut_raw, kinds=[mirt.LIKERT] * n_items)


def fake_state(n_personas: int = 4, dims: int = DIMS, n_items: int = N_ITEMS, seed: int = 1):
    rng = np.random.default_rng(seed)
    answered = np.zeros((n_personas, n_items), dtype=bool)
    answered[0, 0] = True
    answered[1, [1, 2]] = True
    cov = np.tile(np.eye(dims), (n_personas, 1, 1))
    return InterviewState(
        theta=rng.normal(size=(n_personas, dims)),
        cov=cov,
        blur=np.sqrt(np.diagonal(cov, axis1=1, axis2=2)),
        answered=answered,
        n_asked=answered.sum(axis=1),
        step=2,
    )


def fake_cache(tmp_path: Path, *, n_rounds: int = 3, n_personas: int = 12,
               n_items: int = N_ITEMS, seed: int = 3) -> EpisodeCache:
    rng = np.random.default_rng(seed)
    path = tmp_path / "episodes.npz"
    np.savez_compressed(
        path,
        codes=rng.integers(0, 5, size=(n_rounds, n_personas, n_items)).astype(np.int8),
        rounds=np.array([f"ep_rl_{k:03d}" for k in range(n_rounds)]),
        persona_ids=np.array([f"x{k:04d}" for k in range(n_personas)]),
        item_ids=np.array([f"q{k:03d}" for k in range(n_items)]),
    )
    return EpisodeCache(path)


# --------------------------------------------------------------------------
# the Wall, restated for src/rl
# --------------------------------------------------------------------------


def rl_python_files() -> list[Path]:
    return sorted(RL_DIR.rglob("*.py"))


def test_there_are_rl_files_to_check():
    assert len(rl_python_files()) >= 4


def test_src_rl_never_imports_the_grader_or_anything_truth_side():
    """No route from the trainer to planted truth, checked by parsing.

    The repository-wide Wall test already forbids the truth-store accessor and
    the planted directory's path everywhere outside ``src/eval``. This is the
    stronger statement the RL arm needs: ``src/rl`` does not import ``src.eval``
    at all, so the reward, the gradient and every weight are downstream of
    system-side code only. The exploratory oracle arm inverts the dependency --
    ``src/eval`` imports ``src/rl`` and injects its reward -- which is why the
    check is directional.
    """
    # assembled from pieces so this file does not itself contain the literal the
    # repository-wide Wall test forbids outside src/eval
    accessor = "truth" + "_store"
    banned_modules = {"eval", accessor, "recovery"}
    banned_names = {"load_planted_theta", "load_planted_wobble", accessor}
    violations: list[str] = []

    for path in rl_python_files():
        relative = path.relative_to(REPO_ROOT)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if banned_modules & set(alias.name.split(".")):
                        violations.append(f"{relative}:{node.lineno} import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if banned_modules & set(module.split(".")):
                    violations.append(f"{relative}:{node.lineno} from {module}")
                for alias in node.names:
                    if alias.name in banned_names:
                        violations.append(
                            f"{relative}:{node.lineno} from {module} import {alias.name}"
                        )
    assert not violations, "src/rl reaches truth-side code:\n" + "\n".join(violations)


def test_the_reward_is_a_function_of_the_posterior_and_the_cost_alone():
    """Same covariances in, same reward out, whatever else is on the context."""
    rng = np.random.default_rng(0)
    before = np.tile(np.eye(DIMS) * 1.0, (5, 1, 1))
    after = np.tile(np.eye(DIMS) * 0.6, (5, 1, 1))
    base = RewardContext(
        step=3,
        columns=np.arange(5),
        persona_rows=np.arange(5),
        theta_before=rng.normal(size=(5, DIMS)),
        theta_after=rng.normal(size=(5, DIMS)),
        cov_before=before,
        cov_after=after,
        question_cost=0.02,
    )
    expected = DIMS * (1.0 - 0.6) - 0.02
    assert blur_reward(base) == pytest.approx(np.full(5, expected))

    # everything the reward is not allowed to depend on, changed at once
    other = RewardContext(
        step=99,
        columns=np.zeros(5, dtype=int),
        persona_rows=np.arange(5)[::-1],
        theta_before=rng.normal(size=(5, DIMS)),
        theta_after=rng.normal(size=(5, DIMS)),
        cov_before=before,
        cov_after=after,
        question_cost=0.02,
    )
    assert blur_reward(other) == pytest.approx(blur_reward(base))


def test_posterior_trace_is_the_total_declared_variance():
    cov = np.stack([np.diag([1.0, 2.0, 3.0]), np.diag([0.5, 0.5, 0.5])])
    assert posterior_trace(cov) == pytest.approx([6.0, 1.5])


# --------------------------------------------------------------------------
# masking
# --------------------------------------------------------------------------


def test_the_mask_zeroes_answered_items_and_renormalises_the_rest():
    state = fake_state()
    logits = np.random.default_rng(7).normal(size=state.answered.shape)
    probs = np.exp(masked_log_softmax(logits, state.answered))
    assert np.all(probs[state.answered] == 0.0)
    assert probs.sum(axis=1) == pytest.approx(np.ones(state.n_personas))
    # the unmasked entries keep the softmax's relative odds
    row = 0
    free = ~state.answered[row]
    ratio = probs[row][free] / probs[row][free].sum()
    reference = np.exp(logits[row][free] - logits[row][free].max())
    assert ratio == pytest.approx(reference / reference.sum())


def test_a_sampled_action_is_never_an_answered_item():
    state = fake_state(n_personas=64, seed=11)
    policy = MLPPolicy(DIMS, N_ITEMS, hidden=5, horizon=6, seed=2)
    rng = np.random.default_rng(0)
    for _ in range(25):
        columns = policy.act(state, rng)
        assert not np.any(state.answered[np.arange(state.n_personas), columns])


def test_the_greedy_action_is_never_an_answered_item_even_when_it_scores_best():
    """The mask has to beat the logits, not just tie-break them."""
    state = fake_state(n_personas=2, seed=13)
    policy = MLPPolicy(DIMS, N_ITEMS, hidden=5, horizon=6, seed=3)
    policy.b2 = np.zeros(N_ITEMS)
    policy.b2[0] = 50.0  # item 0 is overwhelmingly preferred, and row 0 has it
    policy.w2 = np.zeros_like(policy.w2)
    columns = policy.greedy(state)
    assert columns[0] != 0
    assert columns[1] == 0


def test_sample_actions_follows_the_probabilities():
    probs = np.array([[0.7, 0.3, 0.0], [0.0, 0.0, 1.0]])
    rng = np.random.default_rng(0)
    draws = np.stack([sample_actions(probs, rng) for _ in range(4000)])
    assert np.all(draws[:, 1] == 2)
    assert np.mean(draws[:, 0] == 0) == pytest.approx(0.7, abs=0.03)


def test_the_step_feature_is_clipped_so_a_long_interview_stays_in_distribution():
    policy = MLPPolicy(DIMS, N_ITEMS, hidden=4, horizon=10, seed=0)
    state = fake_state()
    features = policy_module.state_features(state, policy.horizon)
    assert features.shape == (state.n_personas, 2 * DIMS + N_ITEMS + 1)
    assert features[:, -1] == pytest.approx(np.full(state.n_personas, 0.2))
    long = InterviewState(
        theta=state.theta, cov=state.cov, blur=state.blur,
        answered=state.answered, n_asked=state.n_asked, step=999,
    )
    assert policy_module.state_features(long, policy.horizon)[:, -1] == pytest.approx(
        np.ones(state.n_personas)
    )


# --------------------------------------------------------------------------
# the gradient
# --------------------------------------------------------------------------


def objective(policy: MLPPolicy, state, actions, weights, entropy_bonus):
    """The scalar the trainer ascends: mean of ``w log pi(a) + beta H``."""
    x = policy.features(state)
    _, logits = policy.forward(x)
    log_p = masked_log_softmax(logits, state.answered)
    probs = np.exp(log_p)
    chosen = log_p[np.arange(len(actions)), actions]
    entropy = -np.sum(probs * np.where(probs > 0.0, log_p, 0.0), axis=1)
    return float(np.mean(weights * chosen + entropy_bonus * entropy))


def test_the_gradient_matches_finite_differences_on_a_tiny_case():
    state = fake_state(n_personas=5, seed=21)
    policy = MLPPolicy(DIMS, N_ITEMS, hidden=4, horizon=8, seed=4, scale=0.5)
    rng = np.random.default_rng(22)
    actions = np.array([policy.act(state, rng)]).reshape(-1)
    weights = rng.normal(size=state.n_personas)
    beta = 0.05

    x, hidden, probs = policy.distribution(state)
    analytic = policy.backward(x, hidden, probs, actions, weights, entropy_bonus=beta)

    eps = 1e-6
    for name, grad in analytic.items():
        array = getattr(policy, name)
        flat = array.reshape(-1)
        checked = np.linspace(0, flat.size - 1, min(8, flat.size)).astype(int)
        for index in checked:
            original = flat[index]
            flat[index] = original + eps
            up = objective(policy, state, actions, weights, beta)
            flat[index] = original - eps
            down = objective(policy, state, actions, weights, beta)
            flat[index] = original
            numeric = (up - down) / (2 * eps)
            assert grad.reshape(-1)[index] == pytest.approx(numeric, abs=2e-6), (
                f"{name}[{index}] analytic {grad.reshape(-1)[index]} vs numeric {numeric}"
            )


def test_masked_items_receive_no_gradient():
    state = fake_state(n_personas=3, seed=31)
    policy = MLPPolicy(DIMS, N_ITEMS, hidden=4, horizon=8, seed=5, scale=0.5)
    x, hidden, probs = policy.distribution(state)
    actions = np.argmax(probs, axis=1)
    grads = policy.backward(x, hidden, probs, actions, np.ones(3), entropy_bonus=0.1)
    # item 0 is answered by persona 0 only, so its bias still moves through the
    # other two rows; an item answered by everybody must not move at all.
    everywhere = fake_state(n_personas=3, seed=31)
    mask = np.array(everywhere.answered)
    mask[:, 4] = True
    blocked = InterviewState(
        theta=everywhere.theta, cov=everywhere.cov, blur=everywhere.blur,
        answered=mask, n_asked=mask.sum(axis=1), step=everywhere.step,
    )
    x, hidden, probs = policy.distribution(blocked)
    actions = np.argmax(probs, axis=1)
    grads = policy.backward(x, hidden, probs, actions, np.ones(3), entropy_bonus=0.1)
    assert grads["b2"][4] == pytest.approx(0.0)
    assert grads["w2"][:, 4] == pytest.approx(np.zeros(policy.hidden))


def test_returns_to_go_credit_an_action_with_what_came_after_it():
    rewards = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    assert returns_to_go(rewards, 1.0) == pytest.approx(
        np.array([[9.0, 12.0], [8.0, 10.0], [5.0, 6.0]])
    )
    assert returns_to_go(rewards, 0.0) == pytest.approx(rewards)


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------


def test_the_same_seed_gives_the_same_initial_weights():
    a = MLPPolicy(DIMS, N_ITEMS, hidden=6, seed=99)
    b = MLPPolicy(DIMS, N_ITEMS, hidden=6, seed=99)
    c = MLPPolicy(DIMS, N_ITEMS, hidden=6, seed=100)
    for name in a.parameters:
        assert getattr(a, name) == pytest.approx(getattr(b, name))
    assert not np.allclose(a.w2, c.w2)


def test_the_iteration_generator_depends_only_on_the_seed_and_the_index():
    first = iteration_rng(7, 12).random(5)
    again = iteration_rng(7, 12).random(5)
    other = iteration_rng(7, 13).random(5)
    assert first == pytest.approx(again)
    assert not np.allclose(first, other)


def test_two_runs_with_the_same_config_end_at_the_same_weights(tmp_path):
    cache = fake_cache(tmp_path)
    params = fake_params()
    config = TrainConfig(
        horizon=3, iterations=6, batch_personas=4, hidden=5, seed=17,
        checkpoint_every=0, eval_personas=4,
    )
    first = train(cache, params, config, out_dir=tmp_path / "a", verbose=False)
    second = train(cache, params, config, out_dir=tmp_path / "b", verbose=False)
    a, _ = MLPPolicy.load(first["artifacts"]["final_policy"])
    b, _ = MLPPolicy.load(second["artifacts"]["final_policy"])
    for name in a.parameters:
        assert getattr(a, name) == pytest.approx(getattr(b, name))


def test_resuming_from_a_checkpoint_lands_where_an_uninterrupted_run_would(tmp_path):
    cache = fake_cache(tmp_path)
    params = fake_params()
    common = dict(horizon=3, batch_personas=4, hidden=5, seed=23, checkpoint_every=2,
                  eval_personas=4)
    whole = train(cache, params, TrainConfig(iterations=8, **common),
                  out_dir=tmp_path / "whole", verbose=False)
    train(cache, params, TrainConfig(iterations=4, **common),
          out_dir=tmp_path / "split", verbose=False)
    resumed = train(cache, params, TrainConfig(iterations=8, **common),
                    out_dir=tmp_path / "split", verbose=False)
    a, _ = MLPPolicy.load(whole["artifacts"]["final_policy"])
    b, _ = MLPPolicy.load(resumed["artifacts"]["final_policy"])
    for name in a.parameters:
        assert getattr(a, name) == pytest.approx(getattr(b, name))
    assert resumed["costs"]["iterations_run"] == 4


# --------------------------------------------------------------------------
# the loop does something
# --------------------------------------------------------------------------


def test_training_raises_the_declared_reward_and_never_re_asks_an_item(tmp_path):
    cache = fake_cache(tmp_path, n_personas=24, n_rounds=4)
    params = fake_params()
    config = TrainConfig(
        horizon=4, iterations=40, batch_personas=8, hidden=8, lr=0.05, seed=41,
        checkpoint_every=0, eval_personas=8,
    )
    report = train(cache, params, config, out_dir=tmp_path / "run", verbose=False)
    returns = np.array([row["episode_return"] for row in report["history"]])
    assert returns[-10:].mean() > returns[:10].mean()
    assert sum(report["item_choice_counts"]) == config.iterations * config.batch_personas * config.horizon
    # the simulator raises on a repeat, so finishing at all proves the mask held
    assert report["costs"]["episodes_simulated"] == config.iterations * config.batch_personas


def test_the_trainer_uses_whatever_reward_it_is_handed(tmp_path):
    """The seam the exploratory oracle arm goes through."""
    cache = fake_cache(tmp_path)
    params = fake_params()
    seen: list[int] = []

    def constant(ctx: RewardContext) -> np.ndarray:
        seen.append(ctx.step)
        return np.zeros(ctx.theta_after.shape[0])

    config = TrainConfig(horizon=3, iterations=2, batch_personas=4, hidden=4,
                         seed=1, checkpoint_every=0, eval_personas=4)
    train(cache, params, config, out_dir=tmp_path / "run", reward_fn=constant, verbose=False)
    assert seen == [0, 1, 2, 0, 1, 2]


# --------------------------------------------------------------------------
# the quarantine (owner RULING 2)
# --------------------------------------------------------------------------


def save_arm(tmp_path: Path, arm: str, reward: str, name: str) -> Path:
    policy = MLPPolicy(DIMS, N_ITEMS, hidden=4, horizon=5, seed=0)
    return policy.save(tmp_path / name, arm=arm, reward=reward)


def test_the_confirmatory_loader_refuses_the_exploratory_oracle_weights(tmp_path):
    oracle = save_arm(tmp_path, ORACLE_ARM, ORACLE_REWARD, "exploratory_oracle.npz")
    with pytest.raises(ValueError, match="refusing to load"):
        load_confirmatory(oracle)
    # the generic loader still works: the proxy watch has to be able to read it
    loaded, meta = MLPPolicy.load(oracle)
    assert meta["arm"] == ORACLE_ARM and loaded.n_items == N_ITEMS


def test_the_confirmatory_loader_refuses_confirmatory_weights_with_a_truth_reward(tmp_path):
    """The arm label alone is not enough -- the reward name has to agree."""
    mislabelled = save_arm(tmp_path, CONFIRMATORY_ARM, ORACLE_REWARD, "mislabelled.npz")
    with pytest.raises(ValueError, match="refusing to load"):
        load_confirmatory(mislabelled)


def test_the_confirmatory_loader_accepts_the_confirmatory_arm(tmp_path):
    good = save_arm(tmp_path, CONFIRMATORY_ARM, BLUR_REWARD, "policy_final.npz")
    loaded, meta = load_confirmatory(good)
    assert meta["arm"] == CONFIRMATORY_ARM
    assert meta["reward"] == BLUR_REWARD
    assert loaded.n_items == N_ITEMS


def test_the_gate5_episode_runner_cannot_load_the_oracle_arm(tmp_path):
    """The confirmatory entry point, at the one place weights enter a result."""
    oracle = save_arm(tmp_path, ORACLE_ARM, ORACLE_REWARD, "exploratory_oracle.npz")
    rng = np.random.default_rng(0)
    n_personas = 6
    task = {
        "arm": gate5.POLICY,
        "strategy": gate5.POLICY,
        "replicate": 0,
        "round": "ep_policy_000",
        "codes": rng.integers(0, 5, size=(n_personas, N_ITEMS)).astype(np.int16),
        "loadings": fake_params().loadings,
        "cut_raw": fake_params().cut_raw,
        "kinds": [mirt.LIKERT] * N_ITEMS,
        "scripted_columns": [0, 1],
        "seed": 0,
        "theta_true_train": rng.normal(size=(3, DIMS)),
        "theta_true_holdout": rng.normal(size=(3, DIMS)),
        "kappa": np.ones(DIMS),
        "n_train": 3,
        "n_steps": 2,
        "hessian": "analytic",
        "policy_path": str(oracle),
    }
    with pytest.raises(ValueError, match="refusing to load"):
        gate5.run_episode(task)

    # and the same task with confirmatory weights runs
    good = save_arm(tmp_path, CONFIRMATORY_ARM, BLUR_REWARD, "policy_final.npz")
    task["policy_path"] = str(good)
    result = gate5.run_episode(task)
    assert result["strategy"] == gate5.POLICY
    assert result["n_steps"] == 2


def test_a_saved_policy_records_its_arm_its_reward_and_its_config(tmp_path):
    policy = MLPPolicy(DIMS, N_ITEMS, hidden=4, horizon=5, seed=0)
    path = policy.save(
        tmp_path / "p.npz", arm=CONFIRMATORY_ARM, reward=BLUR_REWARD,
        config={"lr": 0.02}, extra={"iteration": 7},
    )
    loaded, meta = MLPPolicy.load(path)
    assert meta["config"]["lr"] == 0.02
    assert meta["extra"]["iteration"] == 7
    assert loaded.horizon == 5
    for name in policy.parameters:
        assert getattr(loaded, name) == pytest.approx(getattr(policy, name))


def test_saving_an_unknown_arm_is_refused(tmp_path):
    policy = MLPPolicy(DIMS, N_ITEMS, hidden=4, seed=0)
    with pytest.raises(ValueError, match="unknown arm"):
        policy.save(tmp_path / "p.npz", arm="whatever", reward=BLUR_REWARD)


# --------------------------------------------------------------------------
# the entry point's split guard
# --------------------------------------------------------------------------


def test_the_entry_point_refuses_to_train_on_a_held_out_persona(tmp_path):
    from src.rl.train import refuse_holdout

    fit = {"persona_holdout_ids": np.array(["p0007", "p0009"])}
    refuse_holdout(["r0001", "r0002"], fit)  # fine
    with pytest.raises(ValueError, match="held-out"):
        refuse_holdout(["r0001", "p0009"], fit)


def test_persona_ids_are_read_in_first_seen_order(tmp_path):
    from src.rl.train import persona_ids_from_jsonl

    path = tmp_path / "answers.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"pid": pid, "item_id": "q001", "answer": 3})
            for pid in ("r0002", "r0001", "r0002", "r0003")
        )
        + "\n",
        encoding="utf-8",
    )
    assert persona_ids_from_jsonl(path) == ["r0002", "r0001", "r0003"]


def test_the_config_file_holds_every_knob_the_trainer_reads():
    document = json.loads(
        (REPO_ROOT / "experiments" / "stage5_rl.json").read_text(encoding="utf-8")
    )
    fields = set(TrainConfig.__dataclass_fields__)
    for name, block in document["profiles"].items():
        keys = set(block) - {"notes", "n_rounds"}
        assert keys <= fields, f"{name}: unknown keys {sorted(keys - fields)}"
        for knob in ("lr", "entropy_bonus", "batch_personas", "seed", "horizon"):
            assert knob in keys, f"{name}: {knob} is not recorded"


def test_the_episode_cache_refuses_a_persona_it_does_not_hold(tmp_path):
    cache = fake_cache(tmp_path)
    assert cache.rows_for(["x0002", "x0000"]).tolist() == [2, 0]
    with pytest.raises(ValueError, match="not in"):
        cache.rows_for(["nobody"])


def test_the_trainer_refuses_a_batch_bigger_than_the_persona_pool(tmp_path):
    cache = fake_cache(tmp_path, n_personas=5)
    with pytest.raises(ValueError, match="only 5 personas"):
        train(
            cache, fake_params(),
            TrainConfig(horizon=2, iterations=1, batch_personas=8, checkpoint_every=0),
            out_dir=tmp_path / "run", verbose=False,
        )


def test_the_trainer_refuses_a_cache_drawn_on_a_different_bank(tmp_path):
    cache = fake_cache(tmp_path, n_items=N_ITEMS)
    with pytest.raises(ValueError, match="candidate items"):
        train(
            cache, fake_params(n_items=N_ITEMS + 1),
            TrainConfig(horizon=2, iterations=1, batch_personas=4, checkpoint_every=0),
            out_dir=tmp_path / "run", verbose=False,
        )
