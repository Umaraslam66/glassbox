"""REINFORCE for the Stage 5 interviewer. Deliberately the simple version.

PRD section 7, Stage 5: "Keep RL simple first; escalate complexity only with
owner approval. If RL is unstable after a bounded effort (log the budget), ship
the heuristic result with the RL attempt documented." So: vanilla policy
gradient, a moving-average baseline, a small entropy bonus, a fixed horizon. No
critic, no trust region, no replay.

The loop
--------
One iteration is one batch of interviews:

1. take a batch of training personas and one cached sitting of their answers;
2. run ``horizon`` questions, sampling each one from the policy;
3. after every question, ask the reward callable what that question was worth;
4. turn the rewards into returns, subtract the baseline, and take one Adam step
   on ``mean[ advantage * log pi(a | s) ] + entropy_bonus * mean[ H(pi) ]``.

Every persona in the batch is an independent trajectory -- a different person,
a different set of answers -- so a batch of 64 is 64 samples of the gradient,
not one.

Where the variance reduction comes from
---------------------------------------
Three cheap things, in order of how much they matter here:

* **Reward-to-go.** An action is credited with what happened after it, not with
  the whole episode. Without this the first question and the last one get the
  same signal.
* **A per-step moving-average baseline.** The return from step 1 is much larger
  than the return from step 24, simply because there is more interview left, so
  one global baseline would leave a large step-dependent bias in the advantage.
  One scalar per step index, updated with an exponential moving average, removes
  it. This is the "moving-average baseline" of the spec, kept per step because a
  single number would be strictly worse for no saving.
* **Advantage normalisation.** Dividing the advantage by its standard deviation
  across the batch, which makes the effective step size independent of how large
  the trace reductions happen to be at that point in training. Switchable, on by
  default, and recorded in the config.

Determinism and resume
----------------------
The batch, the round and the action draws at iteration ``i`` come from a
generator seeded by ``(seed, i)`` alone -- never from a running stream. So the
run is reproducible from the config, and resuming from a checkpoint at iteration
``i`` continues with exactly the draws an uninterrupted run would have made.
Checkpoints hold the weights, the optimiser state, the baselines and the history,
so the entry point can be killed and restarted without losing or repeating work.

Truth never enters
------------------
The trainer calls whichever reward callable it was handed and never constructs
one. The confirmatory callable is :func:`src.rl.reward.blur_reward`, which is
arithmetic on the posterior. The exploratory oracle callable is built in
``src/eval/`` and injected. This module imports nothing from ``src/eval``, and
``tests/test_rl.py`` parses every file under ``src/rl`` to check that no truth
path can be reached from here.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..interview.episodes import AnswerBank
from ..model import mirt
from ..model.sequential_posterior import SequentialPosterior
from .policy import (
    BLUR_REWARD,
    CONFIRMATORY_ARM,
    MLPPolicy,
    masked_log_softmax,
    sample_actions,
)
from .reward import REWARD_DEFINITION, RewardContext, RewardFn, blur_reward, posterior_trace

__all__ = [
    "EpisodeCache",
    "TrainConfig",
    "evaluate_declared",
    "draw_training_curves",
    "rollout",
    "train",
]


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainConfig:
    """Everything that changes an outcome. Written into every artifact."""

    horizon: int = 25
    iterations: int = 600
    batch_personas: int = 64
    lr: float = 0.02
    entropy_bonus: float = 0.01
    question_cost: float = 0.02
    baseline_decay: float = 0.9
    advantage_norm: bool = True
    discount: float = 1.0
    hidden: int = 64
    grad_clip: float = 5.0
    seed: int = 0
    init_scale: float = 0.01
    checkpoint_every: int = 50
    eval_personas: int = 100
    hessian: str = "analytic"

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TrainConfig":
        known = {f for f in cls.__dataclass_fields__}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"unknown config keys: {unknown}")
        return cls(**values)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def iteration_rng(seed: int, iteration: int) -> np.random.Generator:
    """The generator for one iteration, a pure function of the seed and index."""
    return np.random.default_rng([int(seed), int(iteration)])


# --------------------------------------------------------------------------
# the cached episodes
# --------------------------------------------------------------------------


class EpisodeCache:
    """The drawn sittings the trainer replays, and who is in them.

    An episode cache is what :mod:`src.interview.episodes` writes: coded answers
    for ``(rounds, personas, items)``, already through the frozen noise layer.
    The trainer never draws answers itself -- it replays these -- which is what
    keeps training a pure CPU job with no responder in the loop.
    """

    def __init__(self, path: str | Path) -> None:
        archive = np.load(Path(path), allow_pickle=False)
        self.path = str(path)
        self.codes = np.asarray(archive["codes"])
        self.rounds = [str(r) for r in archive["rounds"]]
        self.persona_ids = [str(p) for p in archive["persona_ids"]]
        self.item_ids = [str(i) for i in archive["item_ids"]]
        if self.codes.ndim != 3:
            raise ValueError(f"codes must be (rounds, personas, items), got {self.codes.shape}")
        if self.codes.shape[1] != len(self.persona_ids):
            raise ValueError("codes and persona ids disagree")
        if self.codes.shape[2] != len(self.item_ids):
            raise ValueError("codes and item ids disagree")

    @property
    def n_rounds(self) -> int:
        return int(self.codes.shape[0])

    @property
    def n_personas(self) -> int:
        return int(self.codes.shape[1])

    @property
    def n_items(self) -> int:
        return int(self.codes.shape[2])

    def rows_for(self, persona_ids: Sequence[str]) -> np.ndarray:
        """Row indices of the named personas, in the order given."""
        lookup = {pid: row for row, pid in enumerate(self.persona_ids)}
        missing = [p for p in persona_ids if p not in lookup]
        if missing:
            raise ValueError(
                f"{len(missing)} persona(s) are not in {self.path}, first: {missing[0]}"
            )
        return np.array([lookup[str(p)] for p in persona_ids], dtype=np.int64)

    def bank(self, round_index: int, rows: np.ndarray) -> AnswerBank:
        """One sitting's answers for a batch of personas, kept private."""
        block = self.codes[int(round_index)][np.asarray(rows, dtype=np.int64)]
        return AnswerBank(block, self.rounds[int(round_index)])


# --------------------------------------------------------------------------
# Adam
# --------------------------------------------------------------------------


class Adam:
    """The usual thing, ten lines, so the run does not depend on a step-size guess."""

    def __init__(self, shapes: dict[str, tuple[int, ...]], lr: float,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        self.lr = float(lr)
        self.beta1, self.beta2, self.eps = float(beta1), float(beta2), float(eps)
        self.m = {k: np.zeros(s) for k, s in shapes.items()}
        self.v = {k: np.zeros(s) for k, s in shapes.items()}
        self.t = 0

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        """In-place ASCENT: ``grads`` point the way the objective increases."""
        self.t += 1
        for name, grad in grads.items():
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * grad**2
            m_hat = self.m[name] / (1 - self.beta1**self.t)
            v_hat = self.v[name] / (1 - self.beta2**self.t)
            params[name] += self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def state(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {f"adam_m_{k}": v for k, v in self.m.items()}
        out.update({f"adam_v_{k}": v for k, v in self.v.items()})
        out["adam_t"] = np.array(self.t)
        return out

    def load_state(self, archive: Any) -> None:
        for name in list(self.m):
            self.m[name] = np.asarray(archive[f"adam_m_{name}"], dtype=float)
            self.v[name] = np.asarray(archive[f"adam_v_{name}"], dtype=float)
        self.t = int(archive["adam_t"])


# --------------------------------------------------------------------------
# one batch of episodes
# --------------------------------------------------------------------------


def rollout(
    policy: MLPPolicy,
    posterior: SequentialPosterior,
    bank: AnswerBank,
    persona_rows: np.ndarray,
    reward_fn: RewardFn,
    *,
    horizon: int,
    question_cost: float,
    rng: np.random.Generator,
    greedy: bool = False,
) -> dict[str, Any]:
    """Run one batch of interviews and keep what the gradient needs.

    Returns the per-step features, hidden activations, action distributions,
    actions and rewards, plus the declared-blur trace at every step (system-side
    bookkeeping, recorded whatever reward is in use).
    """
    horizon = int(horizon)
    posterior.reset()
    xs, hs, ps, actions, rewards = [], [], [], [], []
    traces = [float(posterior_trace(posterior.cov).mean())]
    blurs = [float(posterior.blur.mean())]
    entropies: list[float] = []

    for step in range(horizon):
        state = posterior.state()
        x = policy.features(state)
        h, logits = policy.forward(x)
        log_p = masked_log_softmax(logits, state.answered)
        probs = np.exp(log_p)
        if greedy:
            columns = np.argmax(np.where(state.answered, -np.inf, logits), axis=1).astype(np.int64)
        else:
            columns = sample_actions(probs, rng)

        theta_before = np.array(posterior.theta, copy=True)
        cov_before = np.array(posterior.cov, copy=True)
        answers = bank.reveal(columns)
        posterior.observe(columns, answers)

        reward = np.asarray(
            reward_fn(
                RewardContext(
                    step=step,
                    columns=columns,
                    persona_rows=np.asarray(persona_rows, dtype=np.int64),
                    theta_before=theta_before,
                    theta_after=np.array(posterior.theta, copy=True),
                    cov_before=cov_before,
                    cov_after=np.array(posterior.cov, copy=True),
                    question_cost=float(question_cost),
                )
            ),
            dtype=float,
        ).reshape(-1)
        if reward.shape[0] != x.shape[0]:
            raise ValueError(
                f"the reward returned {reward.shape[0]} numbers for "
                f"{x.shape[0]} personas"
            )

        # masked entries have p = 0 and log p = -inf; 0 * -inf is a nan, so the
        # log is zeroed there first rather than the product afterwards.
        entropy = -np.sum(probs * np.where(probs > 0.0, log_p, 0.0), axis=1)
        entropies.append(float(entropy.mean()))
        xs.append(x)
        hs.append(h)
        ps.append(probs)
        actions.append(columns)
        rewards.append(reward)
        traces.append(float(posterior_trace(posterior.cov).mean()))
        blurs.append(float(posterior.blur.mean()))

    return {
        "features": xs,
        "hidden": hs,
        "probs": ps,
        "actions": actions,
        "rewards": np.stack(rewards),  # (horizon, batch)
        "entropy": entropies,
        "trace_by_n": traces,  # length horizon + 1, index 0 is the prior
        "blur_by_n": blurs,
    }


def returns_to_go(rewards: np.ndarray, discount: float) -> np.ndarray:
    """``G_t = sum_{t' >= t} gamma^{t'-t} r_{t'}``, shaped like ``rewards``."""
    rewards = np.asarray(rewards, dtype=float)
    out = np.zeros_like(rewards)
    running = np.zeros(rewards.shape[1])
    for t in range(rewards.shape[0] - 1, -1, -1):
        running = rewards[t] + float(discount) * running
        out[t] = running
    return out


# --------------------------------------------------------------------------
# a system-side readout of a checkpoint
# --------------------------------------------------------------------------


def evaluate_declared(
    policy: MLPPolicy,
    cache: EpisodeCache,
    params: mirt.ItemParams,
    rows: np.ndarray,
    *,
    round_index: int,
    horizon: int,
    hessian: str = "analytic",
) -> dict[str, Any]:
    """What the policy declares about itself: blur and trace against N.

    Greedy, so it is the deterministic arm the grader would run. Everything
    reported here is system-side; the matching truth-side numbers are computed
    later, by a separate pass in ``src/eval`` over the saved checkpoints, and
    never in this loop (owner RULING 2).
    """
    rows = np.asarray(rows, dtype=np.int64)
    posterior = SequentialPosterior(params, rows.size, hessian=hessian)
    bank = cache.bank(round_index, rows)
    trace = rollout(
        policy,
        posterior,
        bank,
        rows,
        blur_reward,
        horizon=horizon,
        question_cost=0.0,
        rng=np.random.default_rng(0),
        greedy=True,
    )
    chosen = np.stack(trace["actions"])  # (horizon, batch)
    return {
        "round": cache.rounds[int(round_index)],
        "n_personas": int(rows.size),
        "horizon": int(horizon),
        "declared_blur_by_n": trace["blur_by_n"],
        "declared_trace_by_n": trace["trace_by_n"],
        "blur_remaining": float(trace["blur_by_n"][-1] / max(trace["blur_by_n"][0], 1e-12)),
        "trace_remaining": float(trace["trace_by_n"][-1] / max(trace["trace_by_n"][0], 1e-12)),
        "n_distinct_items": int(np.unique(chosen).size),
    }


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------


def _checkpoint_path(out_dir: Path) -> Path:
    return out_dir / "checkpoint.npz"


def _snapshot_path(out_dir: Path, iteration: int) -> Path:
    return out_dir / f"policy_iter{int(iteration):06d}.npz"


def train(
    cache: EpisodeCache,
    params: mirt.ItemParams,
    config: TrainConfig,
    *,
    out_dir: str | Path,
    persona_rows: np.ndarray | None = None,
    reward_fn: RewardFn | None = None,
    arm: str = CONFIRMATORY_ARM,
    reward_name: str = BLUR_REWARD,
    resume: bool = True,
    verbose: bool = True,
    progress_every: int = 20,
) -> dict[str, Any]:
    """Train one policy. Returns the run's report; writes checkpoints as it goes.

    ``persona_rows`` restricts training to a subset of the cache's personas and
    is written into the report, so an artifact says exactly whose interviews
    shaped the weights. Passing the held-out personas here would be a split
    violation; the entry point is what refuses to, and it records the persona ids
    it used.
    """
    started = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reward_fn = blur_reward if reward_fn is None else reward_fn

    rows = (
        np.arange(cache.n_personas, dtype=np.int64)
        if persona_rows is None
        else np.asarray(persona_rows, dtype=np.int64)
    )
    if rows.size < config.batch_personas:
        raise ValueError(
            f"batch of {config.batch_personas} asked for, but only {rows.size} "
            "personas are available to train on"
        )
    dims = int(params.loadings.shape[1])
    n_items = int(params.loadings.shape[0])
    if n_items != cache.n_items:
        raise ValueError(
            f"the fit has {n_items} candidate items, the episode cache has {cache.n_items}"
        )

    policy = MLPPolicy(
        dims,
        n_items,
        hidden=config.hidden,
        horizon=config.horizon,
        seed=config.seed,
        scale=config.init_scale,
    )
    optimizer = Adam({k: v.shape for k, v in policy.parameters.items()}, config.lr)
    baseline = np.zeros(config.horizon)
    baseline_seen = False
    item_counts = np.zeros(n_items, dtype=np.int64)
    history: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    start_iteration = 0

    checkpoint = _checkpoint_path(out_dir)
    if resume and checkpoint.is_file():
        archive = np.load(checkpoint, allow_pickle=False)
        policy.set_parameters({k: archive[k] for k in ("w1", "b1", "w2", "b2")})
        optimizer.load_state(archive)
        baseline = np.asarray(archive["baseline"], dtype=float)
        baseline_seen = bool(archive["baseline_seen"])
        item_counts = np.asarray(archive["item_counts"], dtype=np.int64)
        history = json.loads(str(archive["history_json"]))
        checkpoints = json.loads(str(archive["checkpoints_json"]))
        start_iteration = int(archive["iteration"])
        if verbose:
            print(f"resuming from {checkpoint} at iteration {start_iteration}")

    posterior = SequentialPosterior(params, config.batch_personas, hessian=config.hessian)
    eval_rows = rows[: min(int(config.eval_personas), rows.size)]

    def write_checkpoint(iteration: int) -> None:
        payload: dict[str, Any] = {k: v for k, v in policy.parameters.items()}
        payload.update(optimizer.state())
        payload["baseline"] = baseline
        payload["baseline_seen"] = np.array(baseline_seen)
        payload["item_counts"] = item_counts
        payload["iteration"] = np.array(int(iteration))
        payload["history_json"] = np.array(json.dumps(history))
        payload["checkpoints_json"] = np.array(json.dumps(checkpoints))
        np.savez(checkpoint, **payload)

    def snapshot(iteration: int) -> None:
        readout = evaluate_declared(
            policy, cache, params, eval_rows,
            round_index=iteration % cache.n_rounds,
            horizon=config.horizon,
            hessian=config.hessian,
        )
        path = _snapshot_path(out_dir, iteration)
        policy.save(
            path,
            arm=arm,
            reward=reward_name,
            config=config.as_dict(),
            extra={"iteration": int(iteration), "declared": readout},
        )
        checkpoints.append(
            {"iteration": int(iteration), "policy": str(path), "declared": readout}
        )

    if start_iteration == 0 and not checkpoints:
        snapshot(0)

    for iteration in range(start_iteration, config.iterations):
        rng = iteration_rng(config.seed, iteration)
        batch = rng.choice(rows, size=config.batch_personas, replace=False)
        round_index = int(rng.integers(cache.n_rounds))
        bank = cache.bank(round_index, batch)

        trace = rollout(
            policy, posterior, bank, batch, reward_fn,
            horizon=config.horizon,
            question_cost=config.question_cost,
            rng=rng,
        )
        rewards = trace["rewards"]
        returns = returns_to_go(rewards, config.discount)

        if not baseline_seen:
            baseline = returns.mean(axis=1)
            baseline_seen = True
        advantage = returns - baseline[:, None]
        if config.advantage_norm:
            spread = advantage.std(axis=1, keepdims=True)
            advantage = advantage / np.maximum(spread, 1e-8)
        baseline = (
            config.baseline_decay * baseline
            + (1.0 - config.baseline_decay) * returns.mean(axis=1)
        )

        grads = {k: np.zeros_like(v) for k, v in policy.parameters.items()}
        for t in range(config.horizon):
            piece = policy.backward(
                trace["features"][t],
                trace["hidden"][t],
                trace["probs"][t],
                trace["actions"][t],
                advantage[t],
                entropy_bonus=config.entropy_bonus,
            )
            for name in grads:
                grads[name] += piece[name] / config.horizon
            item_counts += np.bincount(trace["actions"][t], minlength=n_items)

        norm = float(np.sqrt(sum(float((g**2).sum()) for g in grads.values())))
        if config.grad_clip and norm > config.grad_clip:
            for name in grads:
                grads[name] *= config.grad_clip / norm
        params_ref = policy.parameters
        optimizer.step(params_ref, grads)

        history.append(
            {
                "iteration": int(iteration),
                "round": cache.rounds[round_index],
                "mean_reward_per_question": float(rewards.mean()),
                "episode_return": float(returns[0].mean()),
                "entropy": float(np.mean(trace["entropy"])),
                "grad_norm": norm,
                "declared_trace_final": float(trace["trace_by_n"][-1]),
                "declared_blur_final": float(trace["blur_by_n"][-1]),
                "n_distinct_items": int(np.unique(np.stack(trace["actions"])).size),
            }
        )

        done = iteration + 1
        if config.checkpoint_every and done % config.checkpoint_every == 0:
            snapshot(done)
            write_checkpoint(done)
        if verbose and (done % progress_every == 0 or done == config.iterations):
            row = history[-1]
            print(
                f"  iter {done:5d}/{config.iterations}  return {row['episode_return']:8.4f}  "
                f"entropy {row['entropy']:6.3f}  blur {row['declared_blur_final']:.4f}  "
                f"|g| {row['grad_norm']:.3f}"
            )

    final = out_dir / "policy_final.npz"
    policy.save(
        final,
        arm=arm,
        reward=reward_name,
        config=config.as_dict(),
        extra={"iteration": int(config.iterations), "n_train_personas": int(rows.size)},
    )
    write_checkpoint(config.iterations)

    report = {
        "arm": arm,
        "reward": reward_name,
        "reward_definition": REWARD_DEFINITION if reward_name == BLUR_REWARD else reward_name,
        "config": config.as_dict(),
        "episodes": {
            "cache": cache.path,
            "n_rounds": cache.n_rounds,
            "n_items": cache.n_items,
            "n_personas_in_cache": cache.n_personas,
        },
        "training_personas": {
            "n": int(rows.size),
            "ids": [cache.persona_ids[int(r)] for r in rows],
        },
        "history": history,
        "checkpoints": checkpoints,
        "item_choice_counts": item_counts.tolist(),
        "item_ids": cache.item_ids,
        "artifacts": {
            "final_policy": str(final),
            "checkpoint": str(checkpoint),
            "snapshots": [c["policy"] for c in checkpoints],
        },
        "costs": {
            "wall_seconds": round(time.time() - started, 1),
            "iterations_run": int(config.iterations - start_iteration),
            "episodes_simulated": int(
                (config.iterations - start_iteration) * config.batch_personas
            ),
            "gpu": "none",
            "api": "none",
        },
    }
    (out_dir / "history.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


# --------------------------------------------------------------------------
# the training curves
# --------------------------------------------------------------------------


def draw_training_curves(report: dict[str, Any], path: str | Path, *, window: int = 20) -> Path:
    """Reward, entropy and the item-choice histogram, in one figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history = report["history"]
    iterations = [row["iteration"] for row in history]
    returns = np.array([row["episode_return"] for row in history], dtype=float)
    entropy = np.array([row["entropy"] for row in history], dtype=float)
    counts = np.array(report["item_choice_counts"], dtype=float)

    def smooth(values: np.ndarray) -> np.ndarray:
        if values.size < window:
            return values
        kernel = np.ones(window) / window
        return np.convolve(values, kernel, mode="valid")

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))

    ax = axes[0]
    ax.plot(iterations, returns, color="#B0B7C3", lw=0.8, label="per iteration")
    if returns.size >= window:
        ax.plot(iterations[window - 1:], smooth(returns), color="#C44E52", lw=1.8,
                label=f"{window}-iteration mean")
    ax.set_xlabel("iteration")
    ax.set_ylabel("episode return (declared variance removed - cost)")
    ax.set_title("reward")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(iterations, entropy, color="#B0B7C3", lw=0.8)
    if entropy.size >= window:
        ax.plot(iterations[window - 1:], smooth(entropy), color="#4C72B0", lw=1.8)
    ax.axhline(float(np.log(report["episodes"]["n_items"])), color="#555555", ls=":", lw=1.0)
    ax.text(
        iterations[-1] if iterations else 0,
        float(np.log(report["episodes"]["n_items"])) - 0.08,
        "uniform over all items", fontsize=7, ha="right", color="#555555",
    )
    ax.set_xlabel("iteration")
    ax.set_ylabel("policy entropy (nats)")
    ax.set_title("exploration")
    ax.grid(alpha=0.25)

    ax = axes[2]
    order = np.argsort(counts)[::-1]
    ax.bar(np.arange(counts.size), counts[order], color="#55A868", width=1.0)
    used = int((counts > 0).sum())
    ax.set_xlabel(f"candidate item, ranked ({used} of {counts.size} ever chosen)")
    ax.set_ylabel("times chosen during training")
    ax.set_title("item-choice histogram")
    ax.grid(alpha=0.25)

    fig.suptitle(
        f"Stage 5 RL interviewer -- {report['arm']} arm, reward: {report['reward']}",
        fontsize=12,
    )
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
