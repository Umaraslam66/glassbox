"""The Stage 5 RL interviewer's policy: one small MLP over the interview state.

What it is
----------
A two-layer network, written in numpy because the repository has no deep
learning dependency and this problem does not need one: 219 inputs, one tanh
hidden layer, 202 outputs -- about 27,000 numbers in total. It runs on a laptop
CPU, and adding a framework to train it would cost more than the training does.

    input   = [ theta_hat (8) | per-dimension blur (8) | answered mask (202) |
                step index (1) ]
    hidden  = tanh(x W1 + b1)
    logits  = hidden W2 + b2
    output  = softmax over the items this persona has NOT been asked

The four input blocks are exactly the four things
:class:`~src.model.sequential_posterior.InterviewState` carries. That is not a
coincidence -- the state object is the complete interface a question-picking
strategy has (see :mod:`src.interview.strategies`), so a policy that reads all
of it and nothing else is the largest policy the Wall permits. There is no
persona identifier, no answer distribution and no planted value on the input,
because there is no route to any of them from here.

Masking
-------
An item a persona has already answered gets a logit of ``-inf`` before the
softmax, so its probability is exactly zero and it receives exactly zero
gradient. This is what keeps the policy legal in the simulator, which raises if
a strategy re-asks a question (:meth:`src.interview.episodes.AnswerBank.reveal`
and :meth:`src.model.sequential_posterior.SequentialPosterior.observe` both
check). Masking in the policy rather than rejection-sampling afterwards also
keeps the log-probability the trainer differentiates the *same* quantity the
action was drawn from, which a rejection scheme would quietly break.

The step feature
----------------
``step / horizon``, clipped to ``[0, 1]``. Clipping matters: the policy is
trained at horizon 25 but the Gate 5 harness runs strategies out to N = 150, and
without the clip every question past the 25th would feed the network a number it
never saw in training. Clipped, a long interview simply looks to the policy like
"still at the end", which is the honest extrapolation.

Greedy at evaluation, sampled during training
---------------------------------------------
:meth:`MLPPolicy.__call__` -- the :class:`src.interview.strategies.Policy`
interface -- takes the arg-max. The graded arm is then deterministic given the
weights, exactly as the info-gain heuristic is deterministic given the fit, so a
replicate's spread is spread in the interview and not in the interviewer.
Training samples instead, because REINFORCE needs the score function of an
action that was actually drawn from the policy.

Quarantine (owner RULING 2)
---------------------------
Every saved policy carries its ``arm`` and the name of the reward it was trained
on. :func:`load_confirmatory` refuses anything that is not the blur-reward
confirmatory arm, and it is the only loader the confirmatory code path calls, so
the exploratory oracle-reward weights cannot be loaded into a confirmatory
result even by mistake. ``tests/test_rl.py`` asserts the refusal.

WALL. Arithmetic on system-side quantities only. Nothing in this module reads a
persona card, an answer distribution or a planted value, and nothing here names
a path to one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..model.sequential_posterior import InterviewState

__all__ = [
    "BLUR_REWARD",
    "CONFIRMATORY_ARM",
    "ORACLE_ARM",
    "ORACLE_REWARD",
    "MLPPolicy",
    "load_confirmatory",
    "masked_log_softmax",
    "sample_actions",
    "state_features",
]

#: The two arms a saved policy can belong to. Only the first is confirmatory.
CONFIRMATORY_ARM = "confirmatory"
ORACLE_ARM = "exploratory_oracle"

#: The reward names that go with them, written out so an artifact says what it is.
BLUR_REWARD = "declared_posterior_variance_reduction_minus_question_cost"
ORACLE_REWARD = "truth_side_squared_error_reduction_minus_question_cost"

_NEG_INF = -1e30


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------


def state_features(state: InterviewState, horizon: int) -> np.ndarray:
    """``(n_personas, 2 * dims + n_items + 1)`` -- the whole state, flattened.

    Order is fixed and recorded here because a saved policy's weights are only
    meaningful against it: point estimate, per-dimension blur, answered mask,
    then the clipped step index.
    """
    n = state.theta.shape[0]
    step = min(float(state.step) / float(max(horizon, 1)), 1.0)
    return np.concatenate(
        [
            np.asarray(state.theta, dtype=float),
            np.asarray(state.blur, dtype=float),
            np.asarray(state.answered, dtype=float),
            np.full((n, 1), step),
        ],
        axis=1,
    )


def masked_log_softmax(logits: np.ndarray, answered: np.ndarray) -> np.ndarray:
    """Log-probabilities over the unasked items; answered items get ``-inf``.

    Stable in the usual way (subtract the row max), and the max is taken over
    the *unmasked* entries so a row whose best item is already answered does not
    silently normalise against it.
    """
    blocked = np.asarray(answered, dtype=bool)
    z = np.where(blocked, _NEG_INF, np.asarray(logits, dtype=float))
    z = z - z.max(axis=1, keepdims=True)
    exp = np.where(blocked, 0.0, np.exp(z))
    total = exp.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore"):
        out = z - np.log(total)
    return np.where(blocked, -np.inf, out)


def sample_actions(probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One item column per persona, drawn from each row of ``probs``.

    Inverse-CDF rather than :meth:`Generator.choice` per row: it is one
    vectorised pass and, more usefully here, it consumes exactly one uniform per
    persona per step, so the stream of draws is a function of the seed and the
    step count alone.
    """
    probs = np.asarray(probs, dtype=float)
    cumulative = np.cumsum(probs, axis=1)
    cumulative[:, -1] = 1.0
    draws = rng.random(probs.shape[0])[:, None]
    return np.argmax(cumulative >= draws, axis=1).astype(np.int64)


# --------------------------------------------------------------------------
# the network
# --------------------------------------------------------------------------


class MLPPolicy:
    """Two layers, tanh in the middle, a categorical distribution at the end.

    ``scale`` shrinks the output layer at initialisation so the first policy is
    very nearly uniform over the unasked items. That is deliberate: an episode
    is only 25 questions out of 202, so a policy that starts opinionated has
    almost no chance of discovering what the other items do.
    """

    def __init__(
        self,
        dims: int,
        n_items: int,
        *,
        hidden: int = 64,
        horizon: int = 25,
        seed: int = 0,
        scale: float = 0.01,
    ) -> None:
        self.dims = int(dims)
        self.n_items = int(n_items)
        self.hidden = int(hidden)
        self.horizon = int(horizon)
        self.seed = int(seed)
        self.n_features = 2 * self.dims + self.n_items + 1

        rng = np.random.default_rng(self.seed)
        # Xavier for the hidden layer, deliberately small for the output layer.
        self.w1 = rng.normal(
            0.0, np.sqrt(1.0 / self.n_features), size=(self.n_features, self.hidden)
        )
        self.b1 = np.zeros(self.hidden)
        self.w2 = rng.normal(
            0.0, float(scale) / np.sqrt(self.hidden), size=(self.hidden, self.n_items)
        )
        self.b2 = np.zeros(self.n_items)

    # ---- shapes ---------------------------------------------------------

    @property
    def parameters(self) -> dict[str, np.ndarray]:
        return {"w1": self.w1, "b1": self.b1, "w2": self.w2, "b2": self.b2}

    @property
    def n_parameters(self) -> int:
        return int(sum(p.size for p in self.parameters.values()))

    def set_parameters(self, values: dict[str, np.ndarray]) -> None:
        for name, array in values.items():
            current = getattr(self, name)
            array = np.asarray(array, dtype=float)
            if array.shape != current.shape:
                raise ValueError(
                    f"{name}: expected shape {current.shape}, got {array.shape}"
                )
            setattr(self, name, array.copy())

    # ---- forward --------------------------------------------------------

    def features(self, state: InterviewState) -> np.ndarray:
        return state_features(state, self.horizon)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(hidden, raw logits)``. The mask is applied by the caller."""
        h = np.tanh(np.asarray(x, dtype=float) @ self.w1 + self.b1)
        return h, h @ self.w2 + self.b2

    def distribution(self, state: InterviewState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(features, hidden, probabilities)`` for one interview state."""
        x = self.features(state)
        h, logits = self.forward(x)
        probs = np.exp(masked_log_softmax(logits, state.answered))
        return x, h, probs

    def act(self, state: InterviewState, rng: np.random.Generator) -> np.ndarray:
        """Sample one item column per persona. Used in training."""
        _, _, probs = self.distribution(state)
        return sample_actions(probs, rng)

    def greedy(self, state: InterviewState) -> np.ndarray:
        """The most likely unasked item per persona. Used in evaluation."""
        x = self.features(state)
        _, logits = self.forward(x)
        masked = np.where(np.asarray(state.answered, dtype=bool), -np.inf, logits)
        return np.argmax(masked, axis=1).astype(np.int64)

    def __call__(self, state: InterviewState) -> np.ndarray:
        """The :class:`src.interview.strategies.Policy` interface: greedy."""
        return self.greedy(state)

    # ---- backward -------------------------------------------------------

    def backward(
        self,
        x: np.ndarray,
        h: np.ndarray,
        probs: np.ndarray,
        actions: np.ndarray,
        weights: np.ndarray,
        *,
        entropy_bonus: float = 0.0,
    ) -> dict[str, np.ndarray]:
        """Gradient of ``mean_p [ w_p log pi(a_p | s_p) + beta H(pi_p) ]``.

        Returned as an ASCENT direction -- the trainer adds it. Two pieces reach
        the logits:

        * the score function, ``d log pi(a) / dz = onehot(a) - p``, scaled by the
          per-persona weight (the advantage);
        * the entropy bonus, ``dH/dz_j = -p_j (log p_j + H)``, which is zero
          wherever ``p_j`` is zero, so masked items stay masked.

        Everything after that is the ordinary two-layer chain rule.
        """
        x = np.asarray(x, dtype=float)
        h = np.asarray(h, dtype=float)
        probs = np.asarray(probs, dtype=float)
        actions = np.asarray(actions, dtype=np.int64)
        weights = np.asarray(weights, dtype=float).reshape(-1)
        n = x.shape[0]

        onehot = np.zeros_like(probs)
        onehot[np.arange(n), actions] = 1.0
        dz = weights[:, None] * (onehot - probs)

        if entropy_bonus:
            safe = np.where(probs > 0.0, probs, 1.0)
            log_p = np.where(probs > 0.0, np.log(safe), 0.0)
            entropy = -np.sum(probs * log_p, axis=1, keepdims=True)
            dz = dz + entropy_bonus * (-probs * (log_p + entropy))

        dz = dz / float(n)
        grad_w2 = h.T @ dz
        grad_b2 = dz.sum(axis=0)
        dh = dz @ self.w2.T
        dpre = dh * (1.0 - h**2)
        grad_w1 = x.T @ dpre
        grad_b1 = dpre.sum(axis=0)
        return {"w1": grad_w1, "b1": grad_b1, "w2": grad_w2, "b2": grad_b2}

    # ---- disk -----------------------------------------------------------

    def save(
        self,
        path: str | Path,
        *,
        arm: str,
        reward: str,
        config: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        """Write the weights plus the two fields the quarantine check reads."""
        if arm not in (CONFIRMATORY_ARM, ORACLE_ARM):
            raise ValueError(f"unknown arm {arm!r}")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            **{k: v for k, v in self.parameters.items()},
            "arm": np.array(str(arm)),
            "reward": np.array(str(reward)),
            "dims": np.array(self.dims),
            "n_items": np.array(self.n_items),
            "hidden": np.array(self.hidden),
            "horizon": np.array(self.horizon),
            "seed": np.array(self.seed),
            "config_json": np.array(json.dumps(config or {}, sort_keys=True)),
            "extra_json": np.array(json.dumps(extra or {}, sort_keys=True)),
        }
        np.savez(path, **payload)
        return path

    @classmethod
    def load(cls, path: str | Path) -> tuple["MLPPolicy", dict[str, Any]]:
        """Any saved policy, with its metadata. Not the confirmatory loader."""
        archive = np.load(Path(path), allow_pickle=False)
        policy = cls(
            dims=int(archive["dims"]),
            n_items=int(archive["n_items"]),
            hidden=int(archive["hidden"]),
            horizon=int(archive["horizon"]),
            seed=int(archive["seed"]),
        )
        policy.set_parameters({k: archive[k] for k in ("w1", "b1", "w2", "b2")})
        meta = {
            "arm": str(archive["arm"]),
            "reward": str(archive["reward"]),
            "config": json.loads(str(archive["config_json"])),
            "extra": json.loads(str(archive["extra_json"])),
            "path": str(path),
        }
        return policy, meta


def load_confirmatory(path: str | Path) -> tuple[MLPPolicy, dict[str, Any]]:
    """Load a policy, refusing anything that is not the confirmatory arm.

    Owner RULING 2: the exploratory oracle-reward policy's weights were shaped
    by planted truth and are quarantined from every confirmatory artifact. This
    is the only loader the confirmatory grading path calls, so the quarantine is
    enforced at the one place weights can enter a graded result rather than by
    convention.
    """
    policy, meta = MLPPolicy.load(path)
    if meta["arm"] != CONFIRMATORY_ARM or meta["reward"] != BLUR_REWARD:
        raise ValueError(
            f"refusing to load {path}: it is the {meta['arm']!r} arm trained on "
            f"reward {meta['reward']!r}. Only the {CONFIRMATORY_ARM!r} arm "
            f"trained on {BLUR_REWARD!r} may enter a confirmatory result "
            "(owner RULING 2)."
        )
    return policy, meta
