"""The confirmatory reward, and the shape any other reward has to fit into.

Owner RULING 2, recorded before any Stage 5 compute:

    The confirmatory RL policy trains on blur-based reward (declared
    posterior-variance reduction minus a per-question cost). Truth-derived
    rewards are FORBIDDEN for the confirmatory policy -- planted truth must
    never shape system-side weights, even via reward on the training batch.

So the confirmatory reward for asking a question is

    r_t = trace(Sigma_before) - trace(Sigma_after) - question_cost

where ``Sigma`` is the system's own Laplace posterior covariance -- the same
declared uncertainty the info-gain heuristic maximises the expected reduction
of, and the same one the Stage 3 backbone reports as blur. Every term is a
number the system computed about itself. There is no planted value anywhere in
this module, and :mod:`src.rl` imports nothing from :mod:`src.eval`, which
``tests/test_rl.py`` checks by parsing the source.

What the per-question cost does, and does not, do
-------------------------------------------------
It is in the reward because the PRD defines the reward as "blur reduction minus
question cost", and it is worth being plain about its effect at a **fixed**
horizon: every episode asks exactly ``N`` questions, so the total cost is
``N * question_cost`` for every possible sequence of actions. A constant added
to every trajectory's return shifts the reward curve and cancels out of the
policy gradient exactly. It is reported, it makes the reward comparable with the
PRD's definition, and it changes no weight. It would start to bite the moment
the policy is allowed to choose when to stop, which is not this experiment.

The interface
-------------
A reward is a callable taking one :class:`RewardContext` and returning one
number per persona. The trainer only ever calls the callable it was handed. That
is what makes the exploratory oracle arm cheap -- it injects a different
callable from ``src/eval/`` and reuses the whole trainer -- and it is also what
makes the confirmatory arm safe, because the confirmatory callable lives here,
in a module that cannot reach truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

__all__ = [
    "REWARD_DEFINITION",
    "RewardContext",
    "RewardFn",
    "blur_reward",
    "make_blur_reward",
    "posterior_trace",
]

#: Quoted verbatim into the training artifact.
REWARD_DEFINITION = (
    "r_t = trace(Sigma_{t-1}) - trace(Sigma_t) - question_cost, where Sigma_t "
    "is the system's own Laplace posterior covariance over the eight latent "
    "dimensions after t answers. This is the realised reduction in TOTAL "
    "declared posterior variance from the question that was actually asked -- "
    "the same A-optimality quantity the info-gain heuristic scores in "
    "expectation, measured after the fact. Every term is system-side: the "
    "posterior the system declares about itself and nothing else. At a fixed "
    "horizon the cost term is the same constant for every action sequence, so "
    "it shifts the reported return and cancels out of the gradient."
)


def posterior_trace(cov: np.ndarray) -> np.ndarray:
    """``trace(Sigma)`` per persona -- the total declared variance."""
    return np.trace(np.asarray(cov, dtype=float), axis1=1, axis2=2)


@dataclass(frozen=True)
class RewardContext:
    """Everything a reward callable is handed after one question is answered.

    The system-side fields (``cov_before``, ``cov_after``, ``theta_before``,
    ``theta_after``, ``columns``, ``step``) are what the confirmatory reward
    uses. ``persona_rows`` is the row each trajectory occupies in the training
    batch's persona list; it exists so the exploratory oracle arm can line its
    own truth table up with the batch, and the confirmatory reward ignores it.
    """

    step: int
    columns: np.ndarray
    persona_rows: np.ndarray
    theta_before: np.ndarray
    theta_after: np.ndarray
    cov_before: np.ndarray
    cov_after: np.ndarray
    question_cost: float


RewardFn = Callable[[RewardContext], np.ndarray]


def blur_reward(ctx: RewardContext) -> np.ndarray:
    """The confirmatory reward. See :data:`REWARD_DEFINITION`."""
    return (
        posterior_trace(ctx.cov_before)
        - posterior_trace(ctx.cov_after)
        - float(ctx.question_cost)
    )


def make_blur_reward() -> RewardFn:
    """The confirmatory reward as a callable, for symmetry with the oracle arm."""
    return blur_reward
