"""Reinforcement learning: interviewer reward computation and policy training.

Where the seam is
-----------------
The Stage 5 interviewing machinery is already built and does not change when a
policy arrives. A policy is a callable

    state -> one item column per persona

where ``state`` is :class:`src.model.sequential_posterior.InterviewState`: the
current point estimate, its covariance, its blur, and a mask of which items each
persona has already been asked. Wrap it in
:class:`src.interview.strategies.PolicyStrategy` and it runs in the same
:class:`src.interview.episodes.EpisodeSimulator` as random, fixed and the
info-gain heuristic, and is graded by the same :mod:`src.eval.gate5` code.

Reward (owner RULING 2, PROJECT_LOG entry "Stage 5 opened")
----------------------------------------------------------
The confirmatory policy's reward is system-side only: declared
posterior-variance reduction minus a per-question cost. Planted truth must never
shape a confirmatory policy's weights, not even through reward on the training
batch -- and there is no route to it from the interview state, by construction.
An optional exploratory truth-reward policy is allowed, but its training code
has to live under ``src/eval/`` (the Wall test forbids truth imports anywhere
else) and its weights stay out of every confirmatory artifact.

Training personas: the freshly minted RL batch, never the original held-out
personas, which stay clean for the confirmatory evaluation.
"""
