"""Question-picking strategies for the Stage 5 adaptive interview.

Four of them, one interface:

* :class:`RandomOrder` -- a seeded permutation of the candidate items, drawn
  fresh per persona. The zero-information ordering, and the denominator of the
  frozen Gate 5 bar.
* :class:`FixedOrder` -- the Stage 3 scripted order. It has 15 items and no
  more, so an episode running it simply stops at 15; the cap is reported, not
  worked around.
* :class:`InfoGain` -- pick the item whose expected reduction in total
  posterior variance is largest, taking the expectation over the answer
  distribution the current posterior predicts. Pure arithmetic on the posterior
  and the fitted item parameters
  (:data:`src.model.sequential_posterior.INFO_GAIN_DEFINITION`).
* :class:`PolicyStrategy` -- the seam for the RL policy. It wraps any callable
  ``state -> item column per persona``; the policy sees the same
  :class:`~src.model.sequential_posterior.InterviewState` every other strategy
  sees and nothing else.

The interface, and why it is this narrow
----------------------------------------
:meth:`Strategy.select` is handed an
:class:`~src.model.sequential_posterior.InterviewState`: the current point
estimate, its covariance and blur, and a mask of which items each persona has
already been asked. :meth:`Strategy.observe` is then told which item was asked
and which answer came back. That is the complete input.

What a strategy therefore cannot see: the persona's answer distribution, any
answer it did not ask for, the persona's identity, and planted truth. This is
structural, not a convention -- the simulator holds the answers privately and
hands over one cell at a time (``src.interview.episodes``), and
``tests/test_episodes.py`` checks that the public surface of the simulator and
the state object carry nothing else.

RL note (Stage 5, owner RULING 2). A policy trained on this interface can use
the declared blur as its reward, because blur is a system-side quantity. It
cannot use planted truth, and there is no route to it from here.
"""

from __future__ import annotations

from typing import Callable, Protocol, Sequence

import numpy as np

from ..model import mirt
from ..model.sequential_posterior import (
    INFO_GAIN_DEFINITION,
    InterviewState,
    expected_variance_reduction,
)

__all__ = [
    "INFO_GAIN_DEFINITION",
    "FixedOrder",
    "InfoGain",
    "Policy",
    "PolicyStrategy",
    "RandomOrder",
    "Strategy",
    "build",
]


class Strategy(Protocol):
    """What an interviewer has to be able to do."""

    #: Short name used in round tags, artifacts and plots.
    name: str

    def max_steps(self) -> int | None:
        """How many questions this strategy can ever ask, or ``None`` for no cap."""

    def select(self, state: InterviewState) -> np.ndarray:
        """One item column per persona, given only ``state``."""

    def observe(self, columns: np.ndarray, answers: np.ndarray) -> None:
        """Told what was asked and what came back. May be a no-op."""

    def reset(self) -> None:
        """Back to the start of an episode."""


class _Base:
    """Shared bookkeeping: nothing here has any access to answers."""

    name = "base"

    def __init__(self, n_personas: int, n_items: int) -> None:
        self.n_personas = int(n_personas)
        self.n_items = int(n_items)

    def max_steps(self) -> int | None:
        return None

    def observe(self, columns: np.ndarray, answers: np.ndarray) -> None:
        return None

    def reset(self) -> None:
        return None

    def _check(self, columns: np.ndarray, state: InterviewState) -> np.ndarray:
        columns = np.asarray(columns, dtype=np.int64).reshape(-1)
        if columns.shape != (self.n_personas,):
            raise ValueError(
                f"{self.name}: must choose one item per persona "
                f"({self.n_personas}), got {columns.shape}"
            )
        if columns.min() < 0 or columns.max() >= self.n_items:
            raise ValueError(f"{self.name}: chose an item column out of range")
        rows = np.arange(self.n_personas)
        if np.any(state.answered[rows, columns]):
            raise ValueError(f"{self.name}: chose an item that was already asked")
        return columns


# --------------------------------------------------------------------------
# random order
# --------------------------------------------------------------------------


class RandomOrder(_Base):
    """A fresh seeded permutation of the candidate items for every persona.

    Per persona rather than one shared order, because the comparison it is the
    denominator of is per persona: an adaptive interviewer picks a different
    sequence for each person, so the thing it has to beat is a random sequence
    for each person. One shared order would make a whole replicate hostage to
    whether that one order happened to start with strong items.

    The seed is the episode's seed; the same seed reproduces the same orders.
    """

    name = "random"

    def __init__(self, n_personas: int, n_items: int, seed: int) -> None:
        super().__init__(n_personas, n_items)
        self.seed = int(seed)
        self.reset()

    def reset(self) -> None:
        rng = np.random.default_rng(self.seed)
        self._order = np.array(
            [rng.permutation(self.n_items) for _ in range(self.n_personas)]
        )

    def select(self, state: InterviewState) -> np.ndarray:
        if state.step >= self.n_items:
            raise ValueError(f"{self.name}: every item has been asked")
        return self._check(self._order[:, state.step], state)


# --------------------------------------------------------------------------
# the scripted order
# --------------------------------------------------------------------------


class FixedOrder(_Base):
    """The Stage 3 scripted order, the same for everybody, and it ends.

    ``columns`` is the scripted items' positions in the candidate list. The
    episode is capped at ``len(columns)`` -- 15 for the frozen Stage 3 script --
    and the cap is what it is: this strategy has nothing else to ask.
    """

    name = "fixed"

    def __init__(self, n_personas: int, n_items: int, columns: Sequence[int]) -> None:
        super().__init__(n_personas, n_items)
        self.columns = np.asarray(list(columns), dtype=np.int64)
        if self.columns.size == 0:
            raise ValueError("the scripted order is empty")
        if len(set(self.columns.tolist())) != self.columns.size:
            raise ValueError("the scripted order repeats an item")

    def max_steps(self) -> int | None:
        return int(self.columns.size)

    def select(self, state: InterviewState) -> np.ndarray:
        if state.step >= self.columns.size:
            raise ValueError(
                f"{self.name}: the script has {self.columns.size} questions and "
                f"step {state.step} was requested"
            )
        column = int(self.columns[state.step])
        return self._check(np.full(self.n_personas, column), state)


# --------------------------------------------------------------------------
# the information heuristic
# --------------------------------------------------------------------------


class InfoGain(_Base):
    """Ask whichever unasked item is expected to shrink the posterior most.

    The score is :data:`src.model.sequential_posterior.INFO_GAIN_DEFINITION`:
    expected reduction in the trace of the posterior covariance, with the
    expectation taken over the answer distribution the current posterior
    predicts. Ties are broken by the lower column index so a run is
    reproducible.

    This uses no randomness at all, so a replicate's only source of variation is
    the persona's own answer noise -- which is the point: any spread in the
    heuristic's curves is spread in the interview, not in the interviewer.
    """

    name = "heuristic"
    definition = INFO_GAIN_DEFINITION

    def __init__(
        self, n_personas: int, params: mirt.ItemParams, *, candidates: Sequence[int] | None = None
    ) -> None:
        super().__init__(n_personas, params.loadings.shape[0])
        self.params = params
        self.candidates = (
            None if candidates is None else np.asarray(list(candidates), dtype=np.int64)
        )

    def scores(self, state: InterviewState) -> np.ndarray:
        """The raw score per persona per item, before masking. Exposed for tests."""
        return expected_variance_reduction(state.theta, state.cov, self.params)

    def select(self, state: InterviewState) -> np.ndarray:
        score = self.scores(state)
        blocked = np.array(state.answered, dtype=bool)
        if self.candidates is not None:
            allowed = np.zeros(self.n_items, dtype=bool)
            allowed[self.candidates] = True
            blocked = blocked | ~allowed[None, :]
        if blocked.all(axis=1).any():
            raise ValueError(f"{self.name}: a persona has no candidate items left")
        masked = np.where(blocked, -np.inf, score)
        return self._check(np.argmax(masked, axis=1), state)


# --------------------------------------------------------------------------
# the RL seam
# --------------------------------------------------------------------------


class Policy(Protocol):
    """What the Stage 5 RL agent has to provide.

    One call, one array: given the interview state -- point estimate,
    covariance, blur, answered mask -- return the item column to ask each
    persona. Everything the policy needs to learn from is in that state and in
    the answers it is told about afterwards.
    """

    def __call__(self, state: InterviewState) -> np.ndarray: ...


class PolicyStrategy(_Base):
    """Wrap a learned policy so an episode can run it. Not trained here.

    This is the interface the later RL agent plugs into, and the reason it is a
    thin wrapper is that the policy must be substitutable for the heuristic
    without the simulator noticing. Supply ``on_observe`` if the policy wants to
    be told what came back (for a recurrent state, say); it receives only the
    chosen columns and the answers, which is all any strategy gets.
    """

    name = "policy"

    def __init__(
        self,
        n_personas: int,
        n_items: int,
        policy: Policy,
        *,
        name: str = "policy",
        on_observe: Callable[[np.ndarray, np.ndarray], None] | None = None,
        max_questions: int | None = None,
        reset_hook: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(n_personas, n_items)
        self.policy = policy
        self.name = name
        self._on_observe = on_observe
        self._max = max_questions
        self._reset = reset_hook

    def max_steps(self) -> int | None:
        return self._max

    def select(self, state: InterviewState) -> np.ndarray:
        return self._check(self.policy(state), state)

    def observe(self, columns: np.ndarray, answers: np.ndarray) -> None:
        if self._on_observe is not None:
            self._on_observe(np.asarray(columns), np.asarray(answers))

    def reset(self) -> None:
        if self._reset is not None:
            self._reset()


# --------------------------------------------------------------------------
# building one by name
# --------------------------------------------------------------------------


def build(
    name: str,
    *,
    n_personas: int,
    params: mirt.ItemParams,
    seed: int = 0,
    scripted_columns: Sequence[int] | None = None,
) -> Strategy:
    """The three non-RL strategies, by name. The policy is plugged in directly."""
    n_items = int(params.loadings.shape[0])
    if name == "random":
        return RandomOrder(n_personas, n_items, seed)
    if name == "fixed":
        if scripted_columns is None:
            raise ValueError("the fixed strategy needs the scripted order")
        return FixedOrder(n_personas, n_items, scripted_columns)
    if name == "heuristic":
        return InfoGain(n_personas, params)
    raise ValueError(f"unknown strategy {name!r}; known: random, fixed, heuristic")
