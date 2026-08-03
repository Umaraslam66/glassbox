"""Mobility noise layer — the parent's mechanism, ported (frozen prereg §8).

Seeded mechanical noise driven by the traveler's recorded answer-token
distribution: on scenarios where the traveler's conviction is low (small gap
between the top two answer probabilities), the answer sometimes flips to the
second choice. Each traveler carries a planted wobble parameter (independent
of φ) scaling their flip propensity. The flip rule:

    p_flip = min(0.5, wobble * a * exp(-b * gap))

with (a, b) tuned at Stage M1 so pooled test-retest agreement lands inside
the frozen 70–90% band, then frozen at Gate M1 exactly as the parent froze
its layer. Deterministic: the flip draw is seeded per (traveler, scenario,
round), so any round can be regenerated bit-identically.

Pure functions only — no paths, no file reads. The vault-side QA driver
feeds it recorded answers and distributions and stores what comes out; the
system side only ever sees the noised output.
"""

from __future__ import annotations

import hashlib

import numpy as np


def mint_wobble(seed: int, ids: list[str], lo: float = 0.4, hi: float = 1.6) -> dict[str, float]:
    """Per-traveler wobble, planted independently of φ. Range is a tuning knob
    recorded in the experiment config (parent convention)."""
    rng = np.random.default_rng(seed)
    values = rng.uniform(lo, hi, size=len(ids))
    return {pid: float(v) for pid, v in zip(ids, values)}


def _u01(base_seed: int, pid: str, sid: str, round_no: int) -> float:
    """Deterministic uniform draw in [0, 1) for one (traveler, scenario, round)."""
    material = f"{base_seed}:{pid}:{sid}:{round_no}".encode()
    digest = hashlib.sha256(material).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def conviction_gap(dist_top: list | None, valid: set[str]) -> tuple[float, str | None]:
    """(top-two probability gap, second-choice letter) from a recorded
    distribution, restricted to valid option letters. No distribution or no
    valid runner-up -> gap 1.0 (never flips)."""
    if not dist_top:
        return 1.0, None
    ranked = [(str(tok).strip().upper(), p) for tok, p in dist_top
              if str(tok).strip().upper() in valid]
    if len(ranked) < 2:
        return 1.0, None
    (t1, p1), (t2, p2) = ranked[0], ranked[1]
    total = sum(p for _, p in ranked)
    if total <= 0:
        return 1.0, None
    return max(0.0, (p1 - p2) / total), t2


def flip_prob(wobble: float, gap: float, a: float, b: float) -> float:
    return float(min(0.5, wobble * a * np.exp(-b * gap)))


def noised_answer(answer: str, dist_top: list | None, valid: set[str],
                  wobble: float, a: float, b: float,
                  base_seed: int, pid: str, sid: str, round_no: int) -> str:
    """One seeded application of the layer to one recorded answer."""
    gap, second = conviction_gap(dist_top, valid)
    if second is None or second == answer:
        # runner-up identical to the recorded answer: flip target is the top
        # token instead (the recorded sample already landed on the runner-up)
        ranked = [(str(t).strip().upper(), p) for t, p in (dist_top or [])
                  if str(t).strip().upper() in valid]
        alternatives = [t for t, _ in ranked if t != answer]
        if not alternatives:
            return answer
        second = alternatives[0]
    if _u01(base_seed, pid, sid, round_no) < flip_prob(wobble, gap, a, b):
        return second
    return answer
