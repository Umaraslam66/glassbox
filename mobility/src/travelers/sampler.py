"""Traveler parameter sampler — the frozen population design, as code.

The design constants here (dimension order, correlation matrix, clip bound,
VOT anchor) are the public design published in PREREGISTRATION_MOBILITY.md
section 2, frozen at commit 57bfcfb. The sampled VALUES are planted truth:
only the grader-side mint (``mobility/src/eval/``) may write them to disk,
and only the grader may ever read them back. This module is pure — it holds
no paths and touches no files.
"""

from __future__ import annotations

import numpy as np

#: Dimension order, frozen (PREREGISTRATION_MOBILITY.md section 2).
DIMS = ("VOT", "SCH", "MOD", "CRW", "PRC", "HAB")

#: Planted correlation matrix, frozen. VOT-PRC and CRW-PRC are designed zeros.
PLANTED_CORR = np.array([
    [1.00, 0.25, 0.20, 0.30, 0.00, -0.10],
    [0.25, 1.00, 0.10, 0.10, -0.15, 0.35],
    [0.20, 0.10, 1.00, 0.25, -0.20, 0.30],
    [0.30, 0.10, 0.25, 1.00, 0.00, 0.10],
    [0.00, -0.15, -0.20, 0.00, 1.00, 0.05],
    [-0.10, 0.35, 0.30, 0.10, 0.05, 1.00],
])

DESIGNED_ZERO_PAIRS = (("VOT", "PRC"), ("CRW", "PRC"))

#: Latent values are clipped to [-CLIP, +CLIP] (~4.6% of draws per dimension).
CLIP = 2.0

#: VOT structural anchor: VOT_i = 12 * exp(0.55 * phi_VOT) euros per hour.
VOT_MEDIAN_EUR_H = 12.0
VOT_SIGMA = 0.55


def sample_phi(seed: int, n: int = 400) -> tuple[list[str], np.ndarray]:
    """Mint ``n`` traveler parameter vectors from the frozen design.

    Deterministic in ``seed``. Returns (traveler ids, n x 6 array in DIMS
    order). Values are multivariate normal with PLANTED_CORR, unit variances,
    zero means, clipped to [-CLIP, CLIP].
    """
    rng = np.random.default_rng(seed)
    chol = np.linalg.cholesky(PLANTED_CORR)
    z = rng.standard_normal((n, len(DIMS)))
    phi = np.clip(z @ chol.T, -CLIP, CLIP)
    ids = [f"t{i:04d}" for i in range(n)]
    return ids, phi


def vot_eur_per_hour(phi_vot: np.ndarray) -> np.ndarray:
    """Map latent VOT to euros per hour per the frozen anchor."""
    return VOT_MEDIAN_EUR_H * np.exp(VOT_SIGMA * np.asarray(phi_vot))
