"""The phi sampler: deterministic, frozen-design-shaped, in bounds.

Pure-sampler tests only — nothing here goes near the vault. The grader-side
mint's tolerance checks are exercised through its own QA output at run time.
"""

from __future__ import annotations

import numpy as np

from mobility.src.travelers.sampler import (
    CLIP,
    DIMS,
    PLANTED_CORR,
    sample_phi,
    vot_eur_per_hour,
)


def test_design_constants_match_the_frozen_prereg() -> None:
    assert DIMS == ("VOT", "SCH", "MOD", "CRW", "PRC", "HAB")
    assert PLANTED_CORR.shape == (6, 6)
    assert np.allclose(PLANTED_CORR, PLANTED_CORR.T)
    assert np.linalg.eigvalsh(PLANTED_CORR).min() > 0
    # designed zeros
    assert PLANTED_CORR[DIMS.index("VOT"), DIMS.index("PRC")] == 0.0
    assert PLANTED_CORR[DIMS.index("CRW"), DIMS.index("PRC")] == 0.0
    assert np.abs(PLANTED_CORR - np.eye(6)).max() <= 0.35


def test_sampler_is_deterministic_in_the_seed() -> None:
    ids_a, phi_a = sample_phi(3100, 400)
    ids_b, phi_b = sample_phi(3100, 400)
    assert ids_a == ids_b
    assert np.array_equal(phi_a, phi_b)
    _, phi_c = sample_phi(3101, 400)
    assert not np.array_equal(phi_a, phi_c)


def test_sample_shape_bounds_and_anchor() -> None:
    ids, phi = sample_phi(3100, 400)
    assert len(ids) == 400 and len(set(ids)) == 400
    assert phi.shape == (400, 6)
    assert np.abs(phi).max() <= CLIP
    vot = vot_eur_per_hour(phi[:, 0])
    assert vot.min() >= 12 * np.exp(-0.55 * CLIP) - 1e-9
    assert vot.max() <= 12 * np.exp(0.55 * CLIP) + 1e-9


def test_large_sample_recovers_the_planted_matrix() -> None:
    """Sanity on the generator itself, not on any minted population."""
    _, phi = sample_phi(9, 200_000)
    realized = np.corrcoef(phi, rowvar=False)
    assert np.abs(realized - PLANTED_CORR).max() < 0.02
