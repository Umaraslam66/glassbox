"""Truth-vault accessor -- the ONLY reader of the mobility planted truth.

Mobility mirror of the parent's Wall rule (PRD_MOBILITY.md section 6). Everything
the grader compares against planted values -- the traveler parameter vectors
phi, the full traveler cards, the wobble parameters, the designed scenario
loadings and attribute matrices -- lives in one directory under
``mobility/data/``, and this module is the only code in the repository allowed
to read it. ``mobility/tests/test_wall_mobility.py`` enforces that with a
static check on every commit, and the parent's own wall test additionally
forbids the rest of the repository from spelling the directory path at all.

The path is assembled from parts on purpose: writing it as a single literal
anywhere outside the exempt graders fails both wall tests.

Stage M0 scaffold: the accessor API grows in Stage M1 when the first travelers
are minted. Nothing here reads files yet; there is nothing to read.
"""

from __future__ import annotations

from pathlib import Path

_MOBILITY_ROOT = Path(__file__).resolve().parents[2]

#: The mobility planted-truth directory. Grader-side code only.
TRUTH_DIR = _MOBILITY_ROOT / "data" / "truth"


def truth_dir() -> Path:
    """Absolute path of the mobility planted-truth directory."""
    return TRUTH_DIR
