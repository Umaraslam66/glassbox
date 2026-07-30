"""Truth store accessor.

THE WALL RULE: this is the only module in the repository allowed to read the
planted-truth directory. Nothing outside ``src/eval/`` may import this module,
and nothing outside ``src/eval/`` may reference the truth directory path in any
form. ``tests/test_wall.py`` enforces both rules statically over every Python
file in the repo, so a violation fails the test suite.

What lives behind the Wall:
  * theta      -- the planted trait vector for a persona
  * card       -- the full persona biography written from theta
  * noise      -- the persona's planted response-inconsistency parameters

The interviewer, item encoder, person encoder and predictor must never see any
of it. They get the public profile, the question text, and the transcript.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_TRUTH_DIRNAME = "truth"

_SUBDIRS = {
    "theta": "theta",
    "card": "cards",
    "noise": "noise",
}


def repo_root() -> Path:
    """Repository root, i.e. the parent of ``src/``."""
    return Path(__file__).resolve().parents[2]


def truth_dir() -> Path:
    """Directory holding the planted truth.

    Override with the ``GLASSBOX_TRUTH_DIR`` environment variable (used by
    tests and by runs that keep their truth store outside the repo).
    """
    override = os.environ.get("GLASSBOX_TRUTH_DIR")
    if override:
        return Path(override)
    return repo_root() / "data" / _TRUTH_DIRNAME


def _read_json(kind: str, persona_id: str) -> Any:
    path = truth_dir() / _SUBDIRS[kind] / f"{persona_id}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"no {kind} on file for persona {persona_id!r} (looked in "
            f"{path.parent}) -- has the persona factory run yet?"
        )
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def load_theta(persona_id: str) -> dict[str, float]:
    """Return the planted trait vector for one persona, as {dimension: value}."""
    return _read_json("theta", persona_id)


def load_card(persona_id: str) -> dict[str, Any]:
    """Return the full persona card (biography plus the public-profile slice)."""
    return _read_json("card", persona_id)


def load_noise(persona_id: str) -> dict[str, Any]:
    """Return the planted response-noise parameters for one persona."""
    return _read_json("noise", persona_id)


def list_persona_ids() -> list[str]:
    """Persona ids that have a planted trait vector on file, sorted."""
    theta_dir = truth_dir() / _SUBDIRS["theta"]
    if not theta_dir.is_dir():
        return []
    return sorted(p.stem for p in theta_dir.glob("*.json"))
