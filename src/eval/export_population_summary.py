"""Export the aggregate population summary that the dashboard reads.

    python -m src.eval.export_population_summary \\
        --truth <planted dir> --public <public dir> \\
        --out results/population_summary.json

Grader-side code: this lives in ``src/eval/``, so it is allowed to read the
planted truth. Boiling that truth down to counts is the entire point of the
module. The dashboard sits outside the Wall and must never look behind it, so
this CLI is the only thing that ever turns planted values into something the
dashboard can open.

What comes out is aggregate and nothing else:

  * one histogram per trait dimension -- 20 bins over [-2, 2], counts only;
  * one histogram of the planted wobble (response inconsistency), counts only;
  * how many personas exist;
  * demographic counts from the *public* profiles (occupation truncated to the
    top 20, plus city size, household and region type in full).

What never comes out: a persona id next to a number, an individual trait value,
an individual wobble, or anything that would let a reader recover one person's
planted truth from the file. Bin counts are the finest grain in here.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..bank.schema import DIMENSION_LABELS, DIMENSIONS
from . import truth_store
from .population_qa import truth_dir

#: Format marker, so a stale summary can be spotted rather than mis-read.
SCHEMA = "glassbox.population_summary/1"

#: Trait histograms are always cut the same way, whatever the batch looks like.
THETA_BINS = 20
THETA_RANGE = (-2.0, 2.0)

#: The wobble histogram is cut over the batch's own span (see ``_span``).
WOBBLE_BINS = 20

#: Public profile fields that get counted, and how many rows each may show.
#: ``None`` means "show every distinct value".
DEMOGRAPHIC_FIELDS: dict[str, int | None] = {
    "occupation": 20,
    "city_size": None,
    "household": None,
    "region_type": None,
}


# --------------------------------------------------------------------------
# counting
# --------------------------------------------------------------------------


def _histogram(values: Sequence[float], bins: int, low: float, high: float) -> dict[str, Any]:
    """Counts per bin plus the edges. No values, no ids -- counts only."""
    counts, edges = np.histogram(np.asarray(values, dtype=float), bins=bins, range=(low, high))
    return {
        "bin_edges": [round(float(edge), 6) for edge in edges],
        "counts": [int(count) for count in counts],
        "n": int(len(values)),
    }


def _span(values: Sequence[float]) -> tuple[float, float]:
    """A plottable low/high for a batch, widened if every value is identical."""
    if not values:
        return (0.0, 1.0)
    low, high = float(min(values)), float(max(values))
    if high - low < 1e-9:
        return (low - 0.05, high + 0.05)
    return (low, high)


def theta_histograms(thetas: Iterable[Mapping[str, float]]) -> dict[str, dict[str, Any]]:
    """One counts-only histogram per dimension, on the frozen [-2, 2] grid."""
    columns: dict[str, list[float]] = {dimension: [] for dimension in DIMENSIONS}
    for theta in thetas:
        for dimension in DIMENSIONS:
            if dimension in theta:
                columns[dimension].append(float(theta[dimension]))
    return {
        dimension: _histogram(values, THETA_BINS, *THETA_RANGE)
        for dimension, values in columns.items()
    }


def wobble_histogram(wobbles: Sequence[float]) -> dict[str, Any]:
    """Counts-only histogram of the planted response inconsistency."""
    low, high = _span(wobbles)
    return _histogram(wobbles, WOBBLE_BINS, low, high)


def demographic_counts(profiles: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Value counts per public demographic field, most common first."""
    fields: dict[str, Any] = {}
    for field, limit in DEMOGRAPHIC_FIELDS.items():
        tally = Counter(
            str(profile[field]) for profile in profiles if profile.get(field) is not None
        )
        ordered = tally.most_common()
        shown = ordered if limit is None else ordered[:limit]
        fields[field] = {
            "n_distinct": len(ordered),
            "n_shown": len(shown),
            "truncated": len(shown) < len(ordered),
            "counts": [{"value": value, "count": int(count)} for value, count in shown],
        }
    return {"n_profiles": len(profiles), "fields": fields}


# --------------------------------------------------------------------------
# reading the inputs
# --------------------------------------------------------------------------


def read_profiles(public: str | Path) -> list[dict[str, Any]]:
    """Every public profile under ``<public>/profiles`` (or ``<public>`` itself)."""
    root = Path(public)
    folder = root / "profiles" if (root / "profiles").is_dir() else root
    if not folder.is_dir():
        return []
    profiles: list[dict[str, Any]] = []
    for path in sorted(folder.glob("*.json")):
        profiles.append(json.loads(path.read_text(encoding="utf-8")))
    return profiles


def collect(planted: str | Path) -> tuple[list[dict[str, float]], list[float], int]:
    """Planted trait vectors and wobbles for every persona on file.

    Returns the vectors, the wobbles and the persona count. Personas are read
    through the store accessor, which is the only door onto the planted files.
    Nothing here keeps the ids: the two lists come back detached from who they
    came from, in the order the store lists them, which is what makes
    everything downstream aggregate by construction.
    """
    thetas: list[dict[str, float]] = []
    wobbles: list[float] = []

    with truth_dir(planted):
        pids = truth_store.list_persona_ids()
        for pid in pids:
            record = truth_store.load_theta(pid)
            theta = record.get("theta", record) if isinstance(record, Mapping) else record
            thetas.append({str(k): float(v) for k, v in dict(theta).items()})
            try:
                noise = truth_store.load_noise(pid)
            except FileNotFoundError:
                continue
            if isinstance(noise, Mapping) and noise.get("wobble") is not None:
                wobbles.append(float(noise["wobble"]))

    return thetas, wobbles, len(pids)


def build_summary(
    thetas: Sequence[Mapping[str, float]],
    wobbles: Sequence[float],
    n_personas: int,
    profiles: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """The whole summary from already-loaded pieces."""
    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_personas": int(n_personas),
        "n_with_wobble": len(wobbles),
        "dimensions": list(DIMENSIONS),
        "dimension_labels": dict(DIMENSION_LABELS),
        "theta": {
            "bins": THETA_BINS,
            "range": [THETA_RANGE[0], THETA_RANGE[1]],
            "histograms": theta_histograms(thetas),
        },
        "wobble": {"bins": WOBBLE_BINS, "histogram": wobble_histogram(list(wobbles))},
        "demographics": demographic_counts(list(profiles)),
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.eval.export_population_summary",
        description="Export aggregate-only population stats for the dashboard.",
    )
    parser.add_argument("--truth", type=Path, required=True,
                        help="the batch's planted directory: theta/ and noise/")
    parser.add_argument("--public", type=Path, required=True,
                        help="the public directory holding profiles/")
    parser.add_argument("--out", type=Path, required=True,
                        help="where to write the summary JSON")
    args = parser.parse_args(argv)

    thetas, wobbles, n_personas = collect(args.truth)
    if n_personas == 0:
        parser.error(f"no personas on file under {args.truth}")

    profiles = read_profiles(args.public)
    summary = build_summary(thetas, wobbles, n_personas, profiles)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    demographics = summary["demographics"]
    print(f"personas {n_personas}   wobbles {len(wobbles)}   "
          f"public profiles {demographics['n_profiles']}")
    for dimension in DIMENSIONS:
        counts = summary["theta"]["histograms"][dimension]["counts"]
        print(f"  {dimension}  {sum(counts)} values across {len(counts)} bins")
    for field, block in demographics["fields"].items():
        note = f", showing top {block['n_shown']}" if block["truncated"] else ""
        print(f"  {field}: {block['n_distinct']} distinct values{note}")
    print("")
    print(f"summary -> {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
