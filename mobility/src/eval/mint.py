"""Grader-side mint: deposit the traveler population in the vault, grade it.

The only writer of mobility planted truth (Wall rule: this directory is the
only code allowed near it). Samples the population with the pure sampler,
writes the values into the vault, and grades the realized draw against the
mint acceptance tolerances frozen in PREREGISTRATION_MOBILITY.md section 2 —
before any card is rendered or any scenario answered. Only aggregates leave
the vault: the QA file under ``mobility/results/`` carries population
statistics, never a single traveler's parameters.

Run: ``python -m mobility.src.eval.mint mobility/experiments/m1_mint.json``
On tolerance failure: remint with the next sequential seed, log every attempt.
"""

from __future__ import annotations

import datetime as _dt
import itertools
import json
import sys
from pathlib import Path

import numpy as np

from mobility.src.eval.truth_vault import truth_dir
from mobility.src.travelers.sampler import (
    CLIP,
    DESIGNED_ZERO_PAIRS,
    DIMS,
    PLANTED_CORR,
    sample_phi,
    vot_eur_per_hour,
)

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"

#: Frozen mint acceptance tolerances (PREREGISTRATION_MOBILITY.md section 2).
TOL = {
    "max_abs_corr_dev": 0.10,
    "designed_zero_abs_max": 0.10,
    "mean_abs_max": 0.10,
    "sd_range": (0.90, 1.02),
    "clip_share_max": 0.07,
    "vot_median_range_eur_h": (10.8, 13.2),
}


def population_qa(phi: np.ndarray) -> dict:
    """Aggregate QA of a realized draw against the frozen design. No per-traveler values."""
    realized = np.corrcoef(phi, rowvar=False)
    dev = realized - PLANTED_CORR
    off = ~np.eye(len(DIMS), dtype=bool)

    zeros = {
        f"{a}-{b}": round(float(realized[DIMS.index(a), DIMS.index(b)]), 4)
        for a, b in DESIGNED_ZERO_PAIRS
    }
    means = phi.mean(axis=0)
    sds = phi.std(axis=0, ddof=1)
    clip_share = ((np.abs(phi) >= CLIP - 1e-12).mean(axis=0))
    vot = vot_eur_per_hour(phi[:, DIMS.index("VOT")])

    # phi-level diversity analogs (choice-based diversity bars come with answers)
    eig = np.linalg.eigvalsh(realized)
    participation_ratio = float(eig.sum() ** 2 / (eig ** 2).sum())
    diff = phi[:, None, :] - phi[None, :, :]
    pair = np.sqrt((diff ** 2).sum(axis=2))
    dists = pair[np.triu_indices(len(phi), k=1)]

    checks = {
        "corr_within_tolerance": bool(np.abs(dev[off]).max() <= TOL["max_abs_corr_dev"]),
        "designed_zeros_ok": bool(all(abs(v) <= TOL["designed_zero_abs_max"] for v in zeros.values())),
        "means_ok": bool(np.abs(means).max() <= TOL["mean_abs_max"]),
        "sds_ok": bool(all(TOL["sd_range"][0] <= s <= TOL["sd_range"][1] for s in sds)),
        "clip_ok": bool(clip_share.max() <= TOL["clip_share_max"]),
        "vot_median_ok": bool(
            TOL["vot_median_range_eur_h"][0] <= float(np.median(vot)) <= TOL["vot_median_range_eur_h"][1]
        ),
        "no_duplicates": bool(dists.min() > 0),
    }
    return {
        "dims": list(DIMS),
        "realized_corr": [[round(float(v), 4) for v in row] for row in realized],
        "planted_corr": [[float(v) for v in row] for row in PLANTED_CORR],
        "max_abs_corr_dev": round(float(np.abs(dev[off]).max()), 4),
        "designed_zeros_realized": zeros,
        "marginals": {
            d: {
                "mean": round(float(means[i]), 4),
                "sd": round(float(sds[i]), 4),
                "clip_share": round(float(clip_share[i]), 4),
            }
            for i, d in enumerate(DIMS)
        },
        "vot_eur_per_hour": {
            "median": round(float(np.median(vot)), 2),
            "p5": round(float(np.percentile(vot, 5)), 2),
            "p95": round(float(np.percentile(vot, 95)), 2),
            "min": round(float(vot.min()), 2),
            "max": round(float(vot.max()), 2),
            "anchor": "12*exp(0.55*phi), median target 12, construction range [4.0, 36.1]",
        },
        "diversity_phi_level": {
            "participation_ratio_realized_corr": round(participation_ratio, 3),
            "pairwise_distance_min": round(float(dists.min()), 4),
            "pairwise_distance_median": round(float(np.median(dists)), 3),
            "note": "choice-based diversity bars (pairwise agreement, choice-matrix "
                    "participation ratio) require answers and are graded at Gate M1 proper",
        },
        "tolerances": {k: (list(v) if isinstance(v, tuple) else v) for k, v in TOL.items()},
        "checks": checks,
        "all_checks_pass": bool(all(checks.values())),
    }


def mint(config_path: Path) -> dict:
    config = json.loads(config_path.read_text())
    seed, n = int(config["seed"]), int(config["n"])
    ids, phi = sample_phi(seed, n)

    qa = population_qa(phi)
    qa["experiment"] = config.get("experiment", "m1_mint")
    qa["seed"] = seed
    qa["n"] = n
    qa["date"] = _dt.date.today().isoformat()
    qa["prereg_commit"] = config.get("prereg_commit")

    if qa["all_checks_pass"]:
        vault = truth_dir()
        vault.mkdir(parents=True, exist_ok=True)
        np.savez(
            vault / "phi_m1.npz",
            ids=np.array(ids), phi=phi, seed=seed,
            dims=np.array(DIMS), prereg_commit=str(config.get("prereg_commit")),
        )
        qa["deposited"] = True
    else:
        qa["deposited"] = False  # remint with the next sequential seed; log the attempt

    out = RESULTS_DIR / f"{qa['experiment']}_qa.json"
    out.write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")
    return qa


if __name__ == "__main__":
    result = mint(Path(sys.argv[1]))
    print(json.dumps({k: result[k] for k in ("experiment", "seed", "n", "all_checks_pass",
                                             "deposited", "max_abs_corr_dev")}, indent=2))
    sys.exit(0 if result["all_checks_pass"] else 1)
