"""M4 world mint + design-time capacity calibration (memo a; zero API).

Mints the M4 population (300 of the 400 travelers, seed 3250) and their
preferred arrival times (PAT ~ N(08:30, 12 min), seed 3260, clipped to
[07:30, 09:30]) into the vault, then calibrates the bottleneck capacity
from planted parameters only, BEFORE any agent runs: the day-1 naive
profile (everyone on route A, departing PAT - 18) must produce a maximum
queue delay inside [12, 18] minutes, target 15 (memo a). The calibration
curve and chosen capacity go to results/m4_capacity_calibration.json.

Run: ``python -m mobility.src.eval.m4_world``
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import numpy as np

from mobility.src.engine.bottleneck import FREE_FLOW_A, vickrey_travel_times
from mobility.src.eval.truth_vault import truth_dir

MOBILITY_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = MOBILITY_ROOT / "results"

DRAW_SEED, PAT_SEED = 3250, 3260
PAT_MEAN, PAT_SD = 8 * 60 + 30, 12.0
PAT_CLIP = (7 * 60 + 30, 9 * 60 + 30)
TARGET_DELAY, DELAY_BAND = 15.0, (12.0, 18.0)


def main() -> dict:
    ids = [f"t{i:04d}" for i in range(400)]
    world_path = truth_dir() / "m4_world.npz"
    if world_path.exists():
        stash = np.load(world_path, allow_pickle=False)
        pids = [str(x) for x in stash["pids"]]
        pat = stash["pat"]
    else:
        rng = np.random.default_rng(DRAW_SEED)
        # draw order preserved on purpose: the pilot is pre-declared as the
        # first 50 positions of this seeded draw
        pids = rng.choice(ids, size=300, replace=False).tolist()
        pat = np.clip(np.random.default_rng(PAT_SEED).normal(
            PAT_MEAN, PAT_SD, size=300), *PAT_CLIP)
        np.savez(world_path, pids=np.array(pids), pat=pat,
                 draw_seed=DRAW_SEED, pat_seed=PAT_SEED)

    naive_deps = pat - FREE_FLOW_A
    curve = []
    for s in np.arange(2.0, 10.01, 0.25):
        travel = vickrey_travel_times(naive_deps, float(s))
        max_delay = float((travel - FREE_FLOW_A).max())
        curve.append({"capacity_per_min": round(float(s), 2),
                      "day1_max_delay_min": round(max_delay, 2)})
    in_band = [c for c in curve
               if DELAY_BAND[0] <= c["day1_max_delay_min"] <= DELAY_BAND[1]]
    chosen = min(in_band, key=lambda c: abs(c["day1_max_delay_min"] - TARGET_DELAY))

    record = {
        "experiment": "m4_capacity_calibration",
        "date": _dt.date.today().isoformat(),
        "rule": "memo (a): capacity chosen at design time, from planted "
                "parameters only, before any agent runs, so the day-1 naive "
                "profile (route A, depart PAT - 18) yields max queue delay "
                "in [12, 18] min, target 15",
        "seeds": {"traveler_draw": DRAW_SEED, "pat": PAT_SEED},
        "population": {"n": len(pids),
                       "pat_mean_clock": "08:30", "pat_sd_min": PAT_SD,
                       "pat_clip": ["07:30", "09:30"]},
        "chosen_capacity_per_min": chosen["capacity_per_min"],
        "achieved_day1_max_delay_min": chosen["day1_max_delay_min"],
        "band": list(DELAY_BAND),
        "curve": curve,
    }
    (RESULTS_DIR / "m4_capacity_calibration.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


if __name__ == "__main__":
    out = main()
    print(json.dumps({k: out[k] for k in
                      ("chosen_capacity_per_min",
                       "achieved_day1_max_delay_min", "band")}, indent=2))
