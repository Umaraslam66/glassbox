"""Gate M4 grader — frozen bars on the completed day-loop (zero API).

Runs only after day 20 exists (Ruling 33b: no grading mid-run). Bars,
verbatim from the frozen prereg §6:
- Peak spreading: S20 <= 0.80 * S1, S_t = the day's maximum share of
  arrivals in any sliding 15-minute window (§5).
- Behavioral coherence: Spearman r(W_i, HAB_i) <= -0.3 and
  Spearman r(X_i, SCH_i) <= -0.3, with W and X exactly as §5/Ruling 8.
Ruling 30's pre-declaration applies: beside each coherence result the
decomposition divides the miss into card transmission vs dynamics via the
measured M1 transmission (SCH 0.4861, HAB 0.5845).

Run: ``python -m mobility.src.eval.m4_bars m4_full``
"""

from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np

from mobility.src.engine.bottleneck import (lateness_exposure, peak_15min_share,
                                            switching_frequency,
                                            tau_interpolator)
from mobility.src.eval.truth_vault import truth_dir
from mobility.src.travelers.sampler import DIMS

MOBILITY_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = MOBILITY_ROOT / "results"
RHO_HAT = {"SCH": 0.4861, "HAB": 0.5845}


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    def rank(v):
        order = np.argsort(v, kind="stable")
        ranks = np.empty(len(v))
        ranks[order] = np.arange(len(v), dtype=float)
        for val in np.unique(v):
            m = v == val
            ranks[m] = ranks[m].mean()
        return ranks
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def main(run_name: str) -> dict:
    rows = [json.loads(l) for l in
            (MOBILITY_ROOT / "data" / "runs" / run_name /
             "trajectories.jsonl").read_text().splitlines()]
    days = sorted({r["day"] for r in rows})
    by_pid: dict[str, dict[int, dict]] = {}
    for r in rows:
        by_pid.setdefault(r["pid"], {})[r["day"]] = r
    pids = sorted(by_pid)
    assert len(days) == 20, "grading requires the completed 20-day run"

    world = np.load(truth_dir() / "m4_world.npz", allow_pickle=False)
    pat = {str(p): float(v) for p, v in zip(world["pids"], world["pat"])}
    stash = np.load(truth_dir() / "phi_m1.npz", allow_pickle=False)
    ids = [str(x) for x in stash["ids"]]
    phi = {p: stash["phi"][ids.index(p)] for p in pids}
    i_sch, i_hab = DIMS.index("SCH"), DIMS.index("HAB")

    # ---- peak spreading ---------------------------------------------------
    S = []
    day_context = []
    for day in days:
        d = [r for r in rows if r["day"] == day]
        arrivals = np.array([r["arrive_min"] for r in d])
        on_a = [r for r in d if r["route"] == "A"]
        S.append(round(peak_15min_share(arrivals), 4))
        day_context.append({
            "day": day, "peak_share": S[-1],
            "route_A_share": round(len(on_a) / len(d), 3),
            "max_queue_delay_min": round(max(r["travel_min"] for r in on_a) - 18, 2)
            if on_a else 0.0,
            "mean_travel_min": round(float(np.mean([r["travel_min"] for r in d])), 2),
        })
    peak_pass = S[-1] <= 0.80 * S[0]

    # ---- frozen indices ---------------------------------------------------
    day_profiles = []
    for day in days:
        d = [r for r in rows if r["day"] == day]
        a = [(r["dep_min"], r["travel_min"]) for r in d if r["route"] == "A"]
        b = [(r["dep_min"], r["travel_min"]) for r in d if r["route"] == "B"]
        day_profiles.append({
            "A": tau_interpolator(np.array([x for x, _ in a]),
                                  np.array([y for _, y in a])),
            "B": tau_interpolator(np.array([x for x, _ in b]) if b else np.array([0.0]),
                                  np.array([y for _, y in b]) if b else np.array([26.0])),
        })
    W, X, hab, sch = [], [], [], []
    for p in pids:
        routes = [by_pid[p][t]["route"] for t in days]
        deps = [by_pid[p][t]["dep_min"] for t in days]
        W.append(switching_frequency(routes))
        X.append(lateness_exposure(deps, routes, day_profiles, pat[p]))
        hab.append(phi[p][i_hab])
        sch.append(phi[p][i_sch])
    W, X, hab, sch = map(np.array, (W, X, hab, sch))
    r_w = spearman(W, hab)
    r_x = spearman(X, sch)

    def quartile_summary(index: np.ndarray, trait: np.ndarray) -> list[dict]:
        qs = np.quantile(trait, [0.25, 0.5, 0.75])
        bins = np.digitize(trait, qs)
        return [{"trait_quartile": q + 1,
                 "n": int((bins == q).sum()),
                 "index_mean": round(float(index[bins == q].mean()), 4)}
                for q in range(4)]

    results = {
        "experiment": "m4_bars",
        "date": _dt.date.today().isoformat(),
        "run": run_name,
        "ruling32_footnote": (
            "Memo (a)'s peak-spreading attainability argument assumed a single "
            "bottleneck; the ratified two-route world adds an escape path. Any "
            "peak-spreading result is interpreted under the two-route config, "
            "and the report notes this config difference from memo (a)."),
        "peak_spreading": {
            "bar": "S20 <= 0.80 * S1 (per-day sliding 15-min maximum share)",
            "S1": S[0], "S20": S[-1],
            "S20_over_S1": round(S[-1] / S[0], 4),
            "trajectory": S,
            "pass": bool(peak_pass),
        },
        "coherence_hab": {
            "bar": "Spearman r(W_i, HAB_i) <= -0.3 (frozen W_i, Ruling 8)",
            "spearman_r": round(r_w, 4),
            "pass": bool(r_w <= -0.3),
            "disattenuated_r": round(r_w / RHO_HAT["HAB"], 4),
            "rho_hat_m1": RHO_HAT["HAB"],
            "W_mean": round(float(W.mean()), 4),
            "quartiles": quartile_summary(W, hab),
        },
        "coherence_sch": {
            "bar": "Spearman r(X_i, SCH_i) <= -0.3 (frozen X_i, Ruling 8, b = 5)",
            "spearman_r": round(r_x, 4),
            "pass": bool(r_x <= -0.3),
            "disattenuated_r": round(r_x / RHO_HAT["SCH"], 4),
            "rho_hat_m1": RHO_HAT["SCH"],
            "X_mean": round(float(X.mean()), 4),
            "quartiles": quartile_summary(X, sch),
        },
        "ruling30_predeclaration": (
            "The M4 coherence bars (|r| >= 0.3) are at elevated risk of "
            "ceiling-inherited misses given measured transmission; if they "
            "miss, the frozen decomposition attributes card vs dynamics."),
        "day_context": day_context,
    }
    (RESULTS_DIR / "m4_bars.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


if __name__ == "__main__":
    out = main(sys.argv[1] if len(sys.argv) > 1 else "m4_full")
    print(json.dumps({
        "peak": {k: out["peak_spreading"][k] for k in
                 ("S1", "S20", "S20_over_S1", "pass")},
        "hab": {k: out["coherence_hab"][k] for k in
                ("spearman_r", "disattenuated_r", "pass")},
        "sch": {k: out["coherence_sch"][k] for k in
                ("spearman_r", "disattenuated_r", "pass")},
    }, indent=2))
