"""Gate M2 diagnostics — the mandated tripwire leakage hunt + fit forensics.

Grader-side, zero API. Two jobs, run after the frozen M2 pipeline reported:

1. The distractor tripwire fired (max |r| 0.22 > 0.15), and the frozen bar
   text requires a documented leakage hunt before Gate M2 reports. The hunt
   correlates distractor choices with PLANTED phi (pre-noise and noised):
   breaches there mean the renderer leaks traits into designed-zero
   scenarios; breaches only against phi_hat mean correlated estimation
   variance, not truth leakage.
2. EXPLORATORY forensics for the honest gate diagnosis (no bars): in-sample
   and held-out-scenario predictive accuracy of the fitted models beside the
   population-marginal baseline, coefficient sign/saturation profile, and
   the designed-score-index contrast (does a simple frozen-QA-style tendency
   index recover more phi signal than the structural estimator?).

Run: ``python -m mobility.src.eval.m2_diagnostics mobility/experiments/m2_recovery.json``
"""

from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np

from mobility.src.eval.m2_recovery import load_noised, phi_hat_from_beta
from mobility.src.eval.truth_vault import truth_dir
from mobility.src.model.mnl import build_design, fit_traveler
from mobility.src.travelers.sampler import DIMS

MOBILITY_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = MOBILITY_ROOT / "results"


def main(config_path: Path) -> dict:
    config = json.loads(config_path.read_text())
    bank = json.loads((MOBILITY_ROOT / "data" / "bank" / "scenarios.json").read_text())
    design = build_design(bank)
    ids = [f"t{i:04d}" for i in range(400)]
    noised = load_noised(MOBILITY_ROOT / "data" / "runs" / "m1" / "answers_noised.jsonl")
    splits = json.loads((MOBILITY_ROOT / "experiments" / "m2_splits.json").read_text())
    held_t = set(splits["held_out_travelers"])
    held_s = set(splits["held_out_scenarios"])
    train_ids = [p for p in ids if p not in held_t]
    heldout_ids = [p for p in ids if p in held_t]
    fit_sids = sorted(s["sid"] for s in bank
                      if s["sid"] not in held_s and s["block"] != "distractor")
    eval_sids = sorted(s["sid"] for s in bank
                       if s["sid"] in held_s and s["block"] != "distractor")

    stash = np.load(truth_dir() / "phi_m1.npz", allow_pickle=False)
    phi = stash["phi"]
    pre = {}
    shard_dir = truth_dir() / "runs" / "m1_sweep_r2" / "answers"
    for pid in ids:
        for line in (shard_dir / f"{pid}.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r["answer"]:
                pre[(pid, r["sid"])] = r["answer"]

    by_traveler: dict[str, dict[str, str]] = {p: {} for p in ids}
    for (pid, sid), ans in noised.items():
        by_traveler[pid][sid] = ans

    # deterministic refit of the main estimates (same code path as the run)
    betas = {p: fit_traveler(design, by_traveler[p], fit_sids) for p in ids}
    beta_mat = np.array([betas[p] for p in ids])
    train_beta = np.array([betas[p] for p in train_ids])
    b0 = float(np.median(np.clip(-train_beta[:, 1], 1e-3, 10.0)))
    hab_vals = -train_beta[:, 6]
    phi_hat = np.array([phi_hat_from_beta(betas[p], b0, float(hab_vals.mean()),
                                          float(hab_vals.std())) for p in ids])

    results: dict = {"experiment": "m2_diagnostics",
                     "date": _dt.date.today().isoformat()}

    # ---- 1. the mandated tripwire hunt ------------------------------------
    hunt = {}
    worst = {"noised_vs_phi": 0.0, "prenoise_vs_phi": 0.0, "noised_vs_phi_hat": 0.0}
    for sid in sorted(s["sid"] for s in bank if s["block"] == "distractor"):
        row = {}
        for label, source, target in [
                ("noised_vs_phi", noised, phi),
                ("prenoise_vs_phi", pre, phi),
                ("noised_vs_phi_hat", noised, phi_hat)]:
            coded = np.array([{"A": 1.0, "B": -1.0}.get(source.get((p, sid)), np.nan)
                              for p in ids])
            ok = ~np.isnan(coded)
            rs = {d: round(float(np.corrcoef(coded[ok], target[ok, k])[0, 1]), 4)
                  for k, d in enumerate(DIMS)}
            row[label] = rs
            worst[label] = max(worst[label], max(abs(v) for v in rs.values()))
        hunt[sid] = row
    results["tripwire_hunt"] = {
        "status": "MANDATED by the frozen distractor bar (fired at 0.2215)",
        "max_abs": {k: round(v, 4) for k, v in worst.items()},
        "per_scenario": hunt,
        "reading": ("breaches against planted phi = renderer leaks traits into "
                    "designed-zero scenarios; breaches only against phi_hat = "
                    "correlated estimation variance, not truth leakage"),
    }

    # ---- 2. exploratory forensics -----------------------------------------
    def accuracy(sids_used, traveler_set):
        model_hits = base_hits = n = 0
        share = {}
        for sid in sids_used:
            counts = {}
            for p in train_ids:
                a = by_traveler[p].get(sid)
                if a:
                    counts[a] = counts.get(a, 0) + 1
            share[sid] = max(counts, key=counts.get) if counts else None
        for p in traveler_set:
            b = betas[p]
            for sid in sids_used:
                ans = by_traveler[p].get(sid)
                if not ans or not design[sid]["informative"]:
                    continue
                u = design[sid]["X"] @ b
                pred = design[sid]["keys"][int(np.argmax(u))]
                model_hits += int(pred == ans)
                base_hits += int(share[sid] == ans)
                n += 1
        return {"model_accuracy": round(model_hits / n, 4),
                "population_majority_baseline": round(base_hits / n, 4),
                "n_choices": n}

    results["fit_quality_exploratory"] = {
        "in_sample_training_scenarios_heldout_travelers":
            accuracy(fit_sids, heldout_ids),
        "held_out_scenarios_heldout_travelers":
            accuracy(eval_sids, heldout_ids),
        "note": "argmax prediction vs the noised choice; baseline = per-scenario "
                "majority option over training travelers (zero-information)",
    }

    with np.errstate(divide="ignore", invalid="ignore"):
        vot_raw = 60.0 * beta_mat[:, 0] / beta_mat[:, 1]
        ratio = beta_mat[:, 5] / beta_mat[:, 0]
    results["saturation_profile_exploratory"] = {
        "b_time_nonnegative": int((beta_mat[:, 0] >= 0).sum()),
        "b_cost_nonnegative": int((beta_mat[:, 1] >= -1e-9).sum()),
        "vot_ratio_outside_winsor_0.5_200": int(
            ((vot_raw < 0.5) | (vot_raw > 200) | ~np.isfinite(vot_raw)).sum()),
        "abs_asc_car_above_3": int((np.abs(beta_mat[:, 3]) > 3).sum()),
        "abs_b_switch_above_3": int((np.abs(beta_mat[:, 6]) > 3).sum()),
        "n_travelers": len(ids),
        "note": "near-deterministic rendered choices push coefficients to "
                "ridge-limited magnitudes (quasi-separation); ratios destabilize",
    }

    # designed-score contrast: the frozen-QA-style tendency index on the same
    # noised training data, graded on the same held-out travelers
    truth_design = {d["sid"]: d for d in json.loads(
        (truth_dir() / "bank" / "design.json").read_text())}
    held_idx = [ids.index(p) for p in heldout_ids]
    contrast = {}
    for k, dim in enumerate(DIMS):
        dim_sids = [sid for sid in fit_sids
                    if truth_design[sid]["primary"] == dim]
        tend, ph = [], []
        for p in heldout_ids:
            scores = [truth_design[sid]["scores"][by_traveler[p][sid]]
                      for sid in dim_sids if by_traveler[p].get(sid)]
            if scores:
                tend.append(float(np.mean(scores)))
                ph.append(phi[ids.index(p), k])
        contrast[dim] = round(float(np.corrcoef(ph, tend)[0, 1]), 4)
    results["designed_score_contrast_exploratory"] = {
        "held_out_r_from_simple_tendency_index": contrast,
        "median": round(float(np.median(list(contrast.values()))), 4),
        "note": "same noised choices, same held-out travelers, same training "
                "scenarios as the structural estimator; the index knows the "
                "designed loadings (grader-side oracle), so this is an upper "
                "reference for signal present in the choices, not a rival "
                "system-side estimator",
    }

    (RESULTS_DIR / "m2_diagnostics.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


if __name__ == "__main__":
    out = main(Path(sys.argv[1]))
    print(json.dumps({
        "hunt_max": out["tripwire_hunt"]["max_abs"],
        "fit_quality": out["fit_quality_exploratory"],
        "saturation": {k: v for k, v in
                       out["saturation_profile_exploratory"].items()
                       if isinstance(v, int)},
        "designed_score_contrast":
            out["designed_score_contrast_exploratory"]
            ["held_out_r_from_simple_tendency_index"],
    }, indent=2))
