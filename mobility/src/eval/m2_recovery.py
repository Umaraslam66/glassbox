"""Stage M2 grader — structural recovery vs planted truth (frozen prereg §5/§6).

Order:
1. Integrity: the official noised world must re-derive bit-identically from
   the vault answers under the RATIFIED noise parameters (Ruling 22).
2. Frozen splits (seed in the experiment config), created once and reused;
   the driver refuses to regenerate an existing split file.
3. System-side fit (mobility.src.model.mnl): per-traveler conditional logit
   on the noised answers over training scenarios, frozen utility verbatim.
4. phi_hat maps = the anchor inversions pre-declared in the config.
5. Bootstrap uncertainty, frozen bars, decomposition, comparison fit.

Run: ``python -m mobility.src.eval.m2_recovery mobility/experiments/m2_recovery.json``
"""

from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np

from mobility.src.eval.truth_vault import truth_dir
from mobility.src.model.mnl import (PARAMS, bootstrap_traveler, build_design,
                                    fit_traveler, shrinkage_refit)
from mobility.src.travelers.noise_layer import noised_answer
from mobility.src.travelers.sampler import DIMS

MOBILITY_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = MOBILITY_ROOT / "results"
EXPERIMENTS_DIR = MOBILITY_ROOT / "experiments"

# frozen anchor constants (prereg section 2), declared in the config
VOT_SCALE, VOT_K = 12.0, 0.55
SCH_MID, SCH_K = 2.828427, 0.519860
CRW_MID, CRW_SLOPE = 1.6, 0.3
PRC_K = 0.458145
PHI_WINSOR = 4.0


def load_noised(path: Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        out[(r["pid"], r["sid"])] = r["answer"]
    return out


def integrity_check(ids, sids, noised, config) -> int:
    """The official noised world must re-derive exactly (Ruling 22 params)."""
    shard_dir = truth_dir() / "runs" / "m1_sweep_r2" / "answers"
    stash = np.load(truth_dir() / "wobble_m1.npz", allow_pickle=False)
    wobble = {str(p): float(w) for p, w in zip(stash["ids"], stash["wobble"])}
    bank = json.loads((MOBILITY_ROOT / "data" / "bank" / "scenarios.json").read_text())
    options = {s["sid"]: {o["key"] for o in s["options"]} for s in bank}
    n = 0
    for pid in ids:
        for line in (shard_dir / f"{pid}.jsonl").read_text().splitlines():
            r = json.loads(line)
            if not r["answer"]:
                continue
            expect = noised_answer(r["answer"], r.get("dist_top"), options[r["sid"]],
                                   wobble[pid], 1.2, 4.0, 3190, pid, r["sid"], 0)
            if noised.get((pid, r["sid"])) != expect:
                raise AssertionError(f"noised world mismatch at {pid}/{r['sid']}")
            n += 1
    return n


def make_or_load_splits(config, ids, bank) -> dict:
    path = EXPERIMENTS_DIR / "m2_splits.json"
    if path.exists():
        return json.loads(path.read_text())
    rng = np.random.default_rng(int(config["splits"]["seed"]))
    held_travelers = sorted(rng.choice(ids, size=80, replace=False).tolist())
    by_block: dict[str, list[str]] = {}
    for s in bank:
        by_block.setdefault(s["block"], []).append(s["sid"])
    held_scenarios = []
    for block in sorted(by_block):
        sids = sorted(by_block[block])
        k = round(len(sids) * 0.2)
        held_scenarios += sorted(rng.choice(sids, size=k, replace=False).tolist())
    splits = {
        "seed": int(config["splits"]["seed"]),
        "created": _dt.date.today().isoformat(),
        "held_out_travelers": held_travelers,
        "held_out_scenarios": sorted(held_scenarios),
        "note": "frozen section 5 splits; created once before any M2 estimation; "
                "the same held-out sets serve all confirmatory results M2-M4",
    }
    path.write_text(json.dumps(splits, indent=2) + "\n", encoding="utf-8")
    return splits


def phi_hat_from_beta(beta: np.ndarray, b0: float,
                      hab_mean: float, hab_sd: float) -> np.ndarray:
    """The pre-declared anchor inversions. Order follows DIMS."""
    b_time, b_cost, b_crw, asc_car, _asc_tr, b_late, b_switch = beta
    den_cost = min(b_cost, -1e-9)
    den_time = min(b_time, -1e-9)
    vot = np.clip(60.0 * b_time / den_cost, 0.5, 200.0)
    phi_vot = np.log(vot / VOT_SCALE) / VOT_K
    ratio = np.clip(min(b_late, -1e-9) / den_time, 0.25, 32.0)
    phi_sch = np.log(ratio / SCH_MID) / SCH_K
    phi_mod = asc_car / 0.75
    m_hat = np.clip(1.0 + 3.0 * b_crw / den_time, 0.5, 4.0)
    phi_crw = (m_hat - CRW_MID) / CRW_SLOPE
    c = np.clip(-b_cost, 1e-3, 10.0)
    phi_prc = np.log(c / b0) / PRC_K
    phi_hab = (-b_switch - hab_mean) / hab_sd
    return np.clip(np.array([phi_vot, phi_sch, phi_mod, phi_crw, phi_prc, phi_hab]),
                   -PHI_WINSOR, PHI_WINSOR)


def main(config_path: Path) -> dict:
    config = json.loads(config_path.read_text())
    bank = json.loads((MOBILITY_ROOT / "data" / "bank" / "scenarios.json").read_text())
    design = build_design(bank)
    ids = [f"t{i:04d}" for i in range(400)]
    noised = load_noised(MOBILITY_ROOT / "data" / "runs" / "m1" / "answers_noised.jsonl")

    n_checked = integrity_check(ids, sorted({s["sid"] for s in bank}), noised, config)

    splits = make_or_load_splits(config, ids, bank)
    held_t = set(splits["held_out_travelers"])
    held_s = set(splits["held_out_scenarios"])
    train_ids = [p for p in ids if p not in held_t]
    heldout_ids = [p for p in ids if p in held_t]
    fit_sids = sorted(s["sid"] for s in bank
                      if s["sid"] not in held_s and s["block"] != "distractor")

    stash = np.load(truth_dir() / "phi_m1.npz", allow_pickle=False)
    assert [str(x) for x in stash["ids"]] == ids
    phi = stash["phi"]

    by_traveler: dict[str, dict[str, str]] = {p: {} for p in ids}
    for (pid, sid), ans in noised.items():
        by_traveler[pid][sid] = ans

    # ---- main per-traveler fits (frozen estimator) ------------------------
    betas: dict[str, np.ndarray] = {}
    for pid in ids:
        b = fit_traveler(design, by_traveler[pid], fit_sids)
        if b is not None:
            betas[pid] = b
    beta_mat = np.array([betas[p] for p in ids])

    # pre-declared normalizations, from TRAINING travelers, frozen from here on
    train_beta = np.array([betas[p] for p in train_ids])
    b0 = float(np.median(np.clip(-train_beta[:, 1], 1e-3, 10.0)))
    hab_vals = -train_beta[:, 6]
    hab_mean, hab_sd = float(hab_vals.mean()), float(hab_vals.std())

    phi_hat = np.array([phi_hat_from_beta(betas[p], b0, hab_mean, hab_sd)
                        for p in ids])

    held_idx = [ids.index(p) for p in heldout_ids]
    train_idx = [ids.index(p) for p in train_ids]

    results: dict = {
        "experiment": config["experiment"],
        "date": _dt.date.today().isoformat(),
        "config": config,
        "integrity": {"noised_world_rederived_identical": True,
                      "n_answers_checked": n_checked},
        "splits_realized": {
            "held_out_travelers": len(held_t),
            "held_out_scenarios": sorted(held_s),
            "training_fit_scenarios": len(fit_sids),
            "training_designed_per_dimension": None,
        },
        "fit_summary": {
            "n_travelers_fit": len(betas),
            "ridge_lambda": 1e-4,
            "mean_beta": {k: round(float(beta_mat[:, i].mean()), 4)
                          for i, k in enumerate(PARAMS)},
            "degenerate_cost_sign": int((beta_mat[:, 1] >= -1e-9).sum()),
            "degenerate_time_sign": int((beta_mat[:, 0] >= 0).sum()),
            "normalizations": {"prc_b0": b0, "hab_mean": hab_mean, "hab_sd": hab_sd},
        },
    }

    # ---- recovery bars ----------------------------------------------------
    def r_per_dim(idx):
        return {d: round(float(np.corrcoef(phi[idx, k], phi_hat[idx, k])[0, 1]), 4)
                for k, d in enumerate(DIMS)}

    rec_held = r_per_dim(held_idx)
    rec_train = r_per_dim(train_idx)
    rho = config["inputs"]["transmission_rho_hat"]
    results["recovery"] = {
        "bar": "r(phi_hat_d, phi_d) >= 0.75 on held-out travelers, all 6 dimensions",
        "held_out": rec_held,
        "held_out_median": round(float(np.median(list(rec_held.values()))), 4),
        "training_context": rec_train,
        "disattenuated_held_out": {
            d: round(min(rec_held[d] / rho[d], 1.0), 4) for d in DIMS},
        "pass_per_dimension": {d: bool(rec_held[d] >= 0.75) for d in DIMS},
        "pass": bool(all(rec_held[d] >= 0.75 for d in DIMS)),
    }

    # ---- VOT in real units + frozen decomposition -------------------------
    vot_true = VOT_SCALE * np.exp(VOT_K * phi[:, 0])
    with np.errstate(divide="ignore", invalid="ignore"):
        vot_raw = 60.0 * beta_mat[:, 0] / beta_mat[:, 1]
    err = np.abs(vot_raw - vot_true) / vot_true
    e_raw = float(np.median(err[held_idx]))
    rho_vot = float(rho["VOT"])
    e_ceiling = float(np.exp(VOT_K * 0.6745 * np.sqrt(1 - rho_vot ** 2)) - 1)
    results["vot_real_units"] = {
        "bar_frozen": "median |VOT_hat - VOT|/VOT <= 0.35 on held-out travelers",
        "e_raw_median": round(e_raw, 4),
        "pass_35": bool(e_raw <= 0.35),
        "reported_25_line": {"threshold": 0.25, "pass": bool(e_raw <= 0.25),
                             "role": "REPORTED scoreboard threshold, never a gate condition (Ruling 7)"},
        "decomposition": {
            "rho_vot_measured_m1": rho_vot,
            "e_ceiling": round(e_ceiling, 4),
            "estimator_overhead": round(e_raw - e_ceiling, 4),
            "note": "the correction removes the ceiling's systematic component only",
        },
        "n_degenerate_ratio_held_out": int((~np.isfinite(vot_raw[held_idx])).sum()
                                           + (vot_raw[held_idx] <= 0).sum()),
    }

    # ---- bootstrap uncertainty + coverage (parent convention) -------------
    rng = np.random.default_rng(3220)
    B = 200
    sigma = np.zeros((len(ids), len(DIMS)))
    n_nonconverged = 0
    for pi, pid in enumerate(ids):
        boots = bootstrap_traveler(design, by_traveler[pid], fit_sids,
                                   betas[pid], B, rng)
        phis = np.array([phi_hat_from_beta(b, b0, hab_mean, hab_sd) for b in boots])
        sigma[pi] = phis.std(axis=0)
    inside = np.abs(phi_hat[held_idx] - phi[held_idx]) <= sigma[held_idx]
    err_lat = phi_hat[held_idx] - phi[held_idx]
    results["coverage"] = {
        "bar": "pooled 60-75% (|phi_hat - phi| <= 1 bootstrap SD, held-out travelers x 6 dims)",
        "pooled_coverage": round(float(inside.mean()), 4),
        "per_dimension_coverage": {d: round(float(inside[:, k].mean()), 4)
                                   for k, d in enumerate(DIMS)},
        "pass": bool(0.60 <= float(inside.mean()) <= 0.75),
        "bootstrap": {"B": B, "seed": 3220, "n_nonconverged": n_nonconverged},
        "diagnostics": {
            "mean_sigma_per_dimension": {d: round(float(sigma[held_idx][:, k].mean()), 4)
                                         for k, d in enumerate(DIMS)},
            "variance_ratio_actual_over_predicted": {
                d: round(float(np.mean(err_lat[:, k] ** 2)
                               / np.mean(sigma[held_idx][:, k] ** 2)), 3)
                for k, d in enumerate(DIMS)},
        },
    }

    # ---- designed-zero separability ---------------------------------------
    r_sep = float(np.corrcoef(phi_hat[held_idx, 0], phi_hat[held_idx, 4])[0, 1])
    results["separability_vot_prc"] = {
        "bar": "|r(phi_hat_VOT, phi_hat_PRC)| <= 0.15 on held-out travelers",
        "r": round(r_sep, 4),
        "pass": bool(abs(r_sep) <= 0.15),
    }

    # ---- distractor tripwire ----------------------------------------------
    distractor_sids = sorted(s["sid"] for s in bank if s["block"] == "distractor")
    trip = {}
    worst = 0.0
    for sid in distractor_sids:
        coded = np.array([{"A": 1.0, "B": -1.0}.get(noised.get((p, sid)), np.nan)
                          for p in ids])
        ok = ~np.isnan(coded)
        rs = {d: round(float(np.corrcoef(coded[ok], phi_hat[ok, k])[0, 1]), 4)
              for k, d in enumerate(DIMS)}
        trip[sid] = rs
        worst = max(worst, max(abs(v) for v in rs.values()))
    results["distractor_tripwire"] = {
        "bar": "every distractor x dimension |r| <= 0.15 (all 400 travelers, noised choices)",
        "max_abs_r": round(worst, 4),
        "pass": bool(worst <= 0.15),
        "per_scenario": trip,
    }

    # ---- comparison fit (frozen 'alongside'; REPORTED, no bar) ------------
    shrunk = shrinkage_refit(design, by_traveler, fit_sids, betas, train_ids)
    S_mat = np.array([shrunk[p] for p in ids])
    mu_t = S_mat[[ids.index(p) for p in train_ids]].mean(axis=0)
    sd_t = S_mat[[ids.index(p) for p in train_ids]].std(axis=0)
    Z = (S_mat - mu_t) / np.maximum(sd_t, 1e-9)
    Zt = Z[train_idx]
    _, _, Vt = np.linalg.svd(Zt - Zt.mean(axis=0), full_matrices=False)
    comps = Vt[:6].T
    scores = (Z - Zt.mean(axis=0)) @ comps
    M = scores[train_idx].T @ phi[train_idx]
    U, _, Vh = np.linalg.svd(M)
    R = U @ Vh
    aligned = scores @ R
    results["latent_comparison_fit"] = {
        "role": "frozen section 5 'alongside' comparison; REPORTED, carries no bar",
        "operationalization": config["comparison_fit"]["operationalization"],
        "held_out_r": {d: round(float(np.corrcoef(phi[held_idx, k],
                                                  aligned[held_idx, k])[0, 1]), 4)
                       for k, d in enumerate(DIMS)},
    }

    # training designed-scenario counts per dimension (context for thin dims)
    truth_design = {d["sid"]: d for d in json.loads(
        (truth_dir() / "bank" / "design.json").read_text())}
    results["splits_realized"]["training_designed_per_dimension"] = {
        d: sum(1 for sid in fit_sids if truth_design[sid]["primary"] == d)
        for d in DIMS}

    (RESULTS_DIR / f"{config['experiment']}.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


if __name__ == "__main__":
    out = main(Path(sys.argv[1]))
    print(json.dumps({
        "recovery_held_out": out["recovery"]["held_out"],
        "recovery_pass": out["recovery"]["pass"],
        "vot_e_raw": out["vot_real_units"]["e_raw_median"],
        "vot_pass_35": out["vot_real_units"]["pass_35"],
        "coverage": out["coverage"]["pooled_coverage"],
        "separability_r": out["separability_vot_prc"]["r"],
        "tripwire_max": out["distractor_tripwire"]["max_abs_r"],
    }, indent=2))
