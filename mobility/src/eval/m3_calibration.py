"""Stage M3 grader — calibration + elasticity realism (frozen prereg §5/§6).

Every analysis choice below was pre-declared in
experiments/m3_calibration.json and the project log BEFORE this module ran
(commit 40d3a2a). Zero API calls.

Sections:
1. Short-survey estimator (K = 12 + profile) with EB prior from training
   travelers; frozen bars: Brier lift >= 25%, ECE <= 0.05, coverage 60-75%.
2. Coverage-decay curve over the K grid (watched, reported).
3. Static elasticity probes from the policy block, vs published ranges,
   with the frozen by-construction caveat verbatim. REPORTED, no bar.
4. Aggregate-vs-individual contrast.
5. Ruling-26 exploratory estimator comparison (labeled; specs fixed at
   M2-close).

Run: ``python -m mobility.src.eval.m3_calibration mobility/experiments/m3_calibration.json``
"""

from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np

from mobility.src.eval.m2_recovery import load_noised, phi_hat_from_beta
from mobility.src.eval.truth_vault import truth_dir
from mobility.src.model.mnl import (PARAMS, build_design, fit_traveler,
                                    shrinkage_refit)
from mobility.src.model.short_survey import (knn_shares, map_fit_with_cov,
                                             predict_probs,
                                             profile_group_shares)
from mobility.src.travelers.noise_layer import noised_answer
from mobility.src.travelers.sampler import DIMS

MOBILITY_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = MOBILITY_ROOT / "results"

CAVEAT = ("By-construction caveat (frozen section 6, verbatim): the "
          "population's planted VOT/PRC distributions partly determine these "
          "numbers; the comparison is a realism check, never recovery "
          "evidence.")


def stratified_permutation(bank, held_s, seed):
    """One seeded block-proportional interleave; nested prefixes give every K."""
    rng = np.random.default_rng(seed)
    blocks: dict[str, list[str]] = {}
    for s in bank:
        if s["block"] != "distractor" and s["sid"] not in held_s:
            blocks.setdefault(s["block"], []).append(s["sid"])
    order = {b: [blocks[b][i] for i in rng.permutation(len(blocks[b]))]
             for b in sorted(blocks)}
    total = sum(len(v) for v in order.values())
    taken = {b: 0 for b in order}
    perm = []
    for n in range(1, total + 1):
        b = max(order, key=lambda k: len(order[k]) / total * n - taken[k])
        perm.append(order[b][taken[b]])
        taken[b] += 1
    return perm


def true_distribution(pre_answer, dist_top, valid, wobble, pid, sid):
    """Frozen section 5: 50 seeded noise-layer samples, rounds 100-149."""
    counts: dict[str, int] = {}
    for rnd in range(100, 150):
        a = noised_answer(pre_answer, dist_top, valid, wobble, 1.2, 4.0,
                          3190, pid, sid, rnd)
        counts[a] = counts.get(a, 0) + 1
    return {o: c / 50 for o, c in counts.items()}


def brier(pred: dict[str, float] | None, target: dict[str, float],
          options: list[str]) -> float | None:
    if pred is None:
        return None
    return sum((pred.get(o, 0.0) - target.get(o, 0.0)) ** 2 for o in options)


def main(config_path: Path) -> dict:
    config = json.loads(config_path.read_text())
    bank = json.loads((MOBILITY_ROOT / "data" / "bank" / "scenarios.json").read_text())
    by_sid = {s["sid"]: s for s in bank}
    design = build_design(bank)
    ids = [f"t{i:04d}" for i in range(400)]
    noised = load_noised(MOBILITY_ROOT / "data" / "runs" / "m1" / "answers_noised.jsonl")
    profiles = {p["pid"]: p for p in json.loads(
        (MOBILITY_ROOT / "data" / "profiles.json").read_text())}
    splits = json.loads((MOBILITY_ROOT / "experiments" / "m2_splits.json").read_text())
    held_t, held_s = set(splits["held_out_travelers"]), set(splits["held_out_scenarios"])
    train_ids = [p for p in ids if p not in held_t]
    heldout_ids = [p for p in ids if p in held_t]
    fit_sids = sorted(s["sid"] for s in bank
                      if s["sid"] not in held_s and s["block"] != "distractor")
    eval_sids = sorted(s["sid"] for s in bank
                       if s["sid"] in held_s and s["block"] != "distractor")

    stash = np.load(truth_dir() / "phi_m1.npz", allow_pickle=False)
    phi = stash["phi"]
    wob = np.load(truth_dir() / "wobble_m1.npz", allow_pickle=False)
    wobble = {str(p): float(w) for p, w in zip(wob["ids"], wob["wobble"])}
    pre, dists = {}, {}
    shard_dir = truth_dir() / "runs" / "m1_sweep_r2" / "answers"
    for pid in ids:
        for line in (shard_dir / f"{pid}.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r["answer"]:
                pre[(pid, r["sid"])] = r["answer"]
                dists[(pid, r["sid"])] = r.get("dist_top")

    by_traveler: dict[str, dict[str, str]] = {p: {} for p in ids}
    for (pid, sid), ans in noised.items():
        by_traveler[pid][sid] = ans

    nrm = config["inputs"]["m2_normalization_constants"]
    b0, hab_mean, hab_sd = nrm["prc_b0"], nrm["hab_mean"], nrm["hab_sd"]

    results: dict = {"experiment": config["experiment"],
                     "date": _dt.date.today().isoformat(), "config": config}

    # ---- EB prior from training travelers (deterministic refit) -----------
    betas = {p: fit_traveler(design, by_traveler[p], fit_sids) for p in ids}
    shrunk = shrinkage_refit(design, by_traveler, fit_sids, betas, train_ids)
    S_train = np.array([shrunk[p] for p in train_ids])
    prior_mean = S_train.mean(axis=0)
    prior_cov = np.cov(S_train.T) + 1e-6 * np.eye(len(PARAMS))
    prior_prec = np.linalg.inv(prior_cov)

    # profile design check: group prior-mean deltas in population-SD units
    sd = np.maximum(S_train.std(axis=0), 1e-9)
    deltas = {}
    for field in ["age_band", "occupation_type", "household", "area_type"]:
        worst = 0.0
        for val in {profiles[p][field] for p in train_ids}:
            members = [i for i, p in enumerate(train_ids)
                       if profiles[p][field] == val]
            if len(members) >= 10:
                d = np.abs(S_train[members].mean(axis=0) - prior_mean) / sd
                worst = max(worst, float(d.max()))
        deltas[field] = round(worst, 3)
    results["profile_design_check"] = {
        "max_group_prior_mean_delta_sd_units": deltas,
        "note": "profile is designed signal-free; posterior uses the "
                "unconditional prior (pre-declared)",
    }

    # ---- K permutation and short-survey fits ------------------------------
    perm = stratified_permutation(bank, held_s, 3230)  # seed pre-declared
    k12 = perm[:12]
    comp = {b: sum(by_sid[s]["block"] == b for s in k12)
            for b in ["mode", "dep_time", "route", "policy"]}
    assert comp == {"mode": 5, "dep_time": 3, "route": 2, "policy": 2}, comp
    results["k12_scenarios"] = k12

    k_grid = config["short_survey"]["k_grid"]
    fits: dict[int, dict[str, tuple]] = {}
    for K in k_grid:
        sub = perm[:K]
        fits[K] = {}
        for pid in heldout_ids:
            out = map_fit_with_cov(design, by_traveler[pid], sub,
                                   prior_mean, prior_prec)
            fits[K][pid] = out

    # ---- frozen bar 1: Brier lift at K = 12 -------------------------------
    base_share, opts = {}, {}
    for sid in eval_sids:
        counts: dict[str, int] = {}
        for p in train_ids:
            a = by_traveler[p].get(sid)
            if a:
                counts[a] = counts.get(a, 0) + 1
        total = sum(counts.values())
        base_share[sid] = {o: c / total for o, c in counts.items()}
        opts[sid] = [o["key"] for o in by_sid[sid]["options"]]

    rows = {"model": [], "marginal": [], "knn": [], "profile": []}
    for pid in heldout_ids:
        beta12 = fits[12][pid][0]
        for sid in eval_sids:
            if (pid, sid) not in pre:
                continue
            p_true = true_distribution(pre[(pid, sid)], dists[(pid, sid)],
                                       set(opts[sid]), wobble[pid], pid, sid)
            preds = {
                "model": predict_probs(design, sid, beta12),
                "marginal": base_share[sid],
                "knn": knn_shares(pid, k12, sid, by_traveler, train_ids, k=20),
                "profile": profile_group_shares(profiles[pid], sid, profiles,
                                                by_traveler, train_ids),
            }
            for name, pr in preds.items():
                b = brier(pr, p_true, opts[sid])
                if b is not None:
                    rows[name].append(b)
    bs = {name: float(np.mean(v)) for name, v in rows.items()}
    lift = 1 - bs["model"] / bs["marginal"]
    results["brier_lift"] = {
        "bar": ">= 25% Brier improvement over the population-marginal baseline",
        "brier": {k: round(v, 4) for k, v in bs.items()},
        "n_cells": len(rows["model"]),
        "lift_vs_marginal": round(lift, 4),
        "pass": bool(lift >= 0.25),
        "knn_lift_reported": round(1 - bs["knn"] / bs["marginal"], 4),
        "profile_only_lift_reported": round(1 - bs["profile"] / bs["marginal"], 4),
        "llm_profile_baseline": config["baseline_not_run"],
    }

    # ---- frozen bar 2: ECE ------------------------------------------------
    conf, hit = [], []
    for pid in heldout_ids:
        beta12 = fits[12][pid][0]
        for sid in eval_sids:
            chosen = by_traveler[pid].get(sid)
            if not chosen:
                continue
            pr = predict_probs(design, sid, beta12)
            for o in opts[sid]:
                conf.append(pr.get(o, 0.0))
                hit.append(1.0 if chosen == o else 0.0)
    conf, hit = np.array(conf), np.array(hit)
    ece = 0.0
    bins = []
    for lo in np.arange(0, 1.0, 0.1):
        m = (conf >= lo) & (conf < lo + 0.1 if lo < 0.9 else conf <= 1.0)
        if m.sum():
            gap = abs(hit[m].mean() - conf[m].mean())
            ece += m.sum() / len(conf) * gap
            bins.append({"bin": f"[{lo:.1f},{lo+0.1:.1f})", "n": int(m.sum()),
                         "confidence": round(float(conf[m].mean()), 4),
                         "accuracy": round(float(hit[m].mean()), 4)})
    results["ece"] = {"bar": "<= 0.05 (10 equal-width bins)",
                      "ece": round(float(ece), 4), "pass": bool(ece <= 0.05),
                      "n_cells": len(conf), "bins": bins}

    # ---- frozen bar 3: coverage at K = 12 + decay curve -------------------
    rng = np.random.default_rng(3240)
    decay = []
    cov_pass = None
    for K in k_grid:
        inside_all, r_dims = [], {}
        phi_hat_k = {}
        for pid in heldout_ids:
            beta, cov = fits[K][pid]
            draws = rng.multivariate_normal(beta, cov, size=200)
            ph = np.array([phi_hat_from_beta(b, b0, hab_mean, hab_sd)
                           for b in draws])
            centre = phi_hat_from_beta(beta, b0, hab_mean, hab_sd)
            sigma = ph.std(axis=0)
            k_idx = ids.index(pid)
            inside_all.append(np.abs(centre - phi[k_idx]) <= sigma)
            phi_hat_k[pid] = centre
        inside = np.array(inside_all)
        centres = np.array([phi_hat_k[p] for p in heldout_ids])
        truth = np.array([phi[ids.index(p)] for p in heldout_ids])
        r_dims = {d: round(float(np.corrcoef(truth[:, k], centres[:, k])[0, 1]), 4)
                  for k, d in enumerate(DIMS)}
        row = {"K": K, "pooled_coverage": round(float(inside.mean()), 4),
               "median_r": round(float(np.median(list(r_dims.values()))), 4),
               "per_dimension_coverage": {d: round(float(inside[:, k].mean()), 4)
                                          for k, d in enumerate(DIMS)}}
        decay.append(row)
        if K == 12:
            cov_pass = row
    results["coverage"] = {
        "bar": "pooled 60-75% at K = 12 (1 Laplace-draw SD, held-out x 6 dims)",
        "at_k12": cov_pass,
        "pass": bool(0.60 <= cov_pass["pooled_coverage"] <= 0.75),
        "decay_curve": decay,
        "note": "decay watched and reported per the frozen bar; a finding, "
                "never a failure",
    }

    # ---- elasticities (REPORTED, no bar) ----------------------------------
    def share(sid, key):
        n = sum(1 for p in ids if by_traveler[p].get(sid))
        c = sum(1 for p in ids if by_traveler[p].get(sid) == key)
        return c / n

    prc = [(by_sid[f"policy_prc_{i:02d}"]["options"][0]["attrs"]["cost_eur"],
            share(f"policy_prc_{i:02d}", "A")) for i in range(15)]
    prc.sort()
    (p_lo, q_lo), (p_hi, q_hi) = prc[0], prc[-1]
    arc_prc = ((q_hi - q_lo) / ((q_hi + q_lo) / 2)) / ((p_hi - p_lo) / ((p_hi + p_lo) / 2))
    lp, lq = np.log([p for p, q in prc]), np.log([max(q, 1e-3) for p, q in prc])
    ols_prc = float(np.polyfit(lp, lq, 1)[0])

    vot_rows = []
    for i in range(4):
        s = by_sid[f"policy_vot_{i:02d}"]
        a = s["options"][0]["attrs"]
        vot_rows.append((a["cost_eur"], -a["time_min"], share(s["sid"], "A")))
    vot_rows.sort()
    (t_lo, m_lo, qv_lo), (t_hi, m_hi, qv_hi) = vot_rows[0], vot_rows[-1]
    arc_vot = ((qv_hi - qv_lo) / ((qv_hi + qv_lo) / 2)) / ((t_hi - t_lo) / ((t_hi + t_lo) / 2))

    mod_rows = {}
    for i in range(6):
        s = by_sid[f"policy_mod_{i:02d}"]
        pen = s["options"][1]["attrs"]["time_min"]
        mod_rows.setdefault(pen, []).append(share(s["sid"], "B"))
    subsidy = {f"+{k} min": round(float(np.mean(v)), 4)
               for k, v in sorted(mod_rows.items())}

    results["elasticities_reported"] = {
        "caveat": CAVEAT,
        "role": "REPORTED realism check; no bar (frozen section 6)",
        "data": "round-0 noised choices, all 400 travelers (non-confirmatory aggregate)",
        "parking_driving_cost": {
            "curve_cost_vs_car_share": [[round(p, 2), round(q, 4)] for p, q in prc],
            "arc_elasticity_min_to_max": round(arc_prc, 3),
            "log_log_ols_slope": round(ols_prc, 3),
            "published_range": "-0.1 to -0.4 (Goodwin, Dargay & Hanly 2004; "
                               "Vaca & Kuzmyak 2005; Litman)",
        },
        "toll_take_up": {
            "curve_toll_vs_take_up": [[round(t, 2), int(m), round(q, 4)]
                                      for t, m, q in vot_rows],
            "arc_elasticity_min_to_max_toll": round(arc_vot, 3),
            "published_range": "-0.1 to -0.45 (Odeck & Brathen 2008 review)",
            "note": "time savings co-vary 16-21 min across the 4 scenarios "
                    "(price per saved minute 0.20-0.23 EUR); stated per the "
                    "pre-declaration",
        },
        "employer_subsidy_uptake": {
            "uptake_by_time_penalty": subsidy,
            "note": "no price variation in this family; no elasticity",
        },
        "transit_fare_gap": "the design table's fare-change probe was never "
                            "built into the bank; the fare elasticity vs the "
                            "published -0.2 to -0.5 range cannot be computed "
                            "and is reported as missing",
    }

    # ---- aggregate vs individual ------------------------------------------
    def share_cells(source):
        pred, obs = [], []
        for sid in eval_sids:
            counts: dict[str, float] = {o: 0.0 for o in opts[sid]}
            n = 0
            for pid in heldout_ids:
                a = by_traveler[pid].get(sid)
                if a:
                    counts[a] += 1
                    n += 1
            for o in opts[sid]:
                obs_share = counts[o] / n
                if source == "k12":
                    ps = [predict_probs(design, sid, fits[12][p][0]).get(o, 0.0)
                          for p in heldout_ids]
                elif source == "m2":
                    ps = [predict_probs(design, sid, betas[p]).get(o, 0.0)
                          for p in heldout_ids]
                else:
                    ps = [base_share[sid].get(o, 0.0)]
                pred.append(float(np.mean(ps)))
                obs.append(obs_share)
        pred, obs = np.array(pred), np.array(obs)
        return {"mae_share_points": round(float(np.abs(pred - obs).mean()) * 100, 2),
                "pearson_r": round(float(np.corrcoef(pred, obs)[0, 1]), 4)}

    results["aggregate_vs_individual"] = {
        "aggregate_share_prediction": {
            "k12_short_survey": share_cells("k12"),
            "m2_full_data_context": share_cells("m2"),
            "population_marginal_reference": share_cells("marginal"),
        },
        "individual_recovery_context": {
            "m2_frozen_median_r": 0.4267,
            "k12_median_r": cov_pass["median_r"],
        },
        "reading": "the parent's aggregate-easy/individual-hard question, "
                   "answered for mobility",
    }

    # ---- Ruling-26 exploratory comparison ---------------------------------
    m2 = json.loads((RESULTS_DIR / "m2_recovery.json").read_text())
    diag = json.loads((RESULTS_DIR / "m2_diagnostics.json").read_text())
    lam = 1.0
    reg_prec = lam * np.eye(len(PARAMS))
    zeros = np.zeros(len(PARAMS))
    reg_betas = {p: fit_traveler(design, by_traveler[p], fit_sids,
                                 prior_mean=zeros, prior_prec=reg_prec)
                 for p in ids}
    reg_mat = np.array([reg_betas[p] for p in ids])
    reg_train = np.array([reg_betas[p] for p in train_ids])
    reg_b0 = float(np.median(np.clip(-reg_train[:, 1], 1e-3, 10.0)))
    rh = -reg_train[:, 6]
    reg_phi = np.array([phi_hat_from_beta(reg_betas[p], reg_b0,
                                          float(rh.mean()), float(rh.std()))
                        for p in ids])
    hidx = [ids.index(p) for p in heldout_ids]
    reg_r = {d: round(float(np.corrcoef(phi[hidx, k], reg_phi[hidx, k])[0, 1]), 4)
             for k, d in enumerate(DIMS)}
    vot_true = 12.0 * np.exp(0.55 * phi[:, 0])
    with np.errstate(divide="ignore", invalid="ignore"):
        reg_vot = 60.0 * reg_mat[:, 0] / reg_mat[:, 1]
    reg_err = float(np.median(np.abs(reg_vot[hidx] - vot_true[hidx]) / vot_true[hidx]))
    results["ruling26_exploratory_comparison"] = {
        "label": "EXPLORATORY - declared at M2-close (Ruling 26); cannot "
                 "overturn frozen verdicts",
        "held_out_r": {
            "frozen_mnl_m2": m2["recovery"]["held_out"],
            "designed_score_index_oracle": diag["designed_score_contrast_exploratory"]
                ["held_out_r_from_simple_tendency_index"],
            "regularized_mnl_lambda_1.0": reg_r,
        },
        "median_r": {
            "frozen_mnl_m2": m2["recovery"]["held_out_median"],
            "designed_score_index_oracle": diag["designed_score_contrast_exploratory"]["median"],
            "regularized_mnl_lambda_1.0": round(float(np.median(list(reg_r.values()))), 4),
        },
        "vot_median_error": {
            "frozen_mnl_m2": m2["vot_real_units"]["e_raw_median"],
            "regularized_mnl_lambda_1.0": round(reg_err, 4),
        },
    }

    (RESULTS_DIR / f"{config['experiment']}.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


if __name__ == "__main__":
    out = main(Path(sys.argv[1]))
    print(json.dumps({
        "brier_lift": {k: out["brier_lift"][k] for k in
                       ("lift_vs_marginal", "pass", "knn_lift_reported",
                        "profile_only_lift_reported")},
        "ece": {k: out["ece"][k] for k in ("ece", "pass")},
        "coverage_k12": out["coverage"]["at_k12"]["pooled_coverage"],
        "coverage_pass": out["coverage"]["pass"],
        "decay": [(r["K"], r["pooled_coverage"], r["median_r"])
                  for r in out["coverage"]["decay_curve"]],
        "elasticity_parking": out["elasticities_reported"]
            ["parking_driving_cost"]["arc_elasticity_min_to_max"],
        "elasticity_toll": out["elasticities_reported"]
            ["toll_take_up"]["arc_elasticity_min_to_max_toll"],
        "aggregate": out["aggregate_vs_individual"]["aggregate_share_prediction"],
        "ruling26_medians": out["ruling26_exploratory_comparison"]["median_r"],
    }, indent=2))
