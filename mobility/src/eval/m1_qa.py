"""Stage M1 QA pipeline (vault-side): gate, transmission, noise tuning, bars.

Order fixed by Gate M1 Ruling 16:

1. The frozen obedience bar on its native instrument — full-bank, pre-noise:
   median across the 6 dimensions of r(φ_d, mean designed-score tendency on
   the scenarios loading on d). **If the median is below 0.5 everything stops
   here**: partial results are written and nothing else runs.
2. Confirmatory card transmission per the frozen §5 estimator (pre-noise,
   split-half reliability corrected, split seed recorded).
3. Wobble mint (planted independently of φ, seed recorded) + noise-layer
   tuning: (a, b) chosen so pooled test-retest agreement lands in the frozen
   70–90% band (target 0.79, parent's realized level), then reported for
   freezing at the gate.
4. Official noised answers (round 0) written to the system side with exactly
   the allowlisted fields; diversity bars; obedience on the official noised
   world; test-retest per the frozen definition (round 0 vs an independent
   seeded round 1 on 30 fixed scenarios per traveler; the fresh-session
   re-ask is operationalized as an independent seeded application of the
   frozen layer, exactly as the parent's addendum did).

Run: ``python -m mobility.src.eval.m1_qa mobility/experiments/m1_qa.json``
"""

from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np

from mobility.src.eval.truth_vault import truth_dir
from mobility.src.travelers.noise_layer import mint_wobble, noised_answer
from mobility.src.travelers.sampler import DIMS

MOBILITY_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = MOBILITY_ROOT / "results"
SYSTEM_RUNS = MOBILITY_ROOT / "data" / "runs" / "m1"


def load_sweep(sweep_name: str) -> tuple[list[str], dict, dict]:
    shard_dir = truth_dir() / "runs" / sweep_name / "answers"
    ids = [f"t{i:04d}" for i in range(400)]
    answers: dict = {}
    dists: dict = {}
    for pid in ids:
        for line in (shard_dir / f"{pid}.jsonl").read_text().splitlines():
            r = json.loads(line)
            answers[(pid, r["sid"])] = r["answer"]
            dists[(pid, r["sid"])] = r.get("dist_top")
    return ids, answers, dists


def obedience_table(ids, answers, design, phi) -> dict:
    out = {}
    for di, dim in enumerate(DIMS):
        dim_sids = [sid for sid, d in design.items() if d["primary"] == dim]
        tendencies, phis = [], []
        for pi, pid in enumerate(ids):
            scores = [design[sid]["scores"][answers[(pid, sid)]]
                      for sid in dim_sids if answers.get((pid, sid))]
            if scores:
                tendencies.append(float(np.mean(scores)))
                phis.append(phi[pi, di])
        out[dim] = {"r": round(float(np.corrcoef(phis, tendencies)[0, 1]), 4),
                    "n_scenarios": len(dim_sids)}
    out["median"] = round(float(np.median([out[d]["r"] for d in DIMS])), 4)
    return out


def transmission_table(ids, answers, design, phi, split_seed: int) -> dict:
    rng = np.random.default_rng(split_seed)
    out = {}
    for di, dim in enumerate(DIMS):
        dim_sids = sorted(sid for sid, d in design.items() if d["primary"] == dim)
        perm = list(rng.permutation(len(dim_sids)))
        half_a = [dim_sids[k] for k in perm[0::2]]
        half_b = [dim_sids[k] for k in perm[1::2]]

        def tendency(pid, subset):
            scores = [design[sid]["scores"][answers[(pid, sid)]]
                      for sid in subset if answers.get((pid, sid))]
            return float(np.mean(scores)) if scores else None

        rows = [(tendency(p, half_a), tendency(p, half_b), tendency(p, dim_sids),
                 phi[pi, di]) for pi, p in enumerate(ids)]
        rows = [t for t in rows if None not in t[:3]]
        ta, tb, tt, ph = map(np.array, zip(*rows))
        r_half = float(np.corrcoef(ta, tb)[0, 1])
        rel = 2 * r_half / (1 + r_half) if r_half > 0 else None
        r_raw = float(np.corrcoef(ph, tt)[0, 1])
        out[dim] = {
            "r_raw": round(r_raw, 4),
            "split_half_reliability": round(rel, 4) if rel else None,
            "rho_hat": round(min(r_raw / np.sqrt(rel), 1.0), 4) if rel else None,
            "n_scenarios": len(dim_sids),
        }
    vals = [out[d]["rho_hat"] for d in DIMS if out[d]["rho_hat"] is not None]
    out["median_rho_hat"] = round(float(np.median(vals)), 4)
    return out


def retest_sets(ids, sids, seed: int, k: int) -> dict[str, list[str]]:
    rng = np.random.default_rng(seed)
    return {pid: [sids[j] for j in rng.choice(len(sids), size=k, replace=False)]
            for pid in ids}


def pooled_agreement(ids, answers, dists, design, bank_options, wobble,
                     a: float, b: float, seed: int, sets: dict) -> float:
    agree = total = 0
    for pid in ids:
        for sid in sets[pid]:
            ans = answers.get((pid, sid))
            if not ans:
                continue
            valid = bank_options[sid]
            r0 = noised_answer(ans, dists.get((pid, sid)), valid, wobble[pid],
                               a, b, seed, pid, sid, 0)
            r1 = noised_answer(ans, dists.get((pid, sid)), valid, wobble[pid],
                               a, b, seed, pid, sid, 1)
            agree += int(r0 == r1)
            total += 1
    return agree / total


def main(config_path: Path) -> dict:
    config = json.loads(config_path.read_text())
    design = {d["sid"]: d for d in json.loads(
        (truth_dir() / "bank" / "design.json").read_text())}
    bank = json.loads((MOBILITY_ROOT / "data" / "bank" / "scenarios.json").read_text())
    bank_options = {s["sid"]: {o["key"] for o in s["options"]} for s in bank}
    loaded_design = {sid: d for sid, d in design.items() if d["primary"]}
    stash = np.load(truth_dir() / "phi_m1.npz", allow_pickle=False)
    ids = [str(x) for x in stash["ids"]]
    phi = stash["phi"]
    ids_all, answers, dists = load_sweep(config["sweep"])
    assert ids_all == ids

    results: dict = {
        "experiment": config["experiment"],
        "date": _dt.date.today().isoformat(),
        "config": config,
    }

    # ---- step 1: Ruling-16 gate — full-bank pre-noise obedience -----------
    obedience_pre = obedience_table(ids, answers, loaded_design, phi)
    results["obedience_full_bank_pre_noise"] = obedience_pre
    results["ruling16_gate"] = {
        "bar": "median >= 0.5 (frozen M1 QA obedience bar on its native instrument)",
        "median": obedience_pre["median"],
        "pass": bool(obedience_pre["median"] >= 0.5),
    }
    if not results["ruling16_gate"]["pass"]:
        results["stopped"] = "Ruling 16: full-bank obedience median < 0.5 — nothing further ran"
        _write(results)
        return results
    if config.get("gate_only"):
        results["stopped"] = ("gate_only (Ruling 19g): gate PASSED but the pipeline stops "
                              "here — noise tuning and transmission await the owner's go")
        _write(results)
        return results

    # ---- step 2: confirmatory transmission (frozen section 5) -------------
    results["transmission_confirmatory"] = transmission_table(
        ids, answers, loaded_design, phi, int(config["split_half_seed"]))

    # ---- step 3: wobble mint + noise tuning -------------------------------
    lo, hi = config["wobble_range"]
    wobble = mint_wobble(int(config["wobble_seed"]), ids, lo, hi)
    np.savez(truth_dir() / "wobble_m1.npz",
             ids=np.array(ids), wobble=np.array([wobble[p] for p in ids]),
             seed=int(config["wobble_seed"]), lo=lo, hi=hi)

    sids = sorted(bank_options)
    sets = retest_sets(ids, sids, int(config["retest_seed"]), int(config["retest_k"]))
    b = float(config["b"])
    seed = int(config["noise_seed"])
    tuning = []
    best = None
    for a in config["a_grid"]:
        agreement = pooled_agreement(ids, answers, dists, design, bank_options,
                                     wobble, float(a), b, seed, sets)
        tuning.append({"a": a, "pooled_agreement": round(agreement, 4)})
        if best is None or abs(agreement - float(config["target"])) < abs(
                best[1] - float(config["target"])):
            best = (float(a), agreement)
    a_star, achieved = best
    results["noise_tuning"] = {
        "mechanism": "seeded flip to second choice; p_flip = min(0.5, wobble * a * exp(-b * gap))",
        "b_fixed": b, "a_grid": tuning, "a_chosen": a_star,
        "achieved_pooled_retest": round(achieved, 4),
        "frozen_band": [0.70, 0.90], "target": config["target"],
        "in_band": bool(0.70 <= achieved <= 0.90),
        "share_of_answers_with_distribution": round(
            sum(1 for v in dists.values() if v) / len(dists), 4),
    }

    # ---- step 4: official noised world + bars -----------------------------
    SYSTEM_RUNS.mkdir(parents=True, exist_ok=True)
    noised: dict = {}
    with (SYSTEM_RUNS / "answers_noised.jsonl").open("w", encoding="utf-8") as fh:
        for pid in ids:
            for sid in sids:
                ans = answers.get((pid, sid))
                if not ans:
                    continue
                out = noised_answer(ans, dists.get((pid, sid)), bank_options[sid],
                                    wobble[pid], a_star, b, seed, pid, sid, 0)
                noised[(pid, sid)] = out
                fh.write(json.dumps({"pid": pid, "sid": sid, "round": 0,
                                     "answer": out}) + "\n")

    # diversity on the official noised world
    code = {"A": 0, "B": 1, "C": 2}
    matrix = np.full((len(ids), len(sids)), -1, dtype=np.int8)
    for pi, pid in enumerate(ids):
        for si, sid in enumerate(sids):
            if (pid, sid) in noised:
                matrix[pi, si] = code[noised[(pid, sid)]]
    agreements = []
    max_agree = 0.0
    for i in range(len(ids)):
        eq = (matrix[i + 1:] == matrix[i]).mean(axis=1)
        if len(eq):
            agreements.append(eq)
            max_agree = max(max_agree, float(eq.max()))
    all_pairs = np.concatenate(agreements)

    scored = np.zeros((len(ids), len(sids)))
    for si, sid in enumerate(sids):
        scores = design[sid]["scores"]
        default = {"A": 1.0, "B": -1.0, "C": 0.0}
        for pi, pid in enumerate(ids):
            ans = noised.get((pid, sid))
            if ans is None:
                scored[pi, si] = np.nan
            elif design[sid]["primary"] is None:
                scored[pi, si] = default[ans]
            else:
                scored[pi, si] = scores[ans]
    col_mean = np.nanmean(scored, axis=0)
    inds = np.where(np.isnan(scored))
    scored[inds] = np.take(col_mean, inds[1])
    col_sd = scored.std(axis=0)
    keep = col_sd > 1e-9
    standardized = (scored[:, keep] - scored[:, keep].mean(axis=0)) / scored[:, keep].std(axis=0)
    eigvals = np.linalg.svd(standardized, compute_uv=False) ** 2 / (len(ids) - 1)
    participation = float(eigvals.sum() ** 2 / (eigvals ** 2).sum())

    results["diversity_official"] = {
        "max_pairwise_agreement": round(max_agree, 4),
        "median_pairwise_agreement": round(float(np.median(all_pairs)), 4),
        "participation_ratio": round(participation, 3),
        "n_constant_scenarios_dropped": int((~keep).sum()),
        "bars": {"no_pair_above": 0.95, "median_at_most": 0.80,
                 "participation_at_least": 4},
        "pass": bool(max_agree <= 0.95 and float(np.median(all_pairs)) <= 0.80
                     and participation >= 4),
    }

    # obedience on the official noised world (frozen QA formulation)
    results["obedience_official_noised"] = obedience_table(
        ids, noised, loaded_design, phi)

    # test-retest: round 0 vs independent round 1 on the fixed per-traveler sets
    stats = {"2opt": [0, 0], "3opt": [0, 0]}
    for pid in ids:
        for sid in sets[pid]:
            ans = answers.get((pid, sid))
            if not ans:
                continue
            valid = bank_options[sid]
            r0 = noised[(pid, sid)]
            r1 = noised_answer(ans, dists.get((pid, sid)), valid, wobble[pid],
                               a_star, b, seed, pid, sid, 1)
            bucket = "2opt" if len(valid) == 2 else "3opt"
            stats[bucket][0] += int(r0 == r1)
            stats[bucket][1] += 1
    pooled = (stats["2opt"][0] + stats["3opt"][0]) / (stats["2opt"][1] + stats["3opt"][1])
    results["test_retest"] = {
        "pooled_agreement": round(pooled, 4),
        "band": [0.70, 0.90],
        "pass": bool(0.70 <= pooled <= 0.90),
        "two_option": {"agreement": round(stats["2opt"][0] / stats["2opt"][1], 4),
                       "chance": 0.5, "n": stats["2opt"][1]},
        "three_option": {"agreement": round(stats["3opt"][0] / stats["3opt"][1], 4),
                         "chance": 0.333, "n": stats["3opt"][1]},
        "note": "fresh-session re-ask operationalized as an independent seeded "
                "application of the frozen layer (parent addendum convention)",
    }

    _write(results)
    return results


def _write(results: dict) -> None:
    (RESULTS_DIR / f"{results['experiment']}.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    out = main(Path(sys.argv[1]))
    summary = {"ruling16_gate": out["ruling16_gate"]}
    if "noise_tuning" in out:
        summary["noise"] = {k: out["noise_tuning"][k] for k in
                            ("a_chosen", "achieved_pooled_retest", "in_band")}
        summary["diversity_pass"] = out["diversity_official"]["pass"]
        summary["retest"] = out["test_retest"]["pooled_agreement"]
        summary["transmission_median"] = out["transmission_confirmatory"]["median_rho_hat"]
    print(json.dumps(summary, indent=2))
