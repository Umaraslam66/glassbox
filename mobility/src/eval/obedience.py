"""The frozen obedience comparison: reasoning-off vs reasoning-on (vault-side).

Runs the check frozen in PREREGISTRATION_MOBILITY.md section 8: 60 travelers
(first 60 of the seed-2141 shuffle), a fixed 24-scenario subset drawn from
the audited bank (seed 2142, stratified per the frozen strata; the drawn IDs
are written into the experiment config before any traveler answers), both
arms answering the identical grid at the frozen temperature, one round,
pre-noise. Frozen rule: Arm A (reasoning off) is confirmed iff its median
per-dimension obedience >= Arm B's median - 0.10 AND >= 0.5; anything else
is a stop-and-ask event.

Also computes the PRE-DECLARED EXPLORATORY transmission early-read from
Arm A (frozen section-5 estimator applied to this subset; labeled
exploratory — the confirmatory number needs the full-bank sweep).

Pre-noise answers are truth-side and stay in the vault run directory; only
aggregates leave. Run:
``python -m mobility.src.eval.obedience mobility/experiments/m1_obedience.json``
"""

from __future__ import annotations

import datetime as _dt
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from mobility.src.engine.or_client import ORClient
from mobility.src.eval.truth_vault import truth_dir
from mobility.src.travelers.sampler import DIMS

MOBILITY_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = MOBILITY_ROOT / "results"

#: Sub-stratification within the frozen strata (pre-declared in the log):
#: (block, primary, count). CRW/PRC slots come from the isolation sub-blocks.
DRAW_PLAN = (
    ("mode", "CRW", 4), ("mode", "PRC", 4),
    ("mode", "MOD", 2), ("mode", "VOT", 1), ("mode", "HAB", 1),
    ("dep_time", "SCH", 6),
    ("route", "VOT", 2), ("route", "SCH", 1), ("route", "HAB", 1),
    ("policy", "PRC", 1), ("policy", "MOD", 1),
)

_ANSWER_SYSTEM_SUFFIX = (
    "\n\nAnswer every question fully in character, based on how this person "
    "would actually choose. Reply with exactly one letter and nothing else."
)


def load_design() -> dict:
    return {d["sid"]: d for d in json.loads(
        (truth_dir() / "bank" / "design.json").read_text())}


def load_public_bank() -> dict:
    return {s["sid"]: s for s in json.loads(
        (MOBILITY_ROOT / "data" / "bank" / "scenarios.json").read_text())}


def draw_selection(config: dict, design: dict) -> tuple[list[str], list[str]]:
    rng_t = np.random.default_rng(int(config["travelers_seed"]))
    all_pids = [f"t{i:04d}" for i in range(400)]
    pids = [all_pids[j] for j in rng_t.permutation(400)[: int(config["n_travelers"])]]

    rng_s = np.random.default_rng(int(config["scenarios_seed"]))
    sids: list[str] = []
    for block, primary, count in DRAW_PLAN:
        if block == "mode" and primary == "CRW":
            pool = sorted(sid for sid, d in design.items() if "crw_iso" in d["tags"])
        elif block == "mode" and primary == "PRC":
            pool = sorted(sid for sid, d in design.items() if "prc_iso" in d["tags"])
        else:
            pool = sorted(sid for sid, d in design.items()
                          if d["block"] == block and d["primary"] == primary)
        take = rng_s.choice(len(pool), size=count, replace=False)
        sids += [pool[k] for k in sorted(take)]
    assert len(sids) == 24 and len(set(sids)) == 24
    return pids, sids


def run_arm(client: ORClient, arm: str, pids: list[str], sids: list[str],
            cards: dict, bank: dict, temperature: float, concurrency: int,
            out_path: Path) -> list[dict]:
    reasoning_on = arm == "on"
    jobs = [(pid, sid) for pid in pids for sid in sids]

    def work(job: tuple[str, str]) -> dict:
        pid, sid = job
        record = client.call(
            cards[pid] + _ANSWER_SYSTEM_SUFFIX, bank[sid]["text"],
            temperature=temperature,
            max_tokens=2048 if reasoning_on else 8,
            reasoning_on=reasoning_on,
            seed_key=f"{pid}:{sid}:arm_{arm}:r0")
        letter = (record["text"] or "?").strip()[:1].upper()
        valid = {o["key"] for o in bank[sid]["options"]}
        return {"pid": pid, "sid": sid, "arm": arm,
                "answer": letter if letter in valid else None}

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        rows = list(pool.map(work, jobs))
    with out_path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return rows


def obedience_by_dim(rows: list[dict], pids: list[str], sids: list[str],
                     design: dict, phi: np.ndarray, ids: list[str]) -> dict:
    """Frozen definition: r across travelers between phi_d and mean designed
    score of the chosen options on scenarios loading on d (scores already
    carry the keying, so no extra sign flip)."""
    answer = {(r["pid"], r["sid"]): r["answer"] for r in rows}
    out = {}
    for di, dim in enumerate(DIMS):
        dim_sids = [sid for sid in sids if design[sid]["primary"] == dim]
        tendencies, phis = [], []
        for pid in pids:
            scores = [design[sid]["scores"][answer[(pid, sid)]]
                      for sid in dim_sids if answer.get((pid, sid))]
            if scores:
                tendencies.append(float(np.mean(scores)))
                phis.append(phi[ids.index(pid), di])
        r = float(np.corrcoef(phis, tendencies)[0, 1]) if len(set(tendencies)) > 1 else 0.0
        out[dim] = {"r": round(r, 4), "n_scenarios": len(dim_sids),
                    "n_travelers": len(tendencies)}
    return out


def split_half_transmission(rows: list[dict], pids: list[str], sids: list[str],
                            design: dict, phi: np.ndarray, ids: list[str]) -> dict:
    """EXPLORATORY early-read of the frozen section-5 estimator on this subset."""
    answer = {(r["pid"], r["sid"]): r["answer"] for r in rows}
    out = {}
    for di, dim in enumerate(DIMS):
        dim_sids = sorted(sid for sid in sids if design[sid]["primary"] == dim)
        if len(dim_sids) < 2:
            out[dim] = {"rho_hat": None, "note": "too few scenarios in subset"}
            continue
        half_a, half_b = dim_sids[0::2], dim_sids[1::2]

        def tendency(pid: str, subset: list[str]) -> float | None:
            scores = [design[sid]["scores"][answer[(pid, sid)]]
                      for sid in subset if answer.get((pid, sid))]
            return float(np.mean(scores)) if scores else None

        rows_t = [(tendency(p, half_a), tendency(p, half_b),
                   tendency(p, dim_sids), phi[ids.index(p), di]) for p in pids]
        rows_t = [t for t in rows_t if None not in t[:3]]
        ta, tb, tt, ph = map(np.array, zip(*rows_t))
        if len(set(ta)) < 2 or len(set(tb)) < 2 or len(set(tt)) < 2:
            out[dim] = {"rho_hat": None, "note": "degenerate tendencies"}
            continue
        r_half = float(np.corrcoef(ta, tb)[0, 1])
        rel = max(2 * r_half / (1 + r_half), 1e-6) if r_half > 0 else None
        r_raw = float(np.corrcoef(ph, tt)[0, 1])
        rho = min(r_raw / np.sqrt(rel), 1.0) if rel else None
        out[dim] = {"r_raw": round(r_raw, 4),
                    "split_half_reliability": round(rel, 4) if rel else None,
                    "rho_hat": round(rho, 4) if rho is not None else None,
                    "n_scenarios": len(dim_sids)}
    return out


def main(config_path: Path) -> dict:
    config = json.loads(config_path.read_text())
    vault = truth_dir()
    run_dir = vault / "runs" / config["experiment"]
    run_dir.mkdir(parents=True, exist_ok=True)

    design = load_design()
    bank = load_public_bank()
    stash = np.load(vault / "phi_m1.npz", allow_pickle=False)
    ids = [str(x) for x in stash["ids"]]
    phi = stash["phi"]

    if "drawn_travelers" not in config:
        pids, sids = draw_selection(config, design)
        config["drawn_travelers"] = pids
        config["drawn_scenarios"] = sids
        config_path.write_text(json.dumps(config, indent=2) + "\n")
    pids, sids = config["drawn_travelers"], config["drawn_scenarios"]

    card_dir = vault / "runs" / "m1_cards" / "cards"
    cards = {pid: json.loads((card_dir / f"{pid}.json").read_text())["card"]
             for pid in pids}

    client = ORClient(cache_dir=run_dir / "cache")
    arms = {}
    for arm in ("off", "on"):
        rows = run_arm(client, arm, pids, sids, cards, bank,
                       float(config["temperature"]), int(config["concurrency"]),
                       run_dir / f"answers_arm_{arm}.jsonl")
        arms[arm] = {
            "obedience": obedience_by_dim(rows, pids, sids, design, phi, ids),
            "n_invalid_answers": sum(1 for r in rows if r["answer"] is None),
        }

    med_off = float(np.median([v["r"] for v in arms["off"]["obedience"].values()]))
    med_on = float(np.median([v["r"] for v in arms["on"]["obedience"].values()]))
    confirmed = (med_off >= med_on - 0.10) and (med_off >= 0.5)

    off_rows = [json.loads(line) for line in
                (run_dir / "answers_arm_off.jsonl").read_text().splitlines()]
    result = {
        "experiment": config["experiment"],
        "date": _dt.date.today().isoformat(),
        "frozen_rule": "off confirmed iff median_off >= median_on - 0.10 AND median_off >= 0.5",
        "median_obedience_off": round(med_off, 4),
        "median_obedience_on": round(med_on, 4),
        "verdict": "CONFIRMED reasoning-off" if confirmed else "STOP-AND-ASK",
        "arms": arms,
        "exploratory_transmission_early_read": {
            "label": "EXPLORATORY (pre-declared in PROJECT_LOG_MOBILITY.md before data existed): "
                     "frozen section-5 estimator on the 60x24 Arm-A subset, not the full bank",
            "by_dim": split_half_transmission(off_rows, pids, sids, design, phi, ids),
        },
        "client": client.stats(),
        "n_travelers": len(pids),
        "n_scenarios": len(sids),
    }
    (RESULTS_DIR / f"{config['experiment']}.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    res = main(Path(sys.argv[1]))
    print(json.dumps({k: res[k] for k in
                      ("median_obedience_off", "median_obedience_on", "verdict",
                       "client")}, indent=2))
