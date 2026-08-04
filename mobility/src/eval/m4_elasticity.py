"""M4 elasticity experiments — the three frozen re-ask experiments.

Frozen §5: arc elasticities on population aggregates for (1) uniform
travel-cost +10%, (2) transit fare −20%, (3) a cordon charge on car
commutes — each the same population re-asked under changed attributes with
everything else fixed. Operationalization pre-declared at commit e837cbf:
scenario sets limited to text-visible money, variants built by exact
substitution of the generator's 'EUR x.xx' strings (two-phase, collision
safe), both arms pre-noise, the 300 M4 travelers, frozen answering config,
$1.00 wall-to-wall cap shared with the day-loop.

Run: ``python -m mobility.src.eval.m4_elasticity``
"""

from __future__ import annotations

import datetime as _dt
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from mobility.src.engine.or_client import ORClient
from mobility.src.eval.truth_vault import truth_dir

MOBILITY_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = MOBILITY_ROOT / "results"
WALL_TO_WALL_CAP = 1.00
CORDON_EUR = 4.00

_SYSTEM_SUFFIX = (
    "\n\nAnswer every question fully in character, based on how this person "
    "would actually choose. Reply with exactly one letter and nothing else."
)

CAVEAT = ("By-construction caveat (frozen section 6, verbatim): the "
          "population's planted VOT/PRC distributions partly determine these "
          "numbers; the interesting quantity is the distortion LLM mediation "
          "introduces relative to the structurally implied elasticities.")


def rewrite(scenario: dict, changes: dict[str, float],
            cordon: bool = False) -> dict:
    """Variant scenario: option key -> new cost. Two-phase substitution so a
    replacement can never be re-matched by a later needle."""
    text = scenario["text"]
    options = json.loads(json.dumps(scenario["options"]))
    marks: dict[str, str] = {}
    seen: dict[str, str] = {}
    for k, opt in enumerate(options):
        key = opt["key"]
        if key not in changes:
            continue
        old, new = opt["attrs"]["cost_eur"], changes[key]
        needle = f"EUR {old:.2f}"
        if needle in seen:
            # equal costs across options replace together on first pass;
            # equal transforms of equal inputs give equal outputs
            assert marks[seen[needle]] == f"EUR {new:.2f}", scenario["sid"]
            mark = seen[needle]
        else:
            mark = f"@@COST{k}@@"
            assert needle in text, f"{scenario['sid']}: {needle} not in text"
            text = text.replace(needle, mark)
            seen[needle] = mark
        opt["label"] = opt["label"].replace(needle, mark)
        replacement = f"EUR {new:.2f}"
        if cordon and opt["attrs"].get("mode") == "car":
            text = text.replace(f"{mark} fuel and parking",
                                f"{mark} fuel, parking and the new EUR "
                                f"{CORDON_EUR:.2f} city charge")
        marks[mark] = replacement
        opt["attrs"]["cost_eur"] = round(new, 2)
    for mark, replacement in marks.items():
        text = text.replace(mark, replacement)
        for opt in options:
            opt["label"] = opt["label"].replace(mark, replacement)
    return {**scenario, "text": text, "options": options}


def build_variants(bank: list[dict]) -> dict[str, list[dict]]:
    by_prefix = lambda p: [s for s in bank if s["sid"].startswith(p)]
    exp1, exp2, exp3 = [], [], []
    for s in by_prefix("mode_vot") + by_prefix("mode_mod") + \
            by_prefix("mode_prc") + by_prefix("mode_hab"):
        exp1.append(rewrite(s, {o["key"]: o["attrs"]["cost_eur"] * 1.10
                                for o in s["options"]
                                if o["attrs"]["cost_eur"] > 0}))
    for s in by_prefix("mode_vot") + by_prefix("mode_prc"):
        exp2.append(rewrite(s, {o["key"]: o["attrs"]["cost_eur"] * 0.80
                                for o in s["options"]
                                if o["attrs"]["cost_eur"] > 0}))
    for s in by_prefix("mode_mod"):
        exp2.append(rewrite(s, {o["key"]: o["attrs"]["cost_eur"] * 0.80
                                for o in s["options"]
                                if o["attrs"].get("mode") == "transit"}))
        exp3.append(rewrite(s, {o["key"]: o["attrs"]["cost_eur"] + CORDON_EUR
                                for o in s["options"]
                                if o["attrs"].get("mode") == "car"}, cordon=True))
    return {"cost_plus10": exp1, "fare_minus20": exp2, "cordon": exp3}


def shares(answers: dict, pids: list[str], scenarios: list[dict],
           mode_only: bool = True) -> dict:
    """Aggregate mode shares over the mode_mod scenarios (the multi-mode
    instrument) and per-sub-block option-A shares (context)."""
    mode_counts: dict[str, int] = {}
    n_mode = 0
    suba: dict[str, list[int]] = {}
    for s in scenarios:
        prefix = s["sid"].rsplit("_", 1)[0]
        modes = {o["key"]: o["attrs"].get("mode") for o in s["options"]}
        for p in pids:
            a = answers.get((p, s["sid"]))
            if not a:
                continue
            suba.setdefault(prefix, [0, 0])
            suba[prefix][0] += int(a == "A")
            suba[prefix][1] += 1
            if s["sid"].startswith("mode_mod"):
                mode_counts[modes[a]] = mode_counts.get(modes[a], 0) + 1
                n_mode += 1
    out = {"mode_shares_mode_mod": {m: round(c / n_mode, 4)
                                    for m, c in sorted(mode_counts.items())}
           if n_mode else {},
           "option_A_share_by_subblock": {k: round(v[0] / v[1], 4)
                                          for k, v in sorted(suba.items())}}
    return out


def arc(q0: float, q1: float, dp_over_pbar: float) -> float:
    return ((q1 - q0) / ((q1 + q0) / 2)) / dp_over_pbar


def main() -> dict:
    bank = json.loads((MOBILITY_ROOT / "data" / "bank" / "scenarios.json").read_text())
    variants = build_variants(bank)
    world = np.load(truth_dir() / "m4_world.npz", allow_pickle=False)
    pids = [str(x) for x in world["pids"]]
    card_dir = truth_dir() / "runs" / "m1_cards_r2" / "cards"
    cards = {p: json.loads((card_dir / f"{p}.json").read_text())["card"]
             for p in pids}

    # baseline arm: cached pre-noise sweep answers, same travelers/scenarios
    base: dict = {}
    shard_dir = truth_dir() / "runs" / "m1_sweep_r2" / "answers"
    for p in pids:
        for line in (shard_dir / f"{p}.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r["answer"]:
                base[(p, r["sid"])] = r["answer"]

    day_loop_cost = json.loads(
        (RESULTS_DIR / "m4_full_summary.json").read_text())["client"]["cost_usd"]
    run_dir = truth_dir() / "runs" / "m4_elasticity"
    run_dir.mkdir(parents=True, exist_ok=True)
    client = ORClient(cache_dir=run_dir / "cache")

    results: dict = {"experiment": "m4_elasticity",
                     "date": _dt.date.today().isoformat(),
                     "caveat": CAVEAT,
                     "role": "REPORTED beside published ranges; no bar",
                     "arms": {}}

    for exp, scenarios in variants.items():
        valid = {s["sid"]: {o["key"] for o in s["options"]} for s in scenarios}
        jobs = [(p, s) for p in pids for s in scenarios]

        def ask(job):
            p, s = job
            if client.stats()["cost_usd"] + day_loop_cost > WALL_TO_WALL_CAP:
                raise SystemExit(
                    f"STOP: wall-to-wall cap {WALL_TO_WALL_CAP} would be "
                    f"exceeded (day loop {day_loop_cost} + elasticity "
                    f"{client.stats()['cost_usd']})")
            record = client.call(
                cards[p] + _SYSTEM_SUFFIX, s["text"], temperature=0.7,
                max_tokens=8, reasoning_on=False,
                seed_key=f"{p}:{s['sid']}:{exp}")
            letter = (record["text"] or "?").strip()[:1].upper()
            return (p, s["sid"]), letter if letter in valid[s["sid"]] else None

        answers: dict = {}
        invalid = 0
        with ThreadPoolExecutor(max_workers=12) as pool:
            for key, letter in pool.map(ask, jobs):
                if letter is None:
                    invalid += 1
                else:
                    answers[key] = letter

        base_here = {k: v for k, v in base.items()
                     if k[1] in valid and k[0] in set(pids)}
        results["arms"][exp] = {
            "n_scenarios": len(scenarios), "n_calls": len(jobs),
            "invalid": invalid,
            "baseline": shares(base_here, pids, scenarios),
            "changed": shares(answers, pids, scenarios),
        }

    # ---- arcs (pre-declared formulas) -------------------------------------
    a = results["arms"]
    e1b = a["cost_plus10"]["baseline"]["mode_shares_mode_mod"]
    e1c = a["cost_plus10"]["changed"]["mode_shares_mode_mod"]
    e2b = a["fare_minus20"]["baseline"]["mode_shares_mode_mod"]
    e2c = a["fare_minus20"]["changed"]["mode_shares_mode_mod"]
    e3b = a["cordon"]["baseline"]["mode_shares_mode_mod"]
    e3c = a["cordon"]["changed"]["mode_shares_mode_mod"]
    mode_mod = [s for s in bank if s["sid"].startswith("mode_mod")]
    car_costs = [o["attrs"]["cost_eur"] for s in mode_mod
                 for o in s["options"] if o["attrs"].get("mode") == "car"]
    pbar_cordon = (2 * np.mean(car_costs) + CORDON_EUR) / 2
    results["elasticities"] = {
        "cost_plus10": {
            "car_share": [e1b.get("car"), e1c.get("car")],
            "transit_share": [e1b.get("transit"), e1c.get("transit")],
            "arc_car": round(arc(e1b["car"], e1c["car"], 0.10 / 1.05), 3),
            "arc_transit": round(arc(e1b["transit"], e1c["transit"], 0.10 / 1.05), 3),
            "published_car_cost": "-0.1 to -0.4 (Goodwin, Dargay & Hanly 2004)",
        },
        "fare_minus20": {
            "transit_share": [e2b.get("transit"), e2c.get("transit")],
            "arc_transit_fare": round(arc(e2b["transit"], e2c["transit"],
                                          -0.20 / 0.90), 3),
            "published_fare": "-0.2 to -0.5 (TRL 2004; Litman) — supplies the "
                              "fare number Ruling 28 recorded as missing "
                              "statically",
        },
        "cordon": {
            "car_share": [e3b.get("car"), e3c.get("car")],
            "car_share_reduction_pct": round(
                100 * (e3c["car"] - e3b["car"]) / e3b["car"], 1),
            "arc_car_vs_cordon": round(arc(e3b["car"], e3c["car"],
                                           CORDON_EUR / pbar_cordon), 3),
            "published_reduction": "-10% to -35% car trips "
                                   "(London/Stockholm program evaluations)",
        },
    }
    results["client"] = client.stats()
    results["wall_to_wall_usd"] = round(client.stats()["cost_usd"] + day_loop_cost, 4)
    (RESULTS_DIR / "m4_elasticity.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


if __name__ == "__main__":
    out = main()
    print(json.dumps({"elasticities": out["elasticities"],
                      "wall_to_wall_usd": out["wall_to_wall_usd"],
                      "client": out["client"]}, indent=2))
