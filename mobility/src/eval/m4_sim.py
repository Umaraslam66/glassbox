"""M4 day-loop — LLM travelers adjusting route + departure time (vault-side).

The world and every design choice were pre-declared at M3 close (commit
4f1b741): two routes (A = 18 min free-flow through the calibrated
bottleneck; B = steady 26 min), day 1 mechanical-naive (route A, depart
PAT − 18), days 2..T chosen by the traveler via the frozen answering config
(qwen reasoning-off, T 0.7, cached, resumable). Menu: ten single-letter
options — A–E route A at {−15, −5, 0, +5, +15} minutes vs yesterday's
departure, F–J route B likewise; the same-route zero-shift option is
labeled "as yesterday". Memory shown: the traveler's own last 3 days.

Raw prompts/completions stay vault-side (they carry the card). The
system-visible trajectory log gets exactly (pid, day, route, dep_min,
travel_min, arrive_min).

Run: ``python -m mobility.src.eval.m4_sim mobility/experiments/m4_pilot.json``
"""

from __future__ import annotations

import datetime as _dt
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from mobility.src.engine.bottleneck import (FREE_FLOW_A, ROUTE_B_TIME,
                                            tau_interpolator,
                                            vickrey_travel_times)
from mobility.src.engine.or_client import ORClient
from mobility.src.eval.truth_vault import truth_dir

MOBILITY_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = MOBILITY_ROOT / "results"

SHIFTS = [-15, -5, 0, 5, 15]
LETTERS = "ABCDEFGHIJ"

_SYSTEM_SUFFIX = (
    "\n\nDecide fully in character, based on how this person would actually "
    "plan tomorrow's commute. Reply with exactly one letter and nothing else."
)


def clock(minutes: float) -> str:
    return f"{int(minutes // 60):02d}:{int(round(minutes % 60)):02d}"


def build_menu(prev_route: str, prev_dep: float) -> tuple[str, list[tuple[str, float]]]:
    """Menu text + [(route, departure)] in option order A..J."""
    plans, lines = [], []
    for base, route in (("A", "A"), ("F", "B")):
        rname = ("the usual main road (fastest when clear)" if route == "A"
                 else "the ring road (longer but steady)")
        for i, shift in enumerate(SHIFTS):
            letter = LETTERS[LETTERS.index(base) + i]
            dep = prev_dep + shift
            tag = " — exactly as yesterday" if route == prev_route and shift == 0 else ""
            lines.append(f"{letter}) Route {route}, {rname}, leave at "
                         f"{clock(dep)}{tag}")
            plans.append((route, dep))
    return "\n".join(lines), plans


def build_prompt(pat: float, memory: list[dict], menu_text: str) -> str:
    mem_lines = []
    for m in memory:
        verdict = ("on time" if m["arrive"] <= pat else
                   f"{int(round(m['arrive'] - pat))} min LATE")
        if m["arrive"] < pat - 1:
            verdict = f"{int(round(pat - m['arrive']))} min early"
        mem_lines.append(
            f"- Day {m['day']}: route {m['route']}, left {clock(m['dep'])}, "
            f"drive took {int(round(m['travel']))} min, arrived "
            f"{clock(m['arrive'])} ({verdict})")
    return (
        f"Your commute. You need to be at work by {clock(pat)}.\n"
        f"Your recent days:\n" + "\n".join(mem_lines) + "\n\n"
        f"Tomorrow morning, which do you choose?\n{menu_text}\n"
        f"Answer with exactly one letter."
    )


def main(config_path: Path) -> dict:
    config = json.loads(config_path.read_text())
    vault = truth_dir()
    world = np.load(vault / "m4_world.npz", allow_pickle=False)
    pids = [str(x) for x in world["pids"]][: config["n_travelers"]]
    pat = {p: float(v) for p, v in zip([str(x) for x in world["pids"]],
                                       world["pat"])}
    calib = json.loads((RESULTS_DIR / "m4_capacity_calibration.json").read_text())
    capacity = float(calib["chosen_capacity_per_min"])

    card_dir = vault / "runs" / config["cards_run"] / "cards"
    cards = {p: json.loads((card_dir / f"{p}.json").read_text())["card"]
             for p in pids}

    run_dir = vault / "runs" / config["experiment"]
    run_dir.mkdir(parents=True, exist_ok=True)
    client = ORClient(cache_dir=run_dir / "cache")
    sys_dir = MOBILITY_ROOT / "data" / "runs" / config["experiment"]
    sys_dir.mkdir(parents=True, exist_ok=True)

    n_days = int(config["n_days"])
    state = {p: {"route": "A", "dep": pat[p] - FREE_FLOW_A} for p in pids}
    memory: dict[str, list[dict]] = {p: [] for p in pids}
    day_profiles: list[dict] = []
    trajectories: list[dict] = []
    invalid_answers = 0
    day_wallclock: list[float] = []

    for day in range(1, n_days + 1):
        t0 = time.time()
        if day > 1:
            menu_cache = {p: build_menu(state[p]["route"], state[p]["dep"])
                          for p in pids}

            def ask(p: str) -> tuple[str, int]:
                menu_text, plans = menu_cache[p]
                prompt = build_prompt(pat[p], memory[p][-int(config["memory_days"]):],
                                      menu_text)
                record = client.call(
                    cards[p] + _SYSTEM_SUFFIX, prompt,
                    temperature=float(config["temperature"]), max_tokens=8,
                    reasoning_on=False, seed_key=f"{p}:day{day}:m4")
                letter = (record["text"] or "?").strip()[:1].upper()
                if letter in LETTERS[: len(plans)]:
                    return p, LETTERS.index(letter)
                return p, -1

            with ThreadPoolExecutor(max_workers=int(config["concurrency"])) as pool:
                for p, choice in pool.map(ask, pids):
                    if choice < 0:
                        invalid_answers += 1
                        choice = LETTERS.index("C") if state[p]["route"] == "A" \
                            else LETTERS.index("H")  # same as yesterday
                    route, dep = menu_cache[p][1][choice]
                    state[p] = {"route": route, "dep": dep}

        on_a = [p for p in pids if state[p]["route"] == "A"]
        deps_a = np.array([state[p]["dep"] for p in on_a])
        travels_a = vickrey_travel_times(deps_a, capacity)
        travel = {p: float(t) for p, t in zip(on_a, travels_a)}
        for p in pids:
            if p not in travel:
                travel[p] = ROUTE_B_TIME
        day_profiles.append({
            "A": tau_interpolator(deps_a, travels_a),
            "B": tau_interpolator(np.array([0.0]), np.array([ROUTE_B_TIME])),
        })
        for p in pids:
            arrive = state[p]["dep"] + travel[p]
            memory[p].append({"day": day, "route": state[p]["route"],
                              "dep": state[p]["dep"], "travel": travel[p],
                              "arrive": arrive})
            trajectories.append({"pid": p, "day": day,
                                 "route": state[p]["route"],
                                 "dep_min": round(state[p]["dep"], 1),
                                 "travel_min": round(travel[p], 2),
                                 "arrive_min": round(arrive, 2)})
        day_wallclock.append(round(time.time() - t0, 1))

    with (sys_dir / "trajectories.jsonl").open("w", encoding="utf-8") as fh:
        for row in trajectories:
            fh.write(json.dumps(row) + "\n")

    stats = client.stats()
    summary = {
        "experiment": config["experiment"],
        "date": _dt.date.today().isoformat(),
        "config": config,
        "capacity_per_min": capacity,
        "n_travelers": len(pids), "n_days": n_days,
        "llm_calls_expected": len(pids) * (n_days - 1),
        "invalid_answers_defaulted": invalid_answers,
        "day_wallclock_s": day_wallclock,
        "client": stats,
        "note": "pilot is mechanics-only per the pre-declaration: no bar "
                "grading on pilot data",
    }
    (RESULTS_DIR / f"{config['experiment']}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    out = main(Path(sys.argv[1]))
    print(json.dumps({k: out[k] for k in
                      ("llm_calls_expected", "invalid_answers_defaulted",
                       "day_wallclock_s", "client")}, indent=2))
