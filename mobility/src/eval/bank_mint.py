"""Bank mint + independent audit (vault-side).

Deposits the frozen bank's design views (primary dimensions, option scores,
tags — planted truth) into the vault, writes the public bank (text +
attributes only) to the system-visible side, and runs the two-part audit the
frozen prereg requires before any traveler answers:

1. Mechanical audit: block counts against the frozen design table, vehicle-map
   discipline (exactly one primary per non-distractor, both score signs
   present), the isolation sub-blocks' hold-everything-else-constant property,
   zero cost variation in the departure-time block, equal costs in the route
   block, attribute ranges, numberless distractors.
2. Independent blind audit: Gemini flash-lite (system-side family, never the
   traveler model) classifies each PUBLIC scenario text by the construct it
   appears designed to reveal; the grader compares those blind labels to the
   designed primaries. Gemini never sees the design.

Run: ``python -m mobility.src.eval.bank_mint mobility/experiments/m1_bank.json``
"""

from __future__ import annotations

import datetime as _dt
import json
import re
import sys
import time
from pathlib import Path

from mobility.src.eval.truth_vault import truth_dir
from mobility.src.scenarios.bank import design_view, generate_bank, public_view

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.llm_client import GeminiClient, load_dotenv_if_present  # noqa: E402

MOBILITY_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = MOBILITY_ROOT / "results"
PUBLIC_BANK = MOBILITY_ROOT / "data" / "bank" / "scenarios.json"

EXPECTED_COUNTS = {
    ("mode", "VOT"): 14, ("mode", "MOD"): 12, ("mode", "CRW"): 10,
    ("mode", "PRC"): 10, ("mode", "HAB"): 4,
    ("dep_time", "SCH"): 30,
    ("route", "VOT"): 13, ("route", "SCH"): 8, ("route", "HAB"): 4,
    ("policy", "PRC"): 15, ("policy", "MOD"): 6, ("policy", "VOT"): 4,
    ("distractor", None): 10,
}

#: Blind labels the auditor may answer, mapped to design dimensions.
AUDIT_LABELS = {"TIME": "VOT", "SCHED": "SCH", "CARPREF": "MOD",
                "CROWD": "CRW", "PRICE": "PRC", "HABIT": "HAB", "NONE": None}

_AUDIT_PROMPT = (
    "A traveler will answer this choice scenario. Which ONE construct is the "
    "scenario mainly designed to reveal about them?\n"
    "TIME = willingness to pay money to save travel time\n"
    "SCHED = attitude to lateness risk, buffers, punctuality\n"
    "CARPREF = preference for car versus transit or bike as such, e.g. when the "
    "options are otherwise close to equal\n"
    "CROWD = tolerance of crowded vehicles\n"
    "PRICE = sensitivity to fares, fees and charges\n"
    "HABIT = sticking with the usual versus trying alternatives\n"
    "NONE = everyday lifestyle choice, none of the above\n"
    "Reply with exactly one word from the list.\n\nScenario:\n"
)


def mechanical_audit(bank: list[dict]) -> dict:
    problems: list[str] = []

    counts: dict = {}
    for s in bank:
        counts[(s["block"], s["primary"])] = counts.get((s["block"], s["primary"]), 0) + 1
    if counts != EXPECTED_COUNTS:
        problems.append(f"block/primary counts off: {counts} != {EXPECTED_COUNTS}")

    for s in bank:
        scores = list(s["scores"].values())
        if s["block"] == "distractor":
            if s["primary"] is not None or any(scores):
                problems.append(f"{s['sid']}: distractor carries design load")
            if re.search(r"\d", s["text"]):
                problems.append(f"{s['sid']}: distractor contains numbers")
            continue
        if s["primary"] is None:
            problems.append(f"{s['sid']}: missing primary")
        if +1 not in scores or -1 not in scores:
            problems.append(f"{s['sid']}: scores must include both signs")
        for opt in s["options"]:
            cost = opt["attrs"].get("cost_eur")
            if cost is not None and not (0 <= cost <= 12.01):
                problems.append(f"{s['sid']}: cost {cost} outside design range")

    # departure-time block: zero cost variation
    for s in bank:
        if s["block"] == "dep_time":
            costs = {opt["attrs"].get("cost_eur", 0.0) for opt in s["options"]}
            if costs != {0.0}:
                problems.append(f"{s['sid']}: dep_time block must have no cost variation")
        if s["block"] == "route":
            costs = {opt["attrs"].get("cost_eur", 0.0) for opt in s["options"]}
            if len(costs) != 1:
                problems.append(f"{s['sid']}: route block must have equal costs")

    # isolation sub-blocks: everything constant across scenarios except the
    # isolated attribute
    crw = [s for s in bank if "crw_iso" in s["tags"]]
    fixed = {(o["attrs"].get("time_min"), o["attrs"].get("cost_eur"))
             for s in crw for o in s["options"]}
    if fixed != {(26, 2.80), (26, 5.50)}:
        problems.append(f"crw_iso: trip not held constant, saw {fixed}")
    if len({s["options"][0]["attrs"]["crowding"] for s in crw}) < 3:
        problems.append("crw_iso: crowding barely varies")

    prc = [s for s in bank if "prc_iso" in s["tags"]]
    if {o["attrs"]["time_min"] for s in prc for o in s["options"]} != {24}:
        problems.append("prc_iso: times not held constant")
    if {o["attrs"].get("mode") for s in prc for o in s["options"]} != {"transit"}:
        problems.append("prc_iso: both options must be the same mode (no MOD loading)")
    savings = sorted(round(s["options"][1]["attrs"]["cost_eur"]
                           - s["options"][0]["attrs"]["cost_eur"], 2) for s in prc)
    if len(set(savings)) != 10 or savings[0] > 0.35 or savings[-1] < 2.0:
        problems.append(f"prc_iso: savings grid must span small-to-large, saw {savings}")
    if any(s["options"][0]["attrs"].get("friction") != 1
           or s["options"][1]["attrs"].get("friction") != 0 for s in prc):
        problems.append("prc_iso: discounted option must carry the friction, full-price none")

    return {"n_problems": len(problems), "problems": problems}


def blind_audit(bank: list[dict], temperature: float = 0.0) -> dict:
    load_dotenv_if_present(REPO_ROOT / ".env")
    client = GeminiClient()
    rows = []
    for s in bank:
        pub = public_view(s)
        for attempt in range(4):
            try:
                resp = client.generate(None, _AUDIT_PROMPT + pub["text"],
                                       temperature=temperature, max_tokens=8)
                break
            except RuntimeError:
                time.sleep(5 * (attempt + 1))
        else:
            rows.append({"sid": s["sid"], "label": "ERROR", "guess": "ERROR"})
            continue
        word = (resp.text or "").strip().upper().split()[0] if resp.text else ""
        word = word if word in AUDIT_LABELS else "NONE"
        rows.append({"sid": s["sid"], "label": word,
                     "guess": AUDIT_LABELS.get(word)})

    by_sid = {r["sid"]: r for r in rows}
    per_block: dict = {}
    for s in bank:
        guess = by_sid[s["sid"]]["guess"]
        hit = guess == s["primary"]
        b = per_block.setdefault(s["block"], {"n": 0, "agree": 0, "misses": []})
        b["n"] += 1
        b["agree"] += int(hit)
        if not hit:
            b["misses"].append(f"{s['sid']}: designed {s['primary']} read as {guess}")
    for b in per_block.values():
        b["agreement"] = round(b["agree"] / b["n"], 3)
    overall = sum(b["agree"] for b in per_block.values()) / len(bank)
    return {"per_block": per_block, "overall_agreement": round(overall, 3)}


def mint_bank(config_path: Path) -> dict:
    config = json.loads(config_path.read_text())
    bank = generate_bank(int(config["seed"]))

    vault = truth_dir()
    (vault / "bank").mkdir(parents=True, exist_ok=True)
    (vault / "bank" / "design.json").write_text(
        json.dumps([design_view(s) for s in bank], indent=1), encoding="utf-8")
    PUBLIC_BANK.parent.mkdir(parents=True, exist_ok=True)
    PUBLIC_BANK.write_text(
        json.dumps([public_view(s) for s in bank], indent=1), encoding="utf-8")

    report = {
        "experiment": config["experiment"],
        "date": _dt.date.today().isoformat(),
        "seed": config["seed"],
        "n_scenarios": len(bank),
        "mechanical_audit": mechanical_audit(bank),
        "blind_audit": blind_audit(bank),
        "config": config,
    }
    (RESULTS_DIR / f"{config['experiment']}_audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    out = mint_bank(Path(sys.argv[1]))
    print(json.dumps({
        "n_scenarios": out["n_scenarios"],
        "mechanical_problems": out["mechanical_audit"]["n_problems"],
        "blind_overall_agreement": out["blind_audit"]["overall_agreement"],
        "blind_per_block": {k: v["agreement"] for k, v in out["blind_audit"]["per_block"].items()},
    }, indent=2))
