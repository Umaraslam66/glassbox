"""Ruling 29 — the frozen LLM-given-profile-only baseline (Gemini, capped $1).

The fourth frozen §5 baseline: an LLM sees ONLY the public profile and the
scenario text as shown to travelers — never a card, never a choice — and
estimates the choice shares of people with that profile. Scored on the
identical Brier cells and p_true as every other M3 baseline; the result is
appended to results/m3_calibration.json beside the frozen baseline list.

Operationalization pre-declared in the project log (commit 4f1b741):
temperature 0, one call per (unique held-out profile, held-out
non-distractor scenario), parse failures fall back to uniform and are
counted, cache in data/runs/m3_llm_profile/, cumulative cost tracked
against the $1.00 cap.

Run: ``python -m mobility.src.eval.m3_llm_profile``
"""

from __future__ import annotations

import datetime as _dt
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from mobility.src.eval.m2_recovery import load_noised
from mobility.src.eval.m3_calibration import brier, true_distribution
from mobility.src.eval.truth_vault import truth_dir

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.llm_client import GeminiClient, load_dotenv_if_present  # noqa: E402

MOBILITY_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = MOBILITY_ROOT / "results"
CACHE_DIR = MOBILITY_ROOT / "data" / "runs" / "m3_llm_profile"
COST_CAP_USD = 1.00
PRICE_IN, PRICE_OUT = 0.30, 2.50  # USD per M tokens, flash-lite

PROMPT = (
    "People with this profile: age {age_band}, occupation {occupation_type}, "
    "household {household}, area type {area_type}.\n\n"
    "Scenario:\n{text}\n\n"
    "Estimate what share of people with exactly this profile would pick each "
    "option. Reply with only the shares on one line, e.g. {example}. "
    "Shares must sum to 1."
)


def profile_key(profile: dict) -> str:
    return "|".join(profile[f] for f in
                    ("age_band", "occupation_type", "household", "area_type"))


def parse_shares(text: str, options: list[str]) -> dict[str, float] | None:
    hits = dict(re.findall(r"\b([A-C])\s*=\s*([01]?\.?\d+)", text or ""))
    try:
        shares = {o: float(hits[o]) for o in options if o in hits}
    except ValueError:
        return None
    if set(shares) != set(options):
        return None
    total = sum(shares.values())
    if total <= 0:
        return None
    return {o: v / total for o, v in shares.items()}


def main() -> dict:
    load_dotenv_if_present(REPO_ROOT / ".env")
    bank = json.loads((MOBILITY_ROOT / "data" / "bank" / "scenarios.json").read_text())
    by_sid = {s["sid"]: s for s in bank}
    profiles = {p["pid"]: p for p in json.loads(
        (MOBILITY_ROOT / "data" / "profiles.json").read_text())}
    splits = json.loads((MOBILITY_ROOT / "experiments" / "m2_splits.json").read_text())
    held = splits["held_out_travelers"]
    eval_sids = sorted(s for s in splits["held_out_scenarios"]
                       if not s.startswith("distr"))
    ids = [f"t{i:04d}" for i in range(400)]
    train_ids = [p for p in ids if p not in set(held)]
    noised = load_noised(MOBILITY_ROOT / "data" / "runs" / "m1" / "answers_noised.jsonl")

    uniq: dict[str, dict] = {}
    for pid in held:
        uniq.setdefault(profile_key(profiles[pid]), profiles[pid])

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / "predictions.jsonl"
    cache: dict[tuple[str, str], dict] = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            r = json.loads(line)
            cache[(r["profile_key"], r["sid"])] = r

    client = GeminiClient()
    state = {"in": 0, "out": 0, "calls": 0, "errors": 0, "parse_failures": 0}

    def cost() -> float:
        return (state["in"] * PRICE_IN + state["out"] * PRICE_OUT) / 1e6

    def one(job: tuple[str, str]) -> tuple[tuple[str, str], dict] | None:
        pkey, sid = job
        s = by_sid[sid]
        options = [o["key"] for o in s["options"]]
        example = " ".join(f"{o}=0.{50 - 10 * i:02d}" for i, o in enumerate(options))
        prompt = PROMPT.format(**uniq[pkey], text=s["text"], example=example)
        for attempt in range(4):
            if cost() > COST_CAP_USD:
                return None
            try:
                resp = client.generate(None, prompt, temperature=0.0, max_tokens=32)
                state["in"] += resp.input_tokens
                state["out"] += resp.output_tokens
                state["calls"] += 1
                shares = parse_shares(resp.text, options)
                if shares is None:
                    state["parse_failures"] += 1
                    shares = {o: 1.0 / len(options) for o in options}
                return (pkey, sid), {"profile_key": pkey, "sid": sid,
                                     "shares": shares}
            except RuntimeError:
                time.sleep(4 * (attempt + 1))
        state["errors"] += 1
        return (pkey, sid), {"profile_key": pkey, "sid": sid,
                             "shares": {o: 1.0 / len(options) for o in options},
                             "error": True}

    jobs = [(pk, sid) for pk in uniq for sid in eval_sids
            if (pk, sid) not in cache]
    print(f"{len(jobs)} calls to make ({len(cache)} cached)")
    with ThreadPoolExecutor(max_workers=4) as pool:
        with cache_path.open("a", encoding="utf-8") as fh:
            for out in pool.map(one, jobs):
                if out is None:
                    raise SystemExit(f"STOP: cost cap {COST_CAP_USD} reached "
                                     f"at ${cost():.3f}")
                key, rec = out
                cache[key] = rec
                fh.write(json.dumps(rec) + "\n")
    if state["errors"] > 0.05 * max(len(jobs), 1):
        raise SystemExit(f"STOP: {state['errors']} persistent errors")

    # ---- score on the identical cells as the other baselines --------------
    stash = np.load(truth_dir() / "wobble_m1.npz", allow_pickle=False)
    wobble = {str(p): float(w) for p, w in zip(stash["ids"], stash["wobble"])}
    pre, dists = {}, {}
    shard_dir = truth_dir() / "runs" / "m1_sweep_r2" / "answers"
    for pid in held:
        for line in (shard_dir / f"{pid}.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r["answer"] and r["sid"] in set(eval_sids):
                pre[(pid, r["sid"])] = r["answer"]
                dists[(pid, r["sid"])] = r.get("dist_top")

    base_share = {}
    for sid in eval_sids:
        counts: dict[str, int] = {}
        for p in train_ids:
            a = noised.get((p, sid))
            if a:
                counts[a] = counts.get(a, 0) + 1
        total = sum(counts.values())
        base_share[sid] = {o: c / total for o, c in counts.items()}

    llm_rows, marg_rows = [], []
    for pid in held:
        pkey = profile_key(profiles[pid])
        for sid in eval_sids:
            if (pid, sid) not in pre:
                continue
            options = [o["key"] for o in by_sid[sid]["options"]]
            p_true = true_distribution(pre[(pid, sid)], dists[(pid, sid)],
                                       set(options), wobble[pid], pid, sid)
            llm_rows.append(brier(cache[(pkey, sid)]["shares"], p_true, options))
            marg_rows.append(brier(base_share[sid], p_true, options))
    bs_llm, bs_marg = float(np.mean(llm_rows)), float(np.mean(marg_rows))

    record = {
        "role": "frozen section 5 baseline 4 (LLM given public profile, no "
                "choices), Ruling 29; reported beside the lift bar",
        "date": _dt.date.today().isoformat(),
        "model": "gemini flash-lite, temperature 0",
        "n_calls_made": state["calls"], "n_cached_before": len(cache) - len(jobs)
            if len(jobs) <= len(cache) else 0,
        "n_cells": len(llm_rows),
        "parse_failures_uniform_fallback": state["parse_failures"],
        "persistent_errors": state["errors"],
        "brier": round(bs_llm, 4),
        "brier_marginal_same_cells": round(bs_marg, 4),
        "lift_vs_marginal": round(1 - bs_llm / bs_marg, 4),
        "cost_usd": round(cost(), 4),
        "cap_usd": COST_CAP_USD,
        "tokens": {"input": state["in"], "output": state["out"]},
    }
    m3_path = RESULTS_DIR / "m3_calibration.json"
    m3 = json.loads(m3_path.read_text())
    m3["llm_profile_baseline_ruling29"] = record
    m3_path.write_text(json.dumps(m3, indent=2) + "\n", encoding="utf-8")
    return record


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
