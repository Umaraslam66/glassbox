"""Stage M0 availability pilot, committed form (Gate M0 Ruling 6).

Measures, for the traveler-side and fallback APIs (PRD_MOBILITY.md section 7):
key validity, latency, burst behavior under rate limits, cost, hidden
reasoning-token spend, answer-token-distribution availability, and
repeated-sample disagreement. Emits a summary JSON with the same shape as the
Stage M0 record at ``mobility/results/m0_pilot_summary.json`` (that file's
numbers stand as the M0 record; this tool reproduces the measurement, not the
numbers).

The one hand-written traveler card and three scenarios below are placeholder
fixtures for measuring the transport, not study data; no output of this tool
feeds any estimator.

Usage:
    python -m mobility.tools.m0_pilot --small [--skip-leonardo] [--out PATH]

``--small`` runs a ~20-request smoke (~$0.002); the default sizes mirror the
original pilot (~100 requests, ~$0.01). Never writes into mobility/data/.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.llm_client import load_dotenv_if_present  # noqa: E402  (parent utility, imported unchanged)

# Field names for answer-token-distribution access come from config, not code:
# the parent Wall test bans a token-distribution keyword from every Python
# file in the repository, and the parent is frozen. Gate M0 Ruling 4 orders
# this exact mechanism; PROJECT_LOG_MOBILITY.md (2026-08-03) has the entry.
FIELDS = json.loads((REPO_ROOT / "mobility" / "config" / "api_fields.json").read_text())

OR_URL = "https://openrouter.ai/api/v1/chat/completions"
GEM_URL = "https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent"

GEMINI_PRICING_USD_PER_M = {"input": 0.30, "output": 2.50,
                            "source": "ai.google.dev/gemini-api/docs/pricing, fetched 2026-08-03"}

CARD = (
    "You are Jonas, 38, a warehouse shift supervisor in a mid-size European city. "
    "You live with your partner and two school-age children in a rowhouse 11 km from "
    "work. Shifts start at 06:30 sharp; being late twice in a month means a formal "
    "warning. Money is planned to the last fifty euros each month. You own an older "
    "diesel wagon and there is a bus line with one change near your house. You have "
    "driven the same route for six years. Answer every question fully in character, "
    "based on how Jonas would actually choose. Reply with exactly one letter (A, B, "
    "or C) and nothing else."
)

SCENARIOS = [
    (
        "It is a normal Tuesday. Options for your trip to work:\n"
        "A) Drive: 22 min door-to-door, fuel+parking EUR 6.10, light traffic expected.\n"
        "B) Bus with one change: 41 min door-to-door, EUR 3.20, moderately crowded.\n"
        "C) E-bike (borrowed): 34 min, free, light rain forecast.\n"
        "Which do you choose? Answer A, B, or C."
    ),
    (
        "Roadworks start on your usual route. Options today:\n"
        "A) Usual road via detour: 35 min, EUR 6.40, arrival time reliable within 5 min.\n"
        "B) Motorway toll route: 24 min, EUR 9.90 including toll, reliable.\n"
        "C) Bus with one change: 44 min, EUR 3.20, can be 10 min late when roads jam.\n"
        "Which do you choose? Answer A, B, or C."
    ),
    (
        "Your city introduces a EUR 4.00 weekday congestion charge for driving into "
        "the zone your workplace is in. Options:\n"
        "A) Keep driving: 22 min, now EUR 10.10 per day.\n"
        "B) Bus with one change: 41 min, EUR 3.20, crowded at that hour.\n"
        "C) Drive to a park-and-ride, then metro: 33 min, EUR 5.60 total.\n"
        "Which do you choose? Answer A, B, or C."
    ),
]


# --------------------------------------------------------------------------
# normalized per-request records (no raw API bodies leave the call functions)
# --------------------------------------------------------------------------

def _env(name: str) -> str:
    import os
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} missing -- populate .env and retry")
    return value


def call_openrouter(scenario: str, temperature: float = 0.7, max_tokens: int = 2048,
                    want_dist: bool = False, no_reasoning: bool = False,
                    tag: str = "") -> dict:
    """One scenario answer; returns a normalized record, never the raw body."""
    f = FIELDS["openrouter"]
    payload: dict = {
        "model": _env("OPENROUTER_MODEL_NAME"),
        "messages": [{"role": "system", "content": CARD},
                     {"role": "user", "content": scenario}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "usage": {"include": True},
    }
    if want_dist:
        payload[f["request_flag"]] = True
        payload[f["request_top_k"]] = 5
    if no_reasoning:
        payload["reasoning"] = {"enabled": False}

    req = urllib.request.Request(
        OR_URL, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {_env('OPENROUTER_API_KEY')}"},
        method="POST")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            body = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:400]
        return {"provider": "openrouter", "tag": tag, "status": exc.code,
                "latency_s": round(time.time() - t0, 2), "error": detail,
                "retry_after": exc.headers.get("Retry-After")}
    except Exception as exc:  # noqa: BLE001
        return {"provider": "openrouter", "tag": tag, "status": -1,
                "latency_s": round(time.time() - t0, 2), "error": str(exc)[:200]}

    choice = (body.get("choices") or [{}])[0]
    usage = body.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    dist_top = None
    dist = choice.get(f["choice_field"])
    if dist and dist.get(f["content_field"]):
        first = dist[f["content_field"]][0]
        alts = first.get(f["alternatives_field"]) or []
        dist_top = [[a.get(f["token_field"]), round(2.718281828 ** a[f["value_field"]], 4)]
                    for a in alts if a.get(f["value_field"]) is not None]
    return {
        "provider": "openrouter", "tag": tag, "status": 200,
        "latency_s": round(time.time() - t0, 2),
        "answer": ((choice.get("message") or {}).get("content") or "").strip()[:8],
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": details.get("reasoning_tokens"),
        "cost_usd": usage.get("cost"),
        "dist_top": dist_top,
        "dist_requested": want_dist,
        "reasoning_off": no_reasoning,
    }


def call_gemini(scenario: str, temperature: float = 0.7, max_tokens: int = 512,
                want_dist: bool = False, tag: str = "") -> dict:
    f = FIELDS["gemini"]
    gen: dict = {"temperature": temperature, "maxOutputTokens": max_tokens}
    if want_dist:
        gen[f["request_flag"]] = True
        gen[f["request_top_k"]] = 5
    payload = {
        "contents": [{"role": "user", "parts": [{"text": scenario}]}],
        "systemInstruction": {"parts": [{"text": CARD}]},
        "generationConfig": gen,
    }
    req = urllib.request.Request(
        GEM_URL.format(m=_env("MODEL_NAME")), data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "x-goog-api-key": _env("GLASSBOX_API_KEY")},
        method="POST")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:400]
        return {"provider": "gemini", "tag": tag, "status": exc.code,
                "latency_s": round(time.time() - t0, 2), "error": detail}
    except Exception as exc:  # noqa: BLE001
        return {"provider": "gemini", "tag": tag, "status": -1,
                "latency_s": round(time.time() - t0, 2), "error": str(exc)[:200]}

    cand = (body.get("candidates") or [{}])[0]
    parts = (cand.get("content") or {}).get("parts") or []
    um = body.get("usageMetadata") or {}
    dist_top = None
    dist = cand.get(f["result_field"])
    if dist:
        chosen = dist.get(f["chosen_field"]) or []
        if chosen:
            dist_top = [[chosen[0].get(f["token_field"]),
                         round(2.718281828 ** chosen[0][f["value_field"]], 4)]]
    return {
        "provider": "gemini", "tag": tag, "status": 200,
        "latency_s": round(time.time() - t0, 2),
        "answer": "".join(p.get("text", "") for p in parts).strip()[:8],
        "prompt_tokens": um.get("promptTokenCount"),
        "completion_tokens": um.get("candidatesTokenCount"),
        "reasoning_tokens": um.get("thoughtsTokenCount"),
        "dist_top": dist_top,
        "dist_requested": want_dist,
    }


def check_leonardo(timeout_s: int = 12) -> dict:
    stamp = _dt.datetime.now().astimezone().isoformat(timespec="minutes")
    try:
        proc = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={timeout_s}",
             "leonardo", "echo ok"],
            capture_output=True, text=True, timeout=timeout_s + 8)
        if proc.returncode == 0 and "ok" in proc.stdout:
            return {"checked": stamp, "result": "login node reachable",
                    "verdict": "reachable"}
        return {"checked": stamp,
                "result": (proc.stderr.strip() or "connection failed")[:200],
                "verdict": "unreachable"}
    except Exception as exc:  # noqa: BLE001
        return {"checked": stamp, "result": str(exc)[:200], "verdict": "unreachable"}


# --------------------------------------------------------------------------
# summary assembly -- same shape as results/m0_pilot_summary.json
# --------------------------------------------------------------------------

def _lat(rows: list[dict], q: float = 0.5) -> float | None:
    ok = sorted(r["latency_s"] for r in rows if r.get("status") == 200)
    if not ok:
        return None
    if q == 0.5:
        return round(statistics.median(ok), 2)
    return round(ok[max(0, int(q * len(ok)) - 1)], 2)


def _dist_verdict(rows: list[dict]) -> str:
    """USABLE when the top answer token leaves real probability elsewhere."""
    tops = [r["dist_top"][0][1] for r in rows
            if r.get("status") == 200 and r.get("dist_top")]
    if not tops:
        return "NOT AVAILABLE: no distribution returned"
    top_median = statistics.median(tops)
    if top_median > 0.99:
        return f"DEGENERATE: answer-token probability ~1.0 (median {top_median:.4f})"
    return f"USABLE: real spread (median top-choice probability {top_median:.2f})"


def _gemini_dist_verdict(probe: dict) -> str:
    if probe.get("status") == 400 and "not enabled" in (probe.get("error") or "").lower():
        return "NOT AVAILABLE: HTTP 400, distribution access not enabled for this model"
    if probe.get("status") == 200 and probe.get("dist_top"):
        return "AVAILABLE"
    return f"NOT AVAILABLE: status {probe.get('status')}"


def build_summary(or_rows: list[dict], gem_rows: list[dict], leonardo: dict,
                  run_label: str) -> dict:
    on = [r for r in or_rows if not r.get("reasoning_off")]
    off = [r for r in or_rows if r.get("reasoning_off")]
    on_ok = [r for r in on if r.get("status") == 200]
    off_ok = [r for r in off if r.get("status") == 200]
    gem_ok = [r for r in gem_rows if r.get("status") == 200]
    gem_probe = next((r for r in gem_rows if r.get("dist_requested")), {})

    def cost_per_1k(rows: list[dict]) -> float | None:
        costs = [r["cost_usd"] for r in rows if r.get("cost_usd") is not None]
        return round(1000 * statistics.mean(costs), 4) if costs else None

    gem_cost = None
    if gem_ok:
        pin = statistics.mean(r.get("prompt_tokens") or 0 for r in gem_ok)
        pout = statistics.mean(r.get("completion_tokens") or 0 for r in gem_ok)
        gem_cost = round((pin * GEMINI_PRICING_USD_PER_M["input"]
                          + pout * GEMINI_PRICING_USD_PER_M["output"]) / 1000, 4)

    or_cost_total = sum(r.get("cost_usd") or 0 for r in or_rows)
    return {
        "experiment": "M0 availability pilot (committed tool)",
        "date": _dt.date.today().isoformat(),
        "run_label": run_label,
        "purpose": "PRD_MOBILITY section 7 availability measurement; "
                   "shape-compatible with the Stage M0 record in results/m0_pilot_summary.json",
        "fixtures_note": "1 hand-written traveler card + 3 scenarios; placeholder fixtures, not study data",
        "leonardo": leonardo,
        "openrouter_qwen37_flash": {
            "requests_total": len(or_rows),
            "reasoning_on": {
                "n": len(on), "ok": len(on_ok),
                "latency_median_s": _lat(on),
                "reasoning_tokens_per_answer_mean":
                    round(statistics.mean(r.get("reasoning_tokens") or 0 for r in on_ok), 1)
                    if on_ok else None,
                "usd_per_1k_scenario_answers": cost_per_1k(on_ok),
                "answer_token_distribution": _dist_verdict(on),
                "n_429": sum(1 for r in on if r.get("status") == 429),
            },
            "reasoning_off": {
                "n": len(off), "ok": len(off_ok),
                "latency_median_s": _lat(off),
                "usd_per_1k_scenario_answers": cost_per_1k(off_ok),
                "answer_token_distribution": _dist_verdict(off),
                "n_429": sum(1 for r in off if r.get("status") == 429),
            },
            "rate_limit_nature": "; ".join(sorted({
                (r.get("error") or "")[:120] for r in or_rows if r.get("status") == 429
            })) or "no 429 observed in this run",
            "repeat_sample_disagreement": _repeat_table(or_rows),
        },
        "gemini_35_flash_lite": {
            "requests_total": len(gem_rows),
            "latency_median_s": _lat(gem_rows),
            "answer_token_distribution": _gemini_dist_verdict(gem_probe),
            "thinking_tokens": max((r.get("reasoning_tokens") or 0) for r in gem_ok) if gem_ok else 0,
            "pricing_usd_per_m": GEMINI_PRICING_USD_PER_M,
            "usd_per_1k_scenario_answers": gem_cost,
        },
        "costs_usd": {
            "openrouter_metered": round(or_cost_total, 5),
            "gemini_computed": round((gem_cost or 0) * len(gem_ok) / 1000, 5),
            "total": round(or_cost_total + (gem_cost or 0) * len(gem_ok) / 1000, 5),
        },
    }


def _repeat_table(or_rows: list[dict]) -> dict:
    out: dict = {}
    for r in or_rows:
        tag = r.get("tag") or ""
        if not tag.startswith("repeat_s"):
            continue
        bucket = out.setdefault(tag.removeprefix("repeat_"), {})
        if r.get("status") == 200:
            letter = (r.get("answer") or "?")[:1]
            bucket[letter] = bucket.get(letter, 0) + 1
        else:
            bucket["rate_limited"] = bucket.get("rate_limited", 0) + 1
    return out or {"note": "no repeat block in this run"}


# --------------------------------------------------------------------------

def run_pilot(small: bool = False, skip_leonardo: bool = False) -> dict:
    n_seq, n_burst, burst_c, n_repeat, n_gem = (2, 4, 2, 6, 3) if small else (8, 24, 8, 10, 6)
    or_rows: list[dict] = []
    gem_rows: list[dict] = []

    or_rows.append(call_openrouter(SCENARIOS[0], want_dist=True, tag="probe"))
    or_rows.append(call_openrouter(SCENARIOS[0], want_dist=True, no_reasoning=True, tag="probe_off"))
    or_rows += [call_openrouter(SCENARIOS[i % 3], tag="seq") for i in range(n_seq)]
    with ThreadPoolExecutor(max_workers=burst_c) as pool:
        or_rows += list(pool.map(
            lambda i: call_openrouter(SCENARIOS[i % 3], tag="burst"), range(n_burst)))
    with ThreadPoolExecutor(max_workers=2) as pool:
        or_rows += list(pool.map(
            lambda i: call_openrouter(SCENARIOS[1], want_dist=True, no_reasoning=True,
                                      tag="repeat_s1"), range(n_repeat)))

    gem_rows.append(call_gemini(SCENARIOS[0], want_dist=True, tag="probe"))
    gem_rows += [call_gemini(SCENARIOS[i % 3], tag="seq") for i in range(n_gem)]

    leonardo = ({"checked": _dt.datetime.now().astimezone().isoformat(timespec="minutes"),
                 "result": "skipped in this run", "verdict": "not checked"}
                if skip_leonardo else check_leonardo())
    return build_summary(or_rows, gem_rows, leonardo,
                         run_label="small" if small else "full")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--small", action="store_true", help="~20-request smoke run")
    parser.add_argument("--skip-leonardo", action="store_true")
    parser.add_argument("--out", type=Path, default=None,
                        help="write summary JSON here (never inside mobility/data/)")
    args = parser.parse_args()

    load_dotenv_if_present(REPO_ROOT / ".env")
    summary = run_pilot(small=args.small, skip_leonardo=args.skip_leonardo)
    text = json.dumps(summary, indent=2)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
