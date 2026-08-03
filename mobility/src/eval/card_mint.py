"""Card factory: renders traveler biographies from planted phi, vault-side.

Lives in the grader's directory because it must read planted truth — the Wall
rule is that nothing else may. It writes full cards INTO the vault (cards are
truth: PREREGISTRATION_MOBILITY.md sections 2 and 4), and emits only the
public profiles to the system-visible side. Profiles are sampled
independently of phi by design — in this population, demographics carry no
parameter signal.

Frozen regime (section 8): cards are written with reasoning ON, two passes —
write, then self-check-and-revise — with a mechanical leak check on top and
one repair attempt for any card that trips it.

Resumable: one JSON per traveler under the vault's run directory; finished
travelers are never re-rendered; the API-level cache below that never
re-queries. Run:
``python -m mobility.src.eval.card_mint mobility/experiments/m1_cards.json``
"""

from __future__ import annotations

import datetime as _dt
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from mobility.src.engine.or_client import ORClient
from mobility.src.eval.truth_vault import truth_dir
from mobility.src.travelers.sampler import DIMS

MOBILITY_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = MOBILITY_ROOT / "results"
PUBLIC_DATA_DIR = MOBILITY_ROOT / "data"

# ---------------------------------------------------------------- profiles

AGE_BANDS = ("25-34", "35-44", "45-54", "55-64")
OCCUPATIONS = ("office administration", "healthcare", "education",
               "retail and service", "logistics and manual", "self-employed")
HOUSEHOLDS = ("single", "couple", "couple with children", "single parent",
              "shared flat")
AREAS = ("inner city", "suburb", "small town")


def sample_profiles(seed: int, ids: list[str]) -> dict[str, dict]:
    """Public profiles, independent of phi by construction (frozen design)."""
    rng = np.random.default_rng(seed)
    profiles = {}
    for pid in ids:
        profiles[pid] = {
            "pid": pid,
            "age_band": str(rng.choice(AGE_BANDS)),
            "occupation_type": str(rng.choice(OCCUPATIONS)),
            "household": str(rng.choice(HOUSEHOLDS)),
            "area_type": str(rng.choice(AREAS)),
        }
    return profiles

# ---------------------------------------------------------------- descriptors

_BINS = (-1.2, -0.4, 0.4, 1.2)  # 5 verbal levels per dimension

_DESCRIPTORS = {
    "VOT": (
        "treats waiting as free time and would never pay to save a few minutes",
        "usually takes the cheaper option even when it is clearly slower",
        "weighs time against money case by case, with no strong pull either way",
        "often pays a bit extra to save meaningful time",
        "pays almost anything to skip a delay; time is the scarcest thing they have",
    ),
    "SCH": (
        "start times barely exist for them; arriving late is a shrug",
        "prefers being roughly on time but does not build in buffers",
        "tries to be punctual and mildly dislikes cutting it close",
        "plans arrivals with a clear safety margin; lateness genuinely bothers them",
        "has hard fixed start times, builds large early buffers, and fears being late",
    ),
    "MOD": (
        "finds owning and driving a car a pure burden; happily car-free",
        "uses whatever mode is practical and feels no attachment to driving",
        "neutral between car, transit and bike; the numbers decide",
        "prefers the car and feels mild reluctance about buses and trains",
        "the car is part of who they are; other modes feel like a downgrade",
    ),
    "CRW": (
        "packed vehicles do not register with them at all",
        "notices crowding but rarely lets it change a plan",
        "prefers a seat but tolerates rush-hour crowds when needed",
        "actively plans around crowded services and values personal space",
        "avoids crowds at real cost; will reroute, retime, or pay to sit alone",
    ),
    "PRC": (
        "barely looks at prices for daily travel; small fees are invisible",
        "knows roughly what trips cost but rarely optimizes",
        "compares prices when the difference is obvious",
        "checks fares and fees and adjusts plans over moderate amounts",
        "recalculates whole routines over small fees and hunts every discount",
    ),
    "HAB": (
        "changes routes, modes and departure times constantly, partly for fun",
        "tries alternatives now and then without much resistance",
        "has loose routines but switches when there is a clear reason",
        "keeps stable routines and needs a good push to change them",
        "same route, same time, for years; change feels genuinely wrong",
    ),
}


def descriptors_for(phi_row: np.ndarray) -> list[str]:
    out = []
    for d, value in zip(DIMS, phi_row):
        level = int(np.searchsorted(_BINS, value))
        out.append(_DESCRIPTORS[d][level])
    return out

# ---------------------------------------------------------------- leak check

_BANNED_TERMS = (
    "value of time", "schedule rigidity", "mode affinity", "crowding sensitivity",
    "price sensitivity", "habit strength", "parameter", "trait", "dimension",
    "scale of", "out of 5", "/5", "score", "vot", "crw", "prc", "hab",
)
_SIGNED_NUMBER = re.compile(r"[+−-]\d+(?:\.\d+)?")


def leak_reasons(card: str) -> list[str]:
    lowered = card.lower()
    reasons = [f"banned term: {t!r}" for t in _BANNED_TERMS
               if re.search(rf"(?<![a-z]){re.escape(t)}(?![a-z])", lowered)]
    if _SIGNED_NUMBER.search(card):
        reasons.append("signed number (parameter-like)")
    return reasons

# ---------------------------------------------------------------- prompts

_WRITER_SYSTEM = (
    "You write concise, realistic character biographies for transport research. "
    "Hard rules: never name behavioral traits or use rating language (no 'high "
    "value of time', no scores, no scales, no signed numbers); show every "
    "tendency through concrete life details — job, family, commute history, "
    "money habits, routines; stay fully consistent with the given public "
    "profile; third person; a plausible European everyday setting; 160-220 "
    "words; output the biography only, no headers."
)

_CHECKER_SYSTEM = (
    "You check a biography against rules and revise it minimally. Rules: no "
    "trait names, no rating or scale language, no signed numbers; every one of "
    "the six tendencies must be visible through at least one concrete life "
    "detail; the public profile facts must be respected; 160-220 words. Return "
    "the corrected biography only — no commentary."
)


def _writer_user(profile: dict, descriptors: list[str]) -> str:
    return (
        f"Public profile: age {profile['age_band']}, works in "
        f"{profile['occupation_type']}, household: {profile['household']}, "
        f"lives in a {profile['area_type']}.\n\n"
        "Write the biography of this person. Give them a plausible first name. "
        "Express ALL six of these tendencies through concrete details, never "
        "naming them:\n"
        + "\n".join(f"- {d}" for d in descriptors)
    )


def _checker_user(profile: dict, descriptors: list[str], draft: str) -> str:
    return (
        f"Public profile: age {profile['age_band']}, {profile['occupation_type']}, "
        f"{profile['household']}, {profile['area_type']}.\n"
        "The six tendencies that must each be visible:\n"
        + "\n".join(f"- {d}" for d in descriptors)
        + "\n\nDraft biography:\n" + draft
    )

# ---------------------------------------------------------------- factory

def render_card(client: ORClient, pid: str, profile: dict,
                descriptors: list[str], temperature: float,
                max_tokens: int, reasoning_budget: int | None = None) -> dict:
    draft = client.call(_WRITER_SYSTEM, _writer_user(profile, descriptors),
                        temperature=temperature, max_tokens=max_tokens,
                        reasoning_on=True, seed_key=f"{pid}:pass1",
                        reasoning_budget=reasoning_budget)
    checked = client.call(_CHECKER_SYSTEM,
                          _checker_user(profile, descriptors, draft["text"]),
                          temperature=temperature, max_tokens=max_tokens,
                          reasoning_on=True, seed_key=f"{pid}:pass2",
                          reasoning_budget=reasoning_budget)
    card = checked["text"]
    leaks = leak_reasons(card)
    repaired = False
    if leaks:
        repair = client.call(
            _CHECKER_SYSTEM,
            _checker_user(profile, descriptors, card)
            + "\n\nThe draft violates these rules, fix them: "
            + "; ".join(leaks),
            temperature=temperature, max_tokens=max_tokens,
            reasoning_on=True, seed_key=f"{pid}:repair",
            reasoning_budget=reasoning_budget)
        card = repair["text"]
        leaks = leak_reasons(card)
        repaired = True
    return {
        "pid": pid,
        "profile": profile,
        "card": card,
        "word_count": len(card.split()),
        "leaks_remaining": leaks,
        "repaired": repaired,
        "truncated": any(r.get("finish") == "length" for r in (draft, checked)),
    }


def mint_cards(config_path: Path) -> dict:
    config = json.loads(config_path.read_text())
    vault = truth_dir()
    run_dir = vault / "runs" / config["experiment"]
    card_dir = run_dir / "cards"
    card_dir.mkdir(parents=True, exist_ok=True)

    stash = np.load(vault / "phi_m1.npz", allow_pickle=False)
    ids = [str(x) for x in stash["ids"]]
    phi = stash["phi"]

    profiles = sample_profiles(int(config["seed_profiles"]), ids)
    client = ORClient(cache_dir=run_dir / "cache")

    todo = [pid for pid in ids if not (card_dir / f"{pid}.json").is_file()]

    def work(pid: str) -> None:
        row = phi[ids.index(pid)]
        record = render_card(client, pid, profiles[pid], descriptors_for(row),
                             float(config["temperature"]), int(config["max_tokens"]),
                             reasoning_budget=config.get("reasoning_budget"))
        (card_dir / f"{pid}.json").write_text(json.dumps(record), encoding="utf-8")

    with ThreadPoolExecutor(max_workers=int(config["concurrency"])) as pool:
        list(pool.map(work, todo))

    # assemble + aggregate QA (aggregates only leave the vault)
    records = [json.loads((card_dir / f"{pid}.json").read_text()) for pid in ids]
    with (run_dir / "cards.jsonl").open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    PUBLIC_DATA_DIR.mkdir(parents=True, exist_ok=True)
    (PUBLIC_DATA_DIR / "profiles.json").write_text(
        json.dumps([profiles[pid] for pid in ids], indent=1), encoding="utf-8")

    words = [r["word_count"] for r in records]
    qa = {
        "experiment": config["experiment"],
        "date": _dt.date.today().isoformat(),
        "n_cards": len(records),
        "n_rendered_this_run": len(todo),
        "leaks_remaining": sum(bool(r["leaks_remaining"]) for r in records),
        "repaired": sum(r["repaired"] for r in records),
        "truncated": sum(r["truncated"] for r in records),
        "word_count": {"min": min(words), "median": int(np.median(words)),
                       "max": max(words)},
        "client": client.stats(),
        "config": config,
    }
    (RESULTS_DIR / f"{config['experiment']}_qa.json").write_text(
        json.dumps(qa, indent=2) + "\n", encoding="utf-8")
    return qa


if __name__ == "__main__":
    summary = mint_cards(Path(sys.argv[1]))
    print(json.dumps({k: summary[k] for k in
                      ("n_cards", "n_rendered_this_run", "leaks_remaining",
                       "repaired", "truncated", "client")}, indent=2))
