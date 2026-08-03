"""The frozen 140-scenario bank, generated deterministically from a seed.

Block structure per PREREGISTRATION_MOBILITY.md section 3 (frozen design
table): 50 mode choice (14 VOT + 12 MOD + 10 CRW-isolating + 10 PRC-isolating
+ 4 HAB switch-vs-stick), 30 departure time (SCH), 25 route/reliability
(13 VOT + 8 SCH + 4 HAB), 25 policy response (15 PRC + 6 MOD + 4 VOT),
10 distractors.

Isolation sub-blocks hold the whole trip constant ACROSS their scenarios and
vary only the isolated attribute (crowding level, or the transit fare), so a
traveler's flip point across the sub-block identifies that dimension free of
the others — the design-table reading recorded for the audit.

Each scenario carries a primary dimension and per-option scores in
{-1, 0, +1}: +1 marks the option a high-phi traveler on the primary dimension
is designed to pick. Scores and primary dimensions are planted truth — the
vault-side bank mint separates them from the public text+attributes file.
This module is pure: no paths, no truth reads.
"""

from __future__ import annotations

import numpy as np

LETTERS = ("A", "B", "C")
ASK = "Which do you choose? Answer with exactly one letter."

CROWD_LEVELS = (
    "seats freely available", "most seats taken", "standing room only",
    "packed shoulder to shoulder",
)


def _sc(sid, block, primary, text, options, scores, tags=()):
    assert len(options) == len(scores) <= 3
    return {
        "sid": sid, "block": block, "primary": primary,
        "text": text + "\n" + ASK,
        "options": [{"key": LETTERS[i], "label": lab, "attrs": attrs}
                    for i, (lab, attrs) in enumerate(options)],
        "scores": {LETTERS[i]: s for i, s in enumerate(scores)},
        "tags": list(tags),
    }


def _mode_vot(rng, i):
    base = int(rng.integers(18, 35))
    save = int(rng.integers(8, 26))
    cost_gap = round(float(rng.uniform(2.5, 9.0)), 2)
    cheap = round(float(rng.uniform(1.5, 3.5)), 2)
    fast, slow = base, base + save
    text = (f"Commute to work this morning:\n"
            f"A) Express connection: {fast} min door-to-door, EUR {cheap + cost_gap:.2f}, {CROWD_LEVELS[1]}.\n"
            f"B) Regular connection: {slow} min door-to-door, EUR {cheap:.2f}, {CROWD_LEVELS[1]}.")
    return _sc(f"mode_vot_{i:02d}", "mode", "VOT", text,
               [(f"Express, {fast} min, EUR {cheap + cost_gap:.2f}",
                 {"time_min": fast, "cost_eur": round(cheap + cost_gap, 2), "crowding": 1}),
                (f"Regular, {slow} min, EUR {cheap:.2f}",
                 {"time_min": slow, "cost_eur": cheap, "crowding": 1})],
               [+1, -1])


def _mode_mod(rng, i):
    t = int(rng.integers(22, 40))
    dt = int(rng.integers(-4, 5))
    c_car = round(float(rng.uniform(3.5, 6.5)), 2)
    c_pt = round(float(rng.uniform(2.5, 4.0)), 2)
    text = (f"A normal weekday trip across town:\n"
            f"A) Drive: {t} min, EUR {c_car:.2f} fuel and parking, {CROWD_LEVELS[0]}.\n"
            f"B) Tram: {t + dt} min, EUR {c_pt:.2f}, {CROWD_LEVELS[1]}.\n"
            f"C) Bike: {t + abs(dt) + 6} min, free, dry weather.")
    return _sc(f"mode_mod_{i:02d}", "mode", "MOD", text,
               [(f"Drive, {t} min, EUR {c_car:.2f}", {"time_min": t, "cost_eur": c_car, "mode": "car"}),
                (f"Tram, {t + dt} min, EUR {c_pt:.2f}", {"time_min": t + dt, "cost_eur": c_pt, "mode": "transit"}),
                (f"Bike, {t + abs(dt) + 6} min, free", {"time_min": t + abs(dt) + 6, "cost_eur": 0.0, "mode": "bike"})],
               [+1, -1, -1])


def _mode_crw_iso(level_idx, i):
    # Whole trip held constant across the sub-block; only crowding varies.
    text = (f"Your usual trip to the center, both options 26 minutes door-to-door:\n"
            f"A) Tram, EUR 2.80, currently {CROWD_LEVELS[level_idx]}.\n"
            f"B) Drive and park, EUR 5.50, alone in the car.")
    return _sc(f"mode_crw_{i:02d}", "mode", "CRW", text,
               [(f"Tram, 26 min, EUR 2.80, {CROWD_LEVELS[level_idx]}",
                 {"time_min": 26, "cost_eur": 2.80, "crowding": level_idx}),
                ("Drive, 26 min, EUR 5.50",
                 {"time_min": 26, "cost_eur": 5.50, "crowding": 0})],
               [-1, +1], tags=("crw_iso",))


#: (base fare, saving) grid for the PRC isolation sub-block. Redesigned under
#: Gate M1 Ruling 10 from design principles only: within ONE mode, equal
#: times, a price saving against a small NON-TIME friction, so no option
#: dominates and no other planted dimension loads. The friction is constant
#: across the sub-block; only the saving moves — the flip point identifies
#: price sensitivity.
_PRC_GRID = (
    (3.60, 0.20), (3.60, 0.35), (3.80, 0.50), (3.40, 0.70), (3.60, 0.90),
    (3.80, 1.10), (3.40, 1.40), (3.60, 1.80), (3.80, 2.20), (3.60, 2.60),
)


def _mode_prc_iso(base, save, i):
    text = (f"Your regular train to work, the same train either way, 24 minutes, "
            f"seats usually available:\n"
            f"A) Discounted fare, EUR {base - save:.2f}: it only exists in the operator's "
            f"app, which does not store payment cards — you dig out your bank card and "
            f"re-enter it before every single trip.\n"
            f"B) Tap your regular card at the gate, EUR {base:.2f}: no steps, full price.")
    return _sc(f"mode_prc_{i:02d}", "mode", "PRC", text,
               [(f"App discount, EUR {base - save:.2f}, card re-entry every trip",
                 {"time_min": 24, "cost_eur": round(base - save, 2), "friction": 1, "mode": "transit"}),
                (f"Tap and go, EUR {base:.2f}",
                 {"time_min": 24, "cost_eur": base, "friction": 0, "mode": "transit"})],
               [+1, -1], tags=("prc_iso",))


def _mode_hab(rng, i):
    gain = int(rng.integers(3, 8))
    text = (f"A colleague shows you a different way to work:\n"
            f"A) Your usual route and mode, as every day: 31 min, EUR 3.10.\n"
            f"B) The new alternative: {31 - gain} min, EUR 3.10, you have never tried it.")
    return _sc(f"mode_hab_{i:02d}", "mode", "HAB", text,
               [("Usual route, 31 min", {"time_min": 31, "cost_eur": 3.10, "habitual": 1}),
                (f"New alternative, {31 - gain} min", {"time_min": 31 - gain, "cost_eur": 3.10, "habitual": 0})],
               [+1, -1], tags=("switch_vs_stick",))


def _dep_time(rng, i):
    slack = int(rng.integers(5, 41))
    p_late = int(rng.choice((5, 10, 15, 20, 25, 30, 35, 40)))
    delay = int(rng.integers(10, 25))
    text = (f"You must be at work at 09:00. Two departures, same ticket price:\n"
            f"A) Early one: arrives 08:{60 - slack if slack < 60 else 59:02d}, "
            f"guaranteed on time, {slack} min early at your desk.\n"
            f"B) Later one: arrives 09:00 sharp if all goes well, but a {p_late}% "
            f"chance of roadworks making you about {delay} min late.")
    return _sc(f"dep_sch_{i:02d}", "dep_time", "SCH", text,
               [(f"Early, {slack} min buffer", {"slack_min": slack, "p_late": 0, "cost_eur": 0.0}),
                (f"Later, {p_late}% risk of {delay} min late",
                 {"slack_min": 0, "p_late": p_late / 100, "late_min": delay, "cost_eur": 0.0})],
               [+1, -1])


def _route_vot(rng, i):
    gap = int(rng.integers(8, 18))
    spread = int(rng.integers(2, 5))
    base = int(rng.integers(20, 34))
    text = (f"Two roads to the same place, same fuel cost:\n"
            f"A) Fast road: {base} min, saves you {gap} min almost every single day "
            f"(a rare jam once or twice a month adds {spread} min).\n"
            f"B) Quiet road: {base + gap} min.")
    return _sc(f"route_vot_{i:02d}", "route", "VOT", text,
               [(f"Fast, {base} min, rare +{spread}",
                 {"time_min": base, "time_max": base + spread, "cost_eur": 0.0}),
                (f"Quiet, {base + gap} min",
                 {"time_min": base + gap, "time_max": base + gap, "cost_eur": 0.0})],
               [+1, -1])


def _route_sch(rng, i):
    base = int(rng.integers(22, 32))
    spread = int(rng.integers(12, 26))
    text = (f"You have a fixed appointment. Two routes, same cost:\n"
            f"A) Usually quicker: {base} min most days, but roughly one day in five "
            f"it jams to {base + spread} min.\n"
            f"B) Steady route: always {base + 6} min.")
    return _sc(f"route_sch_{i:02d}", "route", "SCH", text,
               [(f"Quicker but jams, {base}-{base + spread} min",
                 {"time_min": base, "time_max": base + spread, "p_late": 0.2, "cost_eur": 0.0}),
                (f"Steady, {base + 6} min",
                 {"time_min": base + 6, "time_max": base + 6, "p_late": 0.0, "cost_eur": 0.0})],
               [-1, +1])


def _route_hab(rng, i):
    gain = int(rng.integers(4, 9))
    text = (f"Navigation suggests a route you have never used:\n"
            f"A) Your route of many years: 28 min today.\n"
            f"B) Suggested new route: {28 - gain} min today, unfamiliar streets.")
    return _sc(f"route_hab_{i:02d}", "route", "HAB", text,
               [("Your usual route, 28 min", {"time_min": 28, "habitual": 1, "cost_eur": 0.0}),
                (f"New route, {28 - gain} min", {"time_min": 28 - gain, "habitual": 0, "cost_eur": 0.0})],
               [+1, -1], tags=("switch_vs_stick",))


def _policy_prc(rng, i):
    fee = round(float(rng.uniform(2.0, 6.0)), 2)
    old = round(float(rng.uniform(3.0, 6.0)), 2)
    text = (f"Parking at work was free; from Monday it costs EUR {fee:.2f} per day.\n"
            f"A) Keep driving: 24 min, now EUR {old + fee:.2f} per day in total.\n"
            f"B) Switch to the bus: 38 min, EUR 2.90 per day.")
    return _sc(f"policy_prc_{i:02d}", "policy", "PRC", text,
               [(f"Keep driving, EUR {old + fee:.2f}",
                 {"time_min": 24, "cost_eur": round(old + fee, 2), "mode": "car"}),
                ("Bus, EUR 2.90", {"time_min": 38, "cost_eur": 2.90, "mode": "transit"})],
               [-1, +1], tags=("before_after",))


def _policy_mod(rng, i):
    worth = int(rng.integers(55, 90))
    dt = int(rng.integers(2, 6))
    text = (f"Your employer makes an offer: give up your reserved parking spot and "
            f"receive a free transit pass (worth EUR {worth} per month). The metro "
            f"takes {dt} min longer than driving on your commute:\n"
            f"A) Keep the parking spot and keep driving, at your own cost as today.\n"
            f"B) Take the free pass and switch to the metro.")
    return _sc(f"policy_mod_{i:02d}", "policy", "MOD", text,
               [("Keep the spot, keep driving",
                 {"time_min": 0, "cost_eur": 4.10, "mode": "car"}),
                (f"Free pass, metro, +{dt} min",
                 {"time_min": dt, "cost_eur": 0.0, "mode": "transit"})],
               [+1, -1], tags=("before_after",))


def _policy_vot(rng, i):
    charge = round(float(rng.uniform(2.5, 5.5)), 2)
    save = int(rng.integers(10, 22))
    text = (f"A new toll tunnel opens on your commute:\n"
            f"A) Toll tunnel: saves {save} min, costs EUR {charge:.2f} each way.\n"
            f"B) Old road: as before, free.")
    return _sc(f"policy_vot_{i:02d}", "policy", "VOT", text,
               [(f"Tunnel, -{save} min, EUR {charge:.2f}",
                 {"time_min": -save, "cost_eur": charge}),
                ("Old road, free", {"time_min": 0, "cost_eur": 0.0})],
               [+1, -1], tags=("before_after",))


_DISTRACTORS = (
    ("A free Saturday afternoon:", "Visit the city museum.", "Long walk in the park."),
    ("Dinner tonight:", "Pasta at home.", "Pizza from the place around the corner."),
    ("A gift voucher to spend:", "A stack of novels.", "A cooking class."),
    ("Sunday morning:", "Sleep in late.", "Early swim at the pool."),
    ("Picking a film for the evening:", "A comedy.", "A thriller."),
    ("Evening at home:", "A documentary.", "A quiz show."),
    ("With your afternoon coffee:", "A slice of cake.", "A biscuit."),
    ("Lunch on a workday:", "The canteen soup.", "The canteen salad."),
    ("A summer holiday idea:", "A week by a lake.", "A week of city visits."),
    ("Choosing a dentist appointment slot:", "Tuesday morning.", "Thursday afternoon."),
)


def _distractor(i):
    lead, a, b = _DISTRACTORS[i]
    text = f"{lead}\nA) {a}\nB) {b}"
    return _sc(f"distr_{i:02d}", "distractor", None, text,
               [(a, {}), (b, {})], [0, 0])


def generate_bank(seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    bank: list[dict] = []
    bank += [_mode_vot(rng, i) for i in range(14)]
    bank += [_mode_mod(rng, i) for i in range(12)]
    crw_levels = [0, 1, 1, 2, 2, 2, 3, 3, 3, 3]
    bank += [_mode_crw_iso(crw_levels[i], i) for i in range(10)]
    # No shared-rng draws in the PRC sub-block: its grid is fixed, so the
    # Ruling-10 redesign leaves every other scenario's rng stream untouched.
    bank += [_mode_prc_iso(*_PRC_GRID[i], i) for i in range(10)]
    bank += [_mode_hab(rng, i) for i in range(4)]
    bank += [_dep_time(rng, i) for i in range(30)]
    bank += [_route_vot(rng, i) for i in range(13)]
    bank += [_route_sch(rng, i) for i in range(8)]
    bank += [_route_hab(rng, i) for i in range(4)]
    bank += [_policy_prc(rng, i) for i in range(15)]
    bank += [_policy_mod(rng, i) for i in range(6)]
    bank += [_policy_vot(rng, i) for i in range(4)]
    bank += [_distractor(i) for i in range(10)]
    assert len(bank) == 140
    assert len({s["sid"] for s in bank}) == 140
    return bank


def public_view(scenario: dict) -> dict:
    """What the traveler and the system side may see: text + attributes, no design."""
    return {
        "sid": scenario["sid"],
        "block": scenario["block"],
        "text": scenario["text"],
        "options": scenario["options"],
    }


def design_view(scenario: dict) -> dict:
    """The Wall-protected design: primary dimension, option scores, tags."""
    return {
        "sid": scenario["sid"],
        "block": scenario["block"],
        "primary": scenario["primary"],
        "scores": scenario["scores"],
        "tags": scenario["tags"],
    }
