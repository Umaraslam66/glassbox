"""M4 world mechanics — deterministic Vickrey bottleneck + frozen indices.

Pure functions, no paths, no truth access. The world (pre-declared at M3
close, commit 4f1b741): two routes into the CBD — route A free-flow
FREE_FLOW_A minutes through one point-queue bottleneck of capacity s
vehicles/minute; route B a steady ROUTE_B_TIME minutes, never congested.
Single mode, so the frozen switching index W_i reduces to its route term.

Frozen index formulas (prereg §5, Ruling 8):
  W_i = (1/19) * sum_{t=2..20} 1[route_{i,t} != route_{i,t-1}]
  X_i = (1/19) * sum_{t=2..20} 1[d_{i,t} + tau_{t-1}(d_{i,t}) > PAT_i - b],
        b = 5 minutes, tau_{t-1} = the previous day's realized travel-time
        profile on the chosen route (linear interpolation between observed
        departures, flat beyond the observed range).
"""

from __future__ import annotations

import numpy as np

FREE_FLOW_A = 18.0
ROUTE_B_TIME = 26.0
LATE_BUFFER_MIN = 5.0


def vickrey_travel_times(departures: np.ndarray, capacity_per_min: float,
                         free_flow: float = FREE_FLOW_A) -> np.ndarray:
    """Deterministic FIFO point queue: travel time per traveler, input order.

    Travelers reach the bottleneck at their departure minute, are served at
    ``capacity_per_min``, then drive the free-flow time.
    """
    if len(departures) == 0:
        return np.array([])
    order = np.argsort(departures, kind="stable")
    headway = 1.0 / capacity_per_min
    exits = np.empty(len(order))
    prev = -np.inf
    for k, idx in enumerate(order):
        prev = max(departures[idx], prev + headway)
        exits[k] = prev
    travel = np.empty(len(departures))
    travel[order] = (exits - departures[order]) + free_flow
    return travel


def tau_interpolator(departures: np.ndarray, travels: np.ndarray):
    """Previous-day realized profile tau(d): linear between observed
    departures, flat beyond the observed range (pre-declared)."""
    if len(departures) == 0:
        return lambda d: FREE_FLOW_A
    order = np.argsort(departures, kind="stable")
    xs, ys = departures[order], travels[order]

    def tau(d: float) -> float:
        return float(np.interp(d, xs, ys))

    return tau


def switching_frequency(routes_by_day: list) -> float:
    """Frozen W_i on one traveler's route sequence (day 1 first)."""
    T = len(routes_by_day)
    switches = sum(1 for t in range(1, T)
                   if routes_by_day[t] != routes_by_day[t - 1])
    return switches / (T - 1)


def lateness_exposure(deps_by_day: list, routes_by_day: list,
                      day_profiles: list, pat: float,
                      buffer_min: float = LATE_BUFFER_MIN) -> float:
    """Frozen X_i. ``day_profiles[t]`` maps route -> tau callable for day
    t's realized profile (0-indexed by day; entry t-1 is 'yesterday')."""
    T = len(deps_by_day)
    hits = 0
    for t in range(1, T):
        tau = day_profiles[t - 1][routes_by_day[t]]
        if deps_by_day[t] + tau(deps_by_day[t]) > pat - buffer_min:
            hits += 1
    return hits / (T - 1)


def peak_15min_share(arrivals: np.ndarray) -> float:
    """Frozen §5 peak metric: the MAXIMUM share of the day's arrivals in any
    sliding 15-minute window, 1-minute steps."""
    if len(arrivals) == 0:
        return 0.0
    lo = int(np.floor(arrivals.min()))
    hi = int(np.ceil(arrivals.max()))
    best = 0
    for start in range(lo, hi + 1):
        best = max(best, int(((arrivals >= start) & (arrivals < start + 15)).sum()))
    return best / len(arrivals)
