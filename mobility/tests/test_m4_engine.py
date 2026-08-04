"""Offline tests for the M4 engine mechanics and the frozen index formulas."""

import numpy as np
import pytest

from mobility.src.engine.bottleneck import (lateness_exposure, peak_15min_share,
                                            switching_frequency,
                                            tau_interpolator,
                                            vickrey_travel_times)


def test_uncongested_travelers_drive_free_flow():
    deps = np.array([0.0, 10.0, 20.0])
    travel = vickrey_travel_times(deps, capacity_per_min=2.0, free_flow=18.0)
    assert np.allclose(travel, 18.0)


def test_queue_builds_and_dissipates_fifo():
    # 5 travelers at the same minute, capacity 1/min: delays 0,1,2,3,4
    deps = np.zeros(5)
    travel = vickrey_travel_times(deps, capacity_per_min=1.0, free_flow=18.0)
    assert sorted(np.round(travel - 18.0, 6).tolist()) == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_vickrey_preserves_input_order():
    deps = np.array([30.0, 0.0, 30.0])
    travel = vickrey_travel_times(deps, capacity_per_min=1.0)
    assert travel[1] == pytest.approx(18.0)  # the early traveler never queues


def test_tau_interpolates_and_stays_flat_outside():
    tau = tau_interpolator(np.array([0.0, 10.0]), np.array([20.0, 30.0]))
    assert tau(5.0) == pytest.approx(25.0)
    assert tau(-100.0) == pytest.approx(20.0)
    assert tau(100.0) == pytest.approx(30.0)


def test_switching_frequency_frozen_formula():
    # 20 days, one switch -> 1/19
    routes = ["A"] * 10 + ["B"] * 10
    assert switching_frequency(routes) == pytest.approx(1 / 19)
    assert switching_frequency(["A"] * 20) == 0.0


def test_lateness_exposure_frozen_formula():
    # PAT 100, buffer 5: danger iff dep + tau(dep) > 95
    flat_20 = {"A": lambda d: 20.0, "B": lambda d: 20.0}
    profiles = [flat_20, flat_20, flat_20]
    deps = [70.0, 80.0, 70.0]      # day2: 80+20=100 > 95 late; day3: 90 safe
    routes = ["A", "A", "A"]
    x = lateness_exposure(deps, routes, profiles, pat=100.0)
    assert x == pytest.approx(1 / 2)


def test_peak_share_sliding_window_maximum():
    # 10 arrivals inside one 15-min span, 10 spread far apart
    arrivals = np.concatenate([np.linspace(0, 14, 10),
                               np.arange(100, 400, 30)])
    assert peak_15min_share(arrivals) == pytest.approx(0.5)
    # a shifted-but-not-flattened peak keeps the same sliding maximum
    assert peak_15min_share(arrivals + 37) == pytest.approx(0.5)
