"""Offline tests for the Stage M2 estimator (no network, no truth access)."""

import numpy as np
import pytest

from mobility.src.model.mnl import (PARAMS, build_design, fit_traveler,
                                    option_features)


def test_option_features_mode_block():
    f = option_features("mode", {"time_min": 30, "cost_eur": 2.5, "crowding": 2,
                                 "mode": "car"})
    assert f == [30.0, 2.5, 60.0, 1.0, 0.0, 0.0, 0.0]


def test_option_features_dep_block_uses_slack_and_expected_late():
    f = option_features("dep_time", {"slack_min": 0, "p_late": 0.2, "late_min": 10,
                                     "cost_eur": 0.0})
    assert f[0] == 0.0 and f[5] == pytest.approx(2.0)
    g = option_features("dep_time", {"slack_min": 25, "p_late": 0, "cost_eur": 0.0})
    assert g[0] == 25.0 and g[5] == 0.0


def test_option_features_route_late_from_time_range():
    f = option_features("route", {"time_min": 31, "time_max": 47, "p_late": 0.2,
                                  "cost_eur": 0.0})
    assert f[0] == 31.0 and f[5] == pytest.approx(0.2 * 16)


def test_option_features_habitual_becomes_switch_indicator():
    stick = option_features("mode", {"time_min": 31, "cost_eur": 3.1, "habitual": 1})
    move = option_features("mode", {"time_min": 29, "cost_eur": 3.1, "habitual": 0})
    assert stick[6] == 0.0 and move[6] == 1.0


def _toy_bank(rng):
    bank = []
    for i in range(70):
        kind = i % 3
        if kind == 0:
            t1, t2 = rng.integers(20, 40), rng.integers(30, 55)
            c1, c2 = rng.uniform(2, 9), rng.uniform(0.5, 3)
            opts = [{"key": "A", "attrs": {"time_min": int(t1), "cost_eur": round(c1, 2),
                                           "mode": "car"}},
                    {"key": "B", "attrs": {"time_min": int(t2), "cost_eur": round(c2, 2),
                                           "mode": "transit",
                                           "crowding": int(rng.integers(0, 4))}}]
            block = "mode"
        elif kind == 1:
            opts = [{"key": "A", "attrs": {"slack_min": int(rng.integers(5, 40)),
                                           "p_late": 0, "cost_eur": 0.0}},
                    {"key": "B", "attrs": {"slack_min": 0,
                                           "p_late": float(rng.uniform(0.05, 0.4)),
                                           "late_min": int(rng.integers(5, 25)),
                                           "cost_eur": 0.0}}]
            block = "dep_time"
        else:
            opts = [{"key": "A", "attrs": {"time_min": int(rng.integers(25, 35)),
                                           "habitual": 1, "cost_eur": 0.0}},
                    {"key": "B", "attrs": {"time_min": int(rng.integers(20, 30)),
                                           "habitual": 0, "cost_eur": 0.0}}]
            block = "route"
        bank.append({"sid": f"s{i:03d}", "block": block, "options": opts})
    return bank


def test_synthetic_recovery_orders_travelers():
    """Choices simulated from known coefficients: the fit must recover the
    cross-traveler ordering of the planted coefficients."""
    rng = np.random.default_rng(7)
    bank = _toy_bank(rng)
    design = build_design(bank)
    sids = [s["sid"] for s in bank]
    n = 40
    true_time = -np.exp(rng.normal(np.log(0.08), 0.5, n))
    true_cost = -np.exp(rng.normal(np.log(0.6), 0.5, n))
    true_switch = -np.exp(rng.normal(np.log(0.8), 0.7, n))
    est_time, est_cost, est_switch = [], [], []
    for i in range(n):
        beta = np.array([true_time[i], true_cost[i], -0.01, 0.3, 0.1,
                         2.0 * true_time[i], true_switch[i]])
        answers = {}
        for sid in sids:
            X = design[sid]["X"]
            u = X @ beta
            p = np.exp(u - u.max())
            p /= p.sum()
            answers[sid] = design[sid]["keys"][rng.choice(len(p), p=p)]
        fit = fit_traveler(design, answers, sids)
        est_time.append(fit[0])
        est_cost.append(fit[1])
        est_switch.append(fit[6])
    assert np.corrcoef(true_time, est_time)[0, 1] > 0.7
    # cost varies in only a third of the toy scenarios, so its MLE is noisier
    assert np.corrcoef(true_cost, est_cost)[0, 1] > 0.6
    assert np.corrcoef(true_switch, est_switch)[0, 1] > 0.6


def test_fit_requires_minimum_scenarios():
    rng = np.random.default_rng(3)
    bank = _toy_bank(rng)
    design = build_design(bank)
    answers = {s["sid"]: "A" for s in bank[:5]}
    assert fit_traveler(design, answers, [s["sid"] for s in bank[:5]]) is None


def test_estimator_imports_nothing_from_eval():
    import mobility.src.model.mnl as m
    src = open(m.__file__).read()
    assert "mobility.src.eval" not in src  # the wall test enforces the rest
