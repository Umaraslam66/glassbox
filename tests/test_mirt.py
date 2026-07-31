"""Unit tests for the Stage-2 item-response fit (``src/model/mirt.py``).

The centrepiece is an end-to-end check on a population this file makes up
itself, from a known four-dimensional model with its own random stream. If the
machinery can recover a model it generated, the machinery works, and any
shortfall on the real answer matrix is about the data rather than the code.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.model import mirt


# --------------------------------------------------------------------------
# a small synthetic world, generated here and nowhere else
# --------------------------------------------------------------------------


def make_population(
    *,
    n_personas: int = 400,
    n_likert: int = 90,
    n_binary: int = 40,
    dims: int = 4,
    seed: int = 4242,
    loadings: np.ndarray | None = None,
) -> tuple[mirt.Dataset, np.ndarray, mirt.ItemParams]:
    """Draw people and items from a known model and sample their answers.

    Pass ``loadings`` to generate from a loading matrix of your own; otherwise
    one is drawn here, with a clear primary dimension per item.
    """
    rng = np.random.default_rng(seed)
    kinds = [mirt.LIKERT] * n_likert + [mirt.BINARY] * n_binary
    n_items = len(kinds)

    theta = rng.normal(size=(n_personas, dims))
    if loadings is None:
        loadings = rng.normal(0.0, 0.75, size=(n_items, dims))
        # give each item a clear primary dimension, as a real bank would
        primary = rng.integers(0, dims, size=n_items)
        loadings[np.arange(n_items), primary] += rng.choice([-1.4, 1.4], size=n_items)
    else:
        loadings = np.asarray(loadings, dtype=float)
        if loadings.shape != (n_items, dims):
            raise ValueError(f"loadings must be {(n_items, dims)}, got {loadings.shape}")

    cut_raw = np.zeros((n_items, mirt.LIKERT_CUTS))
    cut_raw[:, 0] = rng.normal(-1.6, 0.6, n_items)
    cut_raw[:, 1:] = mirt.inv_softplus(np.abs(rng.normal(1.1, 0.25, (n_items, 3))))
    params = mirt.ItemParams(loadings, cut_raw, kinds)

    z = theta @ loadings.T
    taus = params.thresholds()
    codes = np.zeros((n_personas, n_items), dtype=np.int16)
    for j, kind in enumerate(kinds):
        if kind == mirt.LIKERT:
            seen = mirt.sigmoid(z[:, j : j + 1] - taus[j][None, :])
            cumulative = np.concatenate(
                [np.ones((n_personas, 1)), seen, np.zeros((n_personas, 1))], axis=1
            )
            probs = cumulative[:, :-1] - cumulative[:, 1:]
            draw = rng.random((n_personas, 1))
            codes[:, j] = (np.cumsum(probs, axis=1) < draw).sum(axis=1)
        else:
            probs = mirt.sigmoid(z[:, j] - cut_raw[j, 0])
            codes[:, j] = (rng.random(n_personas) < probs).astype(np.int16)

    data = mirt.Dataset(
        persona_ids=[f"x{i:04d}" for i in range(n_personas)],
        item_ids=[f"i{j:03d}" for j in range(n_items)],
        item_kinds=kinds,
        codes=codes,
    )
    return data, theta, params


@pytest.fixture(scope="module")
def population():
    return make_population()


def align(fitted: np.ndarray, target: np.ndarray) -> np.ndarray:
    return fitted @ mirt.orthogonal_map(fitted, target)


def per_dim_r(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array([np.corrcoef(a[:, d], b[:, d])[0, 1] for d in range(a.shape[1])])


# --------------------------------------------------------------------------
# the gradients the whole fit rests on
# --------------------------------------------------------------------------


def test_analytic_gradients_match_finite_differences() -> None:
    rng = np.random.default_rng(11)
    kinds = [mirt.LIKERT] * 5 + [mirt.BINARY] * 4
    codes = np.array(
        [
            [
                rng.integers(0, 5) if k == mirt.LIKERT else rng.integers(0, 2)
                for k in kinds
            ]
            for _ in range(12)
        ],
        dtype=np.int16,
    )
    data = mirt.Dataset(
        [f"p{i}" for i in range(12)], [f"q{j}" for j in range(len(kinds))], kinds, codes
    )
    blocks = mirt.make_blocks(data)
    theta = rng.normal(size=(12, 3))
    params = mirt.ItemParams(rng.normal(size=(len(kinds), 3)), rng.normal(size=(len(kinds), 4)), kinds)

    _, g_theta, g_load, g_cut = mirt._loglik_and_grads(theta, params, blocks)

    def value(th, ld, cr):
        return mirt._loglik_and_grads(
            th, mirt.ItemParams(ld, cr, kinds), blocks, want_item_grads=False
        )[0]

    step = 1e-6

    def numeric(array, evaluate):
        out = np.zeros_like(array)
        for index in np.ndindex(array.shape):
            up = array.copy()
            up[index] += step
            down = array.copy()
            down[index] -= step
            out[index] = (evaluate(up) - evaluate(down)) / (2 * step)
        return out

    assert np.allclose(
        g_theta, numeric(theta, lambda a: value(a, params.loadings, params.cut_raw)), atol=1e-5
    )
    assert np.allclose(
        g_load, numeric(params.loadings, lambda a: value(theta, a, params.cut_raw)), atol=1e-5
    )
    numeric_cut = numeric(params.cut_raw, lambda a: value(theta, params.loadings, a))
    mask = np.ones_like(numeric_cut)
    for j, kind in enumerate(kinds):
        if kind == mirt.BINARY:
            mask[j, 1:] = 0.0  # unused columns on a binary item
    assert np.allclose(g_cut * mask, numeric_cut * mask, atol=1e-5)


def test_thresholds_come_out_ordered_whatever_the_parameters() -> None:
    rng = np.random.default_rng(3)
    params = mirt.ItemParams(
        rng.normal(size=(50, 4)), rng.normal(0, 5, size=(50, 4)), [mirt.LIKERT] * 50
    )
    gaps = np.diff(params.thresholds(), axis=1)
    assert (gaps > 0).all(), "the parameterization must keep the cut points in order"


# --------------------------------------------------------------------------
# end to end on the synthetic population
# --------------------------------------------------------------------------


def test_recovers_a_model_it_generated(population) -> None:
    data, theta, params = population
    train = data.subset(data.persona_ids[:300], data.item_ids)

    fit = mirt.fit_joint(train, dims=4, seed=0, max_iters=4000)
    assert fit.converged, "the synthetic fit should reach its convergence check"

    correlations = per_dim_r(align(fit.theta, theta[:300]), theta[:300])
    assert correlations.min() > 0.9, f"trait recovery too weak: {correlations}"


def test_recovers_the_item_loadings_it_generated(population) -> None:
    data, theta, params = population
    train = data.subset(data.persona_ids[:300], data.item_ids)
    fit = mirt.fit_joint(train, dims=4, seed=0, max_iters=4000)

    rotation = mirt.orthogonal_map(fit.theta, theta[:300])
    fitted = fit.params.loadings @ rotation
    per_item = [
        np.corrcoef(fitted[j], params.loadings[j])[0, 1] for j in range(data.n_items)
    ]
    assert np.median(per_item) > 0.9, f"item recovery too weak: {np.median(per_item)}"


def test_scores_unseen_personas_from_frozen_item_parameters(population) -> None:
    data, theta, params = population
    train = data.subset(data.persona_ids[:300], data.item_ids)
    held_out = data.subset(data.persona_ids[300:], data.item_ids)

    fit = mirt.fit_joint(train, dims=4, seed=0, max_iters=4000)
    _, covs, _ = mirt.laplace_covariance(train, fit.params, fit.theta)
    scale = mirt.link_scale(fit.theta, covs)
    theta_train, linked, _ = mirt.apply_link(scale, fit.theta, fit.params, covs)

    scored, _ = mirt.score_personas(held_out, linked, dims=4, seed=0)
    rotation = mirt.orthogonal_map(theta_train, theta[:300])
    correlations = per_dim_r(scored @ rotation, theta[300:])
    assert correlations.min() > 0.9, f"held-out scoring too weak: {correlations}"


def test_blur_coverage_is_near_nominal_when_the_items_are_known(population) -> None:
    """The Laplace interval's own honesty, with item parameters out of the picture.

    Scoring against the parameters the answers were generated from isolates the
    posterior approximation. Nominal is 68 percent; anything in a wide band
    around it means the Hessian and its inverse are doing their job.
    """
    data, theta, params = population
    held_out = data.subset(data.persona_ids[300:], data.item_ids)

    scored, _ = mirt.score_personas(held_out, params, dims=4, seed=0)
    blur, _, bad = mirt.laplace_covariance(held_out, params, scored)
    assert not bad, "every Hessian should be positive definite at a MAP point"

    coverage = float((np.abs(scored - theta[300:]) <= blur).mean())
    assert 0.55 <= coverage <= 0.80, f"blur coverage off nominal: {coverage}"

    predicted = float(np.mean(blur**2))
    actual = float(np.mean((scored - theta[300:]) ** 2))
    assert 0.7 < actual / predicted < 1.5, (
        f"predicted variance {predicted:.4f} does not match actual {actual:.4f}"
    )


def test_the_metric_link_changes_units_and_nothing_else(population) -> None:
    """Linking must leave the likelihood and every correlation untouched."""
    data, theta, _ = population
    train = data.subset(data.persona_ids[:300], data.item_ids)
    fit = mirt.fit_joint(train, dims=4, seed=0, max_iters=1500)
    _, covs, _ = mirt.laplace_covariance(train, fit.params, fit.theta)

    before = mirt.mean_loglik_per_cell(train, fit.theta, fit.params)
    scale = mirt.link_scale(fit.theta, covs)
    theta_linked, linked, covs_linked = mirt.apply_link(scale, fit.theta, fit.params, covs)
    after = mirt.mean_loglik_per_cell(train, theta_linked, linked)

    assert scale > 0
    assert abs(before - after) < 1e-9, "linking moved the likelihood"
    assert np.allclose(
        per_dim_r(align(theta_linked, theta[:300]), theta[:300]),
        per_dim_r(align(fit.theta, theta[:300]), theta[:300]),
    ), "linking moved a correlation"

    spread = float(np.mean(theta_linked**2))
    posterior = float(np.mean(np.diagonal(covs_linked, axis1=1, axis2=2)))
    assert abs(spread + posterior - 1.0) < 1e-9, "the latent variance is not pinned at 1"


def test_restarts_that_agree_are_reported_as_agreeing(population) -> None:
    data, _, _ = population
    train = data.subset(data.persona_ids[:300], data.item_ids)
    fits = [mirt.fit_joint(train, dims=4, seed=s, max_iters=3000) for s in (0, 1)]
    report = mirt.restart_agreement(fits)
    assert report["worst_theta_corr"] > 0.9
    assert report["worst_canonical"] > 0.9
    assert len(report["pairs"]) == 1


def test_a_rotated_solution_is_the_same_solution(population) -> None:
    """Any rotation of a fit has the same likelihood -- that is why alignment is needed."""
    data, _, _ = population
    train = data.subset(data.persona_ids[:300], data.item_ids)
    fit = mirt.fit_joint(train, dims=4, seed=0, max_iters=1200)

    rng = np.random.default_rng(9)
    rotation, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    rotated_params = mirt.ItemParams(
        fit.params.loadings @ rotation, fit.params.cut_raw.copy(), list(fit.params.kinds)
    )
    assert np.isclose(
        mirt.mean_loglik_per_cell(train, fit.theta, fit.params),
        mirt.mean_loglik_per_cell(train, fit.theta @ rotation, rotated_params),
    )


# --------------------------------------------------------------------------
# fitting items against frozen people
# --------------------------------------------------------------------------


def test_fits_new_items_against_frozen_person_vectors(population) -> None:
    data, theta, params = population
    subset_ids = data.item_ids[:40]
    block = data.subset(data.persona_ids[:300], subset_ids)

    fitted, curve = mirt.fit_items(block, theta[:300], dims=4, seed=0)
    assert curve[-1] < curve[0]

    per_item = [
        np.corrcoef(fitted.loadings[j], params.loadings[j])[0, 1] for j in range(len(subset_ids))
    ]
    assert np.median(per_item) > 0.9, f"held-out item recovery too weak: {np.median(per_item)}"


# --------------------------------------------------------------------------
# loading and bookkeeping
# --------------------------------------------------------------------------


def test_loader_reads_the_answer_field_and_the_requested_round(tmp_path) -> None:
    bank_file = tmp_path / "bank.json"
    bank_file.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "item_id": "q1",
                        "type": "likert5",
                        "options": [{"value": v, "label": str(v)} for v in range(1, 6)],
                    },
                    {
                        "item_id": "q2",
                        "type": "binary",
                        "options": [{"value": 1, "label": "yes"}, {"value": 0, "label": "no"}],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    answers = tmp_path / "answers.jsonl"
    answers.write_text(
        "\n".join(
            json.dumps(record)
            for record in [
                {"pid": "p1", "item_id": "q1", "round": "main", "answer": 4, "raw": "1"},
                {"pid": "p1", "item_id": "q2", "round": "main", "answer": "yes", "raw": "no"},
                {"pid": "p2", "item_id": "q1", "round": "main", "answer": 2, "raw": "2"},
                {"pid": "p2", "item_id": "q2", "round": "main", "answer": "no", "raw": "no"},
                {"pid": "p1", "item_id": "q1", "round": "retest", "answer": 1, "raw": "1"},
            ]
        ),
        encoding="utf-8",
    )

    bank = mirt.load_bank(bank_file)
    data = mirt.load_answers(answers, bank)

    assert data.persona_ids == ["p1", "p2"]
    assert data.item_ids == ["q1", "q2"]
    # coded from ``answer``, never from the pre-noise string beside it
    assert data.codes.tolist() == [[3, 1], [1, 0]]

    retest = mirt.load_answers(answers, bank, round_name="retest")
    assert retest.codes.tolist() == [[0, -1]]


def test_missing_report_counts_holes(population) -> None:
    data, _, _ = population
    report = mirt.missing_report(data)
    assert report["complete"] is True
    assert report["n_cells_missing"] == 0
    assert report["n_cells_expected"] == data.n_personas * data.n_items

    holed = mirt.Dataset(data.persona_ids, data.item_ids, data.item_kinds, data.codes.copy())
    holed.codes[0, 0] = -1
    holed_report = mirt.missing_report(holed)
    assert holed_report["complete"] is False
    assert holed_report["n_cells_missing"] == 1


def test_missing_cells_do_not_enter_the_likelihood(population) -> None:
    data, _, params = population
    theta = np.zeros((data.n_personas, 4))
    full = mirt.mean_loglik_per_cell(data, theta, params)

    holed = mirt.Dataset(data.persona_ids, data.item_ids, data.item_kinds, data.codes.copy())
    holed.codes[:, 0] = -1
    blocks_full = mirt.make_blocks(data)
    blocks_holed = mirt.make_blocks(holed)
    loglik_full, _, _, _ = mirt._loglik_and_grads(theta, params, blocks_full)
    loglik_holed, _, _, _ = mirt._loglik_and_grads(theta, params, blocks_holed)

    assert np.isfinite(full)
    assert loglik_holed > loglik_full  # one item's worth of negative terms removed


def test_subset_keeps_labels_and_cells_together(population) -> None:
    data, _, _ = population
    picked_personas = [data.persona_ids[5], data.persona_ids[1]]
    picked_items = [data.item_ids[3], data.item_ids[0]]
    small = data.subset(picked_personas, picked_items)

    assert small.persona_ids == picked_personas
    assert small.item_ids == picked_items
    assert small.codes[0, 0] == data.codes[5, 3]
    assert small.codes[1, 1] == data.codes[1, 0]
    assert small.item_kinds == [data.item_kinds[3], data.item_kinds[0]]
