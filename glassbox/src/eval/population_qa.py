"""Gate 1 population QA: diversity, trait obedience, test-retest.

    python -m src.eval.population_qa \\
        --answers <answers.jsonl> --truth <dir> --public-bank <bank_items.json> \\
        --out <results.json> [--retest-answers <file>]

Grader-side code: this is inside ``src/eval/``, so it is allowed to read the
planted truth. It reads the planted trait vectors through the truth-store
accessor and the bank's designed loadings from ``bank_truth.json`` in the same
directory.

Every number here implements PREREGISTRATION.md sections 5 and 6 as written.
Nothing is invented and nothing is softened:

1. **Diversity.** (a) pairwise agreement between two personas is the fraction
   of items they answer *exactly* the same -- exact match on binary and on
   Likert-5 alike; reported as the max pair, the median pair and a histogram.
   (b) effective dimensionality is the participation ratio (sum L)^2 / sum L^2
   of the eigenvalues L of the covariance of the standardized personas x items
   matrix (Likert as 1-5, binary as 0/1, z-scored per item). The spectrum goes
   into the output for the dashboard.
2. **Trait obedience.** For each dimension, take every closed item designed to
   load on it at |loading| >= 0.5 -- primary items and the cross-loading items
   where it sits at 0.7 or 0.5. Z-score each item across personas, flip the
   sign on negatively-loading items, average per persona, then correlate that
   average across personas with the planted theta. Bar: median of the 8
   correlations >= 0.5.
3. **Test-retest.** Compare each re-asked (persona, item) with the *same*
   persona's main-round answer on the *same* item: exact match on binary,
   within +-1 on Likert-5. Pooled rate, per-persona rates, and the binary and
   Likert rates separately with their chance levels (50% and 52%).

The verdict block states each frozen Gate 1 bar with a pass/fail boolean. A bar
with no data behind it is marked not evaluated and the gate does not pass.
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from ..bank.schema import DIMENSIONS, PRIVATE_FILENAME
from . import truth_store

#: Chance level for binary exact-match agreement under coin-flip answering.
BINARY_CHANCE = 0.5

#: Chance level for Likert-5 within-+-1 agreement under uniform answering:
#: 13 of the 25 equally likely (answer, re-answer) pairs are within one step.
LIKERT_CHANCE = 13 / 25

#: The frozen Gate 1 bars (PREREGISTRATION.md section 6).
MAX_PAIR_AGREEMENT_BAR = 0.95
MEDIAN_PAIR_AGREEMENT_BAR = 0.80
EFFECTIVE_DIM_BAR = 5.0
OBEDIENCE_MEDIAN_BAR = 0.5
RETEST_BAND = (0.70, 0.90)

#: An item counts as "designed to load on dimension d" at this magnitude.
LOADING_THRESHOLD = 0.5

_TRUTH_DIR_ENV = "GLASSBOX_TRUTH_DIR"


# --------------------------------------------------------------------------
# reading the inputs
# --------------------------------------------------------------------------


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def item_types(public_bank: str | Path) -> dict[str, str]:
    """Item id -> "likert5" or "binary", from the public bank."""
    payload = json.loads(Path(public_bank).read_text(encoding="utf-8"))
    return {str(row["item_id"]): str(row["type"]) for row in payload.get("items", [])}


def read_bank_design(truth_dir: str | Path) -> dict[str, dict[str, float]]:
    """Item id -> designed loadings, from ``bank_truth.json`` in the truth dir."""
    path = Path(truth_dir) / PRIVATE_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(row["item_id"]): {str(k): float(v) for k, v in (row.get("loadings") or {}).items()}
        for row in payload.get("items", [])
    }


@contextmanager
def truth_dir(path: str | Path) -> Iterator[None]:
    """Point the truth-store accessor at ``path`` for the duration of the block.

    The accessor takes its directory from the environment so a run can keep its
    planted truth anywhere. Setting it here, and putting it back afterwards,
    keeps the directory a command-line argument rather than a constant.
    """
    previous = os.environ.get(_TRUTH_DIR_ENV)
    os.environ[_TRUTH_DIR_ENV] = str(path)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_TRUTH_DIR_ENV, None)
        else:
            os.environ[_TRUTH_DIR_ENV] = previous


def load_thetas(path: str | Path, pids: Sequence[str]) -> dict[str, dict[str, float]]:
    """Planted trait vectors for the personas we have answers for."""
    found: dict[str, dict[str, float]] = {}
    with truth_dir(path):
        for pid in pids:
            try:
                record = truth_store.load_theta(pid)
            except FileNotFoundError:
                continue
            theta = record.get("theta") if isinstance(record, Mapping) and "theta" in record else record
            found[pid] = {str(k): float(v) for k, v in dict(theta).items()}
    return found


def code_answer(answer: Any, item_type: str) -> float:
    """One stored answer as a number: Likert stays 1-5, binary becomes 1/0."""
    if item_type == "likert5":
        return float(int(answer))
    if item_type == "binary":
        if isinstance(answer, str):
            text = answer.strip().lower()
            if text in ("yes", "no"):
                return 1.0 if text == "yes" else 0.0
            raise ValueError(f"binary answer {answer!r} is neither yes nor no")
        return 1.0 if int(answer) else 0.0
    raise ValueError(f"unknown item type {item_type!r}")


def build_matrix(
    answers: Sequence[Mapping[str, Any]],
    types: Mapping[str, str],
    round_name: str = "main",
) -> tuple[list[str], list[str], np.ndarray]:
    """Personas x items answer matrix for one round; missing cells are NaN."""
    rows = [row for row in answers if str(row.get("round") or "main") == round_name]
    pids = sorted({str(row["pid"]) for row in rows})
    item_ids = sorted({str(row["item_id"]) for row in rows if str(row["item_id"]) in types})
    unknown = {str(row["item_id"]) for row in rows} - set(item_ids)
    if unknown:
        print(
            f"warning: {len(unknown)} item ids in the answers are not in the public "
            f"bank and are ignored, first: {sorted(unknown)[0]}"
        )

    pid_at = {pid: index for index, pid in enumerate(pids)}
    item_at = {item_id: index for index, item_id in enumerate(item_ids)}

    matrix = np.full((len(pids), len(item_ids)), np.nan)
    for row in rows:
        item_id = str(row["item_id"])
        if item_id not in item_at:
            continue
        matrix[pid_at[str(row["pid"])], item_at[item_id]] = code_answer(
            row["answer"], types[item_id]
        )
    return pids, item_ids, matrix


# --------------------------------------------------------------------------
# 1. diversity
# --------------------------------------------------------------------------


def pairwise_agreements(matrix: np.ndarray) -> np.ndarray:
    """Exact-match agreement for every persona pair, over the items both answered."""
    n_people = matrix.shape[0]
    if n_people < 2:
        return np.zeros(0)

    answered = ~np.isnan(matrix)
    chunks: list[np.ndarray] = []
    for index in range(n_people - 1):
        others = matrix[index + 1 :]
        shared = answered[index] & answered[index + 1 :]
        denominator = shared.sum(axis=1)
        # NaN never equals anything, so unanswered cells drop out on their own.
        matches = ((matrix[index] == others) & shared).sum(axis=1)
        chunks.append(
            np.where(denominator > 0, matches / np.maximum(denominator, 1), np.nan)
        )
    return np.concatenate(chunks)


def standardize(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Z-score each item across personas. Returns the matrix and the kept items.

    An item everybody answers the same way has no variance and no z-score; it
    is dropped and counted. Missing cells sit at the item mean (z = 0), which
    is the least-informative filling and keeps the covariance defined.
    """
    answered = ~np.isnan(matrix)
    counts = answered.sum(axis=0)
    with np.errstate(invalid="ignore"):
        means = np.where(counts > 0, np.nansum(np.where(answered, matrix, 0.0), axis=0) / np.maximum(counts, 1), 0.0)
        centered = np.where(answered, matrix - means, 0.0)
        variances = np.where(counts > 1, (centered**2).sum(axis=0) / np.maximum(counts - 1, 1), 0.0)
    stds = np.sqrt(variances)

    keep = (counts > 1) & (stds > 1e-12)
    standardized = np.zeros_like(centered)
    standardized[:, keep] = centered[:, keep] / stds[keep]
    return standardized[:, keep], keep


def effective_dimensionality(standardized: np.ndarray) -> tuple[float, np.ndarray]:
    """Participation ratio (sum L)^2 / sum L^2 and the eigenvalue spectrum.

    The eigendecomposition is over the item-item covariance of the standardized
    matrix, which is the item correlation matrix.
    """
    if standardized.shape[0] < 2 or standardized.shape[1] < 1:
        return float("nan"), np.zeros(0)

    covariance = np.cov(standardized, rowvar=False, ddof=1)
    covariance = np.atleast_2d(covariance)
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = np.clip(eigenvalues, 0.0, None)[::-1]

    total = float(eigenvalues.sum())
    squared = float((eigenvalues**2).sum())
    ratio = (total**2) / squared if squared > 0 else float("nan")
    return ratio, eigenvalues


def _histogram(values: np.ndarray, bins: int, low: float = 0.0, high: float = 1.0) -> dict[str, Any]:
    counts, edges = np.histogram(values, bins=bins, range=(low, high))
    return {"bin_edges": [round(float(e), 6) for e in edges], "counts": [int(c) for c in counts]}


def diversity_report(matrix: np.ndarray, item_ids: Sequence[str]) -> dict[str, Any]:
    """The Gate 1 diversity block: agreement spread plus effective dimensionality."""
    agreements = pairwise_agreements(matrix)
    usable = agreements[~np.isnan(agreements)]

    standardized, keep = standardize(matrix)
    ratio, eigenvalues = effective_dimensionality(standardized)

    report: dict[str, Any] = {
        "n_personas": int(matrix.shape[0]),
        "n_items": int(matrix.shape[1]),
        "n_pairs": int(usable.size),
        "n_pairs_without_shared_items": int(agreements.size - usable.size),
        "max_pair_agreement": _round(usable.max()) if usable.size else None,
        "median_pair_agreement": _round(np.median(usable)) if usable.size else None,
        "mean_pair_agreement": _round(usable.mean()) if usable.size else None,
        "min_pair_agreement": _round(usable.min()) if usable.size else None,
        "percentiles": (
            {
                str(p): _round(np.percentile(usable, p))
                for p in (1, 5, 25, 50, 75, 90, 95, 99)
            }
            if usable.size
            else {}
        ),
        "histogram": _histogram(usable, bins=50) if usable.size else {"bin_edges": [], "counts": []},
        "effective_dimensionality": _round(ratio),
        "eigenvalues": [round(float(v), 6) for v in eigenvalues],
        "n_items_in_eigendecomposition": int(standardized.shape[1]),
        "items_dropped_no_variance": [
            item_ids[index] for index, kept in enumerate(keep) if not kept
        ],
        "n_missing_cells": int(np.isnan(matrix).sum()),
    }
    return report


# --------------------------------------------------------------------------
# 2. trait obedience
# --------------------------------------------------------------------------


def items_loading_on(design: Mapping[str, Mapping[str, float]], dimension: str) -> dict[str, float]:
    """Items designed to load on one dimension, with their signed loading."""
    return {
        item_id: float(loadings[dimension])
        for item_id, loadings in design.items()
        if abs(float(loadings.get(dimension, 0.0))) >= LOADING_THRESHOLD
    }


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def obedience_report(
    pids: Sequence[str],
    item_ids: Sequence[str],
    matrix: np.ndarray,
    design: Mapping[str, Mapping[str, float]],
    thetas: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """Correlation between planted theta and keyed mean answer, per dimension."""
    standardized, keep = standardize(matrix)
    kept_ids = [item_ids[index] for index, flag in enumerate(keep) if flag]
    column_at = {item_id: index for index, item_id in enumerate(kept_ids)}
    answered = ~np.isnan(matrix)
    answered_kept = answered[:, keep]

    per_dimension: dict[str, float | None] = {}
    counts: dict[str, int] = {}
    n_used: dict[str, int] = {}

    for dimension in DIMENSIONS:
        loadings = items_loading_on(design, dimension)
        columns = [(column_at[i], sign) for i, sign in loadings.items() if i in column_at]
        counts[dimension] = len(columns)
        if not columns:
            per_dimension[dimension] = None
            n_used[dimension] = 0
            continue

        indices = np.array([c for c, _ in columns])
        signs = np.sign(np.array([s for _, s in columns]))
        # Z-score first, then flip the negatively keyed items, then average.
        keyed = standardized[:, indices] * signs
        available = answered_kept[:, indices]
        with np.errstate(invalid="ignore"):
            totals = np.where(available, keyed, 0.0).sum(axis=1)
            n_answers = available.sum(axis=1)
            scores = np.where(n_answers > 0, totals / np.maximum(n_answers, 1), np.nan)

        usable = [
            index
            for index, pid in enumerate(pids)
            if pid in thetas and dimension in thetas[pid] and not np.isnan(scores[index])
        ]
        n_used[dimension] = len(usable)
        if len(usable) < 2:
            per_dimension[dimension] = None
            continue

        planted = np.array([float(thetas[pids[index]][dimension]) for index in usable])
        correlation = _pearson(scores[np.array(usable)], planted)
        per_dimension[dimension] = _round(correlation) if correlation is not None else None

    values = [v for v in per_dimension.values() if v is not None]
    return {
        "per_dimension": per_dimension,
        "median": _round(float(np.median(values))) if values else None,
        "n_dimensions_measured": len(values),
        "n_items_per_dimension": counts,
        "n_personas_per_dimension": n_used,
        "n_personas_with_theta": sum(1 for pid in pids if pid in thetas),
    }


# --------------------------------------------------------------------------
# 3. test-retest
# --------------------------------------------------------------------------


def agrees(main: Any, retest: Any, item_type: str) -> bool:
    """Exact match on binary, within one step on Likert-5."""
    if item_type == "binary":
        return code_answer(main, item_type) == code_answer(retest, item_type)
    if item_type == "likert5":
        return abs(code_answer(main, item_type) - code_answer(retest, item_type)) <= 1.0
    raise ValueError(f"unknown item type {item_type!r}")


def retest_report(
    main_answers: Sequence[Mapping[str, Any]],
    retest_answers: Sequence[Mapping[str, Any]],
    types: Mapping[str, str],
) -> dict[str, Any]:
    """Pooled and split test-retest agreement against the same persona's main answer."""
    main_at: dict[tuple[str, str], Any] = {}
    for row in main_answers:
        if str(row.get("round") or "main") == "main":
            main_at[(str(row["pid"]), str(row["item_id"]))] = row["answer"]

    per_persona: dict[str, list[bool]] = {}
    by_type: dict[str, list[bool]] = {"binary": [], "likert5": []}
    unmatched = 0
    pooled: list[bool] = []

    for row in retest_answers:
        if str(row.get("round") or "main") != "retest":
            continue
        pid, item_id = str(row["pid"]), str(row["item_id"])
        item_type = types.get(item_id)
        if item_type is None or (pid, item_id) not in main_at:
            unmatched += 1
            continue
        hit = agrees(main_at[(pid, item_id)], row["answer"], item_type)
        pooled.append(hit)
        per_persona.setdefault(pid, []).append(hit)
        by_type[item_type].append(hit)

    def rate(values: Sequence[bool]) -> float | None:
        return _round(float(np.mean(values))) if values else None

    persona_rates = {pid: _round(float(np.mean(hits))) for pid, hits in sorted(per_persona.items())}
    rates_array = np.array(list(persona_rates.values())) if persona_rates else np.zeros(0)

    return {
        "pooled_agreement": rate(pooled),
        "n_reasked": len(pooled),
        "n_unmatched": unmatched,
        "n_personas": len(per_persona),
        "per_persona": persona_rates,
        "per_persona_histogram": (
            _histogram(rates_array, bins=20) if rates_array.size else {"bin_edges": [], "counts": []}
        ),
        "per_persona_median": _round(float(np.median(rates_array))) if rates_array.size else None,
        "binary": {
            "agreement": rate(by_type["binary"]),
            "n": len(by_type["binary"]),
            "chance": BINARY_CHANCE,
            "rule": "exact match",
        },
        "likert5": {
            "agreement": rate(by_type["likert5"]),
            "n": len(by_type["likert5"]),
            "chance": _round(LIKERT_CHANCE),
            "rule": "within +-1",
        },
    }


# --------------------------------------------------------------------------
# verdict
# --------------------------------------------------------------------------


def _bar(label: str, value: float | None, rule: str, passed: bool | None) -> dict[str, Any]:
    return {
        "bar": label,
        "rule": rule,
        "value": value,
        "pass": passed,
        "evaluated": value is not None,
    }


def verdict(report: Mapping[str, Any]) -> dict[str, Any]:
    """Each frozen Gate 1 bar with its number and a pass/fail boolean."""
    diversity = report["diversity"]
    obedience = report["obedience"]
    retest = report.get("retest") or {}

    maximum = diversity.get("max_pair_agreement")
    median = diversity.get("median_pair_agreement")
    dimensionality = diversity.get("effective_dimensionality")
    obedience_median = obedience.get("median")
    pooled = retest.get("pooled_agreement")

    bars = {
        "diversity_a_no_pair_above_95": _bar(
            "no persona pair agrees on more than 95% of items",
            maximum,
            f"max pair agreement <= {MAX_PAIR_AGREEMENT_BAR}",
            None if maximum is None else bool(maximum <= MAX_PAIR_AGREEMENT_BAR),
        ),
        "diversity_b_median_agreement": _bar(
            "median pairwise agreement at most 80%",
            median,
            f"median pair agreement <= {MEDIAN_PAIR_AGREEMENT_BAR}",
            None if median is None else bool(median <= MEDIAN_PAIR_AGREEMENT_BAR),
        ),
        "diversity_c_effective_dimensionality": _bar(
            "effective dimensionality at least 5",
            dimensionality,
            f"participation ratio >= {EFFECTIVE_DIM_BAR}",
            None
            if dimensionality is None or (isinstance(dimensionality, float) and np.isnan(dimensionality))
            else bool(dimensionality >= EFFECTIVE_DIM_BAR),
        ),
        "trait_obedience_median": _bar(
            "median trait obedience across the 8 dimensions at least 0.5",
            obedience_median,
            f"median r >= {OBEDIENCE_MEDIAN_BAR}",
            None if obedience_median is None else bool(obedience_median >= OBEDIENCE_MEDIAN_BAR),
        ),
        "retest_pooled_band": _bar(
            "pooled test-retest agreement inside the 70-90% band",
            pooled,
            f"{RETEST_BAND[0]} <= pooled agreement <= {RETEST_BAND[1]}",
            None if pooled is None else bool(RETEST_BAND[0] <= pooled <= RETEST_BAND[1]),
        ),
    }
    return {"bars": bars, "all_pass": all(entry["pass"] is True for entry in bars.values())}


def _round(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    if np.isnan(number):
        return None
    return round(number, 6)


# --------------------------------------------------------------------------
# putting it together
# --------------------------------------------------------------------------


def run_qa(
    answers: Sequence[Mapping[str, Any]],
    types: Mapping[str, str],
    design: Mapping[str, Mapping[str, float]],
    thetas: Mapping[str, Mapping[str, float]],
    retest_answers: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """The whole Gate 1 report from already-loaded pieces."""
    pids, item_ids, matrix = build_matrix(answers, types, "main")

    report: dict[str, Any] = {
        "n_personas": len(pids),
        "n_items": len(item_ids),
        "n_answers_main": int((~np.isnan(matrix)).sum()),
        "diversity": diversity_report(matrix, item_ids),
        "obedience": obedience_report(pids, item_ids, matrix, design, thetas),
    }

    retest_rows = list(retest_answers) if retest_answers is not None else list(answers)
    if any(str(row.get("round") or "main") == "retest" for row in retest_rows):
        report["retest"] = retest_report(answers, retest_rows, types)
    else:
        report["retest"] = None

    report["verdict"] = verdict(report)
    return report


def print_table(report: Mapping[str, Any]) -> None:
    """A plain summary, one number per line."""
    diversity = report["diversity"]
    obedience = report["obedience"]
    retest = report.get("retest")

    print(f"personas {report['n_personas']}   items {report['n_items']}   "
          f"answers {report['n_answers_main']}")
    print("")
    print("DIVERSITY")
    print(f"  max pair agreement       {_show(diversity['max_pair_agreement'])}")
    print(f"  median pair agreement    {_show(diversity['median_pair_agreement'])}")
    print(f"  effective dimensionality {_show(diversity['effective_dimensionality'])}"
          f"   (over {diversity['n_items_in_eigendecomposition']} items)")
    print("")
    print("TRAIT OBEDIENCE (r with planted theta)")
    for dimension in DIMENSIONS:
        value = obedience["per_dimension"].get(dimension)
        print(f"  {dimension}  {_show(value)}   "
              f"({obedience['n_items_per_dimension'].get(dimension, 0)} items)")
    print(f"  median  {_show(obedience['median'])}")
    print("")
    print("TEST-RETEST")
    if not retest:
        print("  no retest answers supplied")
    else:
        print(f"  pooled          {_show(retest['pooled_agreement'])}   "
              f"({retest['n_reasked']} re-asked cells, {retest['n_personas']} personas)")
        print(f"  binary exact    {_show(retest['binary']['agreement'])}   "
              f"chance {retest['binary']['chance']}   n={retest['binary']['n']}")
        print(f"  likert within 1 {_show(retest['likert5']['agreement'])}   "
              f"chance {retest['likert5']['chance']}   n={retest['likert5']['n']}")
    print("")
    print("GATE 1 VERDICT")
    for entry in report["verdict"]["bars"].values():
        if entry["pass"] is None:
            mark = "not evaluated"
        else:
            mark = "PASS" if entry["pass"] else "FAIL"
        print(f"  {mark:>13}  {entry['bar']}  [{entry['rule']}, got {_show(entry['value'])}]")
    print(f"  gate 1: {'PASS' if report['verdict']['all_pass'] else 'FAIL'}")


def _show(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.eval.population_qa",
        description="Gate 1 population QA: diversity, trait obedience, test-retest.",
    )
    parser.add_argument("--answers", type=Path, required=True,
                        help="answers.jsonl from the main round")
    parser.add_argument("--truth", type=Path, required=True,
                        help="the batch's planted-truth directory: theta/ plus bank_truth.json")
    parser.add_argument("--public-bank", type=Path, required=True,
                        help="the public bank file, for each item's type")
    parser.add_argument("--out", type=Path, required=True,
                        help="where to write the results JSON")
    parser.add_argument("--retest-answers", type=Path, default=None,
                        help="answers.jsonl holding the retest round, if it lives in its own file")
    args = parser.parse_args(argv)

    answers = read_jsonl(args.answers)
    if not answers:
        parser.error(f"no answers in {args.answers}")

    types = item_types(args.public_bank)
    design = read_bank_design(args.truth)
    pids = sorted({str(row["pid"]) for row in answers})
    thetas = load_thetas(args.truth, pids)
    retest_answers = read_jsonl(args.retest_answers) if args.retest_answers else None

    report = run_qa(answers, types, design, thetas, retest_answers)
    report["inputs"] = {
        "answers": str(args.answers),
        "retest_answers": str(args.retest_answers) if args.retest_answers else None,
        "public_bank": str(args.public_bank),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print_table(report)
    print("")
    print(f"results -> {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
