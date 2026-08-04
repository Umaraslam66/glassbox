"""Stage 3 item encoder: question text -> 8-dimensional loading vector + discrimination.

What it does, in one paragraph
------------------------------
The system-side model (Qwen3.6-27B on Leonardo) is shown one survey item and one
of the eight published trait poles, and asked a single question: if a person
endorses this statement, where does that put them on that dimension? It answers
with one small integer. That is repeated ``k`` times per (item, pole) at a
non-zero temperature with independent seeds, so every item comes back as eight
averaged scores plus eight spreads. A tiny ridge regression, trained on the 202
training items against their **fitted** loadings, turns those scores into a
loading vector. A second ridge predicts the loading vector's length
(discrimination) from the same scores plus a polarization probe, four wording
features and -- when the embedding pass succeeded -- a reduced item embedding.
The 50 held-out items are then predicted zero-shot and scored.

What it is allowed to see
-------------------------
Public item text (``data/public/bank_items.json``), the published pole
descriptions (``src/model/pole_descriptions.py``, byte-guarded against
PREREGISTRATION.md section 2), and the **fitted** item parameters of the 202
training items from ``results/stage2_v2_fit.npz``. Fitted parameters are
system-side quantities: they are what the MIRT fit produced from answers, not
what was planted. It never sees designed loadings, the item bank's truth file,
persona cards, or any trait vector. The Procrustes rotation in
``results/stage2_v2_recovery.json`` is truth-derived and is never read here;
cosine similarity is invariant under a shared orthogonal rotation, so all of
this work stays in the unrotated fitted basis and the graded number is the same.

Why the scores are sampled rather than read off a distribution
--------------------------------------------------------------
The Wall forbids ``src/model/`` from touching token distributions at all
(``tests/test_wall.py`` rule (c)). So confidence is measured the honest way:
``k`` independent samples per (item, pole), and the sample spread is the
confidence feature. This costs k times the generations, which at this scale is
nothing, and it needs no rule change.

Also emitted here: the open-answer pole judgments
-------------------------------------------------
The person encoder's fusion head needs the same kind of judgment over the three
open interview answers ("what does this answer reveal about the speaker on this
dimension?"). Those prompts live in this module because they share the pole
scale, the parser and the same GPU job -- one engine init serves both. This
module does not build the person encoder; it only collects the feature.

Usage
-----
    # 1. emit the judgment prompts for the cluster driver
    python -m src.model.item_encoder emit --out data/runs/stage3_qwen/judge_in.jsonl

    # 2. (Leonardo runs jobs/render_chat_batch.py over that file)

    # 3. train the heads and score the held-out items
    python -m src.model.item_encoder train \\
        --judgments data/runs/stage3_qwen/judge_out.jsonl \\
        --embeddings data/runs/stage3_qwen/item_embeddings.npz
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from src.model.pole_descriptions import DIMENSION_CODES, describe

REPO_ROOT = Path(__file__).resolve().parents[2]

BANK_PATH = REPO_ROOT / "data" / "public" / "bank_items.json"
FIT_PATH = REPO_ROOT / "results" / "stage2_v2_fit.npz"
SPLITS_PATH = REPO_ROOT / "experiments" / "splits_v1.json"
TRANSCRIPTS_PATH = REPO_ROOT / "data" / "runs" / "interview_v1" / "transcripts.jsonl"

#: Sampled-score scale for a pole judgment. Small on purpose: the head only
#: needs "which dimension, which sign, roughly how strongly".
SCORE_MIN, SCORE_MAX = -3, 3

#: Sampled-score scale for the polarization probe (how much a population would
#: split on this item). One-sided, so a wider range costs nothing.
POLAR_MIN, POLAR_MAX = 0, 6

#: Repeats per (item, pole). Five independent draws give a usable spread without
#: making the job big.
DEFAULT_REPEATS = 5

#: Sampling temperature for the judgments. Non-zero by necessity -- at
#: temperature 0 every repeat is the same draw and the spread feature is dead.
DEFAULT_TEMPERATURE = 0.7


# ---------------------------------------------------------------------------
# prompt construction
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = (
    "You are a careful survey methodologist. You read one survey statement and "
    "one description of a personality dimension, and you say how strongly "
    "answering the statement one way indicates one end of that dimension. "
    "You reply with a single number and nothing else -- no words, no "
    "explanation, no punctuation."
)

OPEN_SYSTEM = (
    "You are a careful interview analyst. You read one thing a person said in "
    "an interview and one description of a personality dimension, and you say "
    "where what they said places them on that dimension. You reply with a "
    "single number and nothing else -- no words, no explanation, no "
    "punctuation."
)

POLAR_SYSTEM = (
    "You are a careful survey methodologist. You read one survey statement and "
    "you say how much a broad cross-section of ordinary adults would disagree "
    "with each other when answering it. You reply with a single number and "
    "nothing else -- no words, no explanation, no punctuation."
)

ITEM_SCALE = """\
-3 = that answer is a clear sign of the LOW end
-2 = that answer is a moderate sign of the LOW end
-1 = that answer is a weak sign of the LOW end
0 = this statement tells you nothing about this dimension
1 = that answer is a weak sign of the HIGH end
2 = that answer is a moderate sign of the HIGH end
3 = that answer is a clear sign of the HIGH end"""

OPEN_SCALE = """\
-3 = what they said clearly places them at the LOW end
-2 = what they said moderately suggests the LOW end
-1 = what they said weakly suggests the LOW end
0 = what they said reveals nothing about this dimension
1 = what they said weakly suggests the HIGH end
2 = what they said moderately suggests the HIGH end
3 = what they said clearly places them at the HIGH end"""

POLAR_SCALE = """\
0 = almost everyone would give the same answer
1 = a large majority would answer the same way
2 = most would agree, a noticeable minority would not
3 = opinion would be somewhat divided
4 = opinion would be clearly divided
5 = people would be strongly split
6 = people would be split about as strongly as a question can split them"""


def endorsement_phrase(item: dict) -> str:
    """How a respondent 'endorses' this item, in the item's own answer format."""
    if item.get("type") == "binary":
        return "answers YES to this statement"
    return "answers STRONGLY AGREE to this statement"


def answer_format(item: dict) -> str:
    """One line describing the item's answer options, from the public bank."""
    labels = [str(option["label"]) for option in item.get("options", [])]
    if item.get("type") == "binary":
        return "Answer format: yes / no"
    return "Answer format: a 1-5 agreement scale (" + ", ".join(labels) + ")"


def build_item_pole_prompt(item: dict, code: str) -> tuple[str, str]:
    """The (system, user) turns asking how item ``item`` relates to pole ``code``."""
    user = (
        f"{describe(code)}\n"
        "\n"
        "SURVEY STATEMENT:\n"
        f'"{item["text"]}"\n'
        f"{answer_format(item)}\n"
        "\n"
        f"QUESTION: If a person {endorsement_phrase(item)}, what does that tell "
        "you about where they sit on the trait dimension above?\n"
        "\n"
        f"Reply with exactly one number from this scale:\n{ITEM_SCALE}\n"
        "\n"
        "Your answer (one number only):"
    )
    return JUDGE_SYSTEM, user


def build_polarization_prompt(item: dict) -> tuple[str, str]:
    """The (system, user) turns asking how much a population would split on an item.

    This is the one feature the *design* of the bank cannot supply. A perfect
    design-class oracle only reaches r about 0.57 against fitted discrimination,
    because fitted discrimination is mostly "how hard did the population split on
    this exact wording", not "what was this item labelled". So the model is asked
    that question directly.
    """
    user = (
        "SURVEY STATEMENT:\n"
        f'"{item["text"]}"\n'
        f"{answer_format(item)}\n"
        "\n"
        "QUESTION: If you asked a broad cross-section of ordinary adults this "
        "statement, how much would their answers disagree with each other?\n"
        "\n"
        f"Reply with exactly one number from this scale:\n{POLAR_SCALE}\n"
        "\n"
        "Your answer (one number only):"
    )
    return POLAR_SYSTEM, user


def build_open_answer_prompt(question: str, answer: str, code: str) -> tuple[str, str]:
    """The (system, user) turns asking what an open answer reveals about the speaker.

    The answer text is quoted inside the prompt. It is system-side material: it
    is what the persona actually said out loud in the interview, which is exactly
    what the Wall's rule 2 lets the encoders see.
    """
    user = (
        f"{describe(code)}\n"
        "\n"
        "INTERVIEW QUESTION:\n"
        f"{question}\n"
        "\n"
        "WHAT THE PERSON SAID:\n"
        f'"{answer}"\n'
        "\n"
        "QUESTION: Reading only what they said, where does it place the speaker "
        "on the trait dimension above?\n"
        "\n"
        f"Reply with exactly one number from this scale:\n{OPEN_SCALE}\n"
        "\n"
        "Your answer (one number only):"
    )
    return OPEN_SYSTEM, user


# ---------------------------------------------------------------------------
# batch emission
# ---------------------------------------------------------------------------


def load_bank() -> tuple[list[dict], dict[str, str]]:
    """The 252 closed items and the ``item_id -> text`` map for open prompts."""
    bank = json.loads(BANK_PATH.read_text(encoding="utf-8"))
    open_text = {entry["item_id"]: entry["text"] for entry in bank["open_ended"]}
    return bank["items"], open_text


def iter_item_records(
    items: Sequence[dict], repeats: int = DEFAULT_REPEATS
) -> Iterable[dict]:
    """One record per (item, pole, repeat), plus the polarization repeats.

    ``round`` carries the (kind, pole, repeat) triple because the cluster driver
    derives each record's sampling seed from (pid, item_id, round) -- two records
    that differ only in the repeat index must not share random numbers.
    """
    for item in items:
        for code in DIMENSION_CODES:
            system, user = build_item_pole_prompt(item, code)
            for rep in range(repeats):
                yield {
                    "kind": "item",
                    "item_id": item["item_id"],
                    "pole": code,
                    "rep": rep,
                    "round": f"item|{code}|r{rep}",
                    "system": system,
                    "user": user,
                }
    for item in items:
        system, user = build_polarization_prompt(item)
        for rep in range(repeats):
            yield {
                "kind": "polar",
                "item_id": item["item_id"],
                "pole": "",
                "rep": rep,
                "round": f"polar|r{rep}",
                "system": system,
                "user": user,
            }


def iter_open_records(
    transcripts_path: Path = TRANSCRIPTS_PATH, repeats: int = 1
) -> Iterable[dict]:
    """One record per (persona, open prompt, pole, repeat) over the transcripts.

    Reads only the ``open`` block of each transcript: the prompt text and the
    persona's own free-text answer. Nothing else in the record is touched.
    """
    with transcripts_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            for entry in record.get("open") or []:
                for code in DIMENSION_CODES:
                    system, user = build_open_answer_prompt(
                        entry["question"], entry["answer"], code
                    )
                    for rep in range(repeats):
                        yield {
                            "kind": "open",
                            "pid": record["pid"],
                            "item_id": entry["prompt_id"],
                            "pole": code,
                            "rep": rep,
                            "round": f"open|{code}|r{rep}",
                            "system": system,
                            "user": user,
                        }


def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


# ---------------------------------------------------------------------------
# parsing what came back
# ---------------------------------------------------------------------------

_NUMBER = re.compile(r"[+-]?\d+")


def parse_score(text: str, low: int = SCORE_MIN, high: int = SCORE_MAX) -> int | None:
    """The integer a completion meant, or ``None`` if it did not produce one.

    A clean reply is the whole completion being one number. Anything else falls
    back to the *last* in-range number in the text, because a model that ignores
    "reply with one number" almost always puts its answer at the end. Out-of-range
    values are dropped rather than clipped -- a judgment of "7" on a -3..3 scale is
    a parse failure, not a strong opinion.
    """
    stripped = (text or "").strip()
    if not stripped:
        return None
    candidates = [int(match) for match in _NUMBER.findall(stripped)]
    in_range = [value for value in candidates if low <= value <= high]
    if not in_range:
        return None
    if re.fullmatch(r"[+-]?\d+", stripped) and low <= int(stripped) <= high:
        return int(stripped)
    return in_range[-1]


def is_clean_reply(text: str, low: int = SCORE_MIN, high: int = SCORE_MAX) -> bool:
    """True when the completion is nothing but an in-range number."""
    stripped = (text or "").strip()
    return bool(
        re.fullmatch(r"[+-]?\d+", stripped) and low <= int(stripped) <= high
    )


def load_judgments(path: Path) -> dict[str, Any]:
    """Group a driver output file by kind.

    Returns ``{"item": {item_id: {pole: [scores]}}, "polar": {item_id: [scores]},
    "open": {(pid, prompt_id): {pole: [scores]}}}`` plus parse statistics.
    """
    items: dict[str, dict[str, list[int]]] = {}
    polar: dict[str, list[int]] = {}
    openers: dict[tuple[str, str], dict[str, list[int]]] = {}
    stats = {"records": 0, "clean": 0, "parsed": 0, "unparsed": 0}

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            kind = record.get("kind")
            completion = record.get("completion", "")
            low, high = (POLAR_MIN, POLAR_MAX) if kind == "polar" else (SCORE_MIN, SCORE_MAX)
            value = parse_score(completion, low, high)
            stats["records"] += 1
            if is_clean_reply(completion, low, high):
                stats["clean"] += 1
            if value is None:
                stats["unparsed"] += 1
                continue
            stats["parsed"] += 1

            if kind == "item":
                items.setdefault(record["item_id"], {}).setdefault(
                    record["pole"], []
                ).append(value)
            elif kind == "polar":
                polar.setdefault(record["item_id"], []).append(value)
            elif kind == "open":
                key = (record["pid"], record["item_id"])
                openers.setdefault(key, {}).setdefault(record["pole"], []).append(value)

    return {"item": items, "polar": polar, "open": openers, "stats": stats}


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------

#: Words that soften a claim; an item full of them splits a population less.
HEDGE_WORDS = frozenset(
    """sometimes usually often generally tend tends maybe might somewhat rather
    fairly occasionally mostly typically quite prefer""".split()
)

#: Words that make a claim absolute; an item full of them splits a population more.
ABSOLUTE_WORDS = frozenset(
    """always never every all none completely entirely exactly only nothing
    everyone anyone""".split()
)

WORDING_NAMES = ("is_binary", "n_words", "has_hedge", "has_absolute")


def wording_features(item: dict) -> np.ndarray:
    """The four surface features: answer format, length, hedging, absoluteness."""
    words = re.findall(r"[a-z']+", item["text"].lower())
    return np.array(
        [
            1.0 if item.get("type") == "binary" else 0.0,
            float(len(words)),
            1.0 if any(word in HEDGE_WORDS for word in words) else 0.0,
            1.0 if any(word in ABSOLUTE_WORDS for word in words) else 0.0,
        ],
        dtype=float,
    )


def judgment_features(
    parsed: dict[str, Any], item_ids: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    """``(means, spreads)`` -- one row per item, eight columns each.

    A pole with no parseable sample at all gets mean 0 (the scale's "tells you
    nothing" point) and the largest spread seen, so a missing judgment reads as
    "no information, low confidence" rather than as a strong claim.
    """
    means = np.zeros((len(item_ids), len(DIMENSION_CODES)))
    spreads = np.zeros((len(item_ids), len(DIMENSION_CODES)))
    for row, item_id in enumerate(item_ids):
        by_pole = parsed["item"].get(item_id, {})
        for col, code in enumerate(DIMENSION_CODES):
            samples = by_pole.get(code, [])
            if samples:
                means[row, col] = float(np.mean(samples))
                spreads[row, col] = float(np.std(samples))
            else:
                means[row, col] = 0.0
                spreads[row, col] = float(SCORE_MAX - SCORE_MIN) / 2.0
    return means, spreads


def polarization_features(
    parsed: dict[str, Any], item_ids: Sequence[str]
) -> np.ndarray:
    """``(mean, spread)`` of the polarization probe, one row per item."""
    out = np.zeros((len(item_ids), 2))
    default = (POLAR_MIN + POLAR_MAX) / 2.0
    for row, item_id in enumerate(item_ids):
        samples = parsed["polar"].get(item_id, [])
        out[row, 0] = float(np.mean(samples)) if samples else default
        out[row, 1] = float(np.std(samples)) if samples else 0.0
    return out


def topic_features(items_by_id: dict[str, dict], item_ids: Sequence[str],
                   domains: Sequence[str]) -> np.ndarray:
    """One-hot topic domain -- the non-LLM shortcut baseline."""
    index = {domain: i for i, domain in enumerate(domains)}
    out = np.zeros((len(item_ids), len(domains)))
    for row, item_id in enumerate(item_ids):
        domain = items_by_id[item_id].get("topic_domain")
        if domain in index:
            out[row, index[domain]] = 1.0
    return out


# ---------------------------------------------------------------------------
# ridge with exact leave-one-out
# ---------------------------------------------------------------------------


def _standardizer(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return mean, scale


def _design(X: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Standardized features with an unpenalized intercept column appended."""
    Z = (X - mean) / scale
    return np.hstack([Z, np.ones((Z.shape[0], 1))])


def ridge_fit(X: np.ndarray, Y: np.ndarray, lam: float) -> dict[str, Any]:
    """Ridge, features standardized, intercept unpenalized."""
    mean, scale = _standardizer(X)
    D = _design(X, mean, scale)
    penalty = np.eye(D.shape[1]) * lam
    penalty[-1, -1] = 0.0
    W = np.linalg.solve(D.T @ D + penalty, D.T @ Y)
    return {"mean": mean, "scale": scale, "W": W, "lam": lam}


def ridge_predict(model: dict[str, Any], X: np.ndarray) -> np.ndarray:
    return _design(X, model["mean"], model["scale"]) @ model["W"]


def ridge_loo(X: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    """Exact leave-one-out predictions for the ridge estimator at this lambda.

    Ridge is a linear smoother, so the hat matrix gives every LOO prediction in
    one solve: ``y_loo = y - resid / (1 - h_ii)``. The standardization is treated
    as fixed across folds, which is the usual convention and cannot flatter the
    result here -- it uses no target information at all.
    """
    mean, scale = _standardizer(X)
    D = _design(X, mean, scale)
    penalty = np.eye(D.shape[1]) * lam
    penalty[-1, -1] = 0.0
    G = np.linalg.solve(D.T @ D + penalty, D.T)
    H = D @ G
    fitted = H @ Y
    leverage = np.clip(np.diag(H), 0.0, 1.0 - 1e-9)
    resid = Y - fitted
    return Y - resid / (1.0 - leverage)[:, None]


#: Wide enough that the chosen lambda is never the grid's own edge. The
#: direction head really does want almost no shrinkage -- 16 well-conditioned
#: features against 202 items -- so the grid has to run far enough down that
#: "no regularization" is a choice the search made rather than a wall it hit.
LAMBDA_GRID = tuple(float(10.0 ** e) for e in np.arange(-6.0, 4.01, 0.25))


def unit_rows(M: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return M / norms


def row_cosines(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.sum(unit_rows(A) * unit_rows(B), axis=1)


def select_lambda_direction(X: np.ndarray, Y: np.ndarray) -> tuple[float, list[dict]]:
    """Pick lambda by LOO median cosine on the training items."""
    curve = []
    for lam in LAMBDA_GRID:
        pred = ridge_loo(X, Y, lam)
        cos = row_cosines(pred, Y)
        curve.append(
            {"lam": lam, "median_cosine": float(np.median(cos)),
             "mean_cosine": float(np.mean(cos))}
        )
    best = max(curve, key=lambda row: row["median_cosine"])
    return best["lam"], curve


def select_lambda_scalar(X: np.ndarray, y: np.ndarray) -> tuple[float, list[dict]]:
    """Pick lambda by LOO mean squared error on the training items."""
    Y = y[:, None]
    curve = []
    for lam in LAMBDA_GRID:
        pred = ridge_loo(X, Y, lam)[:, 0]
        curve.append(
            {"lam": lam, "loo_mse": float(np.mean((pred - y) ** 2)),
             "loo_r": float(pearson(pred, y))}
        )
    best = min(curve, key=lambda row: row["loo_mse"])
    return best["lam"], curve


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------------------
# embeddings (optional)
# ---------------------------------------------------------------------------


def reduce_embeddings(
    train: np.ndarray, holdout: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """SVD to ``k`` components, fitted on training rows only, applied frozen.

    202 items against 5120 raw dimensions is a hopeless ratio -- a TF-IDF probe
    at that ratio scored a LOO median cosine of 0.155. So the projection is
    always fitted before any head sees the features, and the held-out rows are
    only ever transformed, never fitted.
    """
    mean = train.mean(axis=0)
    centered = train - mean
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    basis = Vt[:k].T
    return centered @ basis, (holdout - mean) @ basis


# ---------------------------------------------------------------------------
# the encoder itself
# ---------------------------------------------------------------------------


def train_direction_head(X: np.ndarray, Y: np.ndarray) -> dict[str, Any]:
    lam, curve = select_lambda_direction(X, Y)
    model = ridge_fit(X, Y, lam)
    model["lambda_curve"] = curve
    return model


def train_magnitude_head(X: np.ndarray, norms: np.ndarray) -> dict[str, Any]:
    target = np.log(np.maximum(norms, 1e-6))
    lam, curve = select_lambda_scalar(X, target)
    model = ridge_fit(X, target[:, None], lam)
    model["lambda_curve"] = curve
    return model


def predict_loadings(
    direction: dict[str, Any], magnitude: dict[str, Any] | None,
    X_dir: np.ndarray, X_mag: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """``(loading_vectors, predicted_norms)``.

    Direction and magnitude are predicted separately and multiplied. Letting one
    head do both makes the long items dominate the fit and the direction gets
    worse for no gain in magnitude.
    """
    direction_pred = unit_rows(ridge_predict(direction, X_dir))
    if magnitude is None or X_mag is None:
        norms = np.ones(direction_pred.shape[0])
    else:
        norms = np.exp(ridge_predict(magnitude, X_mag)[:, 0])
    return direction_pred * norms[:, None], norms


# ---------------------------------------------------------------------------
# CLI: emit
# ---------------------------------------------------------------------------


def cmd_emit(args: argparse.Namespace) -> None:
    items, _ = load_bank()
    out = Path(args.out)
    records: list[dict] = list(iter_item_records(items, args.repeats))
    n_item = sum(1 for r in records if r["kind"] == "item")
    n_polar = sum(1 for r in records if r["kind"] == "polar")
    n_open = 0
    if not args.no_open:
        open_records = list(iter_open_records(Path(args.transcripts), args.open_repeats))
        records.extend(open_records)
        n_open = len(open_records)
    written = write_jsonl(out, records)

    longest = max(len(r["system"]) + len(r["user"]) for r in records)
    manifest = {
        "out_file": str(out),
        "records": written,
        "item_pole_records": n_item,
        "polarization_records": n_polar,
        "open_answer_records": n_open,
        "items": len(items),
        "poles": len(DIMENSION_CODES),
        "repeats": args.repeats,
        "open_repeats": args.open_repeats,
        "score_scale": [SCORE_MIN, SCORE_MAX],
        "polarization_scale": [POLAR_MIN, POLAR_MAX],
        "longest_prompt_chars": longest,
    }
    manifest_path = out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


# ---------------------------------------------------------------------------
# CLI: train
# ---------------------------------------------------------------------------


def _arm_features(
    blocks: dict[str, np.ndarray], names: Sequence[str]
) -> np.ndarray | None:
    chosen = [blocks[name] for name in names if blocks.get(name) is not None]
    if len(chosen) != len(names):
        return None
    return np.hstack(chosen)


def cmd_train(args: argparse.Namespace) -> None:
    fit = np.load(FIT_PATH, allow_pickle=True)
    splits = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
    items, _ = load_bank()
    items_by_id = {item["item_id"]: item for item in items}
    domains = json.loads(BANK_PATH.read_text(encoding="utf-8"))["topic_domains"]

    train_ids = [str(x) for x in fit["item_train_ids"]]
    holdout_ids = [str(x) for x in fit["item_holdout_ids"]]
    Y_train = np.asarray(fit["train_loadings"], dtype=float)
    Y_holdout = np.asarray(fit["holdout_loadings"], dtype=float)
    a_train = np.asarray(fit["train_discrimination"], dtype=float)
    a_holdout = np.asarray(fit["holdout_discrimination"], dtype=float)

    parsed = load_judgments(Path(args.judgments))

    def blocks_for(ids: Sequence[str]) -> dict[str, np.ndarray | None]:
        means, spreads = judgment_features(parsed, ids)
        return {
            "judge_mean": means,
            "judge_spread": spreads,
            "polar": polarization_features(parsed, ids),
            "wording": np.vstack([wording_features(items_by_id[i]) for i in ids]),
            "topic": topic_features(items_by_id, ids, domains),
            "embed": None,
        }

    tr = blocks_for(train_ids)
    ho = blocks_for(holdout_ids)

    embedding_source = "none"
    embed_components = 0
    embed_component_curve: list[dict] = []
    if args.embeddings:
        emb = np.load(args.embeddings, allow_pickle=True)
        emb_ids = [str(x) for x in emb["item_ids"]]
        lookup = {item_id: row for row, item_id in enumerate(emb_ids)}
        raw_tr = np.vstack([emb["embeddings"][lookup[i]] for i in train_ids])
        raw_ho = np.vstack([emb["embeddings"][lookup[i]] for i in holdout_ids])
        embedding_source = str(emb["source"]) if "source" in emb.files else "provided"

        # How many components, chosen on the 202 TRAINING items only. 202 rows
        # against 5120 raw dimensions is the ratio that made a TF-IDF probe
        # useless, so the reduction is never left to a guess and the held-out
        # rows are only ever transformed by a projection fitted without them.
        if args.embed_components > 0:
            embed_components = args.embed_components
        else:
            log_a = np.log(np.maximum(a_train, 1e-6))
            for k in (4, 8, 16, 32, 64, 128):
                # never let the component count near the row count: the
                # leave-one-out leverages go to 1 and the search stops meaning
                # anything.
                if k > (2 * raw_tr.shape[0]) // 3:
                    continue
                Z, _ = reduce_embeddings(raw_tr, raw_ho, k)
                lam, curve = select_lambda_scalar(Z, log_a)
                best = min(curve, key=lambda row: row["loo_mse"])
                embed_component_curve.append(
                    {"k": k, "lambda": lam, "loo_mse": best["loo_mse"],
                     "loo_r": best["loo_r"]}
                )
            embed_components = int(
                min(embed_component_curve, key=lambda row: row["loo_mse"])["k"]
            )
        tr["embed"], ho["embed"] = reduce_embeddings(raw_tr, raw_ho, embed_components)

    strata = splits.get("strata", {})

    # -------------------------------------------------- arms
    direction_arms = {
        "primary": ("judge_mean", "judge_spread"),
        "judgments_plus_wording": ("judge_mean", "judge_spread", "wording"),
        "topic_only": ("topic",),
        "embeddings_only": ("embed",),
    }
    magnitude_arms = {
        "primary": ("judge_mean", "judge_spread", "polar", "wording", "embed"),
        "no_embeddings": ("judge_mean", "judge_spread", "polar", "wording"),
        "judgments_only": ("judge_mean", "judge_spread"),
        "polarization_only": ("polar",),
        "wording_only": ("wording",),
        "embeddings_only": ("embed",),
        "topic_only": ("topic",),
    }
    if tr["embed"] is None:
        magnitude_arms["primary"] = magnitude_arms["no_embeddings"]

    results: dict[str, Any] = {}

    # -------------------------------------------------- direction
    direction_out = {}
    trained_direction = {}
    for name, names in direction_arms.items():
        X_tr = _arm_features(tr, names)
        X_ho = _arm_features(ho, names)
        if X_tr is None or X_ho is None:
            direction_out[name] = {"skipped": "features unavailable"}
            continue
        head = train_direction_head(X_tr, Y_train)
        trained_direction[name] = (head, X_tr, X_ho)
        pred = unit_rows(ridge_predict(head, X_ho))
        cos = row_cosines(pred, Y_holdout)
        loo = row_cosines(ridge_loo(X_tr, Y_train, head["lam"]), Y_train)
        direction_out[name] = {
            "features": list(names),
            "n_features": int(X_tr.shape[1]),
            "lambda": head["lam"],
            "train_loo_median_cosine": float(np.median(loo)),
            "holdout_median_cosine": float(np.median(cos)),
            "holdout_mean_cosine": float(np.mean(cos)),
            "holdout_frac_ge_0.7": float(np.mean(cos >= 0.7)),
            "holdout_frac_positive": float(np.mean(cos > 0)),
        }

    # constant-direction control: no text at all
    const = unit_rows(np.tile(Y_train.mean(axis=0), (len(holdout_ids), 1)))
    const_cos = row_cosines(const, Y_holdout)
    direction_out["constant_no_text"] = {
        "features": [],
        "holdout_median_cosine": float(np.median(const_cos)),
        "holdout_frac_ge_0.7": float(np.mean(const_cos >= 0.7)),
    }

    primary_head, X_dir_tr, X_dir_ho = trained_direction["primary"]
    primary_pred_dir = unit_rows(ridge_predict(primary_head, X_dir_ho))
    primary_cos = row_cosines(primary_pred_dir, Y_holdout)

    # -------------------------------------------------- magnitude
    magnitude_out = {}
    trained_magnitude = {}
    for name, names in magnitude_arms.items():
        X_tr = _arm_features(tr, names)
        X_ho = _arm_features(ho, names)
        if X_tr is None or X_ho is None:
            magnitude_out[name] = {"skipped": "features unavailable"}
            continue
        head = train_magnitude_head(X_tr, a_train)
        trained_magnitude[name] = (head, X_tr, X_ho)
        pred = np.exp(ridge_predict(head, X_ho)[:, 0])
        loo_log = ridge_loo(X_tr, np.log(np.maximum(a_train, 1e-6))[:, None], head["lam"])[:, 0]
        magnitude_out[name] = {
            "features": list(names),
            "n_features": int(X_tr.shape[1]),
            "lambda": head["lam"],
            "train_loo_r_log": float(pearson(loo_log, np.log(np.maximum(a_train, 1e-6)))),
            "train_loo_r_raw": float(pearson(np.exp(loo_log), a_train)),
            "holdout_r": float(pearson(pred, a_holdout)),
            "holdout_r_log": float(pearson(np.log(pred), np.log(np.maximum(a_holdout, 1e-6)))),
            "holdout_spearman": float(spearman(pred, a_holdout)),
        }

    primary_mag_head, X_mag_tr, X_mag_ho = trained_magnitude["primary"]
    primary_pred_norm = np.exp(ridge_predict(primary_mag_head, X_mag_ho)[:, 0])

    # -------------------------------------------------- the two frozen bars
    predicted_loadings = primary_pred_dir * primary_pred_norm[:, None]
    bar_cosine = float(np.median(primary_cos))
    bar_disc_r = float(pearson(primary_pred_norm, a_holdout))

    # -------------------------------------------------- per stratum + per item
    per_stratum = {}
    for stratum in sorted(set(strata.get(i, "unknown") for i in holdout_ids)):
        mask = np.array([strata.get(i, "unknown") == stratum for i in holdout_ids])
        per_stratum[stratum] = {
            "n": int(mask.sum()),
            "median_cosine": float(np.median(primary_cos[mask])),
            "mean_cosine": float(np.mean(primary_cos[mask])),
            "frac_ge_0.7": float(np.mean(primary_cos[mask] >= 0.7)),
            "discrimination_r": float(pearson(primary_pred_norm[mask], a_holdout[mask])),
        }
    per_kind = {}
    kinds = [str(k) for k in fit["holdout_item_kinds"]]
    for kind in sorted(set(kinds)):
        mask = np.array([k == kind for k in kinds])
        per_kind[kind] = {
            "n": int(mask.sum()),
            "median_cosine": float(np.median(primary_cos[mask])),
            "discrimination_r": float(pearson(primary_pred_norm[mask], a_holdout[mask])),
        }

    order = np.argsort(primary_cos)
    per_item = [
        {
            "item_id": holdout_ids[i],
            "stratum": strata.get(holdout_ids[i], "unknown"),
            "kind": kinds[i],
            "cosine": float(primary_cos[i]),
            "fitted_norm": float(a_holdout[i]),
            "predicted_norm": float(primary_pred_norm[i]),
            "top_judged_pole": DIMENSION_CODES[
                int(np.argmax(np.abs(ho["judge_mean"][i])))
            ],
            "top_predicted_component": int(np.argmax(np.abs(primary_pred_dir[i]))),
            "top_fitted_component": int(np.argmax(np.abs(Y_holdout[i]))),
        }
        for i in order
    ]

    # -------------------------------------------------- diagnostics
    weak_mask = a_holdout <= np.quantile(a_holdout, 0.2)
    diagnostics = {
        "parse": parsed["stats"]
        | {
            "clean_rate": round(parsed["stats"]["clean"] / max(parsed["stats"]["records"], 1), 4),
            "parse_rate": round(parsed["stats"]["parsed"] / max(parsed["stats"]["records"], 1), 4),
        },
        "judgment_spread_mean": float(tr["judge_spread"].mean()),
        "judgment_spread_holdout_mean": float(ho["judge_spread"].mean()),
        "judged_pole_index_equals_predicted_component_index": {
            "value": float(
                np.mean(
                    np.argmax(np.abs(ho["judge_mean"]), axis=1)
                    == np.argmax(np.abs(primary_pred_dir), axis=1)
                )
            ),
            "note": (
                "expected to sit near the 1/8 chance level, and does. The fitted "
                "basis is an arbitrary rotation of the pole basis, so a head that "
                "had learned the rotation properly should NOT line the indices up. "
                "A high number here would mean the head was passing judgments "
                "through rather than rotating them."
            ),
        },
        "fitted_norm_holdout": {
            "median": float(np.median(a_holdout)),
            "min": float(a_holdout.min()),
            "max": float(a_holdout.max()),
        },
        "predicted_norm_holdout": {
            "median": float(np.median(primary_pred_norm)),
            "min": float(primary_pred_norm.min()),
            "max": float(primary_pred_norm.max()),
        },
        "weakest_fitted_quintile": {
            "n": int(weak_mask.sum()),
            "mean_fitted_norm": float(a_holdout[weak_mask].mean()),
            "mean_predicted_norm": float(primary_pred_norm[weak_mask].mean()),
            "median_cosine": float(np.median(primary_cos[weak_mask])),
            "note": (
                "system-side proxy for the standing distractor watch: designed "
                "classes are truth-side, so the weakest FITTED items stand in. "
                "Predicted norms far above fitted here would be the trigger."
            ),
        },
        "cosine_vs_fitted_norm_r": float(pearson(primary_cos, a_holdout)),
        "exploratory_excluding_weakest_quintile": {
            "n": int((~weak_mask).sum()),
            "median_cosine": float(np.median(primary_cos[~weak_mask])),
            "discrimination_r": float(
                pearson(primary_pred_norm[~weak_mask], a_holdout[~weak_mask])
            ),
            "note": (
                "EXPLORATORY, not the bar. A near-zero fitted loading vector has "
                "no stable direction, so its cosine is mostly noise -- this shows "
                "how much of the spread in the confirmatory number comes from "
                "items the fit itself barely resolved. The bar stays the median "
                "over all 50 held-out items."
            ),
        },
        "embedding_source": embedding_source,
        "embed_components": embed_components,
        "embed_component_selection": embed_component_curve,
    }

    results = {
        "schema": "glassbox.stage3.item_encoder/1",
        "grading_reference": (
            "FITTED parameters of the 50 held-out items in results/stage2_v2_fit.npz, "
            "unrotated basis (cosine is invariant under a shared orthogonal rotation, "
            "so the truth-derived Procrustes rotation is never needed system-side)"
        ),
        "model": "Qwen3.6-27B (Leonardo, frozen system-side model)",
        "n_train_items": len(train_ids),
        "n_holdout_items": len(holdout_ids),
        "repeats_per_item_pole": args.repeats_recorded,
        "bars": {
            "median_loading_cosine": {
                "value": bar_cosine,
                "bar": 0.7,
                "pass": bool(bar_cosine >= 0.7),
            },
            "discrimination_r": {
                "value": bar_disc_r,
                "bar": 0.6,
                "pass": bool(bar_disc_r >= 0.6),
            },
        },
        "direction_arms": direction_out,
        "magnitude_arms": magnitude_out,
        "per_stratum": per_stratum,
        "per_item_kind": per_kind,
        "per_item": per_item,
        "diagnostics": diagnostics,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    if args.predictions:
        np.savez(
            args.predictions,
            item_ids=np.array(holdout_ids),
            predicted_loadings=predicted_loadings,
            predicted_norms=primary_pred_norm,
            cosines=primary_cos,
        )
    if args.plot:
        make_plot(
            Path(args.plot),
            primary_cos,
            a_holdout,
            primary_pred_norm,
            primary_head["lambda_curve"],
            per_stratum,
        )
        print(f"[item-encoder] wrote {args.plot}")
    print(json.dumps(results["bars"], indent=2))
    print(f"[item-encoder] wrote {out_path}")


def cmd_open_features(args: argparse.Namespace) -> None:
    """Turn the open-answer judgments into the array the fusion head consumes.

    One row per persona, one column per (open prompt, pole). Unparsed cells are
    NaN in ``scores`` and 0 -- the scale's "reveals nothing" point -- in the
    dense ``features`` matrix, with a companion ``observed`` mask so the
    consumer can tell a genuine zero from a missing one.

    This module does not build the person encoder. It only hands over the
    feature, because the judgments came out of the same GPU job and share this
    module's parser.
    """
    parsed = load_judgments(Path(args.judgments))
    splits = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
    prompt_ids = list(splits["interview_open"])

    pids = sorted({pid for pid, _ in parsed["open"]})
    scores = np.full((len(pids), len(prompt_ids), len(DIMENSION_CODES)), np.nan)
    for row, pid in enumerate(pids):
        for col, prompt_id in enumerate(prompt_ids):
            by_pole = parsed["open"].get((pid, prompt_id), {})
            for depth, code in enumerate(DIMENSION_CODES):
                samples = by_pole.get(code, [])
                if samples:
                    scores[row, col, depth] = float(np.mean(samples))

    observed = np.isfinite(scores)
    dense = np.where(observed, scores, 0.0).reshape(len(pids), -1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        pids=np.array(pids),
        prompt_ids=np.array(prompt_ids),
        poles=np.array(DIMENSION_CODES),
        scores=scores,
        observed=observed,
        features=dense,
    )
    manifest = {
        "out_file": str(out),
        "personas": len(pids),
        "open_prompts": prompt_ids,
        "poles": list(DIMENSION_CODES),
        "feature_columns": dense.shape[1],
        "cells_expected": int(scores.size),
        "cells_observed": int(observed.sum()),
        "coverage": round(float(observed.mean()), 4),
        "score_scale": [SCORE_MIN, SCORE_MAX],
        "layout": "scores[persona, open_prompt, pole]; features = scores flattened, NaN->0",
        "note": (
            "system-side feature for the person encoder's fusion head. Built "
            "from what the persona said out loud in the interview and the "
            "published pole descriptions -- nothing else."
        ),
    }
    out.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


def make_plot(
    path: Path,
    cosines: np.ndarray,
    fitted_norms: np.ndarray,
    predicted_norms: np.ndarray,
    lambda_curve: Sequence[dict],
    per_stratum: dict[str, Any],
) -> None:
    """Four panels: the two bars, where the bars break, and the regularization."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0][0]
    ax.hist(cosines, bins=np.linspace(-1, 1, 33), color="#4C72B0", edgecolor="white")
    ax.axvline(0.7, color="#C44E52", lw=2, label="bar 0.7")
    ax.axvline(float(np.median(cosines)), color="#55A868", lw=2, ls="--",
               label=f"median {np.median(cosines):.3f}")
    ax.set_title("Held-out loading cosine (predicted vs fitted)")
    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("items")
    ax.legend(fontsize=8)

    ax = axes[0][1]
    ax.scatter(fitted_norms, predicted_norms, s=28, color="#4C72B0", alpha=0.8)
    lo = min(fitted_norms.min(), predicted_norms.min())
    hi = max(fitted_norms.max(), predicted_norms.max())
    ax.plot([lo, hi], [lo, hi], color="#8C8C8C", lw=1, ls=":")
    r = pearson(predicted_norms, fitted_norms)
    ax.set_title(f"Discrimination: r = {r:.3f} (bar 0.60)")
    ax.set_xlabel("fitted ||a||")
    ax.set_ylabel("predicted ||a||")

    ax = axes[1][0]
    names = sorted(per_stratum)
    values = [per_stratum[n]["median_cosine"] for n in names]
    counts = [per_stratum[n]["n"] for n in names]
    ax.bar(range(len(names)), values, color="#4C72B0")
    ax.axhline(0.7, color="#C44E52", lw=2)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([f"{n}\n(n={c})" for n, c in zip(names, counts)], fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title("Median cosine by probe stratum")

    ax = axes[1][1]
    lams = [row["lam"] for row in lambda_curve]
    med = [row["median_cosine"] for row in lambda_curve]
    ax.semilogx(lams, med, color="#4C72B0")
    best = max(lambda_curve, key=lambda row: row["median_cosine"])
    ax.axvline(best["lam"], color="#55A868", ls="--",
               label=f"chosen {best['lam']:.3g}")
    ax.set_title("Direction head: leave-one-out on the 202 training items")
    ax.set_xlabel("ridge lambda")
    ax.set_ylabel("LOO median cosine")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return pearson(_ranks(np.asarray(a, float)), _ranks(np.asarray(b, float)))


def _ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    return ranks


# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Stage 3 item encoder.")
    sub = parser.add_subparsers(dest="command", required=True)

    emit = sub.add_parser("emit", help="Write the judgment prompt batch as JSONL.")
    emit.add_argument("--out", required=True)
    emit.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    emit.add_argument("--open-repeats", type=int, default=1)
    emit.add_argument("--transcripts", default=str(TRANSCRIPTS_PATH))
    emit.add_argument("--no-open", action="store_true",
                      help="Emit only the item-side prompts.")
    emit.set_defaults(func=cmd_emit)

    train = sub.add_parser("train", help="Train the heads and score the held-out items.")
    train.add_argument("--judgments", required=True)
    train.add_argument("--embeddings", default=None)
    train.add_argument("--embed-components", type=int, default=0,
                       help="0 (default) picks the count by leave-one-out on the "
                            "202 training items; a positive value pins it.")
    train.add_argument("--repeats-recorded", type=int, default=DEFAULT_REPEATS)
    train.add_argument("--out", default=str(REPO_ROOT / "results" / "stage3_item_encoder.json"))
    train.add_argument("--predictions", default=None)
    train.add_argument("--plot", default=str(REPO_ROOT / "results" / "stage3_item_encoder.png"))
    train.set_defaults(func=cmd_train)

    feats = sub.add_parser(
        "open-features",
        help="Reduce the open-answer judgments to the person encoder's feature array.",
    )
    feats.add_argument("--judgments", required=True)
    feats.add_argument(
        "--out",
        default=str(REPO_ROOT / "data" / "runs" / "stage3_qwen" / "open_answer_features.npz"),
    )
    feats.set_defaults(func=cmd_open_features)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
