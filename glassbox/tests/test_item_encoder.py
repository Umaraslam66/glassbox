"""Unit tests for the Stage 3 item encoder.

Four things are checked, in the order they can hurt:

1. **What the prompt builder reads.** Every file the emit path opens is compared
   against an allowlist of system-side locations. A prompt that quietly picked
   up a designed loading, a persona card or a trait vector would void the stage,
   and by the time it showed up in a good-looking result it would be too late.
2. **What the prompt builder writes.** The rendered prompt must contain the item
   text and the published pole text and nothing else that came from data.
3. **Parsing.** The model replies with sampled text, so the parser is the join
   between a 27B model's manners and an 8-column feature matrix. Fixtures cover
   the clean case, the chatty case, the out-of-range case and the empty case.
4. **The heads.** Ridge fitting is deterministic, and the leave-one-out shortcut
   really is leave-one-out (checked against brute force).
"""

from __future__ import annotations

import builtins
import json
from pathlib import Path

import numpy as np
import pytest

from src.model import item_encoder as ie
from src.model.pole_descriptions import DIMENSION_CODES, HIGH_POLE, LOW_POLE

REPO_ROOT = Path(__file__).resolve().parents[1]

LIKERT_ITEM = {
    "item_id": "q001",
    "text": "I always follow the exact medication schedule my doctor writes down for me.",
    "type": "likert5",
    "options": [
        {"value": 1, "label": "strongly disagree"},
        {"value": 2, "label": "disagree"},
        {"value": 3, "label": "neither agree nor disagree"},
        {"value": 4, "label": "agree"},
        {"value": 5, "label": "strongly agree"},
    ],
    "topic_domain": "health",
}

BINARY_ITEM = {
    "item_id": "q017",
    "text": "I follow the detour signs set up by the road crew.",
    "type": "binary",
    "options": [{"value": 1, "label": "yes"}, {"value": 0, "label": "no"}],
    "topic_domain": "public-services",
}


# ---------------------------------------------------------------------------
# 1. what the prompt builder is allowed to read
# ---------------------------------------------------------------------------

#: Directories the emit path may read from, relative to the repo root. Public
#: item text, the published contract, the frozen splits, and the system-side
#: run artifacts (transcripts, which the Wall test already vets field by field).
ALLOWED_READ_PREFIXES = (
    ("data", "public"),
    ("data", "runs"),
    ("experiments",),
    ("results",),
    ("src",),
    ("PREREGISTRATION.md",),
)

#: Assembled from pieces so this test file does not itself contain the literal
#: the Wall test forbids outside the grader.
FORBIDDEN_FRAGMENTS = ("data" + "/" + "truth", "data" + "\\" + "truth", "bank_truth")


class _Recorder:
    """Records every path handed to open()/Path.open() while it is installed."""

    def __init__(self) -> None:
        self.paths: list[str] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real_open = builtins.open
        real_path_open = Path.open

        def spy_open(file, *args, **kwargs):  # noqa: ANN001
            self.paths.append(str(file))
            return real_open(file, *args, **kwargs)

        def spy_path_open(self_path, *args, **kwargs):  # noqa: ANN001
            self.paths.append(str(self_path))
            return real_path_open(self_path, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", spy_open)
        monkeypatch.setattr(Path, "open", spy_path_open)


def _is_allowed(path: str) -> bool:
    try:
        parts = Path(path).resolve().relative_to(REPO_ROOT).parts
    except ValueError:
        return True  # outside the repo entirely: stdlib, site-packages, tmp
    return any(parts[: len(prefix)] == prefix for prefix in ALLOWED_READ_PREFIXES)


@pytest.mark.skipif(
    not (REPO_ROOT / "data" / "public" / "bank_items.json").is_file(),
    reason="no public bank in this checkout",
)
def test_emitting_prompts_reads_only_system_side_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recorder = _Recorder()
    recorder.install(monkeypatch)

    items, _ = ie.load_bank()
    records = list(ie.iter_item_records(items[:4], repeats=2))
    transcripts = REPO_ROOT / "data" / "runs" / "interview_v1" / "transcripts.jsonl"
    if transcripts.is_file():
        records.extend(list(ie.iter_open_records(transcripts, repeats=1))[:64])
    ie.write_jsonl(tmp_path / "out.jsonl", records)

    assert recorder.paths, "the recorder saw no file access at all -- it is not installed"
    bad = [p for p in recorder.paths if not _is_allowed(p)]
    assert not bad, "emit read files outside the system side:\n" + "\n".join(bad)

    lowered = [p.lower() for p in recorder.paths]
    hits = [p for p in lowered if any(f in p for f in FORBIDDEN_FRAGMENTS)]
    assert not hits, "emit touched planted-truth material:\n" + "\n".join(hits)


def test_module_never_names_the_truth_tree() -> None:
    """A second, static pass over the source itself."""
    source = (REPO_ROOT / "src" / "model" / "item_encoder.py").read_text(encoding="utf-8")
    lowered = source.lower()
    for fragment in FORBIDDEN_FRAGMENTS:
        assert fragment not in lowered, f"item_encoder.py names {fragment!r}"


# ---------------------------------------------------------------------------
# 2. what ends up in the prompt
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("code", DIMENSION_CODES)
def test_item_prompt_carries_the_item_and_the_published_poles(code: str) -> None:
    system, user = ie.build_item_pole_prompt(LIKERT_ITEM, code)
    assert LIKERT_ITEM["text"] in user
    assert LOW_POLE[code] in user
    assert HIGH_POLE[code] in user
    assert "STRONGLY AGREE" in user
    assert system == ie.JUDGE_SYSTEM


def test_binary_items_are_asked_in_their_own_answer_format() -> None:
    _, user = ie.build_item_pole_prompt(BINARY_ITEM, "TRU")
    assert "answers YES" in user
    assert "yes / no" in user
    assert "1-5 agreement scale" not in user


def test_prompts_carry_no_design_labels() -> None:
    """The public bank has no loading information; make sure none leaks in anyway."""
    for code in DIMENSION_CODES:
        _, user = ie.build_item_pole_prompt(LIKERT_ITEM, code)
        for banned in ("loading", "target_dim", "strength_class", "negatively_keyed",
                       "distractor", "primary", "cross"):
            assert banned not in user.lower(), f"{banned} appeared in a prompt"


def test_open_answer_prompt_quotes_the_answer_and_the_question() -> None:
    system, user = ie.build_open_answer_prompt(
        "Tell me about a time you dealt with a bank.",
        "It went fine, the teller knew my name.",
        "TRU",
    )
    assert "Tell me about a time you dealt with a bank." in user
    assert "It went fine, the teller knew my name." in user
    assert LOW_POLE["TRU"] in user
    assert system == ie.OPEN_SYSTEM


def test_polarization_prompt_asks_about_disagreement_not_direction() -> None:
    _, user = ie.build_polarization_prompt(LIKERT_ITEM)
    assert LIKERT_ITEM["text"] in user
    assert "disagree with each other" in user
    for code in DIMENSION_CODES:
        assert LOW_POLE[code] not in user


# ---------------------------------------------------------------------------
# 3. record structure and seeding identity
# ---------------------------------------------------------------------------


def test_every_item_record_gets_its_own_seed_identity() -> None:
    """The cluster driver seeds from (pid, item_id, round).

    Two records sharing that triple would share random numbers, which is exactly
    the bug that inflated the Stage 1 test-retest number. Repeats and poles must
    therefore be distinguishable inside ``round``.
    """
    records = list(ie.iter_item_records([LIKERT_ITEM, BINARY_ITEM], repeats=3))
    identities = [(r.get("pid", ""), r["item_id"], r["round"]) for r in records]
    assert len(set(identities)) == len(identities)

    expected = 2 * len(DIMENSION_CODES) * 3 + 2 * 3
    assert len(records) == expected


def test_open_records_cover_every_pole_for_every_answer(tmp_path: Path) -> None:
    path = tmp_path / "transcripts.jsonl"
    path.write_text(
        json.dumps(
            {
                "pid": "p0001",
                "profile": {"pid": "p0001"},
                "turns": [],
                "open": [
                    {"prompt_id": "oe01", "question": "Q1?", "answer": "A1."},
                    {"prompt_id": "oe16", "question": "Q2?", "answer": "A2."},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    records = list(ie.iter_open_records(path, repeats=1))
    assert len(records) == 2 * len(DIMENSION_CODES)
    assert {r["pole"] for r in records} == set(DIMENSION_CODES)
    assert {r["pid"] for r in records} == {"p0001"}
    # the persona's public profile is deliberately NOT in the prompt
    assert all("profile" not in r["user"] for r in records)


# ---------------------------------------------------------------------------
# 4. parsing what the model sends back
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "completion,expected",
    [
        ("2", 2),
        ("-2", -2),
        ("+3", 3),
        ("  0\n", 0),
        ("3", 3),
        ("-3", -3),
        ("Score: 2", 2),
        ("I would say -1.", -1),
        ("on a -3 to +3 scale I would say 2", 2),  # last in-range number wins
        ("", None),
        ("   ", None),
        ("no idea", None),
        ("7", None),  # out of range is a parse failure, not a strong opinion
        ("-9", None),
    ],
)
def test_parse_score_fixtures(completion: str, expected: int | None) -> None:
    assert ie.parse_score(completion) == expected


def test_polarization_uses_its_own_range() -> None:
    assert ie.parse_score("6", ie.POLAR_MIN, ie.POLAR_MAX) == 6
    assert ie.parse_score("-1", ie.POLAR_MIN, ie.POLAR_MAX) is None
    assert ie.parse_score("6") is None  # out of range on the judgment scale


def test_clean_reply_detection() -> None:
    assert ie.is_clean_reply("2")
    assert ie.is_clean_reply(" -3 ")
    assert not ie.is_clean_reply("Score: 2")
    assert not ie.is_clean_reply("")


def test_load_judgments_groups_and_counts(tmp_path: Path) -> None:
    path = tmp_path / "judge_out.jsonl"
    rows = [
        {"kind": "item", "item_id": "q001", "pole": "TRU", "rep": 0, "completion": "2"},
        {"kind": "item", "item_id": "q001", "pole": "TRU", "rep": 1, "completion": "3"},
        {"kind": "item", "item_id": "q001", "pole": "RSK", "rep": 0, "completion": "0"},
        {"kind": "item", "item_id": "q001", "pole": "RSK", "rep": 1, "completion": "hmm"},
        {"kind": "polar", "item_id": "q001", "pole": "", "rep": 0, "completion": "4"},
        {"kind": "open", "pid": "p1", "item_id": "oe01", "pole": "TRU", "rep": 0,
         "completion": "The answer suggests -2"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    parsed = ie.load_judgments(path)
    assert parsed["item"]["q001"]["TRU"] == [2, 3]
    assert parsed["item"]["q001"]["RSK"] == [0]
    assert parsed["polar"]["q001"] == [4]
    assert parsed["open"][("p1", "oe01")]["TRU"] == [-2]
    assert parsed["stats"] == {"records": 6, "clean": 4, "parsed": 5, "unparsed": 1}


def test_missing_poles_read_as_no_information_low_confidence(tmp_path: Path) -> None:
    path = tmp_path / "j.jsonl"
    path.write_text(
        json.dumps({"kind": "item", "item_id": "q001", "pole": "TRU", "rep": 0,
                    "completion": "3"}) + "\n",
        encoding="utf-8",
    )
    parsed = ie.load_judgments(path)
    means, spreads = ie.judgment_features(parsed, ["q001", "q999"])
    assert means[0, 0] == 3.0
    assert spreads[0, 0] == 0.0
    assert np.all(means[1] == 0.0)          # unknown item: says nothing
    assert np.all(spreads[1] == 3.0)        # ... with the widest possible spread
    assert means[0, 1] == 0.0               # unjudged pole of a known item


# ---------------------------------------------------------------------------
# 5. wording features
# ---------------------------------------------------------------------------


def test_wording_features_read_the_surface_only() -> None:
    is_binary, n_words, hedge, absolute = ie.wording_features(LIKERT_ITEM)
    assert is_binary == 0.0
    assert n_words == 13
    assert hedge == 0.0
    assert absolute == 1.0  # "always"

    is_binary, _, hedge, absolute = ie.wording_features(BINARY_ITEM)
    assert is_binary == 1.0
    assert hedge == 0.0
    assert absolute == 0.0

    hedged = dict(LIKERT_ITEM, text="I sometimes check the label.")
    assert ie.wording_features(hedged)[2] == 1.0


# ---------------------------------------------------------------------------
# 6. the heads
# ---------------------------------------------------------------------------


def _toy(seed: int = 0, n: int = 40, p: int = 6, m: int = 3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    W = rng.normal(size=(p, m))
    Y = X @ W + 0.3 * rng.normal(size=(n, m))
    return X, Y


def test_ridge_is_deterministic() -> None:
    X, Y = _toy()
    a = ie.ridge_fit(X, Y, 1.0)
    b = ie.ridge_fit(X, Y, 1.0)
    assert np.array_equal(a["W"], b["W"])
    assert np.array_equal(ie.ridge_predict(a, X), ie.ridge_predict(b, X))


def test_leave_one_out_shortcut_matches_brute_force() -> None:
    """The hat-matrix identity is the whole basis of the lambda search."""
    X, Y = _toy(seed=3, n=25, p=4, m=2)
    lam = 2.5
    fast = ie.ridge_loo(X, Y, lam)

    mean, scale = ie._standardizer(X)
    D = ie._design(X, mean, scale)
    penalty = np.eye(D.shape[1]) * lam
    penalty[-1, -1] = 0.0

    slow = np.zeros_like(Y)
    for i in range(X.shape[0]):
        keep = np.arange(X.shape[0]) != i
        Di, Yi = D[keep], Y[keep]
        W = np.linalg.solve(Di.T @ Di + penalty, Di.T @ Yi)
        slow[i] = D[i] @ W

    assert np.allclose(fast, slow, atol=1e-8)


def test_direction_head_recovers_a_linear_map_and_is_deterministic() -> None:
    rng = np.random.default_rng(11)
    X = rng.normal(size=(120, 8))
    W = rng.normal(size=(8, 8))
    Y = X @ W
    head_a = ie.train_direction_head(X, Y)
    head_b = ie.train_direction_head(X, Y)
    assert head_a["lam"] == head_b["lam"]
    assert np.array_equal(head_a["W"], head_b["W"])

    cos = ie.row_cosines(ie.ridge_predict(head_a, X), Y)
    assert np.median(cos) > 0.99


def test_magnitude_head_predicts_in_log_space_and_is_deterministic() -> None:
    rng = np.random.default_rng(5)
    X = rng.normal(size=(120, 5))
    norms = np.exp(X @ rng.normal(size=5) * 0.4 + 0.5)
    head_a = ie.train_magnitude_head(X, norms)
    head_b = ie.train_magnitude_head(X, norms)
    assert head_a["lam"] == head_b["lam"]
    assert np.array_equal(head_a["W"], head_b["W"])
    pred = np.exp(ie.ridge_predict(head_a, X)[:, 0])
    assert ie.pearson(pred, norms) > 0.9


def test_direction_and_magnitude_are_predicted_separately() -> None:
    rng = np.random.default_rng(7)
    X = rng.normal(size=(60, 4))
    direction = ie.ridge_fit(X, rng.normal(size=(60, 8)), 1.0)
    magnitude = ie.ridge_fit(X, rng.normal(size=(60, 1)), 1.0)
    vectors, norms = ie.predict_loadings(direction, magnitude, X, X)
    assert np.allclose(np.linalg.norm(vectors, axis=1), norms)


def test_embedding_reduction_is_fitted_on_training_rows_only() -> None:
    rng = np.random.default_rng(13)
    train = rng.normal(size=(50, 200))
    holdout = rng.normal(size=(10, 200))
    a_tr, a_ho = ie.reduce_embeddings(train, holdout, 5)
    assert a_tr.shape == (50, 5)
    assert a_ho.shape == (10, 5)

    # changing a held-out row must not move a single training component
    holdout2 = holdout.copy()
    holdout2[0] += 100.0
    b_tr, b_ho = ie.reduce_embeddings(train, holdout2, 5)
    assert np.allclose(a_tr, b_tr)
    assert not np.allclose(a_ho[0], b_ho[0])


def test_open_feature_export_shapes_and_missing_cells(tmp_path: Path) -> None:
    """The handoff to the person encoder: one row per persona, NaN where unheard."""
    splits = json.loads((REPO_ROOT / "experiments" / "splits_v1.json").read_text())
    prompts = splits["interview_open"]

    rows = []
    for pid in ("p0001", "p0002"):
        for prompt_id in prompts:
            for code in DIMENSION_CODES:
                if pid == "p0002" and prompt_id == prompts[0] and code == "TRU":
                    continue  # one missing cell on purpose
                rows.append(
                    {"kind": "open", "pid": pid, "item_id": prompt_id, "pole": code,
                     "rep": 0, "completion": "1"}
                )
    judgments = tmp_path / "judge_out.jsonl"
    judgments.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    out = tmp_path / "open_answer_features.npz"
    ie.main(["open-features", "--judgments", str(judgments), "--out", str(out)])

    data = np.load(out, allow_pickle=True)
    assert list(data["pids"]) == ["p0001", "p0002"]
    assert data["scores"].shape == (2, len(prompts), 8)
    assert data["features"].shape == (2, len(prompts) * 8)
    assert np.isnan(data["scores"][1, 0, 0])
    assert data["features"][1, 0] == 0.0
    assert not data["observed"][1, 0, 0]
    assert data["observed"].sum() == data["scores"].size - 1


def test_unit_rows_survives_a_zero_vector() -> None:
    M = np.array([[0.0, 0.0], [3.0, 4.0]])
    out = ie.unit_rows(M)
    assert np.allclose(out[1], [0.6, 0.8])
    assert np.all(np.isfinite(out))
