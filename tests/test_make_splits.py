"""Tests for the frozen train / held-out splits (PREREGISTRATION.md section 5).

Two things have to hold, and both are checked here.

  * The splits are **frozen**: the same seed gives the same file, byte for byte,
    and every constraint the orchestrator fixed is actually satisfied.
  * The manifest is **committable**: it goes into git, so it may carry ids,
    stratum labels and counts, and nothing else. The last two tests are the
    strong form of that -- no number in the file is a float, and none of the
    eight dimension codes appears anywhere in the text. Which trait an
    interview item measures is part of the answer key and stays behind the
    Wall.

The bank used here is synthetic and built in a temporary directory, with the
same shape as the real one (252 closed items in the four design cells, 20 open
prompts, 500 personas). The tests never look at the real planted directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.bank.schema import (
    Bank,
    BankSpec,
    ClosedItem,
    DIMENSION_LABELS,
    DIMENSIONS,
    OpenItem,
    TOPIC_DOMAINS,
    closed_item_id,
    open_item_id,
    write_bank,
)
from src.eval import make_splits

N_PERSONAS = 500


def build_bank() -> Bank:
    """A synthetic bank with the real design shape and a workable domain spread."""
    spec = BankSpec()
    closed: list[ClosedItem] = []

    # Primary items: 24 per dimension (16 Likert, 8 binary), each dimension
    # drawing on a rotating window of four topic domains.
    for dim_index, dimension in enumerate(DIMENSIONS):
        window = [TOPIC_DOMAINS[(dim_index + k) % len(TOPIC_DOMAINS)] for k in range(4)]
        for j in range(spec.per_dim):
            sign = -1.0 if j % 5 == 0 else 1.0
            closed.append(
                ClosedItem(
                    item_id=closed_item_id(len(closed)),
                    text=f"primary statement {len(closed)}",
                    item_type="likert5" if j < spec.likert_per_dim else "binary",
                    topic_domain=window[j % 4],
                    loadings={dimension: sign * spec.strong_loading},
                    strength_class="strong",
                    negatively_keyed=sign < 0,
                )
            )

    # Cross-loading items: 0.7 on one dimension, 0.5 on the next.
    for j in range(spec.n_cross):
        main = DIMENSIONS[j % len(DIMENSIONS)]
        second = DIMENSIONS[(j + 1) % len(DIMENSIONS)]
        closed.append(
            ClosedItem(
                item_id=closed_item_id(len(closed)),
                text=f"cross statement {len(closed)}",
                item_type="likert5" if j % 2 else "binary",
                topic_domain=TOPIC_DOMAINS[j % len(TOPIC_DOMAINS)],
                loadings={main: spec.cross_primary_loading, second: spec.cross_secondary_loading},
                strength_class="strong",
            )
        )

    # Designed-weak items.
    for j in range(spec.n_weak):
        closed.append(
            ClosedItem(
                item_id=closed_item_id(len(closed)),
                text=f"weak statement {len(closed)}",
                item_type="likert5",
                topic_domain=TOPIC_DOMAINS[(j + 3) % len(TOPIC_DOMAINS)],
                loadings={DIMENSIONS[j % len(DIMENSIONS)]: spec.weak_loading},
                strength_class="weak",
            )
        )

    # Distractors: no trait signal at all.
    for j in range(spec.n_distractor):
        closed.append(
            ClosedItem(
                item_id=closed_item_id(len(closed)),
                text=f"distractor statement {len(closed)}",
                item_type="binary" if j % 2 else "likert5",
                topic_domain=TOPIC_DOMAINS[(j + 7) % len(TOPIC_DOMAINS)],
                loadings={},
                strength_class="none",
            )
        )

    open_items: list[OpenItem] = []
    for j in range(spec.n_open):
        if j < len(DIMENSIONS):
            targets = (DIMENSIONS[j],)
        else:
            targets = (DIMENSIONS[j % len(DIMENSIONS)], DIMENSIONS[(j + 3) % len(DIMENSIONS)])
        open_items.append(
            OpenItem(
                item_id=open_item_id(j),
                text=f"open prompt {j}",
                target_dims=targets,
                topic_domain=TOPIC_DOMAINS[j % len(TOPIC_DOMAINS)],
            )
        )

    return Bank(closed=closed, open_items=open_items, spec=spec, seed=1, generator="test")


@pytest.fixture(scope="module")
def planted(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A planted directory: bank design, batch manifest, one file per persona."""
    root = tmp_path_factory.mktemp("splits")
    hidden = root / "hidden"
    write_bank(build_bank(), root / "open", hidden)

    (hidden / make_splits.BATCH_MANIFEST_FILENAME).write_text(
        json.dumps({"n": N_PERSONAS, "seed": 548, "date": "2026-07-30"}), encoding="utf-8"
    )
    theta_dir = hidden / "theta"
    theta_dir.mkdir(parents=True, exist_ok=True)
    for index in range(1, N_PERSONAS + 1):
        (theta_dir / f"p{index:04d}.json").write_text("{}", encoding="utf-8")
    return hidden


@pytest.fixture(scope="module")
def bank() -> Bank:
    return build_bank()


@pytest.fixture(scope="module")
def built(planted: Path, bank: Bank) -> tuple[dict, dict]:
    ids = make_splits.read_persona_ids(planted)
    return make_splits.build_splits(bank, ids, created="2026-01-01")


@pytest.fixture(scope="module")
def manifest(built: tuple[dict, dict]) -> dict:
    return built[0]


@pytest.fixture(scope="module")
def detail(built: tuple[dict, dict]) -> dict:
    return built[1]


# --------------------------------------------------------------------------
# Shape and counts
# --------------------------------------------------------------------------


def test_persona_ids_come_from_the_batch_manifest(planted: Path) -> None:
    ids = make_splits.read_persona_ids(planted)
    assert len(ids) == N_PERSONAS
    assert ids[0] == "p0001" and ids[-1] == f"p{N_PERSONAS:04d}"


def test_manifest_has_exactly_the_agreed_keys(manifest: dict) -> None:
    assert set(manifest) == {
        "schema",
        "seed",
        "created",
        "persona_holdout",
        "item_holdout",
        "interview_closed",
        "interview_open",
        "strata",
        "stratum_counts",
        "constraint_report",
    }
    assert manifest["seed"] == make_splits.SPLIT_SEED


def test_holdout_sizes(manifest: dict, bank: Bank) -> None:
    assert len(manifest["persona_holdout"]) == make_splits.N_PERSONA_HOLDOUT
    assert len(manifest["item_holdout"]) == make_splits.N_ITEM_HOLDOUT
    assert len(manifest["interview_closed"]) == make_splits.N_INTERVIEW_CLOSED
    assert len(manifest["interview_open"]) == make_splits.N_INTERVIEW_OPEN
    # 20% of each, as PREREGISTRATION section 5 says.
    assert len(manifest["persona_holdout"]) * 5 == N_PERSONAS
    assert len(manifest["item_holdout"]) * 5 <= len(bank.closed) + 4


def test_holdout_ids_are_real_unique_and_sorted(manifest: dict, planted: Path, bank: Bank) -> None:
    persona_ids = set(make_splits.read_persona_ids(planted))
    item_ids = {item.item_id for item in bank.closed}

    for key, universe in (("persona_holdout", persona_ids), ("item_holdout", item_ids)):
        held = manifest[key]
        assert len(set(held)) == len(held), f"{key} repeats an id"
        assert held == sorted(held), f"{key} is not sorted"
        assert set(held) <= universe, f"{key} names something outside the batch"


def test_open_prompts_are_not_split(manifest: dict) -> None:
    assert manifest["constraint_report"]["open_prompts_split"] == 0


# --------------------------------------------------------------------------
# Interview-set constraints
# --------------------------------------------------------------------------


def test_interview_uses_training_items_only(manifest: dict) -> None:
    assert not set(manifest["interview_closed"]) & set(manifest["item_holdout"])


def test_interview_covers_every_dimension_with_unit_primary_items(
    manifest: dict, bank: Bank
) -> None:
    by_id = {item.item_id: item for item in bank.closed}
    chosen = [by_id[item_id] for item_id in manifest["interview_closed"]]

    assert all(make_splits.is_interview_candidate(item) for item in chosen)

    per_dimension: dict[str, int] = {}
    for item in chosen:
        dimension = make_splits.primary_dimension(item)
        per_dimension[dimension] = per_dimension.get(dimension, 0) + 1

    assert set(per_dimension) == set(DIMENSIONS), "not every dimension is covered"
    assert sorted(per_dimension.values()) == [1] + [2] * 7


def test_interview_domain_cap_leaves_room_for_a_far_stratum(
    manifest: dict, bank: Bank
) -> None:
    by_id = {item.item_id: item for item in bank.closed}
    domains = {by_id[item_id].topic_domain for item_id in manifest["interview_closed"]}
    assert len(domains) <= make_splits.MAX_INTERVIEW_DOMAINS

    all_domains = {item.topic_domain for item in bank.closed}
    assert len(all_domains - domains) >= 5

    # The three open prompts eat into that headroom; something must survive it.
    _, touched = domain_rules(manifest, bank)
    assert all_domains - touched, "the whole interview leaves no domain untouched"


def test_interview_type_mix(manifest: dict, bank: Bank) -> None:
    by_id = {item.item_id: item for item in bank.closed}
    types = [by_id[item_id].item_type for item_id in manifest["interview_closed"]]
    assert types.count("binary") >= make_splits.MIN_INTERVIEW_BINARY
    assert types.count("likert5") >= make_splits.MIN_INTERVIEW_LIKERT


def test_scripted_order_is_a_permutation_of_the_selection(
    manifest: dict, detail: dict
) -> None:
    scripted = manifest["interview_closed"]
    selected = detail["interview_selection_order"]
    assert len(set(scripted)) == len(scripted)
    assert sorted(scripted) == sorted(selected)
    # The selection walks the dimensions in order; the scripted order must not.
    assert scripted != selected


def test_open_prompts_target_three_different_dimensions(
    manifest: dict, bank: Bank
) -> None:
    by_id = {item.item_id: item for item in bank.open_items}
    chosen = [by_id[item_id] for item_id in manifest["interview_open"]]
    dims = [make_splits.open_primary_dimension(item) for item in chosen]
    assert len(set(dims)) == make_splits.N_INTERVIEW_OPEN
    assert len(set(manifest["interview_open"])) == make_splits.N_INTERVIEW_OPEN


def test_constraint_report_agrees_with_the_manifest(manifest: dict, bank: Bank) -> None:
    report = manifest["constraint_report"]
    domain_map, touched = domain_rules(manifest, bank)
    all_domains = {item.topic_domain for item in bank.closed}
    assert report["interview_domains_used"] == len(domain_map)
    assert report["interview_domains_touched_with_open"] == len(touched)
    assert report["domains_untouched_by_interview"] == len(all_domains - touched)
    assert report["dimensions_covered"] == report["dimensions_total"] == len(DIMENSIONS)
    assert report["dimensions_with_two_items"] == 7
    assert report["dimensions_with_one_item"] == 1
    assert report["all_interview_items_primary_unit_loading"] is True
    assert report["interview_from_training_items_only"] is True
    assert report["scripted_order_is_permutation"] is True
    assert report["interview_domains_used"] <= report["interview_domains_allowed"]
    assert report["interview_domains_touched_with_open"] >= report["interview_domains_used"]
    assert report["domains_untouched_by_interview"] <= report["domains_uncovered_by_closed_set"]
    assert report["domains_untouched_by_interview"] > 0
    assert report["interview_binary_items"] >= report["interview_binary_required"]
    assert report["interview_likert_items"] >= report["interview_likert_required"]
    assert report["held_out_items_stratified"] == len(manifest["item_holdout"])
    assert report["closed_items_train"] + report["closed_items_held_out"] == (
        report["closed_items_total"]
    )
    assert report["personas_train"] + report["personas_held_out"] == report["personas_total"]


# --------------------------------------------------------------------------
# Probe strata
# --------------------------------------------------------------------------


def test_every_held_out_item_has_one_stratum(manifest: dict) -> None:
    strata = manifest["strata"]
    assert set(strata) == set(manifest["item_holdout"])
    assert set(strata.values()) <= set(make_splits.STRATA)

    counts = manifest["stratum_counts"]
    assert set(counts) == set(make_splits.STRATA)
    assert sum(counts.values()) == len(manifest["item_holdout"])
    for label in make_splits.STRATA:
        assert counts[label] == sum(1 for s in strata.values() if s == label)


def test_far_stratum_is_not_empty(manifest: dict) -> None:
    assert manifest["stratum_counts"]["far"] > 0


def domain_rules(manifest: dict, bank: Bank) -> tuple[dict, set[str]]:
    """The closed-set domain map and the domains the whole interview touches."""
    by_id = {item.item_id: item for item in bank.closed}
    interview = [by_id[item_id] for item_id in manifest["interview_closed"]]
    domain_map = make_splits.interview_domain_map(interview)

    open_by_id = {item.item_id: item for item in bank.open_items}
    open_chosen = [open_by_id[item_id] for item_id in manifest["interview_open"]]
    touched = make_splits.touched_domains(domain_map, open_chosen)
    return domain_map, touched


def test_strata_follow_the_rule(manifest: dict, bank: Bank) -> None:
    by_id = {item.item_id: item for item in bank.closed}
    domain_map, touched = domain_rules(manifest, bank)

    for item_id, label in manifest["strata"].items():
        item = by_id[item_id]
        dimension = make_splits.primary_dimension(item)
        if label == "far":
            assert item.topic_domain not in touched
        else:
            assert item.topic_domain in touched
            in_domain = domain_map.get(item.topic_domain, set())
            if label == "near":
                assert dimension is not None and dimension in in_domain
            else:
                assert dimension is None or dimension not in in_domain


def test_far_means_untouched_by_the_open_prompts_too(manifest: dict, bank: Bank) -> None:
    """An open prompt's domain can never hold a far item."""
    by_id = {item.item_id: item for item in bank.closed}
    open_by_id = {item.item_id: item for item in bank.open_items}
    open_domains = {open_by_id[i].topic_domain for i in manifest["interview_open"]}

    for item_id, label in manifest["strata"].items():
        if by_id[item_id].topic_domain in open_domains:
            assert label != "far", f"{item_id} sits in a domain an open prompt asks about"


def test_a_domain_only_the_open_prompts_touch_is_same_domain(
    manifest: dict, bank: Bank
) -> None:
    """The rule change itself: closed set silent, open prompt speaks -> same-domain."""
    by_id = {item.item_id: item for item in bank.closed}
    domain_map, _ = domain_rules(manifest, bank)
    open_by_id = {item.item_id: item for item in bank.open_items}
    open_only = {
        open_by_id[i].topic_domain for i in manifest["interview_open"]
    } - set(domain_map)

    moved = [
        item_id
        for item_id in manifest["strata"]
        if by_id[item_id].topic_domain in open_only
    ]
    for item_id in moved:
        assert manifest["strata"][item_id] == "same-domain"


def test_distractors_are_never_near(manifest: dict, bank: Bank) -> None:
    by_id = {item.item_id: item for item in bank.closed}
    for item_id, label in manifest["strata"].items():
        if make_splits.item_class(by_id[item_id]) == make_splits.DISTRACTOR:
            assert label != "near"


# --------------------------------------------------------------------------
# Nothing truth-valued in the committed file
# --------------------------------------------------------------------------


def walk(payload: object) -> list[object]:
    """Every value anywhere in a nested JSON structure, keys included."""
    found: list[object] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            found.append(key)
            found.extend(walk(value))
    elif isinstance(payload, list):
        for value in payload:
            found.extend(walk(value))
    else:
        found.append(payload)
    return found


def test_manifest_holds_no_floats(manifest: dict) -> None:
    floats = [value for value in walk(manifest) if isinstance(value, float)]
    assert not floats, f"the manifest carries numbers with decimals: {floats}"


def test_manifest_names_no_dimension(manifest: dict) -> None:
    text = json.dumps(manifest, indent=2)

    for code in DIMENSIONS:
        assert code not in text, f"dimension code {code} leaked into the manifest"

    for label in DIMENSION_LABELS.values():
        assert label.lower() not in text.lower(), f"dimension label {label!r} leaked"

    lowered = {code.lower() for code in DIMENSIONS}
    for value in walk(manifest):
        if isinstance(value, str):
            assert value.lower() not in lowered, f"{value!r} is a dimension code"


def test_manifest_holds_no_item_text_or_domain(manifest: dict, bank: Bank) -> None:
    text = json.dumps(manifest)
    for domain in TOPIC_DOMAINS:
        assert domain not in text, f"topic domain {domain!r} leaked into the manifest"
    for item in bank.closed[:5] + bank.closed[-5:]:
        assert item.text not in text


# --------------------------------------------------------------------------
# The CLI, and the freeze
# --------------------------------------------------------------------------


def test_cli_writes_the_same_bytes_twice(planted: Path, tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    for out in (first, second):
        code = make_splits.main(
            ["--truth", str(planted), "--out", str(out), "--created", "2026-01-01"]
        )
        assert code == 0

    assert first.read_bytes() == second.read_bytes()


def test_the_seed_is_the_only_thing_that_moves(planted: Path, bank: Bank) -> None:
    ids = make_splits.read_persona_ids(planted)
    frozen, _ = make_splits.build_splits(bank, ids, created="2026-01-01")
    other, _ = make_splits.build_splits(bank, ids, seed=1, created="2026-01-01")
    assert other["persona_holdout"] != frozen["persona_holdout"]
    assert other["item_holdout"] != frozen["item_holdout"]


def test_created_stamp_does_not_touch_the_splits(planted: Path, bank: Bank) -> None:
    ids = make_splits.read_persona_ids(planted)
    early, _ = make_splits.build_splits(bank, ids, created="2026-01-01")
    late, _ = make_splits.build_splits(bank, ids, created="2027-12-31")
    early.pop("created")
    late.pop("created")
    assert early == late
