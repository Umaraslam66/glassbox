"""Data model for the question bank: items, the design spec, and (de)serialization.

Two views of the same bank, and the split between them is a Wall rule:

  * the **public** view is what the interviewer, the encoders and the predictor
    are allowed to see -- item id, text, type, answer options, topic domain;
  * the **private** view adds the planted design -- which traits an item is
    supposed to load on, how strongly, whether agreeing means a high or a low
    trait value, and the note explaining why the slot exists.

The private side is the answer key for Stage 2's item-recovery test. It is
written only to a directory named on the command line, never to a path spelled
out in this file.

Vocabulary used throughout:

  loading         a number per trait dimension. +1.0 means "agreeing with this
                  statement signals a high value on that trait"; -1.0 means
                  agreeing signals a low value. 0 means the trait is irrelevant.
  strength_class  "strong" (designed to discriminate: magnitude 1.0, or the
                  0.7/0.5 pair on cross-loading items), "weak" (magnitude 0.25,
                  deliberately mushy wording), "none" (a distractor: no trait
                  signal at all).
  topic_domain    what the item is *about*, independent of which trait it
                  measures. Needed later to stratify held-out probe items by
                  semantic distance from the interview set.
  amendment       a written, owner-authorised change to one field of one item
                  after the bank was frozen. The bank is final once planted, so
                  every later edit has to say what changed, when, who allowed it
                  and which log entry justifies it. Amendments live on the
                  private side only.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# --------------------------------------------------------------------------
# Fixed vocabularies (PREREGISTRATION section 2 for the dimensions)
# --------------------------------------------------------------------------

#: The eight planted trait dimensions, in their frozen order.
DIMENSIONS: tuple[str, ...] = ("TRU", "RSK", "ENV", "PRC", "TRD", "SOC", "TEC", "LOC")

DIMENSION_LABELS: dict[str, str] = {
    "TRU": "institution trust",
    "RSK": "risk appetite",
    "ENV": "environmentalism",
    "PRC": "price sensitivity",
    "TRD": "traditionalism",
    "SOC": "sociability",
    "TEC": "tech optimism",
    "LOC": "locality attachment",
}

#: What a person at the low end of each dimension looks like.
DIMENSION_LOW: dict[str, str] = {
    "TRU": "assumes officials lie; avoids banks, doctors and forms",
    "RSK": "keeps everything in a savings account; insures against everything",
    "ENV": "annoyed by green rules; will not pay extra for anything green",
    "PRC": "buys what looks good without checking the price",
    "TRD": "breaks conventions; distrusts 'how it has always been done'",
    "SOC": "keeps to themselves; a small circle; declines invitations",
    "TEC": "treats new technology as a scam or a threat",
    "LOC": "would move anywhere; feels no pull to any hometown",
}

#: What a person at the high end of each dimension looks like.
DIMENSION_HIGH: dict[str, str] = {
    "TRU": "defends public services; follows official advice",
    "RSK": "invests, gambles, quits jobs, moves abroad",
    "ENV": "organises recycling; pays green premiums",
    "PRC": "compares unit prices; waits for discounts",
    "TRD": "keeps customs, family roles and established ways",
    "SOC": "joins clubs; knows the neighbours; hosts people",
    "TEC": "early adopter; believes technology solves problems",
    "LOC": "rooted; follows local news, local pride, will not move",
}

#: What items are about. Independent of which trait they measure.
TOPIC_DOMAINS: tuple[str, ...] = (
    "money",
    "public-services",
    "environment",
    "shopping",
    "family-traditions",
    "social-life",
    "technology",
    "neighborhood",
    "work",
    "health",
    "travel",
    "media",
)

ITEM_TYPES: tuple[str, ...] = ("likert5", "binary")
STRENGTH_CLASSES: tuple[str, ...] = ("strong", "weak", "none")

LIKERT_OPTIONS: tuple[dict[str, Any], ...] = (
    {"value": 1, "label": "strongly disagree"},
    {"value": 2, "label": "disagree"},
    {"value": 3, "label": "neither agree nor disagree"},
    {"value": 4, "label": "agree"},
    {"value": 5, "label": "strongly agree"},
)

BINARY_OPTIONS: tuple[dict[str, Any], ...] = (
    {"value": 1, "label": "yes"},
    {"value": 0, "label": "no"},
)

#: File names written by ``generate.py`` and read by ``validate.py``.
PUBLIC_FILENAME = "bank_items.json"
PRIVATE_FILENAME = "bank_truth.json"
OPEN_PRIVATE_FILENAME = "open_items.json"

BANK_FORMAT_VERSION = 1

#: Keys that must never appear anywhere in the public file.
PRIVATE_ONLY_KEYS: tuple[str, ...] = (
    "loadings",
    "strength_class",
    "strength",
    "negatively_keyed",
    "keyed",
    "target_dims",
    "design_notes",
    "seed",
    "spec",
    "amendments",
)

#: Every field an entry in the private file's ``amendments`` list must carry.
AMENDMENT_KEYS: tuple[str, ...] = (
    "item_id",
    "field",
    "old_value",
    "new_value",
    "date",
    "authorized_by",
    "reason",
    "log_citation",
)


def options_for(item_type: str) -> list[dict[str, Any]]:
    """The fixed answer options for an item type."""
    if item_type == "likert5":
        return [dict(option) for option in LIKERT_OPTIONS]
    if item_type == "binary":
        return [dict(option) for option in BINARY_OPTIONS]
    raise ValueError(f"unknown item type {item_type!r}; known: {', '.join(ITEM_TYPES)}")


def normalize_text(text: str) -> str:
    """Fold an item's text down to a comparison key for near-duplicate checks.

    Case, punctuation and whitespace are thrown away, so "I trust the council."
    and "i trust  the Council" collide.
    """
    lowered = text.strip().lower()
    letters_only = re.sub(r"[^a-z0-9 ]+", " ", lowered)
    return re.sub(r"\s+", " ", letters_only).strip()


def closed_item_id(index: int) -> str:
    """Closed-item id for a zero-based position: 0 -> ``q001``."""
    return f"q{index + 1:03d}"


def open_item_id(index: int) -> str:
    """Open-ended item id for a zero-based position: 0 -> ``oe01``."""
    return f"oe{index + 1:02d}"


# --------------------------------------------------------------------------
# The design spec: how many items of each kind the bank must contain
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BankSpec:
    """Counts and loading magnitudes that define the bank's design.

    The defaults are the frozen full design: 252 closed items (192 primary +
    30 cross-loading + 15 designed-weak + 15 distractors) and 20 open-ended
    prompts. ``reduced()`` gives a small version with every design cell still
    present, used by the tests so a round trip does not need 100 API calls.
    """

    likert_per_dim: int = 16
    binary_per_dim: int = 8
    neg_likert_per_dim: int = 5
    neg_binary_per_dim: int = 3
    n_cross: int = 30
    n_weak: int = 15
    n_distractor: int = 15
    n_open: int = 20

    #: Distinct unordered dimension pairs the cross-loading items must cover.
    min_cross_pairs: int = 12
    #: Whether every topic domain has to appear somewhere in the bank.
    require_all_domains: bool = True

    strong_loading: float = 1.0
    cross_primary_loading: float = 0.7
    cross_secondary_loading: float = 0.5
    weak_loading: float = 0.25

    def __post_init__(self) -> None:
        if self.neg_likert_per_dim > self.likert_per_dim:
            raise ValueError("more negatively keyed Likert items than Likert items")
        if self.neg_binary_per_dim > self.binary_per_dim:
            raise ValueError("more negatively keyed binary items than binary items")
        for name in ("n_cross", "n_weak", "n_distractor", "n_open"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must not be negative")

    @property
    def per_dim(self) -> int:
        """Primary items per dimension."""
        return self.likert_per_dim + self.binary_per_dim

    @property
    def n_primary(self) -> int:
        return self.per_dim * len(DIMENSIONS)

    @property
    def n_closed(self) -> int:
        return self.n_primary + self.n_cross + self.n_weak + self.n_distractor

    @classmethod
    def reduced(cls) -> "BankSpec":
        """A miniature bank that still exercises every design cell."""
        return cls(
            likert_per_dim=2,
            binary_per_dim=2,
            neg_likert_per_dim=1,
            neg_binary_per_dim=1,
            n_cross=6,
            n_weak=4,
            n_distractor=3,
            n_open=8,
            min_cross_pairs=6,
            require_all_domains=False,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BankSpec":
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in fields})


# --------------------------------------------------------------------------
# Items
# --------------------------------------------------------------------------


@dataclass
class ClosedItem:
    """One closed-form item: a statement plus its planted design."""

    item_id: str
    text: str
    item_type: str
    topic_domain: str
    loadings: dict[str, float] = field(default_factory=dict)
    strength_class: str = "strong"
    negatively_keyed: bool = False
    design_notes: str = ""

    def __post_init__(self) -> None:
        if self.item_type not in ITEM_TYPES:
            raise ValueError(f"unknown item type {self.item_type!r}")
        if self.strength_class not in STRENGTH_CLASSES:
            raise ValueError(f"unknown strength class {self.strength_class!r}")
        unknown = set(self.loadings) - set(DIMENSIONS)
        if unknown:
            raise ValueError(f"unknown dimensions in loadings: {sorted(unknown)}")

    @property
    def options(self) -> list[dict[str, Any]]:
        return options_for(self.item_type)

    @property
    def dims(self) -> tuple[str, ...]:
        """Dimensions this item actually touches, in frozen dimension order."""
        return tuple(d for d in DIMENSIONS if self.loadings.get(d, 0.0) != 0.0)

    def loading_vector(self) -> tuple[float, ...]:
        """The full 8-vector of designed loadings, zeros included."""
        return tuple(float(self.loadings.get(d, 0.0)) for d in DIMENSIONS)

    def to_public(self) -> dict[str, Any]:
        """The view the system side is allowed to read."""
        return {
            "item_id": self.item_id,
            "text": self.text,
            "type": self.item_type,
            "options": self.options,
            "topic_domain": self.topic_domain,
        }

    def to_private(self) -> dict[str, Any]:
        """The view only the grader is allowed to read."""
        return {
            "item_id": self.item_id,
            "text": self.text,
            "type": self.item_type,
            "topic_domain": self.topic_domain,
            "loadings": {d: float(v) for d, v in self.loadings.items()},
            "strength_class": self.strength_class,
            "negatively_keyed": self.negatively_keyed,
            "design_notes": self.design_notes,
        }

    @classmethod
    def from_private(cls, payload: Mapping[str, Any]) -> "ClosedItem":
        return cls(
            item_id=payload["item_id"],
            text=payload["text"],
            item_type=payload["type"],
            topic_domain=payload["topic_domain"],
            loadings={str(k): float(v) for k, v in (payload.get("loadings") or {}).items()},
            strength_class=payload.get("strength_class", "strong"),
            negatively_keyed=bool(payload.get("negatively_keyed", False)),
            design_notes=payload.get("design_notes", ""),
        )


@dataclass
class OpenItem:
    """One open-ended interview prompt plus the traits it is meant to draw out."""

    item_id: str
    text: str
    target_dims: tuple[str, ...] = ()
    topic_domain: str = ""

    def __post_init__(self) -> None:
        self.target_dims = tuple(self.target_dims)
        unknown = set(self.target_dims) - set(DIMENSIONS)
        if unknown:
            raise ValueError(f"unknown target dimensions: {sorted(unknown)}")

    def to_public(self) -> dict[str, Any]:
        """Text and id only. Which traits it targets stays on the private side."""
        return {"item_id": self.item_id, "text": self.text}

    def to_private(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "text": self.text,
            "target_dims": list(self.target_dims),
            "topic_domain": self.topic_domain,
        }

    @classmethod
    def from_private(cls, payload: Mapping[str, Any]) -> "OpenItem":
        return cls(
            item_id=payload["item_id"],
            text=payload["text"],
            target_dims=tuple(payload.get("target_dims") or ()),
            topic_domain=payload.get("topic_domain", ""),
        )


# --------------------------------------------------------------------------
# The bank
# --------------------------------------------------------------------------


@dataclass
class Bank:
    """A whole question bank: closed items, open prompts, and how it was made."""

    closed: list[ClosedItem] = field(default_factory=list)
    open_items: list[OpenItem] = field(default_factory=list)
    spec: BankSpec = field(default_factory=BankSpec)
    seed: int = 0
    generator: str = ""
    #: Owner-authorised post-freeze edits, newest last. Private side only.
    amendments: list[dict[str, Any]] = field(default_factory=list)

    def public_payload(self) -> dict[str, Any]:
        return {
            "bank_version": BANK_FORMAT_VERSION,
            "n_closed": len(self.closed),
            "n_open": len(self.open_items),
            "topic_domains": list(TOPIC_DOMAINS),
            "items": [item.to_public() for item in self.closed],
            "open_ended": [item.to_public() for item in self.open_items],
        }

    def private_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "bank_version": BANK_FORMAT_VERSION,
            "seed": self.seed,
            "generator": self.generator,
            "dimensions": list(DIMENSIONS),
            "spec": self.spec.to_dict(),
            "n_closed": len(self.closed),
        }
        # Only written once there is something to write, so a bank that was
        # never amended keeps exactly the payload it has always had.
        if self.amendments:
            payload["amendments"] = [dict(entry) for entry in self.amendments]
        payload["items"] = [item.to_private() for item in self.closed]
        return payload

    def open_private_payload(self) -> dict[str, Any]:
        return {
            "bank_version": BANK_FORMAT_VERSION,
            "seed": self.seed,
            "generator": self.generator,
            "n_open": len(self.open_items),
            "items": [item.to_private() for item in self.open_items],
        }

    @classmethod
    def from_private_payloads(
        cls,
        closed_payload: Mapping[str, Any],
        open_payload: Mapping[str, Any] | None = None,
    ) -> "Bank":
        return cls(
            closed=[ClosedItem.from_private(row) for row in closed_payload.get("items", [])],
            open_items=[
                OpenItem.from_private(row)
                for row in ((open_payload or {}).get("items") or [])
            ],
            spec=BankSpec.from_dict(closed_payload.get("spec") or {}),
            seed=int(closed_payload.get("seed", 0)),
            generator=str(closed_payload.get("generator", "")),
            amendments=[dict(entry) for entry in (closed_payload.get("amendments") or [])],
        )


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def write_bank(bank: Bank, public_dir: str | Path, private_dir: str | Path) -> dict[str, Path]:
    """Write the public file and the two private files.

    Both directories come from the caller (the CLI takes them as arguments);
    nothing here knows where the planted design is kept.
    """
    public_dir = Path(public_dir)
    private_dir = Path(private_dir)
    return {
        "public": _write_json(public_dir / PUBLIC_FILENAME, bank.public_payload()),
        "private": _write_json(private_dir / PRIVATE_FILENAME, bank.private_payload()),
        "open_private": _write_json(
            private_dir / OPEN_PRIVATE_FILENAME, bank.open_private_payload()
        ),
    }


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_bank(private_dir: str | Path) -> Bank:
    """Rebuild a bank from the two private files in ``private_dir``."""
    private_dir = Path(private_dir)
    closed_payload = read_json(private_dir / PRIVATE_FILENAME)
    open_path = private_dir / OPEN_PRIVATE_FILENAME
    open_payload = read_json(open_path) if open_path.is_file() else None
    return Bank.from_private_payloads(closed_payload, open_payload)


def collect_keys(payload: Any) -> set[str]:
    """Every mapping key appearing anywhere in a nested JSON-ish structure."""
    found: set[str] = set()
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            found.add(str(key))
            found |= collect_keys(value)
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            found |= collect_keys(value)
    return found


def dimension_pairs(dims: Iterable[str]) -> tuple[str, ...]:
    """A dimension set as a sorted tuple, so TRU/RSK and RSK/TRU are one pair."""
    return tuple(sorted(dims))


def unique_texts(items: Sequence[ClosedItem | OpenItem]) -> dict[str, list[str]]:
    """Map each normalized text to the item ids that use it."""
    buckets: dict[str, list[str]] = {}
    for item in items:
        buckets.setdefault(normalize_text(item.text), []).append(item.item_id)
    return buckets
