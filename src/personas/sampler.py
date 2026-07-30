"""The dice roll: sampling a population of personas with planted traits.

Every persona starts here. We draw a hidden 8-dimensional trait vector theta
from the population distribution frozen in PREREGISTRATION.md section 2, plus
the demographics that become the persona's public profile and a per-persona
wobble level (how much the persona flips answers on questions it does not care
about).

Two things are planted truth and never leave this stage without going behind
the Wall: theta and wobble. The demographics are public by design -- they are
what a real survey would collect.

The whole draw is seed-reproducible. ``sample_population(n, seed)`` returns
bit-identical records for the same arguments, and the draws are made one
persona at a time, so a bigger batch from the same seed starts with exactly the
same people as a smaller one.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

#: Bumped whenever a change here would produce different people from the same
#: seed. Recorded in every batch manifest.
SAMPLER_VERSION = "1.0"

#: The 8 trait dimensions, in the frozen order (PREREGISTRATION.md section 2).
DIMENSIONS: tuple[str, ...] = ("TRU", "RSK", "ENV", "PRC", "TRD", "SOC", "TEC", "LOC")

#: Plain-language name of each dimension. Used by the leak check, never shown
#: to a persona.
DIMENSION_NAMES: dict[str, str] = {
    "TRU": "institution trust",
    "RSK": "risk appetite",
    "ENV": "environmentalism",
    "PRC": "price sensitivity",
    "TRD": "traditionalism",
    "SOC": "sociability",
    "TEC": "tech optimism",
    "LOC": "locality attachment",
}

#: Population correlation matrix, copied cell for cell from PREREGISTRATION.md
#: section 2. Frozen: do not touch without owner sign-off.
CORRELATION_MATRIX: tuple[tuple[float, ...], ...] = (
    # TRU    RSK    ENV    PRC    TRD    SOC    TEC    LOC
    (+1.00, +0.00, +0.10, -0.10, +0.30, +0.15, +0.15, +0.10),  # TRU
    (+0.00, +1.00, +0.00, -0.30, -0.20, +0.15, +0.40, -0.15),  # RSK
    (+0.10, +0.00, +1.00, -0.15, -0.30, +0.10, +0.05, +0.00),  # ENV
    (-0.10, -0.30, -0.15, +1.00, +0.10, +0.00, -0.15, +0.10),  # PRC
    (+0.30, -0.20, -0.30, +0.10, +1.00, +0.10, -0.35, +0.40),  # TRD
    (+0.15, +0.15, +0.10, +0.00, +0.10, +1.00, +0.10, +0.20),  # SOC
    (+0.15, +0.40, +0.05, -0.15, -0.35, +0.10, +1.00, -0.20),  # TEC
    (+0.10, -0.15, +0.00, +0.10, +0.40, +0.20, -0.20, +1.00),  # LOC
)

#: Draws outside this range are clipped (about 4.6% per dimension; accepted in
#: the pre-registration).
THETA_CLIP = 2.0

#: Wobble is sampled uniformly from this range. It is a Stage 1 tuning knob --
#: the test-retest band it has to land in is frozen, this range is not -- so it
#: is a default argument, never a constant baked into the draw.
DEFAULT_WOBBLE_RANGE = (0.10, 0.45)

AGE_RANGE = (18, 79)

CITY_SIZES: tuple[str, ...] = ("village", "small town", "mid-size city", "big city")

HOUSEHOLDS: tuple[str, ...] = (
    "alone",
    "with partner",
    "with partner and kids",
    "with kids, no partner",
    "with parents",
    "shared flat",
)

REGION_TYPES: tuple[str, ...] = ("rural", "suburban", "urban")

#: Region type is drawn conditional on city size, so we never mint a persona
#: living in an "urban village". Rows are in CITY_SIZES order, columns in
#: REGION_TYPES order.
_REGION_GIVEN_CITY_SIZE: dict[str, tuple[float, float, float]] = {
    "village": (0.80, 0.20, 0.00),
    "small town": (0.35, 0.50, 0.15),
    "mid-size city": (0.05, 0.55, 0.40),
    "big city": (0.00, 0.30, 0.70),
}

#: A wide spread of work, on purpose: mode collapse in the persona cards is the
#: first risk in the PRD, and occupation is the strongest diversity lever we
#: control before the card writer ever runs.
OCCUPATIONS: tuple[str, ...] = (
    # trades and manual work
    "electrician",
    "plumber",
    "carpenter",
    "welder",
    "bricklayer",
    "roofer",
    "painter and decorator",
    "car mechanic",
    "heating engineer",
    "crane operator",
    "forklift driver",
    "warehouse picker",
    "long-distance lorry driver",
    "bus driver",
    "taxi driver",
    "train conductor",
    "factory line worker",
    "machine operator",
    "dairy farmer",
    "seasonal farm worker",
    "fisher",
    "gardener and landscaper",
    "waste collection worker",
    # care, health, schools
    "hospital nurse",
    "care assistant in an elderly home",
    "paramedic",
    "midwife",
    "physiotherapist",
    "pharmacist",
    "dental hygienist",
    "social worker",
    "childminder",
    "preschool teacher",
    "primary school teacher",
    "secondary school maths teacher",
    "school administrator",
    "special needs assistant",
    # shops, food, hospitality
    "supermarket cashier",
    "butcher",
    "baker",
    "chef in a small restaurant",
    "waiter",
    "barista",
    "hairdresser",
    "hotel receptionist",
    "cleaner",
    "security guard",
    "small shop owner",
    # offices and professions
    "bookkeeper",
    "accountant",
    "bank branch adviser",
    "insurance claims handler",
    "HR officer",
    "office administrator",
    "sales representative",
    "marketing coordinator",
    "estate agent",
    "lawyer",
    "paralegal",
    "architect",
    "civil engineer",
    "local newspaper journalist",
    "graphic designer",
    "freelance translator",
    "librarian",
    "freelance photographer",
    # public sector and uniforms
    "police officer",
    "firefighter",
    "postal worker",
    "municipal planner",
    "tax office clerk",
    "public health inspector",
    "army logistics sergeant",
    "prison officer",
    # tech
    "software developer",
    "IT support technician",
    "data analyst",
    "network engineer",
    "software tester",
    "cybersecurity analyst",
    # not in paid work
    "university student",
    "vocational college student",
    "apprentice electrician",
    "unemployed, looking for work",
    "stay-at-home parent",
    "retired",
    "retired on disability pension",
)

#: A handful of occupations only make sense in part of the age range, as
#: (min_age, max_age). Everything else is open to all ages.
_OCCUPATION_AGE_LIMITS: dict[str, tuple[int, int]] = {
    "university student": (18, 32),
    "vocational college student": (18, 30),
    "apprentice electrician": (18, 30),
    "retired": (60, 79),
    "retired on disability pension": (45, 79),
    "army logistics sergeant": (20, 60),
    "firefighter": (20, 62),
    "police officer": (21, 63),
    "midwife": (24, 66),
    "lawyer": (26, 75),
    "architect": (26, 75),
}

#: Same idea for who you live with.
_HOUSEHOLD_AGE_LIMITS: dict[str, tuple[int, int]] = {
    "with parents": (18, 38),
    "with partner and kids": (22, 79),
    "with kids, no partner": (22, 79),
    "shared flat": (18, 45),
}


@dataclass(frozen=True)
class Persona:
    """One sampled person: planted truth plus the public demographic slice."""

    pid: str
    theta: dict[str, float]
    wobble: float
    age: int
    occupation: str
    city_size: str
    household: str
    region_type: str

    #: The only fields anything on the system side of the Wall may see.
    PUBLIC_FIELDS = ("pid", "age", "occupation", "city_size", "household", "region_type")

    def public_profile(self) -> dict[str, object]:
        """The demographic slice that is allowed outside the Wall."""
        return {name: getattr(self, name) for name in self.PUBLIC_FIELDS}

    def as_dict(self) -> dict[str, object]:
        """Everything, planted truth included. Never write this to a public file."""
        return asdict(self)


def correlation_matrix() -> np.ndarray:
    """The frozen population correlation matrix as an array."""
    return np.array(CORRELATION_MATRIX, dtype=float)


def _cholesky_factor() -> np.ndarray:
    """Lower-triangular L with L @ L.T = Sigma; fails loudly if Sigma is not PD."""
    try:
        return np.linalg.cholesky(correlation_matrix())
    except np.linalg.LinAlgError as exc:  # pragma: no cover -- frozen matrix is PD
        raise ValueError(
            "the frozen correlation matrix is not positive definite; it cannot "
            "be a covariance and the sampler will not run"
        ) from exc


def _eligible(options: tuple[str, ...], age: int, limits: dict[str, tuple[int, int]]) -> list[str]:
    """Options that make sense at this age, in the original order."""
    keep = [
        option
        for option in options
        if limits.get(option, (0, 200))[0] <= age <= limits.get(option, (0, 200))[1]
    ]
    return keep or list(options)


def _pick(rng: np.random.Generator, options: list[str] | tuple[str, ...]) -> str:
    """One uniform pick, consuming exactly one draw from the stream."""
    return str(options[int(rng.integers(len(options)))])


def _pick_region(rng: np.random.Generator, city_size: str) -> str:
    """Region type, conditional on city size. Consumes exactly one draw."""
    weights = _REGION_GIVEN_CITY_SIZE[city_size]
    roll = float(rng.random())
    running = 0.0
    for region, weight in zip(REGION_TYPES, weights):
        running += weight
        if roll < running:
            return region
    return REGION_TYPES[-1]


def sample_population(
    n: int,
    seed: int,
    wobble_range: tuple[float, float] = DEFAULT_WOBBLE_RANGE,
) -> list[Persona]:
    """Draw ``n`` personas from the frozen population, reproducibly.

    theta comes from a multivariate normal with mean 0, unit variances and the
    frozen correlation matrix, clipped to [-2, 2]. Demographics and the wobble
    level come from the same random stream, one persona at a time, so the first
    ``k`` people of a batch never change when ``n`` grows.

    ``wobble_range`` is the Stage 1 tuning knob; pass it explicitly in an
    experiment config rather than editing the default.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")

    low, high = float(wobble_range[0]), float(wobble_range[1])
    if not 0.0 <= low < high <= 1.0:
        raise ValueError(
            f"wobble_range must be 0 <= low < high <= 1, got {wobble_range}"
        )

    rng = np.random.default_rng(seed)
    chol = _cholesky_factor()
    age_low, age_high = AGE_RANGE

    people: list[Persona] = []
    for index in range(1, n + 1):
        raw = chol @ rng.standard_normal(len(DIMENSIONS))
        clipped = np.clip(raw, -THETA_CLIP, THETA_CLIP)
        theta = {dim: round(float(value), 4) for dim, value in zip(DIMENSIONS, clipped)}

        age = int(rng.integers(age_low, age_high + 1))
        occupation = _pick(rng, _eligible(OCCUPATIONS, age, _OCCUPATION_AGE_LIMITS))
        city_size = _pick(rng, CITY_SIZES)
        household = _pick(rng, _eligible(HOUSEHOLDS, age, _HOUSEHOLD_AGE_LIMITS))
        region_type = _pick_region(rng, city_size)
        wobble = round(float(rng.uniform(low, high)), 2)

        people.append(
            Persona(
                pid=persona_id(index),
                theta=theta,
                wobble=wobble,
                age=age,
                occupation=occupation,
                city_size=city_size,
                household=household,
                region_type=region_type,
            )
        )
    return people


def persona_id(index: int) -> str:
    """Persona id for a 1-based position in the batch: 1 -> ``p0001``."""
    return f"p{index:04d}"


def theta_matrix(people: list[Persona]) -> np.ndarray:
    """Stack the sampled trait vectors into an (n, 8) array, in frozen order."""
    return np.array([[person.theta[dim] for dim in DIMENSIONS] for person in people])
