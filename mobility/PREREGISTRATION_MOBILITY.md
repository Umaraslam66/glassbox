# PREREGISTRATION — GLASSBOX-Mobility

> **STATUS: DRAFT — NOT FROZEN.** Nothing here is binding yet. This document
> becomes the frozen contract only when the owner signs off at Gate M0, with
> whatever amendments the owner orders. Until then every number below is a
> proposal. After the freeze, anything measured differently from what is
> written here is labeled **exploratory** in every report where it appears,
> and deviations require stopping and asking the owner first.

Sign-off: PENDING (Gate M0). Drafted 2026-08-03.

---

## 1. What this study claims and does not claim

We test one claim: **standard travel-demand estimation recovers planted
traveler parameters from the choices of LLM travel agents — or it measurably
does not, and we say where it breaks.** We generate synthetic travelers whose
true behavioral parameters we planted ourselves, hide those values behind a
Wall, have an LLM render each traveler and make travel choices in character,
and grade recovery, calibration, and (reported, not bar-graded) elasticity
realism against published empirical ranges.

This is a parameter recovery study. It does **not** claim anything about real
travelers, and its elasticity comparisons are partly by construction — they
depend on the planted parameter distributions — which every report states
plainly.

**Stage M1 is development and tuning only.** No claims are made from it beyond
its sanity gate. All confirmatory results come from Stages M2–M4 on held-out
travelers and held-out scenarios, measured against the bars in §6.

## 2. Planted truth: traveler parameters, population, noise

**PENDING OWNER CONFIRMATION (Gate M0 decision):** the 6 dimensions and the
correlation matrix below are the orchestrator's proposal, not yet approved.

Each traveler has a hidden parameter vector φ with 6 dimensions, each on
roughly −2 to +2 (clipped at ±2; ~4.6% of draws per dimension clip — same
accepted design choice as the parent):

| # | Code | Parameter | −2 looks like | +2 looks like | Structural anchor |
|---|------|-----------|----------------|----------------|-------------------|
| 1 | VOT | Value of time | time-rich, money-tight; happily waits | pays to skip any delay | VOT_i = 12·exp(0.55·φ_VOT) €/h → median €12/h, range ≈ €4–36/h |
| 2 | SCH | Schedule rigidity | flexible; lateness shrugged off | hard start times; early buffers | late-arrival penalty ratio (β_late/β_early), 1–8× |
| 3 | MOD | Car vs transit/active affinity | car feels like a burden | car is identity | mode-specific constant, ±1.5 utility units on car |
| 4 | CRW | Crowding & comfort sensitivity | indifferent to packed vehicles | avoids crowding at real cost | crowding multiplier on in-vehicle time, 1.0–2.2× at worst level |
| 5 | PRC | Price sensitivity | barely checks fares | restructures trips over small fees | cost-coefficient scaling, ×0.4 to ×2.5 |
| 6 | HAB | Habit strength | experiments constantly | same route for years | switching penalty in utility units (bites mostly in Stage M4) |

**Population distribution:** φ ~ multivariate normal, mean 0, unit variances,
correlation matrix Σ below. Verified positive definite (smallest eigenvalue
0.372); smallest conditional SD 0.844, so no dimension is near-redundant;
all |r| ≤ 0.35.

```
       VOT    SCH    MOD    CRW    PRC    HAB
VOT  +1.00  +0.25  +0.20  +0.30   0.00  -0.10
SCH  +0.25  +1.00  +0.10  +0.10  -0.15  +0.35
MOD  +0.20  +0.10  +1.00  +0.25  -0.20  +0.30
CRW  +0.30  +0.10  +0.25  +1.00   0.00  +0.10
PRC   0.00  -0.15  -0.20   0.00  +1.00  +0.05
HAB  -0.10  +0.35  +0.30  +0.10  +0.05  +1.00
```

Design rationale (fixed with the matrix):

- **VOT–PRC is deliberately zero.** These two are partially redundant by
  construction inside any choice model (both live in the cost/time trade-off).
  Planting them uncorrelated makes their separation a real recovery target:
  the estimator must tell them apart from scenario structure, not lean on a
  population correlation. This is the mobility version of the parent's
  TRU–RSK lesson, and it carries a bar in §6 (Gate M2).
- **CRW–PRC is also zero** — comfort sensitivity must not be recoverable
  through price behavior.
- **SCH–MOD–HAB form the moderately tangled block** (0.35, 0.30, 0.10):
  rigid-schedule, car-identity, high-habit travelers overlap, as they
  plausibly do — the hard case for separation.
- VOT–CRW at +0.30: people who pay for time also pay for comfort.

**Traveler cards:** a lifestyle biography (job, household, commute situation,
constraints) written by the traveler-side model conditioned on φ, showing
parameters through life details, never naming them. Two-pass writing with a
self-check pass from day one — the parent measured a ~0.83 card-transmission
ceiling, and every recovery bar below is set knowing that ceiling exists.
A public profile (age band, occupation type, household, area type) is the only
card content the system side may see. In this population the public profile is
designed to carry no parameter signal (parent rule); real demographics do
carry signal, which is stated as a limitation wherever profile baselines
appear.

**Noise:** each traveler gets a per-traveler wobble parameter, planted
independently of φ. The test–retest band in §6 (70–90% pooled agreement) is
frozen; the mechanism producing it is a Gate M1 decision. The M0 availability
pilot determines the mechanism class (§8): the parent's mechanism consumed
recorded answer-token distributions from local vLLM, which API providers may
not expose usably. Whatever the mechanism, its parameters are frozen at
Gate M1 exactly as the parent froze its noise layer.

**Population size:** 400 travelers, minted from a recorded seed,
re-creatable and extendable at any time.

## 3. The scenario bank (~140 scenarios; design pending owner sign-off)

| Block | n | Choice set | Attributes varied | Primary loadings | Identification design |
|---|---|---|---|---|---|
| Mode choice | 50 | car / transit / bike-or-walk | door-to-door times (deltas 5–35 min), costs (€0–12), crowding (4 labeled levels), weather | VOT, MOD, CRW, PRC | a sub-block varies crowding at equal cost and time (isolates CRW); a sub-block varies cost at equal times (PRC vs VOT separation); attribute deltas span wide ranges so per-traveler coefficients are estimable |
| Departure time | 30 | leave earlier with slack vs later with lateness risk | schedule slack (5–40 min), explicit lateness probabilities (5–40%), no cost differences | SCH primary, VOT secondary | zero cost variation keeps PRC out; explicit probability framing makes the lateness gamble numeric |
| Route / reliability | 25 | faster-but-variable vs slower-but-certain | mean times, stated variability ranges, equal costs | VOT, SCH | mean–variance trade with equal prices — reliability preference identified free of PRC |
| Policy response | 25 | same trip before/after a change | congestion charge (€2–6), fare change (±20–30%), parking fee, employer transit subsidy | PRC primary, MOD, VOT | within-traveler before/after pairs; these double as the static elasticity probes for Stage M4 cross-checks |
| Distractors | 10 | lifestyle choices | none load on φ by design | none | leakage tripwire: any systematic φ̂ correlation with distractor choices triggers a hunt |

A handful (~8) of open-ended prompts ride along for realism QA only; they feed
no estimator.

Bank discipline inherited from the parent: a **vehicle map** (each behavioral
vehicle belongs to exactly one dimension), banned-context lists per dimension,
a correlation-bias budget, and an independent verification audit of the full
bank **before any traveler answers it**. Designed loadings and attribute
matrices are planted truth behind the Wall. HAB is identified statically by a
small set of switch-vs-stick framings inside the mode and route blocks; its
strong test is dynamic (Stage M4), and Gate M2's per-dimension bar knowingly
includes it anyway.

## 4. The Wall, mobility edition (violations void the study)

1. The mobility planted-truth directory (φ, full cards, wobble parameters,
   designed loadings, attribute-loading map) is readable only by
   `mobility/src/eval/` — enforced by `mobility/tests/test_wall_mobility.py`
   (static import + path checks) plus the parent Wall test, which additionally
   bans the rest of the repository from spelling the truth path or touching
   answer-token distributions. Both tests run on every commit.
2. The system side (short-survey estimator, any encoder, the choice predictor)
   sees only: public profile, scenario text + attributes as shown to the
   traveler, and the choices actually made.
3. **Model-family split:** traveler side and system side never share a model
   family (§8). Recorded at the freeze; forced changes later are stop-and-ask
   events with logged amendments, and confirmatory results stay
   whole-stage-coherent on one assignment or are re-run.
4. Field-level allowlists on system-side artifacts (the parent's post-breach
   lesson) are added to the mobility wall test at Stage M1 when the first
   artifact schemas exist — before any system-side code reads them.
5. Surprising results trigger a documented leakage hunt before celebration,
   recorded in PROJECT_LOG_MOBILITY.md.

## 5. Mandatory baselines and fixed analysis choices

Every evaluation reports **lift over the zero-information baseline**; raw
accuracy alone is never reported.

Baselines:
1. **Population marginal** (zero-information): the population choice shares
   for each scenario. Primary reference everywhere.
2. **Public-profile-only** predictor (profile, no choices).
3. **k-NN** over observed choices (nearest travelers by choice overlap).
4. **LLM given public profile but no choices** (the parent's finding 5
   replicated in the travel domain).

Fixed analysis choices:

- **Structural recovery (M2):** per-traveler conditional logit (McFadden MNL)
  over the training scenarios, one utility function fixed before fitting:
  V = β_time·time + β_cost·cost + β_crw·(crowding × in-vehicle time)
  + ASC_car + ASC_transit + β_late·P(late)·delay + β_switch·(non-habitual
  option). VOT_hat = β_time/β_cost in €/h. Alongside: one pooled latent-factor
  comparison fit (mixed logit with shrinkage); its latent axes are aligned by
  orthogonal Procrustes fitted on training travelers only.
- **Recovery metric:** Pearson r(φ̂_d, φ_d) per dimension on held-out
  travelers; VOT additionally in real units (median absolute percentage error
  against planted €/h).
- **Uncertainty:** per-traveler parametric bootstrap (resampling scenarios)
  for M2; posterior SD from the hierarchical estimator for M3. Coverage =
  share of travelers whose planted value lies inside ±1σ (nominal 68%,
  accepted band 60–75%).
- **Short-survey estimator (M3):** K = 12 scenarios + public profile;
  hierarchical shrinkage toward the population fit (empirical Bayes). No
  adaptive selection unless the owner approves it separately.
- **Prediction quality (M3):** Brier score on held-out scenarios' choices,
  against the traveler's true choice distribution estimated per §8's frozen
  noise mechanism (50 samples per scenario, parent convention).
- **Calibration:** ECE with 10 equal-width bins.
- **Fixed splits:** created once, before Stage M2, with a recorded seed: 20%
  of travelers and 20% of scenarios held out; the scenario split is
  stratified by block. The same held-out sets serve all confirmatory results
  in Stages M2–M4.
- **Test–retest (Gate M1):** 30 fixed scenarios re-asked per traveler in a
  fresh session; agreement = exact choice match; binary and multi-alternative
  scenarios reported separately beside their chance levels, pooled band is
  the bar.
- **Elasticities (M4):** arc elasticities on population aggregates for the
  three frozen experiments (uniform travel-cost +10%, transit fare −20%,
  cordon charge on car commutes), each run as the same population re-asked
  under changed attributes with everything else fixed.
- **Peak spreading metric (M4, fixed now):** a day's peak-15-minute share =
  the **maximum** share of that day's arrivals falling in any 15-minute
  window (sliding, 1-minute steps). Defined as a per-day maximum on purpose:
  if the whole peak merely shifts instead of flattening, a fixed window
  would show a spurious fall; the sliding maximum only falls when arrivals
  genuinely spread.

## 6. Bars per gate (DRAFT — frozen only at Gate M0 sign-off)

### Gate M1 — traveler population QA (sanity gate; development stage)
- **Diversity:** (a) no traveler pair with exact-match choice agreement > 95%
  on the full bank; (b) median pairwise agreement ≤ 0.80; (c) participation
  ratio of the travelers × scenarios standardized choice matrix ≥ 4.
- **Obedience:** for each dimension, correlation across travelers between
  planted φ_d and mean standardized choice tendency on the scenarios designed
  to load on d (sign-flipped where negatively keyed). Bar: median across the
  6 dimensions ≥ 0.5.
- **Realistic inconsistency:** pooled test–retest agreement in the 70–90%
  band, produced by the frozen noise mechanism (§8).
- Reasoning-token spend per answer is logged and reported at the gate
  (traveler side runs on a reasoning model; cost honesty).

### Gate M2 — structural recovery (the core claim)
- **Per-dimension recovery:** r(φ̂_d, φ_d) ≥ 0.75 on held-out travelers, all
  6 dimensions. (Set knowing the parent's ~0.83 card-transmission ceiling —
  before any data, which is what makes it legitimate rather than lax.)
- **VOT in real units:** median absolute error ≤ 25% against planted €/h.
- **Uncertainty coverage:** 60–75%.
- **Designed-zero separability:** the recovered VOT–PRC correlation satisfies
  |r(φ̂_VOT, φ̂_PRC) − 0| ≤ 0.15 on held-out travelers.
- **Distractor tripwire:** every distractor scenario's choices correlate with
  every φ̂ dimension at |r| ≤ 0.15; anything above triggers a documented
  leakage hunt before Gate M2 reports.

### Gate M3 — short-survey recovery + calibration
- **Lift:** ≥ 25% Brier improvement over the population-marginal baseline on
  held-out choice prediction, from K = 12 + profile; k-NN reported beside it.
- **Calibration:** ECE ≤ 0.05.
- **Coverage:** 60–75%, with the parent's coverage-decay phenomenon (error
  bars worsening as evidence grows) explicitly watched and reported — its
  reappearance is a finding, not a failure.

### Gate M4 — dynamic mini-world + elasticities
- **Peak spreading (bar):** the peak-15-minute arrival share (per-day
  sliding maximum, §5) falls by ≥ 20% relative from day 1 to day 20 under
  congestion feedback: S₂₀ ≤ 0.80 · S₁.
- **Behavioral coherence (bar):** across travelers, day-to-day switching
  frequency correlates negatively with planted HAB (Spearman r ≤ −0.3), and
  lateness-risk exposure taken correlates negatively with planted SCH
  (Spearman r ≤ −0.3).
- **Elasticities (REPORTED, no bar):** arc elasticities for the three frozen
  experiments beside published empirical ranges, always with the
  by-construction caveat: the population's planted VOT/PRC distributions
  partly determine these numbers; the interesting quantity is the distortion
  LLM mediation introduces relative to the structurally implied elasticities,
  computed and shown.

### Gate M5 — report
- No numeric bars. REPORT_MOBILITY.md in the parent's format: scoreboard
  first, fails first, verbatim frozen bars, limitations, costs. The GATSim
  adapter is out of scope unless the owner approves it before any work.

## 7. Costs

Every experiment logs API spend (and compute, once Leonardo returns) in
`mobility/results/COSTS_MOBILITY.md` at the time it runs, never
retroactively. Reasoning tokens are part of the logged spend.

## 8. Models (assignment recorded at freeze; PENDING pilot + owner sign-off)

- **Traveler side (cards + choices): OpenRouter `qwen/qwen3.7-flash`**
  (Plan B; Leonardo Booster verified unreachable during its Aug 3–17
  maintenance). Caching keyed by (traveler, scenario, round, seed),
  exponential backoff, and a resumable sweep driver are mandatory before the
  first large run — the M0 pilot measured upstream shared-pool 429s outside
  our control (numbers in `results/m0_pilot_summary.json`).
- **Traveler answering regime (Gate M0 Decision 3, approved):** cards are
  written with reasoning **on** (quality work, 400 calls); scenario
  answering runs with reasoning **off** (volume work: ~$0.008 vs ~$0.20 per
  1k answers, hours vs days of wall-clock, and a usable answer-token
  distribution — the M0 pilot measured the reasoning-on distribution as
  degenerate, probability ≈ 1.0 after hidden reasoning). This regime choice
  is verified by the frozen obedience comparison below before the full-bank
  sweep.
- **Obedience comparison, reasoning-on vs reasoning-off (frozen check, runs
  at the start of Stage M1, after the bank audit, before the full-bank
  sweep). Nothing in this check may be improvised after data exists:**
  - **Travelers:** 60 — the first 60 positions of a seeded shuffle
    (seed 2141) of the 400-traveler population.
  - **Scenarios:** a fixed 24-scenario subset drawn from the audited bank by
    a seeded stratified draw (seed 2142): 12 mode-choice (of which 4 from
    the CRW-isolating sub-block and 4 from the PRC-isolating sub-block),
    6 departure-time, 4 route/reliability, 2 policy. The drawn scenario IDs
    are recorded in the experiment config at draw time, before any traveler
    answers anything.
  - **Protocol:** both arms answer the identical traveler × scenario grid at
    the frozen temperature (0.7), one round, no noise layer applied — this
    measures the renderer, not the noise. Arm A = reasoning off; Arm B =
    reasoning on.
  - **Numeric rule (frozen):** per-dimension obedience in each arm = the
    correlation across the 60 travelers between planted φ_d and mean
    standardized choice tendency on the subset scenarios designed to load on
    dimension d (sign-flipped where negatively keyed). Reasoning-off is
    confirmed as the answering regime iff **median obedience across the 6
    dimensions in Arm A ≥ (median in Arm B − 0.10)** and **Arm A median
    ≥ 0.5**. Any other outcome is a stop-and-ask event: the owner picks the
    regime, and the noise mechanism follows the matching branch below.
  - **Pre-declared cost cap:** 60 × 24 × 2 = 2,880 answers, ≈ $0.65 (the
    reasoning-on arm dominates), logged in COSTS_MOBILITY.md at run time.
- **System side (any LLM role): Gemini 3.5 flash-lite** via Google AI Studio
  (~$0.076 per 1k answers measured; no rate-limit hits at concurrency 4).
  The confirmatory estimator itself is pure Python statistical code with no
  LLM; Gemini appears only in system-side auxiliary roles (e.g. the
  profile-only LLM baseline). Gemini exposes no answer-token distributions
  (API refuses the request), which is Wall-convenient: the system side could
  not read conviction signals even by accident.
- **Family split under every combination:** Qwen (traveler) vs Gemini
  (system) at start. If Leonardo returns mid-study and the traveler side
  moves to Gemma 4 31B (Plan A), that is a stop-and-ask amendment; stages
  stay coherent on one assignment or are re-run.
- **Noise mechanism (Gate M0 Decision 3: the parent's mechanism is ported;
  final parameters frozen at Gate M1):** seeded mechanical noise driven by
  the traveler's recorded answer-token distribution (available under the
  approved reasoning-off answering regime), tuned into the frozen band.
  Because the frozen parent Wall test bans a token-distribution keyword from
  every Python file in the repository, mobility code takes the API field
  names for this access from `mobility/config/api_fields.json` rather than
  spelling them in code (Gate M0 Ruling 4; no string-assembly of the banned
  word anywhere), with a comment at the read site and the rationale logged
  in PROJECT_LOG_MOBILITY.md. If the obedience comparison above forces the
  answering regime to reasoning-on, the fallback mechanism is seeded
  mechanical noise calibrated by repeated-sample disagreement (the M0 pilot
  confirmed choice-level disagreement exists under repeated sampling with
  reasoning on). Either way the 70–90% pooled test–retest band is the bar,
  and the mechanism never crosses the Wall: it runs traveler-side, and the
  system side sees only the noised choices.
- Anthropic models are never used for simulating travelers or for grading.

## 9. Provenance

Drafted at Stage M0 by the orchestrator, following the parent
PREREGISTRATION.md as the discipline template. Frozen only with the owner's
Gate M0 sign-off, with amendments applied and recorded exactly as the parent
did.
