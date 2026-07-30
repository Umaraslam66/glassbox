# PREREGISTRATION — Project GLASSBOX

> **STATUS: DRAFT — NOT FROZEN. Awaiting owner sign-off at Gate 0.**
> Once the owner signs off, this document is frozen. After that, nothing in it moves.
> Any measurement done differently from what is written here is labeled **exploratory**
> in every report where it appears. Deviations require stopping and asking the owner first.

Sign-off: ______________________  Date frozen: ______________

---

## 1. What this study claims and does not claim

We test one claim: **the pipeline provably recovers planted ground truth when ground
truth is known.** We generate synthetic people whose true trait values we planted
ourselves, hide those values behind a wall, and measure whether the system recovers
them from interview text alone.

This is a parameter recovery study. It does **not** claim the system predicts real
humans. That claim would require a human panel and is out of scope; the closing
deliverable includes a protocol for that study, not the study itself.

**Stage 1 is development and tuning only.** No claims are made from Stage 1 beyond
its sanity gate. All confirmatory results come from Stages 2–5 on held-out personas
and held-out items, measured against the bars in §6.

## 2. Planted truth: traits, population, noise

**PENDING OWNER CONFIRMATION (Stage 0 decision):** the 8 trait dimensions and the
population correlation matrix below are the orchestrator's proposal, not yet approved.

Each persona has a hidden trait vector θ with 8 dimensions, each on roughly −2 to +2:

| # | Code | Dimension | −2 looks like | +2 looks like |
|---|------|-----------|----------------|----------------|
| 1 | TRU | Institution trust | assumes officials lie; avoids banks, doctors, forms | defends public services; follows official advice |
| 2 | RSK | Risk appetite | savings account only; insures everything | invests, gambles, quits jobs, moves abroad |
| 3 | ENV | Environmentalism | annoyed by green rules; won't pay extra | organizes recycling; pays green premiums |
| 4 | PRC | Price sensitivity | buys quality without checking price | compares unit prices; waits for discounts |
| 5 | TRD | Traditionalism | breaks conventions; distrusts "how it's done" | keeps customs, family roles, established ways |
| 6 | SOC | Sociability | keeps to self; small circle | joins clubs; knows the neighbors; hosts |
| 7 | TEC | Tech optimism | new tech is a scam or a threat | early adopter; tech solves problems |
| 8 | LOC | Locality attachment | would move anywhere; no hometown pull | rooted; local news, local pride, won't move |

**Population distribution:** θ ~ multivariate normal, mean 0, unit variances,
correlation matrix Σ below (verified positive definite; smallest conditional SD 0.83,
so no dimension is near-redundant). Draws are clipped to [−2, 2] (~4.6% of draws
per dimension are clipped; accepted design choice).

```
       TRU    RSK    ENV    PRC    TRD    SOC    TEC    LOC
TRU  +1.00  +0.00  +0.10  -0.10  +0.30  +0.15  +0.15  +0.10
RSK  +0.00  +1.00  +0.00  -0.30  -0.20  +0.15  +0.40  -0.15
ENV  +0.10  +0.00  +1.00  -0.15  -0.30  +0.10  +0.05  +0.00
PRC  -0.10  -0.30  -0.15  +1.00  +0.10  +0.00  -0.15  +0.10
TRD  +0.30  -0.20  -0.30  +0.10  +1.00  +0.10  -0.35  +0.40
SOC  +0.15  +0.15  +0.10  +0.00  +0.10  +1.00  +0.10  +0.20
TEC  +0.15  +0.40  +0.05  -0.15  -0.35  +0.10  +1.00  -0.20
LOC  +0.10  -0.15  +0.00  +0.10  +0.40  +0.20  -0.20  +1.00
```

Design rationale (fixed with the matrix): all |r| ≤ 0.40 so every dimension stays
statistically distinguishable; TRU–RSK and ENV–RSK are deliberately zero (recovery
must work without correlation crutches); TRD–LOC–TEC–RSK form a moderately
correlated block (the hard case: separating related traits). The correlation matrix
itself is planted truth and is a recovery target in Stage 2's dimensionality check.

**Noise:** each persona gets a per-persona wobble parameter (how often it flips or
shifts answers on items it is lukewarm about), sampled independently of θ. The
test–retest **band in §6 (70–90%) is frozen; the noise-sampling range is a Stage 1
tuning knob** and is recorded in the experiment config, not here.

**Population size:** 300–500 personas. Persona cards are written by Gemma 4 31B
conditioned on θ; cards show traits as life details, never as labels or numbers.

## 3. The Wall (leakage rules — violations void the study)

1. `data/truth/` (θ, full cards, noise params) is readable only by `src/eval/`.
   Enforced by a static import test (`tests/test_wall.py`) run on every commit.
2. The interviewer, person encoder, item encoder, and predictor see only: public
   profile, transcript, question text, and closed-form answers actually given in
   that run.
3. Model-family split: personas run on Gemma 4 31B; the interviewer/encoder LLM is
   a different model family. Never the same weights on both sides.
4. Persona cards and θ are never pasted into system-side prompts, logged into
   system-side artifacts, or committed.
5. When any result beats expectations, the first action is a documented leakage
   hunt (prompts, logs, splits, distractor strength), recorded in PROJECT_LOG.md.

## 4. Mandatory baselines

Every evaluation reports lift over baseline; raw accuracy alone is never reported.

1. **Population marginal** (zero-information): predict the population answer
   distribution for every item. This is the primary reference everywhere.
2. **Public-profile-only** predictor (demographics, no interview).
3. **k-NN** over answered items (nearest personas by answer overlap).
4. **LLM given public profile but no interview.**

## 5. Fixed analysis choices

- **Trait recovery:** Pearson r per dimension between θ̂ and planted θ on held-out
  personas, after orthogonal Procrustes alignment of the latent axes (latent spaces
  are rotation-free; alignment is fit on the training split and applied to held-out).
- **Item recovery:** Pearson r between fitted and designed loadings on held-out items.
- **Prediction quality:** Brier score and log loss against the persona's true answer
  distribution, estimated from 50 samples per item at fixed temperature.
- **Calibration:** ECE (10 equal-width bins) and coverage of ±1σ credible intervals
  (nominal 68%, accepted band 60–75%).
- **Diversity clustering (Gate 1):** k-means on personas' standardized full-bank
  answer vectors; k chosen by mean silhouette over k = 10…60; the chosen k is the
  "number of behavioral clusters".
- **Test–retest (Gate 1):** re-ask 30 fixed items per persona in a fresh session;
  agreement = exact match on binary, within ±1 on Likert-5.
- **Predictor:** ordinal logit (graded response form) combining θ̂ with item loadings.
  No LLM anywhere in the predictor.
- **Probe stratification:** held-out probe items stratified by semantic distance from
  the interview set (near / same-domain / far), fixed before Stage 4 runs.
- **Interview length for confirmatory results:** N = 15 scripted closed-form
  questions + 3 open-ended answers, unless a bar below states otherwise.

## 6. Frozen bars per gate

### Gate 1 — population QA (sanity gate; development stage)
- **Diversity:** no persona pair with answer agreement > 95% on the full bank;
  silhouette-chosen k ≥ 30 behavioral clusters.
- **Trait obedience:** median per-dimension correlation (point-biserial/polyserial)
  between planted θ and own bank answers ≥ 0.5.
- **Realistic inconsistency:** population test–retest agreement in the 70–90% band.

### Gate 2 — static recovery (full answer matrix, no interviews)
- **Trait recovery:** per-dimension r(θ̂_fit, θ_true) ≥ 0.8 on held-out personas,
  all 8 dimensions, after Procrustes alignment.
- **Item recovery:** fitted vs designed loadings r ≥ 0.7 on held-out items; designed
  weak items must show low fitted discrimination (ordering sanity check).
- **Blur honesty:** ±1σ intervals cover true θ 60–75% of the time.
- **Dimensionality check (reported, no bar):** held-out likelihood vs number of
  latent dimensions; is the planted 8 recoverable?

### Gate 3 — translators (generalization beyond the fitted set)
- **Item encoder** (held-out items, zero-shot): median loading-vector cosine
  similarity ≥ 0.7; predicted vs fitted discrimination r ≥ 0.6.
- **Person encoder** (at N = 15): per-dimension RMSE(θ̂, θ_true) ≤ 0.5; blur coverage
  60–75%; blur shrinks monotonically in expectation as N grows (plot required).
- **Lift bar:** person-encoder RMSE beats the public-profile-only baseline by ≥ 30%.
  If not, stop and investigate before proceeding.

### Gate 4 — end-to-end prediction + calibration (headline result)
- **Primary lift bar:** Brier score beats the population-marginal baseline by ≥ 25%
  relative, and beats k-NN. If k-NN is close, that is reported prominently.
- **Calibration bars:** ECE ≤ 0.05; correlation between predicted probability and
  empirical 50-sample frequency ≥ 0.9 across all persona × item cells.
- **Generalization curve (reported, no bar):** performance and calibration per probe
  stratum (near / same-domain / far).
- **Aggregate check (reported):** predicted vs true population marginals and 2-way
  cross-tabs, with error bars.

### Gate 5 — adaptive interviewing + RL (efficiency result)
- **(a)** An adaptive interviewer (RL or info-gain heuristic) reaches the Gate-3 blur
  target in ≤ 50% of the questions random ordering needs.
- **(b)** RL vs heuristic on questions-to-target: reported either way; heuristic
  winning is a legitimate finding, not a failure.

## 7. Costs

Every experiment logs compute (Leonardo core-hours) and API cost in
`results/COSTS.md` at the time it runs, not retroactively.

## 8. Models

- Personas (card writing + responding): Gemma 4 31B on Leonardo.
- System side (interviewer, encoders): **decision pending at Gate 0** — Gemini
  flash-lite via API or an approved open-source dense model of a different family
  on Leonardo. Whichever is chosen, the Wall's family-split rule (§3.3) holds.
- Predictor and MIRT fitting: pure Python statistical code, no LLM.
- Anthropic models are never used for simulating humans or grading.
