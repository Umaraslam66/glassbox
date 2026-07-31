# HUMAN_PROTOCOL — the real-human study

**Status: pre-registration template. Freeze it before you recruit anyone.**

This is the study GLASSBOX could not run. GLASSBOX built synthetic people whose
true traits we planted ourselves, hid the traits behind a wall, and measured
whether the pipeline recovered them. That proves the machinery works when truth
is knowable. It does not prove the machinery predicts real people.

This document is the protocol for the study that would. It is written for a team
that has a panel and wants to start next week. Every number in it that came out
of GLASSBOX is labelled with where it came from, and every place where the
human study must differ from the synthetic one is called out.

Companion documents:

- `PREREGISTRATION.md` — the synthetic study's frozen design. This protocol
  copies its discipline: bars are frozen before data, deviations are labelled
  exploratory, results are reported whether they pass or fail.
- `results/REPORT.md` — the synthetic findings this protocol is calibrated on.
- `results/PROJECT_LOG.md` — the chronological record, including every failed
  round. It is not the source of truth for a number; the result files are.

---

## 0. How to read this

Section 1 says what is being tested and what replaces the wall. Section 2 is the
design on one page. Section 3 is the panel size and the power calculation.
Section 4 is the item bank, which is the part most likely to sink the study.
Sections 5–7 are fieldwork, test–retest and the adaptive arm. Section 8 lists
every pre-registered endpoint with a predicted number and an honest interval.
Section 9 is the analysis discipline. Section 10 is a smaller study you can run
this week on existing public data. Section 11 is what none of this can test.

Two things to freeze before fieldwork and never touch again: the item bank
(§4) and the endpoint table (§8). Everything else is engineering.

---

## 1. The claim, and what replaces planted truth

### 1.1 The claim

**Interview a person once with a short set of questions. Predict how that person
would answer questions they were never asked, better than predicting the
population average, with honest confidence.**

That is the whole claim. It is deliberately narrow. It is not "the model
simulates a person", not "the model replaces a survey", and not "the model
knows what someone believes". It is a prediction claim with a named baseline.

### 1.2 The wall-equivalent

In the synthetic study, the wall protected planted truth: an 8-number trait
vector we wrote ourselves, plus the persona's full biography. Only the grader
could read it. Every claim was graded against it.

Real people do not come with a trait vector. The replacement is:

| Synthetic study | Human study |
|---|---|
| Planted trait vector θ | **Does not exist.** No substitute. |
| Persona's true answer distribution (50 seeded samples per item) | **Held-out answers**: real answers to items the model never saw for that person |
| Designed item loadings | **Fitted item parameters** from a calibration half-sample |
| Wall = truth store readable only by `src/eval/` | **Held-out block**: probe answers are stored separately and are never available to the encoder, the interviewer, or the item-parameter fit for that person |

The operational wall for the human study is simple and must be enforced in code,
not by discipline:

1. **Person split.** Each participant's probe-block answers are written to a
   separate store. The interviewer and person encoder read only that
   participant's interview answers and demographics.
2. **Parameter split.** Item parameters used to score a participant are fitted
   on a calibration sample that excludes that participant (§9.2, cross-fitting).
3. **Time split.** The probe block is administered *after* the interview block
   in the same sitting, and the interview block is never re-scored using probe
   answers.
4. **A static test** that fails the build if any module outside the grader
   imports the probe store. Copy `tests/test_wall.py` from this repo; it does
   exactly this and has caught a real leak once already (a pre-noise answer
   field that survived onto the system side, PROJECT_LOG, Stage 2 entry).

### 1.3 What this replacement costs

Everything graded against *answers* survives the swap. Everything graded against
*latent truth* does not. Concretely:

- Prediction quality, calibration against observed answers, efficiency of
  adaptive questioning, test–retest — all measurable.
- Trait recovery (`r(θ̂, θ_true)`), item-loading recovery against designed
  loadings, and **blur honesty against latent truth** — not measurable, at all,
  ever, in a human study.

The last one matters more than it looks, and §11 is about it.

---

## 2. Design at a glance

| | |
|---|---|
| **Participants** | 1,500 completed first visits, nationally representative quotas |
| **Randomisation** | 1:1 to **scripted** interview order (750) or **adaptive** interview order (750) |
| **Session length** | ~25 minutes; 67 closed items + 3 open-ended answers + demographics |
| **Item bank** | 160 closed items (40 interview pool + 120 probe pool) + 12 open-ended prompts |
| **Per participant** | 15 interview items + 12 calibration items + 40 probe items (matrix-sampled), 3 open prompts |
| **Repeat visit** | 400 invited at 14–21 days, ~280 expected; 30 items re-asked from that person's own probe block |
| **Depth subsample** | 300 participants get a 30-question interview instead of 15 |
| **Primary endpoint** | Relative Brier lift of the interview model over the population marginal, on the probe block, scripted arm |
| **Primary test** | One-sided, α = 0.025, H₀: lift ≤ 3%; design effect size 9% |
| **Power at design point** | 0.96 |
| **Model** | Multidimensional graded-response IRT + item text encoder + person encoder. **No LLM in the predictor.** |

**Cost structure** (the shape is fixed; the rates are not): 1,500 main sessions
at ~25 minutes each, plus ~280 retest sessions at ~8 minutes, plus the 300-person
depth subsample at ~35 minutes, plus quota top-ups and a 15% screen-out
allowance. Field spend
**[placeholder — replace with the owner's real panel rates before circulation]**.
Compute is trivial next to the field spend either way: the whole GLASSBOX
synthetic study used 127.7 GPU core-hours end to end.

---

## 3. Panel size and the power calculation

This is the section that determines whether the study is worth funding, so it is
written out fully, including the parts that are unfavourable.

### 3.1 The metric, stated precisely

**Brier score, per cell.** For participant *p* and probe item *j* with `C_j`
answer categories, let `p̂` be the predicted distribution over categories and `y`
the participant's observed answer:

```
Brier(p, j) = (1 / C_j) · Σ_c ( p̂_c − 1[y = c] )²
```

The `1 / C_j` matters. **This is the Gate-4 metric definition carried over
unchanged** — `PREREGISTRATION.md` §5 and the `metric_definitions` block of
`results/stage4_gate4.json` define the Brier score as the mean over that item's
own answer categories, and keeping the divisor is what makes a human result
comparable to the synthetic one at all. Without it, mixing binary and 5-point
items reweights the score: on GLASSBOX's own held-out cells the per-category
form gives a 27.1% lift and the sum form gives 23.1% on identical predictions.
**Write the divisor into the pre-registration or the result is not comparable to
anything.**

**Relative lift.**

```
lift = 1 − mean(Brier_model) / mean(Brier_marginal)
```

where `Brier_marginal` uses the population answer distribution for that item,
estimated on the calibration sample only. This is the zero-information baseline
and it is mandatory in every reported number (PREREGISTRATION.md §4; PRD §2).

### 3.2 The scale problem, and why the synthetic 27.1% is not the number to power against

GLASSBOX graded against each persona's **true answer distribution**, estimated
from 50 seeded re-draws per cell. A real panel gives you **one answer per cell**.
Against a single answer, the Brier score of *every* model — good or bad — carries
an irreducible term equal to the person's own answer randomness, and that term
is the same for both the model and the baseline. It cancels in the difference
but it does not cancel in the ratio, so it inflates the denominator and shrinks
the measured lift.

Measured on GLASSBOX's own Stage-4 cells (`results/stage4_cells.npz`,
`results/stage4_predictions.npz`, `results/stage4_baselines.npz`), by drawing one
answer per cell from each persona's true distribution, 400 times:

| Scale | Marginal Brier | Model Brier | Lift |
|---|---|---|---|
| Distribution-referenced (what Gate 4 measured) | 0.0860 | 0.0627 | **27.1%** |
| Single-answer-referenced (what a panel can measure) | 0.1635 | 0.1402 | **14.3%** |

**Same predictions, same data, same metric — 27.1% becomes 14.3% purely because
the panel observes one answer instead of a distribution.** Anyone who powers a
human study against 27% is powering against a number the study cannot produce.

**Two endpoints, both pre-registered:**

- **E1 (primary, raw):** lift measured against single held-out answers.
  Reference value from GLASSBOX: **14.3%**.
- **E1c (secondary, noise-corrected):** lift after subtracting the irreducible
  answer-noise term, which makes it comparable to the frozen 27.1%.

The correction is exact and uses the test–retest arm. For a person-item with
true answer distribution *q*, the expected single-answer Brier of any prediction
*p̂* is

```
E_y[ Brier(p̂, y) ] = Brier(p̂, q) + (1 − Σ_c q_c²) / C_j
```

and `Σ_c q_c²` is exactly the probability that two independent answers from the
same person to the same item agree. That is precisely what an **exact-match**
test–retest measures. So

```
corrected_lift = mean(Brier_marg − Brier_model) / ( mean(Brier_marg) − noise_term )
noise_term     = mean over cells of (1 − A_j) / C_j ,   A_j = exact-match retest agreement on item j
```

Checked numerically on GLASSBOX's own cells: the correction recovers **0.2712**
against the measured **0.2711**. It is exact when retest agreement is an
unbiased estimate of `Σ q²`.

**It is not unbiased in a real retest, and that is why E1c is secondary.**
Memory of the first visit pushes agreement up (undercorrects); genuine attitude
change over two weeks pushes it down (overcorrects). Both are real and they
pull in opposite directions. Design mitigation: re-ask items **from that
person's own probe block** so the correction is estimated on exactly the cells
it is applied to; use a 14–21 day gap (long enough that verbatim recall of a
mid-survey item is weak, short enough that most attitudes have not moved);
report E1c with a sensitivity band from ±5 points of assumed retest bias.

### 3.3 Effect size: why 9%, honestly

Design effect size is **9% raw lift**, about 63% of GLASSBOX's own single-answer
figure of 14.3%. The reasoning, with the adjustment each factor buys:

**Pushing the human lift down.**

1. **Real people are messier than an 8-dimensional model.** Gate 4's attribution
   analysis (`results/stage4_corr_attribution.json`) found that 66.9% of the gap
   between the model and a truth-side oracle is the model class itself —
   item-level idiosyncrasy that no 8-number trait vector can express. That was
   measured on personas *generated from* an 8-number trait vector. Real people
   are not, so this share can only grow. **×0.75–0.85.**
2. **Real people are less decisive.** GLASSBOX's population marginal puts 49.7%
   of Likert mass on the two extreme categories and 14.7% on the midpoint (raw,
   before the noise layer, the responder was at 63% extreme / 6% midpoint). Real
   Likert data is far more central. Flatter, more overlapping person
   distributions mean the population marginal is already close to everyone, so
   there is less for a person-specific model to win. **×0.75–0.90.**
3. **Real item banks are less clean.** GLASSBOX's bank was rebuilt four times
   against a vehicle map and still shipped with known defects (§4.4). A
   first-pass human bank will be worse. **×0.85–0.95** — and §4 exists to keep
   this factor from being much worse than that.

**Pushing the human lift up.**

4. **No card-rendering ceiling.** The synthetic pipeline lost information twice:
   planting θ, then rendering θ into a biography that an LLM role-played. The
   biography transmitted only ~0.85 of the planted trait signal, and the full
   252-item answer matrix reached only RMSE 0.529 against planted truth
   (PROJECT_LOG, Gate 3 close). A real person has no rendering step — the person
   *is* the truth. **×1.0–1.2.**
5. **Real demographics carry signal.** In the synthetic population they carried
   essentially none: the profile-only baseline collapsed to an intercept
   (+0.1% lift) and an LLM given demographics alone did **worse than knowing
   nothing** (−34.8% lift). Real demographics predict attitudes weakly but not
   zero. Since the model also sees demographics, this **adds** to the primary
   lift. **×1.0–1.1.** It makes the *secondary* endpoint (beating a
   demographics-only model) harder, which is the point of E5.

**The one that does not move the number:** self-report noise. A human's answer
today reflects mood, question order and satisficing. That is the human analogue
of the persona noise layer — and GLASSBOX's noise layer was tuned to land
test–retest at 0.794 pooled, inside the human literature band of roughly
75–85%. The noise level is already approximately right, so no further discount.
If the real panel comes in *below* 70% agreement, the effect size assumption is
broken and §6.4 says what to do.

Multiplying through: 14.3% × (0.75–0.85) × (0.75–0.90) × (0.85–0.95) ×
(1.0–1.2) × (1.0–1.1) spans roughly **5.5%–15%**, centred near **9%**.
The study is powered at 9% and the sensitivity table (§3.6) shows what happens
across that whole range.

### 3.4 Variance structure, measured not assumed

The per-cell quantity being averaged is `d = Brier_marginal − Brier_model`. Its
variance was decomposed on GLASSBOX's own 100 × 50 held-out cells under
single-answer sampling, using crossed (person × item) ANOVA moment estimators:

| Component | Variance | Share |
|---|---|---|
| Between persons | 0.000220 | 1.8% |
| Between items | 0.000710 | 5.7% |
| Residual (cell) | 0.011572 | 92.6% |
| **Total** | **0.012502** | |

Person ICC = 0.018. With 40 probe items per person, the person-clustering design
effect is `1 + 39 × 0.018 = 1.69`.

**The item component is the one that decides the study.** The between-item
standard deviation of the gain is √0.000710 = 0.0266, larger than the mean gain
itself (0.0233). Items differ enormously in how much a person model helps. If
you want the result to generalise to *questions in general* — not just to your
particular 40 — the item variance must be averaged over a large **pool** of
distinct items, and no number of participants substitutes for it.

Under a matrix-sampled design with **J** distinct probe items in the pool and
**k** answered per person:

```
Var( mean d ) = σ²_person / N  +  σ²_item / J  +  σ²_resid / (N · k)
```

The middle term does not depend on N. That is the floor.

### 3.5 The test

- **Unit:** person × probe-item cell.
- **Statistic:** relative Brier lift over the population marginal.
- **Test form:** one-sided test of `H₀: lift ≤ 3%` against `H₁: lift > 3%`,
  implemented as a test that `E[ d − 0.03 · Brier_marginal ] > 0`. This is an
  exact linear contrast, so no ratio approximation is needed.
- **Why 3% and not 0%:** a lift of one or two percent is statistically real and
  commercially worthless. A model that needs a 25-minute interview to beat the
  population average by 2% should not be shipped. The bar encodes that.
- **Flagged for sign-off.** The 3% floor is a judgement call, not a measurement,
  and it is the one number here most worth arguing about. The owner may lower it
  to a nil null (`H₀: lift ≤ 0`) at pre-registration sign-off. The consequence is
  already priced in §3.6: at the recommended design the minimum detectable lift
  at 90% power moves from **8.3%** (against 3%) to **5.5%** (against nil). A nil
  null buys sensitivity and gives up the claim that the detected effect is worth
  anything commercially. Decide once, before fieldwork, and do not revisit.
- **α = 0.025 one-sided** (equivalent to 0.05 two-sided). **Power target 0.90.**
- **Variance:** crossed person × item, as above, inflated by **×1.15** as a
  standing conservatism allowance for real-panel heterogeneity that the
  synthetic population does not have (mode-of-completion effects, satisficers,
  quota clustering).
- **Estimation in practice:** report the model-based SE above *and* a
  two-way cluster bootstrap (resample persons and items independently, 2,000
  replicates). Pre-register the bootstrap as the confirmatory one and the
  closed form as the check. If they disagree by more than 15%, investigate
  before reporting anything.

### 3.6 The answer, and the sensitivity tables

**Recommended design: N = 1,500 first visits (750 per arm), probe pool J = 120,
k = 40 probe items per participant.** Primary endpoint computed on the scripted
arm (N = 750), which is the exact mirror of Gate 4.

At that design, against `H₀: lift ≤ 3%`:

- SE on the lift = **0.0162**
- **Minimum detectable lift at 90% power = 8.3%**
- **Power at the 9% design point = 0.96**
- Power at 8% = 0.87; at 7% = 0.69; at 6% = 0.46

**Sensitivity to panel size** (J = 120, k = 40, H₀ ≤ 3%):

| N (scripted arm) | SE(lift) | Min detectable lift @90% | Power at 9% |
|---|---|---|---|
| 250 | 0.0179 | 8.8% | 0.92 |
| 500 | 0.0166 | 8.4% | 0.95 |
| **750** | **0.0162** | **8.3%** | **0.96** |
| 1,000 | 0.0160 | 8.2% | 0.96 |
| 1,500 | 0.0158 | 8.1% | 0.97 |
| 3,000 | 0.0155 | 8.0% | 0.97 |

**Sensitivity to probe-pool size** (N = 750, k = 40, H₀ ≤ 3%):

| Probe pool J | SE(lift) | Min detectable lift @90% | Power at 9% |
|---|---|---|---|
| 40 | 0.0270 | 11.8% | 0.60 |
| 60 | 0.0223 | 10.2% | 0.77 |
| 80 | 0.0195 | 9.3% | 0.87 |
| **120** | **0.0162** | **8.3%** | **0.96** |
| 160 | 0.0143 | 7.6% | 0.99 |
| 240 | 0.0121 | 6.9% | 1.00 |

**Sensitivity to probe items per person** (N = 750, J = 120, H₀ ≤ 3%):

| k | SE(lift) | Min detectable lift @90% |
|---|---|---|
| 15 | 0.0170 | 8.5% |
| 25 | 0.0165 | 8.4% |
| 40 | 0.0162 | 8.3% |
| 60 | 0.0160 | 8.2% |

**Read those three tables together. This is the headline of the section:**
going from 250 to 3,000 participants moves the standard error by 13%. Going from
40 to 240 distinct probe items moves it by 55%. **Panel size is not the binding
constraint on this endpoint — the number of distinct questions is.** A study
that spends its budget on people and skimps on the item bank is buying the wrong
thing.

**If N is not driven by the primary test, what is it driven by?** All of these
at once:

1. **Item calibration.** Item parameters (loadings and thresholds) must be
   fitted from response data — GLASSBOX's Gate-4 finding is blunt: item
   thresholds predicted from question *text alone* collapse the lift to 2.7%
   (`results/stage4_gate4.json`, `model_text_thresholds` arm). They must be
   learned. With 5-fold cross-fitting, each calibration set holds 1,200
   participants; at k = 40 from a 120-item probe pool that is **400 answers per
   probe item**, and at 12 calibration items from a 40-item interview pool,
   **360 per interview-pool item**. GLASSBOX fitted 8 dimensions from 400
   personas × 202 items and got a median held-out per-item loading recovery of
   0.92 and split-half reliability 0.933, so 400 is a demonstrated-sufficient
   floor in our hands. Below ~300, expect unstable thresholds.
2. **Two randomised arms** (scripted vs adaptive), each needing the primary
   endpoint computable on its own.
3. **The retest subsample** (§6) and the **depth subsample** (§7.3).
4. **Pre-registered subgroups** (age band, education, region) at usable
   precision.
5. **Attrition and screen-outs.** Budget 15% loss between recruitment and a
   usable record.

**Bottom line: N = 1,500. The primary test needs 250. The other 1,250 buy the
item bank calibration, the second arm, the repeat visits and the subgroups.
Say so out loud in the pre-registration — it is honest and it protects the study
from being cut to 400 by someone who read only the power line.**

### 3.7 Assumptions, listed

Every number above rests on these. They are stated so a reviewer can attack them.

| # | Assumption | Where it came from | If wrong |
|---|---|---|---|
| A1 | Per-cell variance of the Brier difference resembles GLASSBOX's, ×1.15 | Measured on `stage4_cells.npz` | Re-estimate from the pilot (§5.4) before locking N |
| A2 | Person ICC ≈ 0.018, item ICC ≈ 0.057 | Same | Same |
| A3 | Population marginal Brier ≈ 0.164 on the single-answer scale | Same | Scales the effect linearly; re-estimate from the pilot |
| A4 | True human lift ≈ 9% raw | §3.3 reasoning chain | Sensitivity table covers 6–14% |
| A5 | Answers within a person are conditionally independent given the trait vector | Standard IRT; used by both the model and the marginal baseline | Both sides are affected; the *difference* is fairly robust, but the aggregate cross-tab check (§8, E10) will show it |
| A6 | Item pool is exchangeable enough to treat items as a random sample | §4 stratification | The item-random analysis becomes a fixed-item analysis and the result no longer generalises to new questions — report it that way |
| A7 | Retest agreement estimates within-person answer randomness | §3.2 identity | E1c becomes uninterpretable; E1 is unaffected |

---

## 4. Item bank design requirements

**This is the non-negotiable section.** GLASSBOX's bank took four QA rounds and
rewrote roughly 115 of 272 items before it was usable. Skipping this section is
the single most likely way for a human study to produce a number that means
nothing.

### 4.1 What went wrong in the synthetic study, in order

From `results/PROJECT_LOG.md`, entries "Bank QA: fix round 1 FAILED verification"
and "Question bank PLANTED after four QA rounds":

1. First audit flagged **95 of 272 items** for sign errors, wrong dimensions and
   near-duplicates.
2. Those fixes were applied, validated mechanically, and then **failed an
   independent verification pass**: 9 of the 95 fixes were still defective.
3. A 25-item control sample of *untouched* items came back **6/25 defective
   (≈24%, CI 9–45%)** — the first audit had a large false-negative rate. The
   bank was worse than anyone had measured.
4. **Systematic wording bias**: environmentalism items written in price
   language, risk items set in technology contexts, traditionalism items leaning
   on paper-versus-digital. The bank was manufacturing the very correlations the
   study existed to test for.
5. **Vehicle collision**: the same behavioural situation ("buy the cheapest X",
   "keep cash rather than a bank account") used as the measuring device for two
   different dimensions in different items, destroying separability.
6. Rebuild against a vehicle map → 16 blocking residuals → scoped fixes → 2 more
   → five micro-fixes → pass.

And even after all that, at fitting time: two items designed as weak fitted with
**flipped signs** (predicted pre-fit by the verifier), one "weak" item fitted at
3.74 discrimination ("labelled weak, not written weak"), and one item was
answered identically by 455 of 500 personas.

### 4.2 The vehicle-map rules — mandatory

A **vehicle** is the concrete behaviour or situation an item uses to measure
something: "waiting for a sale", "using a self-checkout", "attending a
neighbourhood meeting", "reading the terms before signing".

1. **One vehicle, one dimension.** Every vehicle in the bank is assigned to
   exactly one trait dimension, in a written map, before any item is drafted. No
   vehicle is ever used to measure a second dimension, in any item, in any
   wording. Enforce with a machine-readable `vehicle_id` on every item and a
   validator that fails on any vehicle appearing under two dimensions.
2. **Banned-context lists, per dimension.** For each dimension, write down the
   contexts its items may not use, derived from the correlations the study needs
   to be able to separate. Example set from GLASSBOX: environmentalism items may
   not mention price, cost, discount or saving money; risk items may not be set
   in a technology context; traditionalism items may not use paper-versus-digital.
   Enforce with a keyword validator, then a human read.
3. **Correlation-bias budget.** For every pair of dimensions, count the items
   whose wording could plausibly carry signal on both. Set a cap (GLASSBOX used
   "no more than 1 per pair") and enforce it. A bank that violates this budget
   will recover exactly the correlation structure its wording contains, and you
   will not be able to tell that apart from a real finding.
4. **Reverse-keyed items must use distinct vehicles.** A positively-keyed and a
   negatively-keyed item on the same dimension may not be the same situation
   with a negation. If they are, you have measured a wording style, not a trait.
5. **Deliberate weak items and off-dimension distractors.** Include both, tag
   them privately, and check after fitting that weak items fit low
   discrimination and distractors fit near zero. This is the ordering sanity
   check and it is how you find out your instrument is broken. Expect it to
   surprise you — see §4.4.
6. **Probe stratification frozen before any interview.** Split the probe pool by
   semantic distance from the interview pool into **near / same-domain / far**,
   with a recorded rule and a recorded seed, before a single participant is
   interviewed. GLASSBOX froze this at split time (`experiments/splits_v1.json`,
   seed 2026, near 13 / same-domain 19 / far 18) and the per-stratum curve turned
   out to be the most informative result in Gate 4. Suggested human proportions
   for a 120-item probe pool: near 30 / same-domain 45 / far 45.
7. **Independent adversarial verification, full bank, no sampling.** A fresh
   reviewer who did not write the items, checking every item, with a control
   sample of untouched items included so the reviewer's own false-negative rate
   is measured. GLASSBOX's first audit passed 24% defective items; the control
   sample is what caught it.

### 4.3 Bank size and structure for the human study

| Block | Items | Purpose |
|---|---|---|
| Interview pool | 40 closed | The scripted 15 are drawn from here; the adaptive arm selects from all 40 |
| Probe pool | 120 closed | Held-out prediction targets, stratified near/same-domain/far |
| Open-ended prompts | 12 | 3 per participant |
| **Total** | **160 closed + 12 open** | |

Per participant: 15 interview + 12 calibration (matrix-sampled from the
interview pool, so every interview-pool item gets calibrated) + 40 probe
(matrix-sampled from the probe pool) = **67 closed items**, plus 3 open prompts.

Use a balanced incomplete block design for the matrix sampling, so that every
item appears equally often and every item *pair* co-occurs for enough
participants to estimate the covariance structure.

**Dimensionality.** GLASSBOX used 8 trait dimensions with all pairwise
correlations capped at |r| ≤ 0.40. For a human study, do not assume 8. Pre-register
a dimensionality selection rule instead: fit d = 3…12 on the calibration folds,
select by held-out log-likelihood, report the whole curve. GLASSBOX's
split-half curve peaked at exactly the planted d = 8 in the second run and at
6–8 in the first — the selection works, but it is not sharp.

### 4.4 Two findings that will repeat, and are not defects

Write both into the pre-registration so nobody panics in the field.

**Distractors will fit real discrimination.** In GLASSBOX, 4 of 15
deliberately-neutral distractor items fitted material discrimination (median
distractor ‖a‖ 0.578 vs 1.998 for strong items; the worst was a
stairs-versus-lift preference loading 1.69 on tech optimism). Verified not to be
leakage: no planted value ever reached the system side. It is the synthetic
analogue of the "crud factor" — in real survey data, essentially everything
correlates with everything a little, because no person's "neutral" preferences
are truly independent of who they are. **Expect this to be larger in humans, not
smaller.** Report distractor discrimination as an instrument-behaviour finding,
not a failure.

**Some items will be dead.** GLASSBOX shipped with one item answered "no" by 455
of 500 personas and one answered "3" by all 500. Pre-register a drop rule: any
item where one category takes more than 90% of answers in the calibration
sample is dropped from the confirmatory analysis, the drop is reported, and the
analysis is re-run with and without it.

### 4.5 The open-ended prompts

Three per participant, drawn from a pool of 12, fixed by the split seed. Keep
expectations calibrated: in GLASSBOX, the three open answers moved the point
estimate substantially (correlation to the full-matrix estimate rose 0.686 →
0.770) but bought only about **3 closed items' worth of declared precision**.
Text is information-rich and precision-poor under this kind of fusion. Include
them; do not build the study's value on them.

---

## 5. Fieldwork

### 5.1 Session structure

1. Consent, screener, quota assignment.
2. **Demographics block** — age, gender, region, urbanicity, education, income
   band, household composition, employment. This is the input to the
   demographics-only baseline (E5) and it must be collected identically for
   everyone.
3. **Interview block** — 15 closed items (scripted fixed order, or adaptively
   selected, by randomisation) interleaved with 3 open-ended prompts.
4. **Calibration block** — 12 closed items from the interview pool, random order.
5. **Probe block** — 40 closed items from the probe pool, random order.
   *Administered last, always. Never scored back into the interview.*
6. Attention checks: 2 instructed-response items, positions randomised.

The order matters: probes last, so nothing a participant learns from a probe can
contaminate their interview answers.

### 5.2 Quality rules, pre-registered

Exclude a record if any of: failed both attention checks; completion time below
40% of the median; straight-lining (identical response to ≥ 90% of Likert items);
fewer than 60 of 67 closed items answered. Report the exclusion count and re-run
the primary endpoint on the unfiltered sample as a sensitivity analysis.

### 5.3 Mode

One mode for everyone. Mixing web and phone introduces a mode effect that will
show up as person-level variance the model cannot explain, and the study is not
powered to separate it.

### 5.4 Pilot first — this is not optional

Run **n = 120** before locking anything. The pilot's job:

1. Re-estimate the three variance components (§3.4) on real answers and redo the
   power calculation. Assumptions A1–A3 are the shakiest part of this document.
2. Check the marginal answer distributions. If the Likert midpoint is used by
   more than ~35% of respondents on most items, the population is less decisive
   than assumed and the effect size should be revised down.
3. Time the session.
4. Catch bank defects the verification pass missed.

The pilot is **not** part of the confirmatory sample and its participants are
excluded from the main study.

---

## 6. Test–retest

### 6.1 Design

- **400 participants invited back**, stratified across arms and quotas; expect
  ~280 to return (70%).
- **Gap 14–21 days.** Report the actual gap distribution.
- **30 items re-asked**, drawn from **that participant's own probe block**, so
  the noise correction in §3.2 is estimated on the same cells it corrects.
- Fresh session, no reference to previous answers, item order re-randomised.

### 6.2 Agreement definitions — both, always

- **Exact match** on binary and on Likert-5. This is the one that feeds the
  noise correction (§3.2); it must be exact, not within-±1.
- **Within ±1** on Likert-5, for comparability with the survey literature and
  with GLASSBOX's Gate-1 number.

Report both, each next to its chance level (binary exact ≈ 50%; Likert-5
within-±1 ≈ 52% under uniform answering; Likert-5 exact ≈ 20%).

### 6.3 Expected band

**Pooled agreement (binary exact + Likert within-±1) in the 70–90% band, with a
human-literature anchor of roughly 75–85%.**

GLASSBOX's synthetic population landed at **0.794 pooled** (binary exact 0.821,
Likert within-±1 0.781) — the band was frozen *before* the population was tuned
to hit it, and it took a rebuilt noise mechanism to get there. The raw,
un-noised responder sat at **0.98**, which is what a naive LLM persona does: it
answers like a robot. Any real panel that comes back above 0.90 has a survey
design problem (probably order or memory effects), not unusually consistent
respondents.

Precision: with 280 returners × 30 items and the person-level variance measured
in GLASSBOX (SD of per-person agreement 0.096), the standard error on pooled
agreement is **0.006**, i.e. ±1.1 points at 95%. Ample.

### 6.4 What failing the band means

| Result | Reading | Action |
|---|---|---|
| **Above 90%** | Memory or order effects; the two visits are not independent measurements | Lengthen the gap, re-randomise more aggressively, re-run on a fresh subsample. Do **not** report E1c — the correction would be badly biased downward |
| **70–90%** | Normal | Proceed |
| **60–70%** | Noisy panel or noisy items | Report E1 as pre-registered; flag that the effect-size assumption (§3.3 factor 3) was optimistic and the achievable lift is lower for everyone, model and baseline alike. Identify whether the noise is concentrated in specific items and report per-item retest |
| **Below 60%** | The panel or the instrument is not measuring stable attitudes | **The primary endpoint is not interpretable.** Report it anyway, with the retest number front and centre, and stop before drawing product conclusions |

Failing this band is a finding about the panel and the instrument, and it must
be reported in the abstract, not in an appendix. A model cannot predict answers
that the person themselves does not reproduce.

---

## 7. The adaptive-interviewing arm

### 7.1 Ship the heuristic

**The protocol specifies an information-gain heuristic, not a trained policy.**

This is a direct consequence of GLASSBOX's Gate-5 result. A REINFORCE-trained
policy and a hand-coded information-gain heuristic were compared on the same
held-out personas across 51 replicates
(`results/stage5_strategies.json`). On the pre-declared threshold grid:

| Accuracy threshold | Heuristic (median questions) | RL policy | Ratio |
|---|---|---|---|
| RMSE ≤ 0.70 | 24 | 25 | 1.04 |
| RMSE ≤ 0.65 | 40 | 37 | 0.93 |
| RMSE ≤ 0.60 | 77 | 84 | 1.09 |

**RL ties the heuristic.** It does not beat it. Both roughly halve random
ordering (heuristic 0.462 / 0.494 / 0.566 of random at the three thresholds).
A further finding: deployed greedily, the trained policy converged to a
near-**static** 25-question order — the value it learned was *which questions to
ask*, not *how to branch per person*. Given that, a heuristic that needs no
training data, no reward design and no policy checkpoint is the correct thing to
put in front of real participants.

**The heuristic.** At each step, pick the unasked item with the largest expected
reduction in total posterior variance (A-optimality) under the model's current
belief about that participant:

```
score(item i) = Σ_y P(y | θ̂) · [ trace(Σ) − trace( (Σ⁻¹ + c_{i,y} · a_i a_iᵀ)⁻¹ ) ]
```

with Σ the current posterior covariance, `a_i` the item's loading vector and
`c_{i,y}` the curvature answer *y* contributes. By Sherman–Morrison this reduces
to `c · ‖Σa‖² / (1 + c · aᵀΣa)` — a few multiplications per item, no matrix
inverse, fast enough to run live between questions. Implementation and tests are
in `src/interview/` in this repo.

### 7.2 The comparison

Participants are randomised 1:1:

- **Scripted arm (750).** The same 15 interview items, same order, for everyone.
  The order is frozen before fieldwork. This arm carries the primary endpoint.
- **Adaptive arm (750).** 15 items selected sequentially by the heuristic from
  the 40-item interview pool.

Randomisation is at the participant level, stratified by quota cell, with a
recorded seed.

**Why not a random-order arm?** GLASSBOX used random ordering as the reference
because it could simulate it 51 times per persona at no cost. With real people
you cannot. The scripted arm is the honest field comparator: it is what a
company would actually deploy today.

### 7.3 Questions-to-target, without planted truth

GLASSBOX's efficiency bar was defined in trait space — questions needed to reach
a target `RMSE(θ̂, θ_true)`. **That does not exist for real humans.** There is no
θ_true. The bar must be redefined in prediction space:

> **Target = the probe-block Brier score that the scripted 15-question interview
> achieves.** Questions-to-target = the number of adaptive questions needed to
> reach that same Brier score.

This is computable with no extra fieldwork. Each participant's interview answers
arrive in a known order, so the person encoder can be re-run after 1, 2, …, 15
answers and the probe block scored each time, giving a full efficiency curve per
person.

**Depth subsample:** 300 participants (150 per arm) get a **30-question**
interview instead of 15, so the curve extends past the point where the scripted
arm stops. Without this you cannot see where the curves converge, and the
convergence point is the commercially interesting number.

Pre-registered thresholds, chosen to be attainable (GLASSBOX's frozen RMSE ≤ 0.5
target turned out unreachable at any interview length — the full 202-item bank
floored at 0.525 — so the whole Gate-5 bar came back **undefined**; do not repeat
that mistake): report questions-to-target at **50%, 75% and 100%** of the
scripted-15 Brier gain, plus the fraction of the scripted-15 gain reached at
N = 3, 5, 10, 15, 25, 30.

**Pre-registered bar:** the adaptive arm reaches 100% of the scripted arm's
15-question Brier gain in **≤ 10 questions** (median across participants).

---

## 8. Endpoints, bars, and predicted numbers

Everything in this table is frozen before fieldwork. Anything not in this table
is exploratory and must be labelled as such in every report where it appears.

Intervals are **80% subjective prediction intervals** — the range in which we
would be unsurprised to find the answer, given the synthetic evidence. They are
not confidence intervals and they are not bars.

| # | Endpoint | Bar | Predicted | 80% interval | GLASSBOX reference |
|---|---|---|---|---|---|
| **E1** | **Brier lift over population marginal, single-answer scale, scripted arm** | **> 3%, one-sided α=0.025** | **9%** | **5.5–15%** | 14.3% (single-answer) / 27.1% (distribution) |
| E1c | Same, noise-corrected via retest (§3.2) | none — reported | 17% | 10–28% | 27.1% |
| E2 | Beats k-NN over the 15 interview answers | must beat it | beats it by 2 points of lift | −2 to +6 | model 27.1% vs k-NN 24.4%; margin 3.6% relative |
| E3 | ECE, 10 equal-width bins | ≤ 0.05 | 0.030 | 0.015–0.070 | 0.0137 |
| E4 | Reliability-curve correlation (binned, see note) | ≥ 0.95 | 0.97 | 0.92–0.99 | per-cell 0.684 — **not comparable**, see note |
| E5 | Lift over a demographics-only model | > 0, one-sided | 5 points of lift | 2–9 | ≈27 points (model 27.1% vs profile-only 0.1%; the profile baseline was intercept-only) |
| E6 | Test–retest pooled agreement | 70–90% band | 80% | 73–87% | 0.794 |
| E7 | Adaptive questions to reach scripted-15 Brier gain | ≤ 10 median | 8 | 5–13 | heuristic reached random's 52-question accuracy in 24 |
| E8 | Per-stratum lift: far ≥ near | none — reported | model wins far, k-NN wins near | — | model far 28.0% / near 22.6%; k-NN far 21.5% / near 24.9% |
| E9 | Item encoder, zero-shot loading cosine to fitted | ≥ 0.7 median | 0.78 | 0.65–0.88 | 0.861 |
| E9b | Item encoder, zero-shot discrimination correlation | ≥ 0.6 | 0.42 | 0.25–0.58 | 0.504 — **failed** |
| E10 | Predicted vs observed population marginals | none — reported | r ≥ 0.97 | 0.94–0.99 | 0.983 (marginals), 0.967 (2-way cross-tabs) |

### Notes on individual predictions

**E1 — why 9%.** Full reasoning in §3.3. The interval's low end (5.5%) is a
study that technically fails the bar; the high end (15%) is a study that matches
the synthetic result outright. Both are plausible. **A result at 6% would be a
real, publishable, honest finding** — "the interview beats the population
average by about 6% on Brier, which is small but real" — and the protocol should
be written so that outcome is reportable without embarrassment.

**E2 — the k-NN warning.** In GLASSBOX the whole margin of a fitted latent model
over "find the 5 most similar people and average them" was **3.6% relative**.
k-NN sees the same 15 answers. This is the single most likely way for the human
study to produce a deflating result, and it must be reported prominently, not
buried — the same rule `PREREGISTRATION.md` §6 imposes on Gate 4. The redeeming
pattern to look for is the stratum split (E8): k-NN wins on items *near* the
interview, the model wins on items *far* from it. An encoder that only beat
k-NN nearby would be memorising the interview rather than learning a person.

**E3 — expect worse than 0.0137.** GLASSBOX's calibration was measured against
50-sample answer distributions, with a model whose functional form matched the
data-generating process almost exactly. On humans the model is misspecified.
Posterior integration over the person's uncertainty (rather than plugging in the
point estimate) halved ECE in GLASSBOX, 0.0255 → 0.0137; do it, it is nearly
free. **This bar may fail. If it does, report it as a fail.**

**E4 — the per-cell correlation bar cannot be reproduced, and pretending
otherwise would be dishonest.** GLASSBOX's Gate-4 correlation bar compared each
predicted probability to a 50-sample empirical frequency in the same cell. A
panel gives one answer per cell, so the "empirical frequency" is 0 or 1 and the
correlation is a point-biserial with a mechanical ceiling far below 0.9. **Do
not report a per-cell correlation.** Report instead a binned reliability
correlation: bin all predicted probabilities into 20 equal-count bins, correlate
bin mean prediction against bin observed rate. That is a much weaker test — it
checks calibration in aggregate, not per-cell resolution — and the protocol
should say so. GLASSBOX's own attribution analysis showed the per-cell bar was
unreachable for this model class anyway: the full-information ceiling reached
only 0.786 against a bar of 0.90, with 66.9% of the shortfall attributed to
item-level idiosyncrasy no 8-dimensional trait vector expresses.

**E5 — the bar that gets harder with real people.** In the synthetic population
demographics carried essentially zero trait signal: the profile-only ridge
baseline came in at +0.1% lift, indistinguishable from knowing nothing, and an
LLM given demographics alone scored **−34.8%** — worse than predicting the
population average, because its priors tilted per-person in error-adding
directions. That is a clean negative worth carrying into any pitch, and it is
also a population-realism limitation: real demographics do carry signal, so a
demographics-only baseline will be genuinely harder to beat. Predicted: the
demographics-only model gets 2–5% lift over the marginal, and the interview
model beats it by 5 points. **If the interview does not beat demographics by a
clear margin, the product case is weak regardless of E1.**

**E9b — a pre-registered expectation of failure.** The bar stays frozen at 0.6
while the filed prediction is 0.42, following this project's own precedent: at
Stage 3 a projection filed *before any measurement* said the person-encoder RMSE
bar was unreachable, the bar was not moved, the stage ran, and the measurement
came in within 0.083 of the filed number on every dimension (PROJECT_LOG,
"Stage 3 design decided; pre-measurement projection filed"). Filing an expected
fail is what makes the eventual fail informative rather than embarrassing.
GLASSBOX's item encoder predicted loading *direction* well (cosine 0.861) and
*discrimination magnitude* badly (r 0.504 against a 0.6 bar). The sharper reading: it is a weak/strong
detector, not a graded predictor — dropping the weakest-fitted quintile
collapsed the correlation to 0.231. And a clean negative worth repeating:
**directly asking an LLM to rate how much people would disagree about an item
predicted the fitted discrimination at r = −0.003.** Zero. Do not build a
cold-start product on text-predicted item parameters.

**E10 — aggregate numbers are cheap.** GLASSBOX predicted population marginals
at r 0.983 and 1,225 two-way cross-tabs at r 0.967, while per-cell correlation
sat at 0.684. Segment-level claims are easy; individual-level claims are the
hard, honest currency. Report the aggregates, and say plainly that the high
marginal correlation is mostly threshold information the zero-information
baseline already has.

---

## 9. Analysis discipline

### 9.1 Freeze order

1. Item bank written and independently verified (§4.2, rule 7).
2. Vehicle map, banned-context lists and correlation budget recorded.
3. Interview/probe pool split, scripted interview order, probe stratification —
   all fixed with a recorded seed.
4. This endpoint table (§8) frozen.
5. Analysis code written and tested **against simulated data** with a known
   answer, before any real record exists.
6. Pilot (§5.4) → re-estimate variance → confirm or revise N **once**, with the
   revision recorded and dated.
7. Fieldwork.

### 9.2 Cross-fitting

Split participants into 5 folds with a recorded seed. For each fold: fit item
parameters and the population marginal on the other 4 folds; score this fold's
participants using those parameters. No participant is ever scored with
parameters fitted using their own answers. Report the fold-to-fold spread of the
primary endpoint.

### 9.3 The leakage rule

**When a result beats expectation, the first action is a documented hunt for
leakage, not celebration.** Pre-register the triggers:

- Primary lift above 25% (i.e. above the synthetic distribution-referenced
  result, on a scale where 14.3% is the synthetic equivalent).
- Binned reliability correlation above 0.995.
- Any single probe item contributing more than 5% of the total gain.
- Any distractor item fitting discrimination above the median of the intended
  strong items.

Each trigger fires a written investigation covering: participant-split
violations, item-split violations, parameter fitting that saw the probe block,
and duplicate participants across arms. GLASSBOX ran four such hunts. One found a
real confound (the card writer entangling TRU and RSK, two dimensions planted at
zero correlation). One documented real instrument behaviour rather than a defect
(distractors picking up biography coherence). Two came back clean — the
item-encoder direction head, and the two dimensions that beat their filed
projection, which cleared five separate checks. **A hunt that finds nothing is
still worth running and still gets written down.**

### 9.4 Honest-fail reporting

Copy the convention this project used at every gate: **the frozen-bar outcome is
stated first, in one fixed sentence, before any commentary.** For example, the
Gate-4 sentence was:

> "2 of 3 frozen bars pass (Brier lift 27.1% vs marginal, beats k-NN; ECE
> 0.0137). The correlation bar fails at 0.684 vs 0.90, structurally: the
> full-information ceiling of this model class reaches 0.786 — per-cell
> resolution beyond an 8-dimensional trait model, attributed 67% model class /
> 18% interview length / 15% zero-shot item encoder."

Commentary comes after, and never replaces it. Bars do not move after fieldwork
starts. A measurement done differently from this document is labelled
exploratory in every report where it appears.

### 9.5 Costs

Log field spend, incentive cost and compute per analysis run as it happens, not
retroactively.

---

## 10. SocSci210: a calibration replication you can run this week

This section exists because there is a real dataset that lets you test one
narrow part of the claim immediately, with no panel and no field spend.

**Scope note.** The synthetic study banned all external survey datasets (PRD §1).
That ban applies to GLASSBOX itself. This section is an explicit,
owner-sanctioned exception recorded in `results/STAGE6_NOTES.md` and is confined
to the human-study protocol.

### 10.1 The dataset

**SocSci210** — HuggingFace `socratesft/SocSci210`, from the Stanford
HCI / Simile group. Verified from the ACL Anthology camera-ready
(Kolluri, Wu, Park & Bernstein, *Finetuning LLMs for Human Behavior Prediction in
Social Science Experiments*, EMNLP 2025, `2025.emnlp-main.1530`):

- 2,901,390 responses from 400,491 participants across 210 social science
  experiments, spanning 1,197 outcomes and 1,194 conditions (5,998 unique
  stimuli).
- Source: NSF's Time-sharing Experiments for the Social Sciences (TESS).
  Nationally representative, high-powered studies (mean 1,907, median 1,954.5
  participants per study).
- Fields: `sample_id`, `participant`, `demographic` (age, education, employment,
  gender, housing, ideology, income, …), `stimuli`, `response`, `condition_num`,
  `task_num`, `prompt`, `reasoning`, `study_id`.
- Three split mappings shipped in the metadata folder:
  `participant_mapping.json` (study-level seen/unseen splits **by participant**),
  `task_mapping.json` (75/25 by task), `condition_mapping.json` (75/25 by
  condition).

**Use `participant_mapping.json`.** That is the unseen-participant split: within
the 170 training studies, participants are divided once into `participant-train`
(50%) and `participant-eval` (50%). It is the only split that isolates
"generalise to a new person" from "generalise to a new study".

**The shallowness, stated numerically:** 2,901,390 / 400,491 = **7.2 responses
per participant**. GLASSBOX observed each persona on 252 items. This dataset
observes each person on about seven. That single fact determines what this
replication can and cannot do (§10.5).

### 10.2 What is replicated

**The calibration claim only:** do predicted answer distributions match real
respondents' answers?

Design:

1. Fit item parameters (loadings + thresholds) for each study's outcome
   questions on `participant-train`.
2. For each `participant-eval` respondent, hold out one response and use the
   rest (typically 6) plus demographics as the "interview".
3. Predict the held-out response distribution.
4. Grade against the actual answer, on the same three metrics as the panel
   study: Brier lift over the within-condition marginal, ECE, and binned
   reliability correlation.
5. Also compute the paper's own metric (§10.4) so the numbers are comparable to
   the published baseline.

Baselines, all mandatory: within-condition population marginal (zero
information); demographics-only model; k-NN over the observed responses.

### 10.3 Pretraining-contamination check — mandatory, before any number is trusted

TESS studies are publicly archived. Their stimuli, question wordings and in some
cases their published result tables may sit in the pretraining corpus of any
model used here. **The section is void without this check and an honest
statement of its limits.** This is a binding condition recorded in
`results/STAGE6_NOTES.md`.

Note that the Wall's system-side ban on token log-probabilities applies to the
synthetic study only — there is no planted truth here, so log-probability
methods are available and should be used.

**Method, all five, documented and reported whatever they show.**

1. **Verbatim continuation probe.** Feed the first half of each distinctive
   stimulus and ask the model to continue. Score exact and near-exact
   (normalised edit distance) recovery of the held-out half against a matched
   set of newly-written control stimuli. Report the recovery-rate difference.
2. **Metadata elicitation probe.** Ask the model directly: what study is this
   from, what was the response scale, how many participants, what was the main
   finding. Score against the true study record. A model that names the TESS
   study ID has seen it.
3. **Min-K% membership check.** Compute per-token log-probabilities on each
   stimulus and take the mean of the lowest-K% tokens (K = 20). Memorised text
   has anomalously high values on its rarest tokens. Compare stimuli against
   length- and topic-matched controls written after the model's cutoff. Report
   the AUC of the separation; report it even when it is ~0.5.
4. **Temporal control.** Split the 210 studies by archive date into pre-cutoff
   and post-cutoff relative to the model's stated training cutoff. Run the full
   replication separately on each half. **If the lift is the same on both halves,
   contamination is unlikely to be driving the result. If it is much larger on
   the pre-cutoff half, stop and report that instead of the headline number.**
   This is the strongest of the five checks and the only one that tests the
   thing you actually care about.

   **Precondition, decided before the run, not discovered during it.** This check
   requires a *reliably documented* pretraining cutoff for the model being
   checked. For GLASSBOX's frozen system-side model (Qwen3.6-27B) that is
   unlikely to be available, and the same is true of most open-weight
   checkpoints. **If the cutoff is not documented by the model's publisher —
   a date in a model card or technical report, not a guess and not a
   self-report elicited from the model — check #4 is declared INAPPLICABLE.**
   Record the declaration and the reason in the results file, run the remaining
   four checks, and state plainly in every writeup that the contamination
   evidence is weaker because the one check that tests the outcome directly
   could not be run. Do not substitute an inferred cutoff; an inferred date
   produces a temporal split that means nothing and a false sense of clearance.
5. **Demographic-swap control.** Re-run predictions with each respondent's
   demographic block replaced by a random other respondent's from the same
   condition. If the model's predictions barely change, it is reproducing a
   remembered marginal rather than conditioning on the person, and the
   person-level claim is void.

**What the check can rule out:** strong verbatim memorisation of stimulus text;
explicit recall of study identity and design; a large temporal advantage on
studies the model could have seen (**only if check #4 was applicable** — see the
precondition above); predictions that ignore the person entirely.

**What it cannot rule out, and this must be written next to any number:**

- Having absorbed a *summary* of a study's findings from a paper, blog post or
  textbook without memorising its text. Min-K% and continuation probes cannot see
  this.
- Having learned the general shape of American survey response distributions
  from the enormous amount of survey-like text on the internet. This is not
  contamination in the leakage sense, but it means "the model predicts the
  marginal well" carries no information about the person-level claim. Only the
  lift over the within-condition marginal does — which is why that baseline is
  mandatory.
- Contamination of the *base* model of any comparison system, which we cannot
  inspect at all for closed models.
- A negative result on all five checks is weak evidence of cleanliness, not
  proof. Say so in those words. If check #4 was declared inapplicable, a negative
  result on the remaining four is weaker still, and the writeup says *that*.

### 10.4 Reference baseline: Socrates-Qwen-14B

**Camera-ready figures only.** All numbers below were read from the ACL
Anthology camera-ready PDF text of `2025.emnlp-main.1530`, not from a search
summary and not from a secondary source.

> **⚠ Note to future editors — do not "correct" these numbers.** A web search
> summary of the Anthology landing page returns **36% over base / 15% over
> GPT-4o**. Those figures are wrong and appear nowhere in the paper. The
> camera-ready PDF, arXiv v1 (6 Sep 2025) and arXiv v2 (5 Nov 2025) all state
> **26% / 13%**, consistent with Table 2 below (26.3% / 13.2%). Verified by
> extracting the PDF text directly, 2026-08-01. If you are about to raise these
> numbers, you are reading a summariser, not the paper.

**Their metric** (needed to compare at all): for each (condition, outcome) pair,
responses are standardised to [0,1] and the **Wasserstein distance** is computed
between the predicted response distribution across participants in that
condition and the empirical distribution. Distances are averaged within a study,
then across studies. **Lower is better.** They report relative change against a
baseline, because the raw range is narrow.

**Table 2 of the camera-ready — unseen-studies split (170 train / 40 test):**

| Arm | Wasserstein ↓ | vs its base | vs GPT-4o |
|---|---|---|---|
| GPT-4o base | 0.174 | — | — |
| GPT-4o + few-shot (5) | 0.161 | 7.5% | 7.5% |
| Qwen2.5-14B base | 0.205 | — | −17.8% |
| **Qwen2.5-14B + SFT (= Socrates-Qwen-14B)** | **0.151** | **26.3%** | **13.2%** |
| LLaMA3-8B base | 0.219 | — | −25.9% |
| LLaMA3-8B + SFT (= Socrates-Llama-8B) | 0.153 | 30.1% | 12.1% |
| Uniform guess (lower bound) | 0.203 | — | — |
| Empirical best (bootstrap upper bound) | 0.125 | — | — |

Abstract figures, consistent with the table: **26%** more aligned than the base
model on completely unseen studies, **outperforming GPT-4o by 13%**; 71% on
unseen conditions within a study; 10.6% reduction in demographic parity
difference.

Two things in that table are worth more than the headline. **The Qwen2.5-14B
base model (0.205) is no better than a uniform guess (0.203) on distributional
alignment** — an untuned LLM asked to simulate survey respondents is worthless
on this metric. And **the empirical best is 0.125**, so the finetuned model
closes about two thirds of the distance from base to the noise floor.

**On the unseen-participant split specifically (§5.5 of the paper):** with 10% of
participant data, DPO tuning raises individual accuracy from 71% to 75% (a 13%
relative error reduction), and learning saturates at around 10% of participant
data. **The Wasserstein values for this split appear only in the paper's Figure 4
and are not tabulated.** So:

> **`[ACL camera-ready Figure 4 — read off the figure, or recompute by running
> the released Socrates-Qwen-14B checkpoint on your exact split before
> quoting]`**

The right move is to recompute rather than quote: the models are released
(`stanfordhci.github.io/socrates`), so run the checkpoint on the identical split
you use. That removes both the figure-reading error and any split mismatch.

**A fairness caveat that must be stated wherever the comparison appears.** On the
unseen-participant split, Socrates-Qwen-14B has already trained on the study,
its conditions, its outcome questions and half of its participants. That is by
design and is the point of the split — but it means this is *not* a
cold-start comparison. Our model would likewise fit item parameters on the same
`participant-train` half, so the comparison is fair in the sense that both sides
see the same half. It is not fair in the sense of "which method works on a new
study": for that, use the unseen-studies split and compare against 0.151.

### 10.5 What this replication cannot test

Because each person is observed on about seven responses:

- **The person encoder's depth.** GLASSBOX's encoder consumes 15 closed answers
  and 3 open-ended answers and produces an 8-dimensional posterior. With ~6
  observed responses per person there is not enough information to identify more
  than one or two dimensions. This tests calibration, not trait recovery.
- **Blur honesty against latent truth.** There is no latent truth. See §11.
- **Adaptive interviewing.** With no control over what each respondent was asked,
  there is no sequential decision to make and no efficiency curve to draw.
- **Test–retest.** Respondents are observed once.
- **The near/same-domain/far generalisation curve** in any controlled form,
  since the observed and held-out items were not stratified by design.

State all five in any writeup of this section. The replication's value is that
it is real human data and it can be run this week. Its value is not that it
substitutes for the panel study.

---

## 11. What no human study can test

This is the section to read if you only read one.

### 11.1 The finding

GLASSBOX measured something a human study structurally cannot measure.

The person encoder produces a point estimate and a **blur** — its own declared
uncertainty. At 15 questions the blur was well calibrated: ±1σ intervals covered
the planted truth 64.9% of the time, inside the 60–75% band. Good.

Then more evidence was added, and the coverage **fell**:

| Evidence | ±1σ coverage of planted truth |
|---|---|
| 1 answer | 0.746 |
| 15 answers | 0.649 |
| 202 answers | 0.358 |

Nominal is 68%. With the full answer matrix the model's stated uncertainty was
wrong by a factor of about 2.2 in standard deviation, and the decomposition
attributed **88% of that overconfidence to the persona-rendering step and 12% to
answer noise**.

The mechanism is simple and general. The posterior prices the noise it can see —
the randomness in the answers. It cannot price the distortion between the
person's actual disposition and what their answers can express, because nothing
in the observable data reveals that distortion. **More data makes the model more
confident about the wrong thing.** Small samples hide the blind spot; large
samples expose it.

The same pattern showed up in the adaptive loop: across 150 questions the
declared blur fell to 24% of its starting value while true error fell only to
55%, and the gap widened monotonically. And it was shown not to be a
reinforcement-learning artefact — the never-trained heuristic showed a *larger*
declared-versus-true gap than the trained policy.

### 11.2 Why a human study cannot see it

Detecting this required comparing the model's declared uncertainty against
**latent truth**. In a human study the only reference is held-out answers. A
model that is honest about answer noise and blind to everything else will look
perfectly calibrated against held-out answers — which is exactly what GLASSBOX's
Gate-4 ECE of 0.0137 shows, in the same run where coverage against latent truth
was 0.358.

**A human study can confirm that a confidence model is calibrated against
observable data. It cannot detect this class of overconfidence at all.** Both
statements are true of the same model at the same time.

So: E3 in §8 passing is not evidence that the confidence model is honest. It is
evidence that it is honest about the thing it can see. Write that into the
report.

### 11.3 What to do about it

1. **Do not claim honest uncertainty from a human study.** Claim "calibrated
   against held-out answers", which is what was measured.
2. **Run a planted-truth companion.** A synthetic recovery study alongside the
   panel is the only instrument that can see this. That is what GLASSBOX is for,
   and it is the argument for keeping one running permanently next to any
   deployed system.
3. **Get truth-anchored feedback where you can.** Any downstream outcome you can
   observe — a purchase, a vote, a churn event, a follow-up behaviour — is a
   partial anchor that answers alone cannot provide. This is the direction the
   Simile group's confidence-model work points at, and it is the honest answer
   to "how do we know the model's confidence means anything".
4. **Report the coverage-decay curve** from the synthetic study alongside the
   human calibration numbers. It is the exhibit that explains why the human
   calibration number, on its own, does not settle the question.

### 11.4 The other things that go away

- **Trait recovery** — no θ_true, so no `r(θ̂, θ)`. Substitute: stability of θ̂
  across cross-fitting folds and across the repeat visit. That measures
  reliability, not validity, and the difference must be stated.
- **Item recovery against designed loadings** — you can measure whether fitted
  loadings are stable across folds and whether they match the vehicle map's
  intent, but the vehicle map is a design intention, not planted truth.
- **The ordering sanity check on weak items** — partly survives: deliberately
  weak items should still fit low discrimination. GLASSBOX's version of this
  check found that one item labelled weak fitted at 3.74, i.e. it was "labelled
  weak, not written weak". Expect the same and report it.

---

## 12. Deliverables

1. **Pre-registration**, this document with §4 and §8 filled in for the actual
   bank and frozen, posted publicly with a timestamp before recruitment.
2. **Item bank**, with the vehicle map, banned-context lists, correlation budget,
   verification log (including defect counts per round) and the frozen splits.
3. **Analysis code**, tested on simulated data before fieldwork.
4. **Results report**: frozen-bar outcomes first in fixed sentences, then
   commentary, then exploratory analyses clearly labelled, then a limitations
   section that includes §11.
5. **Failed rounds**, listed, with what they cost and what they changed.
6. **Cost log.**

---

## 13. Freeze

| | |
|---|---|
| Version | 1.0 — drafted at GLASSBOX Stage 6 |
| Status | **Template. Not frozen.** Freeze after §4 and §8 are filled in for the real bank and the pilot has confirmed the variance assumptions. |
| Frozen by | *(name, date)* |
| Changes after freeze | Recorded here with date and reason. Bars do not move. |
