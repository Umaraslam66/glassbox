# GLASSBOX — findings

Final report, Stage 6. Written against the frozen bars in `PREREGISTRATION.md` and the
result files in `results/`.

**Citation rule used throughout.** Every number carries a pointer to the file it came
from, in the form `(stage4_gate4.json -> verdict.bars.primary_lift)`. All such files live
in `results/`. Where the project log and a result file disagree, the result file is what
is printed here and the difference is called out.

---

## 1. What this study claims, and the scoreboard

### The claim

GLASSBOX is a **parameter recovery study**. In the frozen wording of
`PREREGISTRATION.md` section 1, we test one claim: *"the pipeline provably recovers
planted ground truth when ground truth is known."* We generated 500 synthetic people,
planted an 8-dimensional trait vector in each one, hid it behind the Wall, and measured
whether the system could recover it from interview text alone.

It does **not** claim the system predicts real humans. That claim needs a human panel and
is out of scope. Everything below is about synthetic people whose answers we control.

Scale: 500 personas (400 training / 100 held out), 252 closed questions (202 training /
50 held out) plus 20 open prompts, 141,000 answer cells per population sweep, three full
sweeps.

### Honest scoreboard

Frozen bar wording is quoted from `PREREGISTRATION.md` section 6. Nothing here is
softened.

#### Gate 1 — population QA (development stage; sanity gate only)

| Frozen bar (verbatim) | Measured | Verdict |
|---|---|---|
| "no persona pair with exact-match answer agreement > 95% on the full bank" | max pairwise 0.7540 (full_sweep_v2_qa.json -> diversity.max_pair_agreement) | **PASS** |
| "median pairwise exact-match agreement across all persona pairs <= 80%" | 0.3849 (full_sweep_v2_qa.json -> diversity.median_pair_agreement) | **PASS** |
| "effective dimensionality (participation ratio, section 5) >= 5" | 31.67 (full_sweep_v2_qa.json -> diversity.effective_dimensionality) | **PASS** |
| "Bar: median across the 8 dimensions >= 0.5" (trait obedience) | median 0.8299, weakest dimension RSK 0.7827 (full_sweep_v2_qa.json -> obedience) | **PASS** |
| "population test-retest agreement ... in the 70-90% pooled band" | 0.7942 pooled; binary exact 0.8214 vs chance 0.50, Likert +/-1 0.7808 vs chance 0.52 (full_sweep_v2_qa.json -> retest) | **PASS** |

Per `PREREGISTRATION.md` section 1, **no claim is made from Stage 1 beyond this sanity
gate.**

One honesty note carried forward from the log: the participation ratio of 31.67 is
measured on the noised answers and partly measures the isotropic noise floor. The raw
(un-noised) figure, 13.89 (full_sweep_v2_raw_qa.json -> diversity.effective_dimensionality),
is the structural number. Do not cite 31.67 as evidence of latent richness.

#### Gate 2 — static recovery (full answer matrix, no interviews)

| Frozen bar (verbatim) | Measured | Verdict |
|---|---|---|
| "per-dimension r(theta-hat_fit, theta_true) >= 0.8 on held-out personas, all 8 dimensions, after Procrustes alignment" | worst RSK 0.7983, next SOC 0.7996; 6 of 8 dimensions clear 0.80; mean 0.8386 (stage2_v2_recovery.json -> trait_recovery_holdout) | **FAIL** |
| "Bar: median across held-out items r >= 0.7" (item recovery) | median per-item r 0.9221 (stage2_v2_recovery.json -> verdict.bars.item_recovery) | **PASS** |
| "Designed weak items must show low fitted discrimination (ordering sanity check)" | 0.8247 chance that a random designed-weak item sits below a random strong item; weak median 0.9197 vs strong median 1.9979 (stage2_v2_recovery.json -> weak_item_ordering) | **PASS** (grader's reading of qualitative wording; both distributions preserved in the file) |
| "+/-1 sigma intervals cover true theta 60-75% of the time" (blur honesty) | pooled coverage 0.3575 (stage2_v2_recovery.json -> blur_honesty.pooled_coverage) | **FAIL** |

**Owner's verdict, verbatim and binding:**

> "2 of 4 frozen bars unmet: trait recovery 0.798/0.7996 on RSK/SOC vs 0.80; blur
> coverage 0.358, structurally attributable to card-rendering distortion invisible to the
> posterior."

*(The result file records the coverage as 0.3575; the owner's fixed phrasing rounds it to
0.358. Same measurement.)*

#### Gate 3 — translators (generalization beyond the fitted set)

| Frozen bar (verbatim) | Measured | Verdict |
|---|---|---|
| "median loading-vector cosine similarity >= 0.7" (item encoder, held-out items, zero-shot) | 0.8607 (stage3_item_encoder.json -> bars.median_loading_cosine) | **PASS** |
| "predicted vs fitted discrimination r >= 0.6" (item encoder) | 0.5040 on the confirmatory raw scale; 0.5770 on the log scale, reported alongside — fails on both (stage3_item_encoder.json -> bars.discrimination_r, magnitude_arms.primary) | **FAIL** |
| "per-dimension RMSE(theta-hat, theta_true) <= 0.5" (person encoder at N = 15) | worst TRD 0.7878, best TEC 0.5595, **0 of 8 dimensions under the bar**; pooled 0.6971 (stage3_gate3.json -> tracks.fused.bars.rmse) | **FAIL** |
| "blur coverage 60-75%" (person encoder at N = 15) | 0.6488 (stage3_gate3.json -> verdict.bars.blur_coverage) | **PASS** |
| "blur shrinks monotonically in expectation as N grows (plot required)" | largest uptick in any mean-blur curve 0.00228, i.e. 0.9% of that dimension's total shrinkage; 15.2% of individual steps rise (stage3_gate3.json -> verdict.bars.monotone_blur) | **PASS by orchestrator ruling** under the frozen "in expectation" wording. The grader's own two mechanical readings both record FAIL in the result file (`mean_curve_verdict`, `strict_per_step_verdict`); both readings are preserved so the call can be re-made. |
| "person-encoder RMSE beats the public-profile-only baseline by >= 30%" | 26.8% fused; 21.4% closed-only (stage3_gate3.json -> tracks.fused.bars.lift) | **FAIL** |

**Owner's verdict, verbatim and binding:**

> "3 bars unmet (item-encoder discrimination 0.504 vs 0.60; person-encoder RMSE, 0/8 dims
> under 0.50; lift 26.8% vs 30%), with the RMSE and lift shortfalls attributable to the
> accepted card-transmission ceiling: the full 252-item matrix itself reaches only RMSE
> 0.529 / 44% lift against planted truth. The 15-question transcript recovers r 0.770 of
> the full-matrix estimate."

*(Two pointers on that sentence. The full-matrix reference is scored over the 202
**training** items on held-out personas — `stage3_gate3.json -> full_matrix_reference.n_items`
= 202, pooled RMSE 0.5292; the bank is 252 items in total. The 44% is that same arm read
as lift: 1 - 0.5292 / 0.9523, from `full_matrix_reference.pooled_rmse` and
`tracks.fused.bars.lift.pooled_baseline_rmse`.)*

The lift bar carries its own clause — "If not, stop and investigate before proceeding."
That investigation was run; it is summarised in section 2 and finding 3.3.

#### Gate 4 — end-to-end prediction + calibration (headline gate)

| Frozen bar (verbatim) | Measured | Verdict |
|---|---|---|
| "Brier score beats the population-marginal baseline by >= 25% relative, and beats k-NN" | Brier 0.06269 vs marginal 0.08601 = **27.1% relative lift**; k-NN 0.06506, so the model beats it (stage4_gate4.json -> verdict.bars.primary_lift) | **PASS** |
| "ECE <= 0.05" | **0.01368** (stage4_gate4.json -> verdict.bars.ece) | **PASS** |
| "correlation between predicted probability and empirical 50-sample frequency >= 0.9 across all persona x item cells" | 0.6844 pooled (binary 0.7000, Likert 0.6154) (stage4_gate4.json -> verdict.bars.correlation) | **FAIL** |

The frozen document requires a close k-NN to be reported prominently, so: **k-NN alone
reaches 24.4% lift** (stage4_gate4.json -> verdict.bars.primary_lift.knn_lift_vs_marginal);
the model's whole margin over it is 3.6% relative.

**Owner's verdict, verbatim and binding:**

> "2 of 3 frozen bars pass (Brier lift 27.1% vs marginal, beats k-NN; ECE 0.0137). The
> correlation bar fails at 0.684 vs 0.90, structurally: the full-information ceiling of
> this model class reaches 0.786 — per-cell resolution beyond an 8-dimensional trait
> model, attributed 67% model class / 18% interview length / 15% zero-shot item encoder."

Owner's commentary, which never replaces the frozen-bar sentence above:

> "the two passing bars are the ones the headline claim rests on (lift + calibration); the
> failing bar measures per-cell resolution no trait-vector model of this dimensionality
> reaches on this population."

#### Gate 5 — adaptive interviewing + RL (efficiency gate)

| Frozen bar (verbatim) | Measured | Verdict |
|---|---|---|
| "(a) Median questions-to-target for the adaptive interviewer (RL or info-gain heuristic) <= 50% of median questions-to-target for random ordering" — target = "per-dimension RMSE(theta-hat, theta_true) <= 0.5" | **No strategy reaches the target at any N <= 150.** 0 of 51 replicates for random, fixed, heuristic and RL; the full 202-item bank floors at pooled 0.5253 / worst dimension 0.5637 (stage5_strategies.json -> frozen_bar.population_curve_crossing; curves.random.pooled_rmse_full_bank) | **UNDEFINED** — the comparison the bar asks for does not exist at the frozen threshold |
| "(b) RL vs heuristic on questions-to-target: reported either way; heuristic winning is a legitimate finding, not a failure" | Undefined at the frozen target for the same reason (stage5_strategies.json -> frozen_bar.comparison.policy_vs_heuristic) | **UNDEFINED** |

This outcome was pre-declared. Owner RULING 1, recorded in the log **before any Stage 5
compute existed**, said the target might be unreachable at any N, that the bar would then
be reported as unmet/undefined and not reinterpreted, and pre-authorised a
labelled-exploratory grid at attainable thresholds. That grid is finding 3.11.

**Owner's verdict, verbatim and binding:**

> "Both frozen bars UNDEFINED at the frozen RMSE <= 0.5 target (full-bank floor 0.525,
> pre-declared in Ruling 1); on the pre-declared exploratory grid, adaptive interviewing
> reaches the same accuracy in <=50% of random's questions at attainable thresholds
> (0.462 and 0.494 at <=0.70 and <=0.65); RL ties the info-gain heuristic; proxy-gaming
> watch fired and was attributed to the card-transmission blind spot, not reward hacking
> (heuristic shows a LARGER declared-vs-true gap than the trained policy; oracle-reward
> arm within 0.024 RMSE of the deployable blur reward)."

### Scoreboard in one line

Gate 2 — 2 of 4 bars unmet. Gate 3 — 3 bars unmet. Gate 4 — 2 of 3 pass, 1 fails.
Gate 5 — both bars undefined at the frozen target. Gate 1 passed all five, and Gate 1 is
development only.

---

## 2. The spine: one measured ceiling explains almost every fail

### The ceiling

A synthetic person is not the trait vector. It is a **life story written from** the trait
vector, and the writing step loses information. We measured how much.

Take the planted trait value on one dimension and correlate it, across held-out people,
with the average answer they actually gave on the items designed for that dimension. No
model fitting — just what the answers themselves carry:

| Dimension | TRU | RSK | ENV | PRC | TRD | SOC | TEC | LOC |
|---|---|---|---|---|---|---|---|---|
| answer-matrix ceiling (r) | 0.839 | 0.821 | 0.871 | 0.807 | 0.818 | 0.770 | 0.866 | 0.861 |

**Persona cards transmit ~85% of the planted signal (measured median 0.830, range
0.770-0.871 across dimensions; stage2_v2_recovery.json -> answer_matrix_ceiling)** — and
nothing downstream can recover what never left the card.

The fit lands on that ceiling, not below it. Per dimension, fitted recovery is within
-0.029 to +0.033 of the model-free ceiling on every axis
(stage2_v2_recovery.json -> answer_matrix_ceiling.fit_minus_answers_per_dimension); the
three MIRT restarts agree to r 0.9997
(stage2_v2_recovery.json -> fit_diagnostics.restart_agreement, worst per-dimension
theta correlation across the three pairs) and split-half reliability of theta-hat is 0.933
(stage2_v2_recovery.json -> reliability_ceiling.reliability_summary.median). The estimator
is not the bottleneck.
The rendering step is.

### What inherits the ceiling, and what does not

The rule turns out to be simple. **Bars graded against planted truth inherit the ceiling.
Bars graded against the person's own answers do not.**

| Bar | Anchored to | Outcome |
|---|---|---|
| Gate 2 trait recovery (r >= 0.80) | planted truth | FAIL — 0.7983 / 0.7996 on the two weakest dimensions |
| Gate 2 blur honesty (60-75%) | planted truth | FAIL — 0.3575 |
| Gate 3 person RMSE (<= 0.5) | planted truth | FAIL — 0 of 8 dimensions |
| Gate 3 lift (>= 30%) | planted truth | FAIL — 26.8% |
| Gate 5 questions-to-target (RMSE <= 0.5) | planted truth | UNDEFINED — unreachable at any N |
| **Gate 4 Brier lift (>= 25%)** | **the person's own answer distribution** | **PASS — 27.1%** |
| **Gate 4 ECE (<= 0.05)** | **the person's own answer distribution** | **PASS — 0.0137** |

Gate 4 was opened on exactly this reasoning, recorded in the log before the stage ran: its
bars grade against each persona's true answer distribution (50 seeded applications of the
frozen noise layer per cell), so the card-transmission ceiling largely drops out and the
bars are meant to be passable at full strength. They were.

The size of the ceiling shows up cleanly in Gate 5. Every strategy converges to the same
floor: pooled RMSE 0.5253 (random), 0.5255 (heuristic), 0.5271 (RL policy) from the full
202-item bank (stage5_strategies.json -> curves.*.pooled_rmse_full_bank). The frozen target
was 0.5 on **every** dimension; the best worst-dimension any arm reaches from the whole
bank is 0.5637. The bar sits below the floor. More questions cannot fix it, which is why
the honest answer is "undefined" and not a large number.

### The confidence-model exhibit: coverage decays as evidence grows

This is the most transferable result in the project.

The system's uncertainty is calibrated on data it can see — answer noise. It cannot see
the card-rendering distortion, because that lives on the far side of the Wall. So its
error bars are honest about the noise and blind to the loss. With one question, sampling
noise dominates and the intervals look well calibrated. As evidence accumulates the
sampling noise shrinks, the invisible rendering error does not, and the intervals become
provably overconfident.

| Evidence | +/-1 sigma coverage of planted truth | Source |
|---|---|---|
| 1 item | 0.7463 | stage3_gate3.json -> anticipation_note.coverage_decay.fused.at_n1 |
| 5 items | 0.6888 | same, at_n5 |
| 10 items | 0.6775 | same, at_n10 |
| 15 items + 3 open answers | 0.6488 | same, at_n15 |
| 202 items (full matrix) | 0.3575 | stage2_v2_recovery.json -> blur_honesty.pooled_coverage |

Nominal is 68%; the accepted band is 60-75%. Coverage starts in band and ends at about
half the nominal rate, and the decay is monotone. Nothing was recalibrated, and per the
standing rule nothing ever could be: tuning system-side uncertainty against planted theta
would be a Wall breach.

The banked finding, in the owner's wording:

> "uncertainty calibrated only against observable data understates error against latent
> ground truth when the persona-rendering step is lossy (measured here: 88% card-induced /
> 12% noise-induced). Honest confidence models need truth-anchored feedback — connect to
> Simile's confidence-model direction in the writeup"

*(The 88/12 split is cited as recorded, not recomputed: it appears in the owner's banked
Gate 2 finding in `results/STAGE6_NOTES.md`, which is the sentence quoted above, and in the
PROJECT_LOG exploratory block at the Gate 2 second attempt. It is not persisted in a result
JSON, and it does not appear in the owner-accepted Gate 2 verdict quoted earlier.)*

The same blind spot reappears in the adaptive interviewer, measured a third way. Over a
150-question interview the system declares away 76% of its uncertainty (declared blur falls
to 0.2362 of its starting value for the heuristic) while removing only 45% of its actual
error (0.5536 remaining) — a gap of 0.317, widening monotonically
(stage5_strategies.json -> exploratory.blur_vs_truth.heuristic). Three stages, three
methods, one phenomenon.

---

## 3. Findings

Findings 3.1-3.10 are the notes banked in `results/STAGE6_NOTES.md` at the Gate 2, 3 and 4
closes on owner instruction. 3.11 and 3.12 come from Gate 5. 3.13-3.15 are carried here
from the log on the owner's instruction.

### 3.1 (banked, Gate 2) Confidence models calibrated only on observable data go silently overconfident

Covered in full in section 2. Measured here as 88% card-induced / 12% noise-induced. The
practical consequence: an honest confidence model needs truth-anchored feedback, which a
deployed system almost never has.

### 3.2 (banked, Gate 3) Demographics carry ~zero trait signal in this population

The public-profile-only ridge baseline selected the intercept-only model in
cross-validation. It predicts the training mean for every persona, and its RMSE equals the
population spread (stage3_gate3.json -> tracks.fused.bars.lift.baseline_is_intercept_only,
with `baseline_rmse_per_dimension` and `population_rms_per_dimension` matching on 7 of 8
dimensions to three decimals, PRC to two). At Gate 4 the same baseline scores +0.1% Brier
lift over the zero-information
marginal — indistinguishable (stage4_gate4.json -> arms.profile.brier 0.08590 against
arms.marginal.brier 0.08601).

Two consequences. First, the Gate 3 "lift over profile" bar and the zero-information bar
are the **same bar** here, which makes 26.8% a clean lift-over-nothing number. Second, this
is a **population-realism limitation**: real demographics do carry signal, so a
lift-over-profile bar would be harder on real humans, not easier.

### 3.3 (banked, Gate 3) 15 questions + 3 open answers recover 77% of what the full bank knows

The 15-question fused estimate correlates r 0.770 (mean over dimensions) with the estimate
from all 202 answered items on the same held-out people
(stage3_gate3.json -> transcript_vs_full_matrix.system_side_agreement_at_n15.fused.r_mean).
Closed items alone give 0.686.

Read that carefully: 7% of the questions get you 77% of the agreement. The interview is
efficient. The ceiling it is efficient *towards* is still the ceiling.

This is also the "stop and investigate" answer the Gate 3 lift bar demanded. The encoder is
not broken and the transcript is not failing to add signal; the shortfall is the accepted
card-transmission ceiling downstream. Even the full 202-item answer matrix only reaches
pooled RMSE 0.529 (44% lift), so 15 closed + 3 open reaching 26.8% is the expected fraction
of a ceiling Gate 2 had already documented.

### 3.4 (banked, Gate 3) Open text moves the estimate a lot and buys almost no declared precision

The three open answers lift the point estimate from r 0.686 to 0.770 against the
full-matrix reference. Measured in units of posterior sharpness, they are worth about
**3 closed items** pooled — and unevenly, from 0.3 to 9.2 items depending on the dimension
(stage3_gate3.json -> open_answers_worth_in_closed_items).

Text is information-rich and precision-poor under our fusion. A system that trusts free
text the way it trusts a scored item will be overconfident.

### 3.5 (banked, Gate 3) Clean negative: asking the model how discriminating an item is predicts nothing

We asked Qwen directly how much people would disagree about each item. Correlation with the
item's actual fitted discrimination: **r = -0.0029**
(stage3_item_encoder.json -> magnitude_arms.polarization_only.holdout_r).

Learned heads on the same model's pole judgments plus embeddings reach r 0.504 — the
information is in the model, it just does not come out through introspection. Ablations:
embeddings only 0.407, judgments only 0.340, wording only 0.281, topic domain only 0.017
(same file, magnitude_arms).

The direction head is a different story and passes its bar comfortably: median held-out
cosine 0.861, against 0.265 for a topic-domain-only baseline and 0.258 for a no-text
constant (stage3_item_encoder.json -> direction_arms). **Which way an item points is
learnable from text; how sharply it separates people is much harder.**

Sharper caveat on the magnitude head, from the same file: it is a weak/strong detector, not
a graded predictor. Excluding the weakest-fitted quintile of held-out items collapses r
from 0.504 to 0.231 (diagnostics.exploratory_excluding_weakest_quintile).

### 3.6 (banked, Gate 3) The coverage-decay curve

Presented as the centrepiece in section 2.

### 3.7 (banked, Gate 4) An LLM given only demographics is worse than knowing nothing

Qwen3.6-27B, given each held-out persona's public profile and no interview, predicted
answer distributions for all 50 probe items (15,000 predictions, 0 parse failures).

| Arm | Brier | Correlation | ECE |
|---|---|---|---|
| Population marginal (zero information) | 0.08601 | 0.554 | 0.0110 |
| **LLM, profile only, no interview** | **0.11594** | **0.336** | **0.0704** |
| Our model (15 + 3 interview) | 0.06269 | 0.684 | 0.0137 |

Source: stage4_gate4.json -> arms. The LLM is **34.8% worse than predicting the population
average**. It tilts per-person in directions that add error, and its calibration is far
worse than either reference: ECE 0.0704, which is 6.4 times the marginal's 0.0110 and 5.1
times our model's 0.0137.

This is the direct evidence against "just prompt an LLM with a persona description". On
this population that approach is worse than a lookup table — and the profile-only ridge
confirms it from the other side (finding 3.2): there is nothing in the demographics to tilt
on, so any tilt is noise.

### 3.8 (banked, Gate 4) Item difficulty cannot be read off the question text

The fully zero-shot item side — loadings *and* thresholds predicted from item text —
collapses to 2.7% Brier lift (stage4_gate4.json -> arms.model_text_thresholds.brier 0.08370
against arms.marginal.brier 0.08601). With thresholds fitted from training personas'
answers and only the loadings zero-shot, the same predictor reaches 27.1%.

Cold-start implication for any real product: you can guess what a new question measures,
but you cannot guess how hard it is. That needs response data.

### 3.9 (banked, Gate 4) Aggregate prediction is dramatically easier than per-person prediction

| What is predicted | Agreement with truth |
|---|---|
| Population marginal per item | r 0.9826, mean absolute gap 0.028 (stage4_gate4.json -> aggregate_check) |
| Two-way cross-tabs, 1,225 item pairs | r 0.9671 (stage4_gate4.json -> cross_tabs) |
| **Individual cell probabilities** | **r 0.6844 — the failed bar** |

Same model, same predictions, three orders of aggregation. Segment-level claims are cheap;
individual-level claims are the hard, honest currency. Anyone quoting a synthetic-panel "we
match the population within 3 points" number is quoting the easy one.

Two caveats stated on the artifact itself. The high marginal correlation is largely
threshold information shared with the marginal baseline, not persona-level signal. And the
cross-tabs are computed under a conditional-independence assumption on **both** sides, so a
real within-person answer dependence would not show up in either.

### 3.10 (banked, Gate 4) There is a per-cell resolution limit no 8-dimensional trait model reaches

The correlation bar failed at 0.684 against 0.90. The exploratory attribution
(stage4_corr_attribution.json -> summary) telescopes the gap exactly:

| Rung | Correlation |
|---|---|
| Population marginal (zero information) | 0.554 |
| k-NN on the 15 interview answers | 0.662 |
| **Confirmatory model** | **0.684** |
| Full 202-item theta-hat + fitted held-out item parameters (ceiling of this model class) | **0.786** |
| Truth-side oracle (the analytic post-noise distribution) | 0.993 |

Attribution of the oracle-to-model gap: **66.9% the 8-dimensional model class, 18.2% the
N=15 interview length, 14.9% the zero-shot item encoder.** Closing everything the project
controls still lands 0.114 under the bar. The oracle at 0.993 proves the metric itself is
sound — the shortfall is real, not an artefact of the measure.

Two further readings from the same file. The pooled form is 81% saturated by item-level
structure the zero-information marginal already captures
(variance.what_the_marginal_already_captures.share_of_the_model_r_the_marginal_already_has
= 0.809; answer-frequency variance splits 32% between-item / 68% between-persona). And
predictions are near-unbiased: OLS slope 0.937, intercept 0.015
(flatness.per_arm.model.ols_empirical_on_predicted). **That is why ECE passes while
correlation fails — the arm is noisy, not miscalibrated.**

Future direction, exploratory and not built here: individual item-level quirks exist that
no low-dimensional trait vector expresses, and k-NN partially captures them on items near
the interview. A hybrid trait model plus collaborative-filtering residual is the obvious
next architecture.

### 3.11 (Gate 5) Adaptive interviewing halves the question count — and the value is question *selection*, not per-person branching

Because the frozen target sits below the floor, the efficiency result lives on the
exploratory grid that owner RULING 1 pre-declared before any Stage 5 compute existed.
Median questions to reach a given accuracy on held-out people, 51 replicates
(stage5_strategies.json -> exploratory.threshold_grid):

| Target (all 8 dims) | Random | Heuristic | RL policy | Heuristic / random | RL / random |
|---|---|---|---|---|---|
| RMSE <= 0.70 | 52 | 24 | 25 | **0.462** | 0.481 |
| RMSE <= 0.65 | 81 | 40 | 37 | **0.494** | 0.457 |
| RMSE <= 0.60 | 136 | 77 | 84 | 0.566 | 0.618 |

Both adaptive arms roughly halve random at the two tighter-but-attainable thresholds. At
0.60 the ratio slips past 0.5 for both — that row is shown because leaving it out would
flatter the result.

Fraction of full-bank information reached (same file,
exploratory.fraction_of_full_matrix):

| N | Random | Fixed script | Heuristic | RL policy |
|---|---|---|---|---|
| 5 | 0.187 | 0.254 | 0.370 | 0.354 |
| 10 | 0.317 | 0.426 | 0.593 | 0.572 |
| 15 | 0.419 | 0.508 | 0.684 | 0.652 |
| 25 | 0.565 | — | 0.787 | 0.801 |

**RL ties the heuristic.** RL-to-heuristic ratios, in the order of the threshold table
above (0.70 / 0.65 / 0.60): 1.04, 0.93, 1.09 — a tie in both directions, and the
never-trained info-gain heuristic is not beaten. Per the PRD a
match is acceptable and the heuristic winning would have been a legitimate finding.

**The interesting part is *how* the RL policy wins.** Deployed greedily, the trained policy
converges to a near-**static** 25-question order: 25 distinct items across 500 people, down
from 177 at initialisation (stage5_rl_proxy_watch.json -> checkpoints[0] and checkpoints[-1],
`n_distinct_items`). The info-gain heuristic, on the same people at the same horizon,
spreads across **118 distinct items**
(stage5_rl_proxy_watch.json -> gate5_harness_arms.heuristic.mean_distinct_items_used).

So on this population, the value of "adaptive" interviewing is **question selection, not
per-person branching**. A good fixed 25-question set captures nearly all of it. That is a
cheaper product than a live adaptive loop, and it is directly testable on real people.
(Past its training horizon the policy spreads out again — 152 distinct items by N=150.)

This lines up with the exploratory note filed at Stage 3: a D-optimal 15-item set was
projected to carry ~33% lift against the frozen set's ~23%. The frozen interview set stayed
frozen and the headroom went to the adaptive interviewer, where it turned out to be mostly
static headroom.

### 3.12 (Gate 5) The deployable reward costs nothing measurable against the forbidden oracle reward

Owner RULING 2 forbade truth-derived rewards for the confirmatory policy: planted truth
must never shape system-side weights, even through a reward on a training batch. The
confirmatory policy therefore trains on **declared posterior-variance reduction minus a
per-question cost** — a signal any deployed system has.

We also trained an exploratory arm with the forbidden reward: identical trainer, identical
seeds, reward replaced by truth-side squared-error reduction, weights quarantined behind a
loader that refuses to load them.

Result: the two policies land **within 0.024 RMSE of each other at every N**. The largest
gap is 0.0242 at N=5 — and it favours the *deployable* policy. At N=25, the training
horizon, the difference is 0.0029 (computed from stage5_rl_proxy_watch.json and
stage5_rl_exploratory_oracle_proxy_watch.json -> checkpoints[-1].truth_rmse_by_n).

**Confidence-driven interviewing needs no truth signal.** For a real product this is the
whole ballgame: you can train the thing that decides what to ask next without ever knowing
the right answer.

*(Number corrected in the log: a first report said +/-0.014; recomputation against the
watch files gave +/-0.024, and the result file wins. Commit c22f45a.)*

### 3.13 The proxy-gaming watch fired, was chased down, and came back clean

RULING 2 also pre-declared a watch: if declared blur shrinks fast while truth-side error
does not follow, that is proxy-reward hacking and a first-class finding, not a bug to hide.

**The watch fired before any RL existed.** On the non-RL arms, declared blur falls to 24% of
its starting value (heuristic) while truth-side error falls only to 55%
(stage5_strategies.json -> exploratory.blur_vs_truth). Training on the proxy then widened
the trained policy's own gap from +0.142 at initialisation to +0.239 at N=25
(stage5_rl_proxy_watch.json -> checkpoints).

That looks damning until you run the control. On identical sittings the **never-trained**
info-gain heuristic shows a *larger* gap than the trained policy: +0.254 against +0.239 at
N=25 (stage5_rl_proxy_watch.json -> gate5_harness_watch.heuristic and checkpoints[-1]). The
trained policy ends with the same error and *more* declared uncertainty than the heuristic.

Verdict: the gap is a property of the **posterior**, not of the policy. It is the
card-transmission blind spot of section 2 expressed inside the adaptive loop. RL adds
nothing malign. Stated precisely: **the proxy is honest about ordering decisions and
dishonest only about absolute confidence** — exactly as the pre-registered watch framed it.

### 3.14 The k-NN stratum pattern: the encoder earns its keep where it should

k-NN on the 15 interview answers is a strong baseline — 24.4% lift against the model's
27.1%. The pooled comparison is close. The stratum lens is what makes it interpretable
(stage4_gate4.json -> per_stratum_comparison):

| Probe stratum | Items | Model lift | k-NN lift | Winner |
|---|---|---|---|---|
| **near** the interview | 13 | 22.6% | **24.9%** | k-NN |
| **same domain** | 19 | **29.3%** | 26.6% | model |
| **far** from the interview | 18 | **28.0%** | 21.5% | model |

This is the opposite of the memorisation pattern. A model that only beat k-NN on items near
the interview would be reciting the transcript. This one loses near the interview — where
direct matching is best — and wins as questions move away, which is what recovering a trait
vector is supposed to buy. k-NN's advantage falls 3.4 points from near to far; the model's
does not fall at all.

### 3.15 Distractors carry real signal — the synthetic "crud factor"

Fifteen items in the bank were written to measure nothing. In the fit, 4 of them come out
with material discrimination: median 0.578 across the 15, top item q249 ("stairs versus
lift") at 1.694 loading on tech optimism, against a strong-item median of 1.998
(stage2_v2_recovery.json -> watch_list.distractors and weak_item_ordering).

A leakage hunt was run and verified: no planted value reaches the system side. The cause is
that a coherent biography colours even nominally neutral preferences. The owner's decision
was to document this as a finding rather than a defect, in these terms:

> "the synthetic analogue of the everything-correlates 'crud factor' in real survey data,
> where no human's 'neutral' preferences are truly orthogonal to their traits."

The standing leakage trigger stayed armed for every later stage: any distractor whose
discrimination grew materially, or any new route by which distractor answers could see
planted values, would have been a hunt. Neither happened.

---

## 4. Leakage discipline: four hunts, documented

The Wall rule (`PREREGISTRATION.md` section 3.5) is that when any result beats
expectations, the first action is a documented leakage hunt — not a celebration. Four
fired. All four are in `PROJECT_LOG.md`; here is what each found.

**Gate 2, hunt 1 — the TRU-RSK confound (real, and not a Wall breach).** Institution trust
and risk appetite were planted at exactly 0.00 correlation, deliberately, so recovery could
not lean on a correlation crutch. The fit recovered -0.270 on held-out people
(stage2_v2_recovery.json -> watch_list.planted_zero_pair). The hunt found the entanglement
in the **answers**, before any fitting: the model-free answer-score correlation is -0.375
(PROJECT_LOG, Gate 2 second attempt — recorded in the log's exploratory block and cited as
recorded; it is not persisted in a result JSON). The bank had been scrubbed for this pair, so the
confound entered at card generation — Gemma writes institution-trusters as financially
cautious people, with bank accounts and insurance. A card-writer confound, not system-side
leakage. It survived a full two-pass card rewrite (section 5.4), shrinking only ~14%.

**Gate 2, hunt 2 — the distractors.** Covered as finding 3.15. Verified clean: no planted
value reaches the system side.

**Gate 3, hunt 3 — the item encoder passed its cosine bar, so we attacked it.** Median
held-out cosine 0.861 is a strong number for a zero-shot head. The control: a
topic-domain-only feature set reaches 0.265, barely above the 0.258 a no-text constant
scores (stage3_item_encoder.json -> direction_arms.topic_only and constant_no_text). The
head is reading the item, not the topic tag. Clean.

**Gate 3, hunt 4 — two dimensions beat their own filed projection by more than 0.10.** TRU
by 0.114 and TEC by 0.138 (stage3_gate3.json -> leakage_hunt.trigger). Five checks were
run: split integrity (zero train/holdout overlap, matches the frozen manifest),
measurement-model scope (fitted on 400 training personas against a system-side target;
selection never saw the holdout), feature provenance, and two more. All clean. The
explanation is the open-answer channel, built for exactly this — and its contribution is
uneven across dimensions, worth 0.3 to 9.2 closed items depending on the axis.

**Gate 4 — no hunt fired.** The grader's own triggers (pooled correlation > 0.95, or lift
> 45%) did not trip. The failed bar was low, not suspiciously high. Nothing beat
expectations, which is itself the honest reason not to hunt.

---

## 5. Honest methods: every round that failed

This section is a feature. A pipeline this long produces wrong numbers on the way to right
ones, and the only defence against fooling yourself is writing the wrong ones down. Each
entry points at its `PROJECT_LOG.md` entry.

### 5.1 The question bank took four QA rounds and one full rebuild

*PROJECT_LOG, "Bank QA: fix round 1 FAILED verification" and "Question bank PLANTED after
four QA rounds".*

The first audit of the 272-item bank flagged **95 items** — sign errors, wrong dimensions,
near-duplicates. Fixes were applied and validated mechanically. Then an independent
adversarial verification pass, with a 25-item control sample of *untouched* items, returned
NO:

- 9 of the 95 fixes were still defective.
- The control sample found **6 of 25 untouched items defective (~24%)** — the first audit
  had a material false-negative rate.
- Systematic wording bias: environment items carrying price language, risk items set in
  tech contexts, tradition items leaning on paper-versus-digital. The bank would have
  **manufactured the very correlations the study has to prove it can recover** (ENV-PRC
  planted at -0.15, RSK-TEC at +0.40, TRD-TEC at -0.35).
- The same behavioural "vehicle" loaded on different dimensions in different items
  (cheapest-X on both environment and price; cash-versus-bank on both trust and risk).

The decision was not another patch round. It was a **full content rebuild against a vehicle
map**: each behavioural vehicle belongs to exactly one dimension, banned-context lists per
dimension, a correlation-bias budget, reverse-keyed items must use distinct vehicles.

Path to acceptance, honestly: initial generation -> 95 flagged -> fixes failed verification
-> full vehicle-map rebuild (101 items, 16 blocking residuals) -> scoped final fixes (~50
items, 2 new blocking) -> five micro-fixes -> PASS. Cumulative: ~115 of 272 items
rewritten, 14 retagged, design table byte-identical throughout.

Known instrument limitations were recorded at planting rather than discovered later —
including that items q229 and q234 might be non-monotonic. Both later fitted with flipped
signs exactly as predicted. Because they were flagged **before any fitting**, correcting
their designed labels was housekeeping, not a rescue; both are training items, so the
confirmatory held-out bar is bit-identical either way, and the grader gained an
`--exclude-items` robustness path that confirms it.

### 5.2 An invalid 98% test-retest number, caused by a seed-coupling bug

*PROJECT_LOG, "Pilot round 1: measurement bug found (seed coupling)".*

The first temperature pilot returned test-retest agreement of ~98% at both temperatures,
far above the frozen 70-90% band. **The number was invalid.** The batch driver passed one
global seed into every request's sampling parameters, and vLLM (verified in the installed
source) seeds a per-request generator from it — so the main round and the retest round drew
identical random numbers by construction. Accidental common-random-numbers coupling.

The fix was per-record seeds (a stable hash of base seed, process id, item, round and
temperature). It did real work: 3-4% of individual answers flipped. **The pooled number did
not move** — 0.9778 / 0.9789 / 0.9778 at three temperatures. The responder is genuinely
near-deterministic, and temperature is not the knob.

That is why the mechanical noise layer exists (section 5.5), and why the log records which
early signals were still safe to read: coupling pushes agreement *up*, so the diversity and
obedience bars passed conservatively.

### 5.3 A GPU memory leak burned a whole job

*PROJECT_LOG, "Pilot round 1"; COSTS.md, job 51171292.*

One pilot attempt died with orphaned workers after `os._exit`, leaking GPU memory between
driver invocations. **5.69 core-hours lost.** Fixed with a `free_gpus()` helper in the
sbatch pattern; the failure has not recurred.

### 5.4 The card confound survived the fix designed to remove it

*PROJECT_LOG, "Gate 2 fix round" and "Stage 2 second attempt".*

Gate 2 failed. The owner ordered a full card regeneration with a two-pass writer: pass 1
writes the biography under an explicit independence block (the eight sides are independent
facts; one side is never evidence of another; surprising combinations stay as given, with
behavioural contrast examples de-entangling institution-trust from financial caution in
both directions); pass 2 re-reads its own draft against the same guidance and rewrites.

It executed well: 500/500 drafts, 500/500 revisions, 489 biographies materially changed,
**zero leak rejects** on ingest against 6 in version 1, and 25% of raw answer cells changed.
The cards are genuinely different.

**It did not fix the confound.** TRU-RSK moved from -0.283 to -0.270 on held-out people;
the model-free answer correlation from -0.437 to -0.375. About a 14% reduction. And the
overall correlation structure got **worse** while individual traits got better: the slope of
recovered on planted off-diagonals fell from 0.71 to 0.650 and their correlation from 0.77
to 0.739 (computed from stage2_v2_recovery.json -> correlations.holdout.recovered against
correlations.planted, off-diagonal entries).

Gate 2's second attempt failed the same two bars. Per the owner's order it was a hard stop:
no third card round, no bar moved, no confirmatory number recalibrated. Cost of the fix
round: 18.52 core-hours for a 0.02 improvement in mean trait recovery.

### 5.5 A pre-registration reading that had to be argued, not assumed

*PROJECT_LOG, "Pilot round 2".*

The prompt-side wobble instruction measurably did nothing. Responder inconsistency was
instead injected by a seeded mechanical noise layer driven by the responder's own recorded
answer-token distribution.

This is recorded as a **design decision requiring owner ratification**, with the argument
written out rather than assumed: frozen section 2 defines wobble as "how often it flips or
shifts answers on items it is lukewarm about", and the layer implements that sentence
literally, with lukewarmness measured from the persona model itself. The band stayed the
frozen bar; only the layer's parameters were tuned, offline on recorded logprobs at zero
GPU cost, and recorded as an addendum in `PREREGISTRATION.md` section 5. Frozen at a = 1.2,
b = 4.0, T_noise = 16, temperature 0.7, seed base 548, and never touched again.

The honest consequence, also recorded: this is a mechanical noise layer, not human
psychology. See section 6.1.

### 5.6 A Wall breach found mid-stage and closed

*PROJECT_LOG, "Stage 2 run".*

`answers_noised.jsonl` carried a pre-noise "raw" answer field that was visible on the
system side. It was stripped, the layer was patched, and a **field-level** Wall test was
added on top of the existing import-level test. The Wall suite now parses every Python file
in the repo on every commit and also checks artifact fields.

### 5.7 A staging mistake that cost 4.60 core-hours

*PROJECT_LOG, "Stage 5: RL batch minted"; COSTS.md, job 51357407.*

The first RL-batch card job wrote all 500 pass-1 drafts, then the bridge rejected all 500
because the per-persona prompt records had never been staged to the cluster — pass 2 had no
input and exited. **4.60 core-hours spent for nothing.** The resume-safe driver made
recovery cheap: the retry skipped all 500 already-rendered drafts with no engine
initialisation and finished the job for 4.75 core-hours.

### 5.8 An RL objective that was provably order-blind, caught in a smoke run

*PROJECT_LOG, "Stage 5 run"; COSTS.md, abandoned run.*

The first confirmatory RL training used an undiscounted fixed-horizon return. The smoke run
exposed why that is broken: with an undiscounted return, every step's return equals the
total uncertainty removed minus whatever is left at the horizon, and the constant
per-question cost cancels out of the gradient — so **the objective is order-invariant**.
The policy learned a good 25-item *set* asked in arbitrary order, and matched random early
in the interview.

Discounting at gamma = 0.9 restores urgency. It changes how returns are aggregated, not
what the reward is, so RULING 2 (blur-only reward) is untouched. The gamma = 1 run was
killed at ~1,200 iterations; the 0.18 core-hours are logged because they were spent.

### 5.9 A number that was wrong in the log and got corrected

*PROJECT_LOG, commit c22f45a.*

The oracle-versus-deployable reward gap was first reported as +/-0.014. It failed
recomputation against the watch files and is +/-0.024. The log was corrected rather than
quietly edited, and the rule stands: **the result file wins.**

### 5.10 Two grader's-reading calls, both preserved in both directions

Two frozen bars are worded qualitatively, so the grader had to interpret them. In both
cases both readings are stored in the result file so the call can be re-made by anyone.

- **Gate 2 weak-item ordering** — "Designed weak items must show low fitted discrimination"
  has no number attached. The grader's rule (median weak discrimination below the 5th
  percentile of strong primaries) passes at 0.825; the full distributions are in the file,
  and 13 of the 15 weak items sit above the weakest strong item.
- **Gate 3 blur monotonicity** — "blur shrinks monotonically **in expectation**". The
  orchestrator ruled PASS on the "in expectation" wording: on the confirmatory fused track
  the mean-blur curves shrink by 0.176-0.300 with a worst uptick of 0.00228 (0.9% of that
  dimension's shrinkage) and 15.2% of individual steps rising; the closed-only track shrinks
  0.19-0.42, which is the range the log records. The fixed-theta control is exactly
  monotone, so the upticks are MAP relocation rather than information loss. The grader's two
  mechanical readings both record FAIL in the result file. Both are printed in the
  scoreboard above.

---

## 6. Limitations

### 6.1 The inconsistency is mechanical, not psychological

Real people are inconsistent because they are ambivalent, tired, primed by the last
question, or performing for an interviewer. Our people are inconsistent because a seeded
noise layer resamples their answer from their own flattened answer distribution, with an
event probability driven by how lukewarm they were on that item.

The layer was built precisely because the prompt-side approach did nothing (section 5.5)
and it lands the frozen test-retest band on the nose. But it is a **noise model, not a
psychology model**. It has no memory, no order effects, no interviewer effects, no
motivated reasoning. A real test-retest of 79% and our synthetic 79% are the same number
for different reasons.

### 6.2 Demographics carry zero trait signal here; on real people they do not

Finding 3.2 in reverse. Our public profiles are informationally empty about traits, which
made two things easy that would not be easy in reality. The lift-over-profile bar collapsed
into lift-over-nothing, and the LLM-with-only-demographics baseline had nothing to work
with. **On real humans a profile-only baseline would be genuinely strong, and the same bars
would be harder to clear.** Do not read finding 3.7 as "LLMs are bad at people"; read it as
"an LLM given a signal-free profile invents signal, and the invention costs you".

### 6.3 One persona model family, one system model

Everything on the persona side is Gemma 4 31B. Everything confirmatory on the system side
is Qwen3.6-27B. The Wall's family-split rule is satisfied, but the study has **n = 1 on
both sides**.

The card-transmission ceiling in particular is a measurement of *Gemma's* rendering
fidelity. A different writer, or a different prompt schedule, would move it — and the whole
downstream story with it. Nothing here establishes that ~85% (measured median 0.830, range
0.770-0.871 across dimensions) is a general property of LLM persona rendering. It is what
we measured, once, with one model, after one rewrite that moved it very little.

### 6.4 The trait model is 8-dimensional and linear, and we can see its edge

The Gate 4 correlation failure is the resolution limit made visible: 66.9% of the
oracle-to-model gap is the model class itself, and the full-information ceiling of that
class is 0.786 against a bar of 0.90 (finding 3.10). Item-level idiosyncrasy exists in this
population that no 8-number summary of a person can express.

Also: the recovered between-trait correlation structure is compressed. The slope of
recovered on planted off-diagonals is 0.650 (section 5.4). Individual traits recover better
than the relationships between them.

### 6.5 The interview is short, scripted, and closed-form-dominated

Confirmatory results use N = 15 closed questions plus 3 open answers, frozen at split time.
Gate 5's adaptive work is closed items only, on recorded distributions, by owner RULING 3 —
no open prompts in the adaptive loop. So the efficiency result is about **selecting scored
questions**, and says nothing about an interviewer that writes its own follow-ups.

### 6.6 Recovery study, not prediction of humans

Restating section 1 because it is the limitation that matters most. Every number here is
measured against truth we planted ourselves, in people we generated ourselves. **Nothing in
this report is evidence that the pipeline would recover anything from a real person.** The
two routes to that evidence — a pre-registered human panel, and a shallow-resolution
calibration replication on real survey responses — are specified in `HUMAN_PROTOCOL.md` at
the repository root. They are a separate deliverable and are not tested here.

### 6.7 Known instrument defects that were accepted, not fixed

Recorded so nobody rediscovers them as findings. Item q246 is answered "3" by all 500
personas raw and is content-free. Item q044 is near-dead (488 of 500 answer "no" raw) and
is a designed-strong item, so it sits near the bottom of the strong-primary distribution:
it fits 0.5713, eleventh-lowest of all 252 items, though the strong-primary minimum of
0.4498 belongs to a different item. Four price-sensitivity
items share one syntactic frame — a possible method factor. Locality has a six-item "would
you live elsewhere" cluster, within-dimension redundancy. Demographic combinations in the
cards are occasionally odd (a 79-year-old barista with children at home); no frozen bar
tests demographic realism.

---

## 7. Costs

| Stage | GPU core-hours | Notes |
|---|---|---|
| 0 — setup and smoke tests | 5.74 | Gemma smoke test on Leonardo |
| 1 — population, bank, pilots, full sweep | 49.06 | includes 5.69 lost to the GPU memory leak (5.3) |
| 2 — static recovery | 0 | MIRT fit is pure numpy on local CPU, 59 s |
| 2-fix — card regeneration v2 + full sweep v2 | 18.52 | the fix round that did not fix it (5.4) |
| 3 — translators | 23.39 | open answers 4.28, Qwen smoke + embeddings 10.05, judgment batch 9.06 |
| 4 — end-to-end prediction | 7.12 | the LLM-no-interview baseline job only; everything else local CPU |
| 5 — adaptive interviewing + RL | 20.00 | includes 4.60 lost to the staging mistake (5.7) |
| **GPU total** | **123.83** | summed from COSTS.md |
| CPU (Stage 5 episodes, RL training, grading) | 5.35 | 1.5 first Gate-5 strategy sweep + 1.3 RL training (incl. 0.18 for the abandoned gamma=1 run) + 2.55 Gate-5 re-grade with the RL arm |
| **Project total** | **~129.2 core-hours** | |

*(The CPU row is summed from every CPU line in COSTS.md. An earlier figure of ~3.9 left out
the 1.5 core-hours of the first Gate-5 strategy sweep, which the later re-grade reproduced
bit-exactly but did not un-spend. Same rule as everywhere else: compute that was spent is
counted.)*

**API spend: about $0.02 for the entire project.** One question-bank generation run on
`gemini-3.5-flash-lite` (~44k input tokens) plus two trivial verification calls (COSTS.md).
Every confirmatory number was produced on Leonardo or on local CPU.

Two things worth noting for anyone budgeting a similar study. First, the largest single
cost is not the science — it is regenerating a population (18.52 core-hours for the v2
cards and sweep, which changed almost nothing). Second, about 10.3 core-hours, roughly 8%
of GPU spend, went to two operational mistakes that were both preventable and both logged.
