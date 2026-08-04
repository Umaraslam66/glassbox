# REPORT — GLASSBOX-Mobility

Stage M5 report, drafted at Gate M5 (2026-08-04). Format per the frozen
reporting spec: scoreboard first, fails first, verbatim frozen bars,
limitations, costs. Every number below is a convenience copy of a committed
result file, named at each table; result files win over this prose.
Preregistration frozen at commit `57bfcfb` (2026-08-03); bars never moved.

**The claim tested (prereg §1, verbatim in spirit):** standard travel-demand
estimation recovers planted traveler parameters from the choices of LLM
travel agents — or it measurably does not, and we say where it breaks.

**One-sentence result: it measurably does not, and we can say exactly where
it breaks — while the same agent population stays believable, probabilistically
calibrated, and aggregate-realistic on published elasticity ranges.**

---

## 1. Scoreboard — every frozen bar, fails first

### FAILED (10 bars)

| # | Stage | Frozen bar (verbatim) | Measured | File |
|---|---|---|---|---|
| 1 | M2 | per-dimension recovery r(φ̂_d, φ_d) ≥ 0.75 on held-out travelers, all 6 dimensions | 0.285–0.603, median 0.427 — **all six fail** | `m2_recovery.json` |
| 2 | M2 | VOT in real units: median absolute error ≤ 35% against planted €/h | **112.2%** | `m2_recovery.json` |
| 3 | M2 | uncertainty coverage 60–75% | **16.7%** (bootstrap 3–9× overconfident) | `m2_recovery.json` |
| 4 | M2 | designed-zero separability \|r(φ̂_VOT, φ̂_PRC)\| ≤ 0.15 | **−0.343** | `m2_recovery.json` |
| 5 | M2 | distractor tripwire: every distractor × dimension \|r\| ≤ 0.15 | **0.222** → mandated hunt ran (§4b) | `m2_recovery.json`, `m2_diagnostics.json` |
| 6 | M3 | Brier lift ≥ 25% over the population-marginal baseline (K = 12 + profile) | **+7.0%** | `m3_calibration.json` |
| 7 | M3 | coverage 60–75% | **30.4%**, decaying 33.3% → 14.4% as K grows (§4d) | `m3_calibration.json` |
| 8 | M4 | peak spreading S₂₀ ≤ 0.80·S₁ | S₁ 0.2633, S₂₀ 0.2933, **ratio 1.114** (§4e) | `m4_bars.json` |
| 9 | M4 | Spearman r(W_i, HAB_i) ≤ −0.3 | **−0.188** (disattenuated −0.322; ceiling-inherited per Ruling 30) | `m4_bars.json` |
| 10 | M1 | diversity (a): no traveler pair with exact-match agreement > 95% on the full bank | **0.9714** — exactly one pair (t0009–t0361, planted distance 2.34; Ruling 21: recorded as FAIL, population untouched) | `m1_qa_r2_full.json` |

The reported-only 25% VOT line (never a gate condition, Ruling 7) also fails.

Also on the permanent record: the **first full-bank obedience gate FAILED**
at median 0.4761 < 0.5 (`m1_qa.json`, Ruling 16). The single authorized
improvement round (Ruling 19: decision-form cards, CRW/HAB scenario
sharpening, documented before rendering) lifted it to the pass below; both
rounds and their costs stand in the log.

### PASSED (6 bars)

| # | Stage | Frozen bar (verbatim) | Measured | File |
|---|---|---|---|---|
| 1 | M1 | obedience: median across the 6 dimensions ≥ 0.5 (pre-noise, full bank) | **0.5922** (round 2; round-1 fail 0.4761 on record) | `m1_qa_r2.json` |
| 2 | M1 | diversity (b): median pairwise agreement ≤ 0.80 | **0.600** | `m1_qa_r2_full.json` |
| 3 | M1 | diversity (c): participation ratio ≥ 4 | **22.15** | `m1_qa_r2_full.json` |
| 4 | M1 | test–retest pooled agreement in 70–90% | **79.8%** (2-option 80.4% vs chance 50%; 3-option 74.0% vs 33%) | `m1_qa_r2_full.json` |
| 5 | M3 | calibration ECE ≤ 0.05 | **0.0473** | `m3_calibration.json` |
| 6 | M4 | Spearman r(X_i, SCH_i) ≤ −0.3 | **−0.361** (disattenuated −0.743) | `m4_bars.json` |

---

## 2. The decomposition tables (declared primary presentation)

Card transmission, measured at M1 with the frozen §5 estimator
(`m1_qa_r2_full.json`): the cards carry a hard ceiling of **median ρ̂ 0.609**
against the parent's 0.83 reference — every dimension below the parent's
worst (0.77).

| | VOT | SCH | MOD | CRW | PRC | HAB |
|---|---|---|---|---|---|---|
| transmission ρ̂ (M1) | 0.670 | 0.486 | 0.733 | 0.634 | 0.516 | 0.585 |
| recovery r (M2, held-out) | 0.447 | 0.406 | 0.320 | 0.603 | 0.285 | 0.591 |
| disattenuated r/ρ̂ (cap 1) | 0.668 | 0.836 | 0.436 | 0.951 | 0.553 | 1.000 |

Reading (Rulings 23/24): SCH, CRW and HAB misses are predominantly
**ceiling-inherited** — the choices carry nearly all the signal the cards
transmit. VOT, MOD and PRC lose heavily **beyond** the ceiling — estimator-side.

**VOT in real units** (`m2_recovery.json`): measured error 112.2%;
ceiling-implied error at ρ̂_VOT = 0.670 is 31.7%; **estimator overhead
+80.5 points — the estimator, not the cards, dominates this miss.** The
correction removes the ceiling's systematic component only.

**M4 coherence decomposition** (Ruling 30, declared before any M4 compute):
HAB −0.188 disattenuates to −0.322 — the dynamics carry a bar-clearing
relationship that the card ceiling attenuates below the bar. SCH −0.361
passes outright and disattenuates to −0.743.

---

## 3. Baselines (frozen §5 list, complete)

Held-out choice prediction, Brier vs the traveler's noise-derived true
distribution, identical cells (`m3_calibration.json`):

| Predictor | Brier | Lift vs marginal |
|---|---|---|
| population marginal (zero information) | 0.1930 | — |
| **k-NN over observed choices (k = 20)** | **0.1278** | **+33.8%** |
| structural K = 12 + profile (the frozen estimator; bar-carrying) | 0.1796 | +7.0% |
| profile-only shares | 0.2205 | −14.2% |
| LLM given profile only (Ruling 29, Gemini) | 0.3217 | **−66.7%** |

Behavioral similarity beats structural estimation roughly five-fold. Both
negative controls hold: the designed signal-free profile helps nobody, and
an LLM reading it invents structure and lands far below zero information
(the parent's finding 5, replicated and amplified).

---

## 4. Findings (registered at their gates; Rulings 25, 27, 31, 35)

**a. Believable but not estimable.** The fitted structural models predict
choices well (in-sample 0.804, held-out scenarios 0.767, vs 0.714 zero
information) and their choice probabilities are nearly honest (ECE 0.0473 —
a passing frozen bar), while every parameter-recovery bar fails. Surface
believability and internal validity are different properties, measured apart.

**b. Persona-everywhere (the tripwire hunt verdict).** Pre-noise distractor
choices correlate with planted traits in 9 of 10 designed-zero scenarios
(max |r| 0.383: sleep-in-vs-early-swim × SCH; the dentist-slot scenario ×
HAB +0.213). Real renderer leakage, not estimation artifact, not a Wall
incident: for LLM travelers no scenario is behaviorally neutral, and
construct-audits of scenario text cannot catch it (`m2_diagnostics.json`).

**c. Near-deterministic rendering starves maximum likelihood.** 301/400
travelers hit ridge-limited constants, 184/400 fit with positive time
coefficients (buffer-buying and texture preferences absorbed into β_time),
half the held-out travelers have degenerate VOT ratios; the VOT–PRC
entanglement (−0.343) is mechanical, through the shared β̂_cost. Ruling-26
exploratory comparison: ridge λ = 1.0 lifts median r 0.427 → 0.507 but
worsens real-unit VOT (112% → 166%); the designed-score oracle reaches
0.600 from the same choices — **specification misfit, not estimation
variance, binds.**

**d. Coverage-decay generalizes across domains.** The parent's phenomenon
reappears exactly: pooled coverage falls monotonically 33.3% → 14.4% as
evidence grows K = 4 → 104 while recovery r rises (`m3_calibration.json`).
More data makes the estimator more confidently wrong about LLM agents.

**e. The metering artifact (methods lesson).** The frozen peak bar anchors
on day 1, but day-1's saturated bottleneck meters arrivals at capacity —
S₁ = 0.2633 equals the capacity ceiling (5.25/min × 15 / 300 = 0.2625)
almost exactly, so the anchor measured the queue, not behavior. From day 2
(once 19% of travelers found the ring road and the metering broke) the peak
fell monotonically 0.387 → 0.293, −24.3% — exploratory context under
Ruling 32's footnote, never a re-grade; the frozen bar stays FAILED
(`m4_bars.json`).

**f. SCH asymmetry.** Schedule rigidity is the weakest static transmitter
(0.486) but the strongest dynamic performer (coherence pass, quartile
exposure 0.445 → 0.119): repeated consequential days elicit what one-shot
scenarios cannot.

**g. The ratchet tail.** 7 of 300 travelers kept taking "leave 15 minutes
earlier" morning after morning, drifting to departures as early as 03:55 —
LLM persona caricature under repetition, findable in the viewer's notable
shortlist.

**h. Aggregate realism holds while individual recovery fails — the headline
contrast, complete at every level** (`m4_elasticity.json`,
`m3_calibration.json`): see §5.

---

## 5. Elasticities (REPORTED, never bar-graded)

By-construction caveat (frozen §6, verbatim): the population's planted
VOT/PRC distributions partly determine these numbers; the comparison is a
realism check, never recovery evidence.

Dynamic re-ask experiments, 300 travelers, both arms pre-noise
(`m4_elasticity.json`):

| Experiment | Result | Published range | Verdict |
|---|---|---|---|
| transit fare −20% | arc **−0.327** | −0.2 to −0.5 (TRL 2004; Litman) | inside |
| cordon charge €4 | car share **−9.9%**, arc −0.186 | −10 to −35% car trips (London/Stockholm) | at the edge |
| uniform cost +10% | car arc **−0.471** (transit cross +0.19) | −0.1 to −0.4 (Goodwin et al. 2004) | slightly beyond |

Static probes from the cached bank (`m3_calibration.json`): parking-cost
demand curve log-log slope −0.201 (inside −0.1..−0.4), toll take-up arc
−0.166 (inside), subsidy uptake correctly ordered. Aggregate share
prediction: population marginals alone reach r 0.977 against held-out share
cells; even the failed individual models track aggregates at 0.887–0.920.
**Aggregates are easy; individuals are hard** — GATSim-style believability
plus macro realism coexist with a complete recovery failure underneath.

---

## 6. Limitations

1. **Transit-fare static gap (Ruling 28).** The design table's fare-change
   probe was never built into the static bank; the fare elasticity above is
   a dynamic-probe result and labeled as such. The static gap stands.
2. **Diversity bar (a) failed** by one pair on a 140-item discrete
   instrument (Ruling 21: recorded verbatim, population untouched).
3. **Two-route config vs memo (a)** (Ruling 32 footnote, verbatim): "Memo
   (a)'s peak-spreading attainability argument assumed a single bottleneck;
   the ratified two-route world adds an escape path. Any peak-spreading
   result is interpreted under the two-route config, and the report notes
   this config difference from memo (a)."
4. **Public profiles are designed signal-free** (parent rule); real
   demographics carry signal, so profile-baseline conclusions transfer to
   real surveys only directionally.
5. The frozen utility has no term for the Ruling-10 PRC scenarios'
   friction attribute (declared pre-fit); PRC overhead partially reflects
   this specification gap.
6. HAB's frozen anchor names no numeric scale; its φ̂ scale was set by
   training-population standardization (pre-declared; correlations
   unaffected, coverage interpretation for HAB is weaker).
7. Elasticity comparisons are partly by construction (§5 caveat) and the
   M4 world is a stylized two-route corridor, not a network.

---

## 7. Wall record

Both wall tests green on every commit (parent 10, mobility 6). No Wall
incidents in any stage. The tripwire event at Gate M2 was renderer-level
leakage of persona into designed-zero scenarios — planted truth never
crossed to the system side; the mandated hunt and verdict are in
`m2_diagnostics.json` and the Gate M2 log entry. The viewer reads a
system-side log whose fields were checked at Gate M5: exactly
{pid, day, route, dep_min, travel_min, arrive_min}, nothing else — PASS.

---

## 8. Costs, seeds, provenance

**Total spend: ≈ $2.34** (COSTS_MOBILITY.md, logged at run time, never
retroactively): M0 ≈ $0.016; M1 ≈ $1.79 (two card rounds, four audit
rounds, two obedience comparisons, two full 56k-answer sweeps — the failed
round is on the record); Ruling-29 baseline $0.142; M4 $0.39 (day-loop
$0.111 + elasticity re-asks $0.278). Zero compute clusters (Leonardo in
maintenance throughout). Traveler side qwen3.7-flash (reasoning-off for
answers, 0 reasoning tokens across 88k+ answer calls); system side Gemini
flash-lite; families never mixed.

**Seeds:** mint 3105 (five failed attempts logged); bank 3200; obedience
draws 2141/2142; wobble 3160; retest 3170; split-half 3180; noise 3190;
M2 splits 3210; bootstrap 3220; M3 K-draw 3230; Laplace 3240; M4 draw
3250; PAT 3260. Population re-creatable from configs + seeds by a stranger.

**Key commits:** freeze `57bfcfb`; M1 gate pass `f3227ec`; M1 close
`a4eac98`; M2 pre-declaration `635b3b1`; M2 `2a84378`; M3 pre-declaration
`40d3a2a`; M3 `84dc792`; M4 pre-declarations `4f1b741`/`e837cbf`; M4 pilot
`a448f25`; M4 `b92b65a`. Rulings 1–35 in PROJECT_LOG_MOBILITY.md.

**Viewer:** `mobility/app/` — `cd mobility/app && python -m http.server
8000`, load `data/runs/m4_full/trajectories.jsonl`. Screenshots:
`results/viewer_pane1_day1.jpg`, `viewer_pane1_day20.jpg`,
`viewer_pane2_peak.jpg`, `viewer_pane3_inspector.jpg`.
