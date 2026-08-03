# Reachability memos — freeze precondition (Gate M0 review, 2026-08-03)

Ordered by the owner before any freeze: for three draft bars, a mechanistic
argument — no new data, no API calls — that the bar is attainable but not
trivially guaranteed. Rule: if any memo shows a bar unreachable or trivially
guaranteed, stop without freezing.

**Verdicts: (a) PASSES. (b) PASSES. (c) FAILS — not shown reachable. The
freeze is therefore STOPPED per the owner's precondition 2.**

All numbers below are derived from the approved planted design
(PREREGISTRATION_MOBILITY.md §2: VOT = 12·e^(0.55·φ) €/h; SCH–HAB r = +0.35;
VOT–SCH r = +0.25) and the parent's published measurements (card-transmission
median 0.830, range 0.770–0.871). The Monte Carlo used to check the
attenuation algebra in memo (c) samples only from these declared
distributions; it touches no study data and no model.

---

## Memo (a) — Peak spreading: ≥ 20% relative fall by day 20

**Bar:** S₂₀ ≤ 0.80·S₁, where Sₜ = the maximum share of day-t arrivals in any
sliding 15-minute window.

**Intended M4 configuration argued against** (fixed here so the argument has
teeth): one Vickrey-style bottleneck, 300 travelers, free-flow 18 min,
preferred arrival times ~N(08:30, 12 min), 20 days, agents remember their own
experienced times over the last 3 days. Bottleneck capacity is calibrated at
design time — from planted parameters only, before any agent runs — so the
day-1 naive departure profile produces a maximum queueing delay of 12–18
minutes.

**Why attainable.** Day 1, travelers depart naively (preferred arrival minus
free-flow), so arrivals compress into roughly 45–60 minutes and S₁ lands
near 0.30–0.45. A ~15-minute peak delay costs the median traveler ≈ €3.00
(median VOT €12/h). Escaping it needs a 15–25 minute departure shift. For a
flexible traveler (SCH < 0, early-arrival penalty of order €3/h) that shift
costs under €1–1.5, and for a low-habit traveler (HAB < 0) the switching
threshold is small — so the shift pays. Under the planted joint distribution,
P(SCH < 0 and HAB < 0) = 0.307 (bivariate normal with r = +0.35), and
travelers with only one of the two below zero shift partially. If 25–40% of
peak-window arrivals leave the window, the sliding-maximum share falls by
roughly 20–35% — the bar sits inside that range, toward its cautious end.

**Why not trivially guaranteed.** (i) If the capacity calibration misses low
(peak delay ~5 min ≈ €1), the incentive drops under most travelers' movement
thresholds and the fall is <10% — an honest miss. (ii) The high-HAB half of
the population anchors in place; the strongly movable mass is only ~31%.
(iii) Bounded 3-day memory can produce overshoot-and-return oscillation
rather than monotone spreading. (iv) The deepest risk is the one the bar
exists to detect: if the LLM renderer does not act on the numeric time and
delay attributes, nothing moves, and the bar fails — which would be a core
finding, not a design accident.

**Verdict: attainable, not trivially guaranteed. PASSES.**

---

## Memo (b) — M4 behavioral coherence: Spearman |r| ≥ 0.3 for HAB and SCH

**Bars:** across the ~300 travelers, (i) day-to-day switching frequency vs
planted HAB: r ≤ −0.3; (ii) lateness-risk exposure taken vs planted SCH:
r ≤ −0.3. With n = 300 the SE of a Spearman r near 0.3 is ≈ 0.058, so the
bar sits ≥ 5 SEs from zero — it tests signal, not luck.

**Why attainable (HAB).** Switching frequency is the direct behavioral trace
of HAB's planted anchor (the switching penalty). Each traveler contributes
19 day-transitions, so the per-traveler switching count is a fairly reliable
index (aggregation washes single-day noise). If cards transmit HAB near the
parent's ~0.8 and the penalty governs choice, planted HAB explains roughly
25–50% of switching-propensity variance, putting the raw correlation at
0.5–0.7 before confounds. The main confound — travelers who start on the
better route have no reason to switch regardless of HAB — adds
HAB-independent variance and deflates the expectation to roughly 0.35–0.55.
That clears 0.3 with real, but not absurd, margin.

**Why attainable (SCH).** Lateness-risk exposure (share of days departing
inside the risk band implied by the previous day's times) is governed by the
late-penalty ratio, SCH's anchor (1–8×). It shares variance with VOT
(planted r = +0.25) and with queue position, so its expected correlation is
lower than HAB's: roughly 0.3–0.5.

**Why not trivially guaranteed.** Fast equilibration would concentrate all
switching in days 2–6 and range-restrict the index; the initial-assignment
confound could be larger than sketched if congestion incentives dominate
preferences; and HAB/SCH transmission could land at the parent's weak end
(~0.77), shaving every number above. SCH in particular has little slack —
if the exposure index is poorly chosen, 0.3 fails. The index definition must
be fixed in the M4 experiment config before any simulated day runs.

**Verdict: attainable, not trivially guaranteed. PASSES.**

---

## Memo (c) — VOT median absolute error ≤ 25% in €/h

**Bar:** median over held-out travelers of |VOT̂ − VOT|/VOT ≤ 0.25, with
VOT̂ = β̂_time/β̂_cost from the per-traveler conditional logit. Under the
approved anchor VOT = 12·e^(0.55·φ), a 25% error corresponds to a latent
error |φ̂ − φ| = ln(1.25)/0.55 = 0.406, i.e. a total error SD budget of
0.406/0.6745 = **0.602** (median of |N(0,σ)| = 0.6745σ).

**The error chain.** Planted φ → card (the parent-measured transmission
ceiling: per-dimension correlation median 0.830, worst 0.770) → choices
(mechanical noise tuned to the frozen 70–90% test–retest band) →
per-traveler MNL estimate → the ratio β̂_time/β̂_cost.

**The number that decides the memo.** Model the ceiling as
φ_card = ρ·φ + √(1−ρ²)·ε with ρ = 0.83. Even a perfect estimator of what
the card contains cannot beat this step. Verified numerically (2M draws):

| Transmission ρ | Ceiling-only median VOT error | SD room left for ALL estimation error |
|---|---|---|
| 0.83, shrinkage estimator | 23.0% | 0.225 |
| 0.83, plain per-traveler MLE (the pre-specified estimator) | 24.2% | 0.148 |
| 0.75 (parent's worst dimension) | 27.8–30.0% | none — bar broken with zero estimation noise |
| 0.90 | 17.5% | 0.414 |

A per-traveler MNL over ~112 discrete choices, with the noise layer flipping
answers at the rate the retest band demands and with the ratio's instability
for low-|β_cost| travelers, plausibly carries a latent estimation SE of
0.2–0.35. The bar therefore requires **either** estimation noise at or below
the extreme optimistic edge (≤ 0.148–0.225), **or** mobility card
transmission materially better than the parent ever measured (ρ ≳ 0.90).
Neither can be asserted before data exists; the second is exactly the kind
of hope a freeze must not encode.

**Contrast, for calibration of judgment:** the companion Gate M2 bar
r(φ̂_VOT, φ_VOT) ≥ 0.75 IS consistent with the same assumptions — expected
r ≈ 0.83/√(1+SE²) = 0.77–0.81 for SE 0.2–0.4 — attainable, not trivial. The
25% real-units bar is the only one that the transferred ceiling arithmetic
cannot support.

**Verdict: NOT SHOWN REACHABLE under the stated assumptions. FAILS the memo
standard — freeze stopped.**
