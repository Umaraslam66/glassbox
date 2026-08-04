# PROJECT LOG — GLASSBOX-Mobility

Chronological map. A cold session reads this first. Entries point at result
files; result files win over log prose. Numbers quoted here are convenience
copies, never the source of truth.

---

## 2026-08-03 — Stage M0: foundations

**Scaffold.** `mobility/` created per PRD_MOBILITY.md §2: `src/` (travelers,
scenarios, engine, model, eval, policy), `experiments/`, `results/`, `app/`,
`data/` (gitignored), `tests/`. `PRD_MOBILITY.md` moved from the repo root
into `mobility/` per its own §2 layout. Root README got its single pointer
line.

**Wall.** `mobility/tests/test_wall_mobility.py` extends the parent's static
pattern to the mobility truth directory: import/name check on the truth-vault
accessor (`mobility/src/eval/truth_vault.py`, the only permitted reader),
truth-path spelling check with mobility-aware exemptions, raw-material check
on `mobility/data/runs/`. Both wall tests green (parent 10 passed; mobility
4 passed + 1 skip for the not-yet-existing data dir).

Two constraints the parent Wall test imposes on all mobility code, discovered
and designed around at scaffold time: (1) no Python file in the repo may
mention answer-token log-probabilities — which is why the M0 availability
pilot lives outside the repo (session scratchpad, `pilot_m0.py`) and why a
mechanism that consumed them would need an owner decision before any mobility
code could parse them; (2) no mobility file may spell the truth-directory
path as a literal — the accessor assembles it from parts.

**Leonardo availability (PRD §7).** Verified unreachable 2026-08-03 ~14:45
CEST: SSH to the login node is closed pre-authentication ("Connection closed
by 131.175.44.5"), consistent with the announced Aug 3–17 Booster
maintenance. Plan B (OpenRouter qwen3.7-flash traveler-side) is in effect.

**M0 availability pilot.** OpenRouter `qwen/qwen3.7-flash` and Gemini 3.5
flash-lite probed for throughput, rate limits, cost, reasoning-token spend,
answer-token log-probability exposure, and repeated-sample disagreement.
Numbers: see `results/m0_pilot_summary.json` (committed) and
COSTS_MOBILITY.md for spend. Raw per-request records stayed in the session
scratchpad (they contain pilot fixtures, not study data). Headline verdicts:
qwen answer-token distributions are degenerate with reasoning on (the model
decides during its hidden reasoning; answer token lands at probability ≈ 1)
but carry real spread with reasoning off, where answers are also ~25× cheaper
and ~20× faster; qwen 429s come from an upstream shared pool, not our key;
Gemini refuses log-probability requests outright. The noise-mechanism
decision in PREREGISTRATION_MOBILITY.md §8 is written around exactly these
three facts.

**Drafts for owner sign-off.** `PREREGISTRATION_MOBILITY.md` written, marked
DRAFT: 6 traveler dimensions + correlation matrix (verified positive
definite, smallest conditional SD 0.844), scenario-bank design table,
all gate bars from PRD §5, model assignment, noise-mechanism proposal
contingent on the pilot's log-probability finding. Nothing frozen.

**Gate M0: HARD STOP.** Awaiting owner decisions: (1) the 6 dimensions +
correlation matrix + scenario-bank design, (2) the preregistration freeze
(bars, splits, model assignment, noise mechanism).

## 2026-08-03 — Gate M0 review executed; freeze STOPPED by reachability memo (c)

**Owner decisions recorded.** (1) Parameters, correlation matrix, and
scenario-bank design APPROVED as drafted. (2) Model assignment APPROVED:
Qwen traveler side / Gemini system side; any move to Gemma is stop-and-ask.
(3) Answering regime Proposal A APPROVED: cards reasoning-on, answers
reasoning-off, parent noise mechanism ported — conditional on the obedience
comparison now pre-declared in full inside PREREGISTRATION_MOBILITY.md §8
(60 travelers by seed 2141, 24 scenarios by stratified seed 2142, numeric
rule frozen-form). Repo stays public; nothing is pushed without an explicit
owner order.

**Ruling 4 (word-ban workaround).** No string-assembly of the banned
token-distribution keyword anywhere in mobility code. The API field names
live in `mobility/config/api_fields.json`; code reads them from config, with
a comment at the read site (`mobility/tools/m0_pilot.py`) and a sentence in
the preregistration §8. The parent Wall test stays untouched and green.

**Ruling 5 (git).** `PRD_MOBILITY.md` removed from version control and kept
local (`mobility/.gitignore`), matching the parent PRD's treatment. Because
the two Stage M0 commits existed only locally, they were rebuilt without the
file rather than rewritten later — commit hashes changed accordingly; the
old hashes existed on no remote. Owner's uncommitted `.claude/CLAUDE.md`
edit left untouched.

**Ruling 6 (pilot provenance).** The pilot recreated as
`mobility/tools/m0_pilot.py` using the Ruling-4 config approach, committed
with `mobility/tests/test_m0_pilot_tool.py`: offline tests hold the tool to
the shape of `results/m0_pilot_summary.json` (fields and sanity, not
numbers), and the opt-in live smoke (`GLASSBOX_M0_SMOKE=1`) passed 5/5
(≈$0.002, in COSTS_MOBILITY.md). The M0 record's numbers stand.

**Freeze preconditions.** Three reachability memos written to
`mobility/docs/REACHABILITY_MEMOS.md` (mechanistic arguments only, no new
data, no API calls). Verdicts: (a) peak spreading — attainable, not trivial,
PASSES; (b) M4 coherence — attainable, not trivial, PASSES; (c) VOT median
error ≤ 25% — **NOT SHOWN REACHABLE**: at the parent-measured transmission
ceiling (median 0.83) the card-writing step alone produces 23.0–24.2% median
VOT error, leaving an estimation-error budget (SD 0.148–0.225 latent) below
any plausible per-traveler MNL noise level; at the parent's worst dimension
(0.77–0.75 range) the ceiling alone breaks the bar. Full arithmetic in the
memo.

**Consequence, per the owner's precondition 2: STOPPED.**
PREREGISTRATION_MOBILITY.md remains DRAFT — nothing frozen, no freeze
commit, no hash to record. Stage M1 free steps (population sampling) NOT
started: they were gated on a successful freeze. Awaiting the owner's ruling
on the VOT bar before anything proceeds.

## 2026-08-03 — Rulings 7–8, FREEZE, Stage M1 mint

**Ruling 7 (VOT bar) applied.** Frozen bar: median VOT absolute error ≤ 35%;
the 25% level retained as a REPORTED scoreboard threshold, never a gate
condition; mobility card transmission measured at M1 (formula in prereg §5)
and the transmission decomposition declared as the primary presentation of
the VOT result; the owner's verbatim sentence recorded in prereg §6. Memo
(c) stands unchanged in `docs/REACHABILITY_MEMOS.md`. Interpretation note,
surfaced at Gate M1a: the ruling's text said "Gate M3 bar," but the one VOT
median-error bar lives in Gate M2 (PRD §5, structural recovery) — applied
there; a move between gates would be an owner-ordered amendment.

**Ruling 8 applied.** Exact formulas for the M4 switching-frequency index
W_i and the lateness-risk exposure index X_i (buffer b = 5 min) written into
prereg §5; the Gate M4 coherence bars now reference them.

**FREEZE.** `PREREGISTRATION_MOBILITY.md` marked FROZEN and committed:
**commit `57bfcfb` (57bfcfba750e20cadc40a4f5ec3c0b32f0b13e88)**. Test counts
at that commit: parent Wall test 10 passed; mobility suite 8 passed +
2 expected skips (no data dir yet; live smoke opt-in); full parent suite
726 passed. Bars never move from here.

**Stage M1 free steps (no card rendering, no scenario answering, zero API
spend).** Population minted per the frozen design. The pure sampler lives in
`src/travelers/sampler.py` (public design constants, no paths); the
grader-side mint (`src/eval/mint.py`, the vault's only writer) deposits φ
and grades the realized draw against the frozen §2 mint tolerances, emitting
aggregates only to `results/m1_mint_qa.json` — the source of truth for every
number below.

**Mint attempt log (frozen sequential-seed policy):** seed 3100 failed
(VOT SD 1.0341 > 1.02), 3101 failed (HAB SD 1.0209), 3102 failed
(corr dev + SD), 3103 failed (corr dev + designed zero + SD), 3104 failed
(corr dev + designed zero), **3105 PASSED all seven checks and was
deposited** (400 travelers, `phi_m1.npz` in the vault, prereg commit 57bfcfb
stamped inside). The [0.90, 1.02] SD band proved the binding tolerance —
joint pass rate ≈ 1 in 6 draws — which is the tolerance doing its job, not a
defect; all attempts recorded here and in `experiments/m1_mint.json`.

**Headline QA of the accepted draw (convenience copies; file wins):** max
absolute correlation deviation from planted 0.0445; designed zeros realized
at VOT–PRC −0.004 and CRW–PRC −0.021; VOT median €11.71/h (p5 €4.82,
p95 €26.83); φ-level participation ratio 5.11; minimum pairwise traveler
distance 0.44 (no duplicates). Choice-based diversity bars are graded at
Gate M1 proper, after rendering is authorized.

## 2026-08-03 — Ruling 9 + Stage M1 proper (cards, bank, obedience comparison)

**Ruling 9 logged:** the VOT bar (35% frozen / 25% reported) is confirmed at
**Gate M2**; the "M3" label in Ruling 7 was the owner's labeling error,
corrected at Gate M1a. No numeric change.

**Execution-order note, surfaced at the gate:** the owner's authorization
listed the obedience comparison (step 2) before the bank build (step 4), but
the frozen prereg draws the comparison's 24 scenarios from the audited bank
and orders it "after the bank audit" — so execution ran cards → bank+audit →
obedience comparison. The frozen document wins over the numbering of a chat
instruction.

**Cards rendered (step 1).** 400/400 through the two-pass reasoning-on
factory with a 512-token reasoning budget (operational knob adopted after a
timing probe; the 34 unbounded-reasoning cards were discarded for
uniformity, ≈$0.056). Quality gate: leak-free 400/400, lengths 143–229
words (median 189), one card (t0267) carries an honest truncated-pass flag
but ends complete and in range. Ledger: 995 calls, $0.189. Numbers:
`results/m1_cards_qa.json`.

**Bank + audit (step 4, run before step 2 per the frozen ordering).** 140
scenarios per the frozen design table; audit round 1 (mechanical 0 problems;
Gemini blind classification 72%) caught four real faults — a 2-cent
coin-flip fare, variance-salient wording on mean-gap route scenarios,
price-mediated MOD policy scenarios, distractors leaking HAB through
novelty/routine wording — all fixed; round 2: 0 mechanical problems, 88%
blind agreement, route and distractors at 100%. Remaining misses documented
as inherent (near-equal MOD vehicles read as "no construct"; isolation-block
anchors read individually). Files: `results/m1_bank_audit_round1.json`,
`results/m1_bank_audit.json`. NOTE, recorded before the obedience data was
read: round 1 also flagged the PRC-isolating scenarios as reading like
car-vs-transit choices; the wording was kept because the sub-block design
identifies via the across-scenario fare flip point. The obedience run below
proved the auditor right and the design wrong.

**Obedience comparison (step 2): STOP-AND-ASK.** Frozen rule fired: Arm A
(reasoning-off) median obedience 0.477 < the 0.5 floor (the relative
condition passed: 0.477 ≥ 0.570 − 0.10). Arm B (reasoning-on) median 0.570.
Numbers: `results/m1_obedience.json`; spend $0.224 against the frozen $0.65
cap. Post-verdict diagnostic on recorded answers (no new API calls): **the
PRC scenarios do not measure PRC** — on equal-time fare choices ~85% of
travelers take the cheaper option regardless of planted price sensitivity
(the cheap option dominates for any character who computes), so the
PRC-scenario tendency correlates with φ_MOD at −0.38 and with φ_PRC at
+0.05, in BOTH arms. Price sensitivity planted as a cost-coefficient
multiplier does not show in dominated-choice direction; it needs
non-dominated price-vs-friction trades within one mode. Every other
dimension shows real signal in both arms (0.43–0.74). Stopped per the
owner's instruction; no bank revision, no re-run without a ruling.

## 2026-08-03 — Rulings 10–13 executed; re-run verdict: STOP-AND-ASK again (different cause)

**Ruling 10 (PRC redesign).** The 10 PRC-isolating scenarios rebuilt from
design principles only: within one mode, equal times, a fare saving (grid
€0.20–€2.60) against a constant non-time friction (per-trip card re-entry in
the app), no dominated option, no MOD loading. Implemented with a fixed grid
so the shared rng stream is untouched — verified: exactly the 10
`mode_prc_*` scenarios changed, the other 130 byte-identical. Audit round 3
(`results/m1_bank_audit.json`; round 2 preserved as
`m1_bank_audit_round2.json`): 0 mechanical problems, 88% blind agreement,
8/10 PRC scenarios read as PRICE and the two stragglers are PRICE at modal
read over 5 repeats (temp-0 flutter, documented, not a fault).

**Ruling 10d re-run (`results/m1_obedience_r2.json`; run 1's verdict stands
in the record).** Same travelers, same drawn scenario IDs (4 corrected PRC
slots), frozen rule verbatim. **The instrument fix worked**: Arm A PRC
obedience 0.047 → 0.454; every dimension now shows real signal (Arm A
0.43–0.73). Verdict still **STOP-AND-ASK**: Arm A median 0.4768 — the
relative condition passed again (0.4768 ≥ 0.5703 − 0.10) and the absolute
0.5 floor failed again, by 0.023. Spend $0.039 (480 fresh calls, 2,400
cache hits).

**Why the floor keeps failing, stated for the owner's ruling:** the 0.5
floor is the frozen M1 QA obedience bar, which is defined on the FULL BANK
(10–38 designed scenarios per dimension, 400 travelers); the comparison
reuses it on a 24-scenario, 60-traveler subset where each dimension's index
sits on 2–7 scenarios (split-half reliabilities 0.60–0.90). The
pre-declared exploratory reliability-corrected read of the same Arm-A data
puts every dimension at 0.49–0.77 with a median ≈ 0.51 — consistent with
the full-bank obedience median clearing the floor through aggregation
alone. The subset rule embedded a full-bank-calibrated floor into a short
instrument; that observation is offered to the owner, not acted on.

**Ruling 12 branch taken: FAILED-AGAIN → STOPPED.** No sweep launched, no
further spend. Sweep machinery (`src/eval/sweep.py`, distribution support
in the client per Ruling 4, experiment config) is built and idle, awaiting
a ruling.

**Ruling 13:** the 512-token reasoning bound + uniform re-render is
recorded as a logged operational amendment (entry above).

## 2026-08-03 — Ruling 16 executed: sweep complete, obedience gate FAILED, stopped

**Sweep (`results/m1_sweep_summary.json`).** 56,000 answers (400 × 140),
reasoning-off, one round, 83 invalid (0.15%), 100% carry recorded
answer-token distributions. Ledger truth: $0.583 (the ~$0.45 estimate was
30% under — prompts carry full cards at ~350 tokens each). Concurrency was
stepped 6 → 12 → 16 against the oscillating upstream pool; 938 retries, no
failures. All pre-noise material vault-side.

**Ruling-16 gate: FAILED.** Full-bank pre-noise obedience, the frozen M1 QA
bar on its native instrument (`results/m1_qa.json`): VOT +0.714, SCH
+0.450, MOD +0.765, CRW +0.361, PRC +0.488, HAB +0.464 — **median 0.4761 <
0.5**. Stopped at that finding per the ruling: no noise tuning, no
transmission computation, no further spend. The orchestrator's
reliability-discount hypothesis from the comparison stage is hereby
recorded as WRONG: full-bank aggregation did not lift the median; the
renderer's obedience genuinely sits just under the bar on four dimensions
(SCH, CRW, PRC, HAB), with CRW weakest at 0.361. An honest bar miss,
surfaced at the gate where it occurred.

## 2026-08-03 — Ruling 19: the one authorized improvement round (documented BEFORE rendering)

**Rulings 18–19 recorded.** Model unchanged (qwen3.7-flash traveler-side —
family split and the distribution-based noise mechanism both preclude a
switch; considered and rejected by the owner, not to be revisited). One
improvement round authorized: card instructions + within-cell CRW/HAB
scenario sharpening; frozen bars untouched; hard rule — no third round.

**Card instruction diff (Ruling 19a, written before any card is rendered):**

1. Writer system prompt gains one rule: *"every tendency must surface as at
   least one concrete decision the person makes because of it (a route
   chosen, a ticket bought, a departure moved, an offer declined) — never as
   an adjective or a summary."*
2. Checker system prompt now instructs: *"verify each of the six tendencies
   appears as at least one concrete decision the person makes because of it;
   where a tendency is only described in adjectives or left implicit,
   rewrite that part into such a decision."* (Ruling 19b.)
3. The five descriptor sentences for each of SCH, CRW, PRC and HAB are
   rewritten from adjective-form to decision-form (examples: SCH very-high
   now reads "builds double buffers before anything with a start time —
   earlier trains taken, coffees skipped — because arriving late is not an
   option"; CRW high now reads "routinely takes a slower or earlier
   connection specifically to avoid packed vehicles"). VOT and MOD
   descriptors are byte-identical to round 1 (they cleared the bar at 0.71
   and 0.77).

**CRW/HAB scenario sharpening (Ruling 19c, design principles only):**

- **CRW sub-block**: round-1 design made escaping the crowd cost €2.70 more
  and a mode change (car), so the escape choice was contaminated by PRC and
  MOD. New design: both options are the SAME transit ticket at the SAME
  price; the escape is the stopping service at a constant +7 minutes; only
  the crowding level on the fast service varies across the 10 scenarios
  (levels 1,1,2,2,2,3,3,3,3,3). The across-scenario constancy doctrine
  (logged at bank build) is preserved and between-option cost equality is
  now literal.
- **HAB scenarios (4 mode + 4 route)**: random gain draws replaced by fixed
  flip-point ladders (mode: 2/4/7/10 min; route: 3/5/8/12 min) so habit
  strength is identified by where each traveler abandons the routine;
  routine texture strengthened ("your route and mode of the last six
  years"). The generator draws-and-discards the same random numbers as
  round 1 so every other scenario in the shared stream stays byte-identical
  (verified after minting).

**Costs pre-stated:** re-render ≈ $0.20, re-sweep ≈ $0.58 (ledger truth
logged at run time). Same frozen gate afterwards: full-bank obedience
median ≥ 0.5. Then STOP either way.

**Round-2 execution, 2026-08-03 evening.** Cards re-rendered clean:
400/400, leak-free, lengths 141–248 (median 184), one truncated-pass flag,
$0.130 ledger (`results/m1_cards_r2_qa.json`); spot-checks confirm
decision-form tendencies. Bank audit round 4 (`m1_bank_audit.json`; round 3
preserved): 0 mechanical problems, 90% blind agreement, exactly 18 sids
changed, all in the two authorized cells, zero misses among them.

**BLOCKED: OpenRouter credits exhausted.** The Ruling-19f re-sweep died on
its first calls with HTTP 402 (insufficient credits; the account balance
ran out after ~$1.33 of cumulative traveler-side spend). Zero shards
written, $0 spent on the dead run, fully resumable. Plan C (Gemini
takeover) is NOT available for the traveler side: Ruling 18 forbids the
model change, the family split reserves Gemini for the system side, and
Gemini refuses distribution access (M0 finding). Stopped; the owner is
asked to top up credits (~$0.60 completes the Ruling-19 gate; a few dollars
covers the remaining study stages).

**Credits added by the owner; sweep r2 completed 2026-08-03 late evening**
(`results/m1_sweep_r2_summary.json`): 56,000 answers, 22 invalid (0.04%),
100% with recorded distributions, $0.578 ledger, 202 retries, no failures.

**RULING-19 GATE: PASSED** (`results/m1_qa_r2.json`, gate_only per Ruling
19g — nothing beyond the gate ran). Full-bank pre-noise obedience, round 1
→ round 2: VOT 0.714 → 0.660, SCH 0.450 → 0.478, MOD 0.765 → 0.707, CRW
0.361 → 0.617, PRC 0.488 → 0.508, HAB 0.464 → 0.567; **median 0.4761 →
0.5922, bar ≥ 0.5 PASSED.** The four targeted dimensions all rose (CRW
+0.256 — the biggest, from the scenario de-contamination + decision-form
cards; HAB +0.103; PRC +0.020; SCH +0.028, still individually under 0.5,
which the gate report states plainly). The two strong dimensions paid a
small price for the decision-form emphasis (VOT −0.055, MOD −0.058), both
still comfortably high. Stopped per Ruling 19g: no noise tuning, no
transmission computation — those await the owner's go with the round-2
population accepted.

**Pre-declaration (written before any obedience-comparison data existed):**
the comparison's Arm A (reasoning-off, the frozen answering regime) will
additionally be used for an **exploratory early-read of card transmission**
ρ̂_d, computed with the frozen §5 estimator but on the 60×24 subset instead
of the full bank. It is labeled exploratory everywhere: the confirmatory
transmission number requires the full-bank pre-noise sweep, which the owner
has not yet authorized. Sub-stratification of the 24-scenario draw within
the frozen strata (recorded in `experiments/m1_obedience.json` at draw time,
before any traveler answered): the 4 non-isolating mode slots = 2 MOD +
1 VOT + 1 HAB; the 4 route slots = 2 VOT + 1 SCH + 1 HAB; the 2 policy
slots = 1 PRC + 1 MOD — this guarantees every dimension is covered so the
frozen median rule is computable.

## 2026-08-03 — Ruling 20 executed: M1 close-out pipeline, zero spend — ONE DIVERSITY BAR FAILED

**Authorization.** Ruling 20 accepted the round-2 population (cards r2 + bank
r4) and authorized the remaining M1 pipeline off the cached sweep-r2 answers:
confirmatory transmission, diversity bars, noise tuning, test-retest. Zero
API calls. Run as experiment `m1_qa_r2_full` (config in `experiments/`) so
the committed Ruling-19 gate artifact `m1_qa_r2.json` stays untouched; the
gate step re-ran deterministically and reproduced its 0.5922 PASS exactly.
All numbers below: `results/m1_qa_r2_full.json` (the source of truth).

**Confirmatory card transmission (frozen §5 estimator, split seed 3180,
reported — no bar).** ρ̂ per dimension: VOT 0.670, SCH 0.486, MOD 0.733,
CRW 0.634, PRC 0.516, HAB 0.585 — **median 0.609 vs the parent's 0.83
reference**. Every mobility dimension sits below the parent's worst (0.77).
Split-half reliabilities are high (0.93–0.97), so the correction is small:
the tendencies are measured reliably; the cards genuinely carry this much
signal and no more. Stated plainly for the owner because the frozen Gate M2
arithmetic consumes these numbers: with ρ̂_VOT = 0.670 the frozen
ceiling formula gives e_ceiling ≈ 31.7% median VOT error — under the frozen
35% bar with little room for estimator overhead — and the r ≥ 0.75
per-dimension recovery bar exceeds the measured transmission ceiling on
every dimension except MOD's near miss (r(φ̂,φ) cannot beat ρ̂ except by
luck). That is a statement of arithmetic, not a proposal to move anything.

**Noise layer tuned (frozen §8 port; parameters to be ratified at the
gate).** Mechanism as ported: p_flip = min(0.5, wobble·a·exp(−b·gap)), gap =
top-two share gap of the recorded answer distribution, wobble per traveler
uniform [0.4, 1.6] (seed 3160, planted independent of φ, deposited as
`wobble_m1.npz`). b fixed at the parent's 4.0; a tuned over the recorded
grid to the 0.79 target: **a = 1.2 chosen (top of the recorded grid),
pooled retest 0.7981, inside the frozen 70–90 band.** 15.3% of answers
flip under the layer. Official noised world (round 0, seed 3190) written
system-side to `data/runs/m1/answers_noised.jsonl` with exactly the
allowlisted fields.

**Test-retest (frozen §5 procedure: 30 fixed scenarios per traveler, seed
3170; fresh session = independent seeded application, parent addendum
convention).** Pooled **0.7981 — PASS** (band 70–90). Two-option 0.804
(chance 0.5, n 10,939); three-option 0.740 (chance 0.333, n 1,055).

**Diversity bars (official noised world). Bars (b) and (c) PASS, bar (a)
FAILED.** Median pairwise agreement 0.600 ≤ 0.80 PASS; participation ratio
22.15 ≥ 4 PASS; max pairwise agreement **0.9714 > 0.95 FAIL** — exactly one
pair, t0009–t0361, agreeing on 136/140 scenarios. Diagnostic (documented
before this entry): their planted φ differ substantially (Euclidean distance
2.34 — VOT −1.18, SCH +1.25, CRW +1.50 apart), so this is not a mint
near-twin; agreement does track φ distance population-wide (r = −0.34 over
sampled pairs), the bank is not dominated (median majority-option share
0.677; 1/140 scenarios above 0.90), and pre-noise there were 31 pairs above
0.95, which the noise layer cut to 1. Verdict offered, not acted on: an
order-statistic tail event over 79,800 pairs on a 140-item discrete
instrument, missing the ceiling by 3 scenarios. An honest miss, surfaced at
the gate where it occurred; the owner rules on what follows.

**Mechanism observation (recorded, not a defect).** Obedience on the noised
world is 0.590 vs 0.592 pre-noise, but SCH *rose* 0.478 → 0.555: 22.9% of
flips land on the distribution's top token (the recorded temperature-0.7
sample had itself landed on the runner-up), so the layer partly de-noises
sampling flukes on low-conviction answers. The frozen obedience bar is
pre-noise, so nothing rides on this; recorded because a noise layer that
raises a trait correlation is the kind of surprise that must be on the
record.

**Cost honesty.** Answer sweep ran reasoning-off: 0 reasoning tokens,
exactly 1 completion token per answer (56k total; `m1_sweep_r2_summary.json`).
This close-out: $0.00 (pure NumPy off the cache). Grand total unchanged
≈ $1.81 (COSTS_MOBILITY.md).

**Tests at close.** Parent suite 726 passed; mobility suite 14 passed +
1 expected skip (opt-in live smoke); wall tests green (parent 10, mobility 6
— the former data-dir skip now runs against the existing vault).

**GATE M1-CLOSE: HARD STOP.** Bars: obedience PASS 0.5922, retest PASS
0.7981, diversity (b)(c) PASS, diversity (a) FAIL 0.9714 > 0.95.
Transmission reported (median 0.609). Noise parameters a = 1.2, b = 4.0
await ratification. Awaiting the owner's ruling on the diversity (a) miss
and the go/no-go for Stage M2.

## 2026-08-04 — Gate M1-close reviewed: Rulings 21–23, Stage M2 authorized

**Ruling 21 (diversity bar a).** Recorded as FAIL, verbatim, per the frozen
definition. The diagnostic (single pair t0009/t0361, planted distance 2.34,
pre-noise 31 pairs → 1 post-noise) stands beside it as context, not
mitigation. The population is NOT modified — no traveler removal, re-roll,
or re-render after seeing answers. M1 closes with this bar failed and all
others passed.

**Ruling 22 (noise parameters RATIFIED).** a = 1.2, b = 4.0, wobble seed
3160, achieved pooled retest 0.7981. Observation logged: a sits at the top
of the pre-recorded grid. The SCH obedience-raise mechanism note
(0.478 → 0.555 via flip restoration) stays on the record as observed
mechanism, no action.

**Ruling 23 (M2 ceiling pre-declaration), owner's verbatim sentence,
recorded before any M2 estimation ran:** "M2 recovery bars are expected to
miss on dimensions where measured transmission < 0.75; these misses, if
they occur, are ceiling-inherited findings. The frozen decomposition
(prereg §5, disattenuation formulas) will attribute each result to card
transmission vs estimator. Declared at M1-close, before any M2 estimation
ran." Bars unchanged. No further card rounds (Ruling 19g stands).

**Stage M2 authorized:** ratified noise layer applied to the cached sweep-r2
answers (seeded, logged); frozen M2 estimation pipeline (per-traveler
conditional logit, held-out split per prereg, no estimator changes); all
frozen M2 bars computed with the decomposition/disattenuation analysis
alongside. Zero new API calls expected; any needed spend is a stop-first
event. STOP at Gate M2.

**M2 pre-declarations, recorded and committed BEFORE any estimation ran**
(full detail in `experiments/m2_recovery.json`; conveniences here):
(1) Frozen splits seed 3210 — 80/400 travelers and 28/140 scenarios held
out, stratified by block, written once to `experiments/m2_splits.json` and
reused through M4. (2) The frozen 7-term utility is fit VERBATIM. Declared
specification gap: the Ruling-10 PRC scenarios carry a 'friction' attribute
that the utility (frozen before that redesign) has no term for; no term is
added; the gap is a declared candidate source of PRC estimator overhead.
(3) φ̂ maps = exact inversions of the frozen §2 anchors (VOT ln(V̂/12)/0.55,
SCH log of the late/early ratio on the 1–8× anchor, MOD ASC_car/0.75, CRW
the linear 1.0–2.2 multiplier inversion, PRC log of the ×0.4–2.5 cost
scaling normalized by the training-median). HAB's frozen anchor names no
numeric range, so φ̂_HAB is standardized on training travelers — the one
map without an anchored scale, declared here. Winsorization bounds and sign
guards are enumerated in the config; counts get reported. (4) Bootstrap
B = 200, seed 3220, resampling each traveler's training scenarios (frozen
§5); coverage per the parent's pooled-cells convention. (5) The 'alongside'
latent comparison fit is operationalized as empirical-Bayes MAP shrinkage +
PCA + orthogonal Procrustes on training travelers only (no scipy in the
repo; carries no bar). (6) Ridge λ = 1e-4 as a numerical existence guard,
reported not tuned. (7) The distractor tripwire uses all 10 distractors ×
all 400 travelers on the noised choices, A=+1/B=−1.

## 2026-08-04 — GATE M2 RUN: ALL FIVE FROZEN BARS FAILED; tripwire hunt found real renderer leakage

Source of truth: `results/m2_recovery.json` (frozen pipeline) and
`results/m2_diagnostics.json` (mandated tripwire hunt + exploratory
forensics). Zero API spend; the noised world re-derived bit-identically
(55,978 answers checked) before anything ran. Splits created at seed 3210
and frozen in `experiments/m2_splits.json`.

**Frozen bars, verbatim (held-out travelers):**
- Per-dimension recovery r ≥ 0.75, all 6: **FAIL, all six.** VOT 0.447,
  SCH 0.406, MOD 0.320, CRW 0.603, PRC 0.285, HAB 0.591 (median 0.427).
- VOT real units, median error ≤ 35%: **FAIL** — e_raw 112.2%. The 25%
  reported line: fail (reported, never a gate condition).
- Uncertainty coverage 60–75%: **FAIL** — pooled 16.7%; the bootstrap is
  3–9× overconfident on every dimension (variance ratios in the file).
- Designed-zero separability |r(VOT̂, PRĈ)| ≤ 0.15: **FAIL** — −0.343.
- Distractor tripwire |r| ≤ 0.15: **FAIL** — max 0.222; the frozen bar's
  mandated leakage hunt was run before this report (below).

**Frozen decomposition (declared primary presentation).** e_ceiling at the
measured ρ̂_VOT = 0.670 is 31.7%; estimator overhead = 112.2% − 31.7% =
**+80.5 points — the estimator, not the card ceiling, dominates the VOT
miss.** Disattenuated r (r/ρ̂, capped at 1): VOT 0.668, SCH 0.836,
MOD 0.436, CRW 0.951, PRC 0.553, HAB 1.000. Read: SCH/CRW/HAB misses are
mostly ceiling-inherited (Ruling 23 anticipated exactly this); VOT/MOD/PRC
carry large estimator-side losses on top of the ceiling.

**Where it breaks (diagnostics file; exploratory except the hunt):**
1. **Quasi-separation.** The rendered choices are far more deterministic
   than a shallow logit: 301/400 travelers hit |ASC_car| > 3, 209/400
   |β_switch| > 3; coefficients grow to ridge-limited magnitudes.
2. **Wrong-sign time coefficients for 184/400 travelers.** The frozen
   utility prices all minutes with one β_time; travelers who consistently
   buy schedule buffers (dep block) or prefer the quiet/uncrowded/habitual
   slower option are fit with positive β_time, which then corrupts
   VOT̂ = β_time/β_cost (220/400 outside [0.5, 200] €/h; 40/80 held-out
   degenerate).
3. **Mechanical VOT–PRC entanglement.** φ̂_VOT and φ̂_PRC share β̂_cost
   (numerator/denominator), so estimation noise induces the −0.343 —
   the parent's TRU–RSK lesson reappearing in ratio form.
4. **The signal is in the choices; the structural estimator loses it.**
   From the SAME noised choices, same held-out travelers, the simple
   designed-score tendency index (grader-side oracle reference) recovers
   r = 0.49–0.71 (median 0.60) vs the structural MNL's 0.29–0.60 (median
   0.43). In-sample choice accuracy is 0.804 vs 0.714 zero-information —
   the model mimics choices adequately while its parameters entangle.
5. **The latent comparison fit does not rescue** (held-out r 0.23–0.54).

**Tripwire hunt verdict (mandated): REAL RENDERER LEAKAGE, not estimation
artifact and not a Wall incident.** Pre-noise distractor choices correlate
with PLANTED φ in 9 of 10 distractors, max |r| 0.383 (distr_03
sleep-in-vs-early-swim × SCH; distr_09 dentist-slot × HAB +0.213; distr_00
museum-vs-park × VOT +0.216). The persona drives designed-zero lifestyle
choices; with LLM travelers no scenario is behaviorally a distractor. The
bank audit could not see this (it classifies constructs, not rendered
behavior). No truth crossed the Wall.

**Honest framing.** This is the study's pre-registered negative direction:
standard travel-demand estimation does NOT recover planted parameters from
these LLM travelers' choices, and the file records where it breaks
(near-deterministic choices, unmodeled texture, ratio instability,
persona-everywhere). Ruling 23's ceiling pre-declaration covers part of the
miss; the estimator overhead beyond it is the new, larger finding.

**Tests at gate:** parent 726 passed; mobility 21 passed + 1 expected skip
(new estimator tests included); both wall tests green. Cost: $0.00 this
stage; grand total unchanged ≈ $1.81.

**GATE M2: HARD STOP.** All five frozen bars failed. Nothing proceeds
without the owner's ruling.

## 2026-08-04 — Gate M2 reviewed: Rulings 24–26, Stage M3 authorized

**Ruling 24 (M2 verdicts recorded verbatim).** All five frozen bars FAIL as
tabled in the entry above; the 25% line fails as reported-only. The
decomposition table is the primary presentation. Ruling 23 attribution
confirmed: SCH/CRW/HAB predominantly ceiling-inherited; VOT/MOD/PRC carry
estimator overhead beyond the ceiling (+80.5 points on VOT). No
re-estimation, no estimator modification, no post-hoc rescue of any bar.

**Ruling 25 (findings registered; report framing fixed now).**
(a) believable-but-not-estimable: choice prediction 0.80 vs 0.71 zero-info
alongside failed parameter recovery; (b) persona-everywhere: 9/10
distractors correlate with planted traits, max 0.383 — no scenario is
behaviorally neutral for LLM travelers; (c) near-deterministic rendering
starves MLE (301/400 ridge-limited, 184/400 positive time coefficients,
degenerate ratios half of held-out). Tripwire hunt verdict accepted: real
renderer leakage, not a Wall incident, not artifact.

**Ruling 26 (exploratory estimator comparison) — EXACT SPECS, written here
BEFORE any comparison fit runs.** Three estimators on identical inputs (the
official noised round-0 choices on the frozen training scenarios), graded
on the same held-out travelers with the same frozen anchor maps:
1. The frozen per-traveler MNL exactly as run at M2 (ridge 1e-4) — numbers
   taken from `results/m2_recovery.json`, not refit.
2. The designed-score tendency index exactly as in
   `results/m2_diagnostics.json` (grader-side oracle reference — it knows
   the designed loadings; stated in every table).
3. ONE regularized MNL variant: identical frozen 7-term utility, identical
   Newton fit, ridge lambda = 1.0 on all 7 coefficients — the single
   pre-declared value, chosen here without any new fit having run; its
   PRC/HAB map normalizations recomputed on training travelers under the
   variant (declared).
Metrics: held-out per-dimension r + median; VOT median real-unit error for
(1) and (3). Every table carries the label "EXPLORATORY — declared at
M2-close (Ruling 26); cannot overturn frozen verdicts." If any spec above
proved insufficient to run without further choices, the comparison would
not run.

**Stage M3 authorized** (calibration + elasticity realism; zero new API
calls; any spend-needing step is stop-first). Pre-declarations, recorded
before implementation (full machine-readable detail in
`experiments/m3_calibration.json`):
- **Short survey (frozen §5):** K = 12 fixed scenarios + public profile,
  hierarchical shrinkage toward the population fit. Operationalized:
  Gaussian prior = moments of the M2 EB-MAP coefficients over training
  travelers (+1e-6 jitter); per-traveler MAP on the K noised answers;
  Laplace posterior at the MAP. The K scenarios: seed 3230, one fixed
  stratified draw from training scenarios (mode 5, dep 3, route 2,
  policy 2), same for every traveler; the decay grid K ∈ {4, 8, 12, 24,
  48, 104} uses nested prefixes of one seeded stratified permutation
  (confirmatory bars at K = 12 only). The public profile is designed
  signal-free (§2), so the posterior uses the unconditional prior; the
  profile enters through the profile-only baseline, and profile-group
  prior-mean deltas are reported as the design check.
- **Brier lift (frozen bar ≥ 25%):** cells = held-out travelers × held-out
  non-distractor scenarios (26); p_true per frozen §5 = 50 seeded
  applications of the ratified noise layer (rounds 100–149) to the recorded
  pre-noise answers; BS = mean over cells of Σ_o (p̂_o − p_true_o)²;
  baseline = training-traveler round-0 shares. k-NN reported beside it
  (k = 20, neighbours by agreement on the same K = 12 answers).
  Profile-only baseline: training shares within the 4-field profile group
  (≥10 members, else drop area_type, else age×occupation, else population).
  The frozen LLM-profile baseline is NOT run — it needs ~$0.8 of Gemini
  spend; surfaced at the gate under the stop-first rule.
- **ECE (frozen bar ≤ 0.05):** pooled option-level cells, predicted p̂_o vs
  the realized round-0 choice indicator, 10 equal-width bins.
- **Coverage (frozen bar 60–75%):** 200 seeded Laplace draws (seed 3240)
  pushed through the frozen anchor maps with the M2 normalization constants
  verbatim (b0 0.21121096372170123, HAB mean 3.9886112024486247, sd
  5.037342415307932); |φ̂ − φ| ≤ 1σ pooled over held-out × 6 dims; the
  decay curve over the K grid is watched and reported (a finding, never a
  failure).
- **Elasticities (REPORTED, no bar; §6 caveat verbatim in every table):**
  round-0 noised choices, all 400 travelers (aggregate realism, not a
  confirmatory quantity). Families from the built bank: (i) parking/driving
  cost — the 15 policy_prc scenarios (drive cost €5.58–€11.65, transit
  alternative fixed): arc elasticity of population car share between the
  cheapest and dearest scenarios (midpoint formula), log-log OLS slope
  across all 15 as robustness; (ii) toll take-up — the 4 policy_vot
  scenarios vs toll level, same formulas; (iii) employer transit subsidy —
  the 6 policy_mod scenarios reported as uptake share vs time penalty
  (no price variation → no elasticity). GAP, stated plainly: the design
  table's transit fare-change probe was never built into the bank, so the
  literature fare-elasticity comparison cannot be computed. Published
  reference ranges, cited in the results file: parking/car-cost elasticity
  −0.1 to −0.4 (Goodwin et al. 2004; Vaca & Kuzmyak 2005; Litman
  synthesis); toll demand −0.1 to −0.45 (Odeck & Bråthen 2008 review);
  transit fare −0.2 to −0.5 (TRL 2004/Litman) recorded for the gap note.
- **Aggregate-vs-individual contrast:** per held-out scenario, predicted
  population option shares from the K = 12 models of held-out travelers
  (mean predicted probability) vs those travelers' observed round-0 shares
  — MAE in share points and Pearson r over scenario-option cells, beside
  the population-marginal reference and the same cells from the M2
  full-data models (context); presented against the individual recovery
  medians as the parent's aggregate-easy/individual-hard answer.

## 2026-08-04 — GATE M3 RUN: lift FAIL, ECE PASS (first bar to pass), coverage FAIL with textbook decay

Source of truth: `results/m3_calibration.json`. Zero API spend. All
choices per the pre-declaration committed at 40d3a2a; the K = 12 draw
realized the declared 5/3/2/2 block composition (scenario IDs in the file).

**Frozen bars, verbatim (held-out travelers, K = 12 + profile):**
- **Brier lift ≥ 25% over population-marginal: FAIL — +7.0%** (Brier 0.1796
  vs 0.1930). Reported beside it per the frozen baseline list: **k-NN
  +33.8%** (0.1278) — the behavioral-similarity baseline clears the level
  the structural estimator misses by a wide margin; profile-only −14.2%
  (worse than zero-information, consistent with the designed signal-free
  profile; group prior-mean deltas 0.16–0.25 SD confirm the design).
- **ECE ≤ 0.05: PASS — 0.0473.** The first frozen bar the study passes in
  M2–M3: the model's choice probabilities are close to honest even though
  its parameters are not recoverable (Ruling 25a extended: believable,
  probabilistically calibrated, still not estimable).
- **Coverage 60–75%: FAIL — 30.4%** at K = 12. **The parent's
  coverage-decay phenomenon reappears, textbook:** pooled coverage falls
  monotonically as evidence grows — 33.3% (K=4) → 30.4% (12) → 21.5% (24)
  → 19.6% (48) → 14.4% (104) — while median r rises 0.20 → 0.40. Watched
  and reported per the frozen bar: a finding, not a failure. Footnotes:
  MOD coverage is 0.0 at every K — a pure location offset (fitted ASC_car
  sits ~+5 utility units above the ±1.5 anchor scale, so the anchor map is
  systematically shifted; correlation unaffected); at K = 4 the prefix
  contains no crowding-bearing scenario, φ̂_CRW equals the prior constant
  for every traveler, its r is undefined and the median shows NaN.
- The frozen **LLM-given-profile-only baseline was NOT run** — it needs
  ~10.4k Gemini calls (≈$0.8); surfaced here per the stop-first rule.

**Elasticities (REPORTED, no bar; §6 by-construction caveat verbatim in
the results file; round-0, all 400):**
- Parking/driving cost (15-scenario demand curve, €5.58–€11.65): arc
  −0.094 (marginally below the published −0.1..−0.4); log-log OLS −0.201
  (inside). Car share falls 0.34 → 0.29–0.32 across the curve.
- Toll take-up (4 scenarios): arc −0.166 (inside −0.1..−0.45), but the
  curve is nearly flat (0.190–0.203) and time savings co-vary 16–21 min —
  stated as pre-declared; a weak-response reading is the honest one.
- Employer subsidy uptake: 50.8% → 48.6% → 46.4% at +2/+3/+5 min — the
  right ordering.
- **Transit fare gap:** the design table's fare-change probe was never
  built into the bank; the fare-elasticity comparison is reported missing.
- Headline: aggregate price response sits at the inelastic edge of the
  published ranges while individual recovery fails — GATSim-style surface
  realism without internal validity, now quantified.

**Aggregate vs individual (the parent's question, answered):** the
population-marginal reference predicts held-out share cells best (MAE 5.2
points, r 0.977); the K = 12 models (9.4, 0.887) and even the M2 full-data
models (8.2, 0.920) track aggregates decently while individual recovery
stands at median r 0.43 (M2) / 0.32 (K = 12). Aggregates are easy —
marginals alone nearly saturate them; individuals are hard.

**Ruling-26 exploratory comparison (EXPLORATORY — declared at M2-close;
cannot overturn frozen verdicts).** Held-out median r: frozen MNL 0.427;
regularized MNL (λ = 1.0) 0.507; designed-score oracle 0.600. The λ = 1.0
variant tames the ASC blow-ups (MOD 0.32 → 0.62) but WORSENS real-unit VOT
error (112% → 166%: shrinking β_cost toward 0 inflates the ratio). Neither
MNL approaches the oracle → specification misfit, not estimation variance,
is the binding constraint. VOT per-dimension table in the results file.

**Tests at gate:** parent 726 passed; mobility 21 passed + 1 expected
skip; both wall tests green. Cost $0.00; grand total unchanged ≈ $1.81.

**GATE M3: HARD STOP.** Lift FAIL, ECE PASS, coverage FAIL (decay
reported). Awaiting the owner's rulings.

## 2026-08-04 — Gate M3 reviewed: Rulings 27–30, Stage M4 pilot authorized

**Ruling 27 (M3 verdicts recorded verbatim).** Lift FAIL (+7.0% vs 25%),
ECE PASS (0.0473 — the study's first passing frozen bar), coverage FAIL
(30.4%) with the coverage-decay phenomenon reproduced (33.3% → 14.4%
monotone). Findings registered for the report: (a) coverage-decay
generalizes across domains; (b) k-NN +33.8% vs structural +7.0% —
behavioral similarity beats structural estimation ~5×; (c) profile-only
−14.2% confirms the negative control; (d) aggregate-easy/individual-hard
quantified (marginals r 0.977 vs individual 0.32–0.43). Ruling-26
comparison recorded as labeled exploratory: regularization helps
correlations, worsens real-unit VOT — specification misfit is binding.
The two footnotes (MOD location offset; K=4 CRW undefined) logged.

**Ruling 28 (transit-fare gap).** Recorded as a pre-existing design
omission, reported as a limitation verbatim. No new scenarios, no
backfill, no re-sweep. The elasticity table ships with the by-construction
caveat and this gap noted.

**Ruling 29 (frozen LLM-profile baseline AUTHORIZED, cap $1.00).**
Operationalization, declared before any call: Gemini flash-lite
(system-side role, family split intact), temperature 0, one call per
(unique held-out profile, held-out non-distractor scenario) — predictions
keyed by profile, so identical profiles share a cached call; the prompt
shows ONLY the public profile fields and the scenario text as shown to
travelers and asks for choice shares ("A=0.xx B=0.yy ..."); parse
failures fall back to uniform and are counted; scored with the identical
Brier cells and p_true as the other baselines; result appended to the M3
record beside the frozen baseline list. Cache in
`data/runs/m3_llm_profile/`; cumulative cost tracked against the $1.00
cap; persistent errors are a stop event.

**Ruling 30 (M4 ceiling pre-declaration), owner's verbatim sentence,
recorded before any M4 compute:** "The M4 coherence bars (|r| ≥ 0.3) are
at elevated risk of ceiling-inherited misses given measured transmission;
if they miss, the frozen decomposition attributes card vs dynamics.
Peak-spreading depends on incentive response, which M2/M3
choice-prediction results (0.80) suggest is present. Declared at M3
close, before any M4 compute." Bars unchanged.

**M4 world pre-declaration (before any simulated day, before the pilot).**
- **Network reconciliation, stated for the owner's veto at the pilot
  gate:** memo (a) argues peak spreading on one Vickrey bottleneck; the
  frozen W_i index counts mode-or-route switches and memo (b) reasons
  explicitly about "travelers who start on the better route" — a
  one-route world would make W_i identically zero and the HAB coherence
  bar uncomputable. The world therefore has TWO routes into the CBD
  (PRD §5's first-listed option): route A, free-flow 18 min through one
  Vickrey bottleneck (capacity calibrated per memo a); route B, steady
  26 min, never congested. Single mode (car): W_i reduces to its route
  term, exactly as the frozen formula allows.
- **Population:** 300 of the 400 travelers, seeded draw (seed 3250);
  preferred arrival times PAT ~ N(08:30, 12 min), seed 3260, clipped to
  [07:30, 09:30]; PATs are planted truth (vault), each traveler sees only
  their own in prompts.
- **Day 1 (mechanical, per memo a):** everyone on route A, departing
  PAT − 18 min. No LLM call.
- **Days 2..20 (LLM):** frozen answering config (qwen3.7-flash,
  reasoning-off, T 0.7, cached, resumable). Menu of 10 single-letter
  options: A–E = route A departing {−15, −5, 0, +5, +15} min relative to
  yesterday's departure; F–J = route B, same shifts; the same-route
  same-time option is labeled "as yesterday". Prompt = card + own PAT +
  own experienced memory of the last 3 days (route, departure, realized
  travel time, early/late vs PAT) + the menu. No forecasts shown; agents
  adjust from experienced times only (PRD §5).
- **Bottleneck mechanics:** deterministic point queue — sorted A-route
  departures served at capacity s veh/min, travel = 18 + queue delay;
  route B constant 26. τ_{t−1}(d) for X_i = linear interpolation between
  the previous day's observed (departure → travel) pairs on the chosen
  route, flat beyond the observed range; route B is constant by
  construction. W_i and X_i exactly as frozen (§5, Ruling 8; b = 5 min).
- **Capacity calibration (design-time, planted parameters only, before
  any agent runs, per memo a):** choose s so the day-1 naive profile's
  maximum queue delay lands in [12, 18] min (target 15); the calibration
  curve and chosen s are logged to
  `results/m4_capacity_calibration.json` before the pilot.
- **PILOT (authorized): 50 travelers × 3 days** — the first 50 positions
  of the seeded 300-draw; day 1 mechanical + 2 LLM days = 100 calls.
  Mechanics only: per-day cost, retry behavior, loop end-to-end, index
  computation. NO bar grading on pilot data. Full-run cost and wall-clock
  extrapolated from measured pilot numbers.
- **Viewer scaffold:** zero-API frontend structure in `mobility/app/`,
  reading the system-side trajectory log (pid, day, route, departure,
  travel, arrival — never φ, never cards); polish deferred to after the
  full run.

## 2026-08-04 — Ruling 29 executed + Gate M4-pilot: mechanics clean end to end

**Ruling-29 baseline result (appended to `results/m3_calibration.json`,
`llm_profile_baseline_ruling29`).** 1,820 Gemini flash-lite calls
(70 unique held-out profiles × 26 scenarios, temperature 0), $0.1421
against the $1.00 cap, zero persistent errors, 73 parse failures (4.0%,
uniform fallback, counted). **Brier 0.3217 vs marginal 0.1930 on the
identical 2,080 cells → lift −66.7%.** The frozen baseline list now
reads: structural K=12 +7.0% (FAIL vs 25%), k-NN +33.8%, profile-only
shares −14.2%, LLM-profile −66.7%. The parent's finding 5 replicates in
the travel domain, amplified: an LLM given a designed signal-free profile
invents demographic structure and predicts far worse than the population
marginal. The negative control holds twice over.

**M4 world minted + capacity calibrated (design-time, before any agent
ran; `results/m4_capacity_calibration.json`).** 300 travelers (seed 3250,
draw order preserved), PAT ~ N(08:30, 12′) clipped [07:30, 09:30] (seed
3260), vault-deposited. Capacity 5.25 veh/min → day-1 naive max queue
delay 14.79 min (target 15, band [12, 18]); full grid curve in the file.

**PILOT (50 × 3, mechanics only per the pre-declaration — nothing here is
graded): CLEAN.** `results/m4_pilot_summary.json`; trajectories
system-side at `data/runs/m4_pilot/trajectories.jsonl`. 100/100 LLM calls,
0 invalid answers, 0 retries, $0.0018, 4.5 s per 50-traveler day at
concurrency 8, reasoning tokens 0. Mechanics observations (not graded):
21/50 travelers changed plan after day 1, one moved to route B, departure
spread widened 13.8 → 16.9 min; queue delay at pilot scale is ~0.1 min as
expected (capacity is calibrated for 300, not 50 — the full run binds).
W_i and X_i compute finite on the pilot trajectories; the frozen formula
implementations carry offline unit tests (`tests/test_m4_engine.py`).

**Full-run extrapolation (300 × 20 days = 5,700 LLM calls, days serial,
travelers parallel):** cost ≈ $0.12 (pilot-measured ~600 prompt tokens ×
5,700 calls; completions 1 token); wall-clock ≈ 9 min at concurrency 8 in
the pilot's clean conditions, budget 20–40 min under the upstream 429
oscillation seen in the sweeps. Cache + resume verified (per-(pid, day)
keys; a crash loses nothing).

**Viewer scaffold** in `mobility/app/` (index.html, viewer.js, style.css):
loads a trajectory log, three panes wired (one-day departures→arrivals,
20-day departure-histogram morph, traveler inspector — public trajectory
only). Structure only; polish after the full run.

**Tests at gate:** parent 726 passed; mobility 28 passed + 1 expected
skip (7 new engine tests); both wall tests green. Costs this entry:
$0.1421 (Ruling 29) + $0.0018 (pilot) → grand total ≈ $1.95.

**GATE M4-PILOT: HARD STOP.** Awaiting the owner's go for the full
300 × 20 run (and the two-route reconciliation stands open to veto).

## 2026-08-04 — Gate M4-pilot reviewed: Rulings 31–33, full run authorized

**Ruling 31 (LLM-profile baseline registered).** −66.7% (Brier 0.3217 vs
marginal 0.1930) appended to the frozen baseline list. Finding registered:
the negative control holds twice over — an LLM given a signal-free profile
invents structure and predicts worse than zero information. Report-grade.

**Ruling 32 (two-route world APPROVED).** Ratified for the full run;
grounds: the frozen W_i formula and HAB coherence bar require route choice
to be computable; declared pre-run at 4f1b741; memo (b) reasons about
route switching. Owner's footnote, recorded verbatim: "Memo (a)'s
peak-spreading attainability argument assumed a single bottleneck; the
ratified two-route world adds an escape path. Any peak-spreading result is
interpreted under the two-route config, and the report notes this config
difference from memo (a)."

**Ruling 33 (full run authorized).** 300 × 20 days, frozen config
(capacity 5.25 veh/min, seeds 3250/3260, memory 3 days, qwen
reasoning-off, days serial, travelers parallel). Rules in force: (a) no
parameter/prompt/capacity/config change once day 1 starts; failures resume
from cache, never restart modified; (b) no grading, no peeking-driven
adjustment during the run — bars only after day 20; (c) per-day cost and
429 behavior logged; budget cap $1.00 wall-to-wall, trajectory above it is
a stop event. Bars afterward exactly as frozen; then the pre-registered
elasticity experiments; then viewer polish (system-side log only).

**M4 elasticity experiments — operationalization pre-declared BEFORE any
run (frozen §5: 'the same population re-asked under changed attributes
with everything else fixed'):**
- **Instrument:** the mode-choice block. A price change a text-reading
  agent cannot see cannot move it, so each experiment's scenario set is
  the mode scenarios whose relevant money amounts render in the text
  (every cost renders as 'EUR x.xx' via the generator's fixed format;
  variants are built by exact string substitution of those formatted
  values, attrs updated to match — same numbers, same wording otherwise).
  mode_crw shows no money and is excluded by this rule.
- **Experiment 1, uniform travel cost +10%:** all visible cost_eur ×1.10;
  set = mode_vot (14) + mode_mod (12) + mode_prc (10) + mode_hab (4) = 40
  scenarios.
- **Experiment 2, transit fare −20%:** transit fares ×0.80; set =
  mode_vot (both options; transit by construction) + mode_mod (tram
  option) + mode_prc (both) = 36 scenarios; mode_hab excluded (mode
  unstated in text).
- **Experiment 3, cordon charge:** +EUR 4.00 (the §3 design-table €2–6
  midpoint) on every car option; set = mode_mod (12); text: 'EUR x.xx
  fuel and parking' → 'EUR y.yy fuel, parking and the new EUR 4.00 city
  charge'.
- **Population:** the 300 M4 travelers. **Both arms pre-noise** (fresh
  answers are pre-noise; baseline = their cached sweep-r2 pre-noise
  answers on the same scenarios) — the comparison measures the renderer's
  response, and the trait-independent noise layer would only blur an
  aggregate. Frozen answering config, one round, no distributions
  requested (not needed; declared).
- **Metrics:** aggregate option/mode shares in both arms; arc elasticity
  e = (ΔQ/Q̄)/(ΔP/P̄) with midpoint denominators (exp1 ΔP/P̄ = 0.0952 on
  all costs, car and transit shares both reported; exp2 −0.2222 on fares,
  transit share; exp3 per-scenario car-cost change averaged for the
  midpoint, car share, plus the plain % car-share reduction beside the
  congestion-charging literature). Published references: fare −0.2..−0.5
  (TRL 2004/Litman — this experiment now supplies the fare number whose
  static probe Ruling 28 recorded as missing); car cost −0.1..−0.4
  (Goodwin et al. 2004); congestion-charge car-trip reduction −10..−35%
  (London/Stockholm program evaluations). Every table carries the frozen
  §6 by-construction caveat.
- **Estimated spend:** (40+36+12) × 300 = 26,400 calls ≈ $0.28; with the
  day-loop ≈ $0.40 wall-to-wall against the $1.00 cap.

## 2026-08-04 — GATE M4 RUN: peak bar FAIL (mechanistically explained), HAB FAIL (ceiling-inherited), SCH PASS; elasticities land on the published ranges

Sources of truth: `results/m4_full_summary.json` (run), `results/m4_bars.json`
(frozen bars), `results/m4_elasticity.json` (reported table). Ruling 33
rules held: zero config changes after day 1, no grading before day 20,
per-day cost logged, wall-to-wall $0.3896 vs the $1.00 cap.

**Run + incident log.** 5,700/5,700 day-loop calls, 0 invalid answers,
125 retries, $0.1112, ~43 min. Incident: days 12–14 rode an upstream 429
congestion wave (913 s / 751 s / 284 s vs the normal 26–56 s per day) on
backoff, no failures, no intervention. Elasticity arms: 26,400/26,400
calls, $0.2784, 2,899 retries, no failures.

**Frozen bars, verbatim:**
- **Peak spreading S20 ≤ 0.80·S1: FAIL** — S1 0.2633, S20 0.2933
  (ratio 1.114). Mechanism, from the day context in the results file:
  day-1's saturated bottleneck meters arrivals at capacity — S1 equals the
  capacity ceiling (5.25/min × 15 / 300 = 0.2625) almost exactly, so the
  frozen day-1 anchor measures the queue, not the behavior. On day 2, 19%
  of travelers escape to route B, the metering breaks and S jumps to
  0.387; from there S falls monotonically to 0.293 — a −24.3% fall from
  day 2 that clears the bar's magnitude but not its frozen anchor.
  Recorded as an honest FAIL under Ruling 32's footnote (the two-route
  escape path changes what the day-1 anchor means); the day-2-rebased
  reading is exploratory context, not a re-grade.
- **HAB coherence r(W, HAB) ≤ −0.3: FAIL** — Spearman −0.188.
  Disattenuated by measured HAB transmission (0.5845): **−0.322** — the
  underlying behavioral relationship clears the bar; the card ceiling
  attenuates it below. A ceiling-inherited miss, exactly as Ruling 30
  pre-declared. Context: route switching is rare (W mean 0.021), and the
  W-by-HAB-quartile means are cleanly monotone (0.034 → 0.012).
- **SCH coherence r(X, SCH) ≤ −0.3: PASS** — Spearman **−0.361**
  (disattenuated −0.743). The study's second passing frozen bar.
  X-by-SCH-quartile means: 0.445 / 0.270 / 0.094 / 0.119 — rigid
  schedules take sharply less lateness risk.
- Behavioral tail noted: 7 of 300 travelers ratcheted −15 min repeatedly
  and drifted to extreme early departures (earliest 03:55) — persona
  caricature at the tail, visible in the viewer.

**Elasticities (REPORTED, §6 caveat verbatim in the file; two of three
inside published ranges):**
| experiment | result | published |
| cost +10% | car arc −0.471 (transit cross +0.19) | −0.1..−0.4 — slightly beyond |
| fare −20% | transit arc **−0.327** | −0.2..−0.5 — inside; supplies Ruling 28's missing fare number |
| cordon €4 | car share −9.9%, arc −0.186 | −10..−35% — at the range edge |
The headline contrast is now complete: aggregate price behavior sits on
or near the empirical ranges at every probe while individual parameter
recovery fails — surface realism without internal validity, measured
statically (M3) and dynamically (M4).

**Viewer: DONE and verified against the full run** (`mobility/app/`;
`cd mobility/app && python -m http.server 8000`, load
`data/runs/m4_full/trajectories.jsonl` or drag it onto the page). Three
panes: one-morning replay (day-1 queue vs day-20 spread), the
histogram-with-ghost + peak sparkline (the S trajectory is the exhibit),
and the traveler inspector (public trajectory only — no φ, no cards).
Time axis percentile-clamped so the 7-traveler early tail cannot stretch
it; verified live in-browser against the real trajectory log.

**Tests at gate:** parent 726 passed; mobility 28 passed + 1 expected
skip; both wall tests green. Stage M4 spend $0.3896; grand total ≈ $2.34.

**GATE M4: HARD STOP.** Peak FAIL (mechanism recorded), HAB FAIL
(ceiling-inherited per Ruling 30), SCH PASS; elasticity table shipped
with caveat. Awaiting the owner's rulings and the M5 report go.

## 2026-08-04 — Gate M4 reviewed: Rulings 34–35, viewer craft pass + Stage M5 authorized

**Ruling 34 (M4 verdicts verbatim).** Peak spreading FAIL (S1 0.2633 →
S20 0.2933 vs frozen anchor); r(W,HAB) −0.188 FAIL, ceiling-inherited per
Ruling 30 (disattenuated −0.322 clears); r(X,SCH) −0.361 PASS
(disattenuated −0.743). The mechanistic peak analysis (day-1 anchor
measured the saturated queue; monotone −24.3% fall from day 2) is
recorded as exploratory context under Ruling 32's footnote — NEVER a
re-grade; the frozen bar stays FAILED. Elasticities recorded with the
verbatim caveat; the fare number fills Ruling 28's gap as a dynamic-probe
result, labeled as such (the static-bank gap remains a limitation).

**Ruling 35 (findings registered).** (a) SCH asymmetry — weakest in
static transmission (0.486), strongest in dynamics (clean coherence
pass): repeated consequential days elicit what one-shot scenarios cannot;
(b) the metering artifact — day-1 anchors in congested systems measure
capacity, not behavior (methods lesson for the harness); (c) the ratchet
tail — 7/300 travelers drifted to extreme early departures: LLM persona
caricature under repetition; (d) aggregate realism holds statically AND
dynamically while individual recovery fails — the study's headline
contrast, complete at every level.

**Authorized:** the one budgeted viewer craft pass (presentation layer
only, zero API, truth-leakage check stated in the report) and Stage M5
(REPORT_MOBILITY.md + plain-language summary, zero API, nothing pushed).

## 2026-08-04 — GATE M5: viewer craft pass done, report + summary drafted

**Viewer (final).** All seven directives implemented: designed dark theme
(44px display title, tabular numerals, 8px grid, amber/cyan kept
everywhere); morning replay with delay-proportional dot glow, the
bottleneck drawn as a lane pinch, live clock + time cursor, day-1 ghost
toggle; histogram with permanent day-1 outline, translucent busiest-15-min
band, animated peak numeral, S-sparkline with S1 reference and the day-2
rebase annotation labeled exploratory; inspector with click-a-dot
selection, per-day delay sparkbars, and a notable-trajectories shortlist
(the 7 early-ratchet travelers by the pre-06:30 criterion + top
switchers); one plain-language caption per pane; keyboard scrubbing
(arrows/space), rAF playback, static-file open works. Verified live
in-browser against the full-run log; screenshots committed:
`results/viewer_pane1_day1.jpg`, `viewer_pane1_day20.jpg`,
`viewer_pane2_peak.jpg`, `viewer_pane3_inspector.jpg`.

**Truth-leakage check (directive 7): PASS.** The viewer's data file
(`data/runs/m4_full/trajectories.jsonl`, 6,000 rows) contains exactly the
system-side fields {pid, day, route, dep_min, travel_min, arrive_min} and
nothing else; route values are public labels; planted parameters, PATs,
wobble and card text are not representable in the schema.

**Stage M5 documents drafted.** `results/REPORT_MOBILITY.md` (scoreboard
first, 10 fails before 6 passes, verbatim frozen bars, both decomposition
tables, findings from Rulings 25/27/31/35, limitations incl. the fare gap,
the diversity single-pair fail and the Ruling-32 footnote, costs, seeds,
commit hashes) and `RESEARCH_SUMMARY_MOBILITY.md` (plain-language; the
believable-calibrated-aggregate-realistic-individually-unmeasurable
headline; parent parallels drawn explicitly).

**Verdict-transcription spot-check:** 28 scoreboard numbers checked
against their result files programmatically — 27 exact string matches, 1
rounding (transmission median 0.6091 reported as 0.609), plus JSON-level
equality asserts on every bar value: all passed.

**Tests at gate:** parent 726 passed; mobility 28 passed + 1 expected
skip; both wall tests green. Stage M5 spend $0.00; grand total ≈ $2.34.
Nothing pushed; repo local per standing rule.

**GATE M5: HARD STOP.** Report and summary await the owner's review.

## 2026-08-04 — Gate M5-final: Viewer V2 (the living corridor) built and verified; Rulings 36–37 NOT ON RECORD — stopped on that portion

**Viewer V2, full visual rebuild per the directive.** A stylized top-down
corridor world on canvas: origin neighbourhood, destination district,
route A crossing a river bridge that visibly pinches (guard-wall
narrowing), route B arcing as the ring road; night-to-dawn ambiance (CSS
dawn layer keyed to the clock); vehicles as glowing comet dots with
persistence trails, amber on A, cyan on B. **Every moving element is
interpolated from the recorded five fields only**: each traveler drives
free-flow to their queue-join point, spends exactly their recorded delay
creeping to the pinch (queue length ∝ delay), releases, and arrives at
their recorded arrival minute; arrivals pool as light in the district.
Experience: morning replay with scrubbing and live clock; twenty-mornings
autoplay with the per-day learning caption ("day N · k travelers have
tried the ring road", computed from the log); 4-beat story mode with
data-driven captions and the t0287 spotlight (ringed, labeled, others
dimmed); share mode (clean 16:9, title + stat strip, day chip hidden);
"p" = PNG frame (canvas composite), "r" = 10-second WebM via native
MediaRecorder (no dependencies). The numbers panes (histogram + day-1
outline, S-sparkline with the exploratory day-2 annotation, inspector
with the ratchet shortlist) moved below the fold, kept.

**Performance, measured:** the first build used canvas shadowBlur per dot
and starved the renderer; rebuilt on pre-rendered glow sprites with
additive compositing and destination-out trail fade — **measured 0.38 ms
per frame (≈2,600 fps headroom) at 1600×900 with 300 travelers**; the
day-loop clamp now also survives occluded-window rAF throttling. Verified
live in-browser; frames judged and re-taken (bloom softened once, share
chip collision fixed once) before finals.

**Final frames committed:** `results/v2_day1_jam.jpg` (the day-1 queue
burning to the pinch), `v2_day20_spread.jpg` (free flow + cyan ring-road
cohort), `v2_beat_ratchet.jpg` (05:34, t0287 alone, ringed and labeled),
`v2_share_mode.jpg` (clean 16:9 share frame), `v2_numbers.jpg` (the
below-fold numbers section). Truth-leakage check re-run at build end:
6,000 rows, fields exactly the system-side allowlist, **PASS**.

**Rulings 36–37: STOPPED, not executed.** The directive says to apply
them "exactly as previously ordered", but no such order exists in this
session, in this log (the ruling trail ends at 35), in the report, or in
cross-session memory. The three summary corrections (Ruling 36) cannot be
known without the ruling text, and inventing them would violate the
study's own discipline. Prepared WITHOUT acting, read-only, toward Ruling
37's stated intent: remote = github.com/Umaraslam66/glassbox; the
would-be push is 29 local commits on main (85aabbf scaffold → the Gate
M5-final commit); exclusion scan clean — no .env/secret/key-named files
tracked, zero tracked files under mobility/data/, PRD_MOBILITY.md
untracked per Ruling 5, both wall tests green. NOTHING PUSHED. Awaiting
the owner's verbatim Ruling 36 corrections and Ruling 37 specification.

## 2026-08-04 — Rulings 36–39 received in full; Ruling 36 executed

**The missing rulings arrived, stated verbatim by the owner:** 36 (three
summary corrections + spot-check re-run), 37 (publication package —
assemble and scan only, NO push without the owner's explicit word),
38 (no browser tools this session; all viewer frames rendered headlessly),
39 (Viewer V3 concept phase: three distinct art directions as
final-quality frames from recorded data only, hard stop after concepts).
Order of work 36 → 37 → 39, stop at Gate M5-final-2.

**Ruling 36 executed.** `RESEARCH_SUMMARY_MOBILITY.md` edited — exactly
the three ordered corrections, nothing else (diff verified):
(a) the peak-spreading sentence now leads with the frozen FAIL verdict on
its day-1 anchor, states the saturated-queue mechanism, and labels the
day-2 −24% fall exploratory (pre-declared footnote); (b) the cordon
result now reads "at the low edge of" the published London/Stockholm
range instead of "matching" it; (c) coverage is cited against the frozen
60–75% band instead of "nominal 68%". REPORT_MOBILITY.md untouched per
the ruling.

**Transcription spot-check re-run on the edited summary
(programmatic, zero API): 18/18 claims verified** against their result
files (script in session scratchpad; every check names its file and
field). Rounding conveniences on record: −0.327→"−0.33", −9.9%→"10%",
16.7%→"17%", 0.6091→"0.61", 112.18%→"112%", −66.65%→"−67%"; the day-2
peak rebase computed from the m4_bars trajectory (0.3867 → 0.2933,
−24.2%) is quoted as "24%". Two pre-existing plain-language roundings
("five-fold" for 4.8×, "half" for 184/400) verified and left as-is per
"no other content changes".

**Tests at this commit:** parent 726 passed; mobility 28 passed +
1 expected skip; both wall tests green.

## 2026-08-04 — Rulings 37–39 executed: publication package scanned (NOT pushed), V3 concept frames committed

**Ruling 37 (publication package — assembled and scanned; NOTHING
PUSHED).** The would-be push is every local commit on `main` from the
85aabbf scaffold to this one, to origin github.com/Umaraslam66/glassbox
(`git push origin main`). Range verification: the range touches only
`mobility/` plus the single root-README pointer line;
`PRD_MOBILITY.md` is absent from EVERY commit tree in the range
(checked tree-by-tree, not diff-by-diff); no tracked file under
`mobility/data/`; `.env` untracked and ignored. Content scan of every
distinct blob version in the range (155 text blobs pre-concepts + this
commit's additions) and all commit messages: zero API-key-shaped
strings, zero `.env` secret values (matched programmatically, three
secret values, never printed), zero absolute local paths, zero personal
emails in file content. All committed images inspected visually: pure
viewer/concept canvas output, no browser UI, no personal tabs. On the
record for the owner: (a) commit author/committer on all commits is the
same public git identity already on origin's existing commits; (b) the
predecessor's Gate M5-final log entry names the remote URL (the repo's
own address — harmless, noted for completeness); (c) a mouse cursor is
visible in some V2 screenshots (cosmetic). Push requires the owner's
explicit word; final test verification at this commit is below.

**Ruling 38 held.** No browser tools this session. All V3 concept
frames were rendered headlessly (node-canvas from npm) and judged by
reading the rendered files back.

**Ruling 39 (V3 concept phase — three directions, committed to
`results/v3_concepts/`, STOPPED for the owner's pick).** All frames
render exclusively from recorded artifacts: the system-side trajectory
log (five fields), the real Stage-M1 biography cards (post-study demo
use per this ruling, provenance stated in a caption inside every
frame), public profiles, and numbers computed from them. No invented
thoughts, dialogue or activities anywhere; avatars are monogram chips
derived from id + profile, stated as such. Every numeric claim in
every frame was re-verified against the extract before acceptance.
- **A — the living town** (`A1_day1_0815`, `A1b_day1_0835`,
  `A2_day20_0815`, `A3_day12_0534`): 300 houses grouped by recorded
  area type (91/115/94), windows lit by recorded departure/arrival,
  queue at the bridge reconstructed to land on exact recorded
  arrivals, office windows lighting per arrival. A1b exists because
  the design brief's assumption "day-1 jam at 08:15" was wrong in the
  data (4 queued at 08:15; the queue peaks at 77 cars at 08:35 —
  consistent with capacity 5.25/min × 14.8 min max delay); both the
  briefed frame and the true peak were rendered, both captioned
  honestly.
- **B — the commuter diary** (`B1_lukas_day12`, `B2_discovery_day2`,
  `B3_lena_day20`): corridor scene beside a traveler dossier —
  verbatim card excerpt (non-adjacent joins marked […]), 20-square
  route ribbon, data-diary lines computed from that traveler's
  recorded minutes, cast strip. Casting deviation, flagged: B2
  features t0122 (first_day_on_B = 2, day-1 queue 14.7 min — 4th
  largest) instead of one of the three earliest adopters, whose day-1
  queues (0.6–1.6 min) offer no before/after contrast.
- **C — Twenty Mornings, One Wall** (`C1_mural_full`, `C2_mural_act1`,
  `C3_mural_earlyrisers`): the whole run as one poster — 20 morning
  strips of arrivals, route A above / ring road below each line,
  named threads (t0287's −245 min ratchet, a 6-switch zig-zag, a
  steady 08:25-every-day traveler), margin notes only where the log
  records something (58 discover the ring road on day 2; peak
  crowding 38.7% day 2 → 29.3% day 20; nobody arrives for 2 h 12 min
  on day 20).
- Data-integrity note: the session brief quoted t0287's day-20
  departure as ~05:56; the recorded value is 03:56 (dep drift
  −245 min). All three directions computed from the data and carry
  the correct value. Renderer scripts and PNG masters remain in the
  session scratchpad (not repo material); frames are reproducible
  from the trajectory log + cards by the same recipe.

**Zero study-API spend this session** (all rendering local; grand
total unchanged ≈ $2.34). **STOPPED at Gate M5-final-2:** awaiting the
owner's direction pick for V3 and the explicit push order.

## 2026-08-04 — Gate M5-final-2 reviewed: Rulings 40–43 received

**Ruling 40 (tooling amendment RECORDED).** The concept-phase use of
parallel design subagents is an owner-authorized amendment for VISUAL
WORK ONLY — zero study-API spend, every frame number verified against
the trajectory log in-session before acceptance. The solo rule stands
unchanged for all study and analysis work. Brief-correction note, on
the record: the session brief circulated t0287's final departure as
~05:56; the recorded value is 03:56 (departure drift −245 min); all
three concept directions computed from the data and rendered the
correct value.

**Ruling 41 (repo restructure APPROVED; parent read-only lifted for
one structural commit only).** Two top-level studies — `glassbox/`
(parent) and `mobility/` — root reduced to README (rewritten landing
page), LICENSE, .gitignore, .claude/, shared pytest config. One
commit, pure `git mv`; only import/path/config fixes allowed, every
edited file listed, wall-test diffs shown verbatim; full verification
at the new HEAD (parent suite identical at 726, mobility 28 +
expected skips, both wall tests green, both PRDs absent from all
commits, untracked material still untracked, exclusion scan re-run on
changed blobs). Any failed check stops everything before pushing.

**Ruling 42 (push AUTHORIZED, conditional).** If every Ruling-41
check is green: `git push origin main`, full range through the
restructure commit — the owner's explicit word, recorded here.

**Ruling 43 (V3 direction DECIDED: the composite).** A = interactive
living-town viewer (hero); B = the traveler dossier as its inspector
layer; C = the mural set shipped as high-res posters in
`results/posters/`. Build rules: headless iteration per Ruling 38;
subagents per Ruling 40; every element data-driven with the
provenance footer everywhere; no invented content; no new study-API
calls; the concept round's B2 casting deviation gets corrected or its
caption rewritten to what the featured traveler's log supports; a
tracked published dataset under `mobility/app/data/` with exactly the
viewer's fields (trajectories, public profiles, card text — planted
parameters EXCLUDED, proven by a field-level check); 60fps target,
scrub + play, guided story beats, share mode, static-file open.
Finals + screenshots committed; second push plan produced; SECOND
PUSH ONLY ON THE OWNER'S WORD.

## 2026-08-04 — Rulings 41–42 executed (restructure + FIRST PUSH), Ruling 43 built: GATE V3

**Restructure (Ruling 41) committed and verified.** Parent study moved
to `glassbox/` by pure `git mv` (142 R100 renames); root reduced to
the rewritten landing README, LICENSE, .gitignore, .claude/, shared
pytest.ini. Every content-edited file, complete list:
`glassbox/tests/test_wall.py` (8 path-constant substitutions — the
test re-anchored to the true repo root so its repo-wide logprob and
truth-path bans still cover mobility, which Ruling 4's config
mechanism depends on), `mobility/tests/test_wall_mobility.py` (parent
grader exemption tuples gained the `glassbox/` prefix),
root `pytest.ini`, `mobility/pytest.ini`, `mobility/tools/m0_pilot.py`
(sys.path bootstrap only), `.gitignore` (npz pattern re-pointed),
root `README.md` (new landing page; the old README moved
byte-identical to `glassbox/README.md`). Verification at the new
HEAD: parent suite 726 passed — identical count; mobility 28 + 1
expected skip; both wall tests green; both PRDs absent from all 34
range commits (tree-by-tree); `.env`, npz, LEONARDO docs,
`glassbox/data/`, `graphify-out/` untracked; exclusion scan clean.

**FIRST PUSH executed (Ruling 42, owner's explicit word):**
`98e2976..d61e531 main -> main`, 34 commits, live at
github.com/Umaraslam66/glassbox.

**Ruling 43 built — V3 composite (this commit).**
- **Published dataset** `mobility/app/data/` (tracked; re-included in
  `mobility/.gitignore` so the change stays inside mobility/):
  trajectories.jsonl (byte-identical to the run log, sha256 8059c84d…),
  profiles.json (300 public profiles), cards.jsonl (pid + name + the
  real Stage-M1 card text, each byte-equal to its leak-checked
  original). **Field-level leakage check: ALL 8 CHECKS PASS** — exact
  field allowlists on all three files, banned planted-truth keys
  (phi/wobble/PAT/loadings/…) absent at every depth, scalar-only
  values, 300-pid consistency. The app works from a public clone with
  no untracked files.
- **The viewer** (`mobility/app/`): shared renderer `town.js` (no DOM
  dependency — the same module renders in the browser and in the node
  verification harness, per Ruling 38), rewritten `viewer.js` /
  `index.html` / `style.css`, `make_bundle.mjs` +
  generated `data-bundle.js` (file:// fallback; embeds the three
  published files as exact bytes with a sha256 manifest — verified
  independently byte-identical). Living-town hero scene; five guided
  story beats (jam → discovery → Hans's switchback → the ratchet →
  spread); click-a-house/car dossier (monogram, profile, verbatim
  card excerpt with adjacency note, computed data-diary, 20-square
  route ribbon); day scrub/play/speed/keyboard; share mode; PNG
  export. EVERY displayed number is computed from the three data
  files at load time — no literal results in the app.
- **Honesty details:** the bridge tag shows observed throughput from
  the log (5.3/min day 1) — the study's 5.25/min capacity is not
  derivable from the published fields and is not displayed; derived
  durations round at the data's own precision (Hans 14.7 min, joint
  4th of day 1, computed with tie-aware ranking); casting fix per
  Ruling 43c — the discovery beat features the actual first three
  onto the ring road (t0116/t0190/t0366 by recorded departure) and
  Hans's beat claims only what his log supports. Provenance footer on
  every frame, in share/export output, and as page HTML.
- **Performance (headless proxy, node-canvas CPU rasterization,
  1920×1080):** busiest static frame 13.8 ms mean (p95 14.2) after a
  sprite mip-ladder + cached light/chrome rebuild; playback 23.4 ms;
  day-20 5.3 ms. Browser GPU compositing has substantial headroom on
  the 60 fps target; measured in-browser numbers await any live
  session the owner runs.
- **Posters (finisher set)** in `results/posters/`: mural_full.png
  3000×4384, mural_act1.png and mural_earlyrisers.png 3840×2160 —
  re-rendered at scale from the approved concept pipeline with a
  byte-for-byte K=1 control against the approved renders; the only
  content change is the slug line (concept IDs dropped), proven by a
  pixel-region diff confined to the slug bounding box.
- **Screenshots:** `results/v3_*.jpg` — the five beats, mid-scrub,
  share mode, and the open dossier, all rendered headlessly by the
  harness through the shared renderer.

**Tests at this commit:** parent 726; mobility 28 + 1 expected skip;
both wall tests green (the app is JS — no new Python); 43d leakage
check ALL PASS. Zero study-API spend; grand total ≈ $2.34.

**GATE V3: HARD STOP.** Committed, NOT pushed. Second push plan
produced and awaiting the owner's word.
