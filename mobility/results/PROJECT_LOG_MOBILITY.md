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
