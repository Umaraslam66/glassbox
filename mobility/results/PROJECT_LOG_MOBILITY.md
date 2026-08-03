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
