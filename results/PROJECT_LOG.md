# Project log

Chronological record of what happened, newest entry at the bottom. Read this first
in a cold session.

This log points at things: experiment configs, result files, commits. It is never
the source of truth for a number — the result file is. If the log and a result file
disagree, the result file wins.

---

## 2026-07-30 — Stage 0: repo scaffolded

Set up the repository: layout per PRD section 4, MIT license, README, gitignore
(data, secrets, and the personal Leonardo ops docs are never committed).

Built two things with actual logic:

- `src/eval/truth_store.py` — the only module allowed to read the planted truth.
- `tests/test_wall.py` — the Wall check from PRD section 6.1. Parses every Python
  file in the repo and fails if anything outside `src/eval/` imports the truth
  store or names the truth directory. Passing on this scaffold.

Also `src/llm_client.py`: one swappable interface with a Gemini implementation
(credentials from the environment, read at call time) and a stub for the local
vLLM path on Leonardo. Not called yet.

Empty so far: the module packages under `src/`, `experiments/`, `app/`.

Still open for Gate 0: Gemma smoke test on Leonardo, Gemini API path verified,
trait dimensions agreed with the owner, `PREREGISTRATION.md` frozen.

## 2026-07-30 — Stage 0: Leonardo set up, smoke test passed, API paths verified

Leonardo: created `[redacted-workspace]/glassbox/` and copied the Gemma 4
31B installation from [sibling-project] (weights, vLLM venv, CUDA compat tools, HF cache —
69 GB, byte-verified; [sibling-project] itself untouched). Smoke test job 51068113 passed on
the first submission: 4 GPUs visible, Gemma answered a role-play prompt in
character. Log: `glassbox/logs/glassbox-smoke-gemma-51068113.out` on Leonardo.
Cost: 5.74 core-hours (see COSTS.md).

API paths verified with one trivial call each: `gemini-3.5-flash-lite` (PRD
default) and the OpenRouter fallback `qwen/qwen3.7-flash` (owner-provided;
reasoning model — spends hidden thinking tokens even on trivial prompts).

System-side model investigation (PRD amendment): Qwen3.6-27B — dense (verified in
its config: no expert keys), non-Gemma family, already on Leonardo in [sibling-project],
proven there with the same venv we now run. Proposed to the owner in the Gate 0
report; not installed pending the decision.

`PREREGISTRATION.md` drafted and committed, marked DRAFT: trait dimensions +
correlation matrix (section 2) and all stage bars from PRD section 7. Awaiting
owner sign-off on (1) trait dimensions, (2) the freeze.

## 2026-07-30 — Gate 0 approved; PREREGISTRATION frozen; Stage 1 started

Owner approved Gate 0 with six amendments (diversity criteria replaced with
pairwise agreement + participation ratio; temperature freeze; fixed splits;
Qwen3.6-27B frozen as the system-side model; tightened wording; test–retest
context reporting). Amendments applied and the document is FROZEN as of today.
One orchestrator addition flagged to the owner: the obedience definition gained
"answers on negatively-loading items are sign-flipped before averaging" —
without it the measure is ill-defined.

Stage 1 work so far:
- Qwen3.6-27B copied [sibling-project] → glassbox on Leonardo, byte- and content-verified,
  zero billed compute. Note: the checkpoint is multimodal (vision configs
  present) — irrelevant for text serving, worth knowing at Stage 3 wiring.
- Persona factory (M0) built and committed: seed-reproducible sampler matching
  the frozen correlation matrix (a test parses the matrix out of
  PREREGISTRATION.md and demands exact equality), card prompts that never name
  a trait or a number, ingest with leak-based card rejection. One design fix
  during review: trait-direction wording now survives into the card prompt for
  |θ| ≥ 0.15 (4-band scheme) instead of being lost below 0.5 — direction reaches
  the card writer for 88% of trait draws instead of 62%.
- Question bank (M1) machinery built and committed; real bank generated with the
  dev API (252 closed + 20 open, validation green). Orchestrator sample review
  found 3 defects (a miskeyed item, a weak item drafted on the wrong dimension,
  a distractor with plausible traditionalism signal) → full-bank audit
  dispatched before any persona answers it.
- STANDING LEAKAGE-HUNT TRIGGER (noted for Stage 2): distractors carry real
  topic-domain tags; if any distractor fits materially nonzero discrimination
  in the Stage-2 MIRT fit, that is a leakage hunt, not a shrug.

## 2026-07-30 — Bank QA: fix round 1 FAILED verification; structured rebuild ordered

Honest record of a failed round. The first bank audit flagged 95 of 272 items
(sign errors, wrong dimensions, near-duplicates). Fixes were applied and
validated mechanically — then an independent verification pass (fresh agent,
adversarial, with a 25-item control sample of untouched items) returned NO:

- 9 of the 95 fixes were still defective (new confounds, unsupported secondary
  loadings, a distractor with price-habit signal).
- Control sample: 6/25 untouched items defective (~24%, CI 9–45%) — the first
  audit had a material false-negative rate.
- Systematic wording bias: ENV items carrying price language, RSK items set in
  tech contexts, TRD leaning on paper-vs-digital — the bank would manufacture
  the very correlations the study must prove it can recover (ENV–PRC planted at
  −0.15, RSK–TEC at +0.40, TRD–TEC at −0.35).
- Same behavioral "vehicle" loaded on different dimensions in different items
  (cheapest-X on ENV and PRC; cash-vs-bank on TRU and RSK), undermining
  separability.
- What WAS clean: zero sign/keying errors remain, public/truth files perfectly
  consistent, no leakage.

Decision: not another patch round. Full content rebuild against a vehicle map
(each behavioral vehicle belongs to exactly one dimension; banned-context lists
per dimension; a correlation-bias budget; reverse-keyed items must use distinct
vehicles). Design table, item ids, and slots preserved. Acceptance requires a
fresh full-bank verification audit — no sampling this time. Policy file kept in
the session scratchpad; rebuild log will be preserved.

Persona side unaffected: 500 personas minted (seed 548, manifest recorded);
Gemma card-writing job running on Leonardo in parallel.

## 2026-07-31 — Cards: 499/500 rendered and leak-checked

Card batch (jobs 51146579 + 51150672, 11.05 core-hours total): 500 biographies
rendered; ingest leak-checker rejected 6 that used trait vocabulary ("risk
tolerance"/"risk appetite") — exactly the leak class it exists to catch. A
fresh-seed retry cleared 5. p0314 leaked the same phrase twice; its re-render
(prompt gains a one-line vocabulary ban, new seed) is folded into the next GPU
job rather than a standalone submission — engine init dominates small jobs.
Independent re-scan of all 499 stored cards: zero leaks. Noted for the record:
demographic combinations can be odd (e.g. a 79-year-old barista with kids at
home) — no frozen bar tests demographic realism; accepted as-is.

Caveat recorded: ingest rewrites card_rejects.jsonl per run (it now lists only
p0314); the round-1 reject list lives in this log and in the raw completions.

## 2026-07-31 — Question bank PLANTED after four QA rounds

The bank (252 closed + 20 open) passed independent verification and is now
final: no further content changes without owner decision. The path there,
honestly: initial generation → audit flagged 95 → fixes failed verification
(9 bad fixes + ~24% residual defects in a control sample) → full vehicle-map
rebuild (101 items; 16 blocking residuals) → scoped final fixes (~50 items;
2 new blocking) → five micro-fixes → PASS. Cumulative: ~115 of 272 items
rewritten, 14 retagged, design table byte-identical throughout. Every round's
findings and fixes are preserved in the session scratchpad logs.

KNOWN INSTRUMENT LIMITATIONS (accepted, watch at Gates 1–2):
- Weak items q229, q234 may be non-monotonic (both poles can disagree) —
  inspect their fitted signs in Stage 2; low discrimination is expected and
  fine, a flipped sign is not.
- PRC− shares one syntactic frame ("without ever checking what it costs")
  across 4 items — possible method factor.
- LOC has a 6-item "would you live elsewhere" cluster (q170/q174/q180/q184/
  q186/q230) — within-dimension redundancy, not cross-dimension leakage.
- q006/q039 sit adjacent on the planted-zero TRU–RSK pair (bank-your-savings
  vehicle); faint opposing side-signals on q040/q196/q200 should wash out —
  check the TRU–RSK recovered correlation specifically.
- q190 brushes the ENV throwaway vehicle (ENV–LOC planted 0.00).
- q166/q145/q157 form a machine-vs-person family within TEC (all TEC-only; no
  cross-dimension harm).
- Closed-only domain split moved (technology 14 / money 22) though the pooled
  frozen table is unchanged — matters only if closed items are ever stratified
  by domain alone.
- Realistic answerability premises (home/garden, employment clusters) and
  style nits accepted as survey realism.

## 2026-07-31 — Pilot round 1: measurement bug found (seed coupling); QA early signals strong

Temperature pilot (60 personas × full bank + 30-item retest at t=0.7 and t=1.0,
jobs 51171292 + 51179813, 14.6 core-hours) returned test–retest ≈ 98% at both
temperatures — far above the frozen 70–90% band. The number is INVALID: the
batch driver passed one global seed into every request's SamplingParams, and
vLLM (verified in the installed source) seeds a per-request generator from it —
so the main and retest rounds drew identical random numbers by construction
(accidental common-random-numbers coupling). Attempt 1 also found a GPU-memory
leak between driver invocations in one job (orphaned workers after os._exit);
fixed with a free_gpus() helper in the sbatch.

Valid early signals from the same run (coupling pushes agreement UP, so these
pass conservatively): max pairwise agreement 0.837 (bar ≤ 0.95), median
pairwise 0.468 (bar ≤ 0.80), participation ratio 10.7 (bar ≥ 5), obedience
median r 0.836–0.841 with all 8 dimensions individually above 0.5. Parse
rejects 0/33,840. The responder is decisive: 63% of Likert answers are 1 or 5,
6% midpoint. Three items had zero variance across 60 personas (q044, q141,
q246) — watch at the full sweep and Stage 2.

Decision: driver gets per-record seeds (stable hash of base seed + pid + item +
round + temperature — uncoupled and reproducible) plus per-record temperature;
clean rerun at t ∈ {0.7, 1.0, 1.3} in one job before touching the wobble
mechanism. p0314's card cleared on attempt 3 with a vocabulary-ban line —
cards now 500/500.

## 2026-07-31 — Pilot round 2 (clean seeds): temperature is not the knob; noise-layer decision

Controlled rerun (same prompt files, per-record seeds, three temperatures,
job 51189257, 7.26 core-hours): test–retest 0.9778 / 0.9789 / 0.9778 at
t = 0.7 / 1.0 / 1.3 — flat, all far above the frozen band. The seed fix did
real work (3–4% of individual answers flipped vs round 1) but the pooled number
did not move: the responder is genuinely near-deterministic. Cross-temperature
agreement 0.94–0.95 confirms enormous logit gaps. All other Gate-1 bars pass in
every arm (obedience median 0.837–0.841, all dims > 0.5; PR 10.7; pairwise max
0.84, median 0.47). Two items are genuinely zero-variance in all arms (q044,
q246) — watch at the full sweep; q141 recovered variance with clean seeds.

DESIGN DECISION (to be ratified by owner at Gate 1): the prompt-side wobble
instruction measurably does nothing; responder inconsistency will instead be
injected by a seeded mechanical noise layer driven by the responder's own
answer-token distribution (recorded logprobs). Noise-event probability =
wobble × (base slip + ambivalence boost × (1 − answer conviction)); on an
event the answer is resampled from the model's own flattened answer
distribution. Why this is not a pre-registration deviation: frozen §2 defines
wobble as "how often it flips or shifts answers on items it is lukewarm
about" — the layer implements that sentence literally, with lukewarmness
measured from the persona model itself, and makes the planted wobble true by
construction. The band stays the frozen bar; layer parameters are the
Stage-1 tuning knob, recorded in the experiment config at freeze. Tuning
happens offline on recorded logprobs — no GPU cost per knob turn.

## 2026-07-31 — Full sweep: Gate 1 PASSES all five bars on 500 personas

Full sweep (job 51200138, 10.75 core-hours; 141,000 answer cells, 16,920
reused byte-identical from the pilot cache, 0 parse rejects, 0 missing
logprobs). Noise layer frozen at a=1.2, b=4.0, T_noise=16, temperature 0.7,
seed-base 548 — addendum recorded in PREREGISTRATION §5.

Gate 1, noised (confirmatory, results/full_sweep_qa.json):
max pairwise agreement 0.746 (bar ≤ 0.95) PASS; median pairwise 0.389
(bar ≤ 0.80) PASS; participation ratio 29.4 (bar ≥ 5) PASS; obedience median
0.827, all 8 dims ≥ 0.753 (bar ≥ 0.5) PASS; pooled test–retest 0.7934 in
70–90 band PASS (binary exact 0.823 vs chance 0.50; Likert ±1 0.779 vs
chance 0.52).

Raw (un-noised) comparison kept in results/full_sweep_raw_qa.json: retest
0.979 (FAIL — robots), wobble gradient +0.04 (meaningless), 267/500 personas
perfectly self-consistent, one zero-variance item (q246). After the layer:
gradient +0.627, 2/500 at perfect consistency, zero flat items.

Honesty note for any reader: the noised participation ratio (29.4 vs raw
13.0) partly measures the isotropic noise floor, not richer planted
structure. The bar is frozen and passes either way; do not cite 29.4 as
evidence of latent richness — the raw 13.0 is the structural number.

Stage-1 compute total: 43.4 core-hours across six jobs (incl. 5.69 lost to a
GPU-memory leak on one failed attempt, since fixed in the sbatch pattern).

## 2026-07-31 — Stage 2 run: Gate 2 FAILS 2 of 4 bars; cause traced to the population

Splits frozen first (seed 2026, experiments/splits_v1.json): 100/500 personas
and 50/252 items held out; scripted interview set (15 closed + 3 open) frozen
at split time; probe strata near 13 / same-domain 19 / far 18 under the
whole-interview distance rule. Wall extended before fitting per owner order:
distributions moved behind the truth store (local + Leonardo), system side
statically logprob-free, consistency rule written into the interview engine.
Mid-stage Wall fix: answers_noised.jsonl carried a "raw" pre-noise answer
field on the system side — stripped, layer patched, field-level wall test
added.

MIRT fit (pure numpy, local CPU, 59 s; 3 restarts agree to r 0.9996): Gate 2
verdict — trait recovery FAIL (5/8 dims ≥ 0.80; SOC 0.781, RSK 0.787, LOC
0.797), item recovery PASS (median per-item r 0.925), blur honesty FAIL
(coverage 0.339 vs 60–75%), weak-item ordering PASS (0.837 rank prob).

Diagnosis (verified, not assumed): the fit is at the ceiling of the answers.
Model-free obedience on the same cells is ~0.76–0.85 per dimension; the fit
matches or exceeds it everywhere; split-half reliability of θ̂ is 0.93. The
bottleneck is upstream: the persona cards transmit only ~0.85 of planted θ,
and the noise layer (needed for the frozen retest band) spends a little more.
Blur fails for the same reason — the posterior is honest about answer noise
but cannot see the card-rendering distortion, so it is overconfident about
planted θ (5.46× predicted error variance; only 5.7% removable by rescaling).
Scale identification (link_scale: pin latent variance to the N(0,1) prior,
truth-free, correlations bit-identical) approved by orchestrator — without it
MAP scale indeterminacy makes coverage meaningless (0.095).

LEAKAGE HUNTS (per §3.5, both documented):
1. TRU–RSK planted at 0.00, recovered −0.28 (held-out); model-free answer-
   score correlation −0.437 — the entanglement exists in the ANSWERS, before
   any fitting. The bank was scrubbed for this pair; the confound therefore
   entered at card generation: Gemma renders institution-trusters as
   financially cautious people (bank accounts, insurance) — a card-writer
   confound, not system-side leakage. No Wall breach.
2. Distractors: 5 of 15 fit materially nonzero discrimination (q249 ‖a‖=1.89
   loading on TEC — stairs-vs-lift reads as tech avoidance; q250 SOC/LOC).
   No planted value reached the system side (verified); the personas' card-
   driven coherence colours even "neutral" preferences. Distractor median
   discrimination 0.574 vs strong 1.957. Recorded as instrument behaviour,
   not leakage.

Also confirmed: q229/q234 fitted signs flip exactly as the bank verifier
predicted pre-fit (their designed +0.25 labels read wrong); the weak class is
"labeled weak, not written weak" (q235 fits 3.53); q044 is near-dead (465/500
answer "no"); dimensionality split-half curve peaks at d = 6–8 (planted 8
recoverable; data alone would choose 6–8). Recovered correlation matrix ≈
0.71 × planted (r = 0.77), heatmap in results/.

Gate 2 is a hard stop: decision goes to the owner (report delivered in chat).

## 2026-07-31 — Gate 2 fix round: owner decisions executed

Owner ordered, in this order: (1) regenerate all 500 cards via the PRD §10
two-pass fallback with per-dimension independence guidance; (2) correct the
q229/q234 designed labels (legitimate only because the bank verifier flagged
both BEFORE any fitting — see the KNOWN INSTRUMENT LIMITATIONS list in the
"Question bank PLANTED" entry above); (3) accept distractor discrimination as
a finding, not a defect; (4) if blur still fails, report the honest fail plus
an exploratory inflation analysis; (5) mandatory checks (TRU–RSK, full
correlation heatmap, watch items) in the report; (6) create
results/STAGE6_NOTES.md (SocSci210 forward note — nothing actioned).

Card regeneration v2 (jobs 51317905): prompts rebuilt at version 2.0 — same
planted θ (verified byte-identical for all 500 before anything was written),
seed 548, truth files untouched. New in the prompt: an independence block
(the eight sides are independent facts; one side is never evidence of
another; surprising combinations stay as given — with behavioural contrast
examples de-entangling institution-trust from financial caution in both
directions) and a vocabulary-ban rule using invented labels so the prompt
contains nothing the leak checker hunts. Pass 2: Gemma re-reads its own draft
against the same guidance, checks direction, strength and cross-bleed per
side, and rewrites. Result: 500/500 pass-1 drafts, 500/500 revisions (489
biographies actually changed, mostly full rewrites), ingest kept 500/500 with
ZERO leak rejects and zero warnings (v1: 6 leak rejects). Orchestrator spot
review confirmed the de-entangling (e.g. a persona who distrusts officials
AND insures everything, each shown through its own details). 8.00 core-hours.

q229/q234 amendment: designed loadings flipped (q229 TEC +0.25 → −0.25, q234
PRC +0.25 → −0.25), recorded in a machine-readable "amendments" block inside
bank_truth.json citing the pre-fit flag; validators extended to accept and
check the block, and the amendment log added to the private-only keys so it
can never reach the public side. Both items are TRAINING items, so the
confirmatory held-out item-recovery bar is bit-identical with or without the
correction — it is housekeeping, not a rescue. Their `negatively_keyed` flags
were deliberately left untouched (metadata-only inconsistency; obedience uses
the loading sign and excludes weak items, so no measurement reads the flag).
The grader gained an --exclude-items robustness path; results carry item
recovery both with the corrected labels (confirmatory) and with the two items
excluded (robustness) — identical on held-out by construction.

## 2026-07-31 — Full sweep v2 + Gate 1 second run: all five bars PASS

All 141,000 answer cells regenerated from the v2 cards (job 51324033, 10.52
core-hours, boost_qos_dbg after a queue-congestion resubmit; nothing reused
from v1 — the driver's resume cache keys on record identity, not prompt text,
so the sweep ran in a fresh run directory to make stale reuse impossible).
Frozen noise layer applied unchanged (a=1.2, b=4.0, T_noise=16, t=0.7, seed
548). Gate 1, noised, confirmatory (results/full_sweep_v2_qa.json): max
pairwise 0.754 (≤0.95) PASS; median pairwise 0.385 (≤0.80) PASS;
participation ratio 31.7 (≥5) PASS; obedience median 0.830, all dims ≥ 0.783
(≥0.5) PASS; test–retest 0.794 in 70–90 band PASS. Raw comparison in
results/full_sweep_v2_raw_qa.json (retest 0.981; PR 13.9 is the structural
number, same honesty note as v1). Watch items: q246 still answered "3" by all
500 raw (content-free, unmoved by new cards); q044 still near-dead (488/500
"no" raw). Note: the v2 responder is slightly MORE deterministic raw (288/500
perfectly self-consistent vs 267) — the frozen layer still lands the pooled
number in band. Obedience moved little (median +0.003 noised); the weakest
dimension RSK rose 0.753 → 0.783. 25% of raw cells changed vs v1 — the cards
are genuinely different; the transmission ceiling just did not move much.

## 2026-07-31 — Stage 2 second attempt: Gate 2 FAILS the same 2 of 4 bars

Same frozen splits (seed 2026), same fit settings, same pipeline; results in
results/stage2_v2_recovery.json (+ fit npz, diagnostics, heatmap, scatter).

Bars: trait recovery FAIL — 6/8 dims clear 0.80; RSK 0.7983 and SOC 0.7996
miss by 0.0017 and 0.0004 (v1: 3 dims under, worst 0.781). Every dimension
improved; mean r 0.819 → 0.839. Item recovery PASS (median held-out r 0.922).
Blur honesty FAIL (coverage 0.358 vs 60–75 band; v1 0.339). Weak-item
ordering PASS (same qualitative caveat as v1). The fit is at the answer
ceiling again (matches or beats model-free obedience on 5/8 dims, worst
shortfall 0.029; restart agreement r 0.9997; split-half reliability 0.933).
The dimensionality curve now peaks at the planted d=8 (v1: 6–8 tie).

Mandatory checks. TRU–RSK (planted 0.00): recovered −0.270 held-out (v1
−0.283); model-free answer-score correlation −0.375 (v1 −0.437) — the card
confound shrank ~14% but survives the two-pass writer. Correlation-structure
recovery got WORSE while per-trait recovery got better: slope of recovered on
planted 0.65 (v1 0.71), r 0.74 (v1 0.77) — the v2 cards transmit individual
traits more faithfully and between-trait structure less. q229/q234 fitted
signs now match the corrected labels (−0.495 and −0.337 vs designed −0.25);
neither is held-out, so the amendment moves no bar. q235 again fits strong
(3.74) — "labeled weak, not written weak" stands. q044 remains a dead strong
item (455/500 "no" noised) and is what drags the strong-primary minimum down.

Blur exploratory block (EXPLORATORY — no confirmatory number recalibrated):
nominal 68% coverage would need σ × 2.18 (v1 2.23); band entry ×1.89. The
overconfidence decomposes as 88% card-rendering-induced / 12% noise-layer-
induced (v1 90/10); pricing the noise-layer share alone would reach coverage
0.435 (still fail), pricing both would reach 0.748 (in band). The posterior
is honest about answer noise and blind to card distortion, exactly as in v1;
the v2 cards cut the card share only ~4% relative.

DISTRACTOR FINDING (owner decision: documented as a finding, not a defect):
distractors fit real discrimination (median ‖a‖ 0.578 vs strong 1.998; 4/15
material, top q249 "stairs-vs-lift" at 1.69 loading on TEC). Verified again:
no planted value reaches the system side; this is coherent-biography colouring
of nominally neutral preferences — the synthetic analogue of the everything-
correlates "crud factor" in real survey data, where no human's "neutral"
preferences are truly orthogonal to their traits. It goes in the final report
in those terms. The standing leakage trigger STAYS ARMED for Stages 3–5:
any distractor whose discrimination GROWS materially at a later stage, or any
new route by which distractor answers could see planted values, is a hunt.

Wobble-recovery relationship flipped to the sensible sign in v2 (wobblier
personas now recover slightly worse, r +0.23; v1 showed −0.10, unexplained).

Fix-round cost: 8.00 + 10.52 core-hours GPU + local CPU ≈ 18.5 core-hours
(orchestrator estimate was ~17). Gate 2 second attempt is the ordered hard
stop: verdict to the owner, no third attempt, no bar moves.
