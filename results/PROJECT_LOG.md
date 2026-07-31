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

## 2026-07-31 — Gate 2 outcome ACCEPTED AS FINAL by the owner

The documented result, in the owner's fixed phrasing, to be carried into
every future report and the final REPORT.md: "2 of 4 frozen bars unmet:
trait recovery 0.798/0.7996 on RSK/SOC vs 0.80; blur coverage 0.358,
structurally attributable to card-rendering distortion invisible to the
posterior." No third card round, no bar moved, no recalibration of any
confirmatory number. The σ×2.18 coverage decomposition stays
exploratory-labeled wherever it appears. A finding derived from this
(uncertainty calibrated only against observable data understates error
against latent ground truth when the rendering step is lossy) is recorded
in results/STAGE6_NOTES.md for the writeup.

ANTICIPATION NOTE for Gate 3 (recorded before any Stage 3 measurement):
the Gate 3 blur-coverage bar (60–75% at N=15) may inherit the same
structural undershoot — the person encoder's uncertainty can only see
answer-side noise, not card-rendering distortion. If Stage 3 coverage
fails while RMSE passes, that is reported as the same phenomenon, not
patched — with the same exploratory decomposition analysis. And the
standing rule, stated here so it is never even attempted: no
truth-anchored calibration on the system side, ever — tuning any
system-side uncertainty against planted θ would be a Wall breach.

## 2026-07-31 — Stage 3 opened (translators)

Scripted interview set: already frozen at split time in
experiments/splits_v1.json (seed 2026) — 15 closed items in scripted order
(the list order is the order; manifest records
scripted_order_is_permutation), all drawn from TRAINING items
(interview_from_training_items_only: true; re-verified today against the
holdout list: zero overlap), 3 open-ended prompts (oe01, oe16, oe15).
Nothing re-drawn. Consistency rule in force for Stage 3: every closed-form
interview answer is produced by the frozen noise layer (a=1.2, b=4.0,
T_noise=16, t=0.7, per-record seeds) applied to the persona's recorded
answer-token distribution from the v2 full sweep — the interview never
re-queries Gemma on a closed item, so an interview answer can never
contradict the persona's own answer distribution. Open-ended answers are
fresh Gemma generations (persona side, batched, cached). Item encoder and
person encoder run system-side on the frozen Qwen3.6-27B; architectures
to be proposed by subagents and decided by the orchestrator before any
training.

## 2026-07-31 — Stage 3 design decided; pre-measurement projection filed

PROJECTION, FILED BEFORE ANY STAGE 3 MEASUREMENT (a prediction, not a
result): from Stage 2's own numbers, the Gate 3 person-encoder RMSE bar
(≤ 0.5 per dimension at N=15) is projected unreachable — the full
252-item fit already sits at RMSE 0.458–0.580 with 5/8 dimensions above
0.50, the transmission ceiling puts 4/8 dimensions above the bar even
with infinite interview data, and Fisher-information plus a direct
15-item MAP scoring of held-out personas (two independent methods that
agree) put N=15 at RMSE 0.70–0.81 on every dimension. Projected lift
from closed items alone: ~23% vs the 30% bar (the open answers are the
only legal lever); projected coverage ~0.60 — near the band floor, with
the anticipation-note blind spot expected to surface as N grows rather
than at N=15. Bars do not move; the stage runs and is measured against
them as frozen. Exploratory Stage-5 note: a D-optimal 15-item set would
carry ~33% lift vs the frozen set's ~23% — the frozen interview set
stays frozen; this is headroom for the adaptive interviewer, nothing
else.

ORCHESTRATOR WALL RULINGS: (1) the PREREGISTRATION §2 pole descriptions
are committed public design text and may be used as system-side prompt
scaffolding — guarded by a byte-identity test against the frozen file,
and never reworded in response to encoder errors; (2) the Gate 3 item
bars are graded against the FITTED parameters of held-out items (the
PRD's wording and the only Wall-clean reading; cosine is
rotation-invariant so all system-side work stays in the fitted basis —
the truth-derived Procrustes rotation is off-limits system-side);
(3) the system-side logprob ban stands — encoder confidence comes from
sampled-score spread, not token probabilities.

OUTCOME ADDENDUM (written after measurement — see the Gate 3 entry
below): the projection held on RMSE (measured 0.627–0.817 closed-only,
within 0.083 of the filed number on every dimension) and was corrected
upward on one input: its r(θ̂₁₅, θ̂_full)=0.758 shared noise with its own
reference; on an independent interview sitting the true number is 0.686
(closed-only) / 0.770 (fused).

ARCHITECTURE DECISIONS (from the proposal in the session scratchpad):
item encoder = hybrid — prompt-elicited Qwen judgments against the
eight pole descriptions → small ridge head for loading DIRECTION;
a separate head on judgments + reduced embeddings + wording features
for discrimination MAGNITUDE (a design-class oracle only reaches
r≈0.57 vs the 0.60 bar, so magnitude is the risk and is measured, not
assumed). Person encoder = MIRT posterior over the answered closed
items as the backbone (their fitted parameters are system-side
quantities; blur monotonicity in N holds by construction) + the three
open answers fused as a learned pseudo-item block in the likelihood
(trained on the 400 training personas against fitted θ̂; blur
calibrated on training-split residuals only). Full-LLM extraction and
amortized inference are ablation arms, not the primary. Qwen3.6-27B has
never run on this cluster in this project: the first Stage 3 GPU job is
a smoke test.

## 2026-07-31 — Stage 3 run: Gate 3 verdict — 2 of 5 bars FAIL, projection confirmed

Machinery (all committed): consistency-rule interview answers (Gemma never
re-queried on closed items; the interview is a genuinely fresh noise
sitting — 33% of the 15 cells differ from the main round), 1,500 open
answers (zero leak rejects), Qwen's first runs in this project (thinking
channel closed via a --no-thinking driver flag; Gemma path byte-identical
with the flag off — verified), 23,340/23,340 pole judgments parsed clean,
item embeddings via vLLM pooling.

ITEM ENCODER (results/stage3_item_encoder.json): median held-out loading
cosine 0.861 PASS (bar 0.7; pre-filed projection band 0.80–0.93; leakage
hunt on the passing bar clean — topic-domain-only reaches 0.265, barely
above a no-text constant). Discrimination r 0.504 FAIL (bar 0.6; raw
scale confirmatory by orchestrator ruling, log-scale 0.577 reported
alongside — fails on both). The sharper caveat: it is a weak/strong
detector, not a graded predictor — excluding the weakest-fitted quintile
collapses r to 0.231. Clean negative worth keeping: asking the model
directly how much people would disagree about an item predicts fitted
discrimination at r −0.003.

PERSON ENCODER (results/stage3_gate3.json; fused = closed posterior +
open-answer pseudo-item block, confirmatory): per-dimension RMSE vs
planted θ FAIL on 0/8 dims (worst TRD 0.788, best TEC 0.559, bar 0.5) —
exactly as the pre-measurement projection filed above said before any
compute was spent. Blur coverage 0.649 PASS (band 60–75) — the
anticipation note did NOT fire at N=15; its mechanism is visible anyway:
coverage decays 0.746 → 0.649 across N=1→15, heading toward the
full-matrix 0.358. Monotone blur shrinkage: ORCHESTRATOR RULING — PASS
under the frozen "in expectation" wording (the mean-blur curves shrink
by 0.19–0.42 with worst uptick 0.00228, i.e. 0.9% of that dimension's
shrinkage; the fixed-θ control is exactly monotone, so upticks are MAP
relocation, not information loss). The strict per-step reading (15.2% of
individual steps rise) is reported in full alongside; like the Gate 2
weak-item check, this is a grader's reading of qualitative frozen
wording, and both readings are preserved so the call can be re-made.
Lift over the public-profile baseline: 26.8% FAIL (bar ≥ 30; closed-only
21.4). The baseline is intercept-only (profile carries ~zero trait
signal), so this bar coincides with lift over zero information.

STOP-AND-INVESTIGATE (the lift bar's own clause) — investigation run and
concluded: the encoder is not broken and the transcript is not failing
to add signal (fused r to the full-matrix estimate is 0.770; the fused
point estimate reaches r 0.71 max against planted per-dimension). The
shortfall is the accepted card-transmission ceiling downstream: even the
FULL 202-item answer matrix only achieves pooled RMSE 0.529 (lift ~44%),
so 15 closed + 3 open reaching 26.8% is the expected fraction of a
ceiling that Gate 2 already documented, not a new defect. A leakage hunt
fired on the other side (TEC and TRU beat the filed projection by >0.10)
and came back clean on five checks; the open-answer channel — built for
exactly this — is what produced the beat, and it is uneven across
dimensions (worth 0.3 to 9.2 closed items depending on axis; ~3 items
pooled). The three open answers move the point estimate strongly
(r 0.686 → 0.770) while buying only ~3 items of declared precision.

Costs, Stage 3: 23.39 GPU core-hours (open answers 4.28; Qwen smoke +
embeddings 10.05; judgment batch 9.06) + local CPU at zero. Gate 3 is a
hard stop per the owner's order: verdict delivered, decision with the
owner.

## 2026-07-31 — Gate 3 outcome ACCEPTED AS FINAL by the owner

The documented result, in the owner's fixed phrasing, to be carried into
every future report and the final REPORT.md: "3 bars unmet (item-encoder
discrimination 0.504 vs 0.60; person-encoder RMSE, 0/8 dims under 0.50;
lift 26.8% vs 30%), with the RMSE and lift shortfalls attributable to the
accepted card-transmission ceiling: the full 252-item matrix itself
reaches only RMSE 0.529 / 44% lift against planted truth. The 15-question
transcript recovers r 0.770 of the full-matrix estimate." The
monotonicity ruling (PASS under the frozen "in expectation" wording, both
readings preserved) and both Gate 3 leakage hunts stand as logged. Four
findings banked in results/STAGE6_NOTES.md on owner instruction:
demographics-carry-no-signal population-realism limitation; the 15+3
interview recovering 77% of the full matrix; the clean negative on
direct-elicitation of discrimination; the coverage decay curve as the
confidence-model exhibit.

## 2026-07-31 — Stage 4 opened (end-to-end prediction + calibration)

Why this is the headline gate: Gate 4 bars grade against each persona's
TRUE ANSWER DISTRIBUTION (50 seeded noise-layer applications per cell,
frozen §5 addendum), not planted θ — so the card-transmission ceiling
that capped Gates 2–3 largely drops out. These bars are meant to be
passable at full strength.

Probe stratification verified frozen and committed with the Stage-2
splits (commit 2ea1f72, seed 2026): near 13 / same-domain 19 / far 18
over the 50 held-out items. No fix needed.

Confirmatory path: held-out persona → N=15 fused transcript θ̂ (Stage 3
person encoder, cached) → held-out probe items with ITEM-ENCODER
zero-shot loadings (never fitted loadings for held-out items) → ordinal-
logit predictor → predicted distribution per cell. Mandatory baselines,
all from legitimate system-side data only: population marginal (training
personas), profile-only, k-NN matched on the 15 interview items (k chosen
on training data, recorded), and Qwen given profile but no interview.
Frozen bars: Brier lift ≥ 25% relative vs marginal AND beats k-NN;
ECE ≤ 0.05 (10 bins); corr(predicted prob, empirical 50-sample
frequency) ≥ 0.9 across cells. Reported, no bar: per-stratum
generalization curve, aggregate marginals + 2-way cross-tabs with error
bars, exploratory θ̂-error vs loading-error decomposition.

## 2026-07-31 — Stage 4 run: Gate 4 verdict — 2 of 3 bars PASS, correlation fails structurally

Machinery (all committed): 50-sample true answer distributions per held-out
cell (250,000 seeded draws of the frozen noise layer, round tags
truth50_s01..s50, byte-reproducible, eval-side only, stored under
data/truth/runs/truth50); graded-response predictor over the fused N=15
person-encoder posterior (512-draw posterior integration) with ITEM-ENCODER
zero-shot loadings for all 50 probe items — probe THRESHOLDS fitted on
training personas' answers with loadings and training θ frozen
(orchestrator decision: threshold information equals what the marginal
baseline already sees; the person-specific signal stays on the zero-shot
path; the fully-zero-shot text-threshold arm is exploratory and collapses
to 2.7% lift); all four mandatory baselines. Integrity: item-encoder re-run
byte-identical to Gate 3 (0.861/0.504), predictor matches mirt.py to
1e-10, 619-test suite green at grading (620 with the dashboard page).

GATE 4 (results/stage4_gate4.json, confirmatory): primary lift PASS —
Brier 0.0627 vs marginal 0.0860, 27.1% relative (bar ≥ 25%), and beats
k-NN. Per the frozen doc, prominently: k-NN alone reaches 24.4% lift; the
model's whole margin over it is 3.6% relative. The stratum lens is the
redeeming pattern: k-NN wins NEAR (24.9 vs 22.6), the model wins
same-domain (29.3 vs 26.6) and FAR (28.0 vs 21.5) — the opposite of
interview memorization; the encoder earns its keep exactly where it
should. ECE PASS — 0.0137 (bar ≤ 0.05); posterior integration halves ECE
vs plug-in (0.0255 → 0.0137). Correlation FAIL — pooled r(predicted prob,
empirical 50-sample frequency) 0.684 vs bar 0.90 (binary 0.700, likert
0.615).

ATTRIBUTION of the correlation fail (exploratory, results/
stage4_corr_attribution.json; sanity-locked to the grader's pooling at
1e-9): the bar is unreachable for this model class — the ceiling arm
(full 202-item θ̂ + FITTED holdout item parameters) reaches only 0.786;
the truth-side oracle reaches 0.9925, so the metric itself is sound. The
oracle-to-model gap telescopes exactly: 66.9% the 8-dim model class
(item-level idiosyncrasy an 8-trait vector cannot express), 18.2% N=15
information, 14.9% zero-shot item encoder. Closing everything the
project controls still lands ~0.11 under the bar. The pooled form is
also 81%-saturated by item-level structure the zero-information marginal
already captures (marginal rung 0.554 of the model's 0.684; frequency
variance splits 32% between-item / 68% between-persona). Predictions are
near-unbiased (OLS slope 0.937, intercept 0.015) — which is why ECE
passes while correlation fails: the arm is noisy, not miscalibrated.

QWEN NO-INTERVIEW BASELINE (job 51349316, 7.12 core-hours, 15,000/15,000
parsed, wall-audited prompts): Brier 0.1159 — LOSES to the
zero-information marginal by 35% relative (corr 0.336). Clean negative
for the report: an LLM given only demographics is worse than predicting
the population distribution — it tilts per-person in directions that add
error, consistent with demographics carrying ~zero trait signal here.
Profile-only ridge confirms from the other side: +0.1% lift,
indistinguishable from the marginal.

AGGREGATE CHECK (reported, no bar): predicted vs true population
marginals r 0.9826 (mean abs gap 0.028; 78% of entries inside the true
95% bootstrap band); 1,225 two-way cross-tabs r 0.9671 — both sides
computed under the same conditional-independence assumption, stated on
the artifact. The high marginal r is threshold information shared with
the marginal baseline, not persona-level signal.

LEAKAGE: no hunt fired. The grader's own triggers (pooled corr > 0.95 or
lift > 45%) did not trip; the failed bar is low, not suspiciously high;
nothing beat expectations. Exploratory 2×2 decomposition: θ̂-error and
item-side error contribute almost exactly equally to the Brier gap
(0.010–0.012 each, interaction −0.002); the full-information arm reaches
0.0416 Brier / 51.7% lift — the Gate-2 ceiling expressed in answer space.
Oracle Brier 0.0016 sits on the analytic 50-sample noise floor 0.0016.

Dashboard page 4 (Calibration) live: verdict table with honest FAIL
banner, reliability diagram + per-arm table, per-stratum lift chart with
the k-NN reading, aggregate scatter, Qwen panel. 620 tests green.

Costs, Stage 4: 7.12 GPU core-hours (Qwen baseline job 51349316) + local
CPU at zero. Project total 103.8 core-hours. Gate 4 is a hard stop:
verdict delivered, decision with the owner.

## 2026-07-31 — Gate 4 outcome ACCEPTED AS FINAL by the owner

The documented result, in the owner's fixed phrasing, frozen-bar outcome
first, to be carried into every future report and the final REPORT.md:
"2 of 3 frozen bars pass (Brier lift 27.1% vs marginal, beats k-NN; ECE
0.0137). The correlation bar fails at 0.684 vs 0.90, structurally: the
full-information ceiling of this model class reaches 0.786 — per-cell
resolution beyond an 8-dimensional trait model, attributed 67% model
class / 18% interview length / 15% zero-shot item encoder." Commentary
(never replacing the frozen-bar sentence): the two passing bars are the
ones the headline claim rests on (lift + calibration); the failing bar
measures per-cell resolution no trait-vector model of this
dimensionality reaches on this population. k-NN verdict reported
prominently as required: 3.6% relative edge overall; k-NN wins NEAR, the
model wins SAME-DOMAIN and FAR. The thresholds-from-training-answers
design decision is approved as legal (same information the marginal
baseline sees); the exploratory zero-shot-threshold collapse (2.7% lift)
stays documented. Four findings banked in results/STAGE6_NOTES.md on
owner instruction (LLM-worse-than-nothing; thresholds-need-response-data
cold-start; aggregate-vs-per-cell asymmetry; per-cell resolution limit
and the hybrid CF direction).

## 2026-07-31 — Stage 5 opened (adaptive interviewing + RL); owner rulings recorded BEFORE any Stage 5 compute

RULING 1 — frozen-target attainability. The frozen Gate 5 target
(the Gate-3 accuracy level: per-dimension RMSE ≤ 0.5) may be unreachable
at ANY N, since the full 252-item matrix reaches only 0.529. The frozen
bar is reported exactly as written (questions-to-target; if no strategy
ever reaches the target, that is reported honestly — the bar is then
unmet/undefined, not reinterpreted). ALONGSIDE it, PRE-DECLARED HERE as
labeled-exploratory, before any results exist: (a) full efficiency
curves — truth-side RMSE (graded by src/eval) vs N for every strategy;
(b) a questions-to-target grid at attainable thresholds (RMSE ≤ 0.60,
0.65, 0.70, all dims), with the ≤50%-of-random comparison computed at
each; (c) fraction-of-full-matrix information reached at N = 5, 10, 15,
25. This pre-declaration is what makes the exploratory analysis
legitimate.

RULING 2 — reward is system-side only. The confirmatory RL policy trains
on blur-based reward (declared posterior-variance reduction minus
per-question cost). Truth-derived rewards are FORBIDDEN for the
confirmatory policy — planted truth must never shape system-side
weights, even via reward on the training batch. Explicit watch for proxy
gaming (blur shrinks fast while truth-side RMSE does not follow): if
observed, it is a first-class finding for the report (proxy-reward
hacking in confidence-driven interviewing), not a bug to hide. Optional,
only if cheap: one exploratory truth-reward policy trained on the RL
batch alone, to measure the deployable-vs-oracle reward gap — clearly
labeled, never confirmatory; its training code must live under src/eval
(the Wall test forbids truth imports anywhere else) and its weights are
quarantined from every confirmatory artifact.

RULING 3 — closed items only for Stage 5. All strategies (random, fixed
Stage-3 order, info-gain heuristic, RL) select among closed TRAINING
items only, identical treatment, no open-ended prompts in the adaptive
loop. Rationale recorded: episodes must run on recorded distributions +
the frozen noise layer (CPU-only, no GPU per episode), and Stage 3
measured open answers as precision-poor. Reading noted: the frozen
N=15+3 clause governs earlier stages' confirmatory interviews; Gate 5's
own wording does not pin open answers.

Execution set by the owner: mint the RL TRAINING BATCH per frozen §2
(fresh personas, recorded new seed, v2 two-pass card writer, leak
checker, full closed-bank GPU sweep with recorded distributions); run
the five Gate-1 QA metrics on the batch as a sanity report (no
re-freeze, no retuning of the frozen noise layer). Confirmatory eval
personas remain the ORIGINAL Stage-2 held-out set — the RL policy never
touches them in training. Sequential posterior machinery is pure CPU
math, no LLM in the loop; RL starts simple per PRD (bounded effort;
heuristic shipped with the RL attempt documented if RL is unstable).
GPU only for the new batch's cards and sweep; RL training and all
episode simulation on CPU.

RULING ADDENDUM (owner clarification, recorded before any Stage 5
compute): the sequential posterior, info-gain heuristic, and RL state
use the Stage-2 FITTED loadings/thresholds for the closed training items
in the adaptive loop — the same machinery as the Stage-3 person-encoder
backbone, preserving comparability with the Gate-3 accuracy level that
defines the frozen Gate 5 target. The earlier execution note's
"item-encoder loadings" applied to Stage 4's held-out probes, where
zero-shot was mandatory. One labeled-exploratory arm reruns the loop
with item-encoder-predicted parameters as the product cold-start
comparison.

## 2026-07-31 — Stage 5: RL batch minted in band; sequential machinery built; non-RL strategies measured

RL TRAINING BATCH (per the owner's execution order): 500 fresh personas,
mint seed 549, new pid namespace r0001–r0500 (factory gained a
--pid-prefix flag; ids disjoint from p0001–p0500 in filenames, resume
caches, and every per-record seed hash — seed-base 548 unchanged for
that reason). v2 two-pass cards: 500/500 ingested, ZERO leak rejects,
no retry round. Full sweep 141,000 cells, 0 parse rejects, 0 missing
logprobs. QA sanity (results/rl_batch_qa.json, frozen layer unchanged):
all five Gate-1 metrics in band and within a few thousandths of the
confirmatory batch (max pairwise 0.746, median 0.385, PR 34.0,
obedience median 0.834 all dims ≥ 0.766, retest 0.798). Raw retest
0.983 sits above band exactly as in v1/v2 — same honesty note; watch
items q246/q044 behave identically. Truth tree at data/truth_rl/
(separate directory, same layout — protects the seed-548 manifest from
overwrite). Cost 20.0 core-hours across three GPU jobs, of which 4.60
was a staging mistake on the first card job (pass 2 had no input; the
resume-safe driver made recovery one engine init). Confirmatory eval
personas remain the original held-out 100, untouched by anything here.

SEQUENTIAL MACHINERY: incremental Bayesian posterior reproduces the
Stage-3 closed-only backbone at every N=1..15 to ~1e-10 (mandatory
integrity test, both Hessian modes; the analytic form is the sweep
default, validated against the Stage-3 central-difference recipe).
Episode simulator = frozen noise layer on recorded distributions with
per-episode round tags (fresh sitting per episode, Gemma never
re-queried); strategies are architecturally blind to everything but
(chosen item, returned answer) — enforced by tests including an AST
check. Grader choices recorded: random = fresh permutation per persona;
fixed = the Stage-3 scripted order, censored at its own 15 items;
info-gain = A-optimality (trace) with plug-in answer probabilities;
blur-calibration constants from Stage 3 applied unchanged, never
refitted.

NON-RL RESULTS (results/stage5_strategies.json; 51 replicates,
held-out 100, N up to 150): the frozen Gate 5 target (all dims RMSE
≤ 0.5) is reached by NO strategy at ANY N — 0/51 replicates for
random, fixed, and heuristic; the full 202-item bank through the same
machinery bottoms out at pooled 0.525 / worst dim 0.56. Per RULING 1
the frozen bar (a) is reported exactly as written: unmet/undefined —
the comparison it asks for does not exist at threshold 0.5. The
pre-declared exploratory grid carries the efficiency result: questions-
to-target medians (random vs heuristic) 52 vs 24 at RMSE ≤ 0.70 (ratio
0.462), 81 vs 40 at ≤ 0.65 (0.494), ≥136 vs 77 at ≤ 0.60 (≤ 0.566,
conservative — 15/51 random replicates never crossed). Fraction of
full-matrix information at N=5/10/15/25: heuristic 0.37/0.59/0.68/0.79,
fixed 0.25/0.43/0.51/—, random 0.19/0.32/0.42/0.57. The heuristic
reaches random's N=52 accuracy in 24 questions and random's N=150
accuracy in ~60; the win is efficiency, not a higher ceiling (both
converge to the 0.525 floor).

PROXY-GAMING WATCH (RULING 2) FIRES BEFORE ANY RL EXISTS: declared blur
falls to 24% of its start (heuristic; 27% random) while truth-side
error falls only to 55%; the gap widens monotonically to ~0.31 at
N=150. This is the coverage-decay blind spot expressed in the adaptive
loop: the posterior prices answer noise, not card distortion, so the
deployable reward is measurably optimistic. First-class finding for
the report; full tracking arrays stored. Exploratory cold-start arm:
encoder-predicted loadings (in-sample LOO, cosine 0.868 vs fitted)
cost +0.03–0.07 pooled RMSE and move the 0.65 crossing 40 → 76
(heuristic) / 81 → 144 (random); the full-bank floor rises 0.525 →
0.550. Suite 686 green. RL trainer build in progress; confirmatory
training on the r batch is next.

## 2026-08-01 — Stage 5 run: Gate 5 verdict — frozen bars undefined at the frozen target; heuristic and RL tie on the pre-declared grid

RL POLICY (all committed; numpy only, no new dependencies): 219→64→202
MLP over (θ̂, blur, answered mask, step), REINFORCE with moving-average
baseline, entropy bonus, Adam; reward = declared posterior-variance
reduction minus per-question cost, blur-only per RULING 2 — an
AST-level test proves src/rl imports nothing truth-side. DESIGN FIX
found by the smoke run, recorded here honestly: with an undiscounted
fixed-horizon return the objective is ORDER-INVARIANT (every step's
return equals total trace removed minus what is left at the horizon;
the constant per-question cost cancels from the gradient), so the smoke
policy learned a good 25-item SET asked in arbitrary order — matching
random early. γ=0.9 discounting restores urgency; it changes return
aggregation, not the reward, so RULING 2 is untouched. The γ=1 run was
abandoned at ~1,200 iterations (0.18 core-hours, logged). Confirmatory
training on the r batch only: 3,000 iterations × 64 episodes × 25
questions = 192,000 simulated interviews, return 3.77→4.12, stable,
entropy 2.25→0.11, 0.45 core-hours. Training-batch guard checks
persona ids.

GATE 5 (results/stage5_strategies.json; 100 original held-out personas,
51 replicates, greedy policy; the five pre-existing arms reproduce
bit-exactly under the new code path):
- Frozen bar (a): UNDEFINED — no strategy (random, fixed, heuristic,
  RL) reaches per-dimension RMSE ≤ 0.5 at any N ≤ 150; the full bank
  floors at pooled 0.525 / worst dim 0.56. Reported exactly as written
  per RULING 1; the comparison the bar asks for does not exist at the
  frozen threshold.
- Frozen bar (b): UNDEFINED at the frozen target; on the pre-declared
  exploratory grid it is a TIE — 0.60: heuristic 77 / RL 84 (1.09);
  0.65: RL 37 / heuristic 40 (0.93); 0.70: heuristic 24 / RL 25 (1.04).
  Match is acceptable per the PRD; the heuristic is not beaten. Both
  adaptive arms roughly halve random (heuristic 0.462/0.494/≤0.566,
  RL 0.48/0.46/0.62 at 0.70/0.65/0.60). Pooled RMSE at N=5/10/15/25:
  RL 0.802/0.709/0.675/0.612 (best arm at its training horizon 25),
  heuristic 0.794/0.699/0.660/0.616, random 0.873/0.817/0.774/0.711.

PROXY-GAMING VERDICT (RULING 2 watch, run to ground): training on the
blur proxy widens the policy's own declared-vs-true gap (+0.14 at init
→ +0.24 trained at N=25), BUT the gap is a property of the POSTERIOR,
not of reward hacking: on identical sittings the never-trained
heuristic shows a LARGER gap than the trained policy (held-out +0.255
vs +0.239 at N=25) — the policy ends with the same error and MORE
declared uncertainty. The Stage-2/3 card-transmission blind spot is
the whole story; RL adds nothing malign to it. And the oracle-reward
exploratory arm (identical trainer, truth-side error-reduction reward,
weights quarantined with a loader-refusal test) lands within ±0.024
RMSE of the confirmatory policy at every N (max gap at N=5; 0.003 at
N=25 — corrected from a first-report ±0.014 that failed recomputation
against the watch files; the result file wins) — THE DEPLOYABLE PROXY
REWARD COSTS NOTHING MEASURABLE. Together: the proxy is honest about
ordering decisions and dishonest only about absolute confidence,
exactly as the pre-registered watch framed it.

ADDITIONAL FINDING: greedily deployed, both trained policies converge
to a near-STATIC 25-question order (the heuristic spreads 118 distinct
items over the same personas) — the RL win is "learned a good fixed
order", not per-person adaptation; past its training horizon it
spreads out again (152/202 items by N=150). Kept for the report next
to the Stage-3 D-optimal note.

Costs, Stage 5: 20.0 GPU core-hours (RL batch: cards 9.35 incl. a 4.60
staging mistake, sweep 10.65) + 3.9 CPU core-hours (training 1.30 incl.
the abandoned run, grading + instrumentation 2.6) — all episode
simulation and RL training on CPU as ordered. Suite 716 green. Project
total ~127.7 core-hours. Gate 5 is a hard stop: verdict delivered,
decision with the owner.

## 2026-08-01 — Gate 5 outcome ACCEPTED AS FINAL by the owner

The documented result, in the owner's fixed phrasing, to be carried into
every future report and the final REPORT.md: "Both frozen bars UNDEFINED
at the frozen RMSE ≤ 0.5 target (full-bank floor 0.525, pre-declared in
Ruling 1); on the pre-declared exploratory grid, adaptive interviewing
reaches the same accuracy in ≤50% of random's questions at attainable
thresholds (0.462 and 0.494 at ≤0.70 and ≤0.65); RL ties the info-gain
heuristic; proxy-gaming watch fired and was attributed to the
card-transmission blind spot, not reward hacking (heuristic shows a
LARGER declared-vs-true gap than the trained policy; oracle-reward arm
within 0.024 RMSE of the deployable blur reward)."

## 2026-08-01 — Stage 6 opened (final deliverables)

Build order set by the owner: (1) results/REPORT.md — full findings,
honest scoreboard first, frozen-bar outcomes always before exploratory
context, the measured-ceiling story as the spine, the coverage-decay
exhibit as the confidence-model centerpiece, all twelve banked
STAGE6_NOTES findings plus the two Gate 5 additions (near-static RL
order — the value of adaptive interviewing here is question SELECTION,
not per-person branching; deployable blur reward within 0.024 RMSE of
the forbidden oracle reward), an honest-methods section listing every
failed round with PROJECT_LOG pointers, limitations, and the cost
table. (2) HUMAN_PROTOCOL.md — pre-registered real-human study with
power calculation for the Gate-4 lift, vehicle-map item-bank
discipline, test–retest plan, adaptive arm, predicted numbers with
intervals, and the SocSci210 shallow-resolution calibration
replication (unseen-participant split; mandatory pretraining-
contamination check before any number is trusted; Socrates-Qwen-14B
ACL camera-ready figures as the reference baseline; honest statement
of what shallow per-person data cannot test). (3) RESEARCH_SUMMARY.md
at repo root, two pages, plain language, no hype. (4) Repo release
audit + README polish + minimal dashboard landing page with gates and
verdicts. Hard stop at Gate 6 for owner review. Drafting delegated;
orchestrator owns final editorial verification: every REPORT.md number
against its result file (result file wins), every frozen-bar sentence
against PREREGISTRATION.md.
