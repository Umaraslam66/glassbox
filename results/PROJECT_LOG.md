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
