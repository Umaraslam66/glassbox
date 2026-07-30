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
