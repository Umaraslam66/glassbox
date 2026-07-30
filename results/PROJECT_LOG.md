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
