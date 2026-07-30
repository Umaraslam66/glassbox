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
