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
