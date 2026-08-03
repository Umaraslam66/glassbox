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
spend).** Population minted per the frozen design: see the entry below and
`results/m1_mint_qa.json` for every number.
