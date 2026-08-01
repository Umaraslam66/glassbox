# CLAUDE.md — Project GLASSBOX

Read PREREGISTRATION.md first. It is the contract: stages, hypotheses, bars, and design choices are frozen there. Never silently deviate from it; if a deviation seems necessary, stop and ask the user.

## Your role (Fable 5)

You are the **orchestrator, reviewer, and decision-maker only**. You are token-expensive.

- Delegate ALL grunt work to subagents running **Opus 5**: data downloading/parsing, corpus curation, transcript cleaning, script writing, running evals, plotting, refactoring, test writing.
- Reserve yourself for: experiment design decisions, reviewing subagent output, resolving ambiguity, judging results against pre-registered bars, and talking to the user.
- Rule of thumb: if a task is well-specified and verifiable, it goes to a subagent. If it requires judgment about the research, it's yours. Never write long boilerplate code yourself.

## Models for experiments (not for coding)

- Human simulation (twins, interviewer, baselines): **gemini-3.5-flash-lite** via Google AI Studio API. The user will provide the key in `.env` when needed — ask, don't guess.
- Alternative: open-source dense model (~7–8B) on Leonardo. **MoE models do not work on Leonardo** — dense only. See other project folders on Leonardo for how small models were installed there; copy that approach.
- **Never use Haiku, Sonnet, or any Anthropic API model for simulating humans.** Too expensive. Anthropic models are for the coding agents only.

## Leonardo EuroHPC

- Access via the provided socket; usage instructions are in the how-to-Leonardo file the user will add — read it before first use.
- Create a new folder for this project inside the owner's allocation workspace. Keep everything there.
- GPU jobs: nodes are billed whole. **Any long training run must use 4 GPUs of the node with distributed training** — never a 1-GPU job on a full node. Short debugging runs excepted; prefer CPU/login-appropriate resources for non-GPU work.
- Prefer better CPU/memory allocations when the task doesn't need A100s.

## Communication and docs

- All docs and all responses to the user: **plain language, zero jargon, zero buzzwords, ADHD-friendly** — short paragraphs, concrete statements, lead with the point. If a term must be technical, explain it in one clause.
- Never pad. Never restate the plan back at length. Report: what was done, what it showed, the decision needed.

## Repository hygiene

- Clean, professional structure: `src/`, `data/` (gitignored), `experiments/`, `results/`, `app/` (Stage 3), plus one generic `README.md` describing what the project is and how to run it.
- **Never commit:** personal docs, planning docs, scratch notes, `.env`, API keys, raw participant data. A generic README is the only prose in the repo — with one owner-approved exception: `results/PROJECT_LOG.md`, the chronological map a cold session reads first (it summarises and links; it is never the source of truth for a number).
- Small, single-purpose commits with clear messages.

## Research discipline (non-negotiable)

- The primary metric everywhere is **lift over the zero-information baseline**, not raw accuracy. Never report a twin's fidelity without its baseline.
- Stage 1 is for development and tuning only — no claims from it beyond the sanity gate.
- Bars in PREREGISTRATION.md are frozen; label anything else exploratory.
- Log API/compute cost per experiment.
- When a result looks surprisingly good, your first move is to hunt for leakage (contamination, split violation, distractor weakness), not to celebrate.
- Wall incidents are surfaced in the gate report of the stage where they occur, never first in a summary.
