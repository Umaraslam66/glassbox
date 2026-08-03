# GLASSBOX-Mobility

A second recovery study on the GLASSBOX harness, aimed at LLM travel agents.
We plant traveler parameters (value of time, schedule rigidity, mode affinity,
crowding sensitivity, price sensitivity, habit strength), let an LLM render
those travelers and make travel choices in character, and grade three things:
does standard travel-demand estimation recover the planted parameters, is the
attached uncertainty honest, and do populations of these agents respond to
price and time changes within empirically documented elasticity ranges.

The contract is `PRD_MOBILITY.md` (here) and, once frozen at Gate M0,
`PREREGISTRATION_MOBILITY.md`. The parent study at the repository root is
finished and read-only; mobility code imports it unchanged or copies and
adapts, never edits.

## Layout

Mirrors the parent: `src/` (travelers, scenarios, engine, model, eval, policy),
`experiments/`, `results/` (logs live here), `app/` (Stage M4 viewer),
`data/` (gitignored), `tests/`.

## Running

```bash
cd mobility && pytest            # mobility tests
pytest tests/test_wall.py        # parent Wall test (from repo root)
```

Both Wall tests must be green on every commit. Start reading at
`results/PROJECT_LOG_MOBILITY.md`; numbers live in result files, never in prose.
