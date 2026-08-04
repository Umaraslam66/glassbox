# GLASSBOX

Recovery studies for LLM-simulated people. The method is the same in both
studies: build synthetic people whose true traits we planted ourselves, hide
that truth behind a wall, let an LLM render each person and answer in
character, and then test whether standard estimation tools can recover the
planted truth from the answers alone. Believability is not evidence;
recovery is.

Two studies live here:

- **[`glassbox/`](glassbox/README.md)** — the parent study (finished,
  audited): survey personas, 30 planted traits, an interview pipeline, and
  the original Wall discipline. Start with
  [`glassbox/RESEARCH_SUMMARY.md`](glassbox/RESEARCH_SUMMARY.md).
- **[`mobility/`](mobility/README.md)** — GLASSBOX-Mobility: the same
  harness turned on LLM travel agents — planted traveler parameters,
  econometric recovery, calibration, elasticity realism, and a 20-day
  congestion mini-world. Start with
  [`mobility/RESEARCH_SUMMARY_MOBILITY.md`](mobility/RESEARCH_SUMMARY_MOBILITY.md).

Run the tests for both studies from the repo root:

```
python -m pytest
```
