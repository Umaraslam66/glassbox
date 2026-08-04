# Stage 6 notes (forward-looking only — nothing here is actioned before Stage 6)

Superseded by results/REPORT.md; retained as provenance — these findings were
banked, with commit timestamps, before the report existed.

Recorded 2026-07-31, on owner instruction at the Gate 2 fix round.

The Stage 6 HUMAN_PROTOCOL will include, alongside the pre-registered new-panel
study, a SocSci210 section: the SocSci210 dataset (HuggingFace
`socratesft/SocSci210`, from the Stanford/Simile group) offers an
immediately-runnable real-data replication of the calibration claim — predicted
answer distributions vs real respondents' answers — at shallow per-person
resolution (each real respondent is observed on far fewer items than our
synthetic personas, so it tests calibration, not deep trait recovery). Two
conditions are mandatory if this section is written: (1) a documented
pretraining-contamination check — the survey items and responses may appear in
the training data of any LLM used, and the section is void without a check and
an honest statement of what it can and cannot rule out; (2) any numbers quoted
from the SocSci210 paper must come from the ACL camera-ready version, not
arXiv v1. Nothing else in this project touches SocSci210 before Stage 6.

## Finding recorded at the Gate 2 close (owner's wording, 2026-07-31)

Finding for the report and pitch: uncertainty calibrated only against
observable data understates error against latent ground truth when the
persona-rendering step is lossy (measured here: 88% card-induced / 12%
noise-induced). Honest confidence models need truth-anchored feedback —
connect to Simile's confidence-model direction in the writeup.

## Findings banked at the Gate 3 close (owner instruction, 2026-07-31)

1. Demographics carry ~zero trait signal in this synthetic population
   (profile-only baseline collapsed to intercept). Real-human demographics
   carry nonzero signal; the lift-over-profile bar would be harder in
   reality. Population-realism limitation.

2. Interview efficiency finding: 15 questions + 3 open answers recover 77%
   of what 252 questions know. Open answers moved point estimates strongly
   but bought only ~3 closed items of declared precision — text is
   information-rich but precision-poor under our fusion.

3. Clean negative: directly asking the system LLM to rate item
   discrimination predicts nothing (r ≈ −0.003); learned heads on
   embeddings do (0.504). Keep for report.

4. Coverage decay curve (0.746 → 0.649 → 0.358 as evidence grows
   1 → 15 → 202 items): small-N sampling noise masks the rendering blind
   spot; more data exposes it. Central exhibit for the confidence-model
   story.

## Findings banked at the Gate 4 close (owner instruction, 2026-07-31)

1. LLM given demographics only performs WORSE than knowing nothing
   (−35% lift): its priors tilt per-person in error-adding directions.
   Directly relevant against naive "just prompt an LLM" persona
   approaches.

2. Item difficulty/thresholds cannot be predicted from question text
   alone (zero-shot thresholds collapse to 2.7% lift); they must be
   learned from response data. Cold-start implication for anything
   built on this.

3. Aggregate prediction is dramatically easier than per-cell: population
   marginals r 0.983, cross-tabs r 0.967 while per-cell correlation sits
   at 0.684. Segment-level claims are cheap; individual-level claims are
   the hard, honest currency.

4. Per-cell resolution limit: individual item-level quirks exist that no
   low-dim trait vector expresses; k-NN partially captures them on NEAR
   items. Future direction (exploratory, not built here): hybrid
   trait-model + collaborative-filtering residual.
