# Stage 6 notes (forward-looking only — nothing here is actioned before Stage 6)

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
