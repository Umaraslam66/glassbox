# GLASSBOX

GLASSBOX is a recovery study. We build synthetic people whose true traits we planted ourselves, hide the truth, and then test whether an interview pipeline can recover it from interview text alone.

## Why

If you want to know whether a system can "learn a person from a conversation", you have to already know who that person is. With real humans you never do — you only have another questionnaire, which is just a second noisy measurement, not the truth. That is the validation problem: every claim about modelling a person is graded against a proxy for the thing it is trying to predict.

So we make the people. Each synthetic person starts as a hidden vector of trait values (how much they trust institutions, how much risk they take, how price-sensitive they are, and so on). A language model turns that vector into a life story that shows the traits without ever naming them. The person then answers interview questions in character.

The system on the other side of the interview never sees the vector or the life story. It sees a short public profile (age, job, city size, household) and the interview transcript. From that it estimates the traits, with an honest uncertainty on each one, and predicts how the person would answer questions they were never asked. The grader compares those estimates and predictions against the planted truth — the real thing, not a proxy.

Every number is reported as a lift over a zero-information baseline (predict the population average and ignore the person). Raw accuracy on its own is never reported.

What this does not establish: these are simulated people, not real ones. A pipeline that recovers planted traits from a language model's role-play has not been shown to recover anything from a human. That is what the follow-up study in `HUMAN_PROTOCOL.md` is for.

## The Wall

The Wall is the rule that keeps the study honest: the planted truth lives in one folder, and exactly one module (`src/eval/truth_store.py`) is allowed to read it. Everything else — the interviewer, the trait estimator, the predictor — is blind to it by construction.

`tests/test_wall.py` enforces this by parsing every Python file in the repo and failing if any file outside `src/eval/` imports the truth store or mentions the truth folder path. There is a second rule too: the people are played by one open-weights model family and the system side runs on a different family, so the same weights are never on both sides of the Wall.

## Results

- `results/REPORT.md` — the findings, every gate's numbers, and the limitations.
- `RESEARCH_SUMMARY.md` — the short version.
- `results/PROJECT_LOG.md` — the chronological map: what was run, when, what it showed, and which file holds each number. Read this first in a cold session. It links; it is never the source of truth for a number.
- `results/COSTS.md` — compute and API spend, per experiment.
- `HUMAN_PROTOCOL.md` — the pre-registered follow-up study on real people.

Six gates were graded against bars frozen in `PREREGISTRATION.md` before any of them ran. Two passed, three failed, one came out undefined. The verdicts are on the dashboard's front page and the numbers behind them are in `results/REPORT.md`; nothing was re-graded after the fact.

## Layout

```
src/personas/   trait sampling, persona card writing, response noise
src/bank/       question generation and item metadata
src/interview/  persona responder runtime, interviewer strategies
src/model/      trait model fitting, item encoder, person encoder, predictor
src/eval/       grader, baselines, metrics, the truth store accessor
src/rl/         reward computation, interviewer training
src/llm_client.py  one small swappable interface to whichever LLM we call
experiments/    one config file per experiment
results/        metrics JSON, plots, the report, the project log, the cost log
app/            minimal local dashboard
tests/
data/           generated people, transcripts, planted truth (never committed)
```

`PRD.md` is the build spec. `PREREGISTRATION.md` is the frozen contract: stages, hypotheses and numeric bars, signed off before Stage 1. `example.md` walks one persona through the whole pipeline with concrete numbers and is the fastest way to understand what this repo does.

## Running it

Python 3.11 or newer. Dependencies: `numpy`, `matplotlib` and `pytest`. Matplotlib is
what the graders use to write the PNG beside every metrics JSON, so the test suite needs
it too. The dashboard adds `streamlit` (which brings `altair` and `pandas`); nothing
outside `app/` needs it.

```bash
python3 -m venv .venv
./.venv/bin/pip install numpy matplotlib pytest streamlit
./.venv/bin/python -m pytest tests/ -q
```

The tests are the thing to run first and after every change. The Wall test is part of them, so a change that leaks the truth fails the suite.

### Reproducing a run

Every experiment is reproducible from a config file plus a seed. The configs are in
`experiments/` (`splits_v1.json` fixes the train/hold-out split, `stage5_rl.json` is
the RL interviewer run); `results/PROJECT_LOG.md` links each config to the result file
it produced.

Truth-side paths are always passed in as arguments, never hardcoded and never
defaulted — that is part of how the Wall is enforced. Mint a population:

```bash
./.venv/bin/python -m src.personas.factory --n 500 --seed 42 \
    --out-truth <truth dir> --out-public data/public
./.venv/bin/python -m src.personas.ingest \
    --completions <persona-model completions.jsonl> --out-truth <truth dir>
```

Train the adaptive (RL) interviewer. One command; run it again to resume from the last
checkpoint. The trainer is handed recorded material by path and gets coded answers
back, and its reward is the system's own declared uncertainty, so no planted value ever
reaches a weight:

```bash
./.venv/bin/python -m src.rl.train \
    --config experiments/stage5_rl.json --profile confirmatory \
    --fit results/stage2_v2_fit.npz \
    --answers <rl batch sweep>/answers.jsonl \
    --recorded <rl batch sweep>/completions.jsonl \
    --noise-dir <rl batch noise dir> \
    --public-bank data/public/bank_items.json \
    --persona-source data/runs/rl_batch/answers_noised.jsonl \
    --episodes data/runs/stage5_rl/episode_codes.npz \
    --out-dir data/runs/stage5_rl/confirmatory \
    --results results --prefix stage5_rl
```

Then grade it beside the non-RL strategies with `python -m src.eval.gate5 --policy
<the final weights>`, and check it for proxy gaming with `python -m
src.eval.rl_proxy_watch`.

Watch a run on the local dashboard — a front page with the six gate verdicts, then one
page per stage (`app/README.md` lists what each page needs):

```bash
./.venv/bin/streamlit run app/dashboard.py
```

## What is and is not committed

- `data/` is never committed. It holds the planted truth, the persona cards and every transcript. Regenerate it with the commands above.
- `results/*.npz` is gitignored: 23 MB of fitted arrays that the fit commands regenerate. The JSON report and the PNG beside each one **are** committed — those are what the dashboard and the writeup read, so every number in this repo is checkable without rerunning anything.
- `.env` is gitignored. The LLM client reads two environment variables at call time, `GLASSBOX_API_KEY` and `MODEL_NAME`. Put them in a `.env` file at the repo root; nothing reads it automatically unless code calls `load_dotenv_if_present()`.

## Models and compute

Two open-weights model families, one on each side of the Wall: a ~30B dense model
plays the personas (writes the life stories, answers in character), and a different
~30B dense family from another vendor runs the system side (pole judgments, item
embeddings, the no-interview baseline). A small hosted flash-tier API model drafted the
question bank. No Anthropic model was used anywhere in the experiments.

The heavy generation ran in batch on a European academic GPU cluster; the model fitting,
grading and RL training are plain NumPy and run on a laptop CPU in minutes. Nothing in
this repo needs a cluster to check: the graders read recorded answers off disk.
Per-experiment compute and API cost are in `results/COSTS.md`.

## License

MIT. See `LICENSE`.
