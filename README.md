# GLASSBOX

GLASSBOX is a recovery study. We build synthetic people whose true traits we planted ourselves, hide the truth, and then test whether an interview pipeline can recover it from interview text alone.

The point: if you want to know whether a system can "learn a person from a conversation", you need to already know who that person is. With real humans you never do. So we make the people. Each synthetic person starts as a hidden vector of trait values (how much they trust institutions, how much risk they take, how price-sensitive they are, and so on). A language model turns that vector into a life story that shows the traits without ever naming them. The person then answers interview questions in character.

The system on the other side of the interview never sees the vector or the life story. It sees a short public profile (age, job, city size, household) and the interview transcript. From that it estimates the traits, with an honest uncertainty on each one, and predicts how the person would answer questions they were never asked. The grader compares those estimates and predictions against the planted truth.

Every number is reported as a lift over a zero-information baseline (predict the population average and ignore the person). Raw accuracy on its own is never reported.

## The Wall

The Wall is the rule that keeps the study honest: the planted truth lives in one folder, and exactly one module (`src/eval/truth_store.py`) is allowed to read it. Everything else — the interviewer, the trait estimator, the predictor — is blind to it by construction.

`tests/test_wall.py` enforces this by parsing every Python file in the repo and failing if any file outside `src/eval/` imports the truth store or mentions the truth folder path. There is a second rule too: the people are played by one model family (Gemma) and the system side runs on a different family, so the same weights are never on both sides of the Wall.

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
results/        metrics, plots, the project log, the cost log
app/            minimal local dashboard
tests/
data/           generated people and transcripts (never committed)
```

`PRD.md` is the build spec. `example.md` walks one persona through the whole pipeline with concrete numbers and is the fastest way to understand what this repo does.

## Running it

Python 3.11 or newer. No dependencies yet beyond the standard library and `pytest`.

```bash
python3 -m venv .venv
./.venv/bin/pip install pytest
./.venv/bin/python -m pytest tests/ -q
```

The tests are the thing to run first and after every change. The Wall test is part of them, so a change that leaks the truth fails the suite.

The LLM client reads two environment variables at call time, `GLASSBOX_API_KEY` and `MODEL_NAME`. Put them in a `.env` file at the repo root; it is gitignored and never read automatically unless code calls `load_dotenv_if_present()`.

## Status

Stage 0: scaffold. Nothing scientific has been measured yet. Progress is logged in `results/PROJECT_LOG.md`, costs in `results/COSTS.md`.

## License

MIT. See `LICENSE`.
