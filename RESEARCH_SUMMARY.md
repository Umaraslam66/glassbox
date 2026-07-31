# GLASSBOX in two pages

Plain-language summary for a technical reader who has not seen the code. Every number is
traceable to a file in `results/`; the sourced version is `results/REPORT.md`.

---

## What we claimed

One claim, frozen before any measurement:

> **The pipeline provably recovers planted ground truth when ground truth is known.**

The trouble with "can an AI learn a person from a conversation?" is that with a real person
you never know the right answer, so you cannot grade it. So we made the people.

We generated 500 synthetic people. Each starts as eight hidden numbers — how much they
trust institutions, how much risk they take, how price-sensitive they are, and so on. A
language model turns those numbers into a life story that shows the traits without naming
them. The person answers all 252 survey questions in character; 50 are held back to test
generalization, so the "full information" reference is the other 202.

The system on the other side gets a short public profile (age, job, city size, household)
and a 15-question transcript. From that it estimates the eight numbers, attaches an
uncertainty to each, and predicts answers to questions the person was never asked. Then we
grade it against what we planted.

Everything is reported as **lift over a zero-information baseline** — over predicting the
population average and ignoring the person. Raw accuracy alone is never reported, because
raw accuracy on survey questions is easy and meaningless.

## How the Wall worked

The Wall is what keeps this from being self-deception. Planted truth lives in one folder
and exactly one module may read it: the grader. Everything else — interviewer, trait
estimator, predictor — is blind by construction, and a test parses every Python file in the
repo and fails the build if anything outside the grader touches the truth store. A second
rule keeps the two sides from being the same machine: the people are played by one model
family (Gemma), the system side runs on another (Qwen). A third is procedural — whenever a
result beat expectations, the first action was a documented leakage hunt, not a
celebration. Four hunts fired; one found a real problem (the story-writing model entangles
two traits we planted as unrelated), reported as a limitation rather than quietly fixed.
The Wall itself was breached once, by a file carrying a pre-noise answer field into
system-side view. It was caught, closed, and a new class of test added.

## Five numbers

| # | Number | What it means |
|---|---|---|
| 1 | **27.1% better than knowing nothing** (Brier 0.0627 against 0.0860) | Predicting a held-out person's answers to unseen questions from a 15-question interview. Also beats a strong nearest-neighbour baseline, which reaches 24.4% on its own. |
| 2 | **Calibration error 0.0137** (bar was 0.05) | When the system says "60% chance they pick option B", it is right about 60% of the time. This is what makes the uncertainty usable rather than decorative. |
| 3 | **Persona cards carry ~85% of the planted signal** (measured median 0.830, range 0.770–0.871 across dimensions) | The story-writing step loses information. This ceiling caps every measurement graded against planted truth, and it is why several bars failed. |
| 4 | **15 closed questions + 3 open answers recover 77% of what 202 questions know** | 7% of the questions, 77% of the agreement. Short interviews are surprisingly sufficient. |
| 5 | **An LLM given only demographics is 34.8% *worse* than knowing nothing** (Brier 0.1159 against 0.0860) | Prompting a model with "you are a 42-year-old teacher in a mid-size city" does not produce a useful person. It invents a tilt, and the invention adds error. |

A sixth, because it is the practical one: an adaptive interviewer reaches a given accuracy
in **roughly half the questions** random ordering needs — ratios 0.46 and 0.49 at the two
accuracy levels the system can actually reach.

## The three most interesting findings

### 1. More evidence made the error bars *worse*

The system's uncertainty is calibrated on what it can see: noise in the answers. It cannot
see the information lost when a trait vector became a life story, because that happened on
the far side of the Wall.

After one question, sampling noise dominates and the error bars are well calibrated —
covering the true value 75% of the time against a nominal 68%. After 15 questions, 65%.
From all 202 questions, **36%**: half the nominal rate. The more it learns, the more
confident it gets, and the invisible error does not shrink with it.

*Why this one matters most:* it transfers directly to systems that will never have ground
truth, which is all of them. A confidence model trained only on observable data looks
calibrated in small samples and goes quietly overconfident as data accumulates, wherever
there is a lossy step it cannot observe. The honest fix needs truth-anchored feedback —
exactly what production systems lack.

### 2. Predicting a group is easy; predicting a person is not

Same model, same predictions, three levels of aggregation:

- Population averages per question: **r = 0.98**
- Two-way cross-tabs across 1,225 question pairs: **r = 0.97**
- Individual person-by-question probabilities: **r = 0.68** — the bar we failed (target 0.90).

*Why it matters:* this is the number most likely to be misused. Synthetic-panel and
digital-twin claims are almost always quoted at the aggregate level ("matches the real
survey within 3 points"), which is the easy measurement. Individual accuracy is the hard,
honest currency, and the gap is enormous.

We know why it failed and how far it could go. The ceiling for this kind of model — the
full 202-question answer matrix *and* perfect item parameters — is 0.786, still below the
bar. Two thirds of the shortfall is an 8-number model of a person being too coarse; the
rest splits between interview length and the item encoder. Per-question quirks exist in
people that no eight-number summary can hold.

### 3. A reward-hacking alarm that was the instrument, not the agent

We trained a reinforcement-learning interviewer to choose questions. It was forbidden from
seeing planted truth, so its reward was its own declared confidence: ask whatever most
reduces my uncertainty. We pre-registered a watch for the obvious failure — the agent learns
to *feel* certain without getting more *accurate*.

The alarm fired. Then the control: on identical interviews the **never-trained** info-gain
heuristic showed an even larger gap. The trained policy ended with the same error and more
declared uncertainty than the untrained comparison. The overconfidence was a property of
the uncertainty model — finding 1 again, in a third place — not reward hacking.

Two things came out of it. A general recipe: before concluding your agent gamed a proxy,
check what an untrained baseline does to the same proxy. And we trained a forbidden version
using real answers as reward — it beat the legal version by **at most 0.0242**, i.e.
nothing. **You can train the thing that decides what to ask next without ever knowing the
right answer.** That is the finding with the most direct product consequence here.

One more, deflating in a useful way: the trained interviewer converged to a nearly **fixed**
25-question order — 25 distinct questions across 500 different people, against 118 for the
heuristic. Here, the value of "adaptive" interviewing is choosing the right questions, not
branching per person. A good fixed questionnaire captures nearly all of it.

## The scoreboard, including the fails

Bars were frozen before anything ran and were not moved afterwards.

| Gate | Result |
|---|---|
| 1 — population sanity | All five bars pass (development only; no claims made from it) |
| 2 — recover traits from the full answer matrix | **2 of 4 bars unmet.** Trait recovery 0.798 and 0.7996 on the two weakest dimensions against a bar of 0.80. Uncertainty coverage 0.3575 against a 60–75% band. |
| 3 — generalize to unseen questions and short interviews | **3 bars unmet.** Item-encoder discrimination 0.504 against 0.60. Trait error at 15 questions: 0 of 8 dimensions under the bar. Lift 26.8% against 30%. |
| 4 — end-to-end prediction and calibration | **2 of 3 bars pass** — 27.1% lift, calibration error 0.0137. The per-person correlation bar fails at 0.684 against 0.90. |
| 5 — adaptive interviewing | **Both bars undefined.** The frozen accuracy target is unreachable at any interview length — all 202 questions bottom out at 0.5253, just above it. We reported that rather than lowering the target. |

Most failures are one thing seen five times. Bars graded against planted truth inherit the
card-writing ceiling (number 3 above) and fail. Bars graded against the person's own
answers — what Gate 4 does, and what a real product would do — pass at full strength. That
pattern was predicted in writing before Gates 3, 4 and 5 ran, and held every time.

The Gate 5 target being unreachable was flagged as a real possibility *before* that stage's
compute was spent, with the fallback analysis pre-authorised — the only reason the fallback
is worth anything. The project cost about 128 core-hours of GPU and roughly two cents of
API spend, of which ~8% of the compute went to two operational mistakes named in the full
report.

## What this does not prove

**It says nothing about real humans.** Every person here is synthetic, with traits we
planted. That was the point — you cannot grade recovery without knowing the answer — but no
result here is evidence the pipeline would recover anything from a real person. The
pre-registered human study is specified separately in `HUMAN_PROTOCOL.md` at the repository
root; it has not been run.

**The inconsistency is mechanical.** Our people answer differently on a retest because a
noise layer resamples them, tuned to a realistic 79% agreement rate. Real people are
inconsistent because they are ambivalent, tired, or primed by the previous question. Same
number, different reasons.

**Demographics are empty here, and are not in reality.** In our population a public profile
predicts nothing about traits. Real demographics carry real signal, which makes "beat the
profile" bars *harder*, not easier. Read number 5 in the table above with that attached.

**One writer, one reader.** All synthetic people are written by one model (Gemma 4 31B);
the system side is one model (Qwen3.6-27B). The ~85% transmission ceiling measures that one
writer, not a law of nature.

**Eight numbers is not a person.** The model of a person is an 8-dimensional linear trait
vector, and the failed per-person bar is its edge showing.

Trust this summary more, not less, because of the list above and because the scoreboard
reports the failures first. The bars were frozen in advance; six were missed and two more
turned out to be unreachable; none were moved.
