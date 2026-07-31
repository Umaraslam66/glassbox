# GLASSBOX in two pages

A plain-language summary for someone technical who has not read the code. Every number
here is traceable to a file in `results/`; the full version with sources is
`results/REPORT.md`.

---

## What we claimed

One claim, written down and frozen before any measurement:

> **The pipeline provably recovers planted ground truth when ground truth is known.**

The problem with "can an AI learn a person from a conversation?" is that with a real
person you never know the right answer, so you cannot grade it. So we made the people.

We generated 500 synthetic people. Each one starts as eight hidden numbers — how much they
trust institutions, how much risk they take, how price-sensitive they are, and so on. A
language model turns those numbers into a life story that shows the traits without ever
naming them. The person then answers 252 survey questions in character.

The system on the other side of the interview gets a short public profile (age, job, city
size, household) and a 15-question transcript. From that it estimates the eight numbers,
attaches an honest uncertainty to each, and predicts how the person would answer questions
they were never asked. Then we grade it against the numbers we planted.

Everything is reported as **lift over a zero-information baseline** — over just predicting
the population average and ignoring the person. Raw accuracy on its own is never reported,
because raw accuracy on survey questions is easy and meaningless.

## How the Wall worked

The Wall is the rule that keeps this from being self-deception. The planted truth lives in
one folder, and exactly one module is allowed to read it — the grader. Every other part of
the system (the interviewer, the trait estimator, the predictor) is blind to it by
construction, and a test that parses every Python file in the repo fails the build if
anything outside the grader imports the truth store or names the truth folder. A second
rule keeps the two sides from being the same machine: the synthetic people are played by
one model family (Gemma) and everything on the system side runs on a different one (Qwen),
so the same weights are never on both sides. A third rule is procedural: whenever a result
came out better than expected, the first action was a documented hunt for leakage rather
than a celebration. Four hunts fired. One found a real problem (the story-writing model
entangles two traits we had deliberately planted as unrelated), and it is reported as a
limitation rather than quietly fixed. The Wall itself was breached once, mid-project, by a
file that carried a pre-noise answer field into system-side view; it was caught, closed,
and a new class of test was added.

## Five numbers

| # | Number | What it means |
|---|---|---|
| 1 | **27.1% better than knowing nothing** (Brier score 0.0627 against 0.0860) | Predicting a held-out person's answers to unseen questions, from a 15-question interview. Also beats a strong nearest-neighbour baseline, which reaches 24.4% on its own. |
| 2 | **Calibration error 0.0137** (bar was 0.05) | When the system says "60% chance they pick option B", it is right about 60% of the time. This is the number that makes the uncertainty usable rather than decorative. |
| 3 | **Persona cards carry only ~85% of the planted signal** (measured 0.83 median, 0.87 best) | The story-writing step loses information. This ceiling caps every downstream measurement graded against the planted truth, and it is why several bars failed. |
| 4 | **15 closed questions + 3 open answers recover 77% of what 252 questions know** | 7% of the questions, 77% of the agreement. Short interviews are surprisingly sufficient. |
| 5 | **An LLM given only demographics is 35% *worse* than knowing nothing** (Brier 0.1159 against 0.0860) | Prompting a model with "you are a 42-year-old teacher in a mid-size city" does not produce a useful person. It invents a tilt, and the invention adds error. |

A sixth, because it is the practical one: an adaptive interviewer reaches a given accuracy
in **roughly half the questions** a random ordering needs (ratios 0.46 and 0.49 at the two
accuracy levels the system can actually reach).

## The three most interesting findings

### 1. More evidence made the error bars *worse*

The system's uncertainty is calibrated on what it can see — noise in the answers. It cannot
see the information lost when a trait vector was turned into a life story, because that
happened on the far side of the Wall.

With one question asked, sampling noise dominates and the error bars are well calibrated:
they cover the true value 75% of the time (nominal is 68%). By 15 questions, 65%. From the
full 252-question bank, **36%** — half the nominal rate. The more you learn, the more
confident the system gets, and the invisible error does not shrink with it.

*Why this is the most interesting one:* it is the only result that transfers directly to
systems that will never have ground truth, which is all of them. Any confidence model
trained only on observable data will look calibrated in small samples and go quietly
overconfident as data accumulates, whenever there is a lossy step it cannot observe. The
honest fix requires truth-anchored feedback, which is exactly what production systems lack.

### 2. Predicting a group is easy; predicting a person is not

Same model, same predictions, three levels of aggregation:

- Population averages per question: **r = 0.98**
- Two-way cross-tabs across 1,225 question pairs: **r = 0.97**
- Individual person-by-question probabilities: **r = 0.68** — and this is the bar we failed
  (the target was 0.90).

*Why this is interesting:* it is the number most likely to be misused by other people.
Synthetic-panel and digital-twin claims are almost always quoted at the aggregate level
("our simulated panel matches the real survey within 3 points"), which is the easy
measurement. Individual-level accuracy is the hard, honest currency, and the gap between
the two is enormous. We failed the individual bar and are reporting it as a fail.

We also know why it failed and how far it could go: the ceiling for this kind of model —
give it the *full* 252-question answer matrix and perfect item parameters — is 0.79, still
below the bar. Two thirds of the shortfall is the 8-number model of a person being too
coarse; the rest splits between interview length and the item encoder. There are
per-question quirks in people that no eight-number summary can express.

### 3. A reward-hacking alarm that turned out to be the instrument, not the agent

We trained a reinforcement-learning interviewer to choose questions. It was forbidden from
ever seeing the planted truth, so its reward was its own declared confidence — "ask the
question that most reduces my uncertainty". We pre-registered a watch for the obvious
failure mode: the agent learns to make itself *feel* certain without becoming *more
accurate*.

The alarm fired. The trained policy's declared confidence ran ahead of its true accuracy by
a widening margin. Then the control run: on identical interviews, the **never-trained**
info-gain heuristic showed an even *larger* gap. The trained policy ended with the same
error and more declared uncertainty than the untrained comparison. The overconfidence was a
property of the uncertainty model — finding 1 above, showing up in a third place — not of
reward hacking.

Two useful things came out of it. First, a general recipe: before concluding your agent
gamed a proxy, check what an untrained baseline does to the same proxy. Second, we trained
a forbidden version of the same agent using the real answers as its reward, and it beat the
legal version by **at most 0.024** — nothing. **You can train the thing that decides what
to ask next without ever knowing the right answer.** That is the finding with the most
direct product consequence in the whole project.

One more, kept because it is deflating in a useful way: the trained interviewer converged
to a nearly **fixed** 25-question order — 25 distinct questions across 500 different
people, against 118 for the heuristic. On this population, the value of "adaptive"
interviewing is choosing the right questions, not branching per person. A good fixed
questionnaire captures nearly all of it.

## The scoreboard, including the fails

We froze the pass/fail bars before running anything and did not move them afterwards.

| Gate | Result |
|---|---|
| 1 — population sanity | All five bars pass (development only, no claims made from it) |
| 2 — recover traits from the full answer matrix | **2 of 4 bars unmet.** Trait recovery 0.798 and 0.7996 on the two weakest dimensions against a bar of 0.80. Uncertainty coverage 0.358 against a 60-75% band. |
| 3 — generalize to unseen questions and short interviews | **3 bars unmet.** Item-encoder discrimination 0.504 against 0.60. Trait error at 15 questions: 0 of 8 dimensions under the bar. Lift 26.8% against 30%. |
| 4 — end-to-end prediction and calibration | **2 of 3 bars pass** — 27.1% lift, calibration error 0.0137. The per-person correlation bar fails at 0.684 against 0.90. |
| 5 — adaptive interviewing | **Both bars undefined.** The frozen accuracy target turned out to be unreachable at any interview length — even all 252 questions bottom out just above it. We reported that rather than lowering the target. |

Most of the failures are one thing seen five times. Bars graded against the planted truth
inherit the card-writing ceiling (number 3 above) and fail. Bars graded against the
person's own answers — which is what Gate 4 does, and what a real product would do — pass
at full strength. That pattern was predicted in writing before Gates 3, 4 and 5 ran, and it
held every time.

Two more honest notes. The Gate 5 target being unreachable was flagged as a real
possibility *before* any of that stage's compute was spent, with the exploratory fallback
analysis pre-authorised in advance — which is the only reason that analysis is worth
anything. And the whole project cost about 128 core-hours of GPU and roughly two cents of
API spend, of which about 8% of the compute went to two operational mistakes that are
listed by name in the full report.

## What this does not prove

**It says nothing about real humans.** Every person in this study is synthetic, with traits
we planted ourselves. That was the point — you cannot grade recovery without knowing the
answer — but it means no result here is evidence that the pipeline would recover anything
from a real person. A human panel is a separate study.

**The inconsistency is mechanical.** Our people give slightly different answers on a retest
because a noise layer resamples them, tuned to match a realistic 79% agreement rate. Real
people are inconsistent because they are ambivalent, tired, or primed by the previous
question. Same number, different reasons.

**Demographics are empty here, and they are not in reality.** In our population, a public
profile predicts nothing about traits — the profile-only baseline collapses to predicting
the average. On real people, demographics carry real signal, which would make the
"beat the profile" bars *harder*, not easier. Finding 5 in the table above should be read
with that caveat attached.

**One writer, one reader.** All the synthetic people are written by one model (Gemma 4
31B); everything on the system side is one model (Qwen3.6-27B). The 85% transmission
ceiling is a measurement of that one writer, not a law of nature.

**Eight numbers is not a person.** The model of a person is an 8-dimensional linear trait
vector, and the failed per-person bar is the edge of it showing. Real individual quirks
exist that this representation cannot hold.

You should trust this summary more, not less, because of the list above and because the
scoreboard reports the failures first. The bars were frozen in advance; six of them were
missed and two more turned out to be unreachable; none of them were moved.

---

## OPEN QUESTIONS FOR REVIEW

1. **"~85%" in headline number 3.** I used "about 85%" as instructed, with the measured
   values (0.83 median, 0.87 best) in the same cell. `results/REPORT.md` currently leads
   with "about 83-87%". If you want one wording in both places, tell me which and I will
   align them.

2. **"Number 5" numbering in the *What this does not prove* section.** The demographics
   caveat refers to "Finding 5 in the table above", meaning the fifth headline number (the
   LLM baseline). If the table gets renumbered, that cross-reference needs updating.

3. **The sixth number.** You asked for five headline numbers; the adaptive-interviewing
   result (0.46 / 0.49 against random) was on your list of things this document must carry,
   so it sits just below the table as an explicit sixth rather than being squeezed into
   five. Say the word and I will fold it into the table or cut it.

4. **Fail count in the closing line.** Now reads "six of them were missed and two more
   turned out to be unreachable", which is the exact arithmetic: 2 unmet at Gate 2, 3 at
   Gate 3, 1 at Gate 4, and both Gate 5 bars undefined. My first draft said "four", which
   was wrong; corrected before commit. Confirm the phrasing reads right for a lay audience,
   since "unreachable" is doing work that "undefined" does in the full report.

5. **Two-page limit.** As rendered this is a little over two printed pages at normal
   margins, mostly because of the two tables. If it must fit two pages exactly, the
   scoreboard table is the cheapest cut — it is fully covered in `results/REPORT.md`.

6. **Cross-reference to the human-panel protocol.** The *What this does not prove* section
   says "a human panel is a separate study" without naming `HUMAN_PROTOCOL.md`, since that
   file does not exist yet. Worth linking once it does.
