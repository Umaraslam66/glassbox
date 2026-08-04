# GLASSBOX-Mobility — what we found, in plain language

*Stage M5 summary, 2026-08-04. The full evidence is in
`results/REPORT_MOBILITY.md`; every number there traces to a committed
result file.*

## The question

A new research line puts LLM "generative agents" inside transport
simulations and validates them by asking people whether the behavior looks
believable. We asked a harder question. We built 400 synthetic travelers,
each **defined** by six hidden numbers — value of time, schedule rigidity,
car affinity, crowding sensitivity, price sensitivity, habit strength. An
LLM wrote each traveler a life story and then answered 140 travel choices
in character. If the agents are really governed by the travelers they play,
the standard tools of transport economics should read those six numbers
back out of the choices. That is the whole test: plant the truth, hide it
behind a wall, try to recover it.

## The answer

**Recovery fails — comprehensively and measurably — while everything a
believability check would look at passes.**

The same population of agents:

- **looks right**: choice prediction beats zero information, and the
  model's probability statements are nearly honest (our calibration bar
  passed — one of only a few bars that did);
- **moves right in aggregate**: cut transit fares 20% and ridership rises
  with an elasticity of −0.33, squarely inside the published empirical
  range; charge €4 to drive into town and car trips drop 10%, at the low
  edge of the published London/Stockholm range; in the 20-day congestion
  simulation, the frozen peak-spreading bar FAILED on its day-1 anchor —
  which we traced to the anchor measuring a saturated queue rather than
  behavior; measured from day 2 (exploratory, pre-declared footnote), the
  peak fell 24% as travelers discovered the alternative route;
- **and is individually unmeasurable**: not one of the six planted
  parameters could be recovered to the pre-registered standard. Value of
  time — the workhorse number of transport appraisal — came back with a
  112% median error against a 35% bar. The estimator's uncertainty
  statements covered the truth 17% of the time against the frozen 60–75%
  acceptance band.

For anyone who wants to use LLM agents as synthetic survey respondents or
policy-simulation citizens, that combination is the warning: **believability
and macro realism are not evidence that the individuals inside the
simulation mean anything.**

## Where it breaks (we can point at it)

1. **The cards only carry so much.** Writing a personality into a life
   story loses signal — we measured that ceiling at 0.61 here (the parent
   study measured 0.83 on survey traits). Three of six parameters failed
   mostly because of this ceiling, not the estimator.
2. **The agents are too decisive for the statistics.** Classical choice
   models expect people to be somewhat random; these agents almost never
   waver, so the fitted coefficients blow up and ratios like value-of-time
   become garbage. Half the travelers came back with the wrong sign on
   time itself.
3. **The persona leaks into everything.** Scenarios designed to measure
   nothing (museum vs park, soup vs salad, even a dentist-slot choice)
   still correlated with the hidden traits in 9 of 10 cases. With an LLM
   there is no such thing as a neutral filler question.
4. **Structure loses to similarity.** A naive "find the 20 most similar
   travelers and copy them" predictor beat the structural economic model
   five-fold. The signal is in the behavior; the frozen economic lens
   throws most of it away.
5. **More data makes it worse.** The parent study's strangest finding —
   uncertainty getting *less* honest as evidence grows — reappeared here
   in a new domain, almost identically. That now looks like a general
   property of estimating LLM-rendered people, not a quirk.

## Parallels with the parent study

GLASSBOX (surveys) and GLASSBOX-Mobility (travel) now agree on every
transferable point: a card-transmission ceiling exists and binds; recovery
below the ceiling is possible but estimator-hostile; coverage decays with
evidence; profile-only baselines stay at zero by design, and an LLM given
only demographics *invents* structure and predicts worse than knowing
nothing (−67% here). Two domains, one lesson.

## Honest failures we kept on the books

The first traveler population failed its obedience gate and cost a full
re-render and re-sweep (both rounds logged with their dollars). One
diversity bar failed by a single traveler pair and stands failed. The
frozen peak-spreading bar failed partly because its day-1 anchor measured
a saturated queue rather than behavior — the fall is visible from day 2 —
and we recorded that as a methods lesson, not a re-grade. The static bank
was built without its planned fare-change probe; the fare elasticity above
comes from the dynamic experiment and says so.

## What this cost

About **$2.34** of API spend, end to end — two full 56,000-answer sweeps, a
20-day 300-agent simulation, three policy experiments, and every audit and
baseline. The study is reproducible from configs and seeds by a stranger.

## The one-line takeaway

An LLM travel-agent population can be believable, calibrated and
aggregate-realistic while carrying almost no recoverable individual truth —
so any benchmark or certification of such agents must plant the truth and
try to read it back, because looking at behavior will not tell you.
