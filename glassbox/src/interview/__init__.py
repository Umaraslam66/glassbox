"""Interview engine: persona responder runtime and interviewer strategies.

CONSISTENCY RULE (holds from Stage 3 onward, no exceptions). Every closed-form
answer a persona gives during an interview must be produced by the frozen noise
layer, ``src.personas.noise_layer``, applied to that responder's recorded
answer-token distribution for that item. The parameters are the ones frozen in
PREREGISTRATION.md section 5, addendum: a = 1.2, b = 4.0, T_noise = 16,
seed-base 548, seeded per record. No interview path may skip the layer, hand
back the responder's un-noised answer, or re-run the layer with different
numbers. The reason is that Stage 4 grades a prediction against the persona's
true answer distribution, and that distribution is defined as repeated seeded
applications of this same layer at these same parameters: interviews and
grading have to come from one world, or the interview and the grader are
describing two different people.

Wall note: applying the layer is persona-side work. Code in this package moves
answers around -- the strings and numbers the persona said -- and never the
distributions behind them. ``src.interview.interview_answers`` hands the
recorded material to ``src.personas.noise_layer`` by path and takes drawn
answers back; the distributions themselves stay on the persona side.

The four numbers below are the whole of the consistency rule in code. They are
frozen (PREREGISTRATION.md section 5, addendum) and are deliberately not
command-line options anywhere in this package: an interview path that could be
run at other settings would be a way to produce answers from a different world
than the one Stage 4 grades against.
"""

#: Flat slip rate of the frozen noise layer.
FROZEN_NOISE_A = 1.2

#: Ambivalence boost of the frozen noise layer.
FROZEN_NOISE_B = 4.0

#: Re-draw flattening temperature of the frozen noise layer.
FROZEN_NOISE_T = 16.0

#: Base for every per-record seed in this project.
FROZEN_SEED_BASE = 548

#: Responder sampling temperature, frozen at Gate 1.
FROZEN_RESPONDER_TEMPERATURE = 0.7
