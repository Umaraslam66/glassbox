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
distributions behind them.
"""
