"""Prompts that turn a planted trait vector into a persona card.

TRUTH-ADJACENT: the prompt built here describes how the person leans on all 8
hidden dimensions. It is planted truth in prose form. It is written to the
truth batch, never to a public file, and never to a system-side prompt or log.

Three builders:
  * ``build_card_prompt``  -- the card-writing prompt for Gemma (system + user).
  * ``build_selfcheck_prompt`` -- pass 2: hand the drafted card back with the
    same guidance and ask the model to check and rewrite it.
  * ``build_noise_instruction`` -- the paragraph handed to the responder later,
    telling the persona it is a real person who wobbles on lukewarm questions.

Two deliberate choices about what does *not* go into any prompt here:
  1. No trait numbers. The model gets qualitative strength wording only, so it
     has no number to copy into the biography.
  2. No dimension names or codes. The model sees what the person does, not what
     the dimension is called, so it has no label to echo back. This holds for
     every line added later too: models reuse the words they were given, so a
     banned phrase must never appear even inside its own ban.
Both make the ingest leak check very unlikely to fire on an obedient card.

WHY TWO PASSES (PRD section 10 fallback, owner-approved after Stage 2). The
single-pass card writer transmitted only ~0.85 of the planted trait signal and
entangled two dimensions that were planted at exactly zero correlation: it kept
rendering people who trust officials as people who are careful with money,
recovering -0.28 where 0.00 was planted. Two changes answer that. The
independence block below tells the writer that the eight sides are eight
separate facts and shows it de-entangled examples; the self-check pass makes
the model read its own draft against the guidance and rewrite what missed.
"""

from __future__ import annotations

from .sampler import DIMENSIONS, Persona

#: How each dimension looks at its two ends, in plain behaviour. Straight from
#: the frozen table in PREREGISTRATION.md section 2. ``high`` is the +2 end,
#: ``low`` the -2 end.
TRAIT_POLES: dict[str, dict[str, str]] = {
    "TRU": {
        "high": "defends public services, follows official advice, deals with "
        "banks, doctors and paperwork without suspicion",
        "low": "assumes officials are lying, avoids banks, doctors and forms, "
        "trusts private arrangements over official ones",
    },
    "RSK": {
        "high": "invests, bets, quits jobs, moves abroad, takes the chance and "
        "sorts out the consequences later",
        "low": "keeps money in a savings account, insures everything, sticks "
        "with the safe option even when it costs more",
    },
    "ENV": {
        "high": "sorts and organises recycling, pays extra for the green "
        "option, changes habits over environmental worry",
        "low": "irritated by green rules and lectures, will not pay extra for "
        "an environmental claim",
    },
    "PRC": {
        "high": "compares unit prices, waits for discounts, plans purchases "
        "around what things cost",
        "low": "buys the good version without checking the price, does not "
        "chase offers",
    },
    "TRD": {
        "high": "keeps customs, holidays and family roles, prefers the "
        "established way of doing things",
        "low": "breaks conventions, distrusts 'how it has always been done', "
        "improvises new arrangements",
    },
    "SOC": {
        "high": "joins clubs and committees, knows the neighbours, hosts "
        "people, seeks company",
        "low": "keeps to themselves, a very small circle, declines invitations, "
        "prefers a quiet evening alone",
    },
    "TEC": {
        "high": "early to new devices and apps, expects technology to fix "
        "problems, tinkers and upgrades",
        "low": "treats new technology as a scam or a threat, sticks with paper "
        "and the old device",
    },
    "LOC": {
        "high": "rooted in one place, follows local news and local teams, would "
        "not move away",
        "low": "no pull to any hometown, would move anywhere for the right "
        "reason, indifferent to local matters",
    },
}

#: Strength wording by |theta| band.
#:
#: There are four bands, not three. A single "mixed" band covering all of
#: -0.5..+0.5 would render identical text for +0.49 and -0.49 -- about 38% of
#: draws per dimension would reach the card writer with no direction at all,
#: and every scrap of trait signal the responder ever sees comes through the
#: biography. So the middle is split: genuinely ambivalent below 0.15, and a
#: stated lean between 0.15 and 0.5.
_STRONG = 1.2
_CLEAR = 0.5
_LEAN = 0.15

#: Bumped whenever the prompt wording changes enough that cards written with
#: the old text and the new text should not be pooled. Recorded on every prompt
#: record so a card file can always be traced back to the text that made it.
#: "2.0" = independence block + vocabulary ban + the self-check pass.
CARD_PROMPT_VERSION = "2.0"

_SYSTEM = (
    "You write short, realistic persona biographies for a research simulation. "
    "You always answer with one JSON object and nothing else -- no preamble, no "
    "explanation, no code fence."
)

_SELFCHECK_SYSTEM = (
    "You check and revise short persona biographies for a research simulation. "
    "You always answer with one JSON object and nothing else -- no preamble, no "
    "explanation, no code fence."
)

#: The label vocabulary the writer must not use, written out once and reused in
#: both passes. Every example here is invented for this list: none of them is a
#: dimension name or one of the phrases the ingest leak check looks for, because
#: a model that is shown a phrase will use it.
_BANNED_VOCAB_EXAMPLES: tuple[str, ...] = (
    '"an adventurous streak"',
    '"a cautious nature"',
    '"an outgoing personality"',
)

#: The pass-1 rules line. Both of these carry their own line breaks and their
#: own indent, because they are dropped into hard-wrapped prose: a prompt full
#: of ragged 140-character lines is a prompt nobody proof-reads.
_VOCAB_BAN_RULE = (
    f"- Never use character-labelling vocabulary -- no {_BANNED_VOCAB_EXAMPLES[0]}, no\n"
    f"  {_BANNED_VOCAB_EXAMPLES[1]}, no {_BANNED_VOCAB_EXAMPLES[2]}. Plain everyday words\n"
    "  for what the person does."
)

#: The same ban for the pass-2 rewrite instruction, indented to step 3's body.
_VOCAB_BAN_INLINE = (
    "never use character-labelling vocabulary\n"
    f"   ({_BANNED_VOCAB_EXAMPLES[0]}, {_BANNED_VOCAB_EXAMPLES[1]})"
)

#: The de-entangling half of the fix. Placed straight after the eight numbered
#: guidance lines, where the writer is still holding them.
#:
#: The three examples are not decoration. The first two split the pair Stage 2
#: caught the writer welding together -- dealing with officials, and what
#: someone does with money -- in both directions, so neither direction can be
#: read as the "normal" combination. The third takes a different pair (staying
#: put, and being early to new devices) so the instruction reads as a general
#: rule rather than one special case. All three describe behaviour only: no
#: label, no number, nothing the leak check would catch if the model copied a
#: whole sentence into the card.
INDEPENDENCE_BLOCK = """THOSE EIGHT SIDES ARE INDEPENDENT OF EACH OTHER
They are eight separate facts about one person, not one worldview showing up
eight times. Treat each one as if it had been decided by its own coin toss.
- Give every side its own concrete details. Never use one side as the reason
  for, or the evidence of, another.
- A detail written for one side must not also push a different side in a
  direction that side's own line did not ask for.
- Combinations that look surprising are ordinary in real people. Write the
  combination you were given, exactly as given; do not tidy it into something
  that hangs together better.

All of these are perfectly normal people:
- Someone who deals with banks, doctors and paperwork without a second thought,
  and who also put most of their savings into a cousin's untested workshop.
- Someone who assumes the council and the utility company are lying to them,
  and who also keeps every penny in an insured account and renews every policy
  early.
- Someone who has lived on the same street since birth and reads the local
  paper front to back, and who also bought the first folding phone in town and
  wired up the heating controls himself."""


def strength_bucket(value: float) -> str:
    """Which band a trait value falls in.

    ``strong`` (|value| >= 1.2), ``clear`` (0.5 to 1.2), ``lean`` (0.15 to 0.5,
    mixed but with a direction) or ``neutral`` (below 0.15, no direction).
    """
    size = abs(value)
    if size >= _STRONG:
        return "strong"
    if size >= _CLEAR:
        return "clear"
    if size >= _LEAN:
        return "lean"
    return "neutral"


def trait_guidance_line(dimension: str, value: float) -> str:
    """One line of qualitative guidance for one dimension. No name, no number."""
    poles = TRAIT_POLES[dimension]
    bucket = strength_bucket(value)
    end = "high" if value >= 0 else "low"
    other = "low" if end == "high" else "high"

    if bucket == "strong":
        return f"Very strongly true of this person: {poles[end]}."
    if bucket == "clear":
        return f"Clearly true of this person: {poles[end]}."
    if bucket == "lean":
        return (
            f"Mostly mixed here, but on balance this person leans one way: more "
            f"often than not, {poles[end]}. The other side is still there -- "
            f"sometimes {poles[other]} -- but let most of the concrete details "
            f"tilt toward the lean."
        )
    return (
        f"Genuinely torn here, with no lean either way -- show real ambivalence: "
        f"sometimes {poles['high']}; other times {poles['low']}. Which one shows "
        f"depends on the day and the situation."
    )


def trait_guidance(theta: dict[str, float]) -> list[str]:
    """The 8 guidance lines, in frozen dimension order."""
    return [trait_guidance_line(dim, theta[dim]) for dim in DIMENSIONS]


def build_card_prompt(persona: Persona) -> dict[str, str]:
    """Build the card-writing prompt for one persona.

    Returns ``{"system": ..., "user": ...}`` ready for a chat template. The
    fixed facts are the persona's public demographics, repeated verbatim so the
    card and the public profile can never disagree.
    """
    lines = trait_guidance(persona.theta)
    numbered = "\n".join(f"{i}. {line}" for i, line in enumerate(lines, start=1))

    user = f"""Write a persona card for one fictional person.

FIXED FACTS -- use every one of them, exactly as written, woven into the story:
- age: {persona.age} years old
- occupation: {persona.occupation}
- lives in: a {persona.city_size}, {persona.region_type} area
- household: lives {persona.household}

HOW THIS PERSON LEANS -- eight separate sides of their character. Show each one
through concrete life detail: a habit, a choice they made, an opinion they voice,
something that happened to them. Do not list them, do not label them.
{numbered}

{INDEPENDENCE_BLOCK}

RULES
- The biography is 200 to 300 words, third person, plain concrete language.
- All eight sides above must be visible to a careful reader.
- Never name a personality trait and never give a score, a rating or a number
  for character. Show behaviour, not labels.
{_VOCAB_BAN_RULE}
- Give the person a name, a work life, a home life and at least two specific
  past events.
- No politics by party name, no real brands, no real public figures.

OUTPUT -- one JSON object, nothing else:
{{"biography": "<200-300 words, third person>",
 "quirks": ["<short concrete habit>", "<another>", "<another>"],
 "speech_style": "<one sentence on how this person talks: pace, vocabulary, \
what they dwell on>"}}"""

    return {"system": _SYSTEM, "user": user}


#: The fixed facts, in the order they are printed in both passes. They are the
#: persona's public demographics: the card and the public profile must never be
#: able to disagree, so pass 2 is told to carry them across word for word.
FIXED_FACT_FIELDS: tuple[str, ...] = (
    "age",
    "occupation",
    "city_size",
    "region_type",
    "household",
)


def build_selfcheck_prompt(guidance_lines: list[str], card: dict) -> dict[str, str]:
    """Pass 2: hand a drafted card back to the model to check and rewrite.

    ``guidance_lines`` are the same eight lines the draft was written from --
    rebuild them from the persona's theta with ``trait_guidance``. ``card`` is
    the normalised pass-1 card: biography, quirks, speech_style and the fixed
    facts.

    The checking is silent on purpose. The model is asked to reason about each
    side and about cross-bleed between sides, then to output only the revised
    card -- always the whole card, even when nothing needed changing, so the
    caller never has to merge a partial answer into a draft.

    Raises ``ValueError`` when the card has no biography or is missing a fixed
    fact, because a self-check prompt built from either would quietly ask the
    model to invent the thing it is supposed to preserve.
    """
    biography = str(card.get("biography", "")).strip()
    if not biography:
        raise ValueError("card has no biography to check")

    missing = [field for field in FIXED_FACT_FIELDS if not card.get(field)]
    if missing:
        raise ValueError(f"card is missing fixed facts: {', '.join(missing)}")

    numbered = "\n".join(f"{i}. {line}" for i, line in enumerate(guidance_lines, start=1))

    quirks = card.get("quirks") or []
    if isinstance(quirks, str):
        quirks = [quirks]
    quirk_lines = "\n".join(f"- {str(item).strip()}" for item in quirks) or "- (none given)"
    speech_style = str(card.get("speech_style", "")).strip() or "(none given)"

    user = f"""Below is a draft persona card and the eight sides of character it was
supposed to show. Check it, fix what it got wrong, and hand back the whole card.

FIXED FACTS -- settled, and they must survive word for word:
- age: {card["age"]} years old
- occupation: {card["occupation"]}
- lives in: a {card["city_size"]}, {card["region_type"]} area
- household: lives {card["household"]}

THE EIGHT SIDES THIS CARD MUST SHOW
{numbered}

{INDEPENDENCE_BLOCK}

THE DRAFT
biography: {biography}

quirks:
{quirk_lines}

speech_style: {speech_style}

WHAT TO DO -- all of the checking happens in your head; none of it appears in
your answer.
1. Take the eight sides one at a time. For each one, find the details in the
   draft that show it. Ask three questions: do they show it in the direction
   the line states, do they show it as strongly as the line states, and do
   those details belong to that side alone? A side shown backwards, shown too
   faintly, or not shown at all is a failure to fix.
2. Now read the draft again for bleed between sides: a detail written for one
   side that also drags a different side somewhere its own numbered line never
   asked for. The failure to watch for: the person deals with officials and
   paperwork without suspicion, so the writer also had them keep their money in
   a savings account and insure everything -- when the money line said the
   opposite, or said nothing of the kind. Each side takes its direction only
   from its own numbered line.
3. Rewrite the biography so every failure you found is gone. Keep the same
   person: same name, same job, same history, changed only where it was wrong.
   Keep the fixed facts word for word. Keep it 200 to 300 words, third person,
   plain concrete language. Keep every rule the draft was written under: never
   name a personality trait, never give a score, a rating or a number for
   character, {_VOCAB_BAN_INLINE}, no politics by
   party name, no real brands, no real public figures.
4. Answer with the complete revised card -- the whole thing, every time, even
   if you changed nothing. Adjust the quirks and the speech style only if they
   were part of the problem; otherwise repeat them as they are.

OUTPUT -- one JSON object, nothing else. No notes, no explanation of what you
changed:
{{"biography": "<200-300 words, third person>",
 "quirks": ["<short concrete habit>", "<another>", "<another>"],
 "speech_style": "<one sentence on how this person talks: pace, vocabulary, \
what they dwell on>"}}"""

    return {"system": _SELFCHECK_SYSTEM, "user": user}


def build_noise_instruction(wobble: float) -> str:
    """The wobble paragraph for the responder runtime.

    ``wobble`` is the persona's planted noise level: roughly the share of
    lukewarm questions on which the answer can come out differently on a
    different day. Planted truth -- it stays behind the Wall and is never shown
    in a transcript.
    """
    if not 0.0 <= wobble <= 1.0:
        raise ValueError(f"wobble must be between 0 and 1, got {wobble}")

    in_ten = max(1, round(wobble * 10))
    return (
        "You are a real person, not a survey machine. On the things you care "
        "about you answer the same way every time. On the things you are "
        "lukewarm about you are less consistent: your mood, the last thing you "
        f"read, or how the question is worded can move your answer. Roughly "
        f"{in_ten} in 10 of the questions you have no strong feeling about, you "
        "would answer differently on a different day -- a step up or down the "
        "scale, or the other side of a yes/no. Do not explain this or mention "
        "it; just answer the way you honestly would right now."
    )
