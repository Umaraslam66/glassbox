"""Prompts that turn a planted trait vector into a persona card.

TRUTH-ADJACENT: the prompt built here describes how the person leans on all 8
hidden dimensions. It is planted truth in prose form. It is written to the
truth batch, never to a public file, and never to a system-side prompt or log.

Two builders:
  * ``build_card_prompt``  -- the card-writing prompt for Gemma (system + user).
  * ``build_noise_instruction`` -- the paragraph handed to the responder later,
    telling the persona it is a real person who wobbles on lukewarm questions.

Two deliberate choices about what does *not* go into the card prompt:
  1. No trait numbers. The model gets qualitative strength wording only, so it
     has no number to copy into the biography.
  2. No dimension names. The model sees what the person does, not what the
     dimension is called, so it cannot echo "risk appetite" into the text.
Both make the ingest leak check very unlikely to fire on an obedient card.
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

_SYSTEM = (
    "You write short, realistic persona biographies for a research simulation. "
    "You always answer with one JSON object and nothing else -- no preamble, no "
    "explanation, no code fence."
)


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

RULES
- The biography is 200 to 300 words, third person, plain concrete language.
- All eight sides above must be visible to a careful reader.
- Never name a personality trait and never give a score, a rating or a number
  for character. Show behaviour, not labels.
- Give the person a name, a work life, a home life and at least two specific
  past events.
- No politics by party name, no real brands, no real public figures.

OUTPUT -- one JSON object, nothing else:
{{"biography": "<200-300 words, third person>",
 "quirks": ["<short concrete habit>", "<another>", "<another>"],
 "speech_style": "<one sentence on how this person talks: pace, vocabulary, \
what they dwell on>"}}"""

    return {"system": _SYSTEM, "user": user}


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
