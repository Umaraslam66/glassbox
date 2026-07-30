"""The prompt that makes one persona answer one closed item.

Two pieces go into every prompt:

  * the **system** message -- who the persona is (biography, habits, way of
    talking), how steady they are (the wobble paragraph), and exactly what an
    answer must look like for this item type;
  * the **user** message -- a session line and the item text, nothing else. No
    lead-in wording, so the item is the only thing steering the answer.

WALL RULE: the persona card and the wobble are planted truth, and this module
never goes looking for them. They arrive as arguments -- a card dict and a
float -- exactly the way ``src/personas/factory.py`` takes its output tree from
the command line. No path to the planted truth is written down here, and
``tests/test_wall.py`` fails the build if one ever is.

The session line is not decoration. A retest round has to be a genuinely fresh
context, not a cheap continuation of a cached prefix, so the first line of the
user message differs between rounds. That places the difference before the item
text, which is where a prefix cache would otherwise start reusing tokens.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

from ..personas.card_prompts import build_noise_instruction

#: The two rounds a persona is ever asked in: the main pass over the bank, and
#: the fixed re-ask that feeds the test-retest number (PREREGISTRATION.md 5).
ROUNDS: tuple[str, ...] = ("main", "retest")

#: How an answer must be written, per item type. Frozen wording: the parser in
#: ``parse_answers.py`` is tuned against exactly these two instructions.
FORMAT_INSTRUCTIONS: dict[str, str] = {
    "likert5": (
        "Answer with a single digit 1-5 where 1=strongly disagree, "
        "5=strongly agree. Output ONLY the digit."
    ),
    "binary": "Answer only yes or no.",
}

_ROLE_LEAD = (
    "You are the person described below. Answer as this person would -- from "
    "their life, their habits and their opinions, not from general knowledge."
)

_TERSE = "Give the answer and nothing else: no explanation, no extra words."


def format_instruction(item_type: str) -> str:
    """The answer-format line for one item type."""
    try:
        return FORMAT_INSTRUCTIONS[item_type]
    except KeyError:
        raise ValueError(
            f"unknown item type {item_type!r}; known: "
            f"{', '.join(sorted(FORMAT_INSTRUCTIONS))}"
        ) from None


def session_id(pid: str, round_name: str, session_seed: int) -> str:
    """A short, reproducible session tag for one persona in one round.

    Same persona, same round, same seed -> same tag, so re-emitting a job file
    does not silently change the prompts. Different round -> different tag,
    which is what makes the retest a fresh sitting.
    """
    if round_name not in ROUNDS:
        raise ValueError(f"unknown round {round_name!r}; known: {', '.join(ROUNDS)}")
    digest = hashlib.sha256(f"{round_name}|{session_seed}|{pid}".encode("utf-8")).hexdigest()
    return f"{round_name}-{digest[:10]}"


def _quirk_lines(quirks: Any) -> list[str]:
    if isinstance(quirks, str):
        quirks = [quirks]
    if not isinstance(quirks, Sequence):
        return []
    return [f"- {str(item).strip()}" for item in quirks if str(item).strip()]


def build_system_prompt(card: Mapping[str, Any], wobble: float, item_type: str) -> str:
    """The role-play system message for one persona answering one item type."""
    biography = str(card.get("biography", "")).strip()
    if not biography:
        raise ValueError(
            f"persona card {card.get('pid', '<no pid>')!r} has no biography; "
            "there is nothing to role-play"
        )

    parts: list[str] = [_ROLE_LEAD, "", "WHO YOU ARE", biography]

    quirks = _quirk_lines(card.get("quirks"))
    if quirks:
        parts += ["", "THINGS YOU DO", *quirks]

    speech_style = str(card.get("speech_style", "")).strip()
    if speech_style:
        parts += ["", "HOW YOU TALK", speech_style]

    parts += ["", "HOW STEADY YOU ARE", build_noise_instruction(wobble)]
    parts += ["", "HOW TO ANSWER", format_instruction(item_type), _TERSE]
    return "\n".join(parts)


def build_user_prompt(item: Mapping[str, Any], session: str) -> str:
    """The user message: the session line, then the item text."""
    text = str(item.get("text", "")).strip()
    if not text:
        raise ValueError(f"item {item.get('item_id', '<no id>')!r} has no text")
    return f"Session: {session}\n\n{text}"


def build_answer_prompt(
    card: Mapping[str, Any],
    wobble: float,
    item: Mapping[str, Any],
    session: str,
) -> dict[str, str]:
    """Build the full prompt for one persona answering one closed item.

    ``card`` is the stored persona card (biography, quirks, speech_style),
    ``wobble`` the persona's planted noise level, ``item`` the **public** item
    record (id, text, type), ``session`` the tag from :func:`session_id`.

    Returns ``{"system": ..., "user": ...}``, ready for a chat template.
    """
    item_type = str(item.get("type", ""))
    return {
        "system": build_system_prompt(card, wobble, item_type),
        "user": build_user_prompt(item, session),
    }
