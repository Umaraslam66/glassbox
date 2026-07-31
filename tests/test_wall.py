"""The Wall test (PRD section 6.1, PREREGISTRATION.md section 3).

Static check over every Python file in the repository. A file outside
``src/eval/`` fails the Wall if it either

  (a) imports the truth-store accessor in any form -- plain ``import``,
      ``from ... import``, package-style ``src.eval.truth_store``, a relative
      import, or an ``importlib`` call with the module name as a string; or
  (b) mentions the planted-truth directory path in its source text.

A file outside ``src/eval/`` **and** ``src/personas/`` also fails if it

  (c) mentions logprobs at all. The recorded answer-token distribution is what
      the persona's true answer distribution is made of -- it is exactly what
      Stage 4 predicts and grades. The system side (interviewer, encoders,
      predictor, k-NN baseline) may see sampled answers and nothing else, so
      ``src/interview/``, ``src/model/``, ``src/rl/``, ``src/bank/`` and
      ``app/`` must not contain the word at all.

And one check looks at the filesystem instead of the source:

  (d) no run directory under ``data/runs/`` holds raw material -- responder
      completions, responder prompts (they carry the persona card) or the raw
      un-noised answers. Those live in a ``runs/<run_id>/`` folder inside the
      planted-truth directory, whose name this file deliberately never spells
      out. Skipped when there is no ``data/`` at all, as in a fresh clone.

Nothing is executed: the files are parsed with ``ast`` and read as text, so this
is safe to run on any commit and catches leaks that a runtime test would miss.

Exemptions, and only these:

  * truth store and truth path (a, b): ``src/eval/`` itself, plus this file (it
    must name the forbidden things in order to look for them) and
    ``tests/test_truth_store.py`` (the accessor's own unit test, which is
    grader-side code by definition).
  * logprobs (c): ``src/eval/`` and ``src/personas/`` -- the grader and the
    persona side both work with the distributions on purpose -- plus this file
    and ``tests/test_noise_layer.py``, the noise layer's own unit test.

Do not add to either list. If new code needs the truth store it belongs in
``src/eval/``; if it needs the answer distributions it belongs in ``src/eval/``
or ``src/personas/``.

Scope note: (c) walks Python files, like every other check here -- it is a rule
about code, not about prose. Notes in ``results/`` may discuss logprobs.
"""

from __future__ import annotations

import ast
from fnmatch import fnmatch
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Directories never walked.
SKIP_DIRS = {".venv", "venv", ".git", "__pycache__", ".pytest_cache", "node_modules", "build", "dist"}

#: Paths (relative to the repo root) allowed to reference the truth store.
EXEMPT_DIRS = (("src", "eval"),)
EXEMPT_FILES = (("tests", "test_wall.py"), ("tests", "test_truth_store.py"))

MODULE_NAME = "truth_store"

#: Paths allowed to mention logprobs: the grader and the persona side.
LOGPROB_EXEMPT_DIRS = (("src", "eval"), ("src", "personas"))
LOGPROB_EXEMPT_FILES = (("tests", "test_wall.py"), ("tests", "test_noise_layer.py"))

#: Matched case-insensitively, so "logprob", "logprobs" and "LogProbs" all hit.
LOGPROB_NEEDLE = "logprob"

#: Where run artifacts live, and what may never be found there.
RUNS_DIR = ("data", "runs")

#: Filename patterns that mean "raw material", matched case-insensitively.
#: Responder prompts carry the persona card; completions carry the answer-token
#: distribution; ``answers.jsonl`` and ``answer_rejects.jsonl`` are the answers
#: before the noise layer touched them. The looser patterns catch the same
#: files under a run-prefixed name, e.g. ``pilot2_all_prompts.jsonl``.
RAW_MATERIAL_PATTERNS = (
    "answers.jsonl",
    "answer_rejects.jsonl",
    "completions*.json*",
    "prompts*.json*",
    "*_prompts.json*",
    "*_prompt_*.json*",
    "*_completions.json*",
    "*_completion_*.json*",
    # per-persona wobble is a planted noise parameter (PREREGISTRATION section 3)
    "noise_stats.json",
)

# Assembled from pieces so that this test file does not itself contain the
# literal it forbids -- keeps the check honest if the exemptions ever change.
TRUTH_PATH_NEEDLES = ("data" + "/" + "truth", "data" + "\\" + "truth")


def python_files() -> list[Path]:
    """Every .py file in the repo, minus the skipped directories."""
    found: list[Path] = []
    for path in REPO_ROOT.rglob("*.py"):
        if SKIP_DIRS & set(path.relative_to(REPO_ROOT).parts):
            continue
        found.append(path)
    return sorted(found)


def _exempt(path: Path, dirs: tuple[tuple[str, ...], ...], files: tuple[tuple[str, ...], ...]) -> bool:
    parts = path.relative_to(REPO_ROOT).parts
    if parts in files:
        return True
    return any(parts[: len(prefix)] == prefix for prefix in dirs)


def is_exempt(path: Path) -> bool:
    """Allowed to reach for the truth store or name its directory."""
    return _exempt(path, EXEMPT_DIRS, EXEMPT_FILES)


def is_logprob_exempt(path: Path) -> bool:
    """Allowed to mention logprobs."""
    return _exempt(path, LOGPROB_EXEMPT_DIRS, LOGPROB_EXEMPT_FILES)


def _names_truth_store(dotted: str | None) -> bool:
    if not dotted:
        return False
    return MODULE_NAME in dotted.split(".")


def import_violations(tree: ast.AST) -> list[str]:
    """Every place in the parsed file that reaches for the truth store."""
    hits: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _names_truth_store(alias.name):
                    hits.append(f"line {node.lineno}: import {alias.name}")

        elif isinstance(node, ast.ImportFrom):
            dots = "." * node.level
            source = f"{dots}{node.module or ''}"
            imported = [alias.name for alias in node.names]
            if _names_truth_store(node.module) or MODULE_NAME in imported:
                hits.append(
                    f"line {node.lineno}: from {source} import {', '.join(imported)}"
                )

        # importlib.import_module("src.eval.truth_store"), __import__(...),
        # spec_from_file_location(...), or any other string that names the
        # module -- string constants are enough to catch all of these.
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if MODULE_NAME in node.value:
                hits.append(f"line {node.lineno}: string naming {MODULE_NAME!r}")

    return hits


def path_violations(source: str) -> list[str]:
    """Line numbers where the source text mentions the truth directory."""
    hits: list[str] = []
    for lineno, line in enumerate(source.splitlines(), start=1):
        if any(needle in line for needle in TRUTH_PATH_NEEDLES):
            hits.append(f"line {lineno}: mentions the planted-truth directory path")
    return hits


def logprob_violations(source: str) -> list[str]:
    """Line numbers where the source text mentions logprobs, with the line."""
    hits: list[str] = []
    for lineno, line in enumerate(source.splitlines(), start=1):
        if LOGPROB_NEEDLE in line.lower():
            hits.append(f"line {lineno}: mentions logprobs -- {line.strip()[:100]}")
    return hits


def is_raw_material(name: str) -> bool:
    """True for a filename that means responder prompts, completions or raw answers."""
    lowered = name.lower()
    return any(fnmatch(lowered, pattern) for pattern in RAW_MATERIAL_PATTERNS)


def test_repo_has_python_files_to_check() -> None:
    """Guard against the walk silently finding nothing."""
    files = python_files()
    assert len(files) >= 5, f"only found {len(files)} python files -- walk is broken"


def test_truth_store_lives_behind_the_wall() -> None:
    """Guard against the accessor being moved, which would make the test vacuous."""
    accessor = REPO_ROOT / "src" / "eval" / f"{MODULE_NAME}.py"
    assert accessor.is_file(), f"expected the truth-store accessor at {accessor}"


def test_no_module_outside_eval_touches_the_truth_store() -> None:
    violations: list[str] = []

    for path in python_files():
        if is_exempt(path):
            continue

        relative = path.relative_to(REPO_ROOT)
        source = path.read_text(encoding="utf-8")

        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            violations.append(f"{relative}: cannot be parsed, so cannot be cleared ({exc})")
            continue

        for hit in import_violations(tree) + path_violations(source):
            violations.append(f"{relative}: {hit}")

    assert not violations, "Wall violations found:\n" + "\n".join(
        f"  - {v}" for v in violations
    )


def test_the_noise_layer_really_does_mention_logprobs() -> None:
    """Guard against the logprob check passing because the word changed."""
    layer = REPO_ROOT / "src" / "personas" / "noise_layer.py"
    assert layer.is_file(), f"expected the noise layer at {layer}"
    assert LOGPROB_NEEDLE in layer.read_text(encoding="utf-8").lower(), (
        f"the noise layer no longer contains {LOGPROB_NEEDLE!r}, so the check "
        "below would pass on any repository -- update the needle"
    )


def test_no_module_outside_eval_and_personas_mentions_logprobs() -> None:
    """The system side never touches the responder's answer distribution.

    Recorded logprobs are the persona's true answer distribution, which is what
    Stage 4 predicts and grades. Reading them anywhere on the system side is
    reading the answer key.
    """
    violations: list[str] = []

    for path in python_files():
        if is_logprob_exempt(path):
            continue
        relative = path.relative_to(REPO_ROOT)
        for hit in logprob_violations(path.read_text(encoding="utf-8")):
            violations.append(f"{relative}: {hit}")

    assert not violations, (
        "logprobs mentioned outside src/eval/ and src/personas/:\n"
        + "\n".join(f"  - {v}" for v in violations)
    )


def test_run_directories_hold_no_raw_material() -> None:
    """``data/runs/`` is system-visible, so the raw material must not be in it.

    Responder prompts (persona cards), completions (answer-token
    distributions) and the un-noised ``answers.jsonl`` all belong in the
    planted-truth directory's own ``runs/<run_id>/`` folder. What may stay
    here: the noised answers, the retest selection, the aggregate noise stats.
    """
    runs = REPO_ROOT.joinpath(*RUNS_DIR)
    if not runs.is_dir():
        pytest.skip("no run directory in this checkout -- nothing to check")

    found = sorted(
        str(path.relative_to(REPO_ROOT))
        for path in runs.rglob("*")
        if path.is_file() and is_raw_material(path.name)
    )

    assert not found, (
        "raw material found on the system side of the Wall -- move it under "
        "the truth store's runs directory:\n" + "\n".join(f"  - {p}" for p in found)
    )
