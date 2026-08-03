"""The mobility Wall test (PRD_MOBILITY.md sections 2 and 6).

Extends the parent's static-check pattern (``tests/test_wall.py``) to the
mobility planted-truth directory. Planted truth for this sub-study -- traveler
parameter vectors phi, full traveler cards, wobble parameters, designed
scenario loadings and attribute matrices -- lives in one directory under
``mobility/data/``, and exactly one module may read it: the truth-vault
accessor in ``mobility/src/eval/``.

Three static checks, parsed with ``ast`` and read as text, nothing executed:

  (a) no file outside ``mobility/src/eval/`` reaches for the truth-vault
      accessor in any form -- plain import, from-import, package-style,
      relative, or an importlib call with the module name in a string;

  (b) no file outside the exempt graders spells the truth-directory path.
      The parent wall test already enforces this repo-wide (its needle is a
      substring of the mobility path); this check repeats it with
      mobility-aware exemptions so the protection stands on its own;

  (c) no run directory under ``mobility/data/runs/`` holds raw material --
      traveler prompts (they carry the card), raw completions, or pre-noise
      answers. Same filename patterns as the parent check it copies. Skipped
      while there is no ``mobility/data/`` at all, as in a fresh clone.

The parent additionally bans any mention of recorded answer-token
distributions outside its own grader and persona side; that check walks the
whole repository, so it already covers every mobility file, and this test
must not (and does not) use the banned word itself. Field-level allowlists on
system-side artifacts (the parent's post-breach lesson) are added here in
Stage M1 when the first artifact schemas exist.

Exemptions, and only these:

  * truth vault and truth path: ``mobility/src/eval/`` plus this file (it must
    name what it forbids). The path check also exempts the parent's grader
    directory and the parent's ``tests/`` directory, both frozen and already
    policed by the parent wall test, because the parent grader legitimately
    names its own truth directory, whose path contains the same substring.

Do not add to the list. New mobility code that needs the truth vault belongs
in ``mobility/src/eval/``.
"""

from __future__ import annotations

import ast
from fnmatch import fnmatch
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Directories never walked.
SKIP_DIRS = {".venv", "venv", ".git", "__pycache__", ".pytest_cache", "node_modules", "build", "dist"}

MODULE_NAME = "truth_vault"

#: Allowed to reach for the truth vault or name its module.
VAULT_EXEMPT_DIRS = (("mobility", "src", "eval"),)
VAULT_EXEMPT_FILES = (("mobility", "tests", "test_wall_mobility.py"),)

#: Additionally allowed to spell a truth-directory path: the parent's grader
#: and the parent's frozen tests, which name the parent's own truth directory
#: and are policed by the parent wall test.
PATH_EXEMPT_DIRS = VAULT_EXEMPT_DIRS + (("src", "eval"), ("tests",))
PATH_EXEMPT_FILES = VAULT_EXEMPT_FILES

#: Assembled from pieces so this file does not contain the literal it forbids.
TRUTH_PATH_NEEDLES = ("data" + "/" + "truth", "data" + "\\" + "truth")

#: Where mobility run artifacts will live, and what may never be found there.
RUNS_DIR = ("mobility", "data", "runs")

#: Filename patterns that mean "raw material", matched case-insensitively.
#: Copied from the parent wall test: prompts carry the traveler card,
#: completions carry the recorded answer-token material, ``answers.jsonl`` is
#: the pre-noise answers, ``noise_stats.json`` is a planted noise parameter.
RAW_MATERIAL_PATTERNS = (
    "answers.jsonl",
    "answer_rejects.jsonl",
    "completions*.json*",
    "prompts*.json*",
    "*_prompts.json*",
    "*_prompt_*.json*",
    "*_completions.json*",
    "*_completion_*.json*",
    "noise_stats.json",
)


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


def is_vault_exempt(path: Path) -> bool:
    """Allowed to reach for the truth vault."""
    return _exempt(path, VAULT_EXEMPT_DIRS, VAULT_EXEMPT_FILES)


def is_path_exempt(path: Path) -> bool:
    """Allowed to spell a truth-directory path."""
    return _exempt(path, PATH_EXEMPT_DIRS, PATH_EXEMPT_FILES)


def _names_vault(dotted: str | None) -> bool:
    if not dotted:
        return False
    return MODULE_NAME in dotted.split(".")


def import_violations(tree: ast.AST) -> list[str]:
    """Every place in the parsed file that reaches for the truth vault."""
    hits: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _names_vault(alias.name):
                    hits.append(f"line {node.lineno}: import {alias.name}")

        elif isinstance(node, ast.ImportFrom):
            dots = "." * node.level
            source = f"{dots}{node.module or ''}"
            imported = [alias.name for alias in node.names]
            if _names_vault(node.module) or MODULE_NAME in imported:
                hits.append(
                    f"line {node.lineno}: from {source} import {', '.join(imported)}"
                )

        # importlib.import_module(...), __import__(...), spec_from_file_location
        # -- string constants are enough to catch all of these.
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if MODULE_NAME in node.value:
                hits.append(f"line {node.lineno}: string naming {MODULE_NAME!r}")

    return hits


def path_violations(source: str) -> list[str]:
    """Line numbers where the source text spells a truth-directory path."""
    hits: list[str] = []
    for lineno, line in enumerate(source.splitlines(), start=1):
        if any(needle in line for needle in TRUTH_PATH_NEEDLES):
            hits.append(f"line {lineno}: spells a planted-truth directory path")
    return hits


def is_raw_material(name: str) -> bool:
    """True for a filename that means prompts, completions or pre-noise answers."""
    lowered = name.lower()
    return any(fnmatch(lowered, pattern) for pattern in RAW_MATERIAL_PATTERNS)


def test_mobility_has_python_files_to_check() -> None:
    """Guard against the walk silently finding nothing."""
    mobility_files = [
        p for p in python_files()
        if p.relative_to(REPO_ROOT).parts[0] == "mobility"
    ]
    assert len(mobility_files) >= 2, (
        f"only found {len(mobility_files)} mobility python files -- walk is broken"
    )


def test_truth_vault_lives_behind_the_wall() -> None:
    """Guard against the accessor being moved, which would make the test vacuous."""
    accessor = REPO_ROOT / "mobility" / "src" / "eval" / f"{MODULE_NAME}.py"
    assert accessor.is_file(), f"expected the truth-vault accessor at {accessor}"


def test_no_module_outside_mobility_eval_touches_the_truth_vault() -> None:
    violations: list[str] = []

    for path in python_files():
        if is_vault_exempt(path):
            continue

        relative = path.relative_to(REPO_ROOT)
        source = path.read_text(encoding="utf-8")

        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            violations.append(f"{relative}: cannot be parsed, so cannot be cleared ({exc})")
            continue

        for hit in import_violations(tree):
            violations.append(f"{relative}: {hit}")

    assert not violations, "mobility Wall violations found:\n" + "\n".join(
        f"  - {v}" for v in violations
    )


def test_no_module_outside_the_graders_spells_a_truth_path() -> None:
    violations: list[str] = []

    for path in python_files():
        if is_path_exempt(path):
            continue
        relative = path.relative_to(REPO_ROOT)
        for hit in path_violations(path.read_text(encoding="utf-8")):
            violations.append(f"{relative}: {hit}")

    assert not violations, "truth-path mentions found:\n" + "\n".join(
        f"  - {v}" for v in violations
    )


def test_mobility_run_directories_hold_no_raw_material() -> None:
    """``mobility/data/runs/`` is system-visible; raw material must not be in it.

    Traveler prompts (they carry the card), raw completions and pre-noise
    answers belong in a ``runs/<run_id>/`` folder inside the mobility
    planted-truth directory, whose name this file deliberately never spells.
    """
    runs = REPO_ROOT.joinpath(*RUNS_DIR)
    if not runs.is_dir():
        pytest.skip("no mobility run directory in this checkout -- nothing to check")

    found = sorted(
        str(path.relative_to(REPO_ROOT))
        for path in runs.rglob("*")
        if path.is_file() and is_raw_material(path.name)
    )

    assert not found, (
        "raw material found on the system side of the mobility Wall -- move it "
        "under the truth vault's runs directory:\n"
        + "\n".join(f"  - {p}" for p in found)
    )


#: The only top-level keys a system-side noised answer record may have. An
#: allowlist on purpose (the parent's post-breach lesson): anything else
#: fails, whether or not there is a name for it yet.
NOISED_ANSWER_KEYS = {"pid", "sid", "round", "answer"}

#: Named separately only so the failure message can say what was found. The
#: allowlist above is the real rule. Assembled needles keep this file clean
#: of the words the parent test bans.
PRE_NOISE_FIELD_NAMES = frozenset({
    "raw", "completion", "answer_raw", "pre_noise", "card", "phi", "wobble",
    "dist_top", "gap", "log" + "probs", "log" + "prob",
})


def test_mobility_noised_answer_files_carry_only_the_allowed_fields() -> None:
    """Field-level allowlist on the noise layer's system-side output."""
    import json

    runs = REPO_ROOT.joinpath(*RUNS_DIR)
    if not runs.is_dir():
        pytest.skip("no mobility run directory in this checkout -- nothing to check")
    found = sorted(runs.rglob("answers_noised.jsonl"))
    if not found:
        pytest.skip("no noised answer files in this checkout -- nothing to check")

    offenders: list[str] = []
    for path in found:
        relative = path.relative_to(REPO_ROOT)
        with path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                keys = set(json.loads(line))
                named = sorted(PRE_NOISE_FIELD_NAMES & keys)
                extra = sorted(keys - NOISED_ANSWER_KEYS)
                if named:
                    offenders.append(f"{relative}: line {lineno} has truth-side field(s) {named}")
                elif extra:
                    offenders.append(f"{relative}: line {lineno} has unexpected field(s) {extra}")
                if named or extra:
                    break  # one finding per file is enough

    assert not offenders, (
        "system-side noised answer files must carry only "
        f"{sorted(NOISED_ANSWER_KEYS)}:\n" + "\n".join(f"  - {o}" for o in offenders)
    )
