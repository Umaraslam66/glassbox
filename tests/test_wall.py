"""The Wall test (PRD section 6.1).

Static check over every Python file in the repository. A file outside
``src/eval/`` fails the Wall if it either

  (a) imports the truth-store accessor in any form -- plain ``import``,
      ``from ... import``, package-style ``src.eval.truth_store``, a relative
      import, or an ``importlib`` call with the module name as a string; or
  (b) mentions the planted-truth directory path in its source text.

Nothing is executed: the files are parsed with ``ast`` and read as text, so this
is safe to run on any commit and catches leaks that a runtime test would miss.

Two files are exempt besides ``src/eval/`` itself, and only these two:
this file (it must name the forbidden things in order to look for them) and
``tests/test_truth_store.py`` (the accessor's own unit test, which is
grader-side code by definition). Do not add to this list -- if new code needs
the truth store, it belongs in ``src/eval/``.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Directories never walked.
SKIP_DIRS = {".venv", "venv", ".git", "__pycache__", ".pytest_cache", "node_modules", "build", "dist"}

#: Paths (relative to the repo root) allowed to reference the truth store.
EXEMPT_DIRS = (("src", "eval"),)
EXEMPT_FILES = (("tests", "test_wall.py"), ("tests", "test_truth_store.py"))

MODULE_NAME = "truth_store"

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


def is_exempt(path: Path) -> bool:
    parts = path.relative_to(REPO_ROOT).parts
    if parts in EXEMPT_FILES:
        return True
    return any(parts[: len(prefix)] == prefix for prefix in EXEMPT_DIRS)


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
