"""Importing a substrax module leaves the process alone.

A module that calls ``logging.basicConfig`` at import changes logging for everything imported after
it, test collection included, and a module that writes ``os.environ`` at import changes the
environment of every test collected after it. Entry points configure both in ``main()``; the
environment a test needs belongs in that test (``monkeypatch.setenv``).
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from pathlib import Path

import pytest

from substrax.testing.source_scans import (
    configures_logging,
    import_time_lines,
    python_files,
    writes_environment,
)


ROOT = Path(__file__).resolve().parents[2]
MODULES = [*python_files(ROOT, ("src", "scripts", "tests")), ROOT / "conftest.py"]

pytestmark = pytest.mark.contract


def test_the_corpus_reaches_the_package_scripts_tests_and_root_conftest() -> None:
    names = {path.relative_to(ROOT).as_posix() for path in MODULES}

    assert {
        "src/substrax/runtime/managed_env.py",
        "scripts/derive_status.py",
        "tests/ci/test_import_side_effects.py",
        "conftest.py",
    } <= names


@pytest.mark.parametrize(
    "matches", [configures_logging, writes_environment], ids=lambda matches: matches.__name__
)
def test_no_module_changes_the_process_at_import(matches: Callable[[ast.AST], bool]) -> None:
    found = [
        f"{path.relative_to(ROOT)}:{line}"
        for path in MODULES
        for line in import_time_lines(path, matches)
    ]

    assert found == []
