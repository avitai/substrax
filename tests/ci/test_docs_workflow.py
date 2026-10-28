"""Every pull request builds the documentation strictly from the lock.

Read the Docs installs the ``docs`` extra with pip from ``pyproject.toml`` and does not fail on
warnings, so the shared gates' Docs job is the one strict build from the lock. The site's build
tools come from the lock, and a relock alone can break that build: mkdocstrings before 1.0.7 with
pymdown-extensions 12.2 or later fails it (``Highlight.__init__() missing 1 required positional
argument: 'md'``). A path filter that left out the lock once let such a relock through unbuilt, so
the build runs on every pull request; it takes about twenty seconds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml


WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"


def _load(name: str) -> dict[str, Any]:
    # BaseLoader keeps every scalar a string, so the `on:` key stays `on` rather than True.
    return yaml.load((WORKFLOWS / name).read_text(encoding="utf-8"), Loader=yaml.BaseLoader)  # noqa: S506


@pytest.mark.parametrize("event", ["pull_request", "push"])
def test_ci_runs_on_every_change_with_no_path_filter(event: str) -> None:
    trigger = _load("ci.yml")["on"][event]

    assert "paths" not in trigger
    assert "paths-ignore" not in trigger


def test_the_gates_build_the_docs_strictly_from_the_lock() -> None:
    steps = _load("python-gates.yml")["jobs"]["docs"]["steps"]
    build = next(step for step in steps if "mkdocs build" in str(step.get("run", "")))

    assert "uv sync --locked" in build["run"]
    assert "mkdocs build --strict" in build["run"]
