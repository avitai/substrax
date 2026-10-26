"""The Documentation workflow builds the site strictly whenever something that can break it changes.

Read the Docs installs the ``docs`` extra with pip from ``pyproject.toml`` and does not fail on
warnings, so ``docs.yml`` is the one strict build from the lock. The site's build tools come from
the lock, and a relock alone can break that build: mkdocstrings before 1.0.7 with
pymdown-extensions 12.2 or later fails it (``Highlight.__init__() missing 1 required positional
argument: 'md'``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml


DOCS_WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "docs.yml"


def _triggers() -> dict[str, Any]:
    # BaseLoader keeps every scalar a string, so the `on:` key stays `on` rather than True.
    return yaml.load(DOCS_WORKFLOW.read_text(), Loader=yaml.BaseLoader)["on"]  # noqa: S506


@pytest.mark.parametrize("event", ["pull_request", "push"])
def test_a_dependency_change_rebuilds_the_docs(event: str) -> None:
    paths = _triggers()[event]["paths"]

    assert "uv.lock" in paths
    assert "pyproject.toml" in paths


def test_both_events_watch_the_same_paths() -> None:
    triggers = _triggers()

    assert triggers["pull_request"]["paths"] == triggers["push"]["paths"]
