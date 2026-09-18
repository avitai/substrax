"""The environment that writes the format-2 fixtures is a committed lock, not a resolution.

An environment resolved at run time floats with PyPI between two jobs of one run (jax 0.11.2
reached PyPI between artifex's unit and end-to-end jobs and broke flax 0.12.9's import in the
second), and one derived from the project's lock drifts away from the release it is meant to
reproduce. The committed lock does neither.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ACTION = REPO_ROOT / ".github" / "actions" / "format2-fixtures" / "action.yml"
INPUTS = REPO_ROOT / "scripts" / "format2_fixture_requirements.in"
LOCK = REPO_ROOT / "scripts" / "format2_fixture_requirements.txt"
PINNED = re.compile(r"^[A-Za-z0-9_.\-]+(\[[^\]]+\])?==\S+")


def _requirement_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_the_action_runs_the_generator_under_the_committed_lock() -> None:
    action = yaml.safe_load(FIXTURE_ACTION.read_text(encoding="utf-8"))
    runs = [str(step.get("run", "")) for step in action["runs"]["steps"]]
    assert any(
        "--with-requirements scripts/format2_fixture_requirements.txt" in run for run in runs
    ), "the action resolves the fixture environment instead of installing the lock"
    assert not any(re.search(r"--with\s", run) for run in runs), (
        "the action adds a floating package"
    )


def test_every_package_of_the_fixture_lock_is_pinned_and_the_inputs_are_kept() -> None:
    locked = _requirement_lines(LOCK)
    assert locked, "the fixture lock is empty"
    for line in locked:
        assert PINNED.match(line), f"unpinned requirement in the fixture lock: {line}"
    locked_names = {re.split(r"[\[=]", line)[0].lower() for line in locked}
    for line in _requirement_lines(INPUTS):
        assert PINNED.match(line), f"an input is not an exact pin: {line}"
        assert line.lower() in {entry.split(";")[0].strip().lower() for entry in locked}, line
    assert {"substrax", "jax", "jaxlib", "flax", "orbax-checkpoint"} <= locked_names
