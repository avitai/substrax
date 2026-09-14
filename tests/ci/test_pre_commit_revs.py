"""Remote pre-commit hooks run the tool versions the lock pins.

CI lints with the locked tools. A hook on another version checks different rules and formats
differently, so a commit the hook accepts can fail CI, and the reverse.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
# Remote hook repositories whose tool the lock also installs, with the locked package each must match.
LOCKED_HOOK_TOOLS = {"https://github.com/astral-sh/ruff-pre-commit": "ruff"}


def _hook_revs(repo: str) -> list[str]:
    config = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8"))
    return [entry["rev"] for entry in config["repos"] if entry["repo"] == repo]


def _locked_versions(package: str) -> list[str]:
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    return [entry["version"] for entry in lock["package"] if entry["name"] == package]


@pytest.mark.parametrize(("repo", "package"), sorted(LOCKED_HOOK_TOOLS.items()))
def test_a_remote_hook_runs_the_locked_version_of_its_tool(repo: str, package: str) -> None:
    locked = _locked_versions(package)

    assert len(locked) == 1, locked
    assert _hook_revs(repo) == [f"v{locked[0]}"]
