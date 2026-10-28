"""Hooks that other repositories run from substrax's ``.pre-commit-hooks.yaml``.

``no-dependabot-config`` refuses a Dependabot configuration: dependency updates arrive through
pull requests the maintainers open, never from a bot. Dependabot alerts stay on; they report
advisories and open nothing. The hook uses pre-commit's ``fail`` language, so it runs no code and
installs no environment.
"""

from __future__ import annotations

import os
import subprocess  # nosec B404
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
HOOK = "no-dependabot-config"


def _git_free_environment(store: Path) -> dict[str, str]:
    """The environment without the ``GIT_*`` variables a commit hook running pytest would set."""
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    return environment | {"PRE_COMMIT_HOME": str(store)}


def _try_hook(consumer: Path, store: Path) -> subprocess.CompletedProcess[str]:
    """Run substrax's hook over every tracked file of ``consumer``, as a consumer's CI does."""
    return subprocess.run(  # noqa: S603  # nosec B603 - this interpreter's pre-commit, fixed arguments
        [sys.executable, "-m", "pre_commit", "try-repo", str(ROOT), HOOK, "--all-files"],
        cwd=consumer,
        env=_git_free_environment(store),
        capture_output=True,
        text=True,
        check=False,
    )


def _consumer(tmp_path: Path, github_file: str) -> Path:
    """A git repository whose index holds a module and one file under ``.github``."""
    repo = tmp_path / "consumer"
    (repo / ".github").mkdir(parents=True)
    (repo / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / ".github" / github_file).write_text("version: 2\n", encoding="utf-8")
    environment = _git_free_environment(tmp_path / "store")
    for command in (["git", "init", "-q"], ["git", "add", "-A"]):
        subprocess.run(  # noqa: S603  # nosec B603 B607 - git with fixed arguments in a temporary repo
            command, cwd=repo, env=environment, check=True
        )
    return repo


@pytest.mark.parametrize("name", ["dependabot.yml", "dependabot.yaml"])
def test_no_dependabot_config_refuses_a_configuration(tmp_path: Path, name: str) -> None:
    """Either spelling of the configuration fails the hook, which names the file."""
    result = _try_hook(_consumer(tmp_path, name), tmp_path / "store")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "no Dependabot configuration" in result.stdout
    assert f".github/{name}" in result.stdout


def test_no_dependabot_config_passes_without_one(tmp_path: Path) -> None:
    """Other files under ``.github`` leave the hook with nothing to check."""
    result = _try_hook(_consumer(tmp_path, "release.yml"), tmp_path / "store")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "no Dependabot configuration" in result.stdout


def test_substrax_carries_no_dependabot_config(tmp_path: Path) -> None:
    """substrax's own tree passes the hook it publishes, so Dependabot opens no pull request."""
    result = _try_hook(ROOT, tmp_path / "store")
    assert result.returncode == 0, result.stdout + result.stderr
