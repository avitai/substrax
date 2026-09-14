"""Root conftest for substrax test suite."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest


_CHILD_TIMEOUT_SECONDS = 180
_CHILD_DROPPED_PREFIXES = ("JAX_", "XLA_")


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for test data."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for test outputs."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def run_interpreter() -> Callable[[list[str], dict[str, str]], subprocess.CompletedProcess[str]]:
    """Run the test's own interpreter on ``argv`` in a fresh process.

    jax fixes its backends and device set when they start, so device-topology and
    process-configuration tests need their own interpreter. The child inherits the parent's
    environment without any ``JAX_*`` or ``XLA_*`` variable, runs on the CPU backend with client
    preallocation off, and then takes ``env`` on top, so settings exported in a developer's shell
    cannot change what a test measures.

    Returns:
        A function of ``(argv, env)`` returning the completed child process.
    """

    def run(argv: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
        inherited = {
            name: value
            for name, value in os.environ.items()
            if not name.startswith(_CHILD_DROPPED_PREFIXES)
        }
        child_env = {
            **inherited,
            "JAX_PLATFORMS": "cpu",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            **env,
        }
        return subprocess.run(  # noqa: S603 - fixed argv: the test's interpreter and its own code
            [sys.executable, *argv],
            env=child_env,
            capture_output=True,
            text=True,
            timeout=_CHILD_TIMEOUT_SECONDS,
            check=False,
        )

    return run
