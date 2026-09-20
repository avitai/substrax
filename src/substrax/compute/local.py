"""The ``local`` backend: runs a job's worker on this machine, detached from the shell.

It runs the project's current environment as it is (``uv run --no-sync``), so a job's ``extras``
and ``accelerator`` are not provisioned here: the machine has what it has. It is the reference
backend the contract tests run against, and it serves a developer with a local accelerator.

Settings (``[tool.substrax.compute.backends.local]``):

- ``python``: the interpreter command that runs the worker, such as ``".venv/bin/python"``;
  ``uv run --no-sync --project <project> python`` when unset.
- ``state_dir``: where runs are kept; :func:`~substrax.compute.backend.default_state_dir` when
  unset.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess  # nosec B404 - runs ps to confirm a pid is still this run's worker
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

from substrax.compute.backend import (
    backend_state_dir,
    new_run,
    RunHandle,
    RunState,
    SPEC_NAME,
    stage_run,
)
from substrax.compute.spec import JobSpec
from substrax.compute.worker import JobManifest, MANIFEST_NAME, read_manifest
from substrax.records import read_record, UNKNOWN_FIELDS_REFUSED
from substrax.typing import JsonValue


NAME = "local"
_OUTPUTS = "outputs"
_WORKER_LOG = "worker.log"
_CANCELLED = "cancelled"
_POLL_SECONDS = 0.2


@dataclass(frozen=True, slots=True, kw_only=True)
class LocalSettings:
    """The ``local`` backend's settings.

    Attributes:
        python: The interpreter command that runs the worker; the project's through ``uv`` when
            ``None``.
        state_dir: Where runs are kept; the default state directory when ``None``.
    """

    __pydantic_config__ = UNKNOWN_FIELDS_REFUSED

    python: str | None = None
    state_dir: str | None = None


class LocalBackend:
    """Runs jobs on this machine."""

    name = NAME

    def __init__(self, settings: Mapping[str, JsonValue] | None = None) -> None:  # noqa: DOC502  # pydantic.ValidationError, a ValueError, is raised by read_record
        """Read the backend's settings.

        Args:
            settings: The settings table; every setting has a default.

        Raises:
            ValueError: If a setting is unknown or of the wrong type.
        """
        chosen = read_record(LocalSettings, settings or {})
        self._python = chosen.python
        self._runs = backend_state_dir(chosen.state_dir, NAME)

    def submit(self, spec: JobSpec, *, project: Path) -> RunHandle:
        """Start the worker on ``spec`` in the background and return its handle.

        Args:
            spec: The job.
            project: The project root.

        Returns:
            The run's handle.
        """
        run = new_run(spec, project)
        root, run_dir = run.project, stage_run(self._runs, run)
        argv = [
            *self._interpreter(root),
            "-m",
            "substrax.compute.worker",
            "--spec",
            str(run_dir / SPEC_NAME),
            "--project",
            str(root),
            "--outputs",
            str(run_dir / _OUTPUTS),
        ]
        log_flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        pid = os.posix_spawnp(
            argv[0],
            argv,
            dict(os.environ),
            file_actions=[
                (os.POSIX_SPAWN_OPEN, 1, str(run_dir / _WORKER_LOG), log_flags, 0o644),
                (os.POSIX_SPAWN_DUP2, 1, 2),
            ],
            setpgroup=0,
        )
        return run.handle(NAME, {"pid": str(pid), "run_dir": str(run_dir)})

    def status(self, handle: RunHandle) -> RunState:
        """Where the run is: from its manifest when it finished, else from its process.

        Args:
            handle: The run.

        Returns:
            Its state.
        """
        run_dir = Path(handle.details["run_dir"])
        manifest = _manifest(run_dir)
        if manifest is not None and manifest.finished:
            return RunState.SUCCEEDED if manifest.succeeded else RunState.FAILED
        if (run_dir / _CANCELLED).exists():
            return RunState.CANCELLED
        if _is_worker(int(handle.details["pid"]), handle.run_id):
            return RunState.RUNNING
        return RunState.FAILED

    def logs(self, handle: RunHandle, *, follow: bool) -> Iterator[str]:
        """The worker's output lines, including every task's, prefixed with the task's name.

        Args:
            handle: The run.
            follow: Keep reading until the run ends.

        Yields:
            str: Each line, without its line ending.
        """
        path = Path(handle.details["run_dir"]) / _WORKER_LOG
        with path.open(encoding="utf-8", errors="replace") as log:
            while True:
                line = log.readline()
                if line:
                    yield line.rstrip("\n")
                    continue
                if not follow or self.status(handle).is_final:
                    yield from (rest.rstrip("\n") for rest in log.readlines())
                    return
                time.sleep(_POLL_SECONDS)

    def fetch(self, handle: RunHandle, destination: Path) -> Path:
        """Copy the run's outputs to ``destination / handle.run_id``.

        Args:
            handle: The run.
            destination: The directory to copy into.

        Returns:
            The copy.
        """
        target = destination / handle.run_id
        shutil.copytree(Path(handle.details["run_dir"]) / _OUTPUTS, target, dirs_exist_ok=True)
        return target

    def cancel(self, handle: RunHandle) -> None:
        """Stop the worker and its tasks; a run that already ended is left as it is.

        Args:
            handle: The run.
        """
        if self.status(handle).is_final:
            return
        (Path(handle.details["run_dir"]) / _CANCELLED).touch()
        pid = int(handle.details["pid"])
        if _is_worker(pid, handle.run_id):
            # The worker leads its own process group, which its tasks inherit.
            os.killpg(pid, signal.SIGTERM)

    def _interpreter(self, project: Path) -> list[str]:
        if self._python is not None:
            return [self._python]
        return ["uv", "run", "--no-sync", "--project", str(project), "python"]


def _manifest(run_dir: Path) -> JobManifest | None:
    path = run_dir / _OUTPUTS / MANIFEST_NAME
    if not path.is_file():
        return None
    return read_manifest(json.loads(path.read_text(encoding="utf-8")))


def _is_worker(pid: int, run_id: str) -> bool:
    """Whether ``pid`` is alive and still this run's worker, not a later process given its pid."""
    completed = subprocess.run(  # noqa: S603  # nosec B603 B607 - ps with a numeric pid, no shell
        ["ps", "-o", "command=", "-p", str(pid)],  # noqa: S607  # ps is on every POSIX PATH
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode == 0 and run_id in completed.stdout
