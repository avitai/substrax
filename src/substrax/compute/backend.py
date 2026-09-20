"""The interface every compute provider implements, and the handles that name submitted runs.

A backend turns a :class:`~substrax.compute.JobSpec` into a run on its provider and reports on it
afterwards. It returns a :class:`RunHandle` from :meth:`ComputeBackend.submit` and needs nothing
else to find the run again, so a run submitted from one shell can be followed, fetched or
cancelled from another. :class:`RunStore` keeps the handles.
"""

from __future__ import annotations

import json
import os
import secrets
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import StrEnum
from pathlib import Path
from typing import Protocol, runtime_checkable

from substrax.compute.spec import JobSpec
from substrax.records import dump_record, read_record, UNKNOWN_FIELDS_REFUSED
from substrax.typing import JsonValue


HANDLE_VERSION = 1
SPEC_NAME = "spec.json"
"""The file a staged run's spec is written to."""
STATE_DIR_ENV = "SUBSTRAX_STATE_DIR"
"""Overrides the directory substrax keeps run handles and local runs in."""


class RunState(StrEnum):
    """Where a run is."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_final(self) -> bool:
        """Whether the run has ended."""
        return self in {RunState.SUCCEEDED, RunState.FAILED, RunState.CANCELLED}


@dataclass(frozen=True, slots=True, kw_only=True)
class RunHandle:
    """Everything needed to find a submitted run again.

    Attributes:
        schema_version: The layout of this record.
        run_id: The run's name, unique across backends; also its outputs' directory name.
        backend: The name of the backend that ran it.
        job: The job's name.
        project: The project root it was submitted from.
        submitted_at: When it was submitted, in UTC.
        details: The provider's own references to the run, such as an app or cluster id.
    """

    __pydantic_config__ = UNKNOWN_FIELDS_REFUSED

    schema_version: int = HANDLE_VERSION
    run_id: str
    backend: str
    job: str
    project: str
    submitted_at: datetime
    details: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class NewRun:
    """A run being submitted: its id, its job and its project, until the provider names it.

    Attributes:
        run_id: The run's id, from :func:`new_run_id`.
        spec: The job.
        project: The resolved project root.
    """

    run_id: str
    spec: JobSpec
    project: Path

    def handle(self, backend: str, details: Mapping[str, str]) -> RunHandle:
        """The handle of this run, submitted now.

        Args:
            backend: The backend's name.
            details: The provider's references to the run.

        Returns:
            The handle.
        """
        return RunHandle(
            run_id=self.run_id,
            backend=backend,
            job=self.spec.name,
            project=str(self.project),
            submitted_at=datetime.now(UTC),
            details=details,
        )


@runtime_checkable
class ComputeBackend(Protocol):
    """A compute provider that runs jobs.

    Every method but :meth:`submit` takes a handle this backend returned, possibly in another
    process, and must work from it alone.
    """

    name: str
    """The name the backend is registered under."""

    def submit(self, spec: JobSpec, *, project: Path) -> RunHandle:
        """Start ``spec`` from the project at ``project`` and return once it is submitted."""
        ...

    def status(self, handle: RunHandle) -> RunState:
        """Where the run is now."""
        ...

    def logs(self, handle: RunHandle, *, follow: bool) -> Iterator[str]:
        """The worker's output lines; with ``follow``, until the run ends."""
        ...

    def fetch(self, handle: RunHandle, destination: Path) -> Path:
        """Copy the run's outputs to ``destination / handle.run_id`` and return that directory."""
        ...

    def cancel(self, handle: RunHandle) -> None:
        """Stop the run; a run that already ended is left as it is."""
        ...


BackendFactory = Callable[[Mapping[str, JsonValue]], ComputeBackend]
"""What a backend's entry point loads: it builds the backend from its settings table."""


def new_run_id(job: str) -> str:
    """Name a new run of ``job`` by its UTC start and a random suffix.

    Args:
        job: The job's name.

    Returns:
        An id such as ``examples-20260919T120000Z-3f9a1c``.
    """
    return f"{job}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{secrets.token_hex(3)}"


def default_state_dir() -> Path:
    """The directory substrax keeps run handles and local runs in.

    Returns:
        ``$SUBSTRAX_STATE_DIR`` when set, else ``$XDG_STATE_HOME/substrax``, else
        ``~/.local/state/substrax``.
    """
    chosen = os.environ.get(STATE_DIR_ENV)
    if chosen:
        return Path(chosen)
    state_home = os.environ.get("XDG_STATE_HOME")
    base = Path(state_home) if state_home else Path.home() / ".local" / "state"
    return base / "substrax"


def backend_state_dir(state_dir: str | None, backend: str) -> Path:
    """The directory a backend keeps its runs' local files in.

    Args:
        state_dir: The backend's ``state_dir`` setting; the default state directory when ``None``.
        backend: The backend's name.

    Returns:
        ``<state_dir>/compute/<backend>``.
    """
    root = Path(state_dir) if state_dir else default_state_dir()
    return root / "compute" / backend


def new_run(spec: JobSpec, project: Path) -> NewRun:
    """Start submitting a run of ``spec`` from ``project``.

    Args:
        spec: The job.
        project: The project root.

    Returns:
        The run, named by :func:`new_run_id`.
    """
    return NewRun(run_id=new_run_id(spec.name), spec=spec, project=project.resolve())


def stage_run(runs: Path, run: NewRun) -> Path:
    """Write the run's spec into a directory of its own under ``runs``.

    Args:
        runs: The backend's directory of runs.
        run: The run.

    Returns:
        Its directory, which holds ``spec.json``.
    """
    run_dir = runs / run.run_id
    run_dir.mkdir(parents=True)
    (run_dir / SPEC_NAME).write_text(json.dumps(dump_record(run.spec)), encoding="utf-8")
    return run_dir


class RunStore:
    """The handles of submitted runs, one JSON file per run."""

    def __init__(self, root: Path | None = None) -> None:
        """Keep handles under ``root``.

        Args:
            root: The directory; ``default_state_dir() / "compute" / "runs"`` when ``None``.
        """
        self.root = root or default_state_dir() / "compute" / "runs"

    def save(self, handle: RunHandle) -> None:
        """Write a handle, replacing an earlier one of the same run.

        Args:
            handle: The handle.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        partial = self.root / f".{handle.run_id}.partial"
        partial.write_text(json.dumps(dump_record(handle), indent=2) + "\n", encoding="utf-8")
        partial.replace(self._path(handle.run_id))

    def load(self, run_id: str) -> RunHandle:
        """Read the handle of a run.

        Args:
            run_id: The run.

        Returns:
            Its handle.

        Raises:
            KeyError: If no handle of that run is kept; the message names the directory.
        """
        path = self._path(run_id)
        if not path.is_file():
            msg = f"no run {run_id!r} in {self.root}"
            raise KeyError(msg)
        return read_record(RunHandle, json.loads(path.read_text(encoding="utf-8")))

    def handles(self) -> list[RunHandle]:
        """Every kept handle, newest first.

        Returns:
            The handles.
        """
        if not self.root.is_dir():
            return []
        found = [
            read_record(RunHandle, json.loads(path.read_text(encoding="utf-8")))
            for path in self.root.glob("*.json")
        ]
        return sorted(found, key=lambda handle: handle.submitted_at, reverse=True)

    def _path(self, run_id: str) -> Path:
        return self.root / f"{run_id}.json"
