"""The job spec: what a remote run executes, independent of the provider that runs it.

A ``JobSpec`` is a record, written by the host that submits a job and read by the worker that runs
it, possibly under another substrax release; ``schema_version`` lets the reader refuse a spec it
does not understand instead of misreading it.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path, PurePosixPath

from substrax.records import read_record, UNKNOWN_FIELDS_REFUSED
from substrax.runtime import JaxRuntime
from substrax.typing import JsonValue


JOB_SPEC_VERSION = 1
"""The job spec layout this release writes and reads."""

# A name becomes a directory under the run's outputs, so it is one path component.
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_ACCELERATOR_KIND = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class MountAccess(StrEnum):
    """Whether a job may change what a mount holds."""

    READ_ONLY = "read-only"
    READ_WRITE = "read-write"


@dataclass(frozen=True, slots=True, kw_only=True)
class Accelerator:
    """The accelerators a job runs on.

    Attributes:
        kind: The accelerator's name as the providers spell it, such as ``"L4"``, ``"A100-80GB"``
            or ``"H100"``; a backend refuses a kind its provider does not offer.
        count: How many.
    """

    __pydantic_config__ = UNKNOWN_FIELDS_REFUSED

    kind: str
    count: int = 1

    def __post_init__(self) -> None:
        """Validate the kind and count.

        Raises:
            ValueError: If ``kind`` is not one word or ``count`` is below one.
        """
        if not _ACCELERATOR_KIND.fullmatch(self.kind):
            msg = f"accelerator kind {self.kind!r} is not one word such as 'L4' or 'A100-80GB'"
            raise ValueError(msg)
        if self.count < 1:
            msg = f"accelerator count must be at least 1, not {self.count}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True, kw_only=True)
class Mount:
    """Named storage a job sees at a path; each backend maps the name to its own storage.

    Attributes:
        name: The storage's name in the backend's settings, such as ``"datasets"``.
        path: The absolute directory the job sees it at.
        access: Whether the job may write to it.
    """

    __pydantic_config__ = UNKNOWN_FIELDS_REFUSED

    name: str
    path: str
    access: MountAccess

    def __post_init__(self) -> None:
        """Validate the name and path.

        Raises:
            ValueError: If the name is not one path component, or the path is not an absolute,
                normalised directory below the root.
        """
        _check_name("mount", self.name)
        posix = PurePosixPath(self.path)
        if not posix.is_absolute() or ".." in posix.parts or str(posix) in {"/", ""}:
            msg = f"mount path {self.path!r} is not an absolute directory below the root"
            raise ValueError(msg)
        if str(posix) != self.path:
            msg = f"mount path {self.path!r} is not normalised; write {str(posix)!r}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True, kw_only=True)
class Task:
    """One command a job runs, from the project's root, in the project's environment.

    Attributes:
        name: The task's name, and the directory its logs and outputs go to.
        argv: The command. A leading ``"python"`` is the project environment's interpreter.
        timeout_seconds: Seconds before the task is killed.
    """

    __pydantic_config__ = UNKNOWN_FIELDS_REFUSED

    name: str
    argv: tuple[str, ...]
    timeout_seconds: float

    def __post_init__(self) -> None:
        """Validate the name, command and timeout.

        Raises:
            ValueError: If the name is not one path component, the command is empty, or the
                timeout is not a positive finite number.
        """
        _check_name("task", self.name)
        if not self.argv:
            msg = f"task {self.name!r} has an empty argv"
            raise ValueError(msg)
        _check_timeout(f"task {self.name!r}", self.timeout_seconds)


@dataclass(frozen=True, slots=True, kw_only=True)
class JobSpec:
    """A job: its tasks and everything the machine that runs them must provide.

    Attributes:
        schema_version: The layout of this record; see :data:`JOB_SPEC_VERSION`.
        name: The job's name, such as ``"examples"``.
        tasks: The commands, run in order; a failed task does not stop the ones after it.
        timeout_seconds: Seconds the whole job may take; the backend stops it after that.
        accelerator: The accelerators it needs; ``None`` runs on CPUs.
        extras: The project's optional dependency groups installed with its locked dependencies.
        runtime: The JAX settings of every task.
        env: Variables set in every task before ``runtime`` is applied.
        mounts: The named storage the tasks read or write, besides the run's outputs.
    """

    __pydantic_config__ = UNKNOWN_FIELDS_REFUSED

    schema_version: int = JOB_SPEC_VERSION
    name: str
    tasks: tuple[Task, ...]
    timeout_seconds: float
    accelerator: Accelerator | None = None
    extras: tuple[str, ...] = ()
    runtime: JaxRuntime = field(default_factory=JaxRuntime)
    env: Mapping[str, str] = field(default_factory=dict)
    mounts: tuple[Mount, ...] = ()

    def __post_init__(self) -> None:
        """Validate the job as a whole.

        Raises:
            ValueError: If the version is not this release's, the name is not one path component,
                there are no tasks, the timeout is not a positive finite number, or two tasks or
                two mounts share a name, or two mounts a path.
        """
        _check_version(self.schema_version)
        _check_name("job", self.name)
        if not self.tasks:
            msg = f"job {self.name!r} needs at least one task"
            raise ValueError(msg)
        _check_timeout(f"job {self.name!r}", self.timeout_seconds)
        _check_unique("task name", (task.name for task in self.tasks))
        _check_unique("mount name", (mount.name for mount in self.mounts))
        _check_unique("mount path", (mount.path for mount in self.mounts))


def read_job_spec(data: Mapping[str, JsonValue]) -> JobSpec:  # noqa: DOC502  # the ValueErrors are raised by _check_version and read_record
    """Read a job spec from its parsed JSON object.

    Args:
        data: The JSON object ``dump_record`` wrote.

    Returns:
        The spec.

    Raises:
        ValueError: If the spec's version is not this release's (the message names both), or a
            field is missing or holds a value its annotation does not admit.
    """
    _check_version(data.get("schema_version"))
    return read_record(JobSpec, data)


def example_tasks(
    paths: Iterable[Path], *, project: Path, timeout_seconds: float
) -> tuple[Task, ...]:
    """Make one task per example script, run by the project's interpreter from its root.

    Args:
        paths: The examples, such as :func:`~substrax.examples.discover_examples` lists them.
        project: The project root the paths lie under.
        timeout_seconds: Each example's budget.

    Returns:
        The tasks, in the order of ``paths``, each named by its path relative to ``project`` with
        dots for separators and without the suffix.

    Raises:
        ValueError: If a path lies outside ``project``.
    """
    root = project.resolve()
    tasks: list[Task] = []
    for path in paths:
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            msg = f"example {path} is outside the project {project}"
            raise ValueError(msg)
        relative = resolved.relative_to(root)
        tasks.append(
            Task(
                name=".".join(relative.with_suffix("").parts),
                argv=("python", relative.as_posix()),
                timeout_seconds=timeout_seconds,
            )
        )
    return tuple(tasks)


def _check_version(version: object) -> None:
    if version != JOB_SPEC_VERSION:
        msg = (
            f"job spec version {version!r} is not the version this substrax reads "
            f"({JOB_SPEC_VERSION}); submit and run the job with the same substrax release"
        )
        raise ValueError(msg)


def _check_name(kind: str, name: str) -> None:
    if not _NAME.fullmatch(name):
        msg = f"{kind} name {name!r} is not one path component of letters, digits, '.', '_', '-'"
        raise ValueError(msg)


def _check_timeout(owner: str, seconds: float) -> None:
    if not (math.isfinite(seconds) and seconds > 0):
        msg = f"{owner} timeout must be a positive finite number of seconds, not {seconds}"
        raise ValueError(msg)


def _check_unique(kind: str, values: Iterable[str]) -> None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            msg = f"two entries share the {kind} {value!r}"
            raise ValueError(msg)
        seen.add(value)
