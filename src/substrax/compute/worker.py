"""The program every backend runs on the remote machine: a job's tasks, their logs and a manifest.

Run as ``python -m substrax.compute.worker --spec spec.json --project . --outputs /outputs`` from
the project's environment. Each task runs from the project root with ``AVITAI_OUTPUT_DIR`` set to
``<outputs>/<task>``, so whatever it writes through
:func:`~substrax.artifacts.resolve_output_dir` lands beside its ``stdout.log`` and ``stderr.log``.
The worker writes ``<outputs>/manifest.json`` after every task, so a run cut short still says what
finished. Its exit code is 0 when every task succeeded and 1 otherwise.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess  # nosec B404 - runs the job's own commands, never through a shell
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import BinaryIO, IO

from substrax.artifacts import OUTPUT_DIR_ENV
from substrax.compute.spec import JobSpec, read_job_spec, Task
from substrax.records import dump_record, read_record, UNKNOWN_FIELDS_REFUSED
from substrax.runtime import child_environment
from substrax.typing import JsonValue


MANIFEST_VERSION = 1
MANIFEST_NAME = "manifest.json"
STDOUT_LOG = "stdout.log"
STDERR_LOG = "stderr.log"
_LOGS = frozenset({STDOUT_LOG, STDERR_LOG})
_PYTHON = "python"


class TaskStatus(StrEnum):
    """How a task ended."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed-out"


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskOutcome:
    """What one task did.

    Attributes:
        name: The task.
        argv: The command that ran, with ``python`` resolved to the interpreter.
        status: How it ended.
        returncode: Its exit code; ``None`` when it could not start.
        seconds: Wall-clock seconds from start to end.
        outputs: The files it wrote under its directory, relative to it, without the logs.
    """

    __pydantic_config__ = UNKNOWN_FIELDS_REFUSED

    name: str
    argv: tuple[str, ...]
    status: TaskStatus
    returncode: int | None
    seconds: float
    outputs: tuple[str, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class JobManifest:
    """What a run of a job did, task by task.

    Attributes:
        schema_version: The layout of this record.
        job: The job's name.
        tasks: The tasks that ran, in order.
        finished: Whether the worker ran every task; a run cut short leaves it ``False``.
    """

    __pydantic_config__ = UNKNOWN_FIELDS_REFUSED

    schema_version: int = MANIFEST_VERSION
    job: str
    tasks: tuple[TaskOutcome, ...] = ()
    finished: bool = False

    @property
    def succeeded(self) -> bool:
        """Whether every task succeeded."""
        return all(task.status is TaskStatus.SUCCEEDED for task in self.tasks)


def read_manifest(data: dict[str, JsonValue]) -> JobManifest:  # noqa: DOC503  # pydantic.ValidationError, a ValueError, is raised by read_record
    """Read a manifest from its parsed JSON object.

    Args:
        data: The JSON object the worker wrote.

    Returns:
        The manifest.

    Raises:
        ValueError: If the manifest's version is not this release's, or a field does not match
            its annotation.
    """
    if data.get("schema_version") != MANIFEST_VERSION:
        msg = (
            f"manifest version {data.get('schema_version')!r} is not the version this substrax "
            f"reads ({MANIFEST_VERSION})"
        )
        raise ValueError(msg)
    return read_record(JobManifest, data)


def run_job(
    spec: JobSpec, *, project: Path, outputs: Path, echo: bool = True, python: str = sys.executable
) -> JobManifest:
    """Run every task of ``spec`` in order and write the manifest after each.

    Each task runs for at most its own budget or what is left of the job's, whichever is less; a
    task the job's budget leaves no time for is recorded as timed out without starting.

    Args:
        spec: The job.
        project: The project root each task runs from.
        outputs: The directory the run's logs, outputs and manifest go to.
        echo: Also copy each line a task writes to this process's standard output or error,
            prefixed with the task's name, so the provider's log shows the run as it goes.
        python: The interpreter a task's leading ``"python"`` names.

    Returns:
        The manifest.
    """
    outputs.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + spec.timeout_seconds
    manifest = JobManifest(job=spec.name)
    for index, task in enumerate(spec.tasks, start=1):
        budget = min(task.timeout_seconds, deadline - time.monotonic())
        outcome = (
            _run_task(
                spec,
                dataclasses.replace(task, timeout_seconds=budget),
                project=project,
                outputs=outputs,
                echo=echo,
                python=python,
            )
            if budget > 0
            else _out_of_time(task, outputs=outputs, python=python)
        )
        manifest = JobManifest(
            job=spec.name,
            tasks=(*manifest.tasks, outcome),
            finished=index == len(spec.tasks),
        )
        _write_json(outputs / MANIFEST_NAME, dump_record(manifest))
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    """Run the job a spec file describes.

    Args:
        argv: The command-line arguments; this process's when ``None``.

    Returns:
        0 when every task succeeded, 1 otherwise.
    """
    parser = argparse.ArgumentParser(prog="python -m substrax.compute.worker")
    parser.add_argument("--spec", type=Path, required=True, help="the job spec's JSON file")
    parser.add_argument("--project", type=Path, default=Path(), help="the project root")
    parser.add_argument("--outputs", type=Path, required=True, help="where the run's files go")
    arguments = parser.parse_args(argv)
    spec = read_job_spec(json.loads(arguments.spec.read_text(encoding="utf-8")))
    manifest = run_job(spec, project=arguments.project, outputs=arguments.outputs)
    return 0 if manifest.succeeded else 1


def _run_task(
    spec: JobSpec, task: Task, *, project: Path, outputs: Path, echo: bool, python: str
) -> TaskOutcome:
    directory = outputs / task.name
    directory.mkdir(parents=True, exist_ok=True)
    argv = (python, *task.argv[1:]) if task.argv[0] == _PYTHON else task.argv
    env = child_environment(spec.runtime, {**spec.env, OUTPUT_DIR_ENV: str(directory.resolve())})
    started = time.perf_counter()
    with (
        (directory / STDOUT_LOG).open("wb") as out_log,
        (directory / STDERR_LOG).open("wb") as err_log,
    ):
        try:
            process = subprocess.Popen(  # noqa: S603  # nosec B603 - the job's command, no shell
                argv, cwd=project, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
        except OSError as error:
            err_log.write(f"{argv[0]} could not start: {error}\n".encode())
            status, returncode = TaskStatus.FAILED, None
        else:
            with process:
                status, returncode = _wait(process, task, (out_log, err_log), echo=echo)
    return TaskOutcome(
        name=task.name,
        argv=argv,
        status=status,
        returncode=returncode,
        seconds=time.perf_counter() - started,
        outputs=_written(directory),
    )


def _out_of_time(task: Task, *, outputs: Path, python: str) -> TaskOutcome:
    """The outcome of a task the job's budget left no time for; it gets empty logs."""
    directory = outputs / task.name
    directory.mkdir(parents=True, exist_ok=True)
    for log in (STDOUT_LOG, STDERR_LOG):
        (directory / log).write_bytes(b"")
    return TaskOutcome(
        name=task.name,
        argv=(python, *task.argv[1:]) if task.argv[0] == _PYTHON else task.argv,
        status=TaskStatus.TIMED_OUT,
        returncode=None,
        seconds=0.0,
        outputs=(),
    )


def _wait(
    process: subprocess.Popen[bytes],
    task: Task,
    logs: tuple[BinaryIO, BinaryIO],
    *,
    echo: bool,
) -> tuple[TaskStatus, int]:
    """Copy the task's streams while it runs; kill it at its timeout."""
    prefix = f"[{task.name}] ".encode()
    streams = (
        (process.stdout, logs[0], sys.stdout.buffer if echo else None),
        (process.stderr, logs[1], sys.stderr.buffer if echo else None),
    )
    copiers = [
        threading.Thread(target=_copy, args=(source, log, mirror, prefix), daemon=True)
        for source, log, mirror in streams
    ]
    for copier in copiers:
        copier.start()
    try:
        returncode = process.wait(timeout=task.timeout_seconds)
        status = TaskStatus.SUCCEEDED if returncode == 0 else TaskStatus.FAILED
    except subprocess.TimeoutExpired:
        process.kill()
        returncode = process.wait()
        status = TaskStatus.TIMED_OUT
    for copier in copiers:
        copier.join()
    return status, returncode


def _copy(source: IO[bytes] | None, log: BinaryIO, mirror: BinaryIO | None, prefix: bytes) -> None:
    """Write each line of ``source`` to the log and, prefixed, to the mirror."""
    if source is None:
        return
    for line in iter(source.readline, b""):
        log.write(line)
        log.flush()
        if mirror is not None:
            mirror.write(prefix + line)
            mirror.flush()


def _written(directory: Path) -> tuple[str, ...]:
    return tuple(
        path.relative_to(directory).as_posix()
        for path in sorted(directory.rglob("*"))
        if path.is_file() and not (path.parent == directory and path.name in _LOGS)
    )


def _write_json(path: Path, payload: dict[str, JsonValue]) -> None:
    """Write through a temporary file, so a reader never sees half a manifest."""
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    partial.replace(path)


if __name__ == "__main__":
    sys.exit(main())
