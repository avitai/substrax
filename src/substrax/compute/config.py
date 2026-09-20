"""A project's compute configuration: the ``[tool.substrax.compute]`` table of its pyproject.toml.

```toml
[tool.substrax.compute]
backend = "modal"                        # used when the command line names none

[tool.substrax.compute.backends.modal]   # passed to the backend as its settings
outputs_volume = "demo-outputs"

[tool.substrax.compute.jobs.examples]
examples = ["examples/metrics"]          # one task per script, as discover_examples lists them
commands = { gpu-tests = ["python", "-m", "pytest", "tests/gpu"] }
extras = ["cuda12"]
accelerator = { kind = "L4", count = 1 }
timeout_seconds = 1800
task_timeout_seconds = 300               # each task's budget; the job's when unset
runtime = { platforms = ["cuda"], xla_flags = ["--xla_gpu_deterministic_ops=true"] }
env = { TF_CPP_MIN_LOG_LEVEL = "1" }
mounts = [{ name = "datasets", path = "/data", access = "read-only" }]
```

A misspelled key is refused rather than ignored.
"""

from __future__ import annotations

import dataclasses
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from substrax.compute.spec import Accelerator, example_tasks, JobSpec, Task
from substrax.examples import discover_examples
from substrax.records import dump_record, read_record, UNKNOWN_FIELDS_REFUSED
from substrax.typing import JsonValue


_TABLE = "[tool.substrax.compute]"
_ACCELERATOR = re.compile(r"(?P<kind>[^:]+)(?::(?P<count>\d+))?")
_TASK_SOURCE_KEYS = ("examples", "commands", "task_timeout_seconds")


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskSources:
    """Where a configured job's tasks come from, and their budget.

    The job's other keys are :class:`~substrax.compute.JobSpec` fields.

    Attributes:
        timeout_seconds: The job's budget, which is also a ``JobSpec`` field.
        examples: Directories, relative to the project, whose example scripts become tasks.
        commands: Further tasks, by name, run after the examples.
        task_timeout_seconds: Each task's budget; ``timeout_seconds`` when ``None``.
    """

    timeout_seconds: float
    examples: tuple[str, ...] = ()
    commands: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    task_timeout_seconds: float | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ComputeConfig:
    """A project's compute configuration.

    Attributes:
        backend: The backend used when none is named; ``None`` requires naming one.
        backends: Each backend's settings, by backend name.
        jobs: Each job's table, by name, read into a ``JobSpec`` when the job is resolved.
    """

    __pydantic_config__ = UNKNOWN_FIELDS_REFUSED

    backend: str | None = None
    backends: Mapping[str, Mapping[str, JsonValue]] = field(default_factory=dict)
    jobs: Mapping[str, Mapping[str, JsonValue]] = field(default_factory=dict)


def read_compute_config(project: Path) -> ComputeConfig:  # noqa: DOC503  # pydantic.ValidationError, a ValueError, is raised by read_record
    """Read the ``[tool.substrax.compute]`` table of the project's pyproject.toml.

    Args:
        project: The project root.

    Returns:
        The configuration.

    Raises:
        LookupError: If pyproject.toml has no such table.
        ValueError: If a key is unknown or a value is of the wrong type.
    """
    document = tomllib.loads((project / "pyproject.toml").read_text(encoding="utf-8"))
    table = document.get("tool", {}).get("substrax", {}).get("compute")
    if table is None:
        msg = f"{project / 'pyproject.toml'} has no {_TABLE} table"
        raise LookupError(msg)
    return read_record(ComputeConfig, table)


def resolve_job(  # noqa: DOC503  # pydantic.ValidationError, a ValueError, is raised by read_record
    config: ComputeConfig,
    name: str,
    *,
    project: Path,
    accelerator: Accelerator | None = None,
) -> JobSpec:
    """Build the spec of a configured job, listing its examples now.

    The job's table holds ``examples``, ``commands`` and ``task_timeout_seconds``, which make its
    tasks, and otherwise the fields of :class:`~substrax.compute.JobSpec` under their own names.

    Args:
        config: The project's configuration.
        name: The job.
        project: The project root.
        accelerator: Replaces the configured accelerator for this run.

    Returns:
        The spec.

    Raises:
        KeyError: If the job is not configured; the message lists the configured ones.
        FileNotFoundError: If an examples directory does not exist.
        ValueError: If a key of the job's table is unknown or a value of the wrong type.
    """
    if name not in config.jobs:
        known = ", ".join(sorted(config.jobs)) or "none"
        msg = f"no job {name!r}; configured: {known}"
        raise KeyError(msg)
    table = dict(config.jobs[name])
    sources = read_record(
        TaskSources,
        {
            **{key: table.pop(key) for key in _TASK_SOURCE_KEYS if key in table},
            "timeout_seconds": table.get("timeout_seconds"),
        },
    )
    budget = sources.task_timeout_seconds or sources.timeout_seconds
    tasks = _example_tasks(sources, name, project, budget) + [
        Task(name=task, argv=argv, timeout_seconds=budget)
        for task, argv in sources.commands.items()
    ]
    spec = read_record(
        JobSpec, {**table, "name": name, "tasks": [dump_record(task) for task in tasks]}
    )
    return spec if accelerator is None else dataclasses.replace(spec, accelerator=accelerator)


def _example_tasks(sources: TaskSources, job: str, project: Path, budget: float) -> list[Task]:
    tasks: list[Task] = []
    for directory in sources.examples:
        root = project / directory
        if not root.is_dir():
            msg = f"examples directory {directory!r} of job {job!r} does not exist in {project}"
            raise FileNotFoundError(msg)
        tasks += example_tasks(discover_examples(root), project=project, timeout_seconds=budget)
    return tasks


def parse_accelerator(text: str) -> Accelerator:
    """Read an accelerator written as ``KIND`` or ``KIND:COUNT``, such as ``H100:8``.

    Args:
        text: The accelerator.

    Returns:
        It.

    Raises:
        ValueError: If ``text`` is not of that form.
    """
    matched = _ACCELERATOR.fullmatch(text)
    if matched is None:
        msg = f"accelerator {text!r} is not KIND or KIND:COUNT, such as 'L4' or 'H100:8'"
        raise ValueError(msg)
    count = matched["count"]
    return Accelerator(kind=matched["kind"], count=int(count) if count else 1)


def require_locked_substrax(project: Path) -> str:
    """Return the substrax the project locks, which the worker runs from.

    Args:
        project: The project root.

    Returns:
        The locked version, or ``"editable"`` when the project is substrax or installs it from a
        path.

    Raises:
        FileNotFoundError: If the project has no uv.lock.
        LookupError: If its uv.lock does not include substrax.
    """
    lock = project / "uv.lock"
    if not lock.is_file():
        msg = f"{project} has no uv.lock; the remote environment is built from it"
        raise FileNotFoundError(msg)
    for package in tomllib.loads(lock.read_text(encoding="utf-8")).get("package", []):
        if package.get("name") == "substrax":
            return package.get("version") or "editable"
    msg = f"substrax is not in {lock}; add it to the project's dependencies to run the worker"
    raise LookupError(msg)
