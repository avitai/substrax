"""A project's compute configuration: the ``[tool.substrax.compute]`` table of its pyproject.toml.

```toml
[tool.substrax.compute]
backend = "modal"                        # used when the command line names none

[tool.substrax.compute.backends.modal]   # passed to the backend as its settings
outputs_volume = "demo-outputs"

[tool.substrax.compute.jobs.examples]
examples = ["examples/metrics"]          # one task per example, as discover_examples lists them
examples_mode = "test"                   # run as its notebook: "test", or "page" to make pages
commands = { gpu-tests = ["python", "-m", "pytest", "tests/gpu"] }
extras = ["cuda12"]
resources = { accelerator = { kind = "L4", count = 1 }, cpu = 8, memory_mib = 32768 }
timeout_seconds = 1800
command_timeout_seconds = 1200           # each task's limit; the job's when unset
runtime = { platforms = ["cuda", "cpu"], xla_flags = ["--xla_gpu_deterministic_ops=true"] }
env = { TF_CPP_MIN_LOG_LEVEL = "1" }
mounts = [{ name = "datasets", path = "/data", access = "read-only" }]

[tool.substrax.examples]                 # each example's budget, substrax.notebooks.config
source_url = "https://github.com/avitai/demo/blob/main"
budget_seconds = 600
```

An example task runs its example as its clean notebook in the job's ``examples_mode``
(:class:`~substrax.compute.ExampleMode`), held by the run to the example's own budget from
``[tool.substrax.examples]`` (its ``budgets`` entry, else ``budget_seconds``), as
``substrax-notebooks page`` and the shared example tests hold it, so a job listing ``examples``
needs that table; a task limit below an example's budget is refused. A misspelled key is refused
rather than ignored.
"""

from __future__ import annotations

import dataclasses
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from substrax.compute.spec import Accelerator, example_tasks, ExampleMode, JobSpec, Task
from substrax.examples import discover_examples, example_source
from substrax.notebooks.config import ExamplesConfig, read_examples_config
from substrax.records import dump_record, read_record, UNKNOWN_FIELDS_REFUSED
from substrax.typing import JsonValue


_TABLE = "[tool.substrax.compute]"
_ACCELERATOR = re.compile(r"(?P<kind>[^:]+)(?::(?P<count>\d+))?")
_TASK_SOURCE_KEYS = ("examples", "examples_mode", "commands", "command_timeout_seconds")


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskSources:
    """Where a configured job's tasks come from, and each task's limit.

    The job's other keys are :class:`~substrax.compute.JobSpec` fields. An example's run is held to
    the example's own budget, from the project's ``[tool.substrax.examples]``.

    Attributes:
        timeout_seconds: The job's budget, which is also a ``JobSpec`` field.
        examples: Directories, relative to the project, whose examples become tasks.
        examples_mode: What each example task does, required with ``examples``.
        commands: Further tasks, by name, run after the examples.
        command_timeout_seconds: Each task's limit; ``timeout_seconds`` when ``None``.
    """

    timeout_seconds: float
    examples: tuple[str, ...] = ()
    examples_mode: ExampleMode | None = None
    commands: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    command_timeout_seconds: float | None = None


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

    The job's table holds ``examples``, ``commands`` and ``command_timeout_seconds``, which make
    its tasks, and otherwise the fields of :class:`~substrax.compute.JobSpec` under their own names.
    An example task's budget is the example's, from the project's ``[tool.substrax.examples]``.

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
        LookupError: If the job lists examples and the project has no ``[tool.substrax.examples]``.
        ValueError: If a key of the job's table is unknown or a value of the wrong type, or the
            examples' settings are refused.
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
    limit = sources.command_timeout_seconds or sources.timeout_seconds
    tasks = _example_tasks(sources, name, project, limit=limit) + [
        Task(name=task, argv=argv, timeout_seconds=limit) for task, argv in sources.commands.items()
    ]
    spec = read_record(
        JobSpec, {**table, "name": name, "tasks": [dump_record(task) for task in tasks]}
    )
    if accelerator is None:
        return spec
    return dataclasses.replace(
        spec, resources=dataclasses.replace(spec.resources, accelerator=accelerator)
    )


def _example_tasks(  # noqa: DOC503  # LookupError and ValueError are raised by read_examples_config
    sources: TaskSources, job: str, project: Path, *, limit: float
) -> list[Task]:
    """The job's example tasks, each run as its notebook in the job's mode.

    Args:
        sources: Where the job's tasks come from.
        job: The job, for the message.
        project: The project root.
        limit: Each task's limit, outside the run's own budget.

    Returns:
        One task per example of each directory, in order.

    Raises:
        ValueError: If the job lists examples without ``examples_mode`` or the reverse, an
            example's budget is longer than ``limit`` (the task would be killed before its run
            reached it), or the examples' settings are refused.
        FileNotFoundError: If an examples directory does not exist; checked before the settings
            are read, which may name examples of a missing directory.
        LookupError: If the project has no pyproject.toml or no ``[tool.substrax.examples]``.
    """
    if bool(sources.examples) != (sources.examples_mode is not None):
        msg = (
            f"job {job!r} sets examples_mode without examples"
            if sources.examples_mode is not None
            else f"job {job!r} lists examples without examples_mode; set it to "
            f"{', '.join(repr(mode.value) for mode in ExampleMode)}"
        )
        raise ValueError(msg)
    if sources.examples_mode is None:
        return []
    for directory in sources.examples:
        if not (project / directory).is_dir():
            msg = f"examples directory {directory!r} of job {job!r} does not exist in {project}"
            raise FileNotFoundError(msg)
    examples = [
        path for directory in sources.examples for path in discover_examples(project / directory)
    ]
    _refuse_budgets_over(
        read_examples_config(project), examples, job=job, project=project, limit=limit
    )
    return list(
        example_tasks(examples, project=project, mode=sources.examples_mode, timeout_seconds=limit)
    )


def _refuse_budgets_over(
    settings: ExamplesConfig, examples: list[Path], *, job: str, project: Path, limit: float
) -> None:
    """Refuse an example whose own budget is longer than its task's limit.

    Args:
        settings: The project's example settings, which give each example's budget.
        examples: The job's examples.
        job: The job, for the message.
        project: The project root.
        limit: Each task's limit.

    Raises:
        ValueError: Naming each such example and its budget.
    """
    over = [
        f"{source} ({budget} s)"
        for source in (example_source(path, repo_root=project) for path in examples)
        if (budget := settings.budget_for(source)) > limit
    ]
    if over:
        msg = (
            f"job {job!r} gives each task {limit} s, less than the budget of "
            f"{', '.join(over)}; raise command_timeout_seconds or timeout_seconds"
        )
        raise ValueError(msg)


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


_LOCAL_SOURCES = frozenset({"editable", "directory", "path"})
"""The keys of a uv.lock ``source`` table that install a package from a local path (uv's lock
``Source``: ``Editable``, ``Directory``, ``Path``); ``registry``, ``git`` and ``url`` are remote."""


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
            local = _LOCAL_SOURCES.intersection(package.get("source", {}))
            return "editable" if local else package.get("version", "editable")
    msg = f"substrax is not in {lock}; add it to the project's dependencies to run the worker"
    raise LookupError(msg)
