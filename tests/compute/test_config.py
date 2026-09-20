"""A project's ``[tool.substrax.compute]`` table: its default backend, backend settings and jobs."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from substrax.compute import Accelerator, Mount, MountAccess, Task
from substrax.compute.config import (
    parse_accelerator,
    read_compute_config,
    require_locked_substrax,
    resolve_job,
)
from substrax.runtime import JaxRuntime


def _project(tmp_path: Path, table: str) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        '[project]\nname = "demo"\n\n' + textwrap.dedent(table), encoding="utf-8"
    )
    return root


FULL = """
[tool.substrax.compute]
backend = "modal"

[tool.substrax.compute.backends.modal]
outputs_volume = "demo-outputs"

[tool.substrax.compute.jobs.examples]
examples = ["examples"]
extras = ["cuda12"]
accelerator = { kind = "L4" }
timeout_seconds = 1800
task_timeout_seconds = 300
runtime = { platforms = ["cuda"], xla_flags = ["--xla_gpu_deterministic_ops=true"] }
env = { TF_CPP_MIN_LOG_LEVEL = "1" }
mounts = [{ name = "datasets", path = "/data", access = "read-only" }]

[tool.substrax.compute.jobs.examples.commands]
gpu-tests = ["python", "-m", "pytest", "tests/gpu"]
"""


def test_a_job_resolves_to_a_spec_with_examples_first_then_commands(tmp_path: Path) -> None:
    project = _project(tmp_path, FULL)
    for name in ("02_second.py", "01_first.py", "_helpers.py"):
        (project / "examples").mkdir(exist_ok=True)
        (project / "examples" / name).write_text("", encoding="utf-8")

    config = read_compute_config(project)
    spec = resolve_job(config, "examples", project=project)

    assert config.backend == "modal"
    assert config.backends == {"modal": {"outputs_volume": "demo-outputs"}}
    assert spec.name == "examples"
    assert spec.tasks == (
        Task(
            name="examples.01_first", argv=("python", "examples/01_first.py"), timeout_seconds=300
        ),
        Task(
            name="examples.02_second", argv=("python", "examples/02_second.py"), timeout_seconds=300
        ),
        Task(
            name="gpu-tests",
            argv=("python", "-m", "pytest", "tests/gpu"),
            timeout_seconds=300,
        ),
    )
    assert spec.accelerator == Accelerator(kind="L4")
    assert spec.extras == ("cuda12",)
    assert spec.timeout_seconds == 1800
    assert spec.runtime == JaxRuntime(
        platforms=("cuda",), xla_flags=("--xla_gpu_deterministic_ops=true",)
    )
    assert spec.env == {"TF_CPP_MIN_LOG_LEVEL": "1"}
    assert spec.mounts == (Mount(name="datasets", path="/data", access=MountAccess.READ_ONLY),)


def test_a_task_without_its_own_budget_gets_the_jobs(tmp_path: Path) -> None:
    project = _project(
        tmp_path,
        """
        [tool.substrax.compute.jobs.probe]
        timeout_seconds = 120
        commands = { probe = ["python", "-c", "import jax; print(jax.devices())"] }
        """,
    )

    spec = resolve_job(read_compute_config(project), "probe", project=project)

    assert spec.tasks[0].timeout_seconds == 120
    assert spec.accelerator is None


def test_the_accelerator_can_be_overridden_per_run(tmp_path: Path) -> None:
    project = _project(tmp_path, FULL)
    (project / "examples").mkdir()
    (project / "examples" / "one.py").write_text("", encoding="utf-8")

    spec = resolve_job(
        read_compute_config(project),
        "examples",
        project=project,
        accelerator=Accelerator(kind="H100", count=2),
    )

    assert spec.accelerator == Accelerator(kind="H100", count=2)


def test_an_unknown_job_is_refused_with_the_configured_names(tmp_path: Path) -> None:
    project = _project(tmp_path, FULL)

    with pytest.raises(KeyError, match="'train'; configured: examples"):
        resolve_job(read_compute_config(project), "train", project=project)


def test_a_job_with_no_tasks_is_refused(tmp_path: Path) -> None:
    project = _project(tmp_path, "[tool.substrax.compute.jobs.empty]\ntimeout_seconds = 60\n")

    with pytest.raises(ValueError, match="at least one task"):
        resolve_job(read_compute_config(project), "empty", project=project)


def test_an_examples_directory_that_does_not_exist_is_refused(tmp_path: Path) -> None:
    project = _project(tmp_path, FULL)

    with pytest.raises(FileNotFoundError, match="examples"):
        resolve_job(read_compute_config(project), "examples", project=project)


def test_a_misspelled_job_field_is_refused_when_the_job_is_resolved(tmp_path: Path) -> None:
    project = _project(
        tmp_path,
        "[tool.substrax.compute.jobs.probe]\ntimeout_seconds = 60\naccelerators = { kind = 'L4' }\n",
    )

    config = read_compute_config(project)

    with pytest.raises(ValueError, match="accelerators"):
        resolve_job(config, "probe", project=project)


def test_a_job_without_a_budget_is_refused(tmp_path: Path) -> None:
    project = _project(
        tmp_path, "[tool.substrax.compute.jobs.probe]\ncommands = { p = ['python', '-V'] }\n"
    )

    with pytest.raises(ValueError, match="timeout_seconds"):
        resolve_job(read_compute_config(project), "probe", project=project)


def test_a_misspelled_top_level_key_is_refused_on_reading(tmp_path: Path) -> None:
    project = _project(tmp_path, "[tool.substrax.compute]\nbackends_default = 'modal'\n")

    with pytest.raises(ValueError, match="backends_default"):
        read_compute_config(project)


def test_a_project_without_the_table_is_refused_naming_it(tmp_path: Path) -> None:
    project = _project(tmp_path, "")

    with pytest.raises(LookupError, match=r"\[tool.substrax.compute\]"):
        read_compute_config(project)


@pytest.mark.parametrize(
    ("text", "expected"),
    [("L4", Accelerator(kind="L4")), ("H100:8", Accelerator(kind="H100", count=8))],
)
def test_an_accelerator_is_written_kind_colon_count(text: str, expected: Accelerator) -> None:
    assert parse_accelerator(text) == expected


@pytest.mark.parametrize("text", ["", "L4:", "L4:two", ":2"])
def test_a_malformed_accelerator_is_refused(text: str) -> None:
    with pytest.raises(ValueError, match="accelerator"):
        parse_accelerator(text)


def test_the_project_must_lock_substrax_for_the_worker_to_run(tmp_path: Path) -> None:
    project = _project(tmp_path, "")
    (project / "uv.lock").write_text('[[package]]\nname = "jax"\nversion = "0.11.1"\n', "utf-8")

    with pytest.raises(LookupError, match=r"substrax.*uv\.lock"):
        require_locked_substrax(project)

    (project / "uv.lock").write_text(
        '[[package]]\nname = "substrax"\nversion = "0.1.15"\n', "utf-8"
    )
    assert require_locked_substrax(project) == "0.1.15"


def test_a_project_without_a_lock_is_refused(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"uv\.lock"):
        require_locked_substrax(_project(tmp_path, ""))


def test_substrax_itself_counts_as_locking_substrax() -> None:
    root = Path(__file__).resolve().parents[2]

    assert require_locked_substrax(root) == "editable"
