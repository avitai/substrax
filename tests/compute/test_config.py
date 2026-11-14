"""A project's ``[tool.substrax.compute]`` table: its default backend, backend settings and jobs."""

from __future__ import annotations

import ast
import re
import subprocess
import textwrap
from pathlib import Path

import pytest

from substrax.compute import (
    Accelerator,
    Mount,
    MountAccess,
    Resources,
    Task,
)
from substrax.compute.config import (
    parse_accelerator,
    read_compute_config,
    require_locked_substrax,
    resolve_job,
)
from substrax.runtime import JaxRuntime


REPOSITORY = Path(__file__).resolve().parents[2]


def _documented_examples() -> list[tuple[str, str]]:
    """Each documented compute example with jobs, and the tracked file that documents it.

    Every tracked Markdown file is read, and every module docstring under ``src``, so a copy of the
    example added anywhere the repository documents it is read too.
    """
    listed = subprocess.run(  # nosec B603 B607 - git with fixed arguments
        ["git", "ls-files", "-z", "--", "*.md", "src/*.py"],  # noqa: S607
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    examples = []
    for relative in filter(None, listed.split("\0")):
        text = (REPOSITORY / relative).read_text(encoding="utf-8")
        if relative.endswith(".py"):
            text = ast.get_docstring(ast.parse(text)) or ""
        examples += [
            (relative, block)
            for block in re.findall(r"```toml\n(.*?)```", text, flags=re.DOTALL)
            if "[tool.substrax.compute.jobs." in block
        ]
    return examples


def test_each_documented_job_lists_its_accelerator_first_and_the_cpu_after(
    tmp_path: Path,
) -> None:
    """``cuda`` alone starts no CPU backend, and jax's host callbacks run on a CPU device.

    With no platform, a CUDA that cannot start leaves the job on the CPU in silence, and a compute
    job has no GPU check of its own; with ``cpu`` first, the CPU is the job's default backend. So a
    job given an accelerator lists the accelerator's platform first and ``cpu`` after it.
    """
    examples = _documented_examples()
    assert {"README.md", "docs/api/compute.md", "src/substrax/compute/config.py"} <= {
        where for where, _ in examples
    }
    for index, (where, block) in enumerate(examples):
        (tmp_path / str(index)).mkdir()
        project = _project(tmp_path / str(index), block)
        (project / "examples" / "metrics").mkdir(parents=True)
        (project / "examples" / "metrics" / "accuracy.py").write_text("", encoding="utf-8")
        settings = read_compute_config(project)
        for job in settings.jobs:
            spec = resolve_job(settings, job, project=project)
            if spec.resources.accelerator is None:
                continue
            platforms = spec.runtime.platforms or ()
            listed = f"{where}: job {job} has an accelerator and lists {platforms or 'no platform'}"
            assert platforms[:1] != ("cpu",), listed
            assert "cpu" in platforms[1:], listed


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
examples_mode = "test"
extras = ["cuda12"]
resources = { accelerator = { kind = "L4" }, cpu = 8, memory_mib = 32768 }
timeout_seconds = 1800
command_timeout_seconds = 1200
runtime = { platforms = ["cuda"], xla_flags = ["--xla_gpu_deterministic_ops=true"] }
env = { TF_CPP_MIN_LOG_LEVEL = "1" }
mounts = [{ name = "datasets", path = "/data", access = "read-only" }]

[tool.substrax.compute.jobs.examples.commands]
gpu-tests = ["python", "-m", "pytest", "tests/gpu"]

[tool.substrax.examples]
source_url = "https://github.com/avitai/demo/blob/main"
budget_seconds = 300
budgets = { "examples/02_second.py" = 900 }
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
    notebooks = ("python", "-m", "substrax.notebooks", "test")
    assert spec.tasks == (
        Task(
            name="examples.01_first",
            argv=(*notebooks, "examples/01_first.py"),
            timeout_seconds=1200,
        ),
        Task(
            name="examples.02_second",
            argv=(*notebooks, "examples/02_second.py"),
            timeout_seconds=1200,
        ),
        Task(
            name="gpu-tests",
            argv=("python", "-m", "pytest", "tests/gpu"),
            timeout_seconds=1200,
        ),
    )
    assert spec.resources == Resources(
        accelerator=Accelerator(kind="L4"), cpu=8.0, memory_mib=32768
    )
    assert spec.extras == ("cuda12",)
    assert spec.timeout_seconds == 1800
    assert spec.runtime == JaxRuntime(
        platforms=("cuda",), xla_flags=("--xla_gpu_deterministic_ops=true",)
    )
    assert spec.env == {"TF_CPP_MIN_LOG_LEVEL": "1"}
    assert spec.mounts == (Mount(name="datasets", path="/data", access=MountAccess.READ_ONLY),)


def test_a_command_without_its_own_budget_gets_the_jobs(tmp_path: Path) -> None:
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
    assert spec.resources == Resources()


def test_the_accelerator_can_be_overridden_per_run(tmp_path: Path) -> None:
    project = _project(tmp_path, FULL)
    (project / "examples").mkdir()
    for name in ("01_first.py", "02_second.py"):
        (project / "examples" / name).write_text("", encoding="utf-8")

    spec = resolve_job(
        read_compute_config(project),
        "examples",
        project=project,
        accelerator=Accelerator(kind="H100", count=2),
    )

    assert spec.resources.accelerator == Accelerator(kind="H100", count=2)
    assert (spec.resources.cpu, spec.resources.memory_mib) == (8.0, 32768)


def test_an_unknown_job_is_refused_with_the_configured_names(tmp_path: Path) -> None:
    project = _project(tmp_path, FULL)

    with pytest.raises(KeyError, match="'train'; configured: examples"):
        resolve_job(read_compute_config(project), "train", project=project)


def _examples_job(tmp_path: Path, job: str, names: tuple[str, ...] = ("01_first.py",)) -> Path:
    """A project with ``FULL``'s settings, its examples job's table replaced by ``job``."""
    head, _, rest = FULL.partition("[tool.substrax.compute.jobs.examples]\n")
    settings = rest[rest.index("[tool.substrax.examples]") :]
    project = _project(tmp_path, f"{head}[tool.substrax.compute.jobs.examples]\n{job}\n{settings}")
    (project / "examples").mkdir()
    for name in names:
        (project / "examples" / name).write_text("", encoding="utf-8")
    return project


def test_a_page_job_writes_each_page_into_the_task_s_outputs(tmp_path: Path) -> None:
    project = _examples_job(
        tmp_path,
        'examples = ["examples"]\nexamples_mode = "page"\ntimeout_seconds = 1200',
        names=("01_first.py", "02_second.py"),
    )

    tasks = resolve_job(read_compute_config(project), "examples", project=project).tasks

    assert [task.argv for task in tasks] == [
        ("python", "-m", "substrax.notebooks", "page", "--into-outputs", f"examples/{name}")
        for name in ("01_first.py", "02_second.py")
    ]
    assert [task.timeout_seconds for task in tasks] == [1200, 1200]


def test_a_job_listing_examples_names_what_their_tasks_do(tmp_path: Path) -> None:
    """Running each example against its page and making its page are different jobs."""
    project = _examples_job(tmp_path, 'examples = ["examples"]\ntimeout_seconds = 600')

    with pytest.raises(ValueError, match="examples_mode"):
        resolve_job(read_compute_config(project), "examples", project=project)


def test_an_examples_mode_without_examples_is_refused(tmp_path: Path) -> None:
    project = _examples_job(
        tmp_path,
        'examples_mode = "test"\ntimeout_seconds = 600\ncommands = { p = ["python", "-V"] }',
    )

    with pytest.raises(ValueError, match="examples_mode"):
        resolve_job(read_compute_config(project), "examples", project=project)


def test_a_task_limit_below_an_example_s_budget_is_refused(tmp_path: Path) -> None:
    """The run holds the example to its budget; a task killed sooner never reaches it."""
    project = _examples_job(
        tmp_path,
        'examples = ["examples"]\nexamples_mode = "test"\ntimeout_seconds = 600',
        names=("01_first.py", "02_second.py"),
    )

    with pytest.raises(ValueError, match=r"examples/02_second\.py.*900"):
        resolve_job(read_compute_config(project), "examples", project=project)


def test_a_job_listing_examples_needs_the_examples_settings(tmp_path: Path) -> None:
    """An example's budget is its own, from ``[tool.substrax.examples]``, held by its run."""
    project = _project(tmp_path, FULL.partition("[tool.substrax.examples]")[0])
    (project / "examples").mkdir()
    (project / "examples" / "01_first.py").write_text("", encoding="utf-8")

    with pytest.raises(LookupError, match=r"\[tool\.substrax\.examples\]"):
        resolve_job(read_compute_config(project), "examples", project=project)


def test_the_former_per_task_budget_key_is_refused(tmp_path: Path) -> None:
    """``task_timeout_seconds`` budgeted examples too; examples now take their own budgets."""
    project = _project(
        tmp_path,
        """
        [tool.substrax.compute.jobs.probe]
        timeout_seconds = 120
        task_timeout_seconds = 60
        commands = { probe = ["python", "-c", "print(1)"] }
        """,
    )

    with pytest.raises(ValueError, match="task_timeout_seconds"):
        resolve_job(read_compute_config(project), "probe", project=project)


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
        "[tool.substrax.compute.jobs.probe]\ntimeout_seconds = 60\n"
        "accelerators = { kind = 'L4' }\n",
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


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ('{ editable = "." }', "editable"),
        ('{ directory = "../substrax" }', "editable"),
        ('{ path = "../substrax" }', "editable"),
        ('{ registry = "https://pypi.org/simple" }', "0.1.18"),
        ('{ git = "https://github.com/avitai/substrax?rev=main#50055ae" }', "0.1.18"),
    ],
)
def test_a_locked_substrax_is_named_by_its_source(
    tmp_path: Path, source: str, expected: str
) -> None:
    """A local source counts as editable even when the lock records its version."""
    project = _project(tmp_path, "")
    (project / "uv.lock").write_text(
        f'[[package]]\nname = "substrax"\nversion = "0.1.18"\nsource = {source}\n', "utf-8"
    )

    assert require_locked_substrax(project) == expected


def test_substrax_itself_counts_as_locking_substrax() -> None:
    root = Path(__file__).resolve().parents[2]

    assert require_locked_substrax(root) == "editable"
