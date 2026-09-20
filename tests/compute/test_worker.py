"""The worker runs a job's tasks the same way on every provider: logs, outputs and a manifest.

Each task here is a few lines of Python without jax, so a child starts in well under a second.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

from substrax.compute import JobSpec, Task
from substrax.compute.worker import (
    JobManifest,
    main,
    MANIFEST_NAME,
    read_manifest,
    run_job,
    STDERR_LOG,
    STDOUT_LOG,
    TaskStatus,
)
from substrax.records import dump_record
from substrax.runtime import JaxRuntime


_BUDGET = 60.0


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A project root holding a ``scripts`` directory."""
    (tmp_path / "project" / "scripts").mkdir(parents=True)
    return tmp_path / "project"


@pytest.fixture
def outputs(tmp_path: Path) -> Path:
    """The directory a run's outputs go to."""
    return tmp_path / "outputs"


def _script(project: Path, name: str, body: str) -> Task:
    path = project / "scripts" / f"{name}.py"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return Task(name=name, argv=("python", f"scripts/{name}.py"), timeout_seconds=_BUDGET)


def _job(*tasks: Task, runtime: JaxRuntime | None = None) -> JobSpec:
    return JobSpec(name="job", tasks=tasks, timeout_seconds=600.0, runtime=runtime or JaxRuntime())


def test_a_task_logs_its_streams_and_its_outputs_land_in_its_directory(
    project: Path, outputs: Path
) -> None:
    task = _script(
        project,
        "figure",
        """
        import os, sys
        from pathlib import Path
        target = Path(os.environ["AVITAI_OUTPUT_DIR"]) / "figure"
        target.mkdir(parents=True)
        (target / "plot.png").write_bytes(b"png")
        print("loss 0.25")
        print("careful", file=sys.stderr)
        """,
    )

    manifest = run_job(_job(task), project=project, outputs=outputs, echo=False)

    (outcome,) = manifest.tasks
    assert outcome.status is TaskStatus.SUCCEEDED
    assert outcome.returncode == 0
    assert outcome.outputs == ("figure/plot.png",)
    assert (outputs / "figure" / STDOUT_LOG).read_text(encoding="utf-8") == "loss 0.25\n"
    assert (outputs / "figure" / STDERR_LOG).read_text(encoding="utf-8") == "careful\n"
    assert manifest.succeeded


def test_the_manifest_on_disk_is_the_one_returned(project: Path, outputs: Path) -> None:
    manifest = run_job(
        _job(_script(project, "ok", "pass")), project=project, outputs=outputs, echo=False
    )

    on_disk = read_manifest(json.loads((outputs / MANIFEST_NAME).read_text(encoding="utf-8")))

    assert on_disk == manifest
    assert on_disk.job == "job"
    assert on_disk.finished


def test_the_manifest_written_between_tasks_is_not_finished(project: Path, outputs: Path) -> None:
    peek = _script(
        project,
        "peek",
        """
        import json, os
        from pathlib import Path
        manifest = Path(os.environ["AVITAI_OUTPUT_DIR"]).parent / "manifest.json"
        print(json.loads(manifest.read_text())["finished"])
        """,
    )

    run_job(
        _job(_script(project, "first", "pass"), peek), project=project, outputs=outputs, echo=False
    )

    assert (outputs / "peek" / STDOUT_LOG).read_text(encoding="utf-8") == "False\n"


def test_a_failed_task_does_not_stop_the_next(project: Path, outputs: Path) -> None:
    failing = _script(project, "failing", "raise SystemExit(3)")
    after = _script(project, "after", "print('ran')")

    manifest = run_job(_job(failing, after), project=project, outputs=outputs, echo=False)

    assert [(o.name, o.status, o.returncode) for o in manifest.tasks] == [
        ("failing", TaskStatus.FAILED, 3),
        ("after", TaskStatus.SUCCEEDED, 0),
    ]
    assert not manifest.succeeded


def test_a_task_past_its_timeout_is_killed_and_recorded(project: Path, outputs: Path) -> None:
    slow = _script(project, "slow", "import time\nprint('started', flush=True)\ntime.sleep(60)")
    slow = Task(name=slow.name, argv=slow.argv, timeout_seconds=1.0)

    manifest = run_job(_job(slow), project=project, outputs=outputs, echo=False)

    (outcome,) = manifest.tasks
    assert outcome.status is TaskStatus.TIMED_OUT
    assert outcome.seconds < 30
    assert (outputs / "slow" / STDOUT_LOG).read_text(encoding="utf-8") == "started\n"


def test_the_jobs_budget_caps_every_task_and_the_tasks_it_leaves_no_time_for(
    project: Path, outputs: Path
) -> None:
    slow = _script(project, "slow", "import time\ntime.sleep(60)")
    never = _script(project, "never", "print('ran')")
    job = JobSpec(name="job", tasks=(slow, never), timeout_seconds=1.0)

    manifest = run_job(job, project=project, outputs=outputs, echo=False)

    assert [(o.name, o.status) for o in manifest.tasks] == [
        ("slow", TaskStatus.TIMED_OUT),
        ("never", TaskStatus.TIMED_OUT),
    ]
    assert manifest.tasks[0].seconds < 30
    assert manifest.tasks[1].returncode is None
    assert not (outputs / "never" / STDOUT_LOG).read_text(encoding="utf-8")
    assert manifest.finished


def test_a_command_that_cannot_start_is_a_failed_task_with_the_reason_logged(
    project: Path, outputs: Path
) -> None:
    missing = Task(name="missing", argv=("no-such-command-here",), timeout_seconds=_BUDGET)

    manifest = run_job(_job(missing), project=project, outputs=outputs, echo=False)

    (outcome,) = manifest.tasks
    assert outcome.status is TaskStatus.FAILED
    assert outcome.returncode is None
    assert "no-such-command-here" in (outputs / "missing" / STDERR_LOG).read_text(encoding="utf-8")


def test_python_is_the_given_interpreter_and_the_runtime_configures_the_task(
    project: Path, outputs: Path
) -> None:
    probe = _script(
        project,
        "probe",
        """
        import os, sys
        print(sys.executable)
        print(os.environ["JAX_PLATFORMS"], os.environ["XLA_FLAGS"])
        print(os.getcwd())
        """,
    )
    runtime = JaxRuntime(platforms=("cpu",), xla_flags=("--xla_cpu_enable_fast_math=false",))

    run_job(_job(probe, runtime=runtime), project=project, outputs=outputs, echo=False)

    lines = (outputs / "probe" / STDOUT_LOG).read_text(encoding="utf-8").splitlines()
    assert Path(lines[0]).resolve() == Path(sys.executable).resolve()
    assert lines[1] == "cpu --xla_cpu_enable_fast_math=false"
    assert Path(lines[2]).resolve() == project.resolve()


def test_echo_copies_each_line_to_the_workers_streams_with_the_task_name(
    project: Path, outputs: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    task = _script(project, "talk", "import sys\nprint('out')\nprint('err', file=sys.stderr)")

    run_job(_job(task), project=project, outputs=outputs, echo=True)

    captured = capfd.readouterr()
    assert "[talk] out\n" in captured.out
    assert "[talk] err\n" in captured.err


def test_main_runs_a_spec_file_and_its_exit_code_says_whether_every_task_succeeded(
    project: Path, outputs: Path, tmp_path: Path
) -> None:
    spec_path = tmp_path / "spec.json"
    ok = _job(_script(project, "ok", "pass"))
    spec_path.write_text(json.dumps(dump_record(ok)), encoding="utf-8")
    failing = _job(_script(project, "bad", "raise SystemExit(1)"))
    failing_path = tmp_path / "failing.json"
    failing_path.write_text(json.dumps(dump_record(failing)), encoding="utf-8")

    succeeded = main(
        ["--spec", str(spec_path), "--project", str(project), "--outputs", str(outputs / "a")]
    )
    failed = main(
        ["--spec", str(failing_path), "--project", str(project), "--outputs", str(outputs / "b")]
    )

    assert (succeeded, failed) == (0, 1)
    assert isinstance(
        read_manifest(json.loads((outputs / "a" / MANIFEST_NAME).read_text(encoding="utf-8"))),
        JobManifest,
    )
