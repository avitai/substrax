"""The job spec: what a remote run executes, as a record every backend reads the same way."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from substrax.compute import (
    Accelerator,
    example_tasks,
    JOB_SPEC_VERSION,
    JobSpec,
    Mount,
    MountAccess,
    read_job_spec,
    Task,
)
from substrax.records import dump_record
from substrax.runtime import JaxRuntime


def _task(name: str = "first", *argv: str) -> Task:
    return Task(name=name, argv=argv or ("python", "-c", "pass"), timeout_seconds=60.0)


def _spec(**changes: object) -> JobSpec:
    base = JobSpec(name="examples", tasks=(_task(),), timeout_seconds=600.0)
    return dataclasses.replace(base, **changes)


class TestJobSpec:
    def test_a_full_spec_round_trips_through_json(self) -> None:
        spec = _spec(
            extras=("cuda12",),
            accelerator=Accelerator(kind="L4", count=2),
            runtime=JaxRuntime(
                platforms=("cuda",), xla_flags=("--xla_gpu_deterministic_ops=true",)
            ),
            env={"TF_CPP_MIN_LOG_LEVEL": "1"},
            mounts=(Mount(name="datasets", path="/data", access=MountAccess.READ_ONLY),),
            tasks=(_task("first"), _task("second", "python", "run.py", "--fast")),
        )

        text = json.dumps(dump_record(spec))

        assert read_job_spec(json.loads(text)) == spec

    def test_the_defaults_ask_for_no_accelerator_extras_mounts_or_settings(self) -> None:
        spec = _spec()

        assert spec.schema_version == JOB_SPEC_VERSION
        assert spec.accelerator is None
        assert spec.extras == ()
        assert spec.mounts == ()
        assert spec.env == {}
        assert spec.runtime == JaxRuntime()

    @pytest.mark.parametrize("name", ["", "has space", "slash/name", "-leading", ".."])
    def test_a_name_that_cannot_be_a_directory_is_refused(self, name: str) -> None:
        with pytest.raises(ValueError, match="name"):
            _spec(name=name)

    def test_a_spec_without_tasks_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one task"):
            _spec(tasks=())

    def test_two_tasks_with_one_name_are_refused(self) -> None:
        with pytest.raises(ValueError, match="'first'"):
            _spec(tasks=(_task("first"), _task("first")))

    @pytest.mark.parametrize("seconds", [0.0, -1.0, float("inf"), float("nan")])
    def test_a_timeout_that_is_not_positive_and_finite_is_refused(self, seconds: float) -> None:
        with pytest.raises(ValueError, match="timeout"):
            _spec(timeout_seconds=seconds)

    def test_two_mounts_with_one_name_or_one_path_are_refused(self) -> None:
        data = Mount(name="data", path="/data", access=MountAccess.READ_ONLY)

        with pytest.raises(ValueError, match="'data'"):
            _spec(mounts=(data, Mount(name="data", path="/other", access=MountAccess.READ_ONLY)))
        with pytest.raises(ValueError, match="'/data'"):
            _spec(mounts=(data, Mount(name="more", path="/data", access=MountAccess.READ_WRITE)))

    def test_a_newer_spec_version_is_refused_naming_both_versions(self) -> None:
        payload = dump_record(_spec())
        payload["schema_version"] = JOB_SPEC_VERSION + 1

        with pytest.raises(ValueError, match=rf"{JOB_SPEC_VERSION + 1}.*{JOB_SPEC_VERSION}"):
            read_job_spec(payload)

    def test_reading_json_applies_the_same_validation_as_construction(self) -> None:
        payload = dump_record(_spec(tasks=(_task("first"), _task("second"))))
        tasks = payload["tasks"]
        assert isinstance(tasks, list)
        second = tasks[1]
        assert isinstance(second, dict)
        second["name"] = "first"

        with pytest.raises(ValueError, match="'first'"):
            read_job_spec(payload)

    def test_a_field_of_the_wrong_json_type_is_refused(self) -> None:
        payload = dump_record(_spec())
        payload["timeout_seconds"] = "600"

        with pytest.raises(ValueError, match="timeout_seconds"):
            read_job_spec(payload)


class TestTask:
    def test_a_task_without_a_command_is_refused(self) -> None:
        with pytest.raises(ValueError, match="argv"):
            Task(name="empty", argv=(), timeout_seconds=1.0)

    def test_a_task_timeout_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="timeout"):
            Task(name="t", argv=("true",), timeout_seconds=0.0)


class TestMount:
    @pytest.mark.parametrize("path", ["data", "./data", "/data/../etc", "/"])
    def test_a_path_that_is_not_an_absolute_directory_below_the_root_is_refused(
        self, path: str
    ) -> None:
        with pytest.raises(ValueError, match="path"):
            Mount(name="data", path=path, access=MountAccess.READ_ONLY)


class TestAccelerator:
    def test_a_count_below_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="count"):
            Accelerator(kind="L4", count=0)

    @pytest.mark.parametrize("kind", ["", "L4:2", "L 4"])
    def test_a_kind_that_is_not_one_word_is_refused(self, kind: str) -> None:
        with pytest.raises(ValueError, match="kind"):
            Accelerator(kind=kind)


class TestExampleTasks:
    def test_each_example_becomes_a_python_task_named_by_its_path(self, tmp_path: Path) -> None:
        project = tmp_path / "repo"
        paths = [project / "examples" / "metrics" / "01_basic.py", project / "examples" / "top.py"]

        tasks = example_tasks(paths, project=project, timeout_seconds=300.0)

        assert tasks == (
            Task(
                name="examples.metrics.01_basic",
                argv=("python", "examples/metrics/01_basic.py"),
                timeout_seconds=300.0,
            ),
            Task(name="examples.top", argv=("python", "examples/top.py"), timeout_seconds=300.0),
        )

    def test_an_example_outside_the_project_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="outside"):
            example_tasks([tmp_path / "x.py"], project=tmp_path / "repo", timeout_seconds=1.0)
