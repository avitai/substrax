"""The ``skypilot`` backend, against a local stand-in for SkyPilot's SDK and ``gcloud``.

``fake_sky/sky.py`` is imported as SkyPilot's ``sky`` module by the SDK bridge; it runs each
task's ``run`` script on this machine with its mounts rewritten to local directories, and
``fake_sky/bin/gcloud`` downloads from those directories, so the contract runs here without a
cloud account.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from substrax.compute import (
    Accelerator,
    JobSpec,
    Mount,
    MountAccess,
    Resources,
    skypilot_backend,
    skypilot_transports,
    Task,
)
from substrax.compute.backend import (
    backend_state_dir,
    ComputeBackend,
    new_run,
    RunHandle,
    RunState,
)
from substrax.compute.skypilot_backend import SkyPilotBackend
from substrax.testing.compute import BackendContract
from substrax.typing import JsonValue


FAKE = Path(__file__).with_name("fake_sky")


@pytest.fixture(autouse=True)
def fake_sky(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the fake SDK and ``gcloud`` at a directory of the test's own."""
    root = tmp_path / "fake-sky"
    monkeypatch.setenv("FAKE_SKY_ROOT", str(root))
    monkeypatch.setenv("PYTHONPATH", str(FAKE) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    monkeypatch.setenv("PATH", f"{FAKE / 'bin'}{os.pathsep}{os.environ['PATH']}")
    return root


def _backend(state_dir: Path, **settings: JsonValue) -> SkyPilotBackend:
    """A backend on the fake SDK."""
    return SkyPilotBackend(
        {
            "outputs_bucket": "gs://outputs",
            "sky": shlex.join([sys.executable, str(FAKE / "sky.py")]),
            "worker_python": sys.executable,
            "sky_python": sys.executable,
            "state_dir": str(state_dir),
            **settings,
        }
    )


def _runs(state_dir: Path) -> Path:
    return backend_state_dir(str(state_dir), "skypilot").resolve()


class TestSkyPilotBackend(BackendContract):
    def make_backend(self, state_dir: Path) -> ComputeBackend:
        return _backend(state_dir)


def _task_file(state_dir: Path) -> dict[str, Any]:
    (path,) = _runs(state_dir).glob("*/task.yaml")
    return json.loads(path.read_text(encoding="utf-8"))


def _spec(**changes: object) -> JobSpec:
    base = JobSpec(
        name="examples",
        tasks=(Task(name="t", argv=("python", "-c", "pass"), timeout_seconds=60.0),),
        timeout_seconds=600.0,
        extras=("cuda12", "docs"),
    )
    return dataclasses.replace(base, **changes)


def _handle(tmp_path: Path) -> RunHandle:
    return new_run(_spec(), tmp_path, settings={}).handle(
        "skypilot", {"cluster": "owned", "job_id": "1"}
    )


def test_the_task_syncs_the_lock_with_the_extras_and_runs_the_worker(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()

    handle = _backend(tmp_path / "state", infra="gcp/us-central1").submit(
        _spec(resources=Resources(accelerator=Accelerator(kind="L4"))), project=project
    )

    task = _task_file(tmp_path / "state")
    assert task["workdir"] == str(project.resolve())
    assert task["resources"] == {
        "use_spot": False,
        "accelerators": "L4:1",
        "infra": "gcp/us-central1",
    }
    assert "uv sync --frozen --extra cuda12 --extra docs" in task["setup"]
    assert "-m substrax.compute.worker" in task["run"]
    assert f"--outputs /outputs/{handle.run_id}" in task["run"]
    assert "timeout" not in task["run"]
    assert task["file_mounts"]["/outputs"] == {"source": "gs://outputs", "mode": "MOUNT"}


def test_the_resources_ask_for_at_least_the_cpu_and_memory_requested(tmp_path: Path) -> None:
    """SkyPilot reads ``<n>+`` as at least ``n`` vCPUs and ``<n>MB+`` as at least ``n`` MiB."""
    (tmp_path / "project").mkdir()
    _backend(tmp_path / "state").submit(
        _spec(resources=Resources(cpu=8.0, memory_mib=65536)), project=tmp_path / "project"
    )

    resources = _task_file(tmp_path / "state")["resources"]
    assert (resources["cpus"], resources["memory"]) == ("8+", "65536MB+")


def test_no_cpu_or_memory_is_asked_for_when_the_job_requests_none(tmp_path: Path) -> None:
    (tmp_path / "project").mkdir()
    _backend(tmp_path / "state").submit(_spec(), project=tmp_path / "project")

    resources = _task_file(tmp_path / "state")["resources"]
    assert "cpus" not in resources
    assert "memory" not in resources


def test_each_mount_is_its_configured_bucket(tmp_path: Path) -> None:
    (tmp_path / "project").mkdir()
    mount = Mount(name="datasets", path="/data", access=MountAccess.READ_ONLY)
    _backend(tmp_path / "state", buckets={"datasets": "gs://demo-data"}).submit(
        _spec(mounts=(mount,)), project=tmp_path / "project"
    )

    mounts = _task_file(tmp_path / "state")["file_mounts"]
    assert mounts["/data"] == {"source": "gs://demo-data", "mode": "MOUNT"}


def test_a_mount_without_a_configured_bucket_is_refused(tmp_path: Path) -> None:
    (tmp_path / "project").mkdir()
    mount = Mount(name="datasets", path="/data", access=MountAccess.READ_ONLY)

    with pytest.raises(LookupError, match="'datasets'"):
        _backend(tmp_path / "state").submit(_spec(mounts=(mount,)), project=tmp_path / "project")


def test_an_outputs_bucket_without_a_download_tool_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="gs:// or s3://"):
        _backend(tmp_path / "state", outputs_bucket="azure://outputs")


def test_cancel_attempts_teardown_when_status_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unavailable status endpoint must not bypass cleanup."""
    backend = _backend(tmp_path / "state")
    calls: list[tuple[str, ...]] = []

    def status(handle: RunHandle) -> RunState:
        del handle
        raise RuntimeError("queue unavailable")

    monkeypatch.setattr(backend, "status", status)
    monkeypatch.setattr(backend, "_sdk", lambda *args: calls.append(args))
    with pytest.raises(ExceptionGroup, match="cleanup"):
        backend.cancel(_handle(tmp_path))
    assert ("down", "owned") in calls


def test_cancel_failure_still_attempts_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancellation and instance teardown are separate operations."""
    backend = _backend(tmp_path / "state")
    handle = _handle(tmp_path)
    (_runs(tmp_path / "state") / handle.run_id).mkdir(parents=True)
    calls: list[tuple[str, ...]] = []

    def sdk(*args: str) -> str:
        calls.append(args)
        if args[0] == "cancel-job":
            raise RuntimeError("cancel failed")
        return ""

    monkeypatch.setattr(backend, "status", lambda _: RunState.RUNNING)
    monkeypatch.setattr(backend, "_sdk", sdk)
    with pytest.raises(ExceptionGroup, match="cleanup"):
        backend.cancel(handle)
    assert ("down", "owned") in calls
    assert not (_runs(tmp_path / "state") / handle.run_id / "cancelled").exists()


def test_cancel_of_a_finished_run_leaves_its_cluster_and_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The published cancel contract is a no-op for terminal runs."""
    backend = _backend(tmp_path / "state")
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(backend, "status", lambda _: RunState.SUCCEEDED)
    monkeypatch.setattr(backend, "_sdk", lambda *args: calls.append(args))
    backend.cancel(_handle(tmp_path))
    assert not calls


def test_unknown_queue_state_is_not_a_scientific_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A provider schema change must remain an infrastructure error."""
    backend = _backend(tmp_path / "state")
    reply = '[{"job_id":"1","status":"NEW_STATE"}]'
    monkeypatch.setattr(skypilot_transports, "run_tool", lambda _: reply)
    with pytest.raises(ValueError, match="Unknown SkyPilot"):
        backend.status(_handle(tmp_path))


def test_a_bucket_run_whose_queue_cannot_be_read_is_an_error_not_a_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cluster may still be running the job; the queue's own error is kept."""

    def unreadable(command: list[str]) -> str:
        del command
        raise RuntimeError("API server unreachable")

    monkeypatch.setattr(skypilot_transports, "run_tool", unreadable)
    with pytest.raises(RuntimeError, match="API server unreachable"):
        _backend(tmp_path / "state").status(_handle(tmp_path))


@pytest.mark.parametrize("provider_status", ["FAILED", "FAILED_SETUP", "FAILED_DRIVER"])
def test_native_failure_statuses_are_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider_status: str
) -> None:
    """Known SDK failures remain failures; future unknown states remain errors."""
    reply = json.dumps([{"job_id": 1, "status": provider_status}])
    monkeypatch.setattr(skypilot_transports, "run_tool", lambda _: reply)
    assert _backend(tmp_path / "state").status(_handle(tmp_path)) is RunState.FAILED


def test_fake_launch_receipt_contains_a_request_not_its_result(tmp_path: Path) -> None:
    """The provider double must model asynchronous request identities."""
    handle = _backend(tmp_path / "state").submit(_spec(), project=tmp_path)
    assert handle.details["request_id"].startswith("request-")


def test_sdk_failure_diagnostic_has_no_embedded_traceback() -> None:
    """A bridge failure must not reintroduce a traceback inside a grouped CLI error."""
    result = subprocess.run(  # noqa: S603 - isolated fake SDK and fixed request
        [
            sys.executable,
            str(Path(skypilot_backend.__file__).with_name("sky_sdk.py")),
            "cancel-request",
            "unknown-request",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "AssertionError" in result.stderr
    assert "Traceback" not in result.stderr
