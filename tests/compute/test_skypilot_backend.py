"""The ``skypilot`` backend, against a local stand-in for the ``sky`` and ``gcloud`` commands.

``fake_sky/sky.py`` runs each task's ``run`` script on this machine with its mounts rewritten to
local directories, and ``fake_sky/bin/gcloud`` downloads from those directories, so the contract
runs here without a cloud account.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shlex
import sys
from pathlib import Path

import pytest

from substrax.compute import Accelerator, JobSpec, Mount, MountAccess, Task
from substrax.compute.backend import ComputeBackend
from substrax.compute.skypilot_backend import SkyPilotBackend
from substrax.testing.compute import BackendContract
from substrax.typing import JsonValue


FAKE = Path(__file__).with_name("fake_sky")


@pytest.fixture(autouse=True)
def fake_sky(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the fake ``sky`` and ``gcloud`` at a directory of the test's own."""
    root = tmp_path / "fake-sky"
    monkeypatch.setenv("FAKE_SKY_ROOT", str(root))
    monkeypatch.setenv("PATH", f"{FAKE / 'bin'}{os.pathsep}{os.environ['PATH']}")
    return root


def _backend(state_dir: Path, **settings: JsonValue) -> SkyPilotBackend:
    return SkyPilotBackend(
        {
            "outputs_bucket": "gs://outputs",
            "sky": shlex.join([sys.executable, str(FAKE / "sky.py")]),
            "worker_python": sys.executable,
            "state_dir": str(state_dir),
            **settings,
        }
    )


class TestSkyPilotBackend(BackendContract):
    def make_backend(self, state_dir: Path) -> ComputeBackend:
        return _backend(state_dir)


def _task_file(state_dir: Path) -> dict[str, object]:
    (path,) = (state_dir / "compute" / "skypilot").glob("*/task.yaml")
    return json.loads(path.read_text(encoding="utf-8"))


def _spec(**changes: object) -> JobSpec:
    base = JobSpec(
        name="examples",
        tasks=(Task(name="t", argv=("python", "-c", "pass"), timeout_seconds=60.0),),
        timeout_seconds=600.0,
        extras=("cuda12", "docs"),
    )
    return dataclasses.replace(base, **changes)


def test_the_task_syncs_the_lock_with_the_extras_and_runs_the_worker(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()

    handle = _backend(tmp_path / "state", infra="gcp/us-central1").submit(
        _spec(accelerator=Accelerator(kind="L4")), project=project
    )

    task = _task_file(tmp_path / "state")
    assert task["workdir"] == str(project.resolve())
    assert task["resources"] == {
        "use_spot": False,
        "accelerators": "L4:1",
        "infra": "gcp/us-central1",
    }
    assert "uv sync --frozen --extra cuda12 --extra docs" in str(task["setup"])
    run = str(task["run"])
    assert "-m substrax.compute.worker" in run
    assert f"--outputs /outputs/{handle.run_id}" in run
    assert "timeout" not in run
    mounts = task["file_mounts"]
    assert isinstance(mounts, dict)
    assert mounts["/outputs"] == {"source": "gs://outputs", "mode": "MOUNT"}


def test_each_mount_is_its_configured_bucket(tmp_path: Path) -> None:
    (tmp_path / "project").mkdir()
    mount = Mount(name="datasets", path="/data", access=MountAccess.READ_ONLY)
    backend = _backend(tmp_path / "state", buckets={"datasets": "gs://demo-data"})

    backend.submit(_spec(mounts=(mount,)), project=tmp_path / "project")

    mounts = _task_file(tmp_path / "state")["file_mounts"]
    assert isinstance(mounts, dict)
    assert mounts["/data"] == {"source": "gs://demo-data", "mode": "MOUNT"}


def test_a_mount_without_a_configured_bucket_is_refused(tmp_path: Path) -> None:
    (tmp_path / "project").mkdir()
    mount = Mount(name="datasets", path="/data", access=MountAccess.READ_ONLY)

    with pytest.raises(LookupError, match="'datasets'"):
        _backend(tmp_path / "state").submit(_spec(mounts=(mount,)), project=tmp_path / "project")


def test_an_outputs_bucket_without_a_download_tool_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="gs:// or s3://"):
        _backend(tmp_path / "state", outputs_bucket="azure://outputs")
