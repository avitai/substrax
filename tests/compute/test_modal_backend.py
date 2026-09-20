"""The ``modal`` backend, against a local stand-in for Modal: the contract, the image, the requests.

``fake_modal/modal`` replaces the Modal client for this module and for the child interpreters its
functions run in, so the whole contract runs here without an account; a real run is recorded in
the release evidence.
"""

from __future__ import annotations

import dataclasses
import importlib
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

from substrax.compute import Accelerator, JobSpec, Mount, MountAccess, Task
from substrax.compute.backend import ComputeBackend
from substrax.testing.compute import BackendContract
from substrax.typing import JsonValue


FAKE = Path(__file__).with_name("fake_modal")


@pytest.fixture(autouse=True)
def fake_modal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    """Make ``import modal`` find the stand-in here and in every child interpreter."""
    for name in [name for name in sys.modules if name == "modal" or name.startswith("modal.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.delitem(sys.modules, "substrax.compute.modal_backend", raising=False)
    monkeypatch.syspath_prepend(str(FAKE))
    monkeypatch.setenv("PYTHONPATH", f"{FAKE}:{Path(__file__).resolve().parents[2] / 'src'}")
    monkeypatch.setenv("FAKE_MODAL_ROOT", str(tmp_path / "fake-modal"))
    fake = importlib.import_module("modal")
    assert Path(fake.__file__ or "").is_relative_to(FAKE)
    yield fake
    for name in [name for name in sys.modules if name == "modal" or name.startswith("modal.")]:
        sys.modules.pop(name)


def _backend(state_dir: Path, **settings: JsonValue) -> ComputeBackend:
    module = importlib.import_module("substrax.compute.modal_backend")
    return module.ModalBackend(
        {"outputs_volume": "outputs", "state_dir": str(state_dir), **settings}
    )


class TestModalBackend(BackendContract):
    def make_backend(self, state_dir: Path) -> ComputeBackend:
        return _backend(state_dir)


def _spec(**changes: object) -> JobSpec:
    base = JobSpec(
        name="examples",
        tasks=(Task(name="t", argv=("python", "-c", "pass"), timeout_seconds=60.0),),
        timeout_seconds=90.5,
        extras=("cuda12",),
    )
    return dataclasses.replace(base, **changes)


def test_the_image_installs_the_lock_with_this_python_then_the_source_then_the_project(
    fake_modal: ModuleType, tmp_path: Path
) -> None:
    project = tmp_path / "project"
    project.mkdir()

    _backend(tmp_path / "state").submit(_spec(), project=project)

    image = fake_modal.LAST["function"].options["image"]
    assert [step for step, _ in image.steps] == [
        "debian_slim",
        "uv_sync",
        "add_local_dir",
        "run_commands",
    ]
    python = f"{sys.version_info.major}.{sys.version_info.minor}"
    assert dict(image.steps)["debian_slim"] == python
    uv_sync = dict(image.steps)["uv_sync"]
    assert uv_sync == {
        "uv_project_dir": str(project.resolve()),
        "extras": ["cuda12"],
        "frozen": True,
        "extra_options": f"--python {python}",
    }
    add_local_dir = dict(image.steps)["add_local_dir"]
    assert add_local_dir["remote_path"] == "/project"
    assert add_local_dir["copy"] is True
    assert ".venv" in add_local_dir["ignore"]
    (install,) = dict(image.steps)["run_commands"]
    assert "--no-deps /project" in install


@pytest.mark.parametrize(
    ("accelerator", "gpu"),
    [(None, None), (Accelerator(kind="L4"), "L4"), (Accelerator(kind="H100", count=2), "H100:2")],
)
def test_the_function_requests_the_accelerator_and_the_timeout(
    fake_modal: ModuleType, tmp_path: Path, accelerator: Accelerator | None, gpu: str | None
) -> None:
    (tmp_path / "project").mkdir()

    _backend(tmp_path / "state").submit(
        _spec(accelerator=accelerator), project=tmp_path / "project"
    )

    options = fake_modal.LAST["function"].options
    assert options["gpu"] == gpu
    assert options["timeout"] == 91
    assert options["serialized"] is True


def test_each_mount_is_its_configured_volume_and_read_only_when_asked(
    fake_modal: ModuleType, tmp_path: Path
) -> None:
    (tmp_path / "project").mkdir()
    mounts = (
        Mount(name="datasets", path="/data", access=MountAccess.READ_ONLY),
        Mount(name="checkpoints", path="/ckpt", access=MountAccess.READ_WRITE),
    )
    backend = _backend(
        tmp_path / "state", volumes={"datasets": "demo-data", "checkpoints": "demo-ckpt"}
    )

    backend.submit(_spec(mounts=mounts), project=tmp_path / "project")

    volumes = fake_modal.LAST["function"].options["volumes"]
    assert {path: (v.name, v.is_read_only) for path, v in volumes.items()} == {
        "/outputs": ("outputs", False),
        "/data": ("demo-data", True),
        "/ckpt": ("demo-ckpt", False),
    }


def test_a_mount_without_a_configured_volume_is_refused(tmp_path: Path) -> None:
    (tmp_path / "project").mkdir()
    mount = Mount(name="datasets", path="/data", access=MountAccess.READ_ONLY)

    with pytest.raises(LookupError, match="'datasets'"):
        _backend(tmp_path / "state").submit(_spec(mounts=(mount,)), project=tmp_path / "project")


def test_the_outputs_volume_is_a_required_setting(tmp_path: Path) -> None:
    module = importlib.import_module("substrax.compute.modal_backend")

    with pytest.raises(ValueError, match="outputs_volume"):
        module.ModalBackend({"state_dir": str(tmp_path)})
