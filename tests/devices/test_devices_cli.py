"""``substrax-devices``: what JAX runs on here, and whether a GPU is there when one is required."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from substrax.devices import cli
from substrax.devices.info import DeviceInfo, DeviceKind


REPOSITORY = Path(__file__).resolve().parents[2]
GPU = DeviceInfo(platform="gpu", kind=DeviceKind.GPU, count=1, device_kinds=("NVIDIA H100",))


def test_the_report_names_the_backend_and_its_devices(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main([]) == 0

    out = capsys.readouterr().out
    assert "Default backend: cpu" in out
    assert "Devices:" in out


def test_json_carries_the_same_report(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--json"]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["default_backend"] == "cpu"
    assert report["device_count"] >= 1
    assert set(report) >= {"python", "platform", "jax_platforms", "jax_version", "device_kinds"}


def test_require_gpu_fails_on_a_cpu_backend(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--require-gpu"]) == 1

    assert "no GPU" in capsys.readouterr().err


def test_require_gpu_passes_on_a_gpu_backend(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "detect_devices", lambda: GPU)

    assert cli.main(["--require-gpu"]) == 0

    assert "NVIDIA H100" in capsys.readouterr().out


def test_a_runtime_that_cannot_list_its_devices_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def broken() -> DeviceInfo:
        raise RuntimeError("Unable to initialize backend 'cuda'")

    monkeypatch.setattr(cli, "detect_devices", broken)

    assert cli.main([]) == 1

    assert "Unable to initialize backend 'cuda'" in capsys.readouterr().err


def test_the_command_is_installed_as_substrax_devices() -> None:
    pyproject = tomllib.loads((REPOSITORY / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["scripts"]["substrax-devices"] == "substrax.devices.cli:main"
