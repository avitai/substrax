"""Tests for device information."""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import pytest
from hypothesis import given, strategies as st

from substrax.devices import detect_devices, DeviceInfo, DeviceKind


_KNOWN_PLATFORMS = {"cpu", "gpu", "cuda", "rocm", "tpu", "metal"}


def test_positive_control_cpu_backend_is_reported_as_cpu() -> None:
    """Under JAX_PLATFORMS=cpu the probe reports the CPU backend and at least one device."""
    info = detect_devices()

    assert info.platform == "cpu"
    assert info.kind is DeviceKind.CPU
    assert info.count >= 1
    assert len(info.device_kinds) == info.count
    assert info.has_accelerator is False


def test_detect_devices_reflects_the_devices_jax_reports(monkeypatch: pytest.MonkeyPatch) -> None:
    """The probe is a pure projection of jax.devices() and jax.default_backend()."""

    class _Device:
        def __init__(self, kind: str) -> None:
            self.device_kind = kind

    fake_devices: list[Any] = [_Device("NVIDIA H100 80GB HBM3"), _Device("NVIDIA H100 80GB HBM3")]
    monkeypatch.setattr(jax, "devices", lambda: fake_devices)
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")

    info = detect_devices()

    assert info.platform == "gpu"
    assert info.kind is DeviceKind.GPU
    assert info.count == 2
    assert info.device_kinds == ("NVIDIA H100 80GB HBM3", "NVIDIA H100 80GB HBM3")
    assert info.has_accelerator is True


@pytest.mark.parametrize(
    ("platform", "kind"),
    [
        ("cpu", DeviceKind.CPU),
        ("gpu", DeviceKind.GPU),
        ("cuda", DeviceKind.GPU),
        ("rocm", DeviceKind.GPU),
        ("tpu", DeviceKind.TPU),
        ("METAL", DeviceKind.METAL),
    ],
)
def test_device_kind_maps_every_backend_name_jax_reports(platform: str, kind: DeviceKind) -> None:
    """Backend names map case-insensitively onto the four accelerator classes."""
    assert DeviceKind.from_platform(platform) is kind


@given(st.text(min_size=1).filter(lambda s: s.lower() not in _KNOWN_PLATFORMS))
def test_unknown_backend_names_map_to_other(platform: str) -> None:
    """A backend this package has never seen is reported, not rejected."""
    assert DeviceKind.from_platform(platform) is DeviceKind.OTHER


def test_device_info_is_immutable() -> None:
    """DeviceInfo is a frozen snapshot."""
    info = DeviceInfo(platform="cpu", kind=DeviceKind.CPU, count=1, device_kinds=("cpu",))

    with pytest.raises(dataclasses.FrozenInstanceError):
        info.count = 2  # type: ignore[misc]
