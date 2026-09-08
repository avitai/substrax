"""Device information: what JAX is running on, as an immutable snapshot."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import cast, Protocol

import jax


class DeviceKind(StrEnum):
    """The accelerator class behind a JAX backend name."""

    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"
    METAL = "metal"
    OTHER = "other"

    @classmethod
    def from_platform(cls, platform: str) -> DeviceKind:
        """Map a backend name as ``jax.default_backend()`` reports it onto a kind.

        Args:
            platform: The backend name, in any case (``"gpu"``, ``"cuda"``, ``"METAL"``).

        Returns:
            The matching kind, or ``OTHER`` for a backend this package has not seen.
        """
        return _PLATFORM_KINDS.get(platform.lower(), cls.OTHER)


_PLATFORM_KINDS: dict[str, DeviceKind] = {
    "cpu": DeviceKind.CPU,
    "gpu": DeviceKind.GPU,
    "cuda": DeviceKind.GPU,
    "rocm": DeviceKind.GPU,
    "tpu": DeviceKind.TPU,
    "metal": DeviceKind.METAL,
}


@dataclass(frozen=True, slots=True, kw_only=True)
class DeviceInfo:
    """Snapshot of the devices visible to the current JAX process.

    Attributes:
        platform: The default backend name exactly as JAX reports it.
        kind: The accelerator class of that backend.
        count: The number of visible devices.
        device_kinds: One ``device_kind`` string per visible device, in JAX order.
    """

    platform: str
    kind: DeviceKind
    count: int
    device_kinds: tuple[str, ...]

    @property
    def has_accelerator(self) -> bool:
        """Whether the default backend is anything other than the CPU."""
        return self.kind is not DeviceKind.CPU


def detect_devices() -> DeviceInfo:
    """Probe the JAX runtime once and return what it reports.

    Errors from the runtime (no backend, uninitialised plugins) propagate; a process that
    cannot see its devices should not be told it is running on a CPU.

    Returns:
        The device snapshot.
    """
    # jaxlib ships no stubs for Device; _DeviceLike names the one attribute this module reads.
    devices = cast(Sequence[_DeviceLike], jax.devices())
    platform: str = jax.default_backend()
    return DeviceInfo(
        platform=platform,
        kind=DeviceKind.from_platform(platform),
        count=len(devices),
        device_kinds=tuple(_device_kind(device) for device in devices),
    )


class _DeviceLike(Protocol):
    """The part of ``jax.Device`` this module reads; jaxlib ships no stubs for it."""

    @property
    def device_kind(self) -> str: ...


def _device_kind(device: _DeviceLike) -> str:
    """Return the runtime's ``device_kind`` string for one device."""
    return str(device.device_kind)
