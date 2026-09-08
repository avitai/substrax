"""Explicit device placement and hardware-aware batch-size recommendations.

The recommendation table follows the JAX performance guide: the critical batch size is
where an accelerator reaches its roofline (TPU v5e 240, H100 298).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec, Sharding, SingleDeviceSharding

from substrax.devices.info import detect_devices, DeviceInfo, DeviceKind
from substrax.typing import PyTree


class HardwareType(StrEnum):
    """Accelerator generations the recommendation table knows about."""

    TPU_V5E = "tpu_v5e"
    TPU_V5P = "tpu_v5p"
    TPU_V4 = "tpu_v4"
    H100 = "h100"
    A100 = "a100"
    V100 = "v100"
    CPU = "cpu"
    UNKNOWN = "unknown"

    @classmethod
    def from_device(cls, platform: str, device_kind: str) -> HardwareType:
        """Classify one device from its backend name and ``device_kind`` string.

        Args:
            platform: The backend name as JAX reports it.
            device_kind: The device's ``device_kind`` string.

        Returns:
            The generation, or ``UNKNOWN`` for an accelerator the table does not cover; the
            table's conservative defaults apply rather than a guessed neighbour.
        """
        kind = DeviceKind.from_platform(platform)
        if kind is DeviceKind.CPU:
            return cls.CPU
        markers = _TPU_MARKERS if kind is DeviceKind.TPU else _GPU_MARKERS
        lowered = device_kind.lower()
        for marker, hardware in markers:
            if marker in lowered:
                return hardware
        return cls.UNKNOWN


_TPU_MARKERS: tuple[tuple[str, HardwareType], ...] = (
    ("v5e", HardwareType.TPU_V5E),
    ("v5 lite", HardwareType.TPU_V5E),
    ("v5p", HardwareType.TPU_V5P),
    ("v4", HardwareType.TPU_V4),
)
_GPU_MARKERS: tuple[tuple[str, HardwareType], ...] = (
    ("h100", HardwareType.H100),
    ("a100", HardwareType.A100),
    ("v100", HardwareType.V100),
)


@dataclass(frozen=True, slots=True, kw_only=True)
class BatchSizeRecommendation:
    """Batch sizes for one accelerator generation.

    Attributes:
        min_batch_size: Below this, throughput degrades sharply.
        critical_batch_size: Where the accelerator reaches its roofline.
        optimal_batch_size: The size to aim for.
        notes: Where the numbers come from.
    """

    min_batch_size: int
    critical_batch_size: int
    optimal_batch_size: int
    notes: str = ""


_RECOMMENDATIONS: dict[HardwareType, BatchSizeRecommendation] = {
    HardwareType.TPU_V5E: BatchSizeRecommendation(
        min_batch_size=64,
        critical_batch_size=240,
        optimal_batch_size=256,
        notes="Critical batch size for reaching roofline on TPU v5e.",
    ),
    HardwareType.TPU_V5P: BatchSizeRecommendation(
        min_batch_size=128,
        critical_batch_size=480,
        optimal_batch_size=512,
        notes="Higher throughput variant, benefits from larger batches.",
    ),
    HardwareType.TPU_V4: BatchSizeRecommendation(
        min_batch_size=64,
        critical_batch_size=192,
        optimal_batch_size=256,
        notes="Similar characteristics to TPU v5e but slightly lower critical batch.",
    ),
    HardwareType.H100: BatchSizeRecommendation(
        min_batch_size=64,
        critical_batch_size=298,
        optimal_batch_size=320,
        notes="Critical batch size for reaching roofline on H100.",
    ),
    HardwareType.A100: BatchSizeRecommendation(
        min_batch_size=32,
        critical_batch_size=240,
        optimal_batch_size=256,
        notes="A100 80GB variant. 40GB variant may need smaller batches.",
    ),
    HardwareType.V100: BatchSizeRecommendation(
        min_batch_size=16,
        critical_batch_size=96,
        optimal_batch_size=128,
        notes="Older generation, memory-limited on 16GB variant.",
    ),
    HardwareType.CPU: BatchSizeRecommendation(
        min_batch_size=1,
        critical_batch_size=16,
        optimal_batch_size=32,
        notes="CPU is memory-bandwidth bound, smaller batches often sufficient.",
    ),
    HardwareType.UNKNOWN: BatchSizeRecommendation(
        min_batch_size=32,
        critical_batch_size=64,
        optimal_batch_size=128,
        notes="Conservative defaults for unknown hardware.",
    ),
}


def detect_hardware_type(info: DeviceInfo | None = None) -> HardwareType:
    """Classify the accelerator generation of the first visible device.

    Args:
        info: A device snapshot; ``None`` probes the runtime.

    Returns:
        The generation.
    """
    snapshot = info if info is not None else detect_devices()
    if snapshot.count == 0:
        return HardwareType.UNKNOWN
    return HardwareType.from_device(snapshot.platform, snapshot.device_kinds[0])


def batch_size_recommendation(hardware_type: HardwareType | None = None) -> BatchSizeRecommendation:
    """Look up the recommendation for a generation, defaulting to the detected one.

    Args:
        hardware_type: The generation; ``None`` detects it.

    Returns:
        The recommendation.
    """
    return _RECOMMENDATIONS[hardware_type if hardware_type is not None else detect_hardware_type()]


@dataclass(frozen=True, slots=True, kw_only=True)
class BatchSizeVerdict:
    """The outcome of validating a batch size.

    Attributes:
        is_valid: Whether the size reaches the minimum.
        message: Why, in one sentence.
    """

    is_valid: bool
    message: str


def validate_batch_size(
    batch_size: int, hardware_type: HardwareType | None = None, *, warn_suboptimal: bool = True
) -> BatchSizeVerdict:
    """Judge a batch size against the recommendation for a generation.

    Args:
        batch_size: The size to judge.
        hardware_type: The generation; ``None`` detects it.
        warn_suboptimal: Whether a valid size below the critical size is called out.

    Returns:
        The verdict.
    """
    kind = hardware_type if hardware_type is not None else detect_hardware_type()
    rec = _RECOMMENDATIONS[kind]
    if batch_size < rec.min_batch_size:
        return BatchSizeVerdict(
            is_valid=False,
            message=(
                f"Batch size {batch_size} is below minimum recommended {rec.min_batch_size} "
                f"for {kind.value}; throughput will degrade sharply."
            ),
        )
    if batch_size < rec.critical_batch_size and warn_suboptimal:
        return BatchSizeVerdict(
            is_valid=True,
            message=(
                f"Batch size {batch_size} is below the critical size {rec.critical_batch_size} "
                f"for {kind.value}; consider increasing it for optimal throughput."
            ),
        )
    if batch_size >= rec.optimal_batch_size:
        return BatchSizeVerdict(
            is_valid=True, message=f"Batch size {batch_size} is optimal for {kind.value}."
        )
    return BatchSizeVerdict(
        is_valid=True, message=f"Batch size {batch_size} is acceptable for {kind.value}."
    )


def place_on_device(data: PyTree, device: Any | None = None) -> PyTree:
    """Copy every array of a pytree onto one device.

    Args:
        data: The pytree.
        device: The target; ``None`` uses the first visible device.

    Returns:
        The pytree with its arrays on that device.
    """
    target = device if device is not None else jax.devices()[0]
    return jax.device_put(data, SingleDeviceSharding(target))


def distribute_batch(data: PyTree, sharding: Sharding) -> PyTree:
    """Place every array of a pytree with one sharding.

    Args:
        data: The pytree.
        sharding: Where the arrays go.

    Returns:
        The placed pytree.
    """
    return jax.device_put(data, sharding)


def replicate_across_devices(data: PyTree, devices: Sequence[Any] | None = None) -> PyTree:
    """Copy every array of a pytree onto each device.

    Args:
        data: The pytree.
        devices: The devices; ``None`` uses every visible device.

    Returns:
        The replicated pytree.
    """
    targets = list(devices) if devices is not None else jax.devices()
    mesh = Mesh(np.array(targets), axis_names=("replica",))
    return jax.device_put(data, NamedSharding(mesh, PartitionSpec(None)))


def shard_batch_dim(
    data: PyTree, mesh: Mesh, *, batch_axis: int = 0, mesh_axis: str = "data"
) -> PyTree:
    """Shard every array along its batch axis; scalars and non-arrays are left as they are.

    Args:
        data: The pytree.
        mesh: The mesh to shard across.
        batch_axis: Which array axis is the batch.
        mesh_axis: Which mesh axis carries it.

    Returns:
        The sharded pytree.
    """

    def maybe_shard(leaf: Any) -> Any:
        if not isinstance(leaf, jax.Array) or leaf.ndim == 0:
            return leaf
        axes: list[str | None] = [None] * leaf.ndim
        if batch_axis < leaf.ndim:
            axes[batch_axis] = mesh_axis
        return jax.device_put(leaf, NamedSharding(mesh, PartitionSpec(*axes)))

    return jax.tree.map(maybe_shard, data)
