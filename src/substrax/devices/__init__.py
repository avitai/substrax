"""Device information and device placement."""

from substrax.devices.info import detect_devices, DeviceInfo, DeviceKind
from substrax.devices.placement import (
    batch_size_recommendation,
    BatchSizeRecommendation,
    BatchSizeVerdict,
    detect_hardware_type,
    distribute_batch,
    HardwareType,
    place_on_device,
    replicate_across_devices,
    shard_batch_dim,
    validate_batch_size,
)


__all__ = [
    "BatchSizeRecommendation",
    "BatchSizeVerdict",
    "DeviceInfo",
    "DeviceKind",
    "HardwareType",
    "batch_size_recommendation",
    "detect_devices",
    "detect_hardware_type",
    "distribute_batch",
    "place_on_device",
    "replicate_across_devices",
    "shard_batch_dim",
    "validate_batch_size",
]
