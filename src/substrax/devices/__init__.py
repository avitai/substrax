"""Device information and device placement."""

from substrax.devices.info import detect_devices, DeviceInfo, DeviceKind
from substrax.devices.placement import (
    BatchSizeRecommendation,
    DevicePlacement,
    get_batch_size_recommendation,
    HardwareType,
    place_on_device,
)


__all__ = [
    "BatchSizeRecommendation",
    "DeviceInfo",
    "DeviceKind",
    "DevicePlacement",
    "HardwareType",
    "detect_devices",
    "get_batch_size_recommendation",
    "place_on_device",
]
