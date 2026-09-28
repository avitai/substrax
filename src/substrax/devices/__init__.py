"""Device information and device placement."""

from substrax.devices.info import (
    detect_devices,
    DeviceInfo,
    DeviceKind,
    DeviceLike,
    visible_devices,
)
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
    "DeviceLike",
    "DevicePlacement",
    "HardwareType",
    "detect_devices",
    "get_batch_size_recommendation",
    "place_on_device",
    "visible_devices",
]
