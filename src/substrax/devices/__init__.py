"""Device information and device placement."""

from substrax.devices.info import (
    detect_devices,
    DeviceInfo,
    DeviceKind,
    DeviceLike,
    GPUUnavailableError,
    require_gpu,
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
    "GPUUnavailableError",
    "HardwareType",
    "detect_devices",
    "get_batch_size_recommendation",
    "place_on_device",
    "require_gpu",
    "visible_devices",
]
