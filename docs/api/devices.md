# Devices

`substrax.devices` answers two questions: what hardware is this process running on, and
where should a batch go.

```python
from substrax.devices import DevicePlacement, detect_devices, get_batch_size_recommendation

info = detect_devices()          # platform, kind, count, memory, name
placement = DevicePlacement()    # defaults to jax.devices()
batch = placement.place_on_device(batch)
recommendation = get_batch_size_recommendation()  # for the detected hardware
```

`DeviceInfo` is a frozen dataclass; `detect_devices()` reads it from `jax.devices()` and
never imports an accelerator plugin itself. `DevicePlacement` carries the batch-size
recommendation table per `HardwareType`.

::: substrax.devices
