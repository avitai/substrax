"""``substrax-devices``: what JAX runs on here, and whether a GPU is there when one is required.

```text
substrax-devices [--require-gpu] [--json]
```

It prints the interpreter, the platform, ``JAX_PLATFORMS``, jax's version, the default backend and
each visible device, as JAX reports them; ``--json`` prints the same report as one JSON object. A
GPU lane runs it with ``--require-gpu`` before its examples, which exits 1 unless the default
backend is a GPU, the check a GPU-required example's ``require_gpu()`` makes. Exit codes: 0 for a
report (and a GPU where one is required), 1 for no GPU where one is required or a runtime that
cannot list its devices, 2 for a refused command.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from collections.abc import Sequence

import jax

from substrax.devices.info import detect_devices, DeviceInfo, GPUUnavailableError, require_gpu
from substrax.typing import JsonValue


_SUCCESS, _UNSUCCESSFUL = 0, 1


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``substrax-devices``.

    Args:
        argv: The arguments after the program name; the process's when ``None``.

    Returns:
        The exit code.
    """
    parser = argparse.ArgumentParser(prog="substrax-devices", description=__doc__)
    parser.add_argument(
        "--require-gpu", action="store_true", help="exit 1 unless the default backend is a GPU"
    )
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    arguments = parser.parse_args(argv)
    try:
        info = detect_devices()
    except RuntimeError as error:
        sys.stderr.write(f"substrax-devices: jax cannot list its devices: {error}\n")
        return _UNSUCCESSFUL
    report = _report(info)
    sys.stdout.write(json.dumps(report, indent=2) if arguments.json else _text(report))
    sys.stdout.write("\n")
    if arguments.require_gpu:
        try:
            require_gpu(info)
        except GPUUnavailableError as error:
            sys.stderr.write(f"substrax-devices: {error}\n")
            return _UNSUCCESSFUL
    return _SUCCESS


def _report(info: DeviceInfo) -> dict[str, JsonValue]:
    """The report as a JSON object."""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "jax_platforms": os.environ.get("JAX_PLATFORMS"),
        "jax_version": jax.__version__,
        "default_backend": info.platform,
        "device_count": info.count,
        "device_kinds": list(info.device_kinds),
    }


def _text(report: dict[str, JsonValue]) -> str:
    """The report as lines for a terminal."""
    kinds = report["device_kinds"]
    devices = [f"  - {kind}" for kind in kinds] if isinstance(kinds, list) else []
    return "\n".join(
        [
            f"Python: {report['python']}",
            f"Platform: {report['platform']}",
            f"JAX_PLATFORMS: {report['jax_platforms'] or 'unset'}",
            f"jax: {report['jax_version']}",
            f"Default backend: {report['default_backend']}",
            "Devices:",
            *devices,
        ]
    )
