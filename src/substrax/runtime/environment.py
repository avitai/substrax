"""The environment variables that apply a ``JaxRuntime`` to a process before jax is imported."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from substrax.runtime.config import JaxRuntime
from substrax.runtime.errors import RuntimeConfigurationError
from substrax.runtime.xla_flags import merge_xla_flags, parse_xla_flags


_XLA_FLAGS_VARIABLE = "XLA_FLAGS"
_CPU_DEVICES_VARIABLE = "JAX_NUM_CPU_DEVICES"
_DEVICE_COUNT_FLAG = "--xla_force_host_platform_device_count"
# jaxlib reads the memory fraction from XLA_CLIENT_MEM_FRACTION, falls back to the deprecated name,
# and raises when both are set (jaxlib 0.11 xla_client.py).
_MEMORY_FRACTION_VARIABLE = "XLA_CLIENT_MEM_FRACTION"
_DEPRECATED_MEMORY_FRACTION_VARIABLE = "XLA_PYTHON_CLIENT_MEM_FRACTION"


def runtime_environment(runtime: JaxRuntime, base: Mapping[str, str]) -> dict[str, str]:  # noqa: DOC502
    """Return the environment variables that apply ``runtime`` to a process not yet running jax.

    Only the variables ``runtime`` sets are returned, and ``base`` is read, never changed. A
    field replaces the variable ``base`` holds, except ``xla_flags``, which merges into ``base``'s
    ``XLA_FLAGS`` by flag name. Pass the result to a child process on top of its environment, or
    write it to ``os.environ`` before importing jax.

    Args:
        runtime: The settings to render.
        base: The environment the variables will be added to.

    Returns:
        Variable names mapped to their values.

    Raises:
        RuntimeConfigurationError: If ``runtime`` sets a memory fraction and ``base`` still holds
            the deprecated ``XLA_PYTHON_CLIENT_MEM_FRACTION``, which jax refuses beside the
            current name.
        XlaFlagConflictError: If a requested flag has a different value in ``base``'s
            ``XLA_FLAGS``.
    """
    _refuse_deprecated_memory_fraction(runtime, base)
    merged_flags = (
        merge_xla_flags(base.get(_XLA_FLAGS_VARIABLE, ""), runtime.xla_flags)
        if runtime.xla_flags
        else None
    )
    values = {
        "JAX_PLATFORMS": None if runtime.platforms is None else ",".join(runtime.platforms),
        _CPU_DEVICES_VARIABLE: _as_text(runtime.cpu_devices),
        "JAX_ENABLE_X64": _as_text(runtime.enable_x64),
        "JAX_DEFAULT_MATMUL_PRECISION": runtime.matmul_precision,
        "JAX_COMPILATION_CACHE_DIR": _as_text(runtime.compilation_cache_dir),
        _XLA_FLAGS_VARIABLE: merged_flags,
        "XLA_PYTHON_CLIENT_PREALLOCATE": _as_text(runtime.preallocate),
        _MEMORY_FRACTION_VARIABLE: _as_text(runtime.memory_fraction),
    }
    return {name: value for name, value in values.items() if value is not None}


def resolve_test_runtime(  # noqa: DOC502
    env: Mapping[str, str],
    *,
    prefix: str,
    cuda_plugin_available: bool,
    default_cpu_devices: int = 8,
) -> JaxRuntime:
    """Choose the backend and the emulated CPU devices of a test run.

    Tests run on the CPU unless ``{prefix}JAX_PLATFORMS`` names an accelerator, which is then used
    without CPU emulation. An inherited ``JAX_PLATFORMS``, such as the ``cuda,cpu`` a developer
    shell exports, does not move tests onto an accelerator, so a local run matches CI. The CPU
    device count is ``{prefix}DEVICE_COUNT`` when set (``0`` for no emulation), otherwise the
    count ``JAX_NUM_CPU_DEVICES`` or the XLA device-count flag already names, otherwise
    ``default_cpu_devices``.

    Args:
        env: The process environment.
        prefix: The prefix of the repository's test variables, such as ``"DATARAX_TEST_"``.
        cuda_plugin_available: Whether a JAX CUDA plugin is installed.
        default_cpu_devices: The emulated device count when nothing else names one.

    Returns:
        The runtime of the test process.

    Raises:
        RuntimeConfigurationError: If CUDA is requested and no JAX CUDA plugin is installed.
        ValueError: If ``{prefix}DEVICE_COUNT`` is not a non-negative integer.
    """
    requested = env.get(f"{prefix}JAX_PLATFORMS", "")
    platforms = tuple(name.strip() for name in requested.split(",") if name.strip())
    if any(name != "cpu" for name in platforms):
        _require_cuda_plugin(
            platforms, f"{prefix}JAX_PLATFORMS={requested!r}", cuda_plugin_available
        )
        return JaxRuntime(platforms=platforms)
    return JaxRuntime(
        platforms=("cpu",), cpu_devices=_test_cpu_devices(env, prefix, default_cpu_devices)
    )


def _refuse_deprecated_memory_fraction(runtime: JaxRuntime, base: Mapping[str, str]) -> None:
    """Raise when a memory fraction would join the deprecated variable jax refuses beside it."""
    if runtime.memory_fraction is not None and _DEPRECATED_MEMORY_FRACTION_VARIABLE in base:
        raise RuntimeConfigurationError(
            f"memory_fraction sets {_MEMORY_FRACTION_VARIABLE}, but the environment also holds the "
            f"deprecated {_DEPRECATED_MEMORY_FRACTION_VARIABLE}, and jax refuses both; unset it"
        )


def _require_cuda_plugin(platforms: tuple[str, ...], request: str, available: bool) -> None:
    """Raise when ``platforms`` asks for CUDA and no JAX CUDA plugin is installed."""
    if not available and any(name.startswith("cuda") for name in platforms):
        raise RuntimeConfigurationError(
            f"{request} asks for CUDA, but no JAX CUDA plugin is installed"
        )


def _test_cpu_devices(env: Mapping[str, str], prefix: str, default: int) -> int | None:
    """The emulated CPU device count of a test run; ``None`` keeps the count already set."""
    variable = f"{prefix}DEVICE_COUNT"
    if variable in env:
        text = env[variable].strip()
        if not text.isdigit():
            raise ValueError(f"{variable} must be a non-negative integer, got {env[variable]!r}")
        return int(text) or None
    inherited_flags = parse_xla_flags(env.get(_XLA_FLAGS_VARIABLE, ""))
    if _CPU_DEVICES_VARIABLE in env or _DEVICE_COUNT_FLAG in inherited_flags:
        return None
    return default


def _as_text(value: bool | float | Path | None) -> str | None:
    """Render a field as jax reads it: a ``bool`` as ``true`` or ``false``, the rest with ``str``."""
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
