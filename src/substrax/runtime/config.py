"""Declarative JAX process settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import get_args, Literal

from substrax.runtime.xla_flags import merge_xla_flags


# The precision names jax accepts for jax_default_matmul_precision (jax 0.11 jax/_src/config.py);
# the dot-algorithm presets it also accepts are left out.
MatmulPrecision = Literal["default", "high", "highest", "bfloat16", "tensorfloat32", "float32"]
_MATMUL_PRECISIONS: tuple[str, ...] = get_args(MatmulPrecision)


@dataclass(frozen=True, slots=True, kw_only=True)
class JaxRuntime:
    """Declarative JAX process settings; ``None`` leaves the process's current value alone.

    :func:`~substrax.runtime.runtime_environment` renders the settings as the environment
    variables jax and XLA read when a process starts, and
    :func:`~substrax.runtime.apply_runtime` applies them to the current process.

    Attributes:
        platforms: The backends jax initialises, the first being the default (``JAX_PLATFORMS``).
        cpu_devices: The number of CPU devices (``JAX_NUM_CPU_DEVICES``). jax prefers it over
            the ``--xla_force_host_platform_device_count`` flag.
        enable_x64: Whether 64-bit types are enabled (``JAX_ENABLE_X64``).
        matmul_precision: The default matrix-multiplication precision
            (``JAX_DEFAULT_MATMUL_PRECISION``).
        compilation_cache_dir: The absolute directory of the persistent compilation cache
            (``JAX_COMPILATION_CACHE_DIR``).
        xla_flags: Flags merged into ``XLA_FLAGS`` by name, each ``--name`` or ``--name=value``.
        preallocate: Whether the accelerator client preallocates its memory
            (``XLA_PYTHON_CLIENT_PREALLOCATE``).
        memory_fraction: The fraction of accelerator memory the client may use, in ``(0, 1]``
            (``XLA_CLIENT_MEM_FRACTION``).
    """

    platforms: tuple[str, ...] | None = None
    cpu_devices: int | None = None
    enable_x64: bool | None = None
    matmul_precision: MatmulPrecision | None = None
    compilation_cache_dir: Path | None = None
    xla_flags: tuple[str, ...] = ()
    preallocate: bool | None = None
    memory_fraction: float | None = None

    def __post_init__(self) -> None:  # noqa: DOC502
        """Validate every field.

        Raises:
            TypeError: If a field holds a value of the wrong type.
            ValueError: If a field holds a value outside its domain.
            XlaFlagConflictError: If one requested flag is given two different values.
        """
        _check_platforms(self.platforms)
        _check_int("cpu_devices", self.cpu_devices)
        _check_bool("enable_x64", self.enable_x64)
        _check_bool("preallocate", self.preallocate)
        _check_matmul_precision(self.matmul_precision)
        _check_cache_dir(self.compilation_cache_dir)
        _check_memory_fraction(self.memory_fraction)
        merge_xla_flags("", self.xla_flags)


def _check_platforms(platforms: tuple[str, ...] | None) -> None:
    """Refuse an empty platform list, and a platform name that is blank or holds a comma."""
    if platforms is not None and (
        not platforms or any(not name.strip() or "," in name for name in platforms)
    ):
        raise ValueError(
            f"platforms must name at least one backend, each without commas, got {platforms!r}"
        )


def _check_int(field: str, value: object) -> None:
    """Refuse a count that is not an int (``bool`` included) or is below one."""
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an int, got {value!r}")
    if value < 1:
        raise ValueError(f"{field} must be at least 1, got {value}")


def _check_bool(field: str, value: object) -> None:
    """Refuse a switch that is not a ``bool``."""
    if value is not None and not isinstance(value, bool):
        raise TypeError(f"{field} must be a bool, got {value!r}")


def _check_matmul_precision(value: object) -> None:
    """Refuse a precision name jax does not accept."""
    if value is not None and value not in _MATMUL_PRECISIONS:
        raise ValueError(f"matmul_precision must be one of {_MATMUL_PRECISIONS}, got {value!r}")


def _check_cache_dir(value: Path | None) -> None:
    """Refuse a relative cache directory, whose meaning would depend on the working directory."""
    if value is not None and not value.is_absolute():
        raise ValueError(f"compilation_cache_dir must be an absolute path, got {value}")


def _check_memory_fraction(value: object) -> None:
    """Refuse a fraction that is not a number (``bool`` included) or lies outside ``(0, 1]``."""
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"memory_fraction must be a float, got {value!r}")
    if not 0.0 < value <= 1.0:  # a NaN fails both comparisons
        raise ValueError(f"memory_fraction must lie in (0, 1], got {value}")
