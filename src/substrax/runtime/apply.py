"""Apply a ``JaxRuntime`` to the current process."""

from __future__ import annotations

import os
import sys
from collections.abc import MutableMapping

from substrax.runtime.config import JaxRuntime
from substrax.runtime.environment import runtime_environment
from substrax.runtime.errors import RuntimeConfigurationError


# Read by jax and XLA only when the backends start, and jax offers no public signal of whether
# they have.
_BACKEND_START_FIELDS = ("platforms", "xla_flags", "preallocate", "memory_fraction")


def apply_runtime(  # noqa: DOC502
    runtime: JaxRuntime, *, environ: MutableMapping[str, str] | None = None
) -> None:
    """Apply ``runtime`` to the current process.

    Before jax is imported, every field is written to ``environ`` as the variables jax and XLA
    read at start-up (see :func:`~substrax.runtime.runtime_environment`). Once jax is imported,
    the CPU device count, 64-bit types, matmul precision and compilation cache directory go
    through ``jax.config.update``; jax refuses a new device count once its backends have started.
    Platforms, XLA flags and client memory settings are read only when the backends start, and jax
    offers no public signal of whether that has happened, so once jax is imported they raise
    instead of being written where they might change nothing.

    Args:
        runtime: The settings to apply.
        environ: The environment written before jax is imported; ``os.environ`` when ``None``.

    Raises:
        RuntimeConfigurationError: If jax is imported and ``runtime`` sets a platform, an XLA
            flag or a client memory setting, or jax refuses the CPU device count.
        XlaFlagConflictError: If a requested flag has a different value in the current
            ``XLA_FLAGS``.
    """
    target = os.environ if environ is None else environ
    if "jax" not in sys.modules:
        target.update(runtime_environment(runtime, target))
        return
    _refuse_backend_start_fields(runtime)
    _update_jax_config(runtime)


def _refuse_backend_start_fields(runtime: JaxRuntime) -> None:
    """Raise, naming them, when ``runtime`` sets fields that jax reads only at backend start."""
    given = [name for name in _BACKEND_START_FIELDS if getattr(runtime, name) not in (None, ())]
    if given:
        raise RuntimeConfigurationError(
            f"{', '.join(given)} must be applied before jax is imported: jax reads them when its "
            "backends start and offers no public signal of whether that has happened"
        )


def _update_jax_config(runtime: JaxRuntime) -> None:
    """Set the fields jax exposes as config options, the CPU device count first."""
    config = sys.modules["jax"].config
    if runtime.cpu_devices is not None:
        try:
            config.update("jax_num_cpu_devices", runtime.cpu_devices)
        except RuntimeError as error:
            raise RuntimeConfigurationError(
                f"cpu_devices={runtime.cpu_devices} cannot be applied: jax has already started "
                "its backends"
            ) from error
    cache_dir = runtime.compilation_cache_dir
    options = {
        "jax_enable_x64": runtime.enable_x64,
        "jax_default_matmul_precision": runtime.matmul_precision,
        "jax_compilation_cache_dir": None if cache_dir is None else str(cache_dir),
    }
    for option, value in options.items():
        if value is not None:
            config.update(option, value)
