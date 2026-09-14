"""Snapshot and restore jax's global configuration, the way jax's own test harness does."""

from __future__ import annotations

import importlib
from collections.abc import Generator
from contextlib import contextmanager


@contextmanager
def restored_jax_config() -> Generator[list[str], None, None]:  # noqa: DOC502
    """Set back every jax configuration value a block changed, and report which ones.

    jax's own test base class snapshots ``jax.config.values`` around each test and fails a test
    that changed a global value (``assert_global_configs_unchanged`` in ``jax/_src/test_util.py``).
    This context manager takes the same snapshot and, on exit, sets every value that differs back
    through ``jax.config.update``, whether the block finished or raised.

    A value read inside a thread-local context manager such as ``jax.enable_x64(True)`` is that
    context's, so a context entered and left inside the block changes nothing.

    Yields:
        list[str]: Empty while the block runs; on exit it holds the sorted names of the values the
        block changed, all of them restored.

    Raises:
        RuntimeError: If jax refuses to set a value back, as it refuses ``jax_num_cpu_devices``
            once its backends have started.
    """
    config = importlib.import_module("jax").config
    before = dict(config.values)
    changed: list[str] = []
    try:
        yield changed
    finally:
        after = config.values
        changed.extend(sorted(name for name, value in before.items() if after.get(name) != value))
        for name in changed:
            config.update(name, before[name])
