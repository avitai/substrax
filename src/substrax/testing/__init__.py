"""Test helpers for JAX packages: fresh-interpreter runs and an opt-in pytest plugin.

``run_python`` runs code or a script in a child interpreter whose JAX settings the test chooses,
with everything the parent's shell exported about jax removed, and returns a ``ChildResult``.
``cuda_is_visible`` probes the CUDA backend in such a child. ``restored_jax_config`` sets back the
global jax configuration a block changed. ``TraceCounter`` counts the traces a jitted function
takes and raises ``RetraceError`` when a block caused an unexpected number. The plugin, ``substrax.testing.pytest_plugin``, is
enabled from a ``conftest.py``: it adds the ``x64``, ``devices`` and ``accelerator`` markers and
fails a test that changes jax's global configuration, as jax's own test harness does.
Importing this package imports no jax.
"""

from substrax.testing.child_process import (
    ChildFailedError,
    ChildResult,
    cuda_is_visible,
    run_python,
)
from substrax.testing.jax_config import restored_jax_config
from substrax.testing.traces import RetraceError, TraceCounter


__all__ = [
    "ChildFailedError",
    "ChildResult",
    "RetraceError",
    "TraceCounter",
    "cuda_is_visible",
    "restored_jax_config",
    "run_python",
]
