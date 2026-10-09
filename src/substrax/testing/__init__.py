"""Test helpers for JAX packages: fresh-interpreter runs and an opt-in pytest plugin.

``run_python`` runs code or a script in a child interpreter whose JAX settings the test chooses,
with everything the parent's shell exported about jax removed, and returns a ``ChildResult``.
``cuda_is_visible`` probes the CUDA backend in such a child. ``restored_jax_config`` sets back the
global jax configuration a block changed. ``TraceCounter`` counts the traces a jitted function
takes and raises ``RetraceError`` when a block caused an unexpected number. ``run_example`` runs one
example per child interpreter with its outputs redirected (``substrax.examples.discover_examples``
lists them). The plugin, ``substrax.testing.pytest_plugin``, is
enabled from a ``conftest.py``: it adds the ``x64``, ``devices`` and ``accelerator`` markers and
fails a test that changes jax's global configuration, as jax's own test harness does.
``counted_calls`` counts the Python functions a block starts on every thread, grouped by
``by_package``, and ``per_iteration`` takes the exact work per iteration from two pass lengths, for
tests that bound work by count instead of time; ``substrax.testing.jax_calls`` adds jax's
dispatches and placements to those counts. ``substrax.testing.compiles`` records the XLA programs compiled inside a block and fails a block
that compiled an unexpected number, and ``substrax.testing.gradients`` checks a module's gradients
against finite differences in float64 (both import jax, so the package does not import them).
``substrax.testing.source_scans`` holds the contract checks a repository runs over its own source:
import-time logging and environment writes, and documented meshes without axis types.
Importing this package imports no jax.
"""

from substrax.testing.calls import (
    by_package,
    counted_calls,
    CountingFailedError,
    per_iteration,
    ToolIdsInUseError,
    UnevenCountError,
)
from substrax.testing.child_process import (
    ChildFailedError,
    ChildResult,
    cuda_is_visible,
    run_python,
)
from substrax.testing.examples import (
    ExampleRun,
    ExampleTimeoutError,
    run_example,
    unavailable_reason,
)
from substrax.testing.jax_config import restored_jax_config
from substrax.testing.traces import RetraceError, TraceCounter


__all__ = [
    "ChildFailedError",
    "ChildResult",
    "CountingFailedError",
    "ExampleRun",
    "ExampleTimeoutError",
    "RetraceError",
    "ToolIdsInUseError",
    "TraceCounter",
    "UnevenCountError",
    "by_package",
    "counted_calls",
    "cuda_is_visible",
    "per_iteration",
    "restored_jax_config",
    "run_example",
    "run_python",
    "unavailable_reason",
]
