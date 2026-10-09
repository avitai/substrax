"""Weights for :func:`~substrax.testing.counted_calls` that count jax's dispatches and placements.

Pass :data:`JAX_CALL_WEIGHTS` as ``weights`` to count, in the block:

- ``jit_dispatch``: calls of a ``jax.jit``-compiled callable from Python code, whether the call
  hits the executable cache or traces. ``nnx.jit`` calls one, and so do the ``jax.numpy``
  functions jax jits itself. Compiles are a different event:
  :mod:`substrax.testing.compiles` counts those.
- ``device_put``: calls of ``jax.device_put``, by identity, so a call through an alias counts.
- ``device_put_leaves``: the leaves of the first argument each ``jax.device_put`` call passes,
  the value placed when it comes first, by position or as ``x=``. A call that passes another
  keyword before ``x`` counts that keyword's leaves.

A call counts only where Python code makes it: a dispatch that jax's C++ code makes, or a
placement inside a traced function, fires no call event. The module imports jax, so
:mod:`substrax.testing` does not import it.
"""

from __future__ import annotations

import operator
import sys
from collections.abc import Mapping
from types import MappingProxyType

import jax

from substrax.testing.calls import CallWeight


_COMPILED_TYPE = type(jax.jit(operator.pos))
"""The type ``jax.jit`` returns (jaxlib's ``PjitFunction``), which jax does not export."""


def jit_dispatches(callee: object, first: object) -> int:
    """Count one for each call of a ``jax.jit``-compiled callable.

    Args:
        callee: The called object.
        first: The first argument the call passes (unused).

    Returns:
        1 for a compiled callable, else 0.
    """
    del first
    return int(isinstance(callee, _COMPILED_TYPE))


def device_puts(callee: object, first: object) -> int:
    """Count one for each call of ``jax.device_put``.

    Args:
        callee: The called object.
        first: The first argument the call passes (unused).

    Returns:
        1 for ``jax.device_put``, else 0.
    """
    del first
    return int(callee is jax.device_put)


def leaves_placed(callee: object, first: object) -> int:
    """Count the leaves of the value each ``jax.device_put`` call places.

    Args:
        callee: The called object.
        first: The first argument the call passes, the value placed.

    Returns:
        The number of leaves of ``first`` for a ``jax.device_put`` call that passes an argument,
        else 0.
    """
    if callee is not jax.device_put or first is sys.monitoring.MISSING:
        return 0
    return len(jax.tree.leaves(first))


JAX_CALL_WEIGHTS: Mapping[str, CallWeight] = MappingProxyType(
    {
        "jit_dispatch": jit_dispatches,
        "device_put": device_puts,
        "device_put_leaves": leaves_placed,
    }
)
"""The three jax counters by key, for ``counted_calls(..., weights=JAX_CALL_WEIGHTS)``."""
