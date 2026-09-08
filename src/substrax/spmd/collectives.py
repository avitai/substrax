"""Metric reductions for SPMD training.

Two families: the plain reductions (``reduce_mean`` and friends) apply ``jnp`` operations to
global arrays and work inside ``nnx.jit`` under a mesh; the ``*_collective`` functions apply
``lax`` collectives and are valid only inside ``jax.shard_map`` (or ``pmap``) bodies.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal

import jax
import jax.numpy as jnp
from jax import lax

from substrax.typing import Metrics


Reduction = Literal["mean", "sum", "max", "min"]

_REDUCTIONS: dict[str, Callable[[jax.Array], jax.Array]] = {
    "mean": jnp.mean,
    "sum": jnp.sum,
    "max": jnp.max,
    "min": jnp.min,
}


def _reduce_arrays(metrics: Metrics, operation: Callable[[jax.Array], jax.Array]) -> Metrics:
    def maybe_reduce(value: Any) -> Any:
        if isinstance(value, jax.Array) and value.ndim > 0:
            return operation(value)
        return value

    return jax.tree.map(maybe_reduce, metrics)


def reduce_mean(metrics: Metrics) -> Metrics:
    """Reduce every non-scalar array to its mean; other leaves pass through."""
    return _reduce_arrays(metrics, jnp.mean)


def reduce_sum(metrics: Metrics) -> Metrics:
    """Reduce every non-scalar array to its sum; other leaves pass through."""
    return _reduce_arrays(metrics, jnp.sum)


def reduce_max(metrics: Metrics) -> Metrics:
    """Reduce every non-scalar array to its maximum; other leaves pass through."""
    return _reduce_arrays(metrics, jnp.max)


def reduce_min(metrics: Metrics) -> Metrics:
    """Reduce every non-scalar array to its minimum; other leaves pass through."""
    return _reduce_arrays(metrics, jnp.min)


def reduce_custom(metrics: Metrics, reductions: Mapping[str, Reduction | None]) -> Metrics:
    """Reduce each metric with its own operation.

    Args:
        metrics: The metrics to reduce.
        reductions: Operation per metric name; a missing name means ``"mean"`` and ``None``
            leaves that metric unreduced.

    Returns:
        The reduced metrics.
    """
    result: Metrics = {}
    for name, value in metrics.items():
        operation = reductions.get(name, "mean")
        reducer = _REDUCTIONS[operation] if operation is not None else None
        if reducer is not None and isinstance(value, jax.Array) and value.ndim > 0:
            result[name] = reducer(value)
        else:
            result[name] = value
    return result


def _apply_collective(metrics: Metrics, collective: Callable[[jax.Array], jax.Array]) -> Metrics:
    def maybe_apply(value: Any) -> Any:
        if isinstance(value, jax.Array):
            return collective(value)
        return value

    return jax.tree.map(maybe_apply, metrics)


def reduce_mean_collective(metrics: Metrics, axis_name: str = "data") -> Metrics:
    """Mean-reduce every array across a mapped axis with ``lax.pmean``."""
    return _apply_collective(metrics, lambda x: lax.pmean(x, axis_name=axis_name))


def reduce_sum_collective(metrics: Metrics, axis_name: str = "data") -> Metrics:
    """Sum-reduce every array across a mapped axis with ``lax.psum``."""
    return _apply_collective(metrics, lambda x: lax.psum(x, axis_name=axis_name))


def all_gather(metrics: Metrics, axis_name: str = "data") -> Metrics:
    """Gather every array across a mapped axis with ``lax.all_gather``."""
    return _apply_collective(metrics, lambda x: lax.all_gather(x, axis_name=axis_name))


def collect_from_devices(metrics: Metrics) -> dict[str, list[Any] | Any]:
    """Split per-device values off the leading axis of each array, outside any mapping.

    Args:
        metrics: Metrics whose arrays carry a leading device axis.

    Returns:
        The metrics with each such array split into a list of per-device values.
    """
    result: dict[str, list[Any] | Any] = {}
    for name, value in metrics.items():
        if isinstance(value, jax.Array) and value.ndim > 0:
            result[name] = [value[index] for index in range(value.shape[0])]
        else:
            result[name] = value
    return result
