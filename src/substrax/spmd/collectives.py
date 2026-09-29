"""Distributed metrics collection utilities.

This module provides functions for collecting and aggregating metrics
across multiple devices in distributed training settings.

Two API variants are provided:

- **Default functions** (reduce_mean, reduce_sum, etc.): Use standard JAX
  operations (jnp.mean, jnp.sum) on global arrays. Work in SPMD contexts
  with nnx.jit + mesh.

- **Collective functions** (reduce_mean_collective, etc.): Use JAX collective
  operations (lax.pmean, lax.psum). Only valid inside pmap or shard_map contexts.
"""

import logging
from collections.abc import Callable, Mapping

import jax
import jax.numpy as jnp
from jax import lax


logger = logging.getLogger(__name__)


def _reduce_arrays[T](
    metrics: T,
    reduce: Callable[[jax.Array], jax.Array],
    *,
    skip_scalars: bool,
) -> T:
    """Apply ``reduce`` to every array leaf of ``metrics``, at any depth.

    Every other leaf, and a 0-d array when ``skip_scalars``, passes through unchanged, so the
    result has the structure of ``metrics``: a gradient tree beside the scalars is reduced leaf by
    leaf.
    """

    def leaf(value: object) -> object:
        if isinstance(value, jax.Array) and not (skip_scalars and value.ndim == 0):
            return reduce(value)
        return value

    return jax.tree.map(leaf, metrics)


# ---------------------------------------------------------------------------
# SPMD-compatible reductions (work with global arrays in nnx.jit)
# ---------------------------------------------------------------------------


def reduce_mean[T](metrics: T) -> T:
    """Compute the mean of metrics using standard JAX operations.

    Works with global arrays in SPMD contexts (nnx.jit + mesh). Every array leaf with at least one
    dimension is reduced, at any depth of ``metrics``; every other leaf passes through unchanged.

    Args:
        metrics: The metrics to reduce: a mapping, or any pytree of them.

    Returns:
        ``metrics`` with every array leaf mean-reduced.
    """
    return _reduce_arrays(metrics, jnp.mean, skip_scalars=True)


def reduce_sum[T](metrics: T) -> T:
    """Compute the sum of metrics using standard JAX operations.

    Works with global arrays in SPMD contexts (nnx.jit + mesh). Every array leaf with at least one
    dimension is reduced, at any depth of ``metrics``; every other leaf passes through unchanged.

    Args:
        metrics: The metrics to reduce: a mapping, or any pytree of them.

    Returns:
        ``metrics`` with every array leaf summed.
    """
    return _reduce_arrays(metrics, jnp.sum, skip_scalars=True)


def reduce_max[T](metrics: T) -> T:
    """Compute the maximum of metrics using standard JAX operations.

    Works with global arrays in SPMD contexts (nnx.jit + mesh). Every array leaf with at least one
    dimension is reduced, at any depth of ``metrics``; every other leaf passes through unchanged.

    Args:
        metrics: The metrics to reduce: a mapping, or any pytree of them.

    Returns:
        ``metrics`` with every array leaf reduced to its maximum.
    """
    return _reduce_arrays(metrics, jnp.max, skip_scalars=True)


def reduce_min[T](metrics: T) -> T:
    """Compute the minimum of metrics using standard JAX operations.

    Works with global arrays in SPMD contexts (nnx.jit + mesh). Every array leaf with at least one
    dimension is reduced, at any depth of ``metrics``; every other leaf passes through unchanged.

    Args:
        metrics: The metrics to reduce: a mapping, or any pytree of them.

    Returns:
        ``metrics`` with every array leaf reduced to its minimum.
    """
    return _reduce_arrays(metrics, jnp.min, skip_scalars=True)


_SPMD_REDUCTION_OPS: dict[str, Callable[[jax.Array], jax.Array]] = {
    "mean": jnp.mean,
    "sum": jnp.sum,
    "max": jnp.max,
    "min": jnp.min,
}


def reduce_custom[V](
    metrics: Mapping[str, V],
    reduce_fn: Mapping[str, str | None] | None = None,
) -> Mapping[str, V | jax.Array]:
    """Apply custom reduction operations to metrics.

    Uses standard JAX operations. Works in SPMD contexts. Each named value is reduced by its
    operation when it is an array; without ``reduce_fn`` this is ``reduce_mean``.

    Args:
        metrics: The metrics to reduce, by name.
        reduce_fn: A dictionary mapping metric names to reduction operations.
            Each operation should be one of {"mean", "sum", "max", "min"}.
            If None, defaults to "mean" for all metrics.

    Returns:
        A dictionary of reduced metrics.
    """
    if reduce_fn is None:
        return reduce_mean(metrics)

    result: dict[str, V | jax.Array] = {}
    for key, value in metrics.items():
        operation = reduce_fn.get(key, "mean")
        op_fn = _SPMD_REDUCTION_OPS.get(operation) if operation else None
        if op_fn is not None and isinstance(value, jax.Array) and value.ndim > 0:
            result[key] = op_fn(value)
        else:
            result[key] = value
    return result


# ---------------------------------------------------------------------------
# Collective reductions (only valid inside pmap or shard_map)
# ---------------------------------------------------------------------------


def reduce_mean_collective[T](
    metrics: T,
    axis_name: str = "batch",
) -> T:
    """Compute the mean of metrics using collective operations.

    Only valid inside a pmap or shard_map context.

    Args:
        metrics: The metrics to reduce: a mapping, or any pytree of them.
        axis_name: The name of the axis to reduce across.

    Returns:
        ``metrics`` with every array leaf averaged across the axis.
    """
    return _reduce_arrays(metrics, lambda x: lax.pmean(x, axis_name=axis_name), skip_scalars=False)


def reduce_sum_collective[T](
    metrics: T,
    axis_name: str = "batch",
) -> T:
    """Compute the sum of metrics using collective operations.

    Only valid inside a pmap or shard_map context.

    Args:
        metrics: The metrics to reduce: a mapping, or any pytree of them.
        axis_name: The name of the axis to reduce across.

    Returns:
        ``metrics`` with every array leaf summed.
    """
    return _reduce_arrays(metrics, lambda x: lax.psum(x, axis_name=axis_name), skip_scalars=False)


def all_gather[T](
    metrics: T,
    axis_name: str = "batch",
) -> T:
    """Gather metrics from all devices.

    Only valid inside a pmap or shard_map context.

    Args:
        metrics: The metrics to gather: a mapping, or any pytree of them.
        axis_name: The name of the axis to gather across.

    Returns:
        ``metrics`` with every array leaf gathered across the axis.
    """
    return _reduce_arrays(
        metrics, lambda x: lax.all_gather(x, axis_name=axis_name), skip_scalars=False
    )


def collect_from_devices[V](metrics: Mapping[str, V]) -> dict[str, list[jax.Array] | V]:
    """Collect metrics from all devices.

    Call outside of a pmapped function to split per-device values
    from the leading device axis.

    Args:
        metrics: The metrics from all devices, with the first dimension
            corresponding to the device axis.

    Returns:
        A dictionary of metrics, with array values split into per-device lists.
    """
    result: dict[str, list[jax.Array] | V] = {}
    for key, value in metrics.items():
        if isinstance(value, jax.Array) and value.ndim > 0:
            result[key] = [value[i] for i in range(value.shape[0])]
        else:
            result[key] = value
    return result
