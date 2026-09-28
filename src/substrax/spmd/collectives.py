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


def _reduce_arrays[V](
    metrics: Mapping[str, V],
    reduce: Callable[[jax.Array], jax.Array],
    *,
    skip_scalars: bool,
) -> dict[str, V | jax.Array]:
    """Apply ``reduce`` to each array value of ``metrics``.

    Every other value, and a 0-d array when ``skip_scalars``, passes through unchanged.
    """
    return {
        name: reduce(value)
        if isinstance(value, jax.Array) and not (skip_scalars and value.ndim == 0)
        else value
        for name, value in metrics.items()
    }


# ---------------------------------------------------------------------------
# SPMD-compatible reductions (work with global arrays in nnx.jit)
# ---------------------------------------------------------------------------


def reduce_mean[V](metrics: Mapping[str, V]) -> dict[str, V | jax.Array]:
    """Compute the mean of metrics using standard JAX operations.

    Works with global arrays in SPMD contexts (nnx.jit + mesh). A value that is not an array
    with at least one dimension passes through unchanged.

    Args:
        metrics: The metrics to reduce.

    Returns:
        A dictionary of mean-reduced metrics.
    """
    return _reduce_arrays(metrics, jnp.mean, skip_scalars=True)


def reduce_sum[V](metrics: Mapping[str, V]) -> dict[str, V | jax.Array]:
    """Compute the sum of metrics using standard JAX operations.

    Works with global arrays in SPMD contexts (nnx.jit + mesh). A value that is not an array
    with at least one dimension passes through unchanged.

    Args:
        metrics: The metrics to reduce.

    Returns:
        A dictionary of summed metrics.
    """
    return _reduce_arrays(metrics, jnp.sum, skip_scalars=True)


def reduce_max[V](metrics: Mapping[str, V]) -> dict[str, V | jax.Array]:
    """Compute the maximum of metrics using standard JAX operations.

    Works with global arrays in SPMD contexts (nnx.jit + mesh). A value that is not an array
    with at least one dimension passes through unchanged.

    Args:
        metrics: The metrics to reduce.

    Returns:
        A dictionary of maximum metrics.
    """
    return _reduce_arrays(metrics, jnp.max, skip_scalars=True)


def reduce_min[V](metrics: Mapping[str, V]) -> dict[str, V | jax.Array]:
    """Compute the minimum of metrics using standard JAX operations.

    Works with global arrays in SPMD contexts (nnx.jit + mesh). A value that is not an array
    with at least one dimension passes through unchanged.

    Args:
        metrics: The metrics to reduce.

    Returns:
        A dictionary of minimum metrics.
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
) -> dict[str, V | jax.Array]:
    """Apply custom reduction operations to metrics.

    Uses standard JAX operations. Works in SPMD contexts.

    Args:
        metrics: The metrics to reduce.
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


def reduce_mean_collective[V](
    metrics: Mapping[str, V],
    axis_name: str = "batch",
) -> dict[str, V | jax.Array]:
    """Compute the mean of metrics using collective operations.

    Only valid inside a pmap or shard_map context.

    Args:
        metrics: The metrics to reduce.
        axis_name: The name of the axis to reduce across.

    Returns:
        A dictionary of mean metrics.
    """
    return _reduce_arrays(metrics, lambda x: lax.pmean(x, axis_name=axis_name), skip_scalars=False)


def reduce_sum_collective[V](
    metrics: Mapping[str, V],
    axis_name: str = "batch",
) -> dict[str, V | jax.Array]:
    """Compute the sum of metrics using collective operations.

    Only valid inside a pmap or shard_map context.

    Args:
        metrics: The metrics to reduce.
        axis_name: The name of the axis to reduce across.

    Returns:
        A dictionary of summed metrics.
    """
    return _reduce_arrays(metrics, lambda x: lax.psum(x, axis_name=axis_name), skip_scalars=False)


def all_gather[V](
    metrics: Mapping[str, V],
    axis_name: str = "batch",
) -> dict[str, V | jax.Array]:
    """Gather metrics from all devices.

    Only valid inside a pmap or shard_map context.

    Args:
        metrics: The metrics to gather.
        axis_name: The name of the axis to gather across.

    Returns:
        A dictionary of gathered metrics.
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
