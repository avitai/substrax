"""Data parallelism utilities.

This module provides functions for data-parallel training in JAX models,
centered on current SPMD APIs via ``nnx.jit`` and meshes.
"""

import logging
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh, PartitionSpec, Sharding

from substrax.typing import PyTree


logger = logging.getLogger(__name__)


def create_data_parallel_sharding(mesh: Mesh, data_axis: str = "data") -> Sharding:
    """Create a Sharding object for data-parallel training.

    Args:
        mesh: The device mesh to use for sharding.
        data_axis: The name of the mesh axis to use for data parallelism.

    Returns:
        A JAX Sharding object for data-parallel training.
    """
    return jax.sharding.NamedSharding(mesh, PartitionSpec(data_axis))


def place_batch_on_shards(batch: PyTree, sharding: Sharding) -> PyTree:
    """Shard a batch of data across devices.

    Args:
        batch: The batch to shard.
        sharding: The sharding specification to use.

    Returns:
        The sharded batch.
    """

    def maybe_shard(x: Any) -> Any:
        if isinstance(x, jax.Array):
            return jax.device_put(x, sharding)
        return x

    return jax.tree.map(maybe_shard, batch)


def spmd_train_step(
    model: nnx.Module,
    optimizer: nnx.Optimizer[Any],
    loss_fn: Callable[[nnx.Module, PyTree], jax.Array],
    batch: PyTree,
) -> jax.Array:
    """Execute a data-parallel training step using SPMD.

    Uses nnx.value_and_grad for automatic differentiation. The XLA compiler
    handles gradient AllReduce automatically when model parameters are
    sharded across devices via jax.set_mesh or explicit NamedSharding.

    This function should be called inside an @nnx.jit decorated function
    with a mesh context active.

    Args:
        model: The NNX model to train.
        optimizer: The NNX optimizer wrapping the model.
        loss_fn: Function (model, batch) -> loss scalar.
        batch: The training batch (should be pre-sharded).

    Returns:
        The loss value.

    Example:
        mesh = jax.make_mesh((4,), ("data",))
        rules = data_parallel_rules()

        @nnx.jit
        def train_step(model, optimizer, batch):
            return spmd_train_step(model, optimizer, my_loss_fn, batch)

        with jax.set_mesh(mesh):
            loss = train_step(model, optimizer, batch)
    """
    loss, grads = nnx.value_and_grad(loss_fn)(model, batch)
    optimizer.update(model, grads)
    return loss


def place_nnx_state_on_shards(
    state: nnx.State[Any, Any],
    mesh: Mesh,
    filter_sharding: nnx.StateSharding | dict[Any, PartitionSpec | Sharding],
) -> nnx.State[Any, Any]:
    """Shard a Flax NNX state tree using current NNX sharding helpers."""
    state_sharding = (
        filter_sharding
        if isinstance(filter_sharding, nnx.StateSharding)
        else nnx.StateSharding(filter_sharding)
    )
    sharded_flat = []
    for path, variable in nnx.to_flat_state(state):
        spec_or_sharding = state_sharding.map_prefix(path, variable)
        sharding = (
            spec_or_sharding
            if isinstance(spec_or_sharding, Sharding)
            else jax.sharding.NamedSharding(mesh, spec_or_sharding)
        )
        value = variable[...]
        if isinstance(value, jax.Array):
            value = jax.reshard(value, sharding)
        else:
            value = jax.device_put(value, sharding)
        sharded_flat.append((path, variable.replace(value=value)))
    return nnx.from_flat_state(sharded_flat)


def reduce_gradient_tree(gradients: Any, reduce_type: str = "mean") -> Any:
    """Reduce gradients using standard JAX operations on global arrays.

    Works in SPMD contexts (inside nnx.jit with mesh). The XLA compiler
    handles cross-device communication automatically.

    Args:
        gradients: The gradients to reduce (global sharded arrays).
        reduce_type: The type of reduction ("mean" or "sum").

    Returns:
        The reduced gradients.

    Raises:
        ValueError: If reduce_type is not "mean" or "sum".
    """
    if reduce_type.lower() == "mean":
        return jax.tree.map(jnp.mean, gradients)
    if reduce_type.lower() == "sum":
        return jax.tree.map(jnp.sum, gradients)
    raise ValueError(f"Unsupported reduce_type: {reduce_type}")
