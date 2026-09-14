"""Data parallelism utilities.

This module provides functions for data-parallel training in JAX models,
centered on current SPMD APIs via ``nnx.jit`` and meshes.
"""

import logging
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec, Sharding

from substrax.mesh import create_named_sharding
from substrax.typing import PyTree


logger = logging.getLogger(__name__)


def create_data_parallel_sharding(mesh: Mesh, data_axis: str = "data") -> NamedSharding:
    """Create the sharding that splits a batch's leading axis over a mesh axis.

    Args:
        mesh: The device mesh to use for sharding.
        data_axis: The name of the mesh axis to use for data parallelism.

    Returns:
        A ``NamedSharding`` over ``mesh`` with spec ``PartitionSpec(data_axis)``.
    """
    return create_named_sharding(mesh, data_axis)


def place_batch_on_shards(batch: PyTree, sharding: Sharding) -> PyTree:
    """Turn this process's batch into global arrays on ``sharding``.

    Array leaves, whether ``jax.Array`` or a NumPy array as a host data loader yields them, go
    to ``jax.make_array_from_process_local_data`` in one call. On a single process that is one
    batched ``jax.device_put``. On several processes each process passes the slice of the
    global batch it loaded, and jax stitches the slices into one global array. Any other leaf,
    such as a string, is returned as it is.

    Args:
        batch: The batch this process loaded.
        sharding: The sharding of the global batch.

    Returns:
        The batch with every array leaf replaced by the global array on ``sharding``.
    """
    leaves, treedef = jax.tree.flatten(batch)
    positions = [
        index for index, leaf in enumerate(leaves) if isinstance(leaf, jax.Array | np.ndarray)
    ]
    placed = jax.make_array_from_process_local_data(
        sharding, [leaves[index] for index in positions]
    )
    for index, leaf in zip(positions, placed, strict=True):
        leaves[index] = leaf
    return jax.tree.unflatten(treedef, leaves)


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
        mesh = DeviceMeshManager.create_device_mesh({"data": 4})
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
