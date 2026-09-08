"""Data-parallel placement and the SPMD training step.

Everything here targets the current JAX SPMD model: global arrays placed with a
``NamedSharding``, ``nnx.jit`` under ``jax.set_mesh``, and the XLA compiler inserting the
gradient all-reduce. There is no ``pmap`` path.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec, Sharding

from substrax.typing import PyTree


GradientReduction = Literal["mean", "sum"]


def create_data_parallel_sharding(mesh: Mesh, data_axis: str = "data") -> NamedSharding:
    """Shard the leading dimension across the data axis.

    Args:
        mesh: The device mesh.
        data_axis: The mesh axis carrying the batch.

    Returns:
        The sharding.
    """
    return NamedSharding(mesh, PartitionSpec(data_axis))


def place_batch_on_shards(batch: PyTree, sharding: Sharding) -> PyTree:
    """Place every array in a batch with the sharding; other leaves are returned as they are.

    Args:
        batch: The batch pytree.
        sharding: Where its arrays go.

    Returns:
        The batch with its arrays placed.
    """

    def maybe_place(value: Any) -> Any:
        if isinstance(value, jax.Array):
            return jax.device_put(value, sharding)
        return value

    return jax.tree.map(maybe_place, batch)


def spmd_train_step(
    model: nnx.Module,
    optimizer: nnx.Optimizer[Any],
    loss_fn: Callable[[nnx.Module, PyTree], jax.Array],
    batch: PyTree,
) -> jax.Array:
    """Run one data-parallel training step: loss, gradients, optimizer update.

    Call it inside an ``nnx.jit``-compiled function with a mesh set; the compiler inserts
    the gradient all-reduce from the input sharding.

    Args:
        model: The model to train.
        optimizer: The optimizer wrapping the model's parameters.
        loss_fn: ``(model, batch) -> scalar loss``.
        batch: The (already placed) batch.

    Returns:
        The loss for this step.
    """
    loss, grads = nnx.value_and_grad(loss_fn)(model, batch)
    optimizer.update(model, grads)
    return loss


def place_nnx_state_on_shards(
    state: PyTree,
    mesh: Mesh,
    filter_sharding: nnx.StateSharding | Mapping[Any, PartitionSpec | Sharding],
) -> PyTree:
    """Place every variable of an NNX state tree according to a filter → sharding mapping.

    Args:
        state: The state tree, as ``nnx.state(model)`` returns it.
        mesh: The device mesh partition specs refer to.
        filter_sharding: An ``nnx.StateSharding`` or a mapping from NNX filters (for
            example ``nnx.Param``) to partition specs or shardings.

    Returns:
        A state tree of the same structure with every array placed.
    """
    state_sharding = (
        filter_sharding
        if isinstance(filter_sharding, nnx.StateSharding)
        else nnx.StateSharding(filter_sharding)
    )
    placed: list[tuple[Any, Any]] = []
    for path, variable in nnx.to_flat_state(state):
        spec_or_sharding = state_sharding.map_prefix(path, variable)
        sharding = (
            spec_or_sharding
            if isinstance(spec_or_sharding, Sharding)
            else NamedSharding(mesh, spec_or_sharding)
        )
        value = variable[...]
        value = (
            jax.reshard(value, sharding)
            if isinstance(value, jax.Array)
            else jax.device_put(value, sharding)
        )
        placed.append((path, variable.replace(value=value)))
    return nnx.from_flat_state(placed)


def reduce_gradient_tree(gradients: PyTree, reduction: GradientReduction = "mean") -> PyTree:
    """Reduce every gradient leaf with ``jnp.mean`` or ``jnp.sum``.

    Works on global arrays inside ``nnx.jit``; the compiler handles the cross-device part.

    Args:
        gradients: The gradient pytree.
        reduction: ``"mean"`` or ``"sum"``.

    Returns:
        The reduced pytree.

    Raises:
        ValueError: If ``reduction`` is neither ``"mean"`` nor ``"sum"``.
    """
    if reduction == "mean":
        return jax.tree.map(jnp.mean, gradients)
    if reduction == "sum":
        return jax.tree.map(jnp.sum, gradients)
    raise ValueError(f"Unsupported reduction {reduction!r}; expected 'mean' or 'sum'.")
