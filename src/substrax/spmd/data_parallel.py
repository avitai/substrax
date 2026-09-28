"""Data parallelism utilities.

This module provides functions for data-parallel training in JAX models,
centered on current SPMD APIs via ``nnx.jit`` and meshes.
"""

import logging
from collections.abc import Callable, Mapping

import jax
import numpy as np
from flax import nnx
from flax.nnx.filterlib import Filter
from jax.sharding import Mesh, NamedSharding, PartitionSpec, Sharding

from substrax.mesh import create_named_sharding
from substrax.nnx_typing import NnxState, PathEntry
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


def place_batch_on_shards(  # noqa: DOC502  # the ValueError is raised by _check_holds
    batch: PyTree, sharding: Sharding | PyTree
) -> PyTree:
    """Turn this process's batch into global arrays, each subtree on its sharding.

    Array leaves, whether ``jax.Array``, a NumPy array as a host data loader yields it, or a NumPy
    scalar, go to ``jax.make_array_from_process_local_data`` in one call. On a single process that
    is one batched ``jax.device_put``. On several processes each process passes the slice of the
    global batch it loaded, and jax stitches the slices into one global array; a leaf on a
    replicated sharding must be equal on every process. Any other leaf, such as a string, is
    returned as it is. Call it on the host batches a loader yields, outside ``jax.jit``: inside a
    traced function the result is not placed on ``sharding``.

    Args:
        batch: The batch this process loaded.
        sharding: One sharding for every array leaf, or a pytree prefix of ``batch`` whose leaves
            are shardings, each applying to its subtree: rows sharded on the batch axis and a
            batch-level field without one replicated, for example.

    Returns:
        The batch with every array leaf replaced by the global array on its sharding.

    Raises:
        ValueError: If a leaf has fewer dimensions than its sharding partitions (a scalar under
            a sharding that splits the batch axis); the message names the leaf's path.
    """
    paths_and_leaves, treedef = jax.tree.flatten_with_path(batch)
    leaves = [leaf for _, leaf in paths_and_leaves]
    shardings = _leaf_shardings(batch, sharding, len(leaves))
    positions = [
        index
        for index, leaf in enumerate(leaves)
        if isinstance(leaf, jax.Array | np.ndarray | np.generic)
    ]
    arrays = [_as_array(leaves[index]) for index in positions]
    for index, array in zip(positions, arrays, strict=True):
        _check_holds(paths_and_leaves[index][0], array, shardings[index])
    placed = jax.make_array_from_process_local_data(
        sharding if isinstance(sharding, Sharding) else [shardings[index] for index in positions],
        arrays,
    )
    for index, leaf in zip(positions, placed, strict=True):
        leaves[index] = leaf
    return jax.tree.unflatten(treedef, leaves)


def _leaf_shardings(batch: PyTree, sharding: Sharding | PyTree, count: int) -> list[Sharding]:
    """The sharding of each of ``batch``'s ``count`` leaves, a prefix broadcast over its subtree."""
    if isinstance(sharding, Sharding):
        return [sharding] * count
    return jax.tree.leaves(
        jax.tree.broadcast(sharding, batch, is_leaf=lambda node: isinstance(node, Sharding))
    )


def _as_array(leaf: jax.Array | np.ndarray | np.generic) -> jax.Array | np.ndarray:
    """A NumPy scalar as a 0-d array; any other array leaf as it is."""
    return np.asarray(leaf) if isinstance(leaf, np.generic) else leaf


def _check_holds(
    path: jax.tree_util.KeyPath[PathEntry], array: jax.Array | np.ndarray, sharding: Sharding
) -> None:
    """Refuse a leaf with fewer dimensions than its named sharding partitions, naming its path."""
    if isinstance(sharding, NamedSharding) and len(sharding.spec) > np.ndim(array):
        msg = (
            f"batch leaf {jax.tree_util.keystr(path)} has {np.ndim(array)} dimension(s), but its "
            f"sharding partitions {len(sharding.spec)} ({sharding.spec}); give that subtree a "
            "replicated sharding in a prefix"
        )
        raise ValueError(msg)


def spmd_train_step[M: nnx.Module](
    model: M,
    optimizer: nnx.Optimizer[M],
    loss_fn: Callable[[M, PyTree], jax.Array],
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
    state: NnxState,
    mesh: Mesh,
    filter_sharding: nnx.StateSharding | Mapping[Filter, PartitionSpec | Sharding],
) -> NnxState:
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
