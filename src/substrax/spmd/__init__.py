"""Data-parallel placement, the SPMD training step and collectives.

A data-parallel gradient needs no reduction after differentiating. Under ``jax.jit`` with the
batch sharded over a mesh axis, the gradient of the batch's loss is already the full gradient:
the compiler inserts the all-reduce (``spmd_train_step`` relies on this). Inside
``jax.shard_map`` each device sees its shard, and the loss is averaged over the axis before
differentiating, ``jax.grad(lambda w: jax.lax.pmean(loss(w, shard), axis))``; the gradient of a
replicated parameter then comes back as the full one. Averaging the gradient afterwards is wrong
by the axis size, because the gradient of a replicated parameter already arrives summed over
the axis. ``reduce_mean_collective`` and ``reduce_sum_collective`` average or sum per-device
metrics inside ``shard_map``.
"""

from substrax.spmd.collectives import (
    all_gather,
    collect_from_devices,
    reduce_custom,
    reduce_max,
    reduce_mean,
    reduce_mean_collective,
    reduce_min,
    reduce_sum,
    reduce_sum_collective,
)
from substrax.spmd.data_parallel import (
    create_data_parallel_sharding,
    place_batch_on_shards,
    place_nnx_state_on_shards,
    spmd_train_step,
)


__all__ = [
    "all_gather",
    "collect_from_devices",
    "create_data_parallel_sharding",
    "place_batch_on_shards",
    "place_nnx_state_on_shards",
    "reduce_custom",
    "reduce_max",
    "reduce_mean",
    "reduce_mean_collective",
    "reduce_min",
    "reduce_sum",
    "reduce_sum_collective",
    "spmd_train_step",
]
