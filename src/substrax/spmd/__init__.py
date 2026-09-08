"""Data-parallel placement, training steps, gradient reduction and collectives."""

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
    Reduction,
)
from substrax.spmd.data_parallel import (
    create_data_parallel_sharding,
    GradientReduction,
    place_batch_on_shards,
    place_nnx_state_on_shards,
    reduce_gradient_tree,
    spmd_train_step,
)


__all__ = [
    "GradientReduction",
    "Reduction",
    "all_gather",
    "collect_from_devices",
    "create_data_parallel_sharding",
    "place_batch_on_shards",
    "place_nnx_state_on_shards",
    "reduce_custom",
    "reduce_gradient_tree",
    "reduce_max",
    "reduce_mean",
    "reduce_mean_collective",
    "reduce_min",
    "reduce_sum",
    "reduce_sum_collective",
    "spmd_train_step",
]
