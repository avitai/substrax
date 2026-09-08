"""Flax NNX state sharding contracts."""

import jax
from flax import nnx
from jax.sharding import PartitionSpec as P

from substrax.spmd import place_nnx_state_on_shards


def test_shard_nnx_state_uses_state_sharding_and_named_sharding() -> None:
    mesh = jax.make_mesh((1,), ("data",))
    model = nnx.Linear(2, 1, rngs=nnx.Rngs(0))
    state = nnx.state(model)

    sharded = place_nnx_state_on_shards(state, mesh, {nnx.Param: P()})

    assert isinstance(sharded, nnx.State)
    for _, variable in nnx.to_flat_state(sharded):
        sharding = variable[...].sharding
        assert isinstance(sharding, jax.sharding.NamedSharding)
        assert sharding.mesh == mesh
