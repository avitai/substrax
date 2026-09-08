"""Tests for data-parallel placement, the SPMD training step and gradient reduction."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx
from hypothesis import given, settings, strategies as st
from hypothesis.extra import numpy as hnp
from jax.sharding import NamedSharding, PartitionSpec

from substrax.mesh import create_data_parallel_mesh, create_device_mesh
from substrax.spmd import (
    create_data_parallel_sharding,
    place_batch_on_shards,
    place_nnx_state_on_shards,
    reduce_gradient_tree,
    spmd_train_step,
)


def _loss(model: nnx.Module, batch: dict[str, jax.Array]) -> jax.Array:
    linear = model
    assert isinstance(linear, nnx.Linear)
    return jnp.mean((linear(batch["x"]) - batch["y"]) ** 2)


class TestCreateDataParallelSharding:
    def test_shards_the_leading_dimension_on_the_data_axis(self) -> None:
        sharding = create_data_parallel_sharding(create_data_parallel_mesh(num_devices=1))
        assert sharding.spec == PartitionSpec("data")

    def test_custom_axis_name(self) -> None:
        sharding = create_data_parallel_sharding(create_device_mesh([("batch", 1)]), "batch")
        assert sharding.spec == PartitionSpec("batch")


class TestPlaceBatchOnShards:
    def test_arrays_are_placed_and_other_leaves_kept(self) -> None:
        sharding = create_data_parallel_sharding(create_data_parallel_mesh(num_devices=1))
        batch = {"inputs": jnp.ones((4, 2)), "targets": jnp.zeros((4,)), "label": "s"}

        result = place_batch_on_shards(batch, sharding)

        assert result["inputs"].sharding == sharding
        assert result["targets"].shape == (4,)
        assert result["label"] == "s"


class TestPlaceNnxStateOnShards:
    def test_every_leaf_lands_on_the_mesh_with_a_named_sharding(self) -> None:
        mesh = jax.make_mesh((1,), ("data",))
        state = nnx.state(nnx.Linear(2, 1, rngs=nnx.Rngs(0)))

        sharded = place_nnx_state_on_shards(state, mesh, {nnx.Param: PartitionSpec()})

        for _, variable in nnx.to_flat_state(sharded):
            sharding = variable[...].sharding
            assert isinstance(sharding, NamedSharding)
            assert sharding.mesh == mesh

    def test_accepts_a_prebuilt_state_sharding(self) -> None:
        mesh = jax.make_mesh((1,), ("data",))
        state = nnx.state(nnx.Linear(2, 1, rngs=nnx.Rngs(0)))
        state_sharding = nnx.StateSharding({nnx.Param: NamedSharding(mesh, PartitionSpec())})

        sharded = place_nnx_state_on_shards(state, mesh, state_sharding)

        assert jax.tree.structure(sharded) == jax.tree.structure(state)


class TestReduceGradientTree:
    @given(
        # XLA flushes subnormals on CPU; see test_collectives.py.
        hnp.arrays(
            np.float32,
            hnp.array_shapes(min_dims=1, max_dims=2, max_side=6),
            elements=st.floats(-100, 100, width=32, allow_subnormal=False),
        )
    )
    @settings(deadline=None)
    def test_mean_and_sum_match_numpy(self, values: np.ndarray) -> None:
        grads = {"w": jnp.asarray(values)}
        assert np.isclose(float(reduce_gradient_tree(grads, "mean")["w"]), values.mean(), rtol=1e-4)
        assert np.isclose(
            float(reduce_gradient_tree(grads, "sum")["w"]), values.sum(), rtol=1e-4, atol=1e-2
        )

    def test_unknown_reduction_raises(self) -> None:
        with pytest.raises(ValueError, match="invalid"):
            reduce_gradient_tree({"w": jnp.array([1.0])}, "invalid")  # type: ignore[arg-type]


class TestSpmdTrainStep:
    @staticmethod
    def _model_and_optimizer() -> tuple[nnx.Linear, nnx.Optimizer[Any]]:
        model = nnx.Linear(2, 1, rngs=nnx.Rngs(0))
        return model, nnx.Optimizer(model, optax.sgd(0.01), wrt=nnx.Param)

    def test_one_step_returns_a_finite_loss_and_moves_the_parameters(self) -> None:
        model, optimizer = self._model_and_optimizer()
        before = jax.tree.map(jnp.copy, nnx.state(model, nnx.Param))
        batch = {"x": jnp.ones((4, 2)), "y": jnp.zeros((4, 1))}

        loss = spmd_train_step(model, optimizer, _loss, batch)

        assert jnp.isfinite(loss)
        after = nnx.state(model, nnx.Param)
        assert any(
            not jnp.array_equal(b, a)
            for b, a in zip(jax.tree.leaves(before), jax.tree.leaves(after), strict=True)
        )

    def test_loss_decreases_over_steps(self) -> None:
        model, optimizer = self._model_and_optimizer()
        batch = {"x": jnp.ones((4, 2)), "y": jnp.zeros((4, 1))}

        first = spmd_train_step(model, optimizer, _loss, batch)
        last = first
        for _ in range(10):
            last = spmd_train_step(model, optimizer, _loss, batch)

        assert float(last) < float(first)

    def test_runs_under_nnx_jit_with_a_mesh(self) -> None:
        """The step is meant to be called inside nnx.jit with a mesh set; prove it compiles."""
        model, optimizer = self._model_and_optimizer()
        mesh = create_data_parallel_mesh(num_devices=1)
        batch = place_batch_on_shards(
            {"x": jnp.ones((4, 2)), "y": jnp.zeros((4, 1))}, create_data_parallel_sharding(mesh)
        )

        @nnx.jit
        def step(m: nnx.Linear, o: nnx.Optimizer[Any], b: dict[str, jax.Array]) -> jax.Array:
            return spmd_train_step(m, o, _loss, b)

        with jax.set_mesh(mesh):
            loss = step(model, optimizer, batch)
        assert jnp.isfinite(loss)
