"""Tests for data parallelism functions.

Tests both SPMD-based and legacy pmap-based data parallel utilities.
"""

from typing import Any

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx

from substrax.mesh import DeviceMeshManager
from substrax.spmd import (
    create_data_parallel_sharding,
    place_batch_on_shards,
    reduce_gradient_tree,
    spmd_train_step,
)


class TestCreateDataParallelSharding:
    """Tests for create_data_parallel_sharding function."""

    def test_single_device_sharding(self) -> None:
        """Test creating sharding with single device mesh."""
        mesh = DeviceMeshManager.create_data_parallel_mesh(num_devices=1)
        sharding = create_data_parallel_sharding(mesh)
        assert sharding.spec == jax.sharding.PartitionSpec("data")  # type: ignore[reportAttributeAccessIssue]

    def test_custom_data_axis(self) -> None:
        """Test creating sharding with custom axis name."""
        mesh = DeviceMeshManager.create_device_mesh([("batch", 1)])
        sharding = create_data_parallel_sharding(mesh, data_axis="batch")
        assert sharding.spec == jax.sharding.PartitionSpec("batch")  # type: ignore[reportAttributeAccessIssue]

    @pytest.mark.skipif(jax.device_count() < 2, reason="Requires 2+ devices")
    def test_multi_device_sharding(self) -> None:
        """Test creating sharding across multiple devices."""
        mesh = DeviceMeshManager.create_data_parallel_mesh(num_devices=2)
        sharding = create_data_parallel_sharding(mesh)
        assert sharding.spec == jax.sharding.PartitionSpec("data")  # type: ignore[reportAttributeAccessIssue]


class TestShardBatch:
    """Tests for place_batch_on_shards function."""

    def test_shards_array_values(self) -> None:
        """Test that jax.Array values are sharded."""
        mesh = DeviceMeshManager.create_data_parallel_mesh(num_devices=1)
        sharding = create_data_parallel_sharding(mesh)
        batch = {"inputs": jnp.ones((4, 2)), "targets": jnp.zeros((4,))}

        result = place_batch_on_shards(batch, sharding)  # type: ignore[reportArgumentType]

        assert result["inputs"].shape == (4, 2)
        assert result["targets"].shape == (4,)

    def test_non_array_values_unchanged(self) -> None:
        """Test that non-array values pass through unchanged."""
        mesh = DeviceMeshManager.create_data_parallel_mesh(num_devices=1)
        sharding = create_data_parallel_sharding(mesh)
        batch = {"data": jnp.ones((4, 2)), "label": "test_string"}

        result = place_batch_on_shards(batch, sharding)  # type: ignore[reportArgumentType]

        assert result["label"] == "test_string"

    @pytest.mark.skipif(jax.device_count() < 2, reason="Requires 2+ devices")
    def test_multi_device_shard(self) -> None:
        """Test sharding across multiple devices."""
        mesh = DeviceMeshManager.create_data_parallel_mesh(num_devices=2)
        sharding = create_data_parallel_sharding(mesh)
        batch = {"inputs": jnp.ones((4, 2)), "targets": jnp.zeros((4,))}

        result = place_batch_on_shards(batch, sharding)  # type: ignore[reportArgumentType]

        assert result["inputs"].shape == (4, 2)
        assert result["targets"].shape == (4,)


class TestReduceGradients:
    """Tests for reduce_gradient_tree (SPMD-compatible)."""

    def test_mean_reduction(self) -> None:
        """Test mean reduction on a gradient pytree."""
        grads = {"w": jnp.array([1.0, 2.0, 3.0]), "b": jnp.array([4.0, 6.0])}
        result = reduce_gradient_tree(grads, "mean")
        assert float(result["w"]) == 2.0
        assert float(result["b"]) == 5.0

    def test_sum_reduction(self) -> None:
        """Test sum reduction on a gradient pytree."""
        grads = {"w": jnp.array([1.0, 2.0, 3.0])}
        result = reduce_gradient_tree(grads, "sum")
        assert float(result["w"]) == 6.0

    def test_unsupported_reduction_raises(self) -> None:
        """Test that unsupported reduce_type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported reduce_type"):
            reduce_gradient_tree({"w": jnp.array([1.0])}, "invalid")


class TestSpmdTrainStep:
    """Tests for spmd_train_step function."""

    def _make_model_and_optimizer(self) -> tuple[nnx.Linear, nnx.Optimizer[Any]]:
        """Create a minimal NNX model and optimizer for testing."""
        model = nnx.Linear(2, 1, rngs=nnx.Rngs(0))
        optimizer = nnx.Optimizer(model, optax.sgd(0.01), wrt=nnx.Param)
        return model, optimizer

    def test_reduces_loss(self) -> None:
        """Test that a single training step produces a finite loss."""
        model, optimizer = self._make_model_and_optimizer()
        batch = {"x": jnp.ones((4, 2)), "y": jnp.zeros((4, 1))}

        def loss_fn(m: nnx.Module, b: dict[str, jax.Array]) -> jax.Array:
            return jnp.mean((m(b["x"]) - b["y"]) ** 2)

        loss = spmd_train_step(model, optimizer, loss_fn, batch)  # type: ignore[reportArgumentType]
        assert jnp.isfinite(loss)

    def test_updates_parameters(self) -> None:
        """Test that parameters change after a training step."""
        model, optimizer = self._make_model_and_optimizer()
        params_before = jax.tree.map(jnp.copy, nnx.state(model, nnx.Param))
        batch = {"x": jnp.ones((4, 2)), "y": jnp.zeros((4, 1))}

        def loss_fn(m: nnx.Module, b: dict[str, jax.Array]) -> jax.Array:
            return jnp.mean((m(b["x"]) - b["y"]) ** 2)

        spmd_train_step(model, optimizer, loss_fn, batch)  # type: ignore[reportArgumentType]
        params_after = nnx.state(model, nnx.Param)

        # At least one parameter leaf must have changed
        leaves_before = jax.tree.leaves(params_before)
        leaves_after = jax.tree.leaves(params_after)
        any_changed = any(
            not jnp.array_equal(b, a) for b, a in zip(leaves_before, leaves_after, strict=True)
        )
        assert any_changed, "Parameters should change after a training step"

    def test_loss_decreases_over_steps(self) -> None:
        """Test that loss decreases over multiple training steps."""
        model, optimizer = self._make_model_and_optimizer()
        batch = {"x": jnp.ones((4, 2)), "y": jnp.zeros((4, 1))}

        def loss_fn(m: nnx.Module, b: dict[str, jax.Array]) -> jax.Array:
            return jnp.mean((m(b["x"]) - b["y"]) ** 2)

        loss_first = spmd_train_step(model, optimizer, loss_fn, batch)  # type: ignore[reportArgumentType]
        loss_last = loss_first
        for _ in range(10):
            loss_last = spmd_train_step(model, optimizer, loss_fn, batch)  # type: ignore[reportArgumentType]

        assert float(loss_last) < float(loss_first)
