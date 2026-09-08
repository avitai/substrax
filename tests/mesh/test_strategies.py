"""Tests for sharding configuration and strategies."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, strategies as st
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from substrax.mesh import (
    DataParallelStrategy,
    FSDPStrategy,
    MultiDimensionalStrategy,
    ParallelismConfig,
    PipelineParallelStrategy,
    ShardingConfig,
    ShardingStrategy,
    TensorParallelStrategy,
)


def _spec_of(array: jax.Array) -> PartitionSpec:
    sharding = array.sharding
    assert isinstance(sharding, NamedSharding)
    return sharding.spec


def _mesh(*axis_names: str) -> Mesh:
    devices = np.array(jax.devices()[:1]).reshape((1,) * len(axis_names))
    return Mesh(devices, axis_names=axis_names)


class TestShardingConfig:
    def test_total_device_count_is_the_product(self) -> None:
        config = ShardingConfig(
            data_parallel_size=2, tensor_parallel_size=4, pipeline_parallel_size=2
        )
        assert config.total_device_count == 16

    @given(st.integers(min_value=1, max_value=256))
    def test_from_device_count_never_exceeds_the_devices(self, device_count: int) -> None:
        assert ShardingConfig.from_device_count(device_count).total_device_count <= device_count

    def test_eight_devices_split_two_by_four(self) -> None:
        config = ShardingConfig.from_device_count(8)
        assert (config.data_parallel_size, config.tensor_parallel_size) == (2, 4)

    def test_config_is_frozen(self) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            ShardingConfig().data_parallel_size = 2  # type: ignore[misc]


class TestParallelismConfig:
    def test_valid_when_mesh_matches_the_sharding(self) -> None:
        config = ParallelismConfig(
            mesh_shape=(2, 4),
            mesh_axis_names=("data", "model"),
            sharding_config=ShardingConfig(data_parallel_size=2, tensor_parallel_size=4),
        )
        assert config.is_valid

    def test_from_sharding_config_names_only_the_parallel_axes(self) -> None:
        config = ParallelismConfig.from_sharding_config(
            ShardingConfig(data_parallel_size=2, tensor_parallel_size=4)
        )
        assert config.mesh_shape == (2, 4)
        assert config.mesh_axis_names == ("data", "model")

    def test_single_device_config_defaults_to_a_data_axis(self) -> None:
        config = ParallelismConfig.from_sharding_config(ShardingConfig())
        assert config.mesh_shape == (1,)
        assert config.mesh_axis_names == ("data",)

    def test_create_mesh_builds_the_declared_topology(self) -> None:
        config = ParallelismConfig.from_sharding_config(ShardingConfig())
        mesh = config.create_mesh()
        assert mesh.axis_names == ("data",)
        assert mesh.devices.shape == (1,)


class TestDataParallelStrategy:
    def test_shards_only_the_batch_dimension(self) -> None:
        strategy = DataParallelStrategy(axis_name="data")
        assert strategy.partition_spec(("batch", "sequence", "hidden")) == PartitionSpec(
            "data", None, None
        )

    def test_shard_places_the_array_with_a_leading_data_axis(self) -> None:
        strategy = DataParallelStrategy(axis_name="data")
        sharded = strategy.shard(jnp.ones((4, 3)), _mesh("data"))
        assert _spec_of(sharded) == PartitionSpec("data", None)


class TestFSDPStrategy:
    def test_small_weights_are_replicated(self) -> None:
        strategy = FSDPStrategy(axis_name="fsdp", min_weight_size=512)
        assert strategy.should_shard(jnp.ones((256, 768))) is False
        assert strategy.should_shard(jnp.ones((2048, 768))) is True

    def test_shards_the_leading_feature_dimension(self) -> None:
        strategy = FSDPStrategy(axis_name="fsdp")
        assert strategy.partition_spec(("out_features", "in_features")) == PartitionSpec(
            "fsdp", None
        )
        assert strategy.partition_spec(("batch", "hidden")) == PartitionSpec(None, None)

    def test_shard_replicates_below_the_threshold(self) -> None:
        strategy = FSDPStrategy(axis_name="fsdp", min_weight_size=8)
        sharded = strategy.shard(jnp.ones((4, 3)), _mesh("fsdp"))
        assert _spec_of(sharded) == PartitionSpec(None, None)


class TestTensorParallelStrategy:
    def test_named_dimension_is_sharded(self) -> None:
        strategy = TensorParallelStrategy(axis_name="model", shard_dimension="in_features")
        assert strategy.linear_weight_spec == PartitionSpec(None, "model")
        assert strategy.partition_spec(("out_features", "in_features")) == PartitionSpec(
            None, "model"
        )

    def test_default_shards_hidden_and_output_features(self) -> None:
        strategy = TensorParallelStrategy(axis_name="model")
        assert strategy.linear_weight_spec == PartitionSpec("model", None)
        assert strategy.attention_qkv_spec == PartitionSpec(None, "model")
        assert strategy.attention_output_spec == PartitionSpec("model", None)
        assert strategy.partition_spec(("batch", "hidden")) == PartitionSpec(None, "model")

    def test_shard_of_a_higher_rank_array_uses_the_last_dimension(self) -> None:
        strategy = TensorParallelStrategy(axis_name="model")
        sharded = strategy.shard(jnp.ones((2, 3, 4)), _mesh("model"))
        assert _spec_of(sharded) == PartitionSpec(None, None, "model")


class TestPipelineParallelStrategy:
    @given(st.integers(min_value=1, max_value=64), st.integers(min_value=1, max_value=16))
    def test_layers_are_assigned_completely_and_evenly(self, num_layers: int, stages: int) -> None:
        assignment = PipelineParallelStrategy(axis_name="pipe", num_stages=stages).assign_layers(
            num_layers
        )
        assert len(assignment) == stages
        assert sum(assignment) == num_layers
        assert max(assignment) - min(assignment) <= 1

    def test_communication_patterns_chain_the_stages(self) -> None:
        strategy = PipelineParallelStrategy(axis_name="pipe", num_stages=4)
        assert strategy.forward_communication_pattern == ((0, 1), (1, 2), (2, 3))
        assert strategy.backward_communication_pattern == ((1, 0), (2, 1), (3, 2))

    def test_tensors_are_replicated_within_a_stage(self) -> None:
        strategy = PipelineParallelStrategy(axis_name="pipe", num_stages=2)
        assert strategy.partition_spec(("batch", "hidden")) == PartitionSpec(None, None)


class TestMultiDimensionalStrategy:
    @staticmethod
    def _config() -> ParallelismConfig:
        return ParallelismConfig(
            mesh_shape=(2, 4),
            mesh_axis_names=("data", "model"),
            sharding_config=ShardingConfig(data_parallel_size=2, tensor_parallel_size=4),
        )

    def test_strategies_satisfy_the_protocol(self) -> None:
        strategy: ShardingStrategy = DataParallelStrategy(axis_name="data")
        assert isinstance(strategy, ShardingStrategy)

    def test_combines_specs_from_every_strategy(self) -> None:
        combined = MultiDimensionalStrategy(
            strategies={
                "data": DataParallelStrategy(axis_name="data"),
                "model": TensorParallelStrategy(axis_name="model", shard_dimension="in_features"),
            },
            config=self._config(),
        )
        assert combined.partition_spec(("batch", "in_features")) == PartitionSpec("data", "model")

    def test_tensor_parallel_wins_a_conflict(self) -> None:
        combined = MultiDimensionalStrategy(
            strategies={
                "fsdp": FSDPStrategy(axis_name="fsdp"),
                "model": TensorParallelStrategy(axis_name="model", shard_dimension="out_features"),
            },
            config=self._config(),
        )
        assert combined.partition_spec(("out_features", "in_features")) == PartitionSpec(
            "model", None
        )

    def test_data_and_fsdp_may_share_an_axis(self) -> None:
        MultiDimensionalStrategy(
            strategies={
                "data": DataParallelStrategy(axis_name="data"),
                "fsdp": FSDPStrategy(axis_name="data"),
            },
            config=self._config(),
        )

    def test_two_strategies_of_one_kind_on_one_axis_conflict(self) -> None:
        with pytest.raises(ValueError, match="data"):
            MultiDimensionalStrategy(
                strategies={
                    "a": DataParallelStrategy(axis_name="data"),
                    "b": DataParallelStrategy(axis_name="data"),
                },
                config=self._config(),
            )

    def test_incompatible_strategies_on_one_axis_conflict(self) -> None:
        with pytest.raises(ValueError, match="Conflicting"):
            MultiDimensionalStrategy(
                strategies={
                    "data": DataParallelStrategy(axis_name="data"),
                    "model": TensorParallelStrategy(axis_name="data"),
                },
                config=self._config(),
            )
