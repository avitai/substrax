"""Test suite for sharding infrastructure components.

This module tests the core sharding strategies, device mesh management,
and parallelism configuration functionality that enables scalable training.
"""

from unittest.mock import Mock, patch

import jax.numpy as jnp
import pytest
from jax.sharding import PartitionSpec

from substrax.mesh.strategies import (
    DataParallelStrategy,
    FSDPStrategy,
    MultiDimensionalStrategy,
    ParallelismConfig,
    PipelineParallelStrategy,
    ShardingConfig,
    ShardingStrategy,
    TensorParallelStrategy,
)


class TestShardingConfig:
    """Test the ShardingConfig dataclass for sharding configuration."""

    def test_sharding_config_creation(self) -> None:
        """Test creating sharding configuration."""
        config = ShardingConfig(
            data_parallel_size=4,
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            fsdp_enabled=True,
            fsdp_min_weight_size=1024,
        )

        assert config.data_parallel_size == 4
        assert config.tensor_parallel_size == 2
        assert config.pipeline_parallel_size == 1
        assert config.fsdp_enabled is True
        assert config.fsdp_min_weight_size == 1024

    def test_sharding_config_validation(self) -> None:
        """Test validation of sharding configuration."""
        # Test total device count validation
        config = ShardingConfig(
            data_parallel_size=2, tensor_parallel_size=4, pipeline_parallel_size=2
        )

        total_devices = config.get_total_device_count()
        assert total_devices == 16  # 2 * 4 * 2

    def test_sharding_config_from_device_count(self) -> None:
        """Test creating optimal config from device count."""
        config = ShardingConfig.from_device_count(8)

        assert isinstance(config, ShardingConfig)
        assert config.get_total_device_count() <= 8


class TestParallelismConfig:
    """Test the ParallelismConfig for multi-dimensional parallelism."""

    def test_parallelism_config_creation(self) -> None:
        """Test creating parallelism configuration."""
        config = ParallelismConfig(
            mesh_shape=(2, 4, 1),
            mesh_axis_names=("data", "model", "pipeline"),
            sharding_config=ShardingConfig(
                data_parallel_size=2, tensor_parallel_size=4, pipeline_parallel_size=1
            ),
        )

        assert config.mesh_shape == (2, 4, 1)
        assert config.mesh_axis_names == ("data", "model", "pipeline")
        assert config.sharding_config.data_parallel_size == 2

    def test_parallelism_config_validation(self) -> None:
        """Test parallelism configuration validation."""
        config = ParallelismConfig(
            mesh_shape=(2, 4),
            mesh_axis_names=("data", "model"),
            sharding_config=ShardingConfig(data_parallel_size=2, tensor_parallel_size=4),
        )

        # Validate mesh shape matches sharding config
        assert config.is_valid()


class TestShardingStrategy:
    """Test the abstract ShardingStrategy base class."""

    def test_sharding_strategy_interface(self) -> None:
        """Test that ShardingStrategy defines correct interface."""
        # Test that it's an abstract base class
        assert hasattr(ShardingStrategy, "get_partition_spec")
        assert hasattr(ShardingStrategy, "apply_sharding")
        assert hasattr(ShardingStrategy, "get_sharding_constraints")

    def test_sharding_strategy_validation(self) -> None:
        """Test strategy validation methods."""
        # This will test the base validation logic
        # when we implement concrete strategies


class TestDataParallelStrategy:
    """Test data parallel sharding strategy."""

    def test_data_parallel_strategy_creation(self) -> None:
        """Test creating data parallel strategy."""
        strategy = DataParallelStrategy(axis_name="data", mesh_axis=0)

        assert strategy.axis_name == "data"
        assert strategy.mesh_axis == 0

    def test_data_parallel_partition_spec(self) -> None:
        """Test data parallel partition specification."""
        strategy = DataParallelStrategy(axis_name="data", mesh_axis=0)

        # Test partition spec for different tensor shapes
        batch_spec = strategy.get_partition_spec(("batch", "sequence", "hidden"))
        assert batch_spec[0] == "data"  # batch dimension
        assert batch_spec[1] is None  # sequence dimension
        assert batch_spec[2] is None  # hidden dimension

    def test_data_parallel_sharding_application(self) -> None:
        """Test applying data parallel sharding to arrays."""
        strategy = DataParallelStrategy(axis_name="data", mesh_axis=0)

        # Mock array and mesh
        mock_array = jnp.ones((8, 512, 768))
        mock_mesh = Mock()

        # Mock the JAX NamedSharding to avoid std::bad_cast with mock objects
        with patch("substrax.mesh.strategies.NamedSharding") as mock_sharding_class:
            mock_sharding = Mock()
            mock_sharding_class.return_value = mock_sharding

            with patch("jax.device_put") as mock_device_put:
                mock_device_put.return_value = mock_array

                sharded_array = strategy.apply_sharding(mock_array, mock_mesh)

                # Verify sharding was applied
                assert sharded_array is not None
                mock_sharding_class.assert_called_once()
                mock_device_put.assert_called_once()


class TestFSDPStrategy:
    """Test Fully Sharded Data Parallel strategy."""

    def test_fsdp_strategy_creation(self) -> None:
        """Test creating FSDP strategy."""
        strategy = FSDPStrategy(axis_name="fsdp", mesh_axis=0, min_weight_size=1024)

        assert strategy.axis_name == "fsdp"
        assert strategy.mesh_axis == 0
        assert strategy.min_weight_size == 1024

    def test_fsdp_weight_sharding(self) -> None:
        """Test FSDP weight parameter sharding."""
        strategy = FSDPStrategy(axis_name="fsdp", mesh_axis=0, min_weight_size=512)

        # Test weight sharding decisions
        large_weight = jnp.ones((2048, 768))  # Should be sharded
        small_weight = jnp.ones((256, 768))  # Should not be sharded

        assert strategy.should_shard_weight(large_weight) is True
        assert strategy.should_shard_weight(small_weight) is False

    def test_fsdp_gradient_sharding(self) -> None:
        """Test FSDP gradient sharding strategy."""
        strategy = FSDPStrategy(axis_name="fsdp", mesh_axis=0)

        gradient_spec = strategy.get_gradient_partition_spec(("out_features", "in_features"))

        # FSDP shards along the first dimension of weights
        assert gradient_spec[0] == "fsdp"  # out_features dimension
        assert gradient_spec[1] is None  # in_features dimension


class TestTensorParallelStrategy:
    """Test tensor parallel sharding strategy."""

    def test_tensor_parallel_strategy_creation(self) -> None:
        """Test creating tensor parallel strategy."""
        strategy = TensorParallelStrategy(
            axis_name="model", mesh_axis=1, shard_dimension="in_features"
        )

        assert strategy.axis_name == "model"
        assert strategy.mesh_axis == 1
        assert strategy.shard_dimension == "in_features"

    def test_tensor_parallel_linear_sharding(self) -> None:
        """Test tensor parallel sharding for linear layers."""
        strategy = TensorParallelStrategy(
            axis_name="model", mesh_axis=1, shard_dimension="in_features"
        )

        # Test weight sharding for linear layer
        weight_spec = strategy.get_linear_weight_spec()
        assert weight_spec[1] == "model"  # in_features dimension
        assert weight_spec[0] is None  # out_features dimension

    def test_tensor_parallel_attention_sharding(self) -> None:
        """Test tensor parallel sharding for attention layers."""
        strategy = TensorParallelStrategy(axis_name="model", mesh_axis=1)

        # Test QKV projection sharding
        qkv_spec = strategy.get_attention_qkv_spec()
        assert qkv_spec[1] == "model"  # hidden dimension

        # Test output projection sharding
        output_spec = strategy.get_attention_output_spec()
        assert output_spec[0] == "model"  # hidden dimension


class TestPipelineParallelStrategy:
    """Test pipeline parallel sharding strategy."""

    def test_pipeline_parallel_strategy_creation(self) -> None:
        """Test creating pipeline parallel strategy."""
        strategy = PipelineParallelStrategy(axis_name="pipeline", mesh_axis=2, num_stages=4)

        assert strategy.axis_name == "pipeline"
        assert strategy.mesh_axis == 2
        assert strategy.num_stages == 4

    def test_pipeline_stage_assignment(self) -> None:
        """Test assigning layers to pipeline stages."""
        strategy = PipelineParallelStrategy(axis_name="pipeline", mesh_axis=2, num_stages=4)

        # Test with 12 layers across 4 stages
        layer_assignments = strategy.assign_layers_to_stages(num_layers=12)

        assert len(layer_assignments) == 4
        assert sum(layer_assignments) == 12
        # Should be roughly balanced: [3, 3, 3, 3]
        assert all(2 <= layers <= 4 for layers in layer_assignments)

    def test_pipeline_communication_pattern(self) -> None:
        """Test pipeline communication patterns."""
        strategy = PipelineParallelStrategy(axis_name="pipeline", mesh_axis=2, num_stages=4)

        # Test forward communication pattern
        forward_pattern = strategy.get_forward_communication_pattern()
        assert len(forward_pattern) == 3  # 4 stages = 3 communications

        # Test backward communication pattern
        backward_pattern = strategy.get_backward_communication_pattern()
        assert len(backward_pattern) == 3


class TestMultiDimensionalStrategy:
    """Test multi-dimensional parallelism strategy."""

    def test_multi_dimensional_strategy_creation(self) -> None:
        """Test creating multi-dimensional strategy."""
        strategies = {
            "data": DataParallelStrategy(axis_name="data", mesh_axis=0),
            "model": TensorParallelStrategy(axis_name="model", mesh_axis=1),
            "fsdp": FSDPStrategy(axis_name="fsdp", mesh_axis=0),
        }
        config = ParallelismConfig(
            mesh_shape=(2, 4),
            mesh_axis_names=("data", "model"),
            sharding_config=ShardingConfig(data_parallel_size=2, tensor_parallel_size=4),
        )

        multi_strategy = MultiDimensionalStrategy(strategies=strategies, config=config)

        assert len(multi_strategy.strategies) == 3
        assert "data" in multi_strategy.strategies
        assert "model" in multi_strategy.strategies
        assert "fsdp" in multi_strategy.strategies

    def test_multi_dimensional_partition_spec_combination(self) -> None:
        """Test combining partition specs from multiple strategies."""
        strategies = {
            "data": DataParallelStrategy(axis_name="data", mesh_axis=0),
            "model": TensorParallelStrategy(axis_name="model", mesh_axis=1),
        }

        config = ParallelismConfig(
            mesh_shape=(2, 4),
            mesh_axis_names=("data", "model"),
            sharding_config=ShardingConfig(data_parallel_size=2, tensor_parallel_size=4),
        )

        multi_strategy = MultiDimensionalStrategy(strategies=strategies, config=config)

        # Test combining specs for a weight tensor
        combined_spec = multi_strategy.get_combined_partition_spec(
            tensor_name="weight", tensor_shape=("out_features", "in_features")
        )

        # Should combine data parallel (batch) and tensor parallel (features)
        assert combined_spec is not None

    def test_multi_dimensional_conflict_resolution(self) -> None:
        """Test resolving conflicts between strategies."""
        # Test when multiple strategies want to shard the same dimension
        strategies = {
            "fsdp": FSDPStrategy(axis_name="fsdp", mesh_axis=0),
            "model": TensorParallelStrategy(axis_name="model", mesh_axis=1),
        }

        config = ParallelismConfig(
            mesh_shape=(2, 4),
            mesh_axis_names=("fsdp", "model"),
            sharding_config=ShardingConfig(data_parallel_size=2, tensor_parallel_size=4),
        )

        multi_strategy = MultiDimensionalStrategy(strategies=strategies, config=config)

        # Test conflict resolution (FSDP vs Tensor Parallel on weights)
        resolved_spec = multi_strategy.resolve_sharding_conflicts(
            tensor_name="weight",
            proposed_specs={
                "fsdp": PartitionSpec("fsdp", None),
                "model": PartitionSpec(None, "model"),
            },
        )

        assert resolved_spec is not None

    def test_multi_dimensional_strategy_has_no_invalid_pipeline_partition_helper(self) -> None:
        """The public strategy should not publish the invalid stage_* partition helper."""
        assert not hasattr(MultiDimensionalStrategy, "create_partition_spec")


class TestIntegrationSharding:
    """Integration tests for sharding infrastructure."""

    def test_strategy_compatibility_validation(self) -> None:
        """Test validation of strategy compatibility."""
        # Test that incompatible strategies are detected
        sharding_config = ShardingConfig(data_parallel_size=2, tensor_parallel_size=4)
        config = ParallelismConfig(
            mesh_shape=(2, 4),
            mesh_axis_names=("data", "model"),
            sharding_config=sharding_config,
        )
        strategies = {
            "data": DataParallelStrategy(axis_name="data", mesh_axis=0),
            "model": TensorParallelStrategy(
                axis_name="data", mesh_axis=1
            ),  # Same axis name - should conflict
        }

        with pytest.raises(ValueError, match="Conflicting strategies"):
            MultiDimensionalStrategy(strategies=strategies, config=config)

    def test_performance_characteristics(self) -> None:
        """Test that sharding configurations meet performance expectations."""
        # This will test that different sharding strategies
        # achieve expected performance characteristics
