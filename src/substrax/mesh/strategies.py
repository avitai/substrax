"""Sharding strategies and mesh-aware configuration for scaling experiments.

This module exposes the retained strategy classes and configuration dataclasses
used by the scaling package. It does not provide a universal parameter-name to
`PartitionSpec` inference layer; callers compose sharding through the concrete
strategy APIs instead.
"""

import math
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jax
from jax import Array
from jax.sharding import Mesh, NamedSharding, PartitionSpec


# Device counts the from_device_count heuristic keys on.
_DATA_PARALLEL_ONLY_MAX_DEVICES = 4
_EIGHT_DEVICES = 8
_MAX_TENSOR_PARALLEL_SIZE = 8
_WEIGHT_MATRIX_NDIM = 2
# Strategy kinds that may share one mesh axis.
_COMPATIBLE_ON_ONE_AXIS = frozenset({"DataParallelStrategy", "FSDPStrategy"})
# Mesh axes that win a per-dimension conflict, highest priority first.
_AXIS_PRIORITY = ("model", "fsdp")


@dataclass
class ShardingConfig:
    """Configuration for multi-dimensional parallelism setup.

    Defines the parallelism dimensions and FSDP settings for a model.
    """

    data_parallel_size: int = 1
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    fsdp_enabled: bool = False
    fsdp_min_weight_size: int = 1024

    def get_total_device_count(self) -> int:
        """Calculate total devices needed for this configuration."""
        return self.data_parallel_size * self.tensor_parallel_size * self.pipeline_parallel_size

    @classmethod
    def from_device_count(cls, device_count: int) -> "ShardingConfig":
        """Create optimal sharding config for given device count.

        Uses heuristics to balance different parallelism dimensions.
        """
        # Simple heuristic: prioritize data parallel, then tensor parallel
        if device_count <= _DATA_PARALLEL_ONLY_MAX_DEVICES:
            return cls(data_parallel_size=device_count)
        if device_count == _EIGHT_DEVICES:
            return cls(data_parallel_size=2, tensor_parallel_size=4)
        # For larger counts, use balanced approach
        tensor_size = min(_MAX_TENSOR_PARALLEL_SIZE, int(math.sqrt(device_count)))
        return cls(data_parallel_size=device_count // tensor_size, tensor_parallel_size=tensor_size)


@dataclass
class ParallelismConfig:
    """Complete parallelism configuration including mesh topology.

    Combines sharding configuration with device mesh setup.
    """

    mesh_shape: tuple[int, ...]
    mesh_axis_names: tuple[str, ...]
    sharding_config: ShardingConfig

    def is_valid(self) -> bool:
        """Validate that mesh shape matches sharding configuration."""
        expected_devices = self.sharding_config.get_total_device_count()
        actual_devices = math.prod(self.mesh_shape)
        return expected_devices == actual_devices

    @classmethod
    def from_sharding_config(cls, config: ShardingConfig) -> "ParallelismConfig":
        """Create parallelism config from sharding configuration."""
        # Build mesh shape from sharding config
        mesh_shape = []
        axis_names = []

        if config.data_parallel_size > 1:
            mesh_shape.append(config.data_parallel_size)
            axis_names.append("data")

        if config.tensor_parallel_size > 1:
            mesh_shape.append(config.tensor_parallel_size)
            axis_names.append("model")

        if config.pipeline_parallel_size > 1:
            mesh_shape.append(config.pipeline_parallel_size)
            axis_names.append("pipeline")

        # Default to data parallel if no dimensions specified
        if not mesh_shape:
            mesh_shape = [1]
            axis_names = ["data"]

        return cls(
            mesh_shape=tuple(mesh_shape), mesh_axis_names=tuple(axis_names), sharding_config=config
        )


class ShardingStrategy(ABC):
    """Abstract base class for sharding strategies.

    Defines the interface that all sharding strategies must implement
    for consistent handling of different parallelism types.
    """

    def __init__(self, axis_name: str, mesh_axis: int) -> None:
        """Initialize sharding strategy.

        Args:
            axis_name: Name of the mesh axis for this strategy
            mesh_axis: Index of the mesh axis
        """
        self.axis_name = axis_name
        self.mesh_axis = mesh_axis

    @abstractmethod
    def get_partition_spec(self, tensor_shape: tuple[str, ...]) -> PartitionSpec:
        """Get partition specification for a tensor with given shape names.

        Args:
            tensor_shape: Tuple of dimension names for the tensor

        Returns:
            PartitionSpec defining how to shard the tensor
        """

    @abstractmethod
    def apply_sharding(self, array: Array, mesh: Mesh) -> Array:
        """Apply sharding to an array using the given mesh.

        Args:
            array: JAX array to shard
            mesh: Device mesh for sharding

        Returns:
            Sharded array
        """

    def get_sharding_constraints(self) -> dict[str, Any]:
        """Get sharding constraints for this strategy.

        Returns:
            Dictionary of sharding constraints
        """
        return {"axis_name": self.axis_name, "mesh_axis": self.mesh_axis}


class DataParallelStrategy(ShardingStrategy):
    """Data parallel sharding strategy.

    Shards the batch dimension across devices while replicating
    model parameters and computation.
    """

    def get_partition_spec(self, tensor_shape: tuple[str, ...]) -> PartitionSpec:
        """Get partition spec for data parallel sharding.

        Only shards the batch dimension, leaves others replicated.
        """
        specs: list[str | None] = []
        for dim_name in tensor_shape:
            if dim_name == "batch":
                specs.append(self.axis_name)
            else:
                specs.append(None)
        return PartitionSpec(*specs)

    def apply_sharding(self, array: Array, mesh: Mesh) -> Array:
        """Apply data parallel sharding to array."""
        # Create named sharding for data parallel
        none_specs = [None] * (array.ndim - 1)
        partition_spec = PartitionSpec(self.axis_name, *none_specs)
        sharding = NamedSharding(mesh, partition_spec)

        # Apply sharding
        return jax.device_put(array, sharding)


class FSDPStrategy(ShardingStrategy):
    """Fully Sharded Data Parallel strategy.

    Shards model parameters across devices to reduce memory usage
    while maintaining training efficiency.
    """

    def __init__(self, axis_name: str, mesh_axis: int, min_weight_size: int = 1024) -> None:
        """Initialize FSDP strategy.

        Args:
            axis_name: Name of the mesh axis
            mesh_axis: Index of the mesh axis
            min_weight_size: Minimum first dimension size to enable sharding
        """
        super().__init__(axis_name, mesh_axis)
        self.min_weight_size = min_weight_size

    def should_shard_weight(self, weight: Array) -> bool:
        """Determine if a weight should be sharded based on its size.

        Args:
            weight: Weight array to check

        Returns:
            True if weight should be sharded, False otherwise
        """
        # Check the first dimension size (what FSDP actually shards)
        return weight.shape[0] >= self.min_weight_size

    def get_partition_spec(self, tensor_shape: tuple[str, ...]) -> PartitionSpec:
        """Get partition spec for FSDP sharding.

        Shards along the first dimension of weight tensors.
        """
        specs: list[str | None] = []
        for i, dim_name in enumerate(tensor_shape):
            if i == 0 and dim_name in ["out_features", "features", "hidden"]:
                specs.append(self.axis_name)
            else:
                specs.append(None)
        return PartitionSpec(*specs)

    def get_gradient_partition_spec(self, tensor_shape: tuple[str, ...]) -> PartitionSpec:
        """Get partition spec for gradient sharding (same as weights)."""
        return self.get_partition_spec(tensor_shape)

    def apply_sharding(self, array: Array, mesh: Mesh) -> Array:
        """Apply FSDP sharding to array."""
        if not self.should_shard_weight(array):
            # Replicate small weights
            partition_spec = PartitionSpec(*([None] * array.ndim))
        else:
            # Shard along first dimension
            none_specs = [None] * (array.ndim - 1)
            partition_spec = PartitionSpec(self.axis_name, *none_specs)

        sharding = NamedSharding(mesh, partition_spec)
        return jax.device_put(array, sharding)


class TensorParallelStrategy(ShardingStrategy):
    """Tensor parallel sharding strategy.

    Shards model computation across devices by splitting tensors
    along specific dimensions (typically features).
    """

    def __init__(self, axis_name: str, mesh_axis: int, shard_dimension: str | None = None) -> None:
        """Initialize tensor parallel strategy.

        Args:
            axis_name: Name of the mesh axis
            mesh_axis: Index of the mesh axis
            shard_dimension: Preferred dimension to shard
                ('in_features' or 'out_features')
        """
        super().__init__(axis_name, mesh_axis)
        self.shard_dimension = shard_dimension

    def get_partition_spec(self, tensor_shape: tuple[str, ...]) -> PartitionSpec:
        """Get partition spec for tensor parallel sharding."""
        specs: list[str | None] = []
        for dim_name in tensor_shape:
            if (self.shard_dimension and dim_name == self.shard_dimension) or (
                not self.shard_dimension and dim_name == "hidden"
            ):
                specs.append(self.axis_name)
            else:
                specs.append(None)
        return PartitionSpec(*specs)

    def get_linear_weight_spec(self) -> PartitionSpec:
        """Get partition spec for linear layer weights."""
        if self.shard_dimension == "in_features":
            # (out, in) -> shard in
            return PartitionSpec(None, self.axis_name)
        # (out, in) -> shard out
        return PartitionSpec(self.axis_name, None)

    def get_attention_qkv_spec(self) -> PartitionSpec:
        """Get partition spec for attention QKV projections."""
        return PartitionSpec(None, self.axis_name)  # Shard output features

    def get_attention_output_spec(self) -> PartitionSpec:
        """Get partition spec for attention output projection."""
        return PartitionSpec(self.axis_name, None)  # Shard input features

    def apply_sharding(self, array: Array, mesh: Mesh) -> Array:
        """Apply tensor parallel sharding to array."""
        # Default to sharding the last dimension if not specified
        if array.ndim == _WEIGHT_MATRIX_NDIM:
            if self.shard_dimension == "in_features":
                partition_spec = PartitionSpec(None, self.axis_name)
            else:
                partition_spec = PartitionSpec(self.axis_name, None)
        else:
            # For other tensors, shard the last dimension
            specs: list[str | None] = [None] * array.ndim
            specs[-1] = self.axis_name
            partition_spec = PartitionSpec(*specs)

        sharding = NamedSharding(mesh, partition_spec)
        return jax.device_put(array, sharding)


class PipelineParallelStrategy(ShardingStrategy):
    """Pipeline parallel sharding strategy.

    Distributes model layers across devices to enable pipeline parallelism
    for very large models that don't fit on single devices.
    """

    def __init__(self, axis_name: str, mesh_axis: int, num_stages: int) -> None:
        """Initialize pipeline parallel strategy.

        Args:
            axis_name: Name of the mesh axis
            mesh_axis: Index of the mesh axis
            num_stages: Number of pipeline stages
        """
        super().__init__(axis_name, mesh_axis)
        self.num_stages = num_stages

    def assign_layers_to_stages(self, num_layers: int) -> list[int]:
        """Assign layers to pipeline stages.

        Args:
            num_layers: Total number of layers in the model

        Returns:
            list of layer counts per stage
        """
        layers_per_stage = num_layers // self.num_stages
        remainder = num_layers % self.num_stages

        assignments = [layers_per_stage] * self.num_stages

        # Distribute remainder layers
        for i in range(remainder):
            assignments[i] += 1

        return assignments

    def get_partition_spec(self, tensor_shape: tuple[str, ...]) -> PartitionSpec:
        """Get partition spec for pipeline parallel sharding.

        Pipeline parallelism doesn't shard individual tensors,
        but rather assigns entire layers to different devices.
        """
        # No sharding of individual tensors in pipeline parallelism
        return PartitionSpec(*([None] * len(tensor_shape)))

    def get_forward_communication_pattern(self) -> list[tuple[int, int]]:
        """Get communication pattern for forward pass.

        Returns:
            list of (source_stage, dest_stage) pairs
        """
        return [(i, i + 1) for i in range(self.num_stages - 1)]

    def get_backward_communication_pattern(self) -> list[tuple[int, int]]:
        """Get communication pattern for backward pass.

        Returns:
            list of (source_stage, dest_stage) pairs
        """
        return [(i + 1, i) for i in range(self.num_stages - 1)]

    def apply_sharding(self, array: Array, mesh: Mesh) -> Array:
        """Apply pipeline parallel sharding to array.

        Pipeline parallelism handles layer assignment rather than
        tensor sharding.
        """
        # Replicate tensors within each pipeline stage
        partition_spec = PartitionSpec(*([None] * array.ndim))
        sharding = NamedSharding(mesh, partition_spec)
        return jax.device_put(array, sharding)


class MultiDimensionalStrategy:
    """Multi-dimensional parallelism strategy combining multiple approaches.

    Combines different sharding strategies (data, tensor, FSDP, pipeline)
    to achieve optimal performance for large-scale training.
    """

    def __init__(
        self, strategies: Mapping[str, ShardingStrategy], config: ParallelismConfig
    ) -> None:
        """Initialize multi-dimensional strategy.

        Args:
            strategies: Dictionary mapping strategy names to strategy instances
            config: Sharding configuration for the multi-dimensional strategy
        """
        self.strategies = strategies
        self.config = config
        self._validate_strategies()

    def _validate_strategies(self) -> None:
        """Validate that strategies sharing a mesh axis are compatible."""
        by_axis: dict[str, list[str]] = {}
        for strategy in self.strategies.values():
            by_axis.setdefault(strategy.axis_name, []).append(type(strategy).__name__)
        for axis_name, strategy_types in by_axis.items():
            if len(strategy_types) > 1:
                _check_axis_compatibility(axis_name, strategy_types)

    def get_combined_partition_spec(
        self, tensor_name: str, tensor_shape: tuple[str, ...]
    ) -> PartitionSpec:
        """Get combined partition spec from all strategies.

        Args:
            tensor_name: Name/type of the tensor
            tensor_shape: Shape dimension names of the tensor

        Returns:
            Combined PartitionSpec
        """
        # Start with all None specs
        combined_specs: list[str | None] = [None] * len(tensor_shape)

        # Apply each strategy, merging non-None specs and resolving conflicts
        for strategy in self.strategies.values():
            strategy_spec = strategy.get_partition_spec(tensor_shape)
            for i, spec in enumerate(strategy_spec):
                if spec is not None:
                    combined_specs[i] = _resolve_spec_conflict(combined_specs[i], spec)

        del tensor_name  # The merge is shape-driven; the name is kept for call-site clarity.
        return PartitionSpec(*combined_specs)

    def resolve_sharding_conflicts(
        self, tensor_name: str, proposed_specs: dict[str, PartitionSpec]
    ) -> PartitionSpec:
        """Resolve conflicts between multiple proposed partition specs.

        Args:
            tensor_name: Name of the tensor
            proposed_specs: Dictionary of strategy names to proposed specs

        Returns:
            Resolved PartitionSpec
        """
        if not proposed_specs:
            return PartitionSpec()

        # Start with first spec
        first_spec = next(iter(proposed_specs.values()))
        result_specs = list(first_spec)

        # Merge other specs with conflict resolution
        for spec in proposed_specs.values():
            for i, spec_value in enumerate(spec):
                if spec_value is not None and result_specs[i] != spec_value:
                    result_specs[i] = _resolve_spec_conflict(result_specs[i], spec_value)

        del tensor_name  # The merge is shape-driven; the name is kept for call-site clarity.
        return PartitionSpec(*result_specs)


def _check_axis_compatibility(axis_name: str, strategy_types: list[str]) -> None:
    """Reject strategies that cannot share ``axis_name``.

    Args:
        axis_name: The mesh axis the strategies share.
        strategy_types: The class names of the strategies on that axis.

    Raises:
        ValueError: If the kinds are not the compatible data-parallel/FSDP pair, or if one
            kind appears more than once.
    """
    if not set(strategy_types) <= _COMPATIBLE_ON_ONE_AXIS:
        raise ValueError(f"Conflicting strategies for axis {axis_name}: {strategy_types}")
    for strategy_type, type_count in Counter(strategy_types).items():
        if type_count > 1:
            raise ValueError(
                f"Conflicting strategies: multiple {strategy_type} instances using axis {axis_name}"
            )


def _resolve_spec_conflict(existing_spec: str | None, new_spec: str) -> str:
    """Pick the axis for one dimension two strategies both claim.

    Tensor parallelism (``model``) wins, then FSDP (``fsdp``); otherwise the existing claim
    stands.

    Args:
        existing_spec: The axis already assigned, or ``None``.
        new_spec: The competing axis.

    Returns:
        The axis to use.
    """
    if existing_spec is None:
        return new_spec
    for winner in _AXIS_PRIORITY:
        if winner in (existing_spec, new_spec):
            return winner
    return existing_spec
