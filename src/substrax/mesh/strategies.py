"""Sharding configuration and the strategies that turn dimension names into partition specs.

A strategy is a pure rule: given the logical names of an array's dimensions it returns a
``PartitionSpec``, and ``shard`` places an array on a mesh accordingly. Model parameters
declared with ``flax.nnx.with_partitioning`` carry the same logical names, so one set of
rules covers both explicit placement and NNX state (see ``flax.nnx.get_named_sharding``).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import jax
from jax import Array
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from substrax.mesh.mesh import create_device_mesh


DimensionNames = tuple[str, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class ShardingConfig:
    """How many devices each parallelism dimension uses.

    Attributes:
        data_parallel_size: Devices along the data axis.
        tensor_parallel_size: Devices along the model (tensor-parallel) axis.
        pipeline_parallel_size: Devices along the pipeline axis.
        fsdp_enabled: Whether parameters are sharded across the data axis.
        fsdp_min_weight_size: Smallest leading dimension worth sharding under FSDP.
    """

    data_parallel_size: int = 1
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    fsdp_enabled: bool = False
    fsdp_min_weight_size: int = 1024

    @property
    def total_device_count(self) -> int:
        """The number of devices the configuration needs."""
        return self.data_parallel_size * self.tensor_parallel_size * self.pipeline_parallel_size

    @classmethod
    def from_device_count(cls, device_count: int) -> ShardingConfig:
        """Pick a configuration for a device count: data parallel first, then tensor parallel.

        Args:
            device_count: The number of available devices.

        Returns:
            A configuration whose ``total_device_count`` never exceeds ``device_count``.
        """
        if device_count <= 4:  # noqa: PLR2004 - the heuristic's own threshold
            return cls(data_parallel_size=device_count)
        if device_count == 8:  # noqa: PLR2004
            return cls(data_parallel_size=2, tensor_parallel_size=4)
        tensor_size = min(8, math.isqrt(device_count))
        return cls(data_parallel_size=device_count // tensor_size, tensor_parallel_size=tensor_size)


@dataclass(frozen=True, slots=True, kw_only=True)
class ParallelismConfig:
    """A sharding configuration together with the mesh topology that realises it.

    Attributes:
        mesh_shape: Devices per mesh axis.
        mesh_axis_names: The axis names, one per entry of ``mesh_shape``.
        sharding_config: The parallelism sizes the mesh must satisfy.
    """

    mesh_shape: tuple[int, ...]
    mesh_axis_names: tuple[str, ...]
    sharding_config: ShardingConfig

    @property
    def is_valid(self) -> bool:
        """Whether the mesh holds exactly the devices the sharding configuration needs."""
        return self.sharding_config.total_device_count == math.prod(self.mesh_shape)

    @classmethod
    def from_sharding_config(cls, config: ShardingConfig) -> ParallelismConfig:
        """Name a mesh axis for every parallelism dimension larger than one.

        Args:
            config: The parallelism sizes.

        Returns:
            The configuration; a single-device config gets a one-element ``data`` axis.
        """
        axes = [
            ("data", config.data_parallel_size),
            ("model", config.tensor_parallel_size),
            ("pipeline", config.pipeline_parallel_size),
        ]
        used = [(name, size) for name, size in axes if size > 1] or [("data", 1)]
        return cls(
            mesh_shape=tuple(size for _, size in used),
            mesh_axis_names=tuple(name for name, _ in used),
            sharding_config=config,
        )

    def create_mesh(self) -> Mesh:
        """Build the mesh this configuration declares over the visible devices.

        Returns:
            The mesh.
        """
        return create_device_mesh(list(zip(self.mesh_axis_names, self.mesh_shape, strict=True)))


@runtime_checkable
class ShardingStrategy(Protocol):
    """A rule mapping logical dimension names onto one mesh axis."""

    @property
    def axis_name(self) -> str:
        """The mesh axis this strategy shards along."""
        ...

    def partition_spec(self, dimension_names: DimensionNames) -> PartitionSpec:
        """Return the partition spec for an array with these logical dimension names."""
        ...

    def shard(self, array: Array, mesh: Mesh) -> Array:
        """Place an array on the mesh according to this strategy."""
        ...


def _leading_axis_spec(ndim: int, axis_name: str | None) -> PartitionSpec:
    return PartitionSpec(axis_name, *([None] * (ndim - 1)))


def _place(array: Array, mesh: Mesh, spec: PartitionSpec) -> Array:
    return jax.device_put(array, NamedSharding(mesh, spec))


@dataclass(frozen=True, slots=True, kw_only=True)
class DataParallelStrategy:
    """Shard the batch dimension, replicate everything else.

    Attributes:
        axis_name: The mesh axis carrying the batch.
        batch_dimension: The logical name of the batch dimension.
    """

    axis_name: str
    batch_dimension: str = "batch"

    def partition_spec(self, dimension_names: DimensionNames) -> PartitionSpec:
        """Shard only the batch dimension."""
        return PartitionSpec(
            *(self.axis_name if name == self.batch_dimension else None for name in dimension_names)
        )

    def shard(self, array: Array, mesh: Mesh) -> Array:
        """Place the array with its leading dimension on the data axis."""
        return _place(array, mesh, _leading_axis_spec(array.ndim, self.axis_name))


@dataclass(frozen=True, slots=True, kw_only=True)
class FSDPStrategy:
    """Shard parameters along their leading feature dimension when they are large enough.

    Attributes:
        axis_name: The mesh axis carrying the shards.
        min_weight_size: Smallest leading dimension worth sharding.
        sharded_dimensions: Logical names of leading dimensions this strategy shards.
    """

    axis_name: str
    min_weight_size: int = 1024
    sharded_dimensions: frozenset[str] = frozenset({"out_features", "features", "hidden"})

    def should_shard(self, weight: Array) -> bool:
        """Whether a weight's leading dimension reaches the sharding threshold."""
        return weight.shape[0] >= self.min_weight_size

    def partition_spec(self, dimension_names: DimensionNames) -> PartitionSpec:
        """Shard the leading dimension when it is a feature dimension."""
        leading = dimension_names[0] if dimension_names else None
        axis = self.axis_name if leading in self.sharded_dimensions else None
        return (
            _leading_axis_spec(len(dimension_names), axis) if dimension_names else PartitionSpec()
        )

    def shard(self, array: Array, mesh: Mesh) -> Array:
        """Place the array sharded on its leading dimension, or replicated if it is small."""
        axis = self.axis_name if self.should_shard(array) else None
        return _place(array, mesh, _leading_axis_spec(array.ndim, axis))


@dataclass(frozen=True, slots=True, kw_only=True)
class TensorParallelStrategy:
    """Shard one feature dimension of every tensor across the model axis.

    Attributes:
        axis_name: The mesh axis carrying the model shards.
        shard_dimension: The logical dimension to shard; ``None`` shards ``hidden`` and the
            output features of linear weights.
    """

    axis_name: str
    shard_dimension: str | None = None

    def partition_spec(self, dimension_names: DimensionNames) -> PartitionSpec:
        """Shard the configured dimension, or ``hidden`` when none is configured."""
        target = self.shard_dimension or "hidden"
        return PartitionSpec(
            *(self.axis_name if name == target else None for name in dimension_names)
        )

    @property
    def linear_weight_spec(self) -> PartitionSpec:
        """The spec for an ``(out_features, in_features)`` linear weight."""
        if self.shard_dimension == "in_features":
            return PartitionSpec(None, self.axis_name)
        return PartitionSpec(self.axis_name, None)

    @property
    def attention_qkv_spec(self) -> PartitionSpec:
        """The spec for attention query/key/value projections (output features sharded)."""
        return PartitionSpec(None, self.axis_name)

    @property
    def attention_output_spec(self) -> PartitionSpec:
        """The spec for the attention output projection (input features sharded)."""
        return PartitionSpec(self.axis_name, None)

    def shard(self, array: Array, mesh: Mesh) -> Array:
        """Place a weight matrix per ``linear_weight_spec``, other arrays on their last axis."""
        if array.ndim == 2:  # noqa: PLR2004 - a weight matrix
            return _place(array, mesh, self.linear_weight_spec)
        return _place(array, mesh, PartitionSpec(*([None] * (array.ndim - 1)), self.axis_name))


@dataclass(frozen=True, slots=True, kw_only=True)
class PipelineParallelStrategy:
    """Assign whole layers to pipeline stages; tensors inside a stage are replicated.

    Attributes:
        axis_name: The mesh axis carrying the stages.
        num_stages: The number of pipeline stages.
    """

    axis_name: str
    num_stages: int

    def assign_layers(self, num_layers: int) -> tuple[int, ...]:
        """Split layers across stages as evenly as possible, earlier stages taking the remainder.

        Args:
            num_layers: The number of layers to place.

        Returns:
            The layer count of each stage.
        """
        base, remainder = divmod(num_layers, self.num_stages)
        return tuple(base + (1 if stage < remainder else 0) for stage in range(self.num_stages))

    @property
    def forward_communication_pattern(self) -> tuple[tuple[int, int], ...]:
        """``(source, destination)`` stage pairs for the forward pass."""
        return tuple((stage, stage + 1) for stage in range(self.num_stages - 1))

    @property
    def backward_communication_pattern(self) -> tuple[tuple[int, int], ...]:
        """``(source, destination)`` stage pairs for the backward pass."""
        return tuple((stage + 1, stage) for stage in range(self.num_stages - 1))

    def partition_spec(self, dimension_names: DimensionNames) -> PartitionSpec:
        """Replicate: pipeline parallelism places layers, not tensor dimensions."""
        return PartitionSpec(*([None] * len(dimension_names)))

    def shard(self, array: Array, mesh: Mesh) -> Array:
        """Place the array replicated within its stage."""
        return _place(array, mesh, PartitionSpec(*([None] * array.ndim)))


_COMPATIBLE_ON_ONE_AXIS = frozenset({DataParallelStrategy, FSDPStrategy})
_AXIS_PRIORITY = ("model", "fsdp")


@dataclass(frozen=True, slots=True, kw_only=True)
class MultiDimensionalStrategy:
    """Combine strategies on distinct mesh axes into one partition spec per tensor.

    Attributes:
        strategies: Named strategies to combine.
        config: The parallelism configuration they realise.
    """

    strategies: Mapping[str, ShardingStrategy]
    config: ParallelismConfig

    def __post_init__(self) -> None:
        """Reject strategies that compete for one mesh axis.

        Raises:
            ValueError: If two strategies of one kind share an axis, or two kinds share an axis
                that are not the compatible data-parallel and FSDP pair.
        """
        by_axis: dict[str, list[type[object]]] = {}
        for strategy in self.strategies.values():
            by_axis.setdefault(strategy.axis_name, []).append(type(strategy))
        for axis, kinds in by_axis.items():
            duplicated = [kind.__name__ for kind, count in Counter(kinds).items() if count > 1]
            if duplicated:
                raise ValueError(
                    f"Conflicting strategies: several {duplicated[0]} on axis {axis!r}"
                )
            if len(kinds) > 1 and not set(kinds) <= _COMPATIBLE_ON_ONE_AXIS:
                names = [kind.__name__ for kind in kinds]
                raise ValueError(f"Conflicting strategies for axis {axis!r}: {names}")

    def partition_spec(self, dimension_names: DimensionNames) -> PartitionSpec:
        """Merge every strategy's spec, resolving conflicts by axis priority.

        Args:
            dimension_names: The logical names of the tensor's dimensions.

        Returns:
            The combined spec.
        """
        proposed = {
            name: strategy.partition_spec(dimension_names)
            for name, strategy in self.strategies.items()
        }
        return self.resolve_sharding_conflicts(proposed, ndim=len(dimension_names))

    @staticmethod
    def resolve_sharding_conflicts(
        proposed_specs: Mapping[str, PartitionSpec], *, ndim: int
    ) -> PartitionSpec:
        """Merge proposed specs dimension by dimension.

        Where two strategies claim one dimension, the ``model`` axis wins, then ``fsdp``,
        otherwise the first claim stands.

        Args:
            proposed_specs: Specs keyed by strategy name.
            ndim: The number of tensor dimensions.

        Returns:
            The merged spec.
        """
        merged: list[str | None] = [None] * ndim
        for spec in proposed_specs.values():
            for index, axis in enumerate(tuple(spec)):
                if axis is None:
                    continue
                current = merged[index]
                merged[index] = axis if current is None else _prefer(current, str(axis))
        return PartitionSpec(*merged)


def _prefer(existing: str, new: str) -> str:
    for winner in _AXIS_PRIORITY:
        if winner in (existing, new):
            return winner
    return existing
