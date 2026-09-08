"""Logical-to-physical mesh axis rules and partition-spec helpers.

``MeshRules`` maps logical dimension names (``data``, ``embed``, ``mlp``, ``heads``) to
physical mesh axis names, following the Flax NNX FSDP examples. The factories build the
two standard configurations, and the helpers turn rules into JAX sharding objects.

Example:
    ```python
    mesh = jax.make_mesh((4, 2), ("data", "model"))
    rules = fsdp_rules(data_axis="data", model_axis="model")
    partition_spec_for_names(rules, "data", "embed")  # PartitionSpec("data", "model")
    ```
"""

from __future__ import annotations

from dataclasses import dataclass

from jax.sharding import Mesh, NamedSharding, PartitionSpec


@dataclass(frozen=True, slots=True, kw_only=True)
class MeshRules:
    """Logical-to-physical mesh axis mapping for SPMD sharding.

    Attributes:
        data: Mesh axis for the batch dimension.
        embed: Mesh axis for embedding dimensions.
        mlp: Mesh axis for MLP hidden dimensions.
        heads: Mesh axis for attention-head dimensions.
    """

    data: str | None = None
    embed: str | None = None
    mlp: str | None = None
    heads: str | None = None

    def __call__(self, *keys: str) -> tuple[str | None, ...]:
        """Look up the mesh axis for each logical dimension name.

        Args:
            *keys: Logical dimension names; each must be an attribute of ``MeshRules``.

        Returns:
            One mesh axis name per key, ``None`` where the dimension is replicated.
        """
        return tuple(getattr(self, key) for key in keys)


def data_parallel_rules(data_axis: str = "data") -> MeshRules:
    """Rules for pure data-parallel training: only the batch dimension is sharded.

    Args:
        data_axis: Mesh axis carrying the batch dimension.

    Returns:
        Rules with only ``data`` mapped.
    """
    return MeshRules(data=data_axis)


def fsdp_rules(data_axis: str = "data", model_axis: str = "model") -> MeshRules:
    """Rules for fully sharded data parallelism.

    The batch dimension is sharded across ``data_axis`` and every model dimension across
    ``model_axis``.

    Args:
        data_axis: Mesh axis carrying the batch dimension.
        model_axis: Mesh axis carrying the model dimensions.

    Returns:
        Rules with ``data`` on the data axis and ``embed``, ``mlp``, ``heads`` on the model axis.
    """
    return MeshRules(data=data_axis, embed=model_axis, mlp=model_axis, heads=model_axis)


def create_named_sharding(mesh: Mesh, *axis_names: str | None) -> NamedSharding:
    """Build a ``NamedSharding`` from a mesh and one axis name per array dimension.

    Args:
        mesh: The device mesh.
        *axis_names: A mesh axis per dimension, ``None`` for a replicated dimension. No names
            at all means fully replicated.

    Returns:
        The sharding.
    """
    return NamedSharding(mesh, PartitionSpec(*axis_names))


def partition_spec_for_names(rules: MeshRules, *logical_names: str) -> PartitionSpec:
    """Map logical dimension names through the rules into a ``PartitionSpec``.

    Args:
        rules: The mapping to apply.
        *logical_names: One logical dimension name per array dimension.

    Returns:
        The partition spec with physical mesh axes.
    """
    return PartitionSpec(*rules(*logical_names))
