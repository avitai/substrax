"""Device-mesh construction on ``jax.make_mesh``.

Functions rather than a manager class: a mesh is built once from a shape and axis names,
and JAX's own ``make_mesh`` already provides topology-aware device ordering.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import numpy as np
from jax.sharding import Mesh


MeshShape = Mapping[str, int] | Sequence[tuple[str, int]]


@dataclass(frozen=True, slots=True, kw_only=True)
class MeshInfo:
    """The shape of a mesh, as ordered ``(axis name, size)`` pairs.

    Attributes:
        total_devices: The number of devices in the mesh.
        axes: One ``(name, size)`` pair per mesh axis, in mesh order.
    """

    total_devices: int
    axes: tuple[tuple[str, int], ...]

    def axis_size(self, name: str) -> int:
        """Return the size of one mesh axis.

        Args:
            name: The axis name.

        Returns:
            The number of devices along that axis.

        Raises:
            KeyError: If the mesh has no axis of that name.
        """
        for axis_name, size in self.axes:
            if axis_name == name:
                return size
        raise KeyError(name)


def _normalise_shape(shape: MeshShape) -> tuple[tuple[str, ...], tuple[int, ...]]:
    pairs = list(shape.items()) if isinstance(shape, Mapping) else list(shape)
    return tuple(name for name, _ in pairs), tuple(size for _, size in pairs)


def create_device_mesh(shape: MeshShape, devices: Sequence[Any] | None = None) -> Mesh:
    """Create a device mesh with the given axis names and sizes.

    Args:
        shape: The mesh shape, as a mapping from axis name to size or as ordered
            ``(name, size)`` pairs.
        devices: The devices to place, in order. ``None`` lets ``jax.make_mesh`` choose a
            topology-aware ordering over every visible device.

    Returns:
        The mesh.

    Raises:
        ValueError: If fewer devices are available than the shape needs.
    """
    axis_names, sizes = _normalise_shape(shape)
    if devices is None:
        return jax.make_mesh(sizes, axis_names)
    needed = math.prod(sizes)
    if len(devices) < needed:
        raise ValueError(
            f"Not enough devices: the mesh needs {needed} but {len(devices)} are available."
        )
    return Mesh(np.array(list(devices[:needed])).reshape(sizes), axis_names=axis_names)


def _visible_devices(needed: int, purpose: str) -> list[Any]:
    devices = jax.devices()
    if len(devices) < needed:
        raise ValueError(
            f"Not enough devices: {purpose} needs {needed} but {len(devices)} are available."
        )
    return list(devices[:needed])


def create_data_parallel_mesh(num_devices: int | None = None) -> Mesh:
    """Create a one-axis ``data`` mesh; a ``ValueError`` names any device shortfall.

    Args:
        num_devices: How many devices to use; ``None`` uses every visible device.

    Returns:
        The mesh.
    """
    count = len(jax.devices()) if num_devices is None else num_devices
    return create_device_mesh([("data", count)], _visible_devices(count, "data parallelism"))


def create_model_parallel_mesh(num_devices: int) -> Mesh:
    """Create a one-axis ``model`` mesh; a ``ValueError`` names any device shortfall.

    Args:
        num_devices: How many devices to use.

    Returns:
        The mesh.
    """
    devices = _visible_devices(num_devices, "model parallelism")
    return create_device_mesh([("model", num_devices)], devices)


def create_hybrid_mesh(data_parallel_size: int, model_parallel_size: int) -> Mesh:
    """Create a two-axis ``(data, model)`` mesh; a ``ValueError`` names any device shortfall.

    Args:
        data_parallel_size: Devices along the data axis.
        model_parallel_size: Devices along the model axis.

    Returns:
        The mesh.
    """
    needed = data_parallel_size * model_parallel_size
    devices = _visible_devices(needed, "hybrid parallelism")
    return create_device_mesh(
        [("data", data_parallel_size), ("model", model_parallel_size)], devices
    )


def mesh_info(mesh: Mesh) -> MeshInfo:
    """Describe a mesh's axes and device count.

    Args:
        mesh: The mesh to describe.

    Returns:
        Its shape as ordered ``(name, size)`` pairs.
    """
    axes = tuple(zip(mesh.axis_names, mesh.devices.shape, strict=True))
    return MeshInfo(total_devices=int(mesh.devices.size), axes=axes)
