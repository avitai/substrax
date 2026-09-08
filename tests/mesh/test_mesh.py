"""Tests for device-mesh construction."""

from __future__ import annotations

import jax
import pytest

from substrax.mesh import (
    create_data_parallel_mesh,
    create_device_mesh,
    create_hybrid_mesh,
    create_model_parallel_mesh,
    mesh_info,
    MeshInfo,
)


class TestCreateDeviceMesh:
    def test_mapping_shape(self) -> None:
        mesh = create_device_mesh({"data": 1})
        assert mesh.devices.shape == (1,)
        assert mesh.axis_names == ("data",)

    def test_sequence_shape(self) -> None:
        mesh = create_device_mesh([("batch", 1)])
        assert mesh.axis_names == ("batch",)

    def test_explicit_devices_are_placed_in_order(self) -> None:
        devices = jax.devices()
        mesh = create_device_mesh([("data", 1)], devices=devices)
        assert mesh.devices[0] == devices[0]

    def test_multiple_axes(self) -> None:
        mesh = create_device_mesh([("data", 1), ("model", 1)])
        assert mesh.devices.shape == (1, 1)
        assert mesh.axis_names == ("data", "model")

    def test_too_few_devices_raises(self) -> None:
        with pytest.raises(ValueError, match="100"):
            create_device_mesh([("data", 100)], devices=jax.devices())


class TestFactories:
    def test_data_parallel_mesh_uses_every_device_by_default(self) -> None:
        mesh = create_data_parallel_mesh()
        assert mesh.devices.shape == (len(jax.devices()),)
        assert mesh.axis_names == ("data",)

    def test_data_parallel_mesh_subset(self) -> None:
        assert create_data_parallel_mesh(num_devices=1).devices.shape == (1,)

    def test_model_parallel_mesh(self) -> None:
        assert create_model_parallel_mesh(num_devices=1).axis_names == ("model",)

    def test_model_parallel_mesh_too_few_devices_raises(self) -> None:
        with pytest.raises(ValueError, match="100"):
            create_model_parallel_mesh(num_devices=100)

    def test_hybrid_mesh(self) -> None:
        mesh = create_hybrid_mesh(data_parallel_size=1, model_parallel_size=1)
        assert mesh.devices.shape == (1, 1)
        assert mesh.axis_names == ("data", "model")

    def test_hybrid_mesh_too_few_devices_raises(self) -> None:
        with pytest.raises(ValueError, match="100"):
            create_hybrid_mesh(data_parallel_size=10, model_parallel_size=10)


class TestMeshInfo:
    def test_single_axis(self) -> None:
        info = mesh_info(create_device_mesh([("data", 1)]))
        assert info == MeshInfo(total_devices=1, axes=(("data", 1),))
        assert info.axis_size("data") == 1

    def test_multiple_axes(self) -> None:
        info = mesh_info(create_hybrid_mesh(data_parallel_size=1, model_parallel_size=1))
        assert info.total_devices == 1
        assert info.axes == (("data", 1), ("model", 1))

    def test_unknown_axis_raises(self) -> None:
        info = mesh_info(create_device_mesh([("data", 1)]))
        with pytest.raises(KeyError):
            info.axis_size("model")
