"""Tests for DeviceMeshManager - device mesh management for JAX distributed training."""

import jax
import pytest

from substrax.mesh import DeviceMeshManager


def test_create_device_mesh_dict_specification() -> None:
    mesh = DeviceMeshManager.create_device_mesh({"data": 1})
    assert mesh.devices.shape == (1,)
    assert mesh.axis_names == ("data",)
    assert mesh.devices.size == 1


def test_create_device_mesh_list_specification() -> None:
    mesh = DeviceMeshManager.create_device_mesh([("batch", 1)])
    assert mesh.devices.shape == (1,)
    assert mesh.axis_names == ("batch",)


def test_create_device_mesh_explicit_devices() -> None:
    devices = jax.devices()
    mesh = DeviceMeshManager.create_device_mesh([("data", 1)], devices=devices)
    assert mesh.devices.shape == (1,)
    assert mesh.devices[0] == devices[0]


def test_create_device_mesh_multiple_axes() -> None:
    mesh = DeviceMeshManager.create_device_mesh([("data", 1), ("model", 1)])
    assert mesh.devices.shape == (1, 1)
    assert mesh.axis_names == ("data", "model")


def test_create_device_mesh_insufficient_devices_error() -> None:
    with pytest.raises(ValueError, match="100"):
        DeviceMeshManager.create_device_mesh([("data", 100)])


def test_create_data_parallel_mesh_default() -> None:
    mesh = DeviceMeshManager.create_data_parallel_mesh()
    assert mesh.devices.shape == (len(jax.devices()),)
    assert mesh.axis_names == ("data",)


def test_create_data_parallel_mesh_single_device() -> None:
    mesh = DeviceMeshManager.create_data_parallel_mesh(num_devices=1)
    assert mesh.devices.shape == (1,)
    assert mesh.axis_names == ("data",)


def test_create_model_parallel_mesh_single_device() -> None:
    mesh = DeviceMeshManager.create_model_parallel_mesh(num_devices=1)
    assert mesh.devices.shape == (1,)
    assert mesh.axis_names == ("model",)


def test_create_model_parallel_mesh_insufficient_devices_error() -> None:
    with pytest.raises(ValueError, match=r"Not enough devices.*100"):
        DeviceMeshManager.create_model_parallel_mesh(num_devices=100)


def test_create_hybrid_mesh_single_device() -> None:
    mesh = DeviceMeshManager.create_hybrid_mesh(data_parallel_size=1, model_parallel_size=1)
    assert mesh.devices.shape == (1, 1)
    assert mesh.axis_names == ("data", "model")
    assert mesh.devices.size == 1


def test_create_hybrid_mesh_insufficient_devices_error() -> None:
    with pytest.raises(ValueError, match=r"Not enough devices.*100"):
        DeviceMeshManager.create_hybrid_mesh(data_parallel_size=10, model_parallel_size=10)


def test_get_mesh_info_single_axis() -> None:
    mesh = DeviceMeshManager.create_device_mesh([("data", 1)])
    info = DeviceMeshManager.get_mesh_info(mesh)
    assert info["total_devices"] == 1
    axes = info["axes"]
    assert isinstance(axes, dict)
    assert axes["data"] == 1


def test_get_mesh_info_multiple_axes() -> None:
    mesh = DeviceMeshManager.create_hybrid_mesh(data_parallel_size=1, model_parallel_size=1)
    info = DeviceMeshManager.get_mesh_info(mesh)
    assert info["total_devices"] == 1
    axes = info["axes"]
    assert isinstance(axes, dict)
    assert axes == {"data": 1, "model": 1}
