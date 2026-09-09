"""Tests for DeviceMeshManager - device mesh management for JAX distributed training."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

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


def test_create_device_mesh_axes_are_auto_by_default() -> None:
    """The mesh leaves sharding inference to XLA, as the data-parallel helpers assume."""
    mesh = DeviceMeshManager.create_device_mesh({"data": 1})
    manual = DeviceMeshManager.create_device_mesh([("data", 1)], jax.devices()[:1])

    assert mesh.axis_types == (jax.sharding.AxisType.Auto,)
    assert manual.axis_types == (jax.sharding.AxisType.Auto,)


def test_create_device_mesh_honours_requested_axis_types() -> None:
    """Explicit sharding-in-types stays available on request, on both paths."""
    explicit = (jax.sharding.AxisType.Explicit,)
    mesh = DeviceMeshManager.create_device_mesh({"data": 1}, axis_types=explicit)
    manual = DeviceMeshManager.create_device_mesh(
        [("data", 1)], jax.devices()[:1], axis_types=explicit
    )

    assert mesh.axis_types == explicit
    assert manual.axis_types == explicit


def test_create_device_mesh_rejects_mismatched_axis_types() -> None:
    with pytest.raises(ValueError, match="axis_types"):
        DeviceMeshManager.create_device_mesh(
            {"data": 1, "model": 1}, axis_types=(jax.sharding.AxisType.Auto,)
        )


def _grad_through_linear(mesh: jax.sharding.Mesh) -> jax.Array:
    """Value-and-grad of a linear layer on a batch sharded along ``data``."""
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
    batch = jax.device_put(jnp.ones((8, 4)), sharding)
    model = nnx.Linear(4, 1, rngs=nnx.Rngs(0))

    @nnx.jit
    def step(module: nnx.Linear, x: jax.Array) -> jax.Array:
        return nnx.value_and_grad(lambda m: jnp.mean(m(x) ** 2))(module)[0]

    return step(model, batch)


def test_default_mesh_differentiates_through_a_batch_sharded_linear() -> None:
    """Data parallelism must not require out_sharding on every contraction."""
    loss = _grad_through_linear(DeviceMeshManager.create_device_mesh({"data": 1}))

    assert jnp.isfinite(loss)


def test_explicit_axes_are_the_control_for_that_guarantee() -> None:
    """Under explicit axis types the same step fails, which is what the default avoids."""
    explicit = DeviceMeshManager.create_device_mesh(
        {"data": 1}, axis_types=(jax.sharding.AxisType.Explicit,)
    )

    # jax exposes no public name for ShardingTypeError; the message is the contract.
    with pytest.raises(Exception, match="Contracting dimensions are sharded"):
        _grad_through_linear(explicit)
