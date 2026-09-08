"""Tests for device placement and the batch-size recommendation table."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, strategies as st
from jax.sharding import Mesh, NamedSharding, PartitionSpec, SingleDeviceSharding

from substrax.devices import (
    batch_size_recommendation,
    BatchSizeRecommendation,
    BatchSizeVerdict,
    detect_hardware_type,
    DeviceInfo,
    DeviceKind,
    distribute_batch,
    HardwareType,
    place_on_device,
    replicate_across_devices,
    shard_batch_dim,
    validate_batch_size,
)


@pytest.fixture
def sample_data() -> dict[str, jax.Array]:
    return {"images": jnp.ones((4, 28, 28, 3)), "labels": jnp.zeros((4,), dtype=jnp.int32)}


@pytest.fixture
def single_device_mesh() -> Mesh:
    return Mesh(np.array(jax.devices()[:1]), axis_names=("data",))


class TestHardwareType:
    @pytest.mark.parametrize(
        ("platform", "device_kind", "expected"),
        [
            ("tpu", "TPU v5 lite", HardwareType.TPU_V5E),
            ("tpu", "TPU v5p", HardwareType.TPU_V5P),
            ("tpu", "TPU v4", HardwareType.TPU_V4),
            ("gpu", "NVIDIA H100 80GB HBM3", HardwareType.H100),
            ("cuda", "NVIDIA A100-SXM4-80GB", HardwareType.A100),
            ("gpu", "Tesla V100-SXM2-16GB", HardwareType.V100),
            ("cpu", "cpu", HardwareType.CPU),
        ],
    )
    def test_known_devices_are_classified(
        self, platform: str, device_kind: str, expected: HardwareType
    ) -> None:
        assert HardwareType.from_device(platform, device_kind) is expected

    @pytest.mark.parametrize(
        ("platform", "device_kind"),
        [("gpu", "NVIDIA B200"), ("tpu", "TPU v7"), ("metal", "Apple M3")],
    )
    def test_unrecognised_accelerators_are_unknown_not_guessed(
        self, platform: str, device_kind: str
    ) -> None:
        assert HardwareType.from_device(platform, device_kind) is HardwareType.UNKNOWN

    def test_positive_control_cpu_process_detects_cpu(self) -> None:
        assert detect_hardware_type() is HardwareType.CPU

    def test_detect_reads_the_first_device_of_a_snapshot(self) -> None:
        info = DeviceInfo(
            platform="gpu",
            kind=DeviceKind.GPU,
            count=2,
            device_kinds=("NVIDIA H100", "NVIDIA H100"),
        )
        assert detect_hardware_type(info) is HardwareType.H100


class TestBatchSizeRecommendation:
    @given(st.sampled_from(list(HardwareType)))
    def test_every_hardware_type_has_an_ordered_recommendation(self, kind: HardwareType) -> None:
        rec = batch_size_recommendation(kind)
        assert isinstance(rec, BatchSizeRecommendation)
        assert 0 < rec.min_batch_size <= rec.critical_batch_size <= rec.optimal_batch_size

    def test_h100_and_tpu_v5e_carry_the_jax_guide_numbers(self) -> None:
        assert batch_size_recommendation(HardwareType.H100).critical_batch_size == 298
        assert batch_size_recommendation(HardwareType.TPU_V5E).critical_batch_size == 240

    def test_defaults_to_the_detected_hardware(self) -> None:
        assert batch_size_recommendation() == batch_size_recommendation(HardwareType.CPU)


class TestValidateBatchSize:
    def test_optimal(self) -> None:
        rec = batch_size_recommendation(HardwareType.H100)
        verdict = validate_batch_size(rec.optimal_batch_size, HardwareType.H100)
        assert verdict == BatchSizeVerdict(is_valid=True, message=verdict.message)
        assert "optimal" in verdict.message

    def test_below_minimum_is_invalid(self) -> None:
        verdict = validate_batch_size(8, HardwareType.H100)
        assert verdict.is_valid is False
        assert "below minimum" in verdict.message

    def test_below_critical_warns_unless_asked_not_to(self) -> None:
        warned = validate_batch_size(100, HardwareType.H100)
        quiet = validate_batch_size(100, HardwareType.H100, warn_suboptimal=False)
        assert warned.is_valid
        assert "critical" in warned.message
        assert quiet.is_valid
        assert "acceptable" in quiet.message


class TestPlacement:
    def test_place_on_device_gives_a_single_device_sharding(self) -> None:
        result = place_on_device(jnp.ones((4, 8)), jax.devices()[0])
        assert isinstance(result.sharding, SingleDeviceSharding)

    def test_place_on_device_defaults_to_the_first_device(
        self, sample_data: dict[str, jax.Array]
    ) -> None:
        result = place_on_device(sample_data)
        assert result["images"].dtype == jnp.float32
        assert result["labels"].dtype == jnp.int32
        np.testing.assert_array_equal(np.asarray(result["labels"]), np.zeros((4,)))

    def test_distribute_batch_applies_the_sharding(self, single_device_mesh: Mesh) -> None:
        sharding = NamedSharding(single_device_mesh, PartitionSpec("data", None))
        result = distribute_batch({"x": jnp.ones((4, 8))}, sharding)
        assert result["x"].sharding == sharding

    def test_replicate_across_devices(self, sample_data: dict[str, jax.Array]) -> None:
        result = replicate_across_devices(sample_data)
        sharding = result["images"].sharding
        assert isinstance(sharding, NamedSharding)
        assert sharding.spec == PartitionSpec(None)

    def test_shard_batch_dim_places_the_batch_axis_and_replicates_scalars(
        self, single_device_mesh: Mesh
    ) -> None:
        data = {"x": jnp.ones((4, 8, 3)), "scalar": jnp.array(1.0)}
        result = shard_batch_dim(data, single_device_mesh)
        x_sharding = result["x"].sharding
        assert isinstance(x_sharding, NamedSharding)
        assert x_sharding.spec == PartitionSpec("data", None, None)
        assert result["scalar"].shape == ()

    def test_shard_batch_dim_custom_axis(self, single_device_mesh: Mesh) -> None:
        result = shard_batch_dim(jnp.ones((8, 4, 3)), single_device_mesh, batch_axis=1)
        sharding = result.sharding
        assert isinstance(sharding, NamedSharding)
        assert sharding.spec == PartitionSpec(None, "data", None)
