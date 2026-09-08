"""Tests for mesh rules and partition-spec helpers."""

from __future__ import annotations

import dataclasses

import jax
import numpy as np
import pytest
from hypothesis import given, strategies as st
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from substrax.mesh import (
    create_named_sharding,
    data_parallel_rules,
    fsdp_rules,
    MeshRules,
    partition_spec_for_names,
)


_LOGICAL_NAMES = st.sampled_from(["data", "embed", "mlp", "heads"])


def _single_device_mesh(*axis_names: str) -> Mesh:
    devices = np.array(jax.devices()[:1]).reshape((1,) * len(axis_names))
    return Mesh(devices, axis_names=axis_names)


class TestMeshRules:
    def test_defaults_map_nothing(self) -> None:
        rules = MeshRules()
        assert rules("data", "embed", "mlp", "heads") == (None, None, None, None)

    def test_call_returns_the_axis_for_each_key(self) -> None:
        rules = MeshRules(data="dp", embed="mp", mlp="mp")
        assert rules("data", "embed", "mlp") == ("dp", "mp", "mp")

    def test_unmapped_keys_return_none(self) -> None:
        rules = MeshRules(data="dp")
        assert rules("data", "heads") == ("dp", None)

    def test_rules_are_frozen(self) -> None:
        rules = MeshRules(data="dp")
        with pytest.raises(dataclasses.FrozenInstanceError):
            rules.data = "mp"  # type: ignore[misc]


class TestFactories:
    def test_data_parallel_rules_map_only_the_data_axis(self) -> None:
        rules = data_parallel_rules(data_axis="batch")
        assert rules == MeshRules(data="batch")

    def test_fsdp_rules_map_model_dimensions_to_the_model_axis(self) -> None:
        rules = fsdp_rules(data_axis="dp", model_axis="mp")
        assert rules == MeshRules(data="dp", embed="mp", mlp="mp", heads="mp")


class TestCreateNamedSharding:
    def test_single_axis(self) -> None:
        sharding = create_named_sharding(_single_device_mesh("data"), "data")
        assert isinstance(sharding, NamedSharding)
        assert sharding.spec == PartitionSpec("data")

    def test_no_axes_is_fully_replicated(self) -> None:
        assert create_named_sharding(_single_device_mesh("data")).spec == PartitionSpec()

    def test_none_marks_a_replicated_dimension(self) -> None:
        sharding = create_named_sharding(_single_device_mesh("data", "model"), "data", None)
        assert sharding.spec == PartitionSpec("data", None)


class TestPartitionSpecForNames:
    def test_maps_logical_names_through_the_rules(self) -> None:
        rules = MeshRules(data="dp", embed="mp")
        assert partition_spec_for_names(rules, "data", "embed") == PartitionSpec("dp", "mp")

    def test_empty_rules_replicate_everything(self) -> None:
        assert partition_spec_for_names(MeshRules(), "data", "embed") == PartitionSpec(None, None)

    @given(st.lists(_LOGICAL_NAMES, min_size=1, max_size=6))
    def test_spec_has_one_entry_per_logical_name(self, names: list[str]) -> None:
        spec = partition_spec_for_names(fsdp_rules(), *names)
        assert len(spec) == len(names)
        assert all(axis in {"data", "model"} for axis in spec)
