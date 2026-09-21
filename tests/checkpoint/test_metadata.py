"""The format-3 metadata record: fixed item names, reserved keys, a version the reader checks."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from substrax.checkpoint import (
    CheckpointMetadata,
    CURRENT_FORMAT_VERSION,
    FORMAT_NAME,
    ITEM_NAMES,
    Producer,
    UnsupportedCheckpointError,
)
from substrax.checkpoint.metadata import (
    check_extra,
    JsonValue,
    library_versions,
    RESERVED_METADATA_KEYS,
)


def _metadata(**overrides: object) -> CheckpointMetadata:
    fields: dict[str, object] = {
        "step": 100,
        "epoch": 2,
        "items": ("model", "optimizer"),
        "libraries": {"jax": "0.11.1"},
        "producer": Producer(name="artifex", version="0.1.10"),
        "metrics": {"loss": 0.25},
        "extra": {"run": "demo"},
        "created_at": "2026-09-17T12:00:00+00:00",
    }
    fields.update(overrides)
    return CheckpointMetadata(**fields)  # type: ignore[arg-type]


class TestContract:
    def test_item_names_are_the_five_training_items(self) -> None:
        assert ITEM_NAMES == ("model", "optimizer", "rng", "data_iterator", "extensions")

    def test_the_current_format_is_one(self) -> None:
        assert CURRENT_FORMAT_VERSION == 1
        assert FORMAT_NAME == "substrax-checkpoint"
        assert _metadata().format == FORMAT_NAME
        assert _metadata().format_version == CURRENT_FORMAT_VERSION

    def test_reserved_keys_are_the_record_fields(self) -> None:
        assert frozenset(CheckpointMetadata.__dataclass_fields__) == RESERVED_METADATA_KEYS

    @pytest.mark.parametrize("key", sorted(RESERVED_METADATA_KEYS))
    def test_extra_cannot_shadow_a_reserved_key(self, key: str) -> None:
        with pytest.raises(ValueError, match=key):
            check_extra({key: 1})

    def test_extra_keeps_json_values(self) -> None:
        extra: dict[str, JsonValue] = {
            "run": "demo",
            "seeds": [1, 2],
            "nested": {"a": None, "b": True},
        }
        assert check_extra(extra) == extra

    def test_a_distribution_that_is_not_installed_is_skipped(self) -> None:
        assert library_versions(("jax", "no-such-distribution-for-this-test")) == {
            "jax": library_versions()["jax"]
        }

    def test_library_versions_name_the_stack(self) -> None:
        versions = library_versions()
        assert {"jax", "flax", "orbax-checkpoint", "substrax"} <= set(versions)
        assert all(isinstance(version, str) and version for version in versions.values())


class TestRoundTrip:
    def test_to_dict_and_back_is_the_identity(self) -> None:
        metadata = _metadata()
        payload = metadata.to_dict()
        assert payload["format"] == FORMAT_NAME
        assert payload["format_version"] == CURRENT_FORMAT_VERSION
        assert payload["items"] == ["model", "optimizer"]
        assert payload["producer"] == {"name": "artifex", "version": "0.1.10"}
        assert CheckpointMetadata.from_dict(payload) == metadata

    def test_a_record_without_producer_or_epoch_round_trips(self) -> None:
        metadata = _metadata(producer=None, epoch=None)
        assert CheckpointMetadata.from_dict(metadata.to_dict()) == metadata

    def test_another_version_number_is_refused(self) -> None:
        payload = _metadata().to_dict()
        payload["format_version"] = CURRENT_FORMAT_VERSION + 1
        with pytest.raises(UnsupportedCheckpointError, match=str(CURRENT_FORMAT_VERSION + 1)):
            CheckpointMetadata.from_dict(payload)

    def test_another_format_name_is_refused(self) -> None:
        payload = _metadata().to_dict()
        payload["format"] = "something-else"
        with pytest.raises(UnsupportedCheckpointError, match="something-else"):
            CheckpointMetadata.from_dict(payload)

    @pytest.mark.parametrize(
        ("key", "value", "location"),
        [
            ("step", "100", ("step",)),
            ("epoch", 2.5, ("epoch",)),
            ("items", "model", ("items",)),
            ("metrics", {"loss": "low"}, ("metrics", "loss")),
            ("libraries", {"jax": 11}, ("libraries", "jax")),
            ("producer", {"name": "artifex"}, ("producer", "version")),
            ("created_at", None, ("created_at",)),
            ("format_version", "3", ("format_version",)),
        ],
    )
    def test_a_field_of_the_wrong_json_type_is_refused_at_its_path(
        self, key: str, value: JsonValue, location: tuple[str, ...]
    ) -> None:
        payload = _metadata().to_dict()
        payload[key] = value
        with pytest.raises(ValidationError) as refused:
            CheckpointMetadata.from_dict(payload)
        assert refused.value.errors()[0]["loc"] == location
