"""The format-3 metadata record written beside every checkpoint's items.

A checkpoint is a step holding named items, each an Orbax pytree item, and one JSON
record describing them: the format and its version, the step and epoch, the item names,
the library versions that wrote it, the producer, the metrics ``best_step`` reads and a
caller's extra values under their own keys.
"""

from __future__ import annotations

import functools
import importlib.metadata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, fields
from datetime import datetime, UTC
from typing import Any

from substrax.checkpoint.errors import UnsupportedCheckpointError
from substrax.typing import JsonValue


FORMAT_NAME = "substrax-checkpoint"
CURRENT_FORMAT_VERSION = 3
ITEM_NAMES = ("model", "optimizer", "rng", "data_iterator", "extensions")
LIBRARY_NAMES = ("jax", "flax", "orbax-checkpoint", "optax", "numpy", "substrax")


@dataclass(frozen=True, slots=True, kw_only=True)
class Producer:
    """The library that wrote a checkpoint, by distribution name and version."""

    name: str
    version: str

    def to_dict(self) -> dict[str, str]:
        """The JSON form."""
        return {"name": self.name, "version": self.version}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Producer:
        """Read the JSON form."""
        return cls(name=str(payload["name"]), version=str(payload["version"]))


@dataclass(frozen=True, slots=True, kw_only=True)
class CheckpointMetadata:
    """What a checkpoint says about itself.

    Attributes:
        format: ``"substrax-checkpoint"``; another name is another producer's file.
        format_version: The on-disk format; a reader refuses a version newer than its own.
        step: The step the checkpoint is addressed by.
        epoch: The epoch the step falls in, when the producer counts epochs.
        items: The item names written, in the order they were given.
        libraries: Distribution versions at write time, informational only.
        producer: The library that wrote the checkpoint, when it says.
        metrics: Scalar metrics, the values ``best_step`` compares.
        extra: The caller's own JSON values; a key here never shadows a field of the record.
        created_at: The write time, ISO 8601 in UTC.
    """

    format: str = FORMAT_NAME
    format_version: int = CURRENT_FORMAT_VERSION
    step: int
    epoch: int | None = None
    items: tuple[str, ...]
    libraries: Mapping[str, str] = field(default_factory=dict)
    producer: Producer | None = None
    metrics: Mapping[str, float] = field(default_factory=dict)
    extra: Mapping[str, JsonValue] = field(default_factory=dict)
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        """The JSON form Orbax writes as the ``metadata`` item."""
        return {
            "format": self.format,
            "format_version": self.format_version,
            "step": self.step,
            "epoch": self.epoch,
            "items": list(self.items),
            "libraries": dict(self.libraries),
            "producer": None if self.producer is None else self.producer.to_dict(),
            "metrics": {name: float(value) for name, value in self.metrics.items()},
            "extra": dict(self.extra),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CheckpointMetadata:
        """Read the JSON form, refusing a record this substrax cannot interpret.

        Args:
            payload: The record as Orbax's ``JsonRestore`` returns it.

        Returns:
            The metadata record.

        Raises:
            UnsupportedCheckpointError: If the record is a format-2 sidecar (those go through
                the migration registry), names another format, or a newer version.
        """
        if "checkpoint_version" in payload:
            raise UnsupportedCheckpointError(
                f"a format 2 checkpoint (checkpoint_version {payload['checkpoint_version']!r}) "
                "is read through the migration registry, not as a format 3 record"
            )
        name = payload.get("format")
        if name != FORMAT_NAME:
            raise UnsupportedCheckpointError(f"not a {FORMAT_NAME} record: format {name!r}")
        version = int(payload.get("format_version", 0))
        if version > CURRENT_FORMAT_VERSION:
            raise UnsupportedCheckpointError(
                f"checkpoint format {version} is newer than the format {CURRENT_FORMAT_VERSION} "
                "this substrax reads; upgrade substrax"
            )
        producer = payload.get("producer")
        epoch = payload.get("epoch")
        return cls(
            format=name,
            format_version=version,
            step=int(payload["step"]),
            epoch=None if epoch is None else int(epoch),
            items=tuple(str(item) for item in payload.get("items", ())),
            libraries={str(k): str(v) for k, v in dict(payload.get("libraries", {})).items()},
            producer=None if producer is None else Producer.from_dict(producer),
            metrics={str(k): float(v) for k, v in dict(payload.get("metrics", {})).items()},
            extra=dict(payload.get("extra", {})),
            created_at=str(payload["created_at"]),
        )


RESERVED_METADATA_KEYS = frozenset(f.name for f in fields(CheckpointMetadata))


def check_extra(extra: Mapping[str, JsonValue] | None) -> dict[str, JsonValue]:
    """Return ``extra`` as a dict, refusing a key that names a field of the record.

    Args:
        extra: The caller's values, or ``None`` for none.

    Returns:
        The values as a new dict.

    Raises:
        ValueError: If a key is one of :data:`RESERVED_METADATA_KEYS`.
    """
    values = dict(extra or {})
    shadowed = sorted(set(values) & RESERVED_METADATA_KEYS)
    if shadowed:
        raise ValueError(
            f"extra metadata cannot use the reserved keys {shadowed}; "
            "they are fields of the checkpoint record"
        )
    return values


def check_item_names(items: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the item names in order, refusing an empty mapping or a name outside the format.

    Args:
        items: The items by name.

    Returns:
        The names, in the mapping's order.

    Raises:
        ValueError: If ``items`` is empty or holds a name that is not in :data:`ITEM_NAMES`.
    """
    names = tuple(items)
    if not names:
        raise ValueError("a checkpoint holds at least one item")
    unknown = [name for name in names if name not in ITEM_NAMES]
    if unknown:
        raise ValueError(f"unknown checkpoint items {unknown}; the format's items are {ITEM_NAMES}")
    return names


@functools.cache
def _installed_versions(names: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    """Distribution versions, read once per process: they cannot change while it runs."""
    versions: list[tuple[str, str]] = []
    for name in names:
        try:
            versions.append((name, importlib.metadata.version(name)))
        except importlib.metadata.PackageNotFoundError:
            continue
    return tuple(versions)


def library_versions(names: Iterable[str] = LIBRARY_NAMES) -> dict[str, str]:
    """The installed version of each named distribution, skipping the ones not installed."""
    return dict(_installed_versions(tuple(names)))


def now_iso() -> str:
    """The current time, ISO 8601 in UTC."""
    return datetime.now(UTC).isoformat()
