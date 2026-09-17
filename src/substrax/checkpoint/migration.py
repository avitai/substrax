"""Migrations from earlier checkpoint formats to the current one, applied in memory on restore."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, UTC
from typing import Any

from substrax.checkpoint.errors import UnsupportedCheckpointError
from substrax.checkpoint.legacy import LegacyLayout
from substrax.checkpoint.metadata import (
    check_item_names,
    CheckpointMetadata,
    CURRENT_FORMAT_VERSION,
    now_iso,
)


Upgrade = Callable[
    [Any, Mapping[str, Any], LegacyLayout], tuple[dict[str, Any], CheckpointMetadata]
]


@dataclass(frozen=True, slots=True, kw_only=True)
class Migration:
    """How one earlier format becomes the current one.

    Attributes:
        source_version: The format version the migration reads.
        applies: Whether a raw metadata record is one of that version.
        upgrade: Turns the payload, its raw metadata and the payload's layout into
            format-3 items and metadata.
    """

    source_version: int
    applies: Callable[[Mapping[str, Any]], bool]
    upgrade: Upgrade


class MigrationRegistry:
    """The migrations a store applies, one per format version below the current one."""

    def __init__(self, migrations: Iterable[Migration]) -> None:
        """Index the migrations by source version and check they cover a contiguous range.

        Args:
            migrations: One migration per format version below the current one.

        Raises:
            ValueError: If two migrations share a version, one targets the current format,
                or the versions covered are not contiguous up to the current format.
        """
        by_version: dict[int, Migration] = {}
        for migration in migrations:
            if migration.source_version >= CURRENT_FORMAT_VERSION:
                raise ValueError(
                    f"a migration reads a format below the current {CURRENT_FORMAT_VERSION}, "
                    f"not {migration.source_version}"
                )
            if migration.source_version in by_version:
                raise ValueError(f"two migrations read format {migration.source_version}")
            by_version[migration.source_version] = migration
        versions = sorted(by_version)
        if versions and versions != list(range(versions[0], CURRENT_FORMAT_VERSION)):
            raise ValueError(
                f"migrations must cover contiguous versions up to {CURRENT_FORMAT_VERSION - 1}, "
                f"got {versions}"
            )
        self._migrations = tuple(by_version[version] for version in versions)

    @property
    def source_versions(self) -> tuple[int, ...]:
        """The format versions the registry reads, ascending."""
        return tuple(migration.source_version for migration in self._migrations)

    def for_metadata(self, raw: Mapping[str, Any]) -> Migration | None:
        """The migration that reads ``raw``, or ``None`` when ``raw`` is a current record."""
        for migration in self._migrations:
            if migration.applies(raw):
                return migration
        return None

    def upgrade(
        self, payload: Any, raw: Mapping[str, Any], layout: LegacyLayout
    ) -> tuple[dict[str, Any], CheckpointMetadata]:
        """Upgrade ``payload`` and its raw metadata through the migration that reads them.

        Args:
            payload: The format-2 payload as restored.
            raw: The metadata record as written beside it.
            layout: How the payload splits into items.

        Returns:
            The items by name and the upgraded metadata.

        Raises:
            UnsupportedCheckpointError: If no migration reads ``raw``.
        """
        migration = self.for_metadata(raw)
        if migration is None:
            raise UnsupportedCheckpointError("no migration reads this checkpoint's metadata")
        return migration.upgrade(payload, raw, layout)


_FORMAT2_CONSUMED_KEYS = frozenset(
    {"step", "timestamp", "loss", "epoch", "metrics", "checkpoint_version"}
)


def _upgrade_format2(
    payload: Any, raw: Mapping[str, Any], layout: LegacyLayout
) -> tuple[dict[str, Any], CheckpointMetadata]:
    """Split a format-2 payload by ``layout`` and map its sidecar onto the record.

    The sidecar's ``loss`` and ``metrics`` become the metrics, ``epoch`` the epoch,
    ``timestamp`` the creation time, and every other key (``model_type``,
    ``physics_metadata``, a producer's own values) lands in ``extra`` beside an
    ``upgraded_from`` marker.
    """
    items = dict(layout.items_of(payload))
    names = check_item_names(items)
    metrics: dict[str, float] = {}
    if raw.get("loss") is not None:
        metrics["loss"] = float(raw["loss"])
    recorded = raw.get("metrics")
    if isinstance(recorded, Mapping):
        metrics.update({str(name): float(value) for name, value in recorded.items()})
    epoch = raw.get("epoch")
    timestamp = raw.get("timestamp")
    created_at = (
        now_iso()
        if timestamp is None
        else datetime.fromtimestamp(float(timestamp), UTC).isoformat()
    )
    extra: dict[str, Any] = {
        key: value for key, value in raw.items() if key not in _FORMAT2_CONSUMED_KEYS
    }
    extra["upgraded_from"] = {"format_version": 2}
    metadata = CheckpointMetadata(
        step=int(raw["step"]),
        epoch=None if epoch is None else int(epoch),
        items=names,
        libraries={},
        producer=None,
        metrics=metrics,
        extra=extra,
        created_at=created_at,
    )
    return items, metadata


FORMAT2_TO_3 = Migration(
    source_version=2,
    applies=lambda raw: raw.get("checkpoint_version") == "2.0",
    upgrade=_upgrade_format2,
)
"""Format 2, substrax 0.1.5 to 0.1.9: one ``model`` payload and a ``checkpoint_version`` sidecar."""

DEFAULT_REGISTRY = MigrationRegistry([FORMAT2_TO_3])
