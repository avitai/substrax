"""The checkpoint store protocol and its Orbax implementation.

A checkpoint is a step holding named items (``model``, ``optimizer``, ``rng``,
``data_iterator``, ``extensions``), each written with Orbax's ``PyTreeSave``, beside one
:class:`~substrax.checkpoint.metadata.CheckpointMetadata` record written with ``JsonSave``.
``PyTreeSave`` carries arrays, typed PRNG keys and plain-Python leaves (ints, floats,
strings, booleans, lists) alike, so restoring a checkpoint never executes code.

Restoring onto templates places every array on its template leaf's device and dtype,
whatever topology the checkpoint was written on; without templates the items come back
as stored. A format-2 checkpoint (substrax 0.1.5 to 0.1.9) is upgraded in memory through
the migration registry, split by the producer's :class:`~substrax.checkpoint.legacy.LegacyLayout`.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable, Self

import orbax.checkpoint as ocp  # type: ignore[import-untyped]

from substrax.checkpoint.errors import CheckpointNotFoundError, CheckpointNotWrittenError
from substrax.checkpoint.legacy import LegacyLayout, MODULE_ONLY_FORMAT2
from substrax.checkpoint.metadata import (
    check_extra,
    check_item_names,
    CheckpointMetadata,
    JsonValue,
    library_versions,
    now_iso,
    Producer,
)
from substrax.checkpoint.migration import DEFAULT_REGISTRY, MigrationRegistry


logger = logging.getLogger(__name__)

METADATA_ITEM = "metadata"
_LEGACY_PAYLOAD_ITEM = "model"

BestMode = Literal["min", "max"]


@dataclass(frozen=True, slots=True, kw_only=True)
class Checkpoint:
    """A restored checkpoint: its step, its items by name and its metadata."""

    step: int
    items: dict[str, Any]
    metadata: CheckpointMetadata


@runtime_checkable
class CheckpointStore(Protocol):
    """Step-addressed checkpoint store contract.

    Implementations persist named items plus a metadata record under an integer step
    and restore them back. Training code depends on this protocol; the Orbax class is
    the one implementation.
    """

    def save(
        self,
        step: int,
        items: Mapping[str, Any],
        *,
        epoch: int | None = None,
        metrics: Mapping[str, float] | None = None,
        producer: Producer | None = None,
        extra: Mapping[str, JsonValue] | None = None,
        overwrite: bool = False,
    ) -> Path:
        """Write ``items`` at ``step`` and return the checkpoint's directory."""
        ...

    def restore(
        self,
        step: int,
        *,
        templates: Mapping[str, Any] | None = None,
        legacy_layout: LegacyLayout | None = None,
    ) -> Checkpoint:
        """Read the checkpoint at ``step``, onto ``templates`` where given."""
        ...

    def read_metadata(
        self, step: int, *, legacy_layout: LegacyLayout | None = None
    ) -> CheckpointMetadata:
        """Read only the metadata record of ``step``."""
        ...

    def list_steps(self) -> list[int]:
        """Every step held, ascending."""
        ...

    def latest_step(self) -> int | None:
        """The newest step, or ``None`` for an empty store."""
        ...

    def best_step(self, metric: str, *, mode: BestMode = "min") -> int | None:
        """The step whose ``metric`` is the lowest or highest, or ``None`` when none has it."""
        ...

    def delete(self, step: int) -> None:
        """Remove the checkpoint at ``step``."""
        ...

    def close(self) -> None:
        """Release backend resources."""
        ...


def _restore_arg(template: Any) -> Any:
    """Orbax restore arguments for one item.

    Without a template the item is restored as it was stored: the checkpoint describes
    its own tree, so list lengths and plain leaves come back as saved, and each array
    returns to the device placement recorded at save time.

    With one, the template's tree must match the checkpoint, and every array is restored
    onto its template leaf's sharding and dtype. A bare ``PyTreeRestore(template)`` gives
    Orbax only the tree structure: the per-leaf ``restore_args`` are what carry placement,
    and without them Orbax falls back to the saved sharding file, which names devices the
    restoring process may not have.
    """
    if template is None:
        return ocp.args.PyTreeRestore()  # type: ignore[reportCallIssue]
    return ocp.args.PyTreeRestore(  # type: ignore[arg-type, reportCallIssue]
        template, restore_args=ocp.checkpoint_utils.construct_restore_args(template)
    )


class OrbaxCheckpointStore:
    """Orbax-backed :class:`CheckpointStore`.

    Wraps :class:`orbax.checkpoint.CheckpointManager` with step-int addressing and
    retention by ``max_to_keep``. The manager is opened on first use, so constructing a
    store writes nothing; the directory appears with the first save. The store doubles as
    a context manager so backend resources are released deterministically::

        with OrbaxCheckpointStore(path, max_to_keep=5) as store:
            store.save(step, {"model": nnx.state(model)}, metrics={"loss": loss})
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        max_to_keep: int | None = 5,
        registry: MigrationRegistry = DEFAULT_REGISTRY,
    ) -> None:
        """Remember where the checkpoints live and how many to keep.

        Args:
            directory: Directory the checkpoints are stored under; created on the first save.
            max_to_keep: How many of the newest steps Orbax retains; ``None`` keeps every one.
            registry: The migrations applied to checkpoints of earlier formats.

        Raises:
            ValueError: If ``directory`` is empty or whitespace-only.
        """
        if not str(directory).strip():
            raise ValueError("Checkpoint directory cannot be empty")
        self.directory = Path(directory).resolve()
        self.max_to_keep = max_to_keep
        self._registry = registry
        self._manager: Any = None

    def __enter__(self) -> Self:
        """Enter the context manager."""
        return self

    def __exit__(self, *_exc_info: object) -> None:
        """Close the backend on context exit."""
        self.close()

    def _open(self) -> Any:
        """The Orbax manager, opened on first use."""
        if self._manager is None:
            options = ocp.CheckpointManagerOptions(max_to_keep=self.max_to_keep, create=True)
            self._manager = ocp.CheckpointManager(self.directory, options=options)
            logger.info("Opened OrbaxCheckpointStore at %s", self.directory)
        return self._manager

    def _steps(self) -> list[int]:
        if self._manager is None and not self.directory.exists():
            return []
        return sorted(int(step) for step in self._open().all_steps())

    def _require_step(self, step: int) -> Any:
        """The manager, once ``step`` is known to exist.

        Args:
            step: The step that must exist.

        Returns:
            The open Orbax manager.

        Raises:
            CheckpointNotFoundError: If the store holds no checkpoint at ``step``.
        """
        if step not in self._steps():
            raise CheckpointNotFoundError(step, self.directory)
        return self._open()

    def _admit_write(self, step: int, *, overwrite: bool) -> int | None:
        """Check the write rules for ``step`` and return the latest step before the write.

        An existing step is deleted when ``overwrite`` is set and refused otherwise; a step
        below the latest is refused, since a store's steps only grow.

        Args:
            step: The step about to be written.
            overwrite: Whether an existing step is replaced.

        Returns:
            The latest step the store held before the write, or ``None`` for an empty store.

        Raises:
            CheckpointNotWrittenError: If ``step`` is written and ``overwrite`` is false, or
                below the latest step.
        """
        steps = self._steps()
        latest = steps[-1] if steps else None
        if step in steps:
            if not overwrite:
                raise CheckpointNotWrittenError(step=step, latest_step=latest, reason="exists")
            self._open().delete(step)
        elif latest is not None and step < latest:
            raise CheckpointNotWrittenError(step=step, latest_step=latest, reason="below_latest")
        return latest

    def _raw_metadata(self, manager: Any, step: int) -> dict[str, Any]:
        restored = manager.restore(
            step,
            args=ocp.args.Composite(**{METADATA_ITEM: ocp.args.JsonRestore()}),  # type: ignore[reportCallIssue]
        )
        return dict(restored[METADATA_ITEM])

    def save(
        self,
        step: int,
        items: Mapping[str, Any],
        *,
        epoch: int | None = None,
        metrics: Mapping[str, float] | None = None,
        producer: Producer | None = None,
        extra: Mapping[str, JsonValue] | None = None,
        overwrite: bool = False,
    ) -> Path:
        """Write ``items`` at ``step`` with a format-3 metadata record.

        Args:
            step: Non-negative step the checkpoint is addressed by.
            items: The pytrees to write, by item name (one of
                :data:`~substrax.checkpoint.metadata.ITEM_NAMES`).
            epoch: The epoch the step falls in, when the caller counts epochs.
            metrics: Scalar metrics recorded for ``best_step``.
            producer: The library writing the checkpoint.
            extra: The caller's own JSON values, under keys that are not fields of the record.
            overwrite: Replace an existing checkpoint at ``step`` instead of refusing.

        Returns:
            The checkpoint's directory, ``directory / str(step)``.

        Raises:
            ValueError: If ``step`` is negative, ``items`` is empty or names an unknown item,
                or ``extra`` shadows a reserved key.
            CheckpointNotWrittenError: If ``step`` is already written and ``overwrite`` is
                false, if it is below the latest step, or if Orbax declines the write.
        """
        if step < 0:
            raise ValueError("Step must be a non-negative integer")
        names = check_item_names(items)
        checked_extra = check_extra(extra)
        latest = self._admit_write(step, overwrite=overwrite)

        metadata = CheckpointMetadata(
            step=step,
            epoch=epoch,
            items=names,
            libraries=library_versions(),
            producer=producer,
            metrics={name: float(value) for name, value in (metrics or {}).items()},
            extra=checked_extra,
            created_at=now_iso(),
        )
        args = ocp.args.Composite(  # type: ignore[reportCallIssue]
            **{name: ocp.args.PyTreeSave(items[name]) for name in names},  # type: ignore[reportCallIssue]
            **{METADATA_ITEM: ocp.args.JsonSave(metadata.to_dict())},  # type: ignore[reportCallIssue]
        )
        manager = self._open()
        written = manager.save(step, args=args, force=True)  # type: ignore[reportCallIssue]
        manager.wait_until_finished()
        if not written:
            raise CheckpointNotWrittenError(step=step, latest_step=latest, reason="rejected")
        logger.info("Saved checkpoint step %s with items %s", step, names)
        return self.directory / str(step)

    def restore(  # noqa: DOC503  # raised by _require_step, from_dict and Orbax
        self,
        step: int,
        *,
        templates: Mapping[str, Any] | None = None,
        legacy_layout: LegacyLayout | None = None,
    ) -> Checkpoint:
        """Read the checkpoint at ``step``.

        Args:
            step: The step to read.
            templates: Pytrees by item name; each named item is restored onto its template's
                leaves (device placement and dtype included), and an item without a template
                comes back as stored.
            legacy_layout: How a format-2 payload splits into items; the module-only layout
                when not given. Read only for a format-2 checkpoint.

        Returns:
            The restored items and the checkpoint's metadata, upgraded to the current format.

        Raises:
            CheckpointNotFoundError: If the store holds no checkpoint at ``step``.
            UnsupportedCheckpointError: If the checkpoint's format is newer than this substrax
                reads, or is not a substrax checkpoint.
            ValueError: If ``templates`` names an item the checkpoint lacks, or a template's
                tree does not match the checkpoint's.
        """
        manager = self._require_step(step)
        raw = self._raw_metadata(manager, step)
        migration = self._registry.for_metadata(raw)
        if migration is not None:
            layout = MODULE_ONLY_FORMAT2 if legacy_layout is None else legacy_layout
            template = None if templates is None else layout.template_of(templates)
            restored = manager.restore(
                step,
                args=ocp.args.Composite(**{_LEGACY_PAYLOAD_ITEM: _restore_arg(template)}),  # type: ignore[reportCallIssue]
            )
            items, metadata = migration.upgrade(restored[_LEGACY_PAYLOAD_ITEM], raw, layout)
            return Checkpoint(step=step, items=items, metadata=metadata)

        metadata = CheckpointMetadata.from_dict(raw)
        given = dict(templates or {})
        missing = sorted(set(given) - set(metadata.items))
        if missing:
            raise ValueError(
                f"templates name items the checkpoint lacks: {missing}; "
                f"step {step} holds {list(metadata.items)}"
            )
        restored = manager.restore(
            step,
            args=ocp.args.Composite(  # type: ignore[reportCallIssue]
                **{name: _restore_arg(given.get(name)) for name in metadata.items}
            ),
        )
        items = {name: restored[name] for name in metadata.items}
        return Checkpoint(step=step, items=items, metadata=metadata)

    def read_metadata(  # noqa: DOC502  # raised by _require_step and from_dict
        self, step: int, *, legacy_layout: LegacyLayout | None = None
    ) -> CheckpointMetadata:
        """Read the metadata record of ``step`` without its items.

        A format-2 checkpoint has no record of its items, so its payload is read as stored
        and split by ``legacy_layout`` (the module-only layout when not given) to name them.

        Args:
            step: The step to read.
            legacy_layout: How a format-2 payload splits into items.

        Returns:
            The metadata record, upgraded to the current format.

        Raises:
            CheckpointNotFoundError: If the store holds no checkpoint at ``step``.
            UnsupportedCheckpointError: If the checkpoint's format is newer than this substrax
                reads, or is not a substrax checkpoint.
        """
        manager = self._require_step(step)
        raw = self._raw_metadata(manager, step)
        if self._registry.for_metadata(raw) is not None:
            return self.restore(step, legacy_layout=legacy_layout).metadata
        return CheckpointMetadata.from_dict(raw)

    def list_steps(self) -> list[int]:
        """Every step held, ascending."""
        return self._steps()

    def latest_step(self) -> int | None:
        """The newest step, or ``None`` for an empty store."""
        steps = self._steps()
        return steps[-1] if steps else None

    def best_step(self, metric: str, *, mode: BestMode = "min") -> int | None:
        """The step whose ``metric`` is the lowest (``"min"``) or highest (``"max"``).

        Steps whose metadata lacks the metric are skipped.

        Args:
            metric: The metric name, as recorded in each step's ``metrics``.
            mode: ``"min"`` for the lowest value, ``"max"`` for the highest.

        Returns:
            The best step, or ``None`` when no step records the metric.

        Raises:
            ValueError: If ``mode`` is neither ``"min"`` nor ``"max"``.
        """
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        scored = [
            (step, value)
            for step in self._steps()
            if (value := self.read_metadata(step).metrics.get(metric)) is not None
        ]
        if not scored:
            return None
        chooser = min if mode == "min" else max
        return chooser(scored, key=lambda pair: pair[1])[0]

    def delete(self, step: int) -> None:  # noqa: DOC502  # raised by _require_step
        """Remove the checkpoint at ``step``.

        Args:
            step: The step to remove.

        Raises:
            CheckpointNotFoundError: If the store holds no checkpoint at ``step``.
        """
        self._require_step(step).delete(step)
        logger.info("Deleted checkpoint step %s", step)

    def close(self) -> None:
        """Close the Orbax manager, if one was opened."""
        if self._manager is not None:
            self._manager.close()
            self._manager = None
            logger.info("Closed OrbaxCheckpointStore at %s", self.directory)
