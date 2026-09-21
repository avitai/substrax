"""The checkpoint store protocol and its Orbax implementation.

A checkpoint is a step holding named items (``model``, ``optimizer``, ``rng``,
``data_iterator``, ``extensions``), each written with Orbax's ``PyTreeSave``, beside one
:class:`~substrax.checkpoint.metadata.CheckpointMetadata` record written with ``JsonSave``.
``PyTreeSave`` carries arrays, typed PRNG keys and plain-Python leaves (ints, floats,
strings, booleans, lists) alike, so restoring a checkpoint never executes code.

On disk each item is one Orbax item holding the pytree under a single ``tree`` node:
Orbax 0.11.33, the floor, refuses an item that is a bare array (a PRNG key on its own
fails its ``if not item`` check), and the node makes every item a mapping.

Restoring onto templates places every array on its template leaf's device, whatever
topology the checkpoint was written on; without templates the items come back as stored.
A template whose dtype differs from a saved array's is refused before any array is read,
unless the caller asks for the cast with ``cast_dtypes=True``: Orbax would otherwise cast every
array to its template's dtype. ``save`` writes every array's dtype as a ``dtypes`` JSON item
beside the metadata record, so the comparison reads one small file; a checkpoint written without
it is compared against Orbax's per-array metadata. An item restored without a template is
compared after it is read, which catches a 64-bit array jax creates at 32 bits while its x64
mode is off.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable, Self

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp  # type: ignore[import-untyped]
from pydantic import TypeAdapter

from substrax.checkpoint.errors import (
    CheckpointDtypeMismatchError,
    CheckpointNotFoundError,
    CheckpointNotWrittenError,
    DtypeMismatch,
)
from substrax.checkpoint.metadata import (
    check_extra,
    check_item_names,
    CheckpointMetadata,
    library_versions,
    now_iso,
    Producer,
)
from substrax.typing import JsonValue


logger = logging.getLogger(__name__)

METADATA_ITEM = "metadata"
# Every array's dtype by item and leaf path, a JSON item beside the metadata record so the
# record that best_step reads for every step stays small.
DTYPES_ITEM = "dtypes"
ITEM_NODE = "tree"
# The two JSON items as Orbax's JsonRestore returns them, checked before they are read.
_JSON_OBJECT: TypeAdapter[dict[str, JsonValue]] = TypeAdapter(dict[str, JsonValue])
_RECORDED_DTYPES: TypeAdapter[dict[str, dict[str, str]]] = TypeAdapter(dict[str, dict[str, str]])

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
        cast_dtypes: bool = False,
    ) -> Checkpoint:
        """Read the checkpoint at ``step``, onto ``templates`` where given."""
        ...

    def read_metadata(self, step: int) -> CheckpointMetadata:
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


def _wrapped(template: Any) -> Any:
    """The on-disk form of an item's template: the pytree under the ``tree`` node."""
    return None if template is None else {ITEM_NODE: template}


# Orbax stores a typed PRNG key by its data, which is uint32 for every jax key implementation.
_KEY_DATA_DTYPE = "uint32"


def _dtype_name(dtype: Any) -> str:
    """The name a dtype is recorded and compared by; a typed PRNG key by its data's dtype."""
    if jnp.issubdtype(dtype, jax.dtypes.prng_key):
        return _KEY_DATA_DTYPE
    return str(np.dtype(dtype))


def _leaf_dtypes(tree: Any, *, node_depth: int) -> dict[str, str]:
    """The dtype name of every leaf of ``tree`` that has one, keyed by its path joined with ``/``.

    The path is Orbax's flat key for the leaf with its first ``node_depth`` keys dropped (1 for
    an item under the ``tree`` node). Leaves without a dtype, strings and Python numbers, are
    left out. Arrays, ``jax.ShapeDtypeStruct`` and Orbax's own array metadata all qualify.
    """
    return {
        "/".join(str(part) for part in key[node_depth:]): _dtype_name(dtype)
        for key, leaf in ocp.tree.to_flat_dict(tree).items()
        if (dtype := getattr(leaf, "dtype", None)) is not None
    }


def _dtype_mismatches(
    item: str, saved: Mapping[str, str], tree: Any, *, node_depth: int
) -> list[DtypeMismatch]:
    """The leaves of one item's ``tree`` whose dtype differs from the saved array's.

    ``tree`` is the item's template before a restore, or the restored item after one. A leaf
    the checkpoint lacks is left to Orbax's own structure check.
    """
    return [
        DtypeMismatch(item=item, leaf=leaf, saved=saved[leaf], restored=name)
        for leaf, name in _leaf_dtypes(tree, node_depth=node_depth).items()
        if leaf in saved and saved[leaf] != name
    ]


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
    ) -> None:
        """Remember where the checkpoints live and how many to keep.

        Args:
            directory: Directory the checkpoints are stored under; created on the first save.
            max_to_keep: How many of the newest steps Orbax retains; ``None`` keeps every one.

        Raises:
            ValueError: If ``directory`` is empty or whitespace-only.
        """
        if not str(directory).strip():
            raise ValueError("Checkpoint directory cannot be empty")
        self.directory = Path(directory).resolve()
        self.max_to_keep = max_to_keep
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

    def _raw_metadata(self, manager: Any, step: int) -> dict[str, JsonValue]:
        restored = manager.restore(
            step,
            args=ocp.args.Composite(**{METADATA_ITEM: ocp.args.JsonRestore()}),  # type: ignore[reportCallIssue]
        )
        return _JSON_OBJECT.validate_python(restored[METADATA_ITEM], strict=True)

    def _recorded_dtypes(self, manager: Any, step: int) -> dict[str, dict[str, str]]:
        """The ``dtypes`` item of ``step``; empty for a checkpoint written without one."""
        if not (self.directory / str(step) / DTYPES_ITEM).is_dir():
            return {}
        restored = manager.restore(
            step,
            args=ocp.args.Composite(**{DTYPES_ITEM: ocp.args.JsonRestore()}),  # type: ignore[reportCallIssue]
        )
        return _RECORDED_DTYPES.validate_python(restored[DTYPES_ITEM], strict=True)

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
        """Write ``items`` at ``step`` with a metadata record and every array's dtype.

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
        dtypes = {name: _leaf_dtypes(_wrapped(items[name]), node_depth=1) for name in names}
        args = ocp.args.Composite(  # type: ignore[reportCallIssue]
            **{name: ocp.args.PyTreeSave({ITEM_NODE: items[name]}) for name in names},  # type: ignore[reportCallIssue]
            **{METADATA_ITEM: ocp.args.JsonSave(metadata.to_dict())},  # type: ignore[reportCallIssue]
            **{DTYPES_ITEM: ocp.args.JsonSave(dtypes)},  # type: ignore[reportCallIssue]
        )
        manager = self._open()
        written = manager.save(step, args=args, force=True)  # type: ignore[reportCallIssue]
        manager.wait_until_finished()
        if not written:
            raise CheckpointNotWrittenError(step=step, latest_step=latest, reason="rejected")
        logger.info("Saved checkpoint step %s with items %s", step, names)
        return self.directory / str(step)

    def _refuse_dtype_changes(
        self,
        step: int,
        trees: Mapping[str, tuple[Any, int]],
        recorded: Mapping[str, Mapping[str, str]],
        *,
        restored: bool,
    ) -> None:
        """Raise when an item would come back with another dtype than it was saved with.

        An item's saved dtypes come from ``recorded``, the metadata record's, when it holds
        the item, and otherwise from Orbax's per-array metadata, which opens every array's
        store: a checkpoint written before substrax recorded dtypes pays that once per restore.

        Args:
            step: The checkpoint's step.
            trees: Each on-disk item's template, or restored tree, with its node depth (1 under
                ``tree``).
            recorded: The dtypes the metadata record holds, by item.
            restored: Whether ``trees`` are restored items, whose dtypes this process chose.

        Raises:
            CheckpointDtypeMismatchError: If any leaf's dtype differs from the saved array's.
        """
        mismatches: list[DtypeMismatch] = []
        with contextlib.ExitStack() as stack:
            handler: Any = None
            for item, (tree, node_depth) in trees.items():
                saved = recorded.get(item)
                if saved is None:
                    if handler is None:
                        handler = stack.enter_context(
                            contextlib.closing(ocp.PyTreeCheckpointHandler())
                        )
                    metadata = handler.metadata(self.directory / str(step) / item)
                    saved = _leaf_dtypes(getattr(metadata, "tree", metadata), node_depth=node_depth)
                mismatches += _dtype_mismatches(item, saved, tree, node_depth=node_depth)
        if mismatches:
            raise CheckpointDtypeMismatchError(
                step=step,
                mismatches=tuple(mismatches),
                x64_disabled=restored and not jax.config.jax_enable_x64,
            )

    def _read_items(
        self, manager: Any, step: int, metadata: CheckpointMetadata, templates: dict[str, Any]
    ) -> Checkpoint:
        """Read every item of the checkpoint, each onto its template where given."""
        restored = manager.restore(
            step,
            args=ocp.args.Composite(  # type: ignore[reportCallIssue]
                **{name: _restore_arg(_wrapped(templates.get(name))) for name in metadata.items}
            ),
        )
        items = {name: restored[name][ITEM_NODE] for name in metadata.items}
        return Checkpoint(step=step, items=items, metadata=metadata)

    def _restore_items(
        self,
        manager: Any,
        step: int,
        raw: Mapping[str, JsonValue],
        *,
        templates: dict[str, Any],
        cast_dtypes: bool,
    ) -> Checkpoint:
        """Read the checkpoint, comparing every item's dtypes with the saved ones.

        Templates are compared before any array is read; items without one after.
        """
        metadata = CheckpointMetadata.from_dict(raw)
        missing = sorted(set(templates) - set(metadata.items))
        if missing:
            raise ValueError(
                f"templates name items the checkpoint lacks: {missing}; "
                f"step {step} holds {list(metadata.items)}"
            )
        if cast_dtypes:
            return self._read_items(manager, step, metadata, templates)
        recorded = self._recorded_dtypes(manager, step)
        templated = {name: (_wrapped(tree), 1) for name, tree in templates.items()}
        self._refuse_dtype_changes(step, templated, recorded, restored=False)
        checkpoint = self._read_items(manager, step, metadata, templates)
        untemplated = {
            name: (_wrapped(tree), 1)
            for name, tree in checkpoint.items.items()
            if name not in templates
        }
        self._refuse_dtype_changes(step, untemplated, recorded, restored=True)
        return checkpoint

    def restore(  # noqa: DOC502  # raised by _require_step, the format reader and Orbax
        self,
        step: int,
        *,
        templates: Mapping[str, Any] | None = None,
        cast_dtypes: bool = False,
    ) -> Checkpoint:
        """Read the checkpoint at ``step``.

        Args:
            step: The step to read.
            templates: Pytrees by item name; each named item is restored onto its template's
                leaves (device placement included), and an item without a template comes back
                as stored. Leaves may be arrays or ``jax.ShapeDtypeStruct``.
            cast_dtypes: Accept an array coming back with another dtype than it was saved with:
                cast to its template leaf's, or, without a template, a 64-bit array at 32 bits
                while jax's x64 mode is off. Without it, either is refused.

        Returns:
            The restored items and the checkpoint's metadata.

        Raises:
            CheckpointNotFoundError: If the store holds no checkpoint at ``step``.
            UnsupportedCheckpointError: If the checkpoint is in another format than the one
                this substrax reads, or is not a substrax checkpoint.
            CheckpointDtypeMismatchError: If an array would come back with another dtype than
                it was saved with and ``cast_dtypes`` is false.
            ValueError: If ``templates`` names an item the checkpoint lacks, or a template's
                tree does not match the checkpoint's.
        """
        manager = self._require_step(step)
        raw = self._raw_metadata(manager, step)
        return self._restore_items(
            manager, step, raw, templates=dict(templates or {}), cast_dtypes=cast_dtypes
        )

    def read_metadata(  # noqa: DOC502  # raised by _require_step and from_dict
        self, step: int
    ) -> CheckpointMetadata:
        """Read the metadata record of ``step`` without its items.

        Args:
            step: The step to read.

        Returns:
            The metadata record.

        Raises:
            CheckpointNotFoundError: If the store holds no checkpoint at ``step``.
            UnsupportedCheckpointError: If the checkpoint is in another format than the one
                this substrax reads, or is not a substrax checkpoint.
        """
        manager = self._require_step(step)
        return CheckpointMetadata.from_dict(self._raw_metadata(manager, step))

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
