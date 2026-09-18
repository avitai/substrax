"""Checkpoints: named items beside one metadata record, in format 3.

A checkpoint is a step holding the items a training loop owns (``model``, ``optimizer``,
``rng``, ``data_iterator``, ``extensions``), each an Orbax pytree item, and one
:class:`CheckpointMetadata` record. :class:`OrbaxCheckpointStore` is the one
implementation of the :class:`CheckpointStore` protocol; it refuses to overwrite or
reorder steps unless told to, restores onto templates so a checkpoint written on one
device topology lands on whatever the restoring process has, and upgrades format-2
checkpoints (substrax 0.1.5 to 0.1.9) in memory through the migration registry.
``upgrade_checkpoints`` and ``python -m substrax.checkpoint upgrade`` rewrite a root in
the current format into a new root.
"""

from substrax.checkpoint.directory import resolve_checkpoint_dir
from substrax.checkpoint.errors import (
    CheckpointDtypeMismatchError,
    CheckpointNotFoundError,
    CheckpointNotWrittenError,
    DtypeMismatch,
    UnsupportedCheckpointError,
)
from substrax.checkpoint.legacy import LegacyLayout, MODULE_ONLY_FORMAT2
from substrax.checkpoint.metadata import (
    CheckpointMetadata,
    CURRENT_FORMAT_VERSION,
    FORMAT_NAME,
    ITEM_NAMES,
    Producer,
)
from substrax.checkpoint.migration import Migration, MigrationRegistry
from substrax.checkpoint.store import Checkpoint, CheckpointStore, OrbaxCheckpointStore
from substrax.checkpoint.upgrade import upgrade_checkpoints


__all__ = [
    "CURRENT_FORMAT_VERSION",
    "FORMAT_NAME",
    "ITEM_NAMES",
    "MODULE_ONLY_FORMAT2",
    "Checkpoint",
    "CheckpointDtypeMismatchError",
    "CheckpointMetadata",
    "CheckpointNotFoundError",
    "CheckpointNotWrittenError",
    "CheckpointStore",
    "DtypeMismatch",
    "LegacyLayout",
    "Migration",
    "MigrationRegistry",
    "OrbaxCheckpointStore",
    "Producer",
    "UnsupportedCheckpointError",
    "resolve_checkpoint_dir",
    "upgrade_checkpoints",
]
