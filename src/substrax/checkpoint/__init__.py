"""Checkpoints: named items beside one metadata record.

A checkpoint is a step holding the items a training loop owns (``model``, ``optimizer``,
``rng``, ``data_iterator``, ``extensions``), each an Orbax pytree item, and one
:class:`CheckpointMetadata` record. :class:`OrbaxCheckpointStore` is the one
implementation of the :class:`CheckpointStore` protocol; it refuses to overwrite or
reorder steps unless told to, restores onto templates so a checkpoint written on one
device topology lands on whatever the restoring process has, and records each array's
dtype at save so a restore that would change one is refused rather than silent.
"""

from substrax.checkpoint.directory import resolve_checkpoint_dir
from substrax.checkpoint.errors import (
    CheckpointDtypeMismatchError,
    CheckpointNotFoundError,
    CheckpointNotWrittenError,
    DtypeMismatch,
    UnsupportedCheckpointError,
)
from substrax.checkpoint.metadata import (
    CheckpointMetadata,
    CURRENT_FORMAT_VERSION,
    FORMAT_NAME,
    ITEM_NAMES,
    Producer,
)
from substrax.checkpoint.store import Checkpoint, CheckpointStore, OrbaxCheckpointStore


__all__ = [
    "CURRENT_FORMAT_VERSION",
    "FORMAT_NAME",
    "ITEM_NAMES",
    "Checkpoint",
    "CheckpointDtypeMismatchError",
    "CheckpointMetadata",
    "CheckpointNotFoundError",
    "CheckpointNotWrittenError",
    "CheckpointStore",
    "DtypeMismatch",
    "OrbaxCheckpointStore",
    "Producer",
    "UnsupportedCheckpointError",
    "resolve_checkpoint_dir",
]
