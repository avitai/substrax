"""Rewrite a checkpoint root in the current format, into a new root."""

from __future__ import annotations

from pathlib import Path

from substrax.checkpoint.legacy import LegacyLayout
from substrax.checkpoint.store import OrbaxCheckpointStore


def upgrade_checkpoints(  # noqa: DOC503  # CheckpointDtypeMismatchError is raised by restore
    source: str | Path, destination: str | Path, *, legacy_layout: LegacyLayout | None = None
) -> list[int]:
    """Write every step under ``source`` to ``destination`` in the current format.

    The source is read as it is (a format-2 root through the migration registry, split
    by ``legacy_layout``) and never modified; the destination must not exist or must be
    empty, so an upgrade never rewrites a root in place. Items are copied as stored, so
    the arrays keep their saved placement and dtype: a step whose arrays this process
    cannot restore at their saved dtype is refused rather than written narrowed.

    Args:
        source: The root to read.
        destination: The new root to write.
        legacy_layout: How a format-2 payload splits into items; module-only when not given.

    Returns:
        The steps written, ascending.

    Raises:
        ValueError: If ``source`` and ``destination`` are the same directory, or the
            destination exists and is not empty.
        CheckpointDtypeMismatchError: If a step's arrays would come back with another dtype
            than they were saved with, such as a 64-bit array while jax's x64 mode is off; the
            destination holds the steps written before it.
    """
    source_dir = Path(source).resolve()
    destination_dir = Path(destination).resolve()
    if source_dir == destination_dir:
        raise ValueError(f"source and destination are the same directory: {source_dir}")
    if destination_dir.exists() and any(destination_dir.iterdir()):
        raise ValueError(f"destination {destination_dir} is not empty")

    with (
        OrbaxCheckpointStore(source_dir, max_to_keep=None) as reader,
        OrbaxCheckpointStore(destination_dir, max_to_keep=None) as writer,
    ):
        steps = reader.list_steps()
        for step in steps:
            checkpoint = reader.restore(step, legacy_layout=legacy_layout)
            metadata = checkpoint.metadata
            writer.save(
                step,
                checkpoint.items,
                epoch=metadata.epoch,
                metrics=metadata.metrics,
                producer=metadata.producer,
                extra=metadata.extra,
            )
    return steps
