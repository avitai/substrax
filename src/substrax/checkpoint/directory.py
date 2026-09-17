"""Where a run keeps its checkpoints."""

from __future__ import annotations

from pathlib import Path


CHECKPOINTS_SUBDIRECTORY = "checkpoints"


def resolve_checkpoint_dir(checkpoint_dir: Path | None, run_dir: Path | None) -> Path:
    """Return the directory a run's checkpoints go to, without creating it.

    An explicit ``checkpoint_dir`` wins; otherwise the run's directory holds a
    ``checkpoints`` subdirectory. A relative path resolves against the working directory.
    The store creates the directory on its first save, so resolving one writes nothing.

    Args:
        checkpoint_dir: The directory the caller chose, or ``None``.
        run_dir: The run's directory, whose ``checkpoints`` subdirectory is the default.

    Returns:
        The absolute directory the checkpoints go to.

    Raises:
        ValueError: If neither directory is given.
    """
    if checkpoint_dir is not None:
        return Path(checkpoint_dir).resolve()
    if run_dir is not None:
        return (Path(run_dir) / CHECKPOINTS_SUBDIRECTORY).resolve()
    raise ValueError("pass checkpoint_dir or run_dir: a checkpoint store needs a directory")
