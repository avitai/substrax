"""The errors a checkpoint store raises."""

from __future__ import annotations

from pathlib import Path
from typing import Literal


WriteRefusal = Literal["exists", "below_latest", "rejected"]


class CheckpointNotWrittenError(RuntimeError):
    """A save was refused, so the step on disk is whatever it was before the call.

    The error carries ``step``, the step the caller asked to write; ``latest_step``, the
    newest step the store held at the time (``None`` for an empty store); and ``reason``:
    ``"exists"`` when the step is already written and ``overwrite`` was not given,
    ``"below_latest"`` when the step is older than the newest one (a store's steps only
    grow), ``"rejected"`` when Orbax declined the write.
    """

    def __init__(self, *, step: int, latest_step: int | None, reason: WriteRefusal) -> None:
        """Record the refused step, the store's latest step and the reason."""
        self.step = step
        self.latest_step = latest_step
        self.reason = reason
        detail = {
            "exists": "it is already written; pass overwrite=True to replace it",
            "below_latest": f"it is below the latest step {latest_step}",
            "rejected": "the Orbax manager declined the write",
        }[reason]
        super().__init__(f"checkpoint step {step} was not written: {detail}")


class CheckpointNotFoundError(FileNotFoundError):
    """The store holds no checkpoint at the requested step."""

    def __init__(self, step: int, directory: Path) -> None:
        """Record the missing step and the store's directory."""
        self.step = step
        self.directory = directory
        super().__init__(f"no checkpoint at step {step} under {directory}")


class UnsupportedCheckpointError(ValueError):
    """The checkpoint's metadata describes a format this substrax cannot read."""
