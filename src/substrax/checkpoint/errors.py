"""The errors a checkpoint store raises."""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True, kw_only=True)
class DtypeMismatch:
    """One array a restore would give another dtype than the one it was saved with.

    ``item`` is the Orbax item on disk and ``leaf`` the array's path inside it joined with
    ``/``. ``saved`` is the dtype written; ``restored`` the dtype the restore gives it: the
    template leaf's, or, without a template, the one this process creates for the saved array.
    """

    item: str
    leaf: str
    saved: str
    restored: str


# The 32-bit dtype jax creates for a 64-bit one while its x64 mode is off.
_WITHOUT_X64 = {
    "float64": "float32",
    "int64": "int32",
    "uint64": "uint32",
    "complex128": "complex64",
}


class CheckpointDtypeMismatchError(ValueError):
    """A restore would give saved arrays another dtype, so it is refused.

    Orbax casts every array to its template leaf's dtype, and without a template jax creates a
    64-bit array at 32 bits while its x64 mode is off. ``step`` is the checkpoint's step and
    ``mismatches`` every array concerned; ``restore(..., cast_dtypes=True)`` accepts the cast.
    """

    def __init__(
        self, *, step: int, mismatches: tuple[DtypeMismatch, ...], x64_disabled: bool = False
    ) -> None:
        """Record the step and every array concerned.

        Args:
            step: The checkpoint's step.
            mismatches: Every array whose restored dtype differs from its saved one.
            x64_disabled: Whether jax's x64 mode was off for a restore without templates, which
                adds how to keep the saved precision to the message.
        """
        self.step = step
        self.mismatches = mismatches
        listed = "; ".join(
            f"{mismatch.item} {mismatch.leaf}: saved {mismatch.saved}, restored as {mismatch.restored}"
            for mismatch in mismatches
        )
        message = (
            f"checkpoint step {step} would restore arrays at another dtype than they were saved "
            f"with ({listed}); pass cast_dtypes=True to accept the cast"
        )
        narrowed = any(_WITHOUT_X64.get(m.saved) == m.restored for m in mismatches)
        if x64_disabled and narrowed:
            message += (
                ", or enable jax's x64 mode (JAX_ENABLE_X64=1 or jax.enable_x64(True)) to keep "
                "the 64-bit arrays"
            )
        super().__init__(message)
