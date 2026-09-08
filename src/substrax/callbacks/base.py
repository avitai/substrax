"""The training-callback protocol, its no-op base and the dispatching list."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any, Protocol, runtime_checkable

from flax import nnx


@runtime_checkable
class TrainerLike(Protocol):
    """What a callback may ask of the trainer that drives it."""

    @property
    def model(self) -> nnx.Module:
        """The model being trained."""
        ...


@runtime_checkable
class TrainingCallback(Protocol):
    """Lifecycle hooks a training loop fires; implement any subset via ``BaseCallback``.

    Parameters are positional-only so implementations may name them as they like.
    """

    def on_train_begin(self, trainer: TrainerLike, /) -> None:
        """Called once before the first epoch."""
        ...

    def on_train_end(self, trainer: TrainerLike, /) -> None:
        """Called once after the last epoch."""
        ...

    def on_epoch_begin(self, trainer: TrainerLike, epoch: int, /) -> None:
        """Called before each epoch."""
        ...

    def on_epoch_end(self, trainer: TrainerLike, epoch: int, logs: dict[str, Any], /) -> None:
        """Called after each epoch with that epoch's metrics."""
        ...

    def on_batch_begin(self, trainer: TrainerLike, batch: int, /) -> None:
        """Called before each batch."""
        ...

    def on_batch_end(self, trainer: TrainerLike, batch: int, logs: dict[str, Any], /) -> None:
        """Called after each batch with that batch's metrics."""
        ...

    def on_validation_begin(self, trainer: TrainerLike, /) -> None:
        """Called before a validation pass."""
        ...

    def on_validation_end(self, trainer: TrainerLike, logs: dict[str, Any], /) -> None:
        """Called after a validation pass with its metrics."""
        ...


class BaseCallback:
    """No-op implementation of every hook; subclasses override what they need."""

    __slots__ = ()

    def on_train_begin(self, _trainer: TrainerLike) -> None:
        """Called once before the first epoch."""

    def on_train_end(self, _trainer: TrainerLike) -> None:
        """Called once after the last epoch."""

    def on_epoch_begin(self, _trainer: TrainerLike, _epoch: int) -> None:
        """Called before each epoch."""

    def on_epoch_end(self, _trainer: TrainerLike, _epoch: int, _logs: dict[str, Any]) -> None:
        """Called after each epoch with that epoch's metrics."""

    def on_batch_begin(self, _trainer: TrainerLike, _batch: int) -> None:
        """Called before each batch."""

    def on_batch_end(self, _trainer: TrainerLike, _batch: int, _logs: dict[str, Any]) -> None:
        """Called after each batch with that batch's metrics."""

    def on_validation_begin(self, _trainer: TrainerLike) -> None:
        """Called before a validation pass."""

    def on_validation_end(self, _trainer: TrainerLike, _logs: dict[str, Any]) -> None:
        """Called after a validation pass with its metrics."""


class CallbackList:
    """Dispatches every hook to its callbacks in registration order."""

    __slots__ = ("_callbacks",)

    def __init__(self, callbacks: Iterable[TrainingCallback] | None = None) -> None:
        """Wrap an initial set of callbacks.

        Args:
            callbacks: The callbacks to start with; each receives every hook, in this order.
        """
        self._callbacks: list[TrainingCallback] = list(callbacks) if callbacks else []

    def add(self, callback: TrainingCallback) -> None:
        """Append a callback."""
        self._callbacks.append(callback)

    def remove(self, callback: TrainingCallback) -> None:
        """Remove a callback; raises ``ValueError`` if it was never added."""
        self._callbacks.remove(callback)

    def __len__(self) -> int:
        """The number of callbacks."""
        return len(self._callbacks)

    def __iter__(self) -> Iterator[TrainingCallback]:
        """Iterate over the callbacks in dispatch order."""
        return iter(self._callbacks)

    def on_train_begin(self, trainer: TrainerLike) -> None:
        """Dispatch ``on_train_begin``."""
        for callback in self._callbacks:
            callback.on_train_begin(trainer)

    def on_train_end(self, trainer: TrainerLike) -> None:
        """Dispatch ``on_train_end``."""
        for callback in self._callbacks:
            callback.on_train_end(trainer)

    def on_epoch_begin(self, trainer: TrainerLike, epoch: int) -> None:
        """Dispatch ``on_epoch_begin``."""
        for callback in self._callbacks:
            callback.on_epoch_begin(trainer, epoch)

    def on_epoch_end(self, trainer: TrainerLike, epoch: int, logs: dict[str, Any]) -> None:
        """Dispatch ``on_epoch_end``."""
        for callback in self._callbacks:
            callback.on_epoch_end(trainer, epoch, logs)

    def on_batch_begin(self, trainer: TrainerLike, batch: int) -> None:
        """Dispatch ``on_batch_begin``."""
        for callback in self._callbacks:
            callback.on_batch_begin(trainer, batch)

    def on_batch_end(self, trainer: TrainerLike, batch: int, logs: dict[str, Any]) -> None:
        """Dispatch ``on_batch_end``."""
        for callback in self._callbacks:
            callback.on_batch_end(trainer, batch, logs)

    def on_validation_begin(self, trainer: TrainerLike) -> None:
        """Dispatch ``on_validation_begin``."""
        for callback in self._callbacks:
            callback.on_validation_begin(trainer)

    def on_validation_end(self, trainer: TrainerLike, logs: dict[str, Any]) -> None:
        """Dispatch ``on_validation_end``."""
        for callback in self._callbacks:
            callback.on_validation_end(trainer, logs)
