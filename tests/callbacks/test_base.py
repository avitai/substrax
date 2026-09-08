"""Tests for the callback protocol and the dispatching list."""

from __future__ import annotations

from typing import Any

from flax import nnx

from substrax.callbacks import BaseCallback, CallbackList, TrainerLike, TrainingCallback


class _Trainer:
    """The smallest thing that satisfies TrainerLike."""

    def __init__(self) -> None:
        self._model = nnx.Linear(1, 1, rngs=nnx.Rngs(0))

    @property
    def model(self) -> nnx.Module:
        return self._model


class _Recorder(BaseCallback):
    """Records every hook it receives, in order."""

    __slots__ = ("events", "name")

    def __init__(self, name: str, events: list[str]) -> None:
        self.name = name
        self.events = events

    def on_train_begin(self, _trainer: TrainerLike) -> None:
        self.events.append(f"{self.name}:train_begin")

    def on_epoch_end(self, _trainer: TrainerLike, epoch: int, logs: dict[str, Any]) -> None:
        self.events.append(f"{self.name}:epoch_end:{epoch}:{logs.get('loss')}")

    def on_validation_end(self, _trainer: TrainerLike, logs: dict[str, Any]) -> None:
        self.events.append(f"{self.name}:validation_end:{len(logs)}")


def test_base_callback_satisfies_the_protocol_and_is_a_no_op() -> None:
    callback = BaseCallback()
    trainer = _Trainer()
    assert isinstance(callback, TrainingCallback)
    assert isinstance(trainer, TrainerLike)
    callback.on_train_begin(trainer)
    callback.on_train_end(trainer)
    callback.on_epoch_begin(trainer, 0)
    callback.on_epoch_end(trainer, 0, {})
    callback.on_batch_begin(trainer, 0)
    callback.on_batch_end(trainer, 0, {})
    callback.on_validation_begin(trainer)
    callback.on_validation_end(trainer, {})


def test_callback_list_dispatches_in_registration_order() -> None:
    events: list[str] = []
    callbacks = CallbackList([_Recorder("a", events), _Recorder("b", events)])
    trainer = _Trainer()

    callbacks.on_train_begin(trainer)
    callbacks.on_epoch_end(trainer, 3, {"loss": 0.5})
    callbacks.on_validation_end(trainer, {"val_loss": 0.4})

    assert events == [
        "a:train_begin",
        "b:train_begin",
        "a:epoch_end:3:0.5",
        "b:epoch_end:3:0.5",
        "a:validation_end:1",
        "b:validation_end:1",
    ]


def test_callback_list_add_remove_len_and_iteration() -> None:
    events: list[str] = []
    first = _Recorder("a", events)
    second = _Recorder("b", events)
    callbacks = CallbackList()
    assert len(callbacks) == 0

    callbacks.add(first)
    callbacks.add(second)
    assert list(callbacks) == [first, second]

    callbacks.remove(first)
    assert list(callbacks) == [second]


def test_callback_list_never_shares_a_default_list() -> None:
    first = CallbackList()
    second = CallbackList()
    first.add(BaseCallback())
    assert len(second) == 0
