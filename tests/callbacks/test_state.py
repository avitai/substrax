"""Callback state: a stopped run resumes with the stopping decision it would have made.

Each stateful piece hands its bookkeeping to ``get_state`` and takes it back with ``set_state``,
so a checkpoint written mid-run and restored into fresh objects continues exactly: the same
best value, the same count of epochs without improvement, the same stopping epoch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from substrax.callbacks import (
    BaseCallback,
    BestMetricTracker,
    CallbackList,
    EarlyStopping,
    EarlyStoppingCallback,
    EarlyStoppingConfig,
    TrainerLike,
)
from substrax.checkpoint import OrbaxCheckpointStore
from substrax.typing import Checkpointable


_LOSSES = [1.0, 0.8, 0.9, 0.85, 0.95, 0.99, 0.7]


def _stopping_epoch(callback: EarlyStoppingCallback, losses: list[float], start: int) -> int | None:
    for epoch, loss in enumerate(losses, start=start):
        callback.on_epoch_end(None, epoch, {"val_loss": loss})  # type: ignore[arg-type]
        if callback.should_stop:
            return callback.stopped_epoch
    return None


def _early_stopping() -> EarlyStoppingCallback:
    return EarlyStoppingCallback(EarlyStoppingConfig(monitor="val_loss", patience=3))


class _Counter(BaseCallback):
    """A stateful callback of the caller's own: counts epochs."""

    __slots__ = ("epochs",)

    def __init__(self) -> None:
        self.epochs = 0

    def on_epoch_end(self, _trainer: TrainerLike, _epoch: int, _logs: dict[str, Any]) -> None:
        self.epochs += 1

    def get_state(self) -> dict[str, Any]:
        return {"epochs": self.epochs}

    def set_state(self, state: dict[str, Any]) -> None:
        self.epochs = state["epochs"]


def test_the_stateful_pieces_satisfy_the_protocol() -> None:
    for piece in (
        BestMetricTracker(mode="min", min_delta=0.0),
        EarlyStopping(patience=2),
        _early_stopping(),
        CallbackList(),
    ):
        assert isinstance(piece, Checkpointable)
    assert not isinstance(BaseCallback(), Checkpointable)


def test_a_restored_tracker_continues_where_it_stood() -> None:
    tracker = BestMetricTracker(mode="min", min_delta=0.0)
    for value in (1.0, 0.5, 0.7):
        tracker.register(value)

    restored = BestMetricTracker(mode="min", min_delta=0.0)
    restored.set_state(tracker.get_state())

    assert (restored.best, restored.num_bad_epochs) == (0.5, 1)
    assert restored.register(0.6) is tracker.register(0.6) is False
    assert restored.num_bad_epochs == tracker.num_bad_epochs == 2


def test_a_restored_stopper_keeps_its_patience_used() -> None:
    stopper = EarlyStopping(patience=2)
    for value in (1.0, 1.1):
        stopper.update(value)

    restored = EarlyStopping(patience=2)
    restored.set_state(stopper.get_state())
    restored.update(1.2)

    assert restored.should_stop


def test_a_resumed_early_stopping_run_stops_at_the_uninterrupted_epoch() -> None:
    uninterrupted = _stopping_epoch(_early_stopping(), _LOSSES, start=0)

    first = _early_stopping()
    assert _stopping_epoch(first, _LOSSES[:3], start=0) is None
    resumed = _early_stopping()
    resumed.set_state(first.get_state())

    assert uninterrupted == 4
    assert _stopping_epoch(resumed, _LOSSES[3:], start=3) == uninterrupted


def test_a_stopping_decision_survives_the_restore() -> None:
    stopped = _early_stopping()
    _stopping_epoch(stopped, _LOSSES, start=0)

    restored = _early_stopping()
    restored.set_state(stopped.get_state())

    assert (restored.should_stop, restored.stopped_epoch) == (True, 4)


def test_state_written_before_the_metric_appears_survives_the_checkpoint_store(
    tmp_path: Path,
) -> None:
    callbacks = CallbackList([_early_stopping(), BaseCallback(), _Counter()])
    callbacks.on_epoch_end(None, 0, {"loss": 1.0})  # type: ignore[arg-type]

    with OrbaxCheckpointStore(tmp_path) as store:
        store.save(0, {"extensions": {"callbacks": callbacks.get_state()}})
        saved = store.restore(0).items["extensions"]["callbacks"]

    restored = CallbackList([_early_stopping(), BaseCallback(), _Counter()])
    restored.set_state(saved)
    first, _, counter = restored

    assert isinstance(first, EarlyStoppingCallback)
    assert isinstance(counter, _Counter)
    assert (first.best_score, first.wait_count, first.stopped_epoch) == (None, 0, None)
    assert counter.epochs == 1


def test_the_list_restores_each_stateful_callback_by_position() -> None:
    callbacks = CallbackList([_Counter(), BaseCallback(), _Counter()])
    for epoch in range(3):
        callbacks.on_epoch_end(None, epoch, {})  # type: ignore[arg-type]
    first, _, last = callbacks
    assert isinstance(first, _Counter)
    assert isinstance(last, _Counter)
    last.epochs = 7

    restored = CallbackList([_Counter(), BaseCallback(), _Counter()])
    restored.set_state(callbacks.get_state())
    restored_first, _, restored_last = restored

    assert isinstance(restored_first, _Counter)
    assert isinstance(restored_last, _Counter)
    assert (restored_first.epochs, restored_last.epochs) == (3, 7)


def test_a_list_built_differently_from_the_saved_one_is_refused() -> None:
    state = CallbackList([_Counter(), BaseCallback()]).get_state()

    with pytest.raises(ValueError, match="callback"):
        CallbackList([BaseCallback(), _Counter()]).set_state(state)


def test_the_callback_refuses_state_of_another_shape() -> None:
    with pytest.raises(ValueError, match=r"stopped_epoch"):
        _early_stopping().set_state({"stopper": {"best": 0.5, "num_bad_epochs": 0}})


@pytest.mark.parametrize(
    "state",
    [{"best": 0.5}, {"best": 0.5, "num_bad_epochs": 1, "patience": 3}],
    ids=["missing", "extra"],
)
def test_a_tracker_refuses_state_of_another_shape(state: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match=r"num_bad_epochs|patience"):
        BestMetricTracker(mode="min", min_delta=0.0).set_state(state)
