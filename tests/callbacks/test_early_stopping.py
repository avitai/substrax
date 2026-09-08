"""Tests for best-metric tracking and early stopping."""

from __future__ import annotations

import math
from typing import Any

import jax.numpy as jnp
import pytest
from hypothesis import given, strategies as st

from substrax.callbacks import (
    BaseCallback,
    BestMetricTracker,
    EarlyStopping,
    EarlyStoppingCallback,
    EarlyStoppingConfig,
    PlateauMode,
)


class TestBestMetricTracker:
    def test_first_value_is_an_improvement_and_sets_the_best(self) -> None:
        tracker = BestMetricTracker(mode=PlateauMode.MIN, min_delta=0.0)
        assert tracker.register(1.0) is True
        assert tracker.best == 1.0
        assert tracker.num_bad_epochs == 0

    def test_stagnation_counts_and_improvement_resets(self) -> None:
        tracker = BestMetricTracker(mode="min", min_delta=0.0)
        tracker.register(1.0)
        assert tracker.register(1.0) is False
        assert tracker.num_bad_epochs == 1
        assert tracker.register(0.5) is True
        assert tracker.num_bad_epochs == 0

    def test_min_delta_requires_a_meaningful_improvement(self) -> None:
        tracker = BestMetricTracker(mode="min", min_delta=0.1)
        tracker.register(1.0)
        assert tracker.register(0.95) is False

    def test_max_mode(self) -> None:
        tracker = BestMetricTracker(mode="max", min_delta=0.0)
        assert tracker.register(0.5) is True
        assert tracker.register(0.7) is True
        assert tracker.register(0.6) is False

    def test_negative_min_delta_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="min_delta"):
            BestMetricTracker(mode="min", min_delta=-0.1)


class TestEarlyStopping:
    def test_stops_after_patience_without_improvement(self) -> None:
        stopper = EarlyStopping(patience=3)
        assert stopper.update(1.0) is True
        for _ in range(2):
            assert stopper.update(1.0) is False
            assert not stopper.should_stop
        stopper.update(1.0)
        assert stopper.should_stop

    def test_improvement_resets_the_counter(self) -> None:
        stopper = EarlyStopping(patience=2)
        stopper.update(1.0)
        stopper.update(1.0)
        assert stopper.update(0.5) is True
        assert not stopper.should_stop
        assert stopper.best == pytest.approx(0.5)

    def test_rejects_non_positive_patience(self) -> None:
        with pytest.raises(ValueError, match="patience"):
            EarlyStopping(patience=0)

    @given(st.lists(st.floats(0.0, 1.0, allow_nan=False), min_size=1, max_size=30))
    def test_a_strictly_improving_sequence_never_stops(self, values: list[float]) -> None:
        stopper = EarlyStopping(patience=1)
        current = 10.0
        for value in sorted(values, reverse=True):
            current = current - 1.0 - value  # strictly decreasing
            stopper.update(current)
        assert not stopper.should_stop

    @given(st.integers(min_value=1, max_value=10))
    def test_a_constant_sequence_stops_exactly_at_patience(self, patience: int) -> None:
        stopper = EarlyStopping(patience=patience)
        stopper.update(1.0)
        for _ in range(patience - 1):
            stopper.update(1.0)
            assert not stopper.should_stop
        stopper.update(1.0)
        assert stopper.should_stop

    def test_non_finite_values_stop_when_checked(self) -> None:
        stopper = EarlyStopping(patience=10, check_finite=True)
        stopper.update(1.0)
        stopper.update(math.nan)
        assert stopper.should_stop

    def test_non_finite_values_are_ignored_when_unchecked(self) -> None:
        stopper = EarlyStopping(patience=10, check_finite=False)
        stopper.update(1.0)
        stopper.update(math.inf)
        assert not stopper.should_stop

    def test_stopping_threshold_in_both_modes(self) -> None:
        low = EarlyStopping(patience=10, stopping_threshold=0.1)
        low.update(0.5)
        assert not low.should_stop
        low.update(0.09)
        assert low.should_stop

        high = EarlyStopping(patience=10, mode="max", stopping_threshold=0.99)
        high.update(0.9)
        high.update(0.995)
        assert high.should_stop

    def test_divergence_threshold_applies_in_min_mode(self) -> None:
        stopper = EarlyStopping(patience=10, divergence_threshold=10.0)
        stopper.update(1.0)
        stopper.update(15.0)
        assert stopper.should_stop


class TestEarlyStoppingCallback:
    def test_is_a_callback_and_reads_the_monitored_metric(self) -> None:
        callback = EarlyStoppingCallback(EarlyStoppingConfig(monitor="loss", patience=2))
        assert isinstance(callback, BaseCallback)
        trainer: Any = object()

        callback.on_epoch_end(trainer, 0, {"loss": 1.0})
        callback.on_epoch_end(trainer, 1, {"loss": 1.1})
        assert not callback.should_stop
        callback.on_epoch_end(trainer, 2, {"loss": 1.2})
        assert callback.should_stop
        assert callback.stopped_epoch == 2

    def test_missing_metric_is_skipped_until_it_appears(self) -> None:
        callback = EarlyStoppingCallback(EarlyStoppingConfig(monitor="val_loss", patience=3))
        trainer: Any = object()

        callback.on_epoch_end(trainer, 0, {"loss": 1.0})
        assert callback.best is None
        callback.on_epoch_end(trainer, 1, {"val_loss": 0.5})
        assert callback.best == pytest.approx(0.5)

    def test_jax_scalars_are_read_as_floats(self) -> None:
        callback = EarlyStoppingCallback(EarlyStoppingConfig(monitor="loss", patience=3))
        trainer: Any = object()

        callback.on_epoch_end(trainer, 0, {"loss": jnp.array(1.0)})
        callback.on_epoch_end(trainer, 1, {"loss": jnp.array(0.9)})
        assert callback.best == pytest.approx(0.9)

    def test_config_defaults_and_immutability(self) -> None:
        config = EarlyStoppingConfig()
        assert (config.monitor, config.patience, config.mode) == ("val_loss", 10, PlateauMode.MIN)
        with pytest.raises(AttributeError):
            config.patience = 3  # type: ignore[misc]
