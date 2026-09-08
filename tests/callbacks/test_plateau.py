"""Tests for the metric-driven training-control callbacks."""

from __future__ import annotations

import pytest

from substrax.callbacks import EarlyStopping


class TestEarlyStopping:
    def test_stops_after_patience_without_improvement(self) -> None:
        stopper = EarlyStopping(patience=3)
        assert stopper.update(1.0) is True  # first value is an improvement
        assert not stopper.should_stop
        for _ in range(2):
            assert stopper.update(1.0) is False  # no improvement
            assert not stopper.should_stop
        assert stopper.update(1.0) is False  # third stagnant epoch
        assert stopper.should_stop

    def test_improvement_resets_the_counter(self) -> None:
        stopper = EarlyStopping(patience=2)
        stopper.update(1.0)
        stopper.update(1.0)  # 1 bad epoch
        assert stopper.update(0.5) is True  # improvement resets
        assert not stopper.should_stop
        assert stopper.best == pytest.approx(0.5)

    def test_min_delta_requires_meaningful_improvement(self) -> None:
        stopper = EarlyStopping(patience=5, min_delta=0.1)
        stopper.update(1.0)
        assert stopper.update(0.95) is False  # below the 0.1 threshold

    def test_max_mode_tracks_increasing_metric(self) -> None:
        stopper = EarlyStopping(patience=2, mode="max")
        assert stopper.update(0.5) is True
        assert stopper.update(0.7) is True
        assert stopper.update(0.6) is False

    def test_rejects_non_positive_patience(self) -> None:
        with pytest.raises(ValueError, match="patience"):
            EarlyStopping(patience=0)
