"""Tests for EarlyStoppingCallback callback.

Following TDD principles - these tests define the expected behavior
for the EarlyStoppingCallback callback.
"""

import time
from unittest.mock import MagicMock

import jax.numpy as jnp
import pytest

from substrax.callbacks import BaseCallback, EarlyStoppingCallback, EarlyStoppingConfig


class TestEarlyStoppingConfig:
    """Test EarlyStoppingConfig dataclass."""

    def test_config_exists(self) -> None:
        """EarlyStoppingConfig should be importable."""

        assert EarlyStoppingConfig is not None

    def test_config_default_values(self) -> None:
        """EarlyStoppingConfig should have sensible defaults."""

        config = EarlyStoppingConfig()
        assert config.monitor == "val_loss"
        assert config.min_delta == 0.0
        assert config.patience == 10
        assert config.mode == "min"
        assert config.check_finite is True
        assert config.stopping_threshold is None
        assert config.divergence_threshold is None

    def test_config_custom_values(self) -> None:
        """EarlyStoppingConfig should accept custom values."""

        config = EarlyStoppingConfig(
            monitor="accuracy",
            min_delta=0.01,
            patience=5,
            mode="max",
            check_finite=False,
            stopping_threshold=0.99,
            divergence_threshold=10.0,
        )
        assert config.monitor == "accuracy"
        assert config.min_delta == 0.01
        assert config.patience == 5
        assert config.mode == "max"
        assert config.check_finite is False
        assert config.stopping_threshold == 0.99
        assert config.divergence_threshold == 10.0


class TestEarlyStoppingBasic:
    """Test basic EarlyStoppingCallback functionality."""

    def test_early_stopping_exists(self) -> None:
        """EarlyStoppingCallback should be importable."""

        assert EarlyStoppingCallback is not None

    def test_early_stopping_inherits_base_callback(self) -> None:
        """EarlyStoppingCallback should inherit from BaseCallback."""

        callback = EarlyStoppingCallback(EarlyStoppingConfig())
        assert isinstance(callback, BaseCallback)

    def test_early_stopping_initializes_state(self) -> None:
        """EarlyStoppingCallback should initialize tracking state."""

        callback = EarlyStoppingCallback(EarlyStoppingConfig())
        assert callback.wait_count == 0
        assert callback.best_score is None
        assert callback.stopped_epoch is None

    def test_early_stopping_has_should_stop_property(self) -> None:
        """EarlyStoppingCallback should have should_stop property."""

        callback = EarlyStoppingCallback(EarlyStoppingConfig())
        assert hasattr(callback, "should_stop")
        assert callback.should_stop is False


class TestEarlyStoppingMinMode:
    """Test EarlyStoppingCallback in 'min' mode (lower is better)."""

    def test_improvement_resets_wait_count(self) -> None:
        """Improvement should reset wait count."""

        callback = EarlyStoppingCallback(EarlyStoppingConfig(monitor="loss", patience=3))
        trainer_mock = MagicMock()

        # First epoch - establishes baseline
        callback.on_epoch_end(trainer_mock, 0, {"loss": 1.0})
        assert callback.best_score == 1.0
        assert callback.wait_count == 0

        # No improvement
        callback.on_epoch_end(trainer_mock, 1, {"loss": 1.0})
        assert callback.wait_count == 1

        # Improvement - should reset
        callback.on_epoch_end(trainer_mock, 2, {"loss": 0.9})
        assert callback.best_score == 0.9
        assert callback.wait_count == 0

    def test_no_improvement_increments_wait_count(self) -> None:
        """No improvement should increment wait count."""

        callback = EarlyStoppingCallback(EarlyStoppingConfig(monitor="loss", patience=3))
        trainer_mock = MagicMock()

        callback.on_epoch_end(trainer_mock, 0, {"loss": 1.0})
        callback.on_epoch_end(trainer_mock, 1, {"loss": 1.1})
        assert callback.wait_count == 1

        callback.on_epoch_end(trainer_mock, 2, {"loss": 1.2})
        assert callback.wait_count == 2

    def test_patience_exceeded_sets_should_stop(self) -> None:
        """Exceeding patience should set should_stop."""

        callback = EarlyStoppingCallback(EarlyStoppingConfig(monitor="loss", patience=2))
        trainer_mock = MagicMock()

        callback.on_epoch_end(trainer_mock, 0, {"loss": 1.0})
        assert callback.should_stop is False

        callback.on_epoch_end(trainer_mock, 1, {"loss": 1.1})
        assert callback.should_stop is False

        callback.on_epoch_end(trainer_mock, 2, {"loss": 1.2})
        assert callback.should_stop is True
        assert callback.stopped_epoch == 2


class TestEarlyStoppingMaxMode:
    """Test EarlyStoppingCallback in 'max' mode (higher is better)."""

    def test_improvement_in_max_mode(self) -> None:
        """Higher values should be improvements in max mode."""

        callback = EarlyStoppingCallback(
            EarlyStoppingConfig(monitor="accuracy", mode="max", patience=3)
        )
        trainer_mock = MagicMock()

        callback.on_epoch_end(trainer_mock, 0, {"accuracy": 0.8})
        assert callback.best_score == 0.8

        # Higher is improvement
        callback.on_epoch_end(trainer_mock, 1, {"accuracy": 0.85})
        assert callback.best_score == 0.85
        assert callback.wait_count == 0

        # Lower is not improvement
        callback.on_epoch_end(trainer_mock, 2, {"accuracy": 0.84})
        assert callback.wait_count == 1


class TestEarlyStoppingMinDelta:
    """Test min_delta threshold for improvements."""

    def test_min_delta_ignores_small_improvements(self) -> None:
        """Small improvements below min_delta should not count."""

        callback = EarlyStoppingCallback(
            EarlyStoppingConfig(monitor="loss", min_delta=0.1, patience=3)
        )
        trainer_mock = MagicMock()

        callback.on_epoch_end(trainer_mock, 0, {"loss": 1.0})
        assert callback.best_score == 1.0

        # Small improvement (0.05 < min_delta=0.1) - should not count
        callback.on_epoch_end(trainer_mock, 1, {"loss": 0.95})
        assert callback.wait_count == 1

        # Large improvement (0.15 > min_delta=0.1) - should count
        callback.on_epoch_end(trainer_mock, 2, {"loss": 0.8})
        assert callback.best_score == 0.8
        assert callback.wait_count == 0


class TestEarlyStoppingCheckFinite:
    """Test check_finite for NaN/Inf detection."""

    def test_nan_triggers_stop(self) -> None:
        """NaN values should trigger immediate stop when check_finite=True."""

        callback = EarlyStoppingCallback(
            EarlyStoppingConfig(monitor="loss", check_finite=True, patience=10)
        )
        trainer_mock = MagicMock()

        callback.on_epoch_end(trainer_mock, 0, {"loss": 1.0})
        callback.on_epoch_end(trainer_mock, 1, {"loss": float("nan")})
        assert callback.should_stop is True

    def test_inf_triggers_stop(self) -> None:
        """Inf values should trigger immediate stop when check_finite=True."""

        callback = EarlyStoppingCallback(
            EarlyStoppingConfig(monitor="loss", check_finite=True, patience=10)
        )
        trainer_mock = MagicMock()

        callback.on_epoch_end(trainer_mock, 0, {"loss": 1.0})
        callback.on_epoch_end(trainer_mock, 1, {"loss": float("inf")})
        assert callback.should_stop is True

    def test_check_finite_disabled(self) -> None:
        """NaN should not trigger stop when check_finite=False."""

        callback = EarlyStoppingCallback(
            EarlyStoppingConfig(monitor="loss", check_finite=False, patience=10)
        )
        trainer_mock = MagicMock()

        callback.on_epoch_end(trainer_mock, 0, {"loss": 1.0})
        callback.on_epoch_end(trainer_mock, 1, {"loss": float("nan")})
        # Should not stop due to NaN, but wait_count increases
        assert callback.should_stop is False


class TestEarlyStoppingThresholds:
    """Test stopping and divergence thresholds."""

    def test_stopping_threshold_min_mode(self) -> None:
        """Reaching stopping_threshold should trigger stop in min mode."""

        callback = EarlyStoppingCallback(
            EarlyStoppingConfig(monitor="loss", mode="min", stopping_threshold=0.1)
        )
        trainer_mock = MagicMock()

        callback.on_epoch_end(trainer_mock, 0, {"loss": 0.5})
        assert callback.should_stop is False

        callback.on_epoch_end(trainer_mock, 1, {"loss": 0.09})
        assert callback.should_stop is True

    def test_stopping_threshold_max_mode(self) -> None:
        """Reaching stopping_threshold should trigger stop in max mode."""

        callback = EarlyStoppingCallback(
            EarlyStoppingConfig(monitor="accuracy", mode="max", stopping_threshold=0.99)
        )
        trainer_mock = MagicMock()

        callback.on_epoch_end(trainer_mock, 0, {"accuracy": 0.9})
        assert callback.should_stop is False

        callback.on_epoch_end(trainer_mock, 1, {"accuracy": 0.995})
        assert callback.should_stop is True

    def test_divergence_threshold(self) -> None:
        """Exceeding divergence_threshold should trigger stop."""

        callback = EarlyStoppingCallback(
            EarlyStoppingConfig(monitor="loss", divergence_threshold=10.0)
        )
        trainer_mock = MagicMock()

        callback.on_epoch_end(trainer_mock, 0, {"loss": 1.0})
        assert callback.should_stop is False

        callback.on_epoch_end(trainer_mock, 1, {"loss": 15.0})
        assert callback.should_stop is True


class TestEarlyStoppingMissingMetric:
    """Test behavior when monitored metric is missing."""

    def test_missing_metric_does_not_crash(self) -> None:
        """Missing metric should not crash, just skip."""

        callback = EarlyStoppingCallback(EarlyStoppingConfig(monitor="val_loss"))
        trainer_mock = MagicMock()

        # Metric not present
        callback.on_epoch_end(trainer_mock, 0, {"loss": 1.0})
        assert callback.best_score is None
        assert callback.wait_count == 0

    def test_metric_appears_later(self) -> None:
        """Callback should work when metric appears after first epoch."""

        callback = EarlyStoppingCallback(EarlyStoppingConfig(monitor="val_loss", patience=3))
        trainer_mock = MagicMock()

        # First epoch - metric missing
        callback.on_epoch_end(trainer_mock, 0, {"loss": 1.0})
        assert callback.best_score is None

        # Second epoch - metric appears
        callback.on_epoch_end(trainer_mock, 1, {"val_loss": 0.5})
        assert callback.best_score == 0.5


class TestEarlyStoppingJaxArrays:
    """Test EarlyStoppingCallback with JAX arrays."""

    def test_works_with_jax_arrays(self) -> None:
        """Should work with JAX array values."""

        callback = EarlyStoppingCallback(EarlyStoppingConfig(monitor="loss", patience=3))
        trainer_mock = MagicMock()

        callback.on_epoch_end(trainer_mock, 0, {"loss": jnp.array(1.0)})
        callback.on_epoch_end(trainer_mock, 1, {"loss": jnp.array(0.9)})
        assert callback.best_score == pytest.approx(0.9)
        assert callback.wait_count == 0


class TestEarlyStoppingOverhead:
    """Test that EarlyStoppingCallback has minimal overhead."""

    def test_overhead_is_minimal(self) -> None:
        """EarlyStoppingCallback overhead should be minimal."""
        callback = EarlyStoppingCallback(EarlyStoppingConfig(monitor="loss"))
        trainer_mock = MagicMock()
        logs = {"loss": 0.5}

        # Warmup
        for i in range(100):
            callback.on_epoch_end(trainer_mock, i, logs)

        # Reset state
        callback = EarlyStoppingCallback(EarlyStoppingConfig(monitor="loss"))

        # Benchmark
        iterations = 10_000
        start = time.perf_counter()
        for i in range(iterations):
            callback.on_epoch_end(trainer_mock, i, {"loss": 0.5 - i * 0.00001})
        elapsed = time.perf_counter() - start

        # Should be < 10 microseconds per call
        avg_time_us = (elapsed / iterations) * 1_000_000
        assert avg_time_us < 10.0, f"EarlyStoppingCallback overhead too high: {avg_time_us:.3f}us"
