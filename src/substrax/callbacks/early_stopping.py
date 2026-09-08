"""Early stopping callback driven by epoch logs.

Monitors a metric in the epoch logs and stops training when it stops improving, reaches a
goal, diverges, or becomes non-finite. The best-so-far and stagnation bookkeeping is
``BestMetricTracker``; this class adds the log lookup, the thresholds and the epoch record.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from substrax.callbacks.base import BaseCallback, TrainerLike
from substrax.callbacks.plateau import BestMetricTracker, PlateauMode


@dataclass(frozen=True, slots=True, kw_only=True)
class EarlyStoppingConfig:
    """Configuration for early stopping.

    Attributes:
        monitor: Metric name to monitor (e.g., "val_loss", "accuracy").
        min_delta: Minimum change to qualify as an improvement.
        patience: Number of epochs with no improvement before stopping.
        mode: "min" if lower is better, "max" if higher is better.
        check_finite: If True, stop when metric becomes NaN or Inf.
        stopping_threshold: Stop immediately when metric reaches this value.
        divergence_threshold: Stop if metric exceeds this value (min mode only).
    """

    monitor: str = "val_loss"
    min_delta: float = 0.0
    patience: int = 10
    mode: Literal["min", "max"] = "min"
    check_finite: bool = True
    stopping_threshold: float | None = None
    divergence_threshold: float | None = None


class EarlyStoppingCallback(BaseCallback):
    """Stop training when a monitored metric stops improving."""

    __slots__ = ("_stopped_epoch", "_tracker", "config")

    def __init__(self, config: EarlyStoppingConfig) -> None:
        """Initialize early stopping callback.

        Args:
            config: Early stopping configuration.
        """
        self.config = config
        self._tracker = BestMetricTracker(mode=config.mode, min_delta=config.min_delta)
        self._stopped_epoch: int | None = None

    @property
    def wait_count(self) -> int:
        """Epochs since the last improvement."""
        return self._tracker.num_bad_epochs

    @property
    def best_score(self) -> float | None:
        """The best monitored value so far, or ``None`` before the metric first appears."""
        best = self._tracker.best
        return None if math.isinf(best) else best

    @property
    def stopped_epoch(self) -> int | None:
        """The epoch at which stopping was decided, if it has been."""
        return self._stopped_epoch

    @property
    def should_stop(self) -> bool:
        """Whether training should stop."""
        return self._stopped_epoch is not None

    def on_epoch_end(self, _trainer: TrainerLike, epoch: int, logs: dict[str, Any]) -> None:
        """Read the monitored metric, if present, and decide whether to stop.

        Args:
            _trainer: The trainer instance (unused).
            epoch: Current epoch number.
            logs: Dictionary of metrics from this epoch.
        """
        value = logs.get(self.config.monitor)
        if value is None:
            return
        current = float(value)
        if self._stops_immediately(current):
            self._stopped_epoch = epoch
            return
        self._tracker.register(current)
        if self._tracker.num_bad_epochs >= self.config.patience:
            self._stopped_epoch = epoch

    def _stops_immediately(self, current: float) -> bool:
        """Whether ``current`` ends training before patience is considered."""
        if self.config.check_finite and not math.isfinite(current):
            return True
        return self._meets_threshold(current) or self._diverged(current)

    def _meets_threshold(self, current: float) -> bool:
        """Whether ``current`` reaches the stopping threshold (the training goal)."""
        threshold = self.config.stopping_threshold
        if threshold is None:
            return False
        if self._tracker.mode is PlateauMode.MIN:
            return current <= threshold
        return current >= threshold

    def _diverged(self, current: float) -> bool:
        """Whether ``current`` exceeds the divergence threshold in min mode."""
        threshold = self.config.divergence_threshold
        return (
            threshold is not None and self._tracker.mode is PlateauMode.MIN and current > threshold
        )
