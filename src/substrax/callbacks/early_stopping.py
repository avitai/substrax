"""Best-metric tracking and early stopping.

``BestMetricTracker`` is the bookkeeping every plateau-driven control shares (best value
so far, consecutive epochs without a ``min_delta`` improvement). ``EarlyStopping`` composes
it with a patience and optional thresholds and is driven by ``update(value)``;
``EarlyStoppingCallback`` composes ``EarlyStopping`` with the callback protocol and reads
the monitored metric out of the epoch logs. Learning-rate plateau decay is
``optax.contrib.reduce_on_plateau``, not part of this package.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from substrax.callbacks.base import BaseCallback, TrainerLike


class PlateauMode(StrEnum):
    """Whether a monitored metric improves by decreasing or increasing."""

    MIN = "min"
    MAX = "max"


class BestMetricTracker:
    """Best-so-far and stagnation bookkeeping for a monitored metric."""

    __slots__ = ("_best", "_min_delta", "_mode", "_num_bad_epochs")

    def __init__(self, *, mode: PlateauMode | str, min_delta: float) -> None:
        """Set up the tracker.

        Args:
            mode: ``"min"`` (lower is better) or ``"max"`` (higher is better).
            min_delta: The smallest absolute change that counts as an improvement.

        Raises:
            ValueError: If ``min_delta`` is negative.
        """
        if min_delta < 0.0:
            raise ValueError(f"min_delta must be non-negative, got {min_delta}.")
        self._mode = PlateauMode(mode)
        self._min_delta = float(min_delta)
        self._best = math.inf if self._mode is PlateauMode.MIN else -math.inf
        self._num_bad_epochs = 0

    @property
    def mode(self) -> PlateauMode:
        """The improvement direction."""
        return self._mode

    @property
    def best(self) -> float:
        """The best value registered so far (``inf`` or ``-inf`` before the first)."""
        return self._best

    @property
    def num_bad_epochs(self) -> int:
        """Consecutive registrations without a ``min_delta`` improvement."""
        return self._num_bad_epochs

    def is_improvement(self, value: float) -> bool:
        """Whether ``value`` beats the best so far by at least ``min_delta``."""
        if self._mode is PlateauMode.MIN:
            return value < self._best - self._min_delta
        return value > self._best + self._min_delta

    def register(self, value: float) -> bool:
        """Record a value; return whether it improved on the best so far.

        Args:
            value: The latest metric value.

        Returns:
            ``True`` on an improvement (which also resets the stagnation count).
        """
        if self.is_improvement(value):
            self._best = float(value)
            self._num_bad_epochs = 0
            return True
        self._num_bad_epochs += 1
        return False

    def reset_stagnation(self) -> None:
        """Clear the stagnation count without touching the best value."""
        self._num_bad_epochs = 0


class EarlyStopping:
    """Signal to stop once a monitored metric stagnates, diverges, or reaches a goal."""

    __slots__ = (
        "_check_finite",
        "_divergence_threshold",
        "_patience",
        "_stopped",
        "_stopping_threshold",
        "_tracker",
    )

    def __init__(
        self,
        *,
        patience: int,
        min_delta: float = 0.0,
        mode: PlateauMode | str = PlateauMode.MIN,
        check_finite: bool = True,
        stopping_threshold: float | None = None,
        divergence_threshold: float | None = None,
    ) -> None:
        """Set up the stopper.

        Args:
            patience: Epochs without a ``min_delta`` improvement before stopping.
            min_delta: The smallest absolute change that counts as an improvement.
            mode: ``"min"`` (lower is better) or ``"max"`` (higher is better).
            check_finite: Whether a NaN or infinite value stops immediately.
            stopping_threshold: Stop once the metric reaches this value (the goal).
            divergence_threshold: In ``"min"`` mode, stop once the metric exceeds this value.

        Raises:
            ValueError: If ``patience`` is not positive.
        """
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}.")
        self._tracker = BestMetricTracker(mode=mode, min_delta=min_delta)
        self._patience = patience
        self._check_finite = check_finite
        self._stopping_threshold = stopping_threshold
        self._divergence_threshold = divergence_threshold
        self._stopped = False

    @property
    def best(self) -> float:
        """The best value registered so far."""
        return self._tracker.best

    @property
    def num_bad_epochs(self) -> int:
        """Consecutive updates without a ``min_delta`` improvement."""
        return self._tracker.num_bad_epochs

    @property
    def should_stop(self) -> bool:
        """Whether training should stop."""
        return self._stopped or self._tracker.num_bad_epochs >= self._patience

    def update(self, value: float) -> bool:
        """Record the latest metric value.

        Args:
            value: The metric value for this epoch.

        Returns:
            Whether it improved on the best so far.
        """
        if self._check_finite and not math.isfinite(value):
            self._stopped = True
            return False
        if self._reached_goal(value) or self._diverged(value):
            self._stopped = True
        return self._tracker.register(value)

    def _reached_goal(self, value: float) -> bool:
        if self._stopping_threshold is None:
            return False
        if self._tracker.mode is PlateauMode.MIN:
            return value <= self._stopping_threshold
        return value >= self._stopping_threshold

    def _diverged(self, value: float) -> bool:
        return (
            self._divergence_threshold is not None
            and self._tracker.mode is PlateauMode.MIN
            and value > self._divergence_threshold
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class EarlyStoppingConfig:
    """How ``EarlyStoppingCallback`` reads and judges the epoch logs.

    Attributes:
        monitor: The metric name to read from the epoch logs.
        min_delta: The smallest absolute change that counts as an improvement.
        patience: Epochs without an improvement before stopping.
        mode: ``"min"`` (lower is better) or ``"max"`` (higher is better).
        check_finite: Whether a NaN or infinite value stops immediately.
        stopping_threshold: Stop once the metric reaches this value.
        divergence_threshold: In ``"min"`` mode, stop once the metric exceeds this value.
    """

    monitor: str = "val_loss"
    min_delta: float = 0.0
    patience: int = 10
    mode: PlateauMode = PlateauMode.MIN
    check_finite: bool = True
    stopping_threshold: float | None = None
    divergence_threshold: float | None = None


class EarlyStoppingCallback(BaseCallback):
    """Drive an ``EarlyStopping`` from the epoch logs a training loop emits."""

    __slots__ = ("_config", "_stopped_epoch", "_stopper")

    def __init__(self, config: EarlyStoppingConfig) -> None:
        """Set up the callback.

        Args:
            config: What to monitor and when to stop.
        """
        self._config = config
        self._stopper = EarlyStopping(
            patience=config.patience,
            min_delta=config.min_delta,
            mode=config.mode,
            check_finite=config.check_finite,
            stopping_threshold=config.stopping_threshold,
            divergence_threshold=config.divergence_threshold,
        )
        self._stopped_epoch: int | None = None

    @property
    def should_stop(self) -> bool:
        """Whether training should stop."""
        return self._stopper.should_stop

    @property
    def stopped_epoch(self) -> int | None:
        """The epoch at which stopping was decided, if it has been."""
        return self._stopped_epoch

    @property
    def best(self) -> float | None:
        """The best monitored value so far, or ``None`` before the metric first appears."""
        best = self._stopper.best
        return None if math.isinf(best) else best

    def on_epoch_end(self, _trainer: TrainerLike, epoch: int, logs: dict[str, Any]) -> None:
        """Read the monitored metric, if present, and decide whether to stop."""
        value = logs.get(self._config.monitor)
        if value is None:
            return
        self._stopper.update(float(value))
        if self._stopper.should_stop and self._stopped_epoch is None:
            self._stopped_epoch = epoch
