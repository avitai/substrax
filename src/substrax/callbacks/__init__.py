"""Training callbacks, best-metric tracking and early stopping."""

from substrax.callbacks.base import BaseCallback, CallbackList, TrainerLike, TrainingCallback
from substrax.callbacks.early_stopping import EarlyStoppingCallback, EarlyStoppingConfig
from substrax.callbacks.plateau import BestMetricTracker, EarlyStopping, PlateauMode


__all__ = [
    "BaseCallback",
    "BestMetricTracker",
    "CallbackList",
    "EarlyStopping",
    "EarlyStoppingCallback",
    "EarlyStoppingConfig",
    "PlateauMode",
    "TrainerLike",
    "TrainingCallback",
]
