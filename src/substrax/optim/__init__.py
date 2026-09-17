"""One optimizer specification, one builder over optax, and the learning rate read on device.

``OptimizerConfig`` names the optimizer in optax's terms; ``create_transformation`` and
``create_optimizer`` build it, passing a schedule as the base optimizer's learning rate and
the weight-decay filter as a static mask; ``current_learning_rate`` reads the rate the last
update applied. ``EXCLUDE_BIAS_AND_NORM_SCALE`` is the filter that decays neither biases nor
normalisation scales.
"""

from substrax.optim.build import create_optimizer, create_transformation
from substrax.optim.config import (
    DECOUPLED_DECAY_TYPES,
    MOMENTUM_TYPES,
    OPTIMIZER_TYPES,
    OptimizerConfig,
    OptimizerType,
)
from substrax.optim.filters import (
    EXCLUDE_BIAS_AND_NORM_SCALE,
    mask_callable,
    mask_from_filter,
    weight_decay_mask,
)
from substrax.optim.learning_rate import current_learning_rate


__all__ = [
    "DECOUPLED_DECAY_TYPES",
    "EXCLUDE_BIAS_AND_NORM_SCALE",
    "MOMENTUM_TYPES",
    "OPTIMIZER_TYPES",
    "OptimizerConfig",
    "OptimizerType",
    "create_optimizer",
    "create_transformation",
    "current_learning_rate",
    "mask_callable",
    "mask_from_filter",
    "weight_decay_mask",
]
