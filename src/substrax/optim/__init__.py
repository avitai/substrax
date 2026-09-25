"""One optimizer specification, one builder over optax, and the learning rate read on device.

``OptimizerConfig`` names the optimizer in optax's terms; ``create_transformation`` and
``create_optimizer`` build it, passing a schedule as the base optimizer's learning rate and
the weight-decay filter as a static mask; ``current_learning_rate`` reads the rate the last
update applied. ``EXCLUDE_BIAS_AND_NORM_SCALE`` is the filter that decays neither biases nor
normalisation scales. ``update_with_line_search`` steps an optimizer that searches along its
direction (``optax.lbfgs``); ``switch_at`` hands over from one transformation to another after a
fixed number of updates; ``with_strong_state_types`` keeps a transformation's state types fixed
across updates, so a jitted step compiles once.
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
from substrax.optim.line_search import switch_at, update_with_line_search
from substrax.optim.state_types import with_strong_state_types


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
    "switch_at",
    "update_with_line_search",
    "weight_decay_mask",
    "with_strong_state_types",
]
