"""The optimizer specification: one frozen dataclass whose fields follow optax."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, get_args, Literal, TYPE_CHECKING

from flax import nnx


if TYPE_CHECKING:
    import optax


OptimizerType = Literal["adam", "adamw", "sgd", "rmsprop", "adagrad", "lamb", "radam", "nadam"]
OPTIMIZER_TYPES: Final[tuple[str, ...]] = get_args(OptimizerType)
DECOUPLED_DECAY_TYPES: Final = frozenset({"adamw", "lamb"})
MOMENTUM_TYPES: Final = frozenset({"sgd", "rmsprop"})


# The eleven fields mirror optax's arguments, each a documented knob; one over pylint's ten.
@dataclass(frozen=True, slots=True, kw_only=True)
class OptimizerConfig:  # pylint: disable=too-many-instance-attributes
    """What an optimizer is, in optax's terms.

    A schedule is passed as the base optimizer's learning rate, never as a
    ``scale_by_schedule`` placed before an adaptive optimizer, whose normalisation would
    cancel the scaling. The names map onto other libraries as follows.

    | Field | PyTorch | Flax NNX |
    | --- | --- | --- |
    | ``learning_rate`` | ``lr`` | the base optax alias's ``learning_rate`` |
    | ``b1``, ``b2`` | ``betas`` | ``b1``, ``b2`` |
    | ``weight_decay`` | ``weight_decay`` on a parameter group | ``optax.adamw(weight_decay=...)`` |
    | ``weight_decay_filter`` | the parameter group | ``optax.adamw(mask=...)``, True where decayed |
    | ``wrt`` | ``model.parameters()`` | ``nnx.Optimizer(model, tx, wrt=...)`` |

    Attributes:
        optimizer_type: The optax alias; ``nadam`` is ``adam`` with Nesterov momentum.
        learning_rate: A positive constant, or an ``optax.Schedule`` of the step count.
        b1: First-moment decay of the Adam family and LAMB.
        b2: Second-moment decay of the Adam family and LAMB.
        eps: The denominator offset.
        momentum: Momentum of ``sgd`` and ``rmsprop``; ``None`` for no momentum.
        weight_decay: Decoupled weight decay, ``adamw`` and ``lamb`` only.
        weight_decay_filter: A Flax NNX filter naming the parameters that are decayed, the
            polarity of optax's ``mask``; ``None`` decays every leaf in ``wrt``, as
            ``torch.optim.AdamW(model.parameters())`` does. ``EXCLUDE_BIAS_AND_NORM_SCALE``
            is the Hugging Face and timm convention.
        gradient_clip_norm: Global-norm clipping applied before the update, or ``None``.
        gradient_clip_value: Element-wise clipping applied before the update, or ``None``.
        wrt: The filter naming the parameters the optimizer updates.
    """

    optimizer_type: OptimizerType = "adam"
    learning_rate: float | optax.Schedule
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8
    momentum: float | None = None
    weight_decay: float = 0.0
    weight_decay_filter: nnx.filterlib.Filter | None = None
    gradient_clip_norm: float | None = None
    gradient_clip_value: float | None = None
    wrt: nnx.filterlib.Filter = nnx.Param

    def __post_init__(self) -> None:  # noqa: DOC502
        """Refuse a specification optax would silently misread.

        Raises:
            TypeError: If ``weight_decay_filter`` is a string, which NNX reads as a tag, not
                a path.
            ValueError: If both clip fields are set, ``weight_decay`` is negative or set for an
                optimizer without decoupled decay, ``momentum`` is set for an optimizer that
                has none, or a constant ``learning_rate`` is not positive.
        """
        _check_clipping(self.gradient_clip_norm, self.gradient_clip_value)
        _check_weight_decay(self.optimizer_type, self.weight_decay)
        _check_momentum(self.optimizer_type, self.momentum)
        _check_filter(self.weight_decay_filter)
        _check_learning_rate(self.learning_rate)


def _check_clipping(clip_norm: float | None, clip_value: float | None) -> None:
    if clip_norm is not None and clip_value is not None:
        raise ValueError(
            "gradient_clip_norm and gradient_clip_value cannot both be set; choose one"
        )


def _check_weight_decay(optimizer_type: str, weight_decay: float) -> None:
    if weight_decay < 0.0:
        raise ValueError(f"weight_decay must be non-negative, got {weight_decay}")
    if weight_decay > 0.0 and optimizer_type not in DECOUPLED_DECAY_TYPES:
        raise ValueError(
            f"{optimizer_type} has no decoupled weight decay; weight_decay is for "
            f"{' and '.join(sorted(DECOUPLED_DECAY_TYPES))}"
        )


def _check_momentum(optimizer_type: str, momentum: float | None) -> None:
    if momentum is not None and optimizer_type not in MOMENTUM_TYPES:
        raise ValueError(
            f"{optimizer_type} takes no momentum; momentum is for "
            f"{' and '.join(sorted(MOMENTUM_TYPES))}"
        )


def _check_filter(weight_decay_filter: nnx.filterlib.Filter | None) -> None:
    if isinstance(weight_decay_filter, str):
        raise TypeError(
            f"weight_decay_filter {weight_decay_filter!r} is a string, which NNX reads as a tag, "
            "not a path; use nnx.PathContains or a predicate"
        )


def _check_learning_rate(learning_rate: float | optax.Schedule) -> None:
    if not callable(learning_rate) and learning_rate <= 0.0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}")
