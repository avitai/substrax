"""The builder: an optax transformation, and an NNX optimizer, from a specification."""

from __future__ import annotations

import jax.numpy as jnp
import optax
from flax import nnx

from substrax.optim.config import OptimizerConfig
from substrax.optim.filters import mask_callable, weight_decay_mask


# Optax saturates its int32 rejection counter. This limit cannot be exceeded,
# so apply_if_finite never gives up and applies an invalid gradient.
_NEVER_APPLY_NON_FINITE = int(jnp.iinfo(jnp.int32).max)


def create_transformation(  # noqa: DOC502
    model: nnx.Module, config: OptimizerConfig
) -> optax.GradientTransformation:
    """Build the optax transformation ``config`` describes for ``model``.

    Clipping, when configured, comes first; the base alias follows, wrapped in
    ``optax.inject_hyperparams`` so its learning rate, a schedule of optax's own step count
    when one is given, can be read on device with ``current_learning_rate``. The weight-decay
    filter becomes a mask of Python booleans over the parameter tree, never optimizer state,
    so the state has the same structure whatever the filter.

    Args:
        model: The model the optimizer updates, which fixes the mask's tree.
        config: The specification.

    Returns:
        The transformation.

    Raises:
        ValueError: If ``config.optimizer_type`` is not an optax alias this builder knows, or
            the weight-decay filter selects no parameter.
    """
    base = _base_transformation(model, config)
    if config.gradient_clip_norm is not None:
        return optax.chain(optax.clip_by_global_norm(config.gradient_clip_norm), base)
    if config.gradient_clip_value is not None:
        return optax.chain(optax.clip(config.gradient_clip_value), base)
    return base


def create_optimizer[M: nnx.Module](model: M, config: OptimizerConfig) -> nnx.Optimizer[M]:
    """Build a native optimizer that skips non-finite gradients without giving up.

    Optax owns rejection and holds the inner optimizer state and its schedule count.
    NNX's attempted-update count still advances. Configured clipping is inside the
    guarded transformation; no clip is selected by default. This checks gradients,
    not the loss, mutable model state or the resulting optimizer/parameter state.

    Args:
        model: The model the optimizer updates.
        config: The specification.

    Returns:
        The optimizer; step it with ``optimizer.update(model, grads)``.
    """
    transformation = optax.apply_if_finite(
        create_transformation(model, config), max_consecutive_errors=_NEVER_APPLY_NON_FINITE
    )
    return nnx.Optimizer(model, transformation, wrt=config.wrt)


def _base_transformation(
    model: nnx.Module, config: OptimizerConfig
) -> optax.GradientTransformation:
    kind = config.optimizer_type
    learning_rate = config.learning_rate
    if kind in {"adam", "nadam"}:
        return optax.inject_hyperparams(optax.adam, static_args=("nesterov",))(
            learning_rate=learning_rate,
            b1=config.b1,
            b2=config.b2,
            eps=config.eps,
            nesterov=kind == "nadam",
        )
    if kind == "radam":
        return optax.inject_hyperparams(optax.radam)(
            learning_rate=learning_rate, b1=config.b1, b2=config.b2, eps=config.eps
        )
    if kind in {"adamw", "lamb"}:
        alias = optax.adamw if kind == "adamw" else optax.lamb
        weight_decay_mask(model, config)  # refuses a filter that selects nothing
        return optax.inject_hyperparams(alias, static_args=("mask",))(
            learning_rate=learning_rate,
            b1=config.b1,
            b2=config.b2,
            eps=config.eps,
            weight_decay=config.weight_decay,
            mask=mask_callable(config),
        )
    if kind == "sgd":
        momentum = {} if config.momentum is None else {"momentum": config.momentum}
        return optax.inject_hyperparams(optax.sgd)(learning_rate=learning_rate, **momentum)
    if kind == "rmsprop":
        momentum = {} if config.momentum is None else {"momentum": config.momentum}
        return optax.inject_hyperparams(optax.rmsprop)(
            learning_rate=learning_rate, eps=config.eps, **momentum
        )
    return optax.inject_hyperparams(optax.adagrad)(learning_rate=learning_rate, eps=config.eps)
