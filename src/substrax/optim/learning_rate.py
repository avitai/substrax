"""The learning rate an optimizer last applied, read from its state on device."""

from __future__ import annotations

from typing import Any

import jax
from flax import nnx


def current_learning_rate(optimizer: nnx.Optimizer[Any]) -> jax.Array:
    """Return the learning rate the last ``update`` applied, as an array on device.

    ``create_transformation`` wraps the base alias in ``optax.inject_hyperparams``, which
    stores the rate it evaluated at optax's own step count. Before the first update the value
    is the schedule at step zero; a step a wrapper such as ``optax.apply_if_finite`` skips
    leaves it unchanged, because optax's count does not advance.

    Args:
        optimizer: An optimizer built over ``create_transformation``.

    Returns:
        The learning rate, a scalar array.

    Raises:
        ValueError: If the optimizer's state holds no injected learning rate.
    """
    state = _injected_state(optimizer.opt_state)
    if state is None:
        raise ValueError(
            "the optimizer state holds no injected learning rate; build the transformation with "
            "substrax.optim.create_transformation"
        )
    return state.hyperparams["learning_rate"][...]


def _injected_state(state: Any) -> Any | None:
    """Find the ``inject_hyperparams`` state holding ``learning_rate``, through any wrapper."""
    hyperparams = getattr(state, "hyperparams", None)
    if hyperparams is not None and "learning_rate" in hyperparams:
        return state
    children: list[Any] = []
    if isinstance(state, tuple):
        children.extend(state)
    elif isinstance(state, dict):
        children.extend(state.values())
    inner = getattr(state, "inner_state", None)
    if inner is not None:
        children.append(inner)
    for child in children:
        found = _injected_state(child)
        if found is not None:
            return found
    return None
