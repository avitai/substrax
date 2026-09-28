"""The learning rate an optimizer last applied, read from its state on device."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
from flax import nnx


def current_learning_rate[M](optimizer: nnx.Optimizer[M]) -> jax.Array:
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
    rate = state.hyperparams["learning_rate"]
    # nnx.Optimizer holds each state leaf in a Variable; optax types the leaf ArrayLike.
    return jnp.asarray(rate[...] if isinstance(rate, nnx.Variable) else rate)


def _injected_state(state: object) -> optax.InjectStatefulHyperparamsState | None:
    """Find the ``inject_hyperparams`` state holding ``learning_rate``, through any wrapper."""
    if (
        isinstance(state, optax.InjectStatefulHyperparamsState)
        and "learning_rate" in state.hyperparams
    ):
        return state
    for child in _children(state):
        found = _injected_state(child)
        if found is not None:
            return found
    return None


def _children(state: object) -> list[object]:
    """The states a wrapper holds: a tuple's or a dict's entries, and an ``inner_state``."""
    children: list[object] = []
    if isinstance(state, tuple):
        children.extend(state)
    elif isinstance(state, dict):
        children.extend(state.values())
    inner: object = getattr(state, "inner_state", None)
    if inner is not None:
        children.append(inner)
    return children
