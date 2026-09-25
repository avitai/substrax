"""Line-search optimizers on an NNX model, and one optax optimizer handing over to another.

``optax.lbfgs`` searches along its direction with the loss as a function of the parameters, so
its ``update`` takes ``value``, ``grad`` and ``value_fn``, which ``nnx.Optimizer.update`` passes
through. ``update_with_line_search`` builds them from the model: the loss of the parameters
``optimizer.wrt`` selects, with the rest of the model's state (batch statistics, RNG counts)
held as it is. ``switch_at`` is the first optimizer for a fixed number of updates and the
second after them, the Adam-then-L-BFGS schedule of Rathore et al. (ICML 2024, "Challenges in
Training PINNs: A Loss Landscape Perspective"). Both are plain optax and ``nnx.Optimizer``, so
a step using them traces once under ``nnx.jit``, the switch included.

A line search evaluates the loss and gradient at the point it accepts, and optax keeps them in
the optimizer state. Using them for the next step saves one evaluation per step, and is right
only while the objective is the one they were computed for: a new minibatch or resampled
collocation points make them the previous objective's. Quasi-Newton methods that allow the
objective to change evaluate on a consistent sample instead (Schraudolph, Yu and Guenter 2007,
section 3.2.3; Berahas, Nocedal and Takac 2016, eq. 2.3), and optax documents the reuse for a
non-stochastic setting only. ``update_with_line_search`` therefore asks on every call whether
the objective changed.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from substrax.optim.state_types import with_strong_state_types


def update_with_line_search[M: nnx.Module](
    model: M,
    optimizer: nnx.Optimizer[Any],
    loss_fn: Callable[[M], jax.Array],
    *,
    objective_changed: bool | jax.Array,
) -> jax.Array:
    """Update ``model`` with an optimizer that evaluates the loss along its search direction.

    The loss, its gradient and the loss as a function of the ``optimizer.wrt`` parameters are
    passed to ``optimizer.update`` as ``value``, ``grad`` and ``value_fn``, the extra arguments
    ``optax.lbfgs`` and any transformation built on it read. When ``objective_changed`` is false
    and the state holds the value and gradient the previous line search computed at the current
    parameters, those are used; otherwise they are evaluated. ``objective_changed`` is traced,
    so either answer runs the same compiled step. Under ``vmap`` (an ensemble of models) the
    choice depends on each member's stored value, and a ``lax.cond`` over a batched predicate
    runs both branches, so each step evaluates the loss as it would with ``objective_changed``
    true: the result is the same and the saving is lost. The loss is taken in the parameters' dtype:
    optax's line search keeps its values in that dtype, and a wider loss (a float32 model's
    loss computed in float64 under x64) would make the branches of its ``lax.cond`` disagree.

    Args:
        model: The model; updated in place.
        optimizer: An ``nnx.Optimizer`` over ``model`` whose transformation takes the extra
            arguments and whose state is strongly typed, such as
            ``with_strong_state_types(optax.lbfgs())`` or a ``switch_at`` ending in it.
        loss_fn: The scalar loss of the model.
        objective_changed: Whether ``loss_fn`` is a different objective from the previous
            call's (a new batch, resampled points), or the parameters moved outside the line
            search since it; ``False`` only for the same objective at the point it accepted.

    Returns:
        The loss before the update, in the parameters' dtype.

    Raises:
        ValueError: If the parameters are of more than one floating dtype, which leaves the
            line search no single dtype to work in, or if the optimizer state has weakly typed
            leaves, which would compile the step again once its first update types them.
    """
    weak = _weakly_typed_paths(optimizer.opt_state)
    if weak:
        msg = (
            f"the optimizer state has weakly typed leaves ({', '.join(weak)}), so a jitted step "
            "would compile again once its first update types them; build the transformation "
            "with substrax.optim.with_strong_state_types"
        )
        raise ValueError(msg)
    graphdef, params, rest = nnx.split(model, optimizer.wrt, ...)
    dtypes = _floating_dtypes(params)
    if len(dtypes) != 1:
        names = ", ".join(sorted(dtype.name for dtype in dtypes)) or "none"
        msg = f"a line search needs one floating parameter dtype, got {names}"
        raise ValueError(msg)
    (dtype,) = dtypes

    def value_fn(candidate: nnx.State[Any, Any]) -> jax.Array:
        return loss_fn(nnx.merge(graphdef, candidate, rest)).astype(dtype)

    pure_params = nnx.as_pure(params)
    opt_state = nnx.as_pure(optimizer.opt_state)
    stored_value = optax.tree.get(opt_state, "value")
    stored_grad = optax.tree.get(opt_state, "grad")
    if stored_value is None or stored_grad is None:
        value, grad_arrays = jax.value_and_grad(value_fn)(pure_params)
    else:
        reuse = jnp.logical_not(objective_changed) & jnp.isfinite(stored_value)
        value, grad_arrays = jax.lax.cond(
            reuse,
            lambda: (stored_value, stored_grad),
            lambda: jax.value_and_grad(value_fn)(pure_params),
        )
    # nnx.Optimizer selects the gradients by their variable type, so they go back into a copy
    # of the parameter state.
    grads = jax.tree.map(jnp.zeros_like, params)
    nnx.replace_by_pure_dict(grads, nnx.to_pure_dict(grad_arrays))
    optimizer.update(model, grads, value=value, grad=grad_arrays, value_fn=value_fn)
    return value


def switch_at(
    first: optax.GradientTransformation,
    second: optax.GradientTransformation,
    *,
    step: int,
) -> optax.GradientTransformationExtraArgs:
    """The ``first`` transformation for ``step`` updates, then ``second``.

    Each runs under ``optax.conditionally_transform`` on the update count, so the switch is a
    branch of the traced step rather than a Python decision, and each keeps its own state from
    initialisation. Extra arguments reach a transformation that takes them (``optax.lbfgs``'s
    ``value``, ``grad`` and ``value_fn``).

    Args:
        first: The transformation for the first ``step`` updates, such as ``optax.adam``.
        second: The transformation from update ``step`` on, such as ``optax.lbfgs()``.
        step: The number of updates ``first`` makes.

    Returns:
        The combined transformation, its initial state strongly typed.

    Raises:
        ValueError: If ``step`` is not positive.
    """
    if step <= 0:
        msg = f"step must be positive, got {step}"
        raise ValueError(msg)
    return with_strong_state_types(
        optax.chain(
            _conditionally(first, lambda count: count < step),
            _conditionally(second, lambda count: count >= step),
        )
    )


def _conditionally(
    inner: optax.GradientTransformation, condition: Callable[[jax.Array], jax.Array]
) -> optax.GradientTransformationExtraArgs:
    """``inner`` on the updates ``condition`` admits, with extra arguments if it takes them."""

    def should_transform(step: jax.typing.ArrayLike, **_extra_args: Any) -> jax.Array:
        return condition(jnp.asarray(step))

    return optax.conditionally_transform(
        inner,
        should_transform,
        forward_extra_args=isinstance(inner, optax.GradientTransformationExtraArgs),
    )


def _floating_dtypes(params: nnx.State[Any, Any]) -> set[jnp.dtype]:
    """The floating dtypes among the parameters.

    Args:
        params: The parameter state.

    Returns:
        Every floating dtype a parameter has.
    """
    return {
        jnp.dtype(leaf.dtype)
        for leaf in jax.tree.leaves(params)
        if jnp.issubdtype(leaf.dtype, jnp.floating)
    }


def _weakly_typed_paths(opt_state: Any) -> list[str]:
    """The paths of the weakly typed leaves of an optimizer state.

    Args:
        opt_state: The optimizer's state.

    Returns:
        Each weakly typed leaf's path, in tree order.
    """
    return [
        jax.tree_util.keystr(path)
        for path, leaf in jax.tree_util.tree_flatten_with_path(nnx.as_pure(opt_state))[0]
        if isinstance(leaf, jax.Array) and jax.typeof(leaf).weak_type
    ]
