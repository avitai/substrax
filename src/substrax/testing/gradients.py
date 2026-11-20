"""Gradient checks that assert values: a module's autodiff gradient against finite differences.

A test that only checks a gradient exists, is finite or is non-zero passes on a wrong gradient.
These checks compare the gradient of a loss, in a module's ``nnx.Param`` leaves or in its input,
with finite differences through ``jax.test_util.check_grads`` (forward and reverse mode; a function
with a ``jax.custom_vjp`` has no forward mode, so it is checked with ``modes=("rev",)``).

The comparison is along one random direction of unit norm, as JAX's automatic-differentiation docs
(``docs/automatic-differentiation.md``, "finite differences in a random direction") and PyTorch's
fast ``gradcheck`` (``torch/autograd/gradcheck.py``, ``_vec_from_tensor``) take it:
``check_grads`` differentiates ``t -> loss(m + t * u)`` at ``t = 0``, so its finite-difference step
is a length in parameter space whatever the number of parameters. Applied to the parameters
directly, ``check_grads`` draws an unnormalised normal direction of norm about ``sqrt(n)``; on a
module with thousands of parameters its step then grows a hundredfold and its truncation error
fails a correct gradient.

The step is 1e-6, PyTorch's ``gradcheck`` default (``torch/autograd/gradcheck.py``, ``eps``), not
``check_grads``' own 1e-4. A central difference's truncation error grows as the step squared,
and at 1e-4 it exceeded the 1e-5 tolerance on correct gradients: a module whose third derivative
along the direction is large, and one whose parameters sit within a step of a ``leaky_relu`` kink,
where the difference straddles the kink. In float64 the rounding error a smaller step brings, about
``2.2e-16 * |loss| / step``, stays near 1e-10 relative.

They run on a float64 copy of the module: a derivative summed over many output elements is below a
float32 finite difference's resolution (``check_grads``'s default tolerance is 1e-5 in float64
against 2e-3 in float32). The copy is split in tree mode, which a module holding an
``nnx.RngKey`` needs, and the caller's module keeps its own dtype and values.

The module imports jax and flax, so ``substrax.testing`` does not import it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.test_util import check_grads
from jax.typing import ArrayLike

from substrax.nnx_typing import NnxState


_SEED = 0
"""Seed of the random direction, so a failure is reproducible."""

_STEP = 1e-6
"""Finite-difference step along the unit direction, PyTorch ``gradcheck``'s float64 default."""


def check_parameter_gradients[M: nnx.Module](  # noqa: DOC502  # raised by check_grads and _refuse_all_zero
    module: M,
    loss: Callable[[M], jax.Array],
    *,
    modes: Sequence[str] = ("fwd", "rev"),
    allow_zero: bool = False,
) -> NnxState:
    """Check the gradient of ``loss`` in ``module``'s ``nnx.Param`` leaves, and return it.

    Args:
        module: The module; it is not changed.
        loss: A scalar function of the module.
        modes: The autodiff modes to check, ``"fwd"`` and/or ``"rev"``.
        allow_zero: Accept a gradient that is zero everywhere; refused otherwise, since a loss
            that does not depend on the parameters passes a finite-difference check.

    Returns:
        The gradient, in float64, with the structure of the module's ``nnx.Param`` state.

    Raises:
        AssertionError: If the gradient differs from finite differences, or is zero everywhere
            and ``allow_zero`` is ``False``.
    """
    graphdef, params, rest = nnx.split(module, nnx.Param, ..., graph=False)
    with jax.enable_x64(True):
        params64, rest64 = as_float64(params), as_float64(rest)
        direction = _unit_direction(params64)

        def of_params(values: NnxState) -> jax.Array:
            return loss(nnx.merge(graphdef, values, rest64))

        def along(step: jax.Array) -> jax.Array:
            return of_params(jax.tree.map(lambda p, u: p + step * u, params64, direction))

        check_grads(
            jax.jit(along),
            (jnp.zeros((), jnp.float64),),
            order=1,
            modes=tuple(modes),
            eps=_STEP,
        )
        gradient: NnxState = jax.jit(jax.grad(of_params))(params64)
    _refuse_all_zero(gradient, allow_zero=allow_zero)
    return gradient


def check_input_gradients[M: nnx.Module](  # noqa: DOC502  # raised by check_grads and _refuse_all_zero
    module: M,
    loss: Callable[[M, jax.Array], jax.Array],
    value: ArrayLike,
    *,
    modes: Sequence[str] = ("fwd", "rev"),
    allow_zero: bool = False,
) -> jax.Array:
    """Check the gradient of ``loss`` in its input ``value``, and return it.

    Args:
        module: The module; it is not changed.
        loss: A scalar function of the module and the input.
        value: The input the gradient is taken at, cast to float64.
        modes: The autodiff modes to check, ``"fwd"`` and/or ``"rev"``.
        allow_zero: Accept a gradient that is zero everywhere; refused otherwise.

    Returns:
        The gradient in float64, shaped like ``value``.

    Raises:
        AssertionError: If the gradient differs from finite differences, or is zero everywhere
            and ``allow_zero`` is ``False``.
    """
    graphdef, state = nnx.split(module, graph=False)
    with jax.enable_x64(True):
        copy = nnx.merge(graphdef, as_float64(state))
        at = jnp.asarray(value, dtype=jnp.float64)
        direction = _unit_direction(at)

        def of_input(x: jax.Array) -> jax.Array:
            return loss(copy, x)

        def along(step: jax.Array) -> jax.Array:
            return of_input(at + step * direction)

        check_grads(
            jax.jit(along),
            (jnp.zeros((), jnp.float64),),
            order=1,
            modes=tuple(modes),
            eps=_STEP,
        )
        gradient = jax.jit(jax.grad(of_input))(at)
    _refuse_all_zero(gradient, allow_zero=allow_zero)
    return gradient


def as_float64[T](tree: T) -> T:
    """The tree with its floating arrays cast to float64 and every other leaf as it is.

    An integer array, such as a class label that indexes a table, stays an integer, as PyTorch's
    ``gradcheck`` perturbs only floating inputs; a leaf that is not an array, such as a Python
    value a module keeps in a Variable, has no dtype and is left as it is.

    Args:
        tree: Any pytree.

    Returns:
        The tree, with each floating array leaf in float64.
    """
    return jax.tree.map(
        lambda leaf: (
            leaf.astype(jnp.float64)
            if isinstance(leaf, jax.Array | np.ndarray) and jnp.issubdtype(leaf.dtype, jnp.floating)
            else leaf
        ),
        tree,
    )


def _unit_direction[T](tree: T) -> T:
    """A random direction over the tree's floating arrays, with unit Euclidean norm overall."""
    leaves, structure = jax.tree.flatten(tree)
    keys = jax.random.split(jax.random.key(_SEED), max(len(leaves), 1))
    drawn = [
        jax.random.normal(key, leaf.shape, leaf.dtype)
        if jnp.issubdtype(leaf.dtype, jnp.inexact)
        else jnp.zeros_like(leaf)
        for key, leaf in zip(keys, leaves, strict=False)
    ]
    norm = jnp.sqrt(sum(jnp.sum(jnp.abs(each) ** 2) for each in drawn))
    return jax.tree.unflatten(structure, [each / norm for each in drawn])


def _refuse_all_zero(gradient: NnxState | jax.Array, *, allow_zero: bool) -> None:
    if allow_zero:
        return
    if not any(np.any(np.asarray(leaf) != 0) for leaf in jax.tree.leaves(gradient)):
        msg = (
            "the gradient is zero everywhere, so a finite-difference check proves nothing; pass "
            "allow_zero=True if the loss is meant not to depend on what is differentiated"
        )
        raise AssertionError(msg)
