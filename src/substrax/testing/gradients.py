"""Gradient checks that assert values: a module's autodiff gradient against finite differences.

A test that only checks a gradient exists, is finite or is non-zero passes on a wrong gradient.
These checks compare the gradient of a loss, in a module's ``nnx.Param`` leaves or in its input,
with finite differences through ``jax.test_util.check_grads`` (forward and reverse mode along one
random direction; a function with a ``jax.custom_vjp`` has no forward mode, so it is checked with
``modes=("rev",)``). They run on a float64 copy of the module: a derivative summed over many output
elements is below a float32 finite difference's resolution (``check_grads``'s default tolerance is
1e-5 in float64 against 2e-3 in float32). The copy is split in tree mode, which a module holding an
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
        params64, rest64 = _float64(params), _float64(rest)

        def of_params(values: NnxState) -> jax.Array:
            return loss(nnx.merge(graphdef, values, rest64))

        check_grads(jax.jit(of_params), (params64,), order=1, modes=tuple(modes))
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
        copy = nnx.merge(graphdef, _float64(state))
        at = jnp.asarray(value, dtype=jnp.float64)

        def of_input(x: jax.Array) -> jax.Array:
            return loss(copy, x)

        check_grads(jax.jit(of_input), (at,), order=1, modes=tuple(modes))
        gradient = jax.jit(jax.grad(of_input))(at)
    _refuse_all_zero(gradient, allow_zero=allow_zero)
    return gradient


def _float64[T](tree: T) -> T:
    """The tree with its floating leaves cast to float64 and every other leaf as it is."""
    return jax.tree.map(
        lambda leaf: leaf.astype(jnp.float64) if jnp.issubdtype(leaf.dtype, jnp.floating) else leaf,
        tree,
    )


def _refuse_all_zero(gradient: NnxState | jax.Array, *, allow_zero: bool) -> None:
    if allow_zero:
        return
    if not any(np.any(np.asarray(leaf) != 0) for leaf in jax.tree.leaves(gradient)):
        msg = (
            "the gradient is zero everywhere, so a finite-difference check proves nothing; pass "
            "allow_zero=True if the loss is meant not to depend on what is differentiated"
        )
        raise AssertionError(msg)
