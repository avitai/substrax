"""An optimizer state whose types do not change on its first update.

``jax.jit`` keys its cache on each argument's type, weak typing included, so a state leaf created
weakly typed (``jnp.asarray(0)``) and returned strongly typed by the first update compiles the
step a second time. ``optax.lbfgs`` initialises three line-search counters that way
(``optax/_src/linesearch.py``; upstream PR #1108 was closed unmerged), and a loop that jits one
step per call pays the extra compile. ``jax.lax.while_loop`` and ``scan`` promote weakly typed
carries themselves, which is why optax's own loop examples never meet it.
``with_strong_state_types`` gives each weakly typed leaf its own dtype explicitly, the fix optax
applied to its backtracking line search (PR #1175) and ``reduce_on_plateau`` (#1440).
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import optax


def with_strong_state_types(
    transformation: optax.GradientTransformation,
) -> optax.GradientTransformationExtraArgs:
    """``transformation`` with every weakly typed leaf of its initial state strongly typed.

    The values and dtypes are unchanged; only the weak type goes, so the state a jitted step
    receives has the types its update returns. Updates are ``transformation``'s own, extra
    arguments included.

    Args:
        transformation: The optax transformation, such as ``optax.lbfgs()``.

    Returns:
        The transformation with a strongly typed initial state.
    """
    inner = optax.with_extra_args_support(transformation)

    def init(params: optax.Params) -> optax.OptState:
        return jax.tree.map(_strongly_typed, inner.init(params))

    return optax.GradientTransformationExtraArgs(init, inner.update)


def _strongly_typed(leaf: Any) -> Any:
    """``leaf`` as a strongly typed array of its own dtype, if it is a weakly typed one."""
    if isinstance(leaf, jax.Array) and jax.typeof(leaf).weak_type:
        return jnp.asarray(leaf, dtype=leaf.dtype)
    return leaf
