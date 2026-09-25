"""An optimizer state that keeps its types across updates, so a jitted step compiles once."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from substrax.optim import with_strong_state_types
from substrax.testing import TraceCounter


def _weak_paths(tree: object) -> list[str]:
    return [
        jax.tree_util.keystr(path)
        for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]
        if jax.typeof(leaf).weak_type
    ]


PARAMS = jnp.asarray([1.0, -2.0, 0.5])


def _value(params: jax.Array) -> jax.Array:
    return jnp.sum(params**2)


def _array(updates: optax.Updates) -> jax.Array:
    """The updates of an array parameter, which are an array."""
    assert isinstance(updates, jax.Array)
    return updates


def test_optax_lbfgs_still_initialises_its_line_search_counters_weakly() -> None:
    # The upstream defect with_strong_state_types exists for (optax PR #1108, closed unmerged;
    # optax/_src/linesearch.py initialises three counters with jnp.asarray(0) / (jnp.inf)).
    # When this fails, optax has fixed it: delete the adapter and this test.
    assert _weak_paths(optax.lbfgs().init(PARAMS)) == [
        "[2].info.num_linesearch_steps",
        "[2].info.decrease_error",
        "[2].info.curvature_error",
    ]


def test_the_adapted_state_has_no_weakly_typed_leaf() -> None:
    state = with_strong_state_types(optax.lbfgs()).init(PARAMS)

    assert _weak_paths(state) == []


def test_the_adapted_state_holds_the_same_values_and_dtypes() -> None:
    plain = optax.lbfgs().init(PARAMS)
    adapted = with_strong_state_types(optax.lbfgs()).init(PARAMS)

    for original, strong in zip(jax.tree.leaves(plain), jax.tree.leaves(adapted), strict=True):
        assert strong.dtype == original.dtype
        np.testing.assert_array_equal(strong, original)


@pytest.mark.parametrize(
    ("param_dtype", "x64"),
    [(jnp.float32, False), (jnp.float32, True), (jnp.float64, True)],
)
def test_a_jitted_update_compiles_once(param_dtype: jnp.dtype, x64: bool) -> None:
    # x64 changes the default dtypes optax's weakly typed counters take; the state must keep
    # its types in every precision configuration, not only the default one.
    with jax.enable_x64(x64):
        solver = with_strong_state_types(optax.lbfgs())
        counter = TraceCounter()

        def step(params: jax.Array, state: optax.OptState) -> tuple[jax.Array, optax.OptState]:
            grad = jax.grad(_value)(params)
            updates, state = solver.update(
                grad, state, params, value=_value(params), grad=grad, value_fn=_value
            )
            return params + _array(updates), state

        update = jax.jit(counter.wrap(step))
        params = jnp.asarray([1.0, -2.0, 0.5], dtype=param_dtype)
        state = solver.init(params)

        with counter.expect(new_traces=1):
            for _ in range(3):
                params, state = update(params, state)


def test_the_updates_are_the_wrapped_transformations() -> None:
    plain, adapted = optax.lbfgs(), with_strong_state_types(optax.lbfgs())
    value, grad = _value(PARAMS), jax.grad(_value)(PARAMS)

    expected, _ = plain.update(
        grad, plain.init(PARAMS), PARAMS, value=value, grad=grad, value_fn=_value
    )
    got, _ = adapted.update(
        grad, adapted.init(PARAMS), PARAMS, value=value, grad=grad, value_fn=_value
    )

    np.testing.assert_allclose(_array(got), _array(expected))


def test_a_transformation_without_extra_arguments_keeps_working() -> None:
    adapted = with_strong_state_types(optax.sgd(0.1))
    state = adapted.init(PARAMS)

    updates, _ = adapted.update(jnp.ones(3), state, PARAMS)

    np.testing.assert_allclose(_array(updates), -0.1 * jnp.ones(3))
