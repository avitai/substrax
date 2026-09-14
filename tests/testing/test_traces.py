"""``TraceCounter`` counts how often a traced function's Python body runs.

jax runs a jitted function's body once per trace, caching the result of ``trace_to_jaxpr``, and
``nnx.jit`` calls the user function once inside the function jax traces. Counting body executions
therefore counts traces under both.
"""

from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from substrax.testing import RetraceError, TraceCounter


def _double(x: jax.Array) -> jax.Array:
    return x * 2.0


def test_a_jitted_function_traces_once_for_calls_of_one_shape() -> None:
    counter = TraceCounter()
    step = jax.jit(counter.wrap(_double))

    with counter.expect(new_traces=1):
        step(jnp.ones(3))
    with counter.expect(new_traces=0):
        step(jnp.ones(3))
        step(jnp.zeros(3))

    assert counter.count == 1


def test_a_new_shape_traces_again() -> None:
    counter = TraceCounter()
    step = jax.jit(counter.wrap(_double))
    step(jnp.ones(3))

    with counter.expect(new_traces=1):
        step(jnp.ones(4))


def test_nnx_jit_over_a_module_traces_once() -> None:
    counter = TraceCounter()
    model = nnx.Linear(2, 1, rngs=nnx.Rngs(0))

    def forward(module: nnx.Linear, x: jax.Array) -> jax.Array:
        return module(x)

    step = nnx.jit(counter.wrap(forward))

    with counter.expect(new_traces=1):
        step(model, jnp.ones((4, 2)))
        step(model, jnp.zeros((4, 2)))


def test_a_vmap_inside_a_jit_traces_the_body_once_per_outer_trace() -> None:
    counter = TraceCounter()
    step = jax.jit(jax.vmap(counter.wrap(_double)))

    with counter.expect(new_traces=1):
        step(jnp.ones((5, 3)))
        step(jnp.ones((5, 3)))


def test_an_exception_raised_while_tracing_still_counts() -> None:
    counter = TraceCounter()

    def failing(x: jax.Array) -> jax.Array:
        raise ValueError(f"cannot trace {x.shape}")

    with pytest.raises(ValueError, match="cannot trace"):
        jax.jit(counter.wrap(failing))(jnp.ones(2))

    assert counter.count == 1


def test_expectation_blocks_nest() -> None:
    counter = TraceCounter()
    step = jax.jit(counter.wrap(_double))

    with counter.expect(new_traces=2):
        with counter.expect(new_traces=1):
            step(jnp.ones(2))
        with counter.expect(new_traces=1):
            step(jnp.ones(3))


def _rebuild_the_jit_on_every_call(counter: TraceCounter, calls: int) -> None:
    with counter.expect(new_traces=1):
        for _ in range(calls):
            jax.jit(counter.wrap(_double))(jnp.ones(3))


def test_a_jit_rebuilt_inside_a_loop_fails_the_expectation_with_the_observed_count() -> None:
    """Control: the check that a step compiles once must catch a step rebuilt on every call."""
    with pytest.raises(RetraceError, match=r"expected 1 new trace.*observed 3") as caught:
        _rebuild_the_jit_on_every_call(TraceCounter(), calls=3)

    assert (caught.value.expected, caught.value.observed) == (1, 3)


def test_a_block_that_raises_reports_its_own_error_not_a_count() -> None:
    counter = TraceCounter()

    with pytest.raises(KeyError, match="inside the block"), counter.expect(new_traces=5):
        raise KeyError("inside the block")


def test_counters_are_independent() -> None:
    first, second = TraceCounter(), TraceCounter()

    jax.jit(first.wrap(_double))(jnp.ones(2))

    assert (first.count, second.count) == (1, 0)


def test_the_wrapped_function_keeps_its_signature_for_static_arguments() -> None:
    counter = TraceCounter()

    def scaled(x: jax.Array, factor: int) -> jax.Array:
        return x * factor

    wrapped = counter.wrap(scaled)
    step = jax.jit(wrapped, static_argnames=("factor",))

    assert inspect.signature(wrapped) == inspect.signature(scaled)
    with counter.expect(new_traces=2):
        step(jnp.ones(2), factor=2)
        step(jnp.ones(2), factor=2)
        step(jnp.ones(2), factor=3)


def test_an_eager_call_counts_as_a_body_execution() -> None:
    counter = TraceCounter()

    with counter.expect(new_traces=2):
        counter.wrap(_double)(jnp.ones(2))
        counter.wrap(_double)(jnp.ones(2))


def test_a_negative_expectation_raises() -> None:
    with pytest.raises(ValueError, match="new_traces"), TraceCounter().expect(new_traces=-1):
        pass
