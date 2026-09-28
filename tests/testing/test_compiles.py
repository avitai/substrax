"""``compiled_programs`` and ``expect_compiles`` see the executables jax builds inside a block.

jax records ``/jax/core/compile/backend_compile_duration`` around ``compile_or_get_cached``, which
runs only when its in-process executable cache misses, and names the program ``jit(<function>)``.
Eager operations compile too (``jnp.ones`` builds ``jit(broadcast_in_dim)``), so each test makes
its inputs before the block.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from substrax.testing.compiles import CompileCountError, compiled_programs, expect_compiles


def _triple(x: jax.Array) -> jax.Array:
    return x * 3.0


def test_a_first_call_compiles_once_and_a_second_call_not_at_all() -> None:
    step, x = jax.jit(_triple), jnp.ones(5)

    with compiled_programs() as first:
        step(x)
    with compiled_programs() as second:
        step(x)

    assert first == ["jit(_triple)"]
    assert second == []


def test_an_unexpected_compile_is_refused_by_name() -> None:
    step, x = jax.jit(_triple), jnp.ones(6)

    with pytest.raises(CompileCountError, match=r"jit\(_triple\)") as refused, expect_compiles(0):
        step(x)

    assert refused.value.expected == 0
    assert refused.value.names == ("jit(_triple)",)


def test_the_expected_number_of_compiles_passes() -> None:
    step, x = jax.jit(_triple), jnp.ones(7)

    with expect_compiles(1):
        step(x)
    with expect_compiles(0):
        step(x)


def test_a_negative_count_is_refused() -> None:
    with pytest.raises(ValueError, match="at least 0"), expect_compiles(-1):
        pass


def test_nothing_is_recorded_after_the_block() -> None:
    step, x = jax.jit(_triple), jnp.ones(8)
    with compiled_programs() as names:
        pass
    step(x)

    assert names == []


def test_nothing_is_recorded_after_a_block_that_raised() -> None:
    step, x = jax.jit(_triple), jnp.ones(9)
    with pytest.raises(RuntimeError), compiled_programs() as names:
        raise RuntimeError("inside the block")
    step(x)

    assert names == []


def test_nested_blocks_each_see_the_compiles_inside_them() -> None:
    outer_step, inner_step, x = jax.jit(_triple), jax.jit(lambda v: v + 1.0), jnp.ones(10)

    with compiled_programs() as outer:
        outer_step(x)
        with compiled_programs() as inner:
            inner_step(x)

    assert outer == ["jit(_triple)", "jit(<lambda>)"]
    assert inner == ["jit(<lambda>)"]
