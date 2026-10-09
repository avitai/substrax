"""``JAX_CALL_WEIGHTS`` counts compiled dispatches, ``jax.device_put`` calls and leaves placed.

Each function is jitted and called once before the counted block, so the block counts dispatches
of cached executables and no trace or compile.
"""

from __future__ import annotations

import sys
import threading

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from substrax.testing import by_package, counted_calls
from substrax.testing.jax_calls import (
    device_puts,
    JAX_CALL_WEIGHTS,
    jit_dispatches,
    leaves_placed,
)


def _scale(x: jax.Array) -> jax.Array:
    return x * 2.0


def _counted(work: object) -> dict[str, int]:
    assert callable(work)
    with counted_calls(by_package("jax"), weights=JAX_CALL_WEIGHTS) as counts:
        work()
    return {key: counts[key] for key in JAX_CALL_WEIGHTS}


def test_the_weights_are_the_three_jax_counters() -> None:
    assert dict(JAX_CALL_WEIGHTS) == {
        "jit_dispatch": jit_dispatches,
        "device_put": device_puts,
        "device_put_leaves": leaves_placed,
    }


def test_each_call_of_a_jitted_function_is_one_dispatch() -> None:
    step, x = jax.jit(_scale), jnp.ones(3)
    step(x)

    def work() -> None:
        for _ in range(3):
            step(x)

    assert _counted(work)["jit_dispatch"] == 3


def test_an_nnx_jitted_call_is_one_dispatch() -> None:
    model = nnx.Linear(2, 2, rngs=nnx.Rngs(0))
    step, x = nnx.jit(lambda m, v: m(v)), jnp.ones((1, 2))
    step(model, x)

    assert _counted(lambda: step(model, x))["jit_dispatch"] == 1


def _scale_on_the_host(x: np.ndarray) -> np.ndarray:
    return x * 2.0


def test_a_plain_python_function_is_no_dispatch() -> None:
    x = np.ones(3)

    assert _counted(lambda: _scale_on_the_host(x))["jit_dispatch"] == 0


def test_each_placement_counts_once_with_the_leaves_it_placed() -> None:
    batch = {"image": np.ones((2, 3)), "label": np.zeros(2), "mask": np.ones(2, bool)}
    jax.block_until_ready(jax.device_put(batch))

    counts = _counted(lambda: jax.device_put(batch))

    assert counts["device_put"] == 1
    assert counts["device_put_leaves"] == 3


def test_a_second_placement_is_counted_through_an_alias() -> None:
    put, batch = jax.device_put, {"x": np.ones(4), "y": np.ones(4)}
    put(batch)

    def twice() -> None:
        put(batch)
        put(batch)

    counts = _counted(twice)

    assert counts["device_put"] == 2
    assert counts["device_put_leaves"] == 4


def test_a_placement_by_keyword_counts_its_leaves() -> None:
    counts = _counted(lambda: jax.device_put(x={"a": np.ones(2), "b": np.ones(2)}))

    assert counts["device_put"] == 1
    assert counts["device_put_leaves"] == 2


def test_a_placement_with_no_argument_counts_the_call_but_no_leaves() -> None:
    def no_argument() -> None:
        with pytest.raises(TypeError):
            jax.device_put()  # pyright: ignore[reportCallIssue]

    counts = _counted(no_argument)

    assert counts["device_put"] == 1
    assert counts["device_put_leaves"] == 0


def test_placements_and_dispatches_on_a_thread_are_identical_across_repeats() -> None:
    step, batch = jax.jit(_scale), np.ones(8, np.float32)
    step(jax.device_put(batch))

    def pass_on_a_thread() -> None:
        def run() -> None:
            for _ in range(20):
                step(jax.device_put(batch)).block_until_ready()

        worker = threading.Thread(target=run)
        worker.start()
        worker.join()

    runs = [_counted(pass_on_a_thread) for _ in range(3)]

    assert runs[0] == {"jit_dispatch": 20, "device_put": 20, "device_put_leaves": 20}
    assert all(counts == runs[0] for counts in runs)


def test_the_weights_read_the_callee_and_the_value_placed() -> None:
    missing = sys.monitoring.MISSING
    tree = {"a": np.ones(2), "b": (np.ones(1), np.ones(1))}

    assert jit_dispatches(jax.jit(_scale), missing) == 1
    assert jit_dispatches(_scale, missing) == 0
    assert device_puts(jax.device_put, tree) == 1
    assert device_puts(jax.device_get, tree) == 0
    assert leaves_placed(jax.device_put, tree) == 3
    assert leaves_placed(jax.device_put, missing) == 0
    assert leaves_placed(jax.device_get, tree) == 0
