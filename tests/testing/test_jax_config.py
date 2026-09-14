"""``restored_jax_config`` sets back the global jax configuration a block changed."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from substrax.testing import restored_jax_config


def test_a_changed_value_is_set_back_and_named() -> None:
    x64 = jax.config.jax_enable_x64
    precision = jax.config.jax_default_matmul_precision

    with restored_jax_config() as changed:
        jax.config.update("jax_enable_x64", not x64)
        jax.config.update(
            "jax_default_matmul_precision", "highest" if precision != "highest" else "high"
        )
        assert changed == []

    assert changed == ["jax_default_matmul_precision", "jax_enable_x64"]
    assert (jax.config.jax_enable_x64, jax.config.jax_default_matmul_precision) == (x64, precision)


def test_a_thread_local_context_left_inside_the_block_changes_nothing() -> None:
    with restored_jax_config() as changed, jax.enable_x64(True):
        assert jnp.ones(1).dtype == jnp.float64

    assert changed == []


def test_a_value_set_to_what_it_already_was_is_not_reported() -> None:
    with restored_jax_config() as changed:
        jax.config.update("jax_enable_x64", jax.config.jax_enable_x64)

    assert changed == []


def _flip_x64_and_raise() -> None:
    jax.config.update("jax_enable_x64", not jax.config.jax_enable_x64)
    raise RuntimeError("the block failed")


def test_values_are_set_back_when_the_block_raises() -> None:
    x64 = jax.config.jax_enable_x64

    with pytest.raises(RuntimeError, match="the block failed"), restored_jax_config():
        _flip_x64_and_raise()

    assert jax.config.jax_enable_x64 is x64
