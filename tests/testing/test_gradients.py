"""``check_parameter_gradients`` and ``check_input_gradients`` assert gradient values.

Each checks a module's autodiff gradient against finite differences in float64, forward and
reverse, on a float64 copy split in tree mode, and returns the gradient.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from substrax.nnx_typing import NnxState
from substrax.testing.gradients import check_input_gradients, check_parameter_gradients


_X = np.array([0.5, -1.0, 2.0])
_C = np.array([1.5, -0.25])


class _Affine(nnx.Module):
    """``y = x @ w + b``, with fixed parameters so the gradients have closed forms."""

    def __init__(self) -> None:
        self.w = nnx.Param(jnp.arange(6.0, dtype=jnp.float32).reshape(3, 2) / 10)
        self.b = nnx.Param(jnp.array([0.1, -0.2], dtype=jnp.float32))

    def __call__(self, x: jax.Array) -> jax.Array:
        return x @ self.w[...] + self.b[...]


def _loss_in_parameters(module: _Affine) -> jax.Array:
    return jnp.sum(module(jnp.asarray(_X, dtype=module.w[...].dtype)) * _C)


def _loss_in_input(module: _Affine, x: jax.Array) -> jax.Array:
    return jnp.sum(module(x) * _C)


def _value(state: NnxState, name: str) -> np.ndarray:
    """The array held by the variable ``name`` in ``state``."""
    variable = state[name]
    assert isinstance(variable, nnx.Variable)
    return np.asarray(variable[...])


def test_parameter_gradients_equal_their_closed_form() -> None:
    gradient = check_parameter_gradients(_Affine(), _loss_in_parameters)

    np.testing.assert_allclose(_value(gradient, "w"), np.outer(_X, _C), rtol=1e-12)
    np.testing.assert_allclose(_value(gradient, "b"), _C, rtol=1e-12)


def test_the_input_gradient_equals_its_closed_form() -> None:
    module = _Affine()

    gradient = check_input_gradients(module, _loss_in_input, _X)

    np.testing.assert_allclose(gradient, np.asarray(module.w[...], np.float64) @ _C, rtol=1e-6)


@jax.custom_jvp
def _doubled_tangent(x: jax.Array) -> jax.Array:
    return x


def _doubled_tangent_jvp(
    primals: tuple[jax.Array], tangents: tuple[jax.Array]
) -> tuple[jax.Array, jax.Array]:
    return primals[0], 2.0 * tangents[0]


_doubled_tangent.defjvp(_doubled_tangent_jvp)


@jax.custom_vjp
def _doubled_cotangent(x: jax.Array) -> jax.Array:
    return x


def _identity_forward(x: jax.Array) -> tuple[jax.Array, None]:
    return x, None


def _doubled_backward(_: None, cotangent: jax.Array) -> tuple[jax.Array]:
    return (2.0 * cotangent,)


_doubled_cotangent.defvjp(_identity_forward, _doubled_backward)


def test_a_wrong_gradient_fails_the_check() -> None:
    def loss(module: _Affine) -> jax.Array:
        return jnp.sum(_doubled_tangent(module.w[...]))

    with pytest.raises(AssertionError):
        check_parameter_gradients(_Affine(), loss)


def test_a_wrong_custom_vjp_fails_the_reverse_mode_check() -> None:
    """A ``custom_vjp`` function has no forward mode, so it is checked in reverse mode only."""

    def loss(module: _Affine) -> jax.Array:
        return jnp.sum(_doubled_cotangent(module.w[...]))

    with pytest.raises(AssertionError):
        check_parameter_gradients(_Affine(), loss, modes=("rev",))


def test_an_all_zero_gradient_is_refused_unless_allowed() -> None:
    def loss(module: _Affine) -> jax.Array:
        return jnp.sum(module.w[...] * 0.0)

    with pytest.raises(AssertionError, match="zero"):
        check_parameter_gradients(_Affine(), loss)

    gradient = check_parameter_gradients(_Affine(), loss, allow_zero=True)
    assert not np.any(_value(gradient, "w"))


def test_the_module_keeps_its_own_dtype_and_values() -> None:
    module = _Affine()
    before = np.asarray(module.w[...])

    check_parameter_gradients(module, _loss_in_parameters)

    assert module.w[...].dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(module.w[...]), before)


class _Keyed(_Affine):
    """An affine module that also holds a random key, as datarax's stochastic operators do."""

    def __init__(self) -> None:
        super().__init__()
        self.key = nnx.RngKey(jax.random.key(0))


def test_a_module_holding_a_key_is_checked_in_its_parameters() -> None:
    gradient = check_parameter_gradients(_Keyed(), _loss_in_parameters)

    np.testing.assert_allclose(_value(gradient, "w"), np.outer(_X, _C), rtol=1e-12)


class _LogOffset(nnx.Module):
    """A valid logarithm whose default central stencil leaves its domain."""

    def __init__(self) -> None:
        self.offset = nnx.Param(jnp.array(0.0, dtype=jnp.float32))


def _log_loss(module: _LogOffset) -> jax.Array:
    return jnp.log1p(10000.0 * module.offset[...])


def _log_input_loss(_module: _LogOffset, value: jax.Array) -> jax.Array:
    return jnp.log1p(10000.0 * value)


@pytest.mark.parametrize("in_parameters", [False, True])
def test_explicit_step_keeps_a_logarithm_stencil_inside_its_domain(in_parameters: bool) -> None:
    """Step control checks the unchanged analytic derivative at the valid zero input."""
    module = _LogOffset()
    check = (
        (lambda: check_parameter_gradients(module, _log_loss))
        if in_parameters
        else lambda: check_input_gradients(module, _log_input_loss, 0.0)
    )
    with pytest.raises(AssertionError):
        check()
    if in_parameters:
        gradient = check_parameter_gradients(module, _log_loss, eps=1e-8)
        observed = _value(gradient, "offset")
    else:
        observed = np.asarray(check_input_gradients(module, _log_input_loss, 0.0, eps=1e-8))
    np.testing.assert_allclose(observed, 10000.0, rtol=1e-12)
    assert module.offset[...].dtype == jnp.float32
    assert float(module.offset[...]) == 0.0


@pytest.mark.parametrize("eps", [0.0, -1.0, float("nan"), float("inf"), True])
@pytest.mark.parametrize("in_parameters", [False, True])
def test_invalid_finite_difference_steps_are_refused(eps: float, in_parameters: bool) -> None:
    """A malformed diagnostic must fail before evaluation instead of hiding bad derivatives."""
    check = (
        (lambda: check_parameter_gradients(_Affine(), _loss_in_parameters, eps=eps))
        if in_parameters
        else lambda: check_input_gradients(_Affine(), _loss_in_input, _X, eps=eps)
    )
    with pytest.raises(ValueError, match="finite positive"):
        check()


@pytest.mark.parametrize("eps", [1e-4, 1e-6, 1e-8])
def test_explicit_steps_still_refuse_an_incorrect_derivative(eps: float) -> None:
    """Selecting a step never bypasses the comparison in either public checker."""
    module = _Affine()
    with pytest.raises(AssertionError):
        check_parameter_gradients(module, lambda m: jnp.sum(_doubled_tangent(m.w[...])), eps=eps)
    with pytest.raises(AssertionError):
        check_input_gradients(module, lambda m, x: jnp.sum(_doubled_tangent(m(x))), _X, eps=eps)
