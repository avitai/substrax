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


class _Configured(_Affine):
    """An affine module that also keeps a Python value in a Variable."""

    def __init__(self) -> None:
        super().__init__()
        self.rate = nnx.Variable(0.5)


def test_a_python_value_in_the_state_is_left_as_it_is() -> None:
    gradient = check_parameter_gradients(_Configured(), _loss_in_parameters)

    np.testing.assert_allclose(_value(gradient, "w"), np.outer(_X, _C), rtol=1e-12)


class _Wide(nnx.Module):
    """Two 64-wide layers through ``tanh``: 8,320 parameters with real curvature."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.first = nnx.Linear(64, 64, rngs=rngs)
        self.second = nnx.Linear(64, 64, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.second(jnp.tanh(self.first(x)))


def test_a_wide_module_with_a_correct_gradient_passes() -> None:
    """The finite-difference step is a length in parameter space, whatever the parameter count.

    An unnormalised random direction has norm about sqrt(8320) = 91 here, so ``check_grads``'
    1e-4 step moved the parameters by about 1e-2 and its truncation error failed a gradient
    autodiff computes exactly.
    """
    module = _Wide(rngs=nnx.Rngs(0))
    x = 3.0 * jnp.linspace(-1.0, 1.0, 4 * 64).reshape(4, 64)

    check_parameter_gradients(
        module, lambda m: jnp.sum(jnp.sin(m(x.astype(m.first.kernel[...].dtype)))), modes=("rev",)
    )


class _Scalar(nnx.Module):
    """One parameter, so the unit direction is the parameter axis and the step is exact."""

    def __init__(self, value: float) -> None:
        self.p = nnx.Param(jnp.array(value, dtype=jnp.float32))


def test_a_correct_gradient_with_strong_curvature_passes() -> None:
    """The step keeps the central difference's truncation error below the tolerance.

    For ``sin(150 p)`` at 0 a central difference with step ``h`` errs, relative to the derivative,
    by about ``(150 h)^2 / 6``. ``check_grads`` steps by its ``eps`` times a random tangent (1.76
    at its seed): at ``eps = 1e-4`` the error is 1.2e-4, above its float64 tolerance of 1e-5, and
    at ``eps = 1e-6`` it is 1.2e-8.
    """
    check_parameter_gradients(_Scalar(0.0), lambda m: jnp.sin(150.0 * m.p[...]))


def test_a_correct_gradient_near_a_kink_passes() -> None:
    """A step shorter than the distance to a kink does not difference across it.

    The parameter is 5e-5 from ``leaky_relu``'s kink. ``check_grads`` steps by its ``eps`` times a
    random tangent (1.76 at its seed): at ``eps = 1e-4`` the step reaches 1.8e-4, straddles the
    kink and differences to 1.26 where the derivative gives 1.76; at ``eps = 1e-6`` it stays on
    one side.
    """
    check_parameter_gradients(_Scalar(5e-5), lambda m: jax.nn.leaky_relu(m.p[...], 0.2))
