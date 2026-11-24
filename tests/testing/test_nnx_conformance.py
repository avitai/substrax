"""Each NNX conformance check passes a correct module and fails one built to break it."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

import substrax.rng
from substrax.testing.nnx_conformance import (
    Case,
    CHECKS,
    ConformanceError,
    module_classes,
)


class Good(nnx.Module):
    """A dense layer with dropout and batch statistics, written to every check."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.linear = nnx.Linear(3, 4, rngs=rngs)
        self.norm = nnx.BatchNorm(4, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.dropout(nnx.silu(self.norm(self.linear(x))))


class RowwiseGood(nnx.Module):
    """A dense layer whose rows are independent in every mode."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.linear = nnx.Linear(3, 4, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.dropout(jnp.tanh(self.linear(x)))


class ModeWithoutView(nnx.Module):
    """A custom dropout leaf owning its mode: ``.eval()`` reaches it, ``nnx.view`` does not."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.linear = nnx.Linear(3, 4, rngs=rngs)
        self.deterministic = False
        self.stream = rngs.dropout.fork()

    def __call__(self, x: jax.Array) -> jax.Array:
        y = jnp.tanh(self.linear(x))
        if self.deterministic:
            return y
        return jnp.where(jax.random.bernoulli(self.stream(), 0.5, y.shape), 2 * y, 0.0)


class CustomLeafGood(ModeWithoutView):
    """The same leaf with ``set_view``, as flax's own stochastic layers define it."""

    def set_view(self, deterministic: bool | None = None) -> None:
        """Set the mode ``nnx.view`` asks for; ``None`` keeps it."""
        if deterministic is not None:
            self.deterministic = deterministic


class StoredRngs(nnx.Module):
    """Keeps an ``nnx.Rngs``, whose ``default`` stream answers ``nnx.view``'s ``set_view`` probe."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.rngs = rngs
        self.child = RowwiseGood(rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.child(x)


class CallTimeGood(nnx.Module):
    """Takes its dropout key at call time, as flax's gemma example does."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.linear = nnx.Linear(3, 4, rngs=rngs)
        self.dropout = nnx.Dropout(0.5)

    def __call__(self, x: jax.Array, *, rngs: nnx.Rngs | None = None) -> jax.Array:
        return self.dropout(jnp.tanh(self.linear(x)), rngs=rngs)


class CallTimeIgnoresKey(nnx.Module):
    """Declares a call-time stream but draws from a stream it holds, so the key changes nothing."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.linear = nnx.Linear(3, 4, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, rngs=rngs)

    def __call__(self, x: jax.Array, *, rngs: nnx.Rngs | None = None) -> jax.Array:  # noqa: ARG002
        return self.dropout(jnp.tanh(self.linear(x)))


class CallTimeFixedMask(nnx.Module):
    """Draws its mask from a key it fixes itself, so different keys give one mask."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.linear = nnx.Linear(3, 4, rngs=rngs)
        self.dropout = nnx.Dropout(0.5)

    def __call__(self, x: jax.Array, *, rngs: nnx.Rngs | None = None) -> jax.Array:  # noqa: ARG002
        return self.dropout(jnp.tanh(self.linear(x)), rngs=nnx.Rngs(dropout=jax.random.key(7)))


class CallTimeHostRandom(nnx.Module):
    """Adds noise from NumPy's global generator, which no key or module state controls."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.linear = nnx.Linear(3, 4, rngs=rngs)
        self.dropout = nnx.Dropout(0.5)

    def __call__(self, x: jax.Array, *, rngs: nnx.Rngs | None = None) -> jax.Array:
        y = self.dropout(jnp.tanh(self.linear(x)), rngs=rngs)
        return y + jnp.asarray(np.random.default_rng().normal(size=y.shape), y.dtype)


class CallTimeEvalOverride(nnx.Module):
    """Keeps dropout on in ``.eval()`` with the caller's key, so eval output depends on the key."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.linear = nnx.Linear(3, 4, rngs=rngs)
        self.dropout = nnx.Dropout(0.5)

    def __call__(self, x: jax.Array, *, rngs: nnx.Rngs | None = None) -> jax.Array:
        return self.dropout(jnp.tanh(self.linear(x)), deterministic=False, rngs=rngs)


class LazyParam(nnx.Module):
    """Creates its weight on the first call."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.rngs = rngs

    def __call__(self, x: jax.Array) -> jax.Array:
        if not hasattr(self, "kernel"):
            self.kernel = nnx.Param(jax.random.normal(self.rngs.params(), (3, 4)))
        return x @ self.kernel[...]


class PythonValueInVariable(nnx.Module):
    """Keeps a hyperparameter in a Variable."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.linear = nnx.Linear(3, 4, rngs=rngs)
        self.rate = nnx.Variable(0.5)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.linear(x)


class CachesPythonHash(nnx.Module):
    """Caches a Python-int digest of its weight in a Variable that starts as None."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.linear = nnx.Linear(3, 4, rngs=rngs)
        self.digest: nnx.Cache[int | None] = nnx.Cache(None)

    def __call__(self, x: jax.Array) -> jax.Array:
        self.digest.set_value(2**40 + id(self))
        return self.linear(x)


class RngsHolder(nnx.Module):
    """Keeps the Rngs it was given."""

    def __init__(self, rngs: nnx.Rngs) -> None:
        self.rngs = rngs


class SharedRngs(nnx.Module):
    """Holds one Rngs in two places, which tree mode refuses."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.rngs = rngs
        self.holder = RngsHolder(rngs)
        self.child = RowwiseGood(rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.child(x)


class EvalOverride(nnx.Module):
    """Forwards ``deterministic=False``, so dropout stays on after ``.eval()``."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.linear = nnx.Linear(3, 4, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, rngs=rngs)

    def __call__(self, x: jax.Array, deterministic: bool = False) -> jax.Array:
        return self.dropout(self.linear(x), deterministic=deterministic)


class DropoutNeverOn(nnx.Module):
    """Claims dropout but always calls it deterministically."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.linear = nnx.Linear(3, 4, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.dropout(self.linear(x), deterministic=True)


class FrozenBatchNorm(nnx.Module):
    """Always reads its running statistics, so ``.train()`` never updates them."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.norm = nnx.BatchNorm(3, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.norm(x, use_running_average=True)


class Retraces(nnx.Module):
    """Changes a static attribute on every call, so each jitted call traces again."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.linear = nnx.Linear(3, 4, rngs=rngs)
        self.calls = 0

    def __call__(self, x: jax.Array) -> jax.Array:
        self.calls += 1
        return self.linear(x)


class BatchCoupled(nnx.Module):
    """Centres over the batch, so a row's output depends on the other rows."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.linear = nnx.Linear(3, 4, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        y = self.linear(x)
        return y - jnp.mean(y, axis=0)


class WritesSharedState(nnx.Module):
    """Records its last output in a Variable, so mapping it writes a row into shared state."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.linear = nnx.Linear(3, 4, rngs=rngs)
        self.last = nnx.BatchStat(jnp.zeros(4))

    def __call__(self, x: jax.Array) -> jax.Array:
        y = self.linear(x)
        self.last[...] = jnp.mean(y, axis=0)
        return y


class PythonBranch(nnx.Module):
    """Branches in Python on a traced value, which ``nnx.jit`` refuses."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.linear = nnx.Linear(3, 4, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        y = self.linear(x)
        return y if jnp.sum(y) > 0 else -y


class EmbedsLabels(nnx.Module):
    """Looks integer labels up in a table, so its input must stay an integer in the float64 check.

    The table is a plain ``nnx.Param``: ``nnx.Embed`` fixes its compute dtype when built
    (``self.dtype = dtype or self.embedding.dtype``), so a float64 copy of it still computes in
    float32, which is not what this module tests.
    """

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.table = nnx.Param(jax.random.normal(rngs.params(), (5, 4)))

    def __call__(self, labels: jax.Array) -> jax.Array:
        return jnp.tanh(jnp.take(self.table[...], labels, axis=0))


class DetachedParams(nnx.Module):
    """Stops the gradient at its weights."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.linear = nnx.Linear(3, 4, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return jax.lax.stop_gradient(self.linear(x))


def _wrong_square(x: jax.Array) -> jax.Array:
    """``x**2`` whose derivative is ``3 x``: the extra term has value zero and gradient ``x``."""
    square = x**2
    return square + 0.5 * (square - jax.lax.stop_gradient(square))


class WrongDerivative(nnx.Module):
    """Squares through a derivative that is 1.5 times too large."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.linear = nnx.Linear(3, 4, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return _wrong_square(self.linear(x))


def _inputs(key: jax.Array) -> tuple[jax.Array, ...]:
    return (jax.random.normal(key, (4, 3)),)


def _case(
    module: Callable[..., nnx.Module],
    *,
    has_dropout: bool = False,
    has_batch_stats: bool = False,
    is_rowwise: bool = True,
    rng_streams: tuple[str, ...] = (),
) -> Case:
    return Case(
        build=lambda rngs: module(rngs=rngs),
        inputs=_inputs,
        has_dropout=has_dropout,
        has_batch_stats=has_batch_stats,
        is_rowwise=is_rowwise,
        rng_streams=rng_streams,
    )


GOOD = _case(Good, has_dropout=True, has_batch_stats=True, is_rowwise=False)
ROWWISE = _case(RowwiseGood, has_dropout=True)
CUSTOM_LEAF = _case(CustomLeafGood, has_dropout=True)
CALL_TIME = _case(CallTimeGood, has_dropout=True, rng_streams=("dropout",))

# Each broken module, the check it must fail, and the claims its case makes.
BROKEN = [
    ("state_structure", _case(LazyParam)),
    ("state_structure", _case(CachesPythonHash)),
    ("state_holds_arrays", _case(PythonValueInVariable)),
    ("tree_mode_split", _case(SharedRngs, has_dropout=True)),
    ("eval_deterministic", _case(EvalOverride, has_dropout=True)),
    ("train_dropout_varies", _case(DropoutNeverOn, has_dropout=True)),
    ("train_dropout_varies", _case(CallTimeIgnoresKey, has_dropout=True, rng_streams=("dropout",))),
    ("train_dropout_varies", _case(CallTimeFixedMask, has_dropout=True, rng_streams=("dropout",))),
    ("train_dropout_varies", _case(CallTimeHostRandom, has_dropout=True, rng_streams=("dropout",))),
    ("eval_deterministic", _case(CallTimeEvalOverride, has_dropout=True, rng_streams=("dropout",))),
    ("view_matches_modes", _case(ModeWithoutView, has_dropout=True)),
    ("view_matches_modes", _case(StoredRngs, has_dropout=True)),
    ("batch_stats", _case(FrozenBatchNorm, has_batch_stats=True, is_rowwise=False)),
    ("jit_matches_eager", _case(Retraces)),
    ("rows_independent", _case(BatchCoupled)),
    ("rows_independent", _case(WritesSharedState)),
    ("gradient_finite", _case(DetachedParams)),
    ("gradient_matches_finite_differences", _case(WrongDerivative)),
]

# Each check that runs a module under a transform, and a module the transform refuses: a
# refusal is the module failing the check, so it raises ConformanceError like any other.
REFUSED = [
    ("jit_matches_eager", _case(PythonBranch)),
    ("jit_matches_eager", _case(CachesPythonHash)),
    ("rows_independent", _case(CachesPythonHash)),
    ("gradient_finite", _case(EvalOverride, has_dropout=True)),
    ("gradient_finite", _case(WritesSharedState)),
    ("gradient_matches_finite_differences", _case(EvalOverride, has_dropout=True)),
    ("gradient_matches_finite_differences", _case(WritesSharedState)),
]


@pytest.mark.parametrize("check", sorted(CHECKS))
@pytest.mark.parametrize(
    "case",
    [GOOD, ROWWISE, CUSTOM_LEAF, CALL_TIME],
    ids=["batch-norm", "rowwise", "custom-leaf", "call-time"],
)
def test_a_correct_module_passes_every_check_it_claims(check: str, case: Case) -> None:
    run, applies = CHECKS[check]
    if applies(case):
        run(case)


@pytest.mark.parametrize(("check", "case"), BROKEN, ids=[check for check, _ in BROKEN])
def test_a_module_built_to_break_a_check_fails_it(check: str, case: Case) -> None:
    run, applies = CHECKS[check]

    assert applies(case)
    with pytest.raises(ConformanceError):
        run(case)


@pytest.mark.parametrize(("check", "case"), REFUSED, ids=[check for check, _ in REFUSED])
def test_a_module_a_transform_refuses_fails_the_check(check: str, case: Case) -> None:
    run, applies = CHECKS[check]

    assert applies(case)
    with pytest.raises(ConformanceError, match="refused"):
        run(case)


def test_integer_inputs_stay_integers_in_the_gradient_checks() -> None:
    """Only floating inputs are cast to float64; an integer label indexes a table."""
    case = Case(
        build=lambda rngs: EmbedsLabels(rngs=rngs),
        inputs=lambda key: (jax.random.randint(key, (4,), 0, 5),),
    )

    CHECKS["gradient_finite"][0](case)
    CHECKS["gradient_matches_finite_differences"][0](case)


def test_every_check_has_a_module_that_breaks_it() -> None:
    assert {check for check, _ in BROKEN} == set(CHECKS)


def test_module_classes_finds_concrete_modules_in_a_package() -> None:
    """Positive control: flax's own layers are found; a package defining none yields none."""
    assert module_classes(substrax.rng) == {}
    found = module_classes(nnx)

    assert "flax.nnx.nn.linear.Linear" in found
    assert all(issubclass(cls, nnx.Module) for cls in found.values())
