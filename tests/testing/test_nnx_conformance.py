"""Each NNX conformance check passes a correct module and fails one built to break it."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax import core as jax_core

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


class CallTimeThreadsHeldLeaves(nnx.Module):
    """Two dropout leaves that hold streams, both given the call-time key, which overrides them."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.linear = nnx.Linear(3, 4, rngs=rngs)
        self.first = nnx.Dropout(0.5, rngs=rngs)
        self.second = nnx.Dropout(0.5, rngs=rngs)

    def __call__(self, x: jax.Array, *, rngs: nnx.Rngs | None = None) -> jax.Array:
        return self.second(self.first(jnp.tanh(self.linear(x)), rngs=rngs), rngs=rngs)


class CallTimePartlyThreaded(CallTimeThreadsHeldLeaves):
    """Gives the call-time key to one dropout leaf; the other draws from the stream it holds."""

    def __call__(self, x: jax.Array, *, rngs: nnx.Rngs | None = None) -> jax.Array:
        return self.second(self.first(jnp.tanh(self.linear(x)), rngs=rngs))


class CallTimeHeldNoise(nnx.Module):
    """Takes its dropout key at call time and adds noise from a stream it holds."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.linear = nnx.Linear(3, 4, rngs=rngs)
        self.dropout = nnx.Dropout(0.5)
        self.noise = rngs.sample.fork()

    def __call__(self, x: jax.Array, *, rngs: nnx.Rngs | None = None) -> jax.Array:
        y = self.dropout(jnp.tanh(self.linear(x)), rngs=rngs)
        return y + jax.random.normal(self.noise(), y.shape, y.dtype)


class CallTimeHeldKey(nnx.Module):
    """Takes its dropout key at call time and adds noise from a raw key it holds and splits."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.linear = nnx.Linear(3, 4, rngs=rngs)
        self.dropout = nnx.Dropout(0.5)
        self.key = nnx.Variable(rngs.sample())

    def __call__(self, x: jax.Array, *, rngs: nnx.Rngs | None = None) -> jax.Array:
        y = self.dropout(jnp.tanh(self.linear(x)), rngs=rngs)
        noise_key, self.key[...] = jax.random.split(self.key[...])
        return y + jax.random.normal(noise_key, y.shape, y.dtype)


class CallTimeEvalOverride(nnx.Module):
    """Keeps dropout on in ``.eval()`` with the caller's key, so eval output depends on the key."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.linear = nnx.Linear(3, 4, rngs=rngs)
        self.dropout = nnx.Dropout(0.5)

    def __call__(self, x: jax.Array, *, rngs: nnx.Rngs | None = None) -> jax.Array:
        return self.dropout(jnp.tanh(self.linear(x)), deterministic=False, rngs=rngs)


class DrawGood(nnx.Module):
    """A learnable Gaussian whose call draws a sample with the key it is given."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.linear = nnx.Linear(3, 4, rngs=rngs)

    def __call__(self, x: jax.Array, *, rngs: nnx.Rngs) -> jax.Array:
        mean = jnp.tanh(self.linear(x))
        return mean + jax.random.normal(rngs.sample(), mean.shape, mean.dtype)


class DrawIgnoresKey(nnx.Module):
    """Draws with a key it fixes itself, so the caller's key changes nothing."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.linear = nnx.Linear(3, 4, rngs=rngs)

    def __call__(self, x: jax.Array, *, rngs: nnx.Rngs) -> jax.Array:  # noqa: ARG002
        mean = jnp.tanh(self.linear(x))
        return mean + jax.random.normal(jax.random.key(7), mean.shape, mean.dtype)


class DrawHeldStream(DrawGood):
    """Adds noise from a stream it holds to the draw the caller's key controls."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        super().__init__(rngs=rngs)
        self.noise = rngs.sample.fork()

    def __call__(self, x: jax.Array, *, rngs: nnx.Rngs) -> jax.Array:
        y = super().__call__(x, rngs=rngs)
        return y + jax.random.normal(self.noise(), y.shape, y.dtype)


class DrawHeldTypedKey(DrawGood):
    """Adds noise from a typed key it holds and never advances."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        super().__init__(rngs=rngs)
        self.key = nnx.Variable(rngs.sample())

    def __call__(self, x: jax.Array, *, rngs: nnx.Rngs) -> jax.Array:
        y = super().__call__(x, rngs=rngs)
        return y + jax.random.normal(self.key[...], y.shape, y.dtype)


class DrawHeldRawKey(DrawGood):
    """Adds noise from a legacy ``uint32`` key it holds and never advances."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        super().__init__(rngs=rngs)
        self.key = nnx.Variable(jax.random.key_data(rngs.sample()))

    def __call__(self, x: jax.Array, *, rngs: nnx.Rngs) -> jax.Array:
        y = super().__call__(x, rngs=rngs)
        return y + jax.random.normal(jax.random.wrap_key_data(self.key[...]), y.shape, y.dtype)


class DrawOnlyWhenTraining(nnx.Module):
    """Returns the mean in ``.eval()``, so a caller sampling an evaluated model gets no sample."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.linear = nnx.Linear(3, 4, rngs=rngs)
        self.dropout = nnx.Dropout(0.5)

    def __call__(self, x: jax.Array, *, rngs: nnx.Rngs) -> jax.Array:
        return self.dropout(jnp.tanh(self.linear(x)) + 1.0, rngs=rngs)


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


class ReturnsLabel(nnx.Module):
    """Returns a string beside its array, as a metrics dict naming its kind does."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.linear = nnx.Linear(3, 4, rngs=rngs)

    def __call__(self, x: jax.Array) -> dict[str, object]:
        return {"y": self.linear(x), "kind": "linear"}


class ReturnsPythonFloat(nnx.Module):
    """Returns a host float, as a metric calling float() on its value does."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.linear = nnx.Linear(3, 4, rngs=rngs)

    def __call__(self, x: jax.Array) -> dict[str, object]:
        y = self.linear(x)
        return {"y": y, "mean": float(jnp.mean(y))}


class IntegerDropoutGood(nnx.Module):
    """Returns integer class ids, picked after dropout, as a token sampler or classifier does."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.linear = nnx.Linear(3, 8, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return jnp.argmax(self.dropout(self.linear(x)), axis=-1)


class IntegerBatchCoupled(nnx.Module):
    """Returns integer class ids of features centred on the batch mean."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.linear = nnx.Linear(3, 8, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        h = self.linear(x)
        return jnp.argmax(h - jnp.mean(h, axis=0), axis=-1)


class AlternatesOutputKey(nnx.Module):
    """Names its one output differently on alternate calls, so two calls differ only in keys."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.linear = nnx.Linear(3, 4, rngs=rngs)
        self.calls = nnx.BatchStat(jnp.zeros((), jnp.int32))

    def __call__(self, x: jax.Array) -> dict[str, jax.Array]:
        self.calls[...] += 1
        key = "even" if int(self.calls[...]) % 2 == 0 else "odd"
        return {key: self.linear(x)}


class TracedStructureDiffers(nnx.Module):
    """Returns one more output when called eagerly than under a transform."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.linear = nnx.Linear(3, 4, rngs=rngs)

    def __call__(self, x: jax.Array) -> dict[str, jax.Array]:
        y = self.linear(x)
        if isinstance(y, jax_core.Tracer):
            return {"y": y}
        return {"y": y, "eager_only": y}


class ReturnsNothing(nnx.Module):
    """Computes and returns nothing, as a training-lifecycle hook does."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.linear = nnx.Linear(3, 4, rngs=rngs)

    def __call__(self, x: jax.Array) -> None:
        del x


class ReturnsBatchSummary(nnx.Module):
    """Returns a per-row output and a scalar summary of the batch."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.linear = nnx.Linear(3, 4, rngs=rngs)

    def __call__(self, x: jax.Array) -> dict[str, jax.Array]:
        y = self.linear(x)
        return {"y": y, "batch_size": jnp.asarray(x.shape[0])}


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
    is_differentiable: bool = True,
    rng_streams: tuple[str, ...] = (),
    draws: bool = False,
) -> Case:
    return Case(
        build=lambda rngs: module(rngs=rngs),
        inputs=_inputs,
        has_dropout=has_dropout,
        has_batch_stats=has_batch_stats,
        is_rowwise=is_rowwise,
        is_differentiable=is_differentiable,
        rng_streams=rng_streams,
        draws=draws,
    )


def _draw_case(
    module: Callable[..., nnx.Module], *, streams: tuple[str, ...] = ("sample",)
) -> Case:
    """A case for a module whose call draws a sample with a call-time key."""
    return _case(module, is_rowwise=False, rng_streams=streams, draws=True)


def _call_time_case(module: Callable[..., nnx.Module]) -> Case:
    """A case for a module with dropout that takes its dropout key at call time."""
    return _case(module, has_dropout=True, rng_streams=("dropout",))


GOOD = _case(Good, has_dropout=True, has_batch_stats=True, is_rowwise=False)
ROWWISE = _case(RowwiseGood, has_dropout=True)
CUSTOM_LEAF = _case(CustomLeafGood, has_dropout=True)
CALL_TIME = _call_time_case(CallTimeGood)
CALL_TIME_HELD = _call_time_case(CallTimeThreadsHeldLeaves)
PARTLY_THREADED = _call_time_case(CallTimePartlyThreaded)
HELD_NOISE = _call_time_case(CallTimeHeldNoise)
HELD_KEY = _call_time_case(CallTimeHeldKey)
INTEGER_OUTPUT = _case(IntegerDropoutGood, has_dropout=True, is_differentiable=False)
VARYING_KEYS = _case(AlternatesOutputKey, has_dropout=True, is_differentiable=False)
DRAW = _draw_case(DrawGood)

# Each broken module, the check it must fail, the claims its case makes, and the failure its defect
# must produce: a module that fails the check for another reason does not test that clause.
BROKEN = [
    ("state_structure", _case(LazyParam), "changed the state's structure"),
    ("state_structure", _case(CachesPythonHash), "changed the state's structure"),
    ("state_holds_arrays", _case(PythonValueInVariable), "Variables holding Python values"),
    ("tree_mode_split", _case(SharedRngs, has_dropout=True), "tree mode refused"),
    ("eval_deterministic", _case(EvalOverride, has_dropout=True), r"two \.eval\(\) calls differ"),
    (
        "train_dropout_varies",
        _case(DropoutNeverOn, has_dropout=True),
        r"two \.train\(\) calls agree",
    ),
    ("train_dropout_varies", _call_time_case(CallTimeIgnoresKey), "different keys agree"),
    ("train_dropout_varies", _call_time_case(CallTimeFixedMask), "different keys agree"),
    ("train_dropout_varies", _call_time_case(CallTimeHostRandom), "the same key differ"),
    ("train_dropout_varies", PARTLY_THREADED, "the same key differ"),
    ("train_dropout_varies", HELD_NOISE, "the same key differ"),
    ("train_dropout_varies", HELD_KEY, "the same key differ"),
    ("eval_deterministic", _call_time_case(CallTimeEvalOverride), "with different keys differ"),
    (
        "output_follows_key",
        _draw_case(DrawIgnoresKey),
        r"\.eval\(\) calls with different keys agree",
    ),
    ("output_follows_key", _draw_case(DrawHeldStream), "the same key differ"),
    ("output_follows_key", _draw_case(DrawHeldTypedKey), "the same key differ"),
    ("output_follows_key", _draw_case(DrawHeldRawKey), "the same key differ"),
    (
        "output_follows_key",
        _draw_case(DrawOnlyWhenTraining, streams=("dropout",)),
        r"\.eval\(\) calls with different keys agree",
    ),
    ("view_matches_modes", _case(ModeWithoutView, has_dropout=True), "outputs differ"),
    ("view_matches_modes", _case(StoredRngs, has_dropout=True), "RNG streams advanced apart"),
    (
        "batch_stats",
        _case(FrozenBatchNorm, has_batch_stats=True, is_rowwise=False),
        "left the running statistics unchanged",
    ),
    ("jit_matches_eager", _case(Retraces), "traced again"),
    ("jit_matches_eager", _case(TracedStructureDiffers), "differ in structure"),
    ("rows_independent", _case(BatchCoupled), "rows mapped one at a time"),
    ("rows_independent", _case(WritesSharedState), "nnx.vmap refused"),
    (
        "rows_independent",
        _case(IntegerBatchCoupled, is_differentiable=False),
        "integer outputs differ",
    ),
    ("gradient_finite", _case(DetachedParams), "zero everywhere"),
    ("gradient_matches_finite_differences", _case(WrongDerivative), "finite differences"),
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
    [GOOD, ROWWISE, CUSTOM_LEAF, CALL_TIME, CALL_TIME_HELD, INTEGER_OUTPUT, DRAW],
    ids=[
        "batch-norm",
        "rowwise",
        "custom-leaf",
        "call-time",
        "call-time-held",
        "integer-output",
        "draw",
    ],
)
def test_a_correct_module_passes_every_check_it_claims(check: str, case: Case) -> None:
    run, applies = CHECKS[check]
    if applies(case):
        run(case)


@pytest.mark.parametrize(
    ("check", "case", "failure"), BROKEN, ids=[check for check, _, _ in BROKEN]
)
def test_a_module_built_to_break_a_check_fails_it(check: str, case: Case, failure: str) -> None:
    run, applies = CHECKS[check]

    assert applies(case)
    with pytest.raises(ConformanceError, match=failure):
        run(case)


@pytest.mark.parametrize(("check", "case"), REFUSED, ids=[check for check, _ in REFUSED])
def test_a_module_a_transform_refuses_fails_the_check(check: str, case: Case) -> None:
    run, applies = CHECKS[check]

    assert applies(case)
    with pytest.raises(ConformanceError, match="refused"):
        run(case)


@pytest.mark.parametrize("check", sorted(CHECKS))
@pytest.mark.parametrize(
    ("module", "leaf"), [(ReturnsLabel, "str"), (ReturnsPythonFloat, "float")], ids=["str", "float"]
)
def test_a_call_returning_a_non_array_fails_every_check_it_reaches(
    check: str, module: Callable[..., nnx.Module], leaf: str
) -> None:
    """A case's call returns a pytree of arrays; any other leaf is reported, not crashed on."""
    run, applies = CHECKS[check]
    case = _case(module, has_dropout=True, has_batch_stats=True)
    if not applies(case) or check == "state_holds_arrays":  # it inspects state, never calls
        return

    # Under jax.grad the float module fails at float(tracer), before it returns: a refusal.
    with pytest.raises(ConformanceError, match=f"a {leaf}, not an array|refused the module"):
        run(case)


def test_outputs_whose_keys_vary_with_dropout_count_as_differing() -> None:
    """Two train calls naming their outputs differently differ, though their values agree."""
    CHECKS["train_dropout_varies"][0](VARYING_KEYS)


@pytest.mark.parametrize("check", sorted(CHECKS))
def test_a_call_returning_no_arrays_fails_every_check_it_reaches(check: str) -> None:
    """With nothing to compare, a check would pass having checked nothing."""
    run, applies = CHECKS[check]
    case = _case(ReturnsNothing)
    if not applies(case) or check == "state_holds_arrays":  # it inspects state, never calls
        return

    with pytest.raises(ConformanceError, match="returned no arrays"):
        run(case)


def test_an_output_without_a_batch_axis_fails_rows_independent() -> None:
    """A batch summary cannot be compared row by row; the case must not claim rows."""
    with pytest.raises(ConformanceError, match="no leading batch axis"):
        CHECKS["rows_independent"][0](_case(ReturnsBatchSummary))


def test_integer_inputs_stay_integers_in_the_gradient_checks() -> None:
    """Only floating inputs are cast to float64; an integer label indexes a table."""
    case = Case(
        build=lambda rngs: EmbedsLabels(rngs=rngs),
        inputs=lambda key: (jax.random.randint(key, (4,), 0, 5),),
    )

    CHECKS["gradient_finite"][0](case)
    CHECKS["gradient_matches_finite_differences"][0](case)


def test_every_check_has_a_module_that_breaks_it() -> None:
    assert {check for check, _, _ in BROKEN} == set(CHECKS)


def test_module_classes_finds_concrete_modules_in_a_package() -> None:
    """Positive control: flax's own layers are found; a package defining none yields none."""
    assert module_classes(substrax.rng) == {}
    found = module_classes(nnx)

    assert "flax.nnx.nn.linear.Linear" in found
    assert all(issubclass(cls, nnx.Module) for cls in found.values())


def test_a_case_that_draws_owes_the_key_check_and_not_eval_determinism() -> None:
    """Sampling is the call's job, so two keys must differ in ``.eval()`` too."""
    owed = {check for check, (_, applies) in CHECKS.items() if applies(DRAW)}

    assert "output_follows_key" in owed
    assert "eval_deterministic" not in owed
    assert not CHECKS["output_follows_key"][1](CALL_TIME), "a dropout-only case does not draw"


@pytest.mark.parametrize("graph", [True, False], ids=["graph-mode", "tree-mode"])
def test_a_correct_draw_passes_the_key_check_in_both_nnx_modes(graph: bool) -> None:
    with nnx.set_graph_mode(graph):
        CHECKS["output_follows_key"][0](DRAW)


def test_a_case_that_draws_names_its_call_time_streams() -> None:
    """A module that draws takes its key at call time; holding one is the defect to find."""
    with pytest.raises(ValueError, match=r"draws.*rng_streams"):
        _case(DrawGood, is_rowwise=False, draws=True)


def test_a_case_that_draws_does_not_claim_independent_rows() -> None:
    """One key drawn for a batch and the same key drawn per row give different samples."""
    with pytest.raises(ValueError, match=r"draws.*is_rowwise"):
        _case(DrawGood, rng_streams=("sample",), draws=True)


def test_the_conformance_streams_are_the_standard_streams() -> None:
    from substrax.testing import nnx_conformance, pytest_plugin  # noqa: PLC0415

    assert nnx_conformance.STREAMS is substrax.rng.STANDARD_STREAMS
    assert not hasattr(pytest_plugin, "RNGS_FIXTURE_STREAMS"), (
        "the fixture reads the one definition"
    )
