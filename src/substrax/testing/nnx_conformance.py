"""Properties every ``nnx.Module`` owes under NNX's transforms, checked on a repository's cases.

A repository describes how to build and call each of its module classes as a :class:`Case`,
and runs every check of :data:`CHECKS` whose claim the case makes. :func:`module_classes` gives
the denominator: every concrete ``nnx.Module`` a package defines, so a class without a case is
found rather than silently skipped. The checks:

- ``state_structure``: a forward pass leaves the state's paths, types, shapes and dtypes, and
  the graph definition, as ``__init__`` built them. A parameter created on the first call is
  missing from an ``nnx.Optimizer`` built before it.
- ``state_holds_arrays``: every Variable holds an array; a Python value in a Variable is traced,
  comes back from ``nnx.jit`` as an array and cannot be differentiated in tree mode.
- ``tree_mode_split``: ``nnx.split(graph=False)`` succeeds and the merged module computes what
  the module does. Tree mode, flax FLIP 5310's planned default, refuses shared Variables.
- ``eval_deterministic``: after ``.eval()`` two calls agree exactly and no RNG stream advances.
- ``train_dropout_varies``: after ``.train()`` a module with dropout draws a fresh mask per call.
- ``batch_stats``: ``.train()`` updates ``nnx.BatchStat``; ``.eval()`` leaves it alone.
- ``view_matches_modes``: ``nnx.view(module, deterministic=..., use_running_average=...)``
  computes what ``.eval()`` and ``.train()`` do and advances the RNG streams they advance.
  ``.eval()`` sets every ``deterministic`` attribute, but ``nnx.view`` reaches a module only
  through its ``set_view``, so a custom leaf owning its mode without one keeps its old mode in a
  view; and ``nnx.view`` calls ``set_view`` on anything answering ``hasattr``, which a stored
  ``nnx.Rngs`` with a ``default`` stream does by drawing a key.
- ``jit_matches_eager``: ``nnx.jit`` computes what an eager call does, and a second call of the
  same shapes does not trace again.
- ``rows_independent``: in ``.eval()``, ``nnx.vmap`` over single rows reproduces the batched call.
- ``gradient_finite``: the float32 parameter gradient is finite and not zero everywhere, taken
  over the ``nnx.Param`` part of a split, the form tree mode keeps.
- ``gradient_matches_finite_differences``: in float64 the parameter gradient agrees with finite
  differences (:func:`substrax.testing.gradients.check_parameter_gradients`). A finite difference
  across a kink is no derivative, so a case gives a module a smooth activation where it can.

Two computations of one function are compared with ``DEPTH * eps(dtype) * scale``, where
``DEPTH`` bounds the roundings an output of a small test module passes through and ``scale`` is
the output's largest magnitude, at least 1; a semantic difference (a skipped branch, a constant
key, a wrong mode) is of the order of ``scale``. Checks comparing two runs start both from cloned
state, RNG counts included, so a module that draws randomness in the wrong mode fails
``eval_deterministic`` alone rather than every check. A failed check raises
:class:`ConformanceError`, and so does a transform refusing the module: flax's errors (a Variable
mutated from another trace level, as dropout drawn or statistics written in ``.eval()`` are under
a gradient), JAX's tracer errors (Python control flow on a traced value) and the ``ValueError`` of
a split or a mapping flax cannot do. The refusal is chained as the error's cause.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from types import ModuleType

import jax
import jax.numpy as jnp
from flax import nnx
from flax.errors import FlaxError
from flax.typing import Key

from substrax.nnx_typing import NnxState
from substrax.rng import rngs_from_seed
from substrax.testing.gradients import as_float64, check_parameter_gradients
from substrax.testing.traces import TraceCounter
from substrax.typing import PyTree


DEPTH = 1e4
"""A bound on the roundings any output of a small test module passes through."""

STREAMS = ("default", "params", "dropout", "sample")
"""The streams a case's module is built with."""

_SEED = 0


class ConformanceError(AssertionError):
    """A module does not have a property a conformance check requires of it."""


_REFUSALS = (FlaxError, jax.errors.JAXTypeError, ValueError)
"""What flax and JAX raise when a transform refuses a module (see the module docstring)."""


@contextmanager
def _refused(transform: str) -> Generator[None]:
    """Raise a transform's refusal of the module as a :class:`ConformanceError`."""
    try:
        yield
    except _REFUSALS as error:
        message = f"{transform} refused the module: {type(error).__name__}: {error}"
        raise ConformanceError(message) from error


def _call(module: nnx.Module, *inputs: jax.Array) -> PyTree:
    return module(*inputs)


@dataclass(frozen=True, slots=True, kw_only=True)
class Case:
    """How to build one module, give it inputs and call it, and what the module claims.

    Attributes:
        build: Builds the module from an ``nnx.Rngs`` with the :data:`STREAMS`.
        inputs: Returns the positional array inputs from a key; every array has a leading batch
            axis.
        call: Calls the module on the inputs and returns a pytree of arrays.
        has_dropout: The module applies dropout, so calls after ``.train()`` differ.
        has_batch_stats: The module keeps ``nnx.BatchStat`` running statistics.
        is_rowwise: In ``.eval()``, output row ``i`` depends on input row ``i`` alone.
        is_differentiable: The output is differentiable in the parameters.
    """

    build: Callable[[nnx.Rngs], nnx.Module]
    inputs: Callable[[jax.Array], tuple[jax.Array, ...]]
    call: Callable[..., PyTree] = _call
    has_dropout: bool = False
    has_batch_stats: bool = False
    is_rowwise: bool = True
    is_differentiable: bool = True


def module_classes(package: ModuleType) -> dict[str, type[nnx.Module]]:
    """Every concrete ``nnx.Module`` subclass a package defines, by qualified name.

    Every module of the package is imported, so a class reachable only through an import that
    fails is not skipped: the import error propagates.

    Args:
        package: An imported package.

    Returns:
        ``{"<module>.<qualname>": class}`` for classes defined in the module that holds them,
        abstract ones excluded.
    """
    modules = [package]
    if hasattr(package, "__path__"):
        modules += [
            importlib.import_module(info.name)
            for info in pkgutil.walk_packages(package.__path__, f"{package.__name__}.")
        ]
    return {
        f"{value.__module__}.{value.__qualname__}": value
        for module in modules
        for value in vars(module).values()
        if inspect.isclass(value)
        and issubclass(value, nnx.Module)
        and value.__module__ == module.__name__
        and not inspect.isabstract(value)
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConformanceError(message)


def _built(case: Case) -> tuple[nnx.Module, tuple[jax.Array, ...]]:
    rngs = rngs_from_seed(_SEED, streams=STREAMS)
    return case.build(rngs), case.inputs(rngs_from_seed(_SEED, streams=("inputs",)).inputs())


def _arrays(tree: PyTree) -> list[jax.Array]:
    return [
        leaf for leaf in jax.tree.leaves(tree) if jnp.issubdtype(jnp.result_type(leaf), jnp.inexact)
    ]


def _value(leaf: PyTree) -> PyTree:
    return leaf.get_value() if isinstance(leaf, nnx.Variable) else leaf


def _path(path: tuple[Key, ...]) -> str:
    return "/".join(map(str, path))


def _require_same_function(actual: PyTree, expected: PyTree, what: str) -> None:
    pairs = list(zip(_arrays(actual), _arrays(expected), strict=True))
    for a, b in pairs:
        scale = max(1.0, float(jnp.max(jnp.abs(b))))
        bound = DEPTH * float(jnp.finfo(b.dtype).eps) * scale
        error = float(jnp.max(jnp.abs(a - b))) if a.shape == b.shape else float("inf")
        _require(error <= bound, f"{what}: max |difference| {error:.3g} exceeds {bound:.3g}")


def _structure(module: nnx.Module) -> list[tuple[str, str, tuple[int, ...], str]]:
    return [
        (
            _path(path),
            type(leaf).__name__,
            jnp.shape(jnp.asarray(_value(leaf))),
            str(jnp.asarray(_value(leaf)).dtype),
        )
        for path, leaf in nnx.to_flat_state(nnx.state(module))
    ]


def _rng_counts(module: nnx.Module) -> list[int]:
    return [int(count[...]) for _, count in nnx.to_flat_state(nnx.state(module, nnx.RngCount))]


def _batch_stats(module: nnx.Module) -> list[jax.Array]:
    return [leaf[...] for _, leaf in nnx.to_flat_state(nnx.state(module, nnx.BatchStat))]


def _scalar(output: PyTree) -> jax.Array:
    return sum((jnp.sum(jnp.sin(leaf)) for leaf in _arrays(output)), start=jnp.zeros(()))


def check_state_structure(case: Case) -> None:
    """A forward pass leaves the state and graph definition as ``__init__`` built them."""
    module, inputs = _built(case)
    module.train()
    before, graph = _structure(module), nnx.graphdef(module)
    case.call(module, *inputs)
    _require(_structure(module) == before, "a forward pass changed the state's structure")
    _require(nnx.graphdef(module) == graph, "a forward pass changed the graph definition")


def check_state_holds_arrays(case: Case) -> None:
    """Every Variable holds an array, not a Python value."""
    module, _ = _built(case)
    python_values = [
        _path(path)
        for path, leaf in nnx.to_flat_state(nnx.state(module))
        if not isinstance(_value(leaf), jax.Array)
    ]
    _require(not python_values, f"Variables holding Python values: {python_values}")


def check_tree_mode_split(case: Case) -> None:
    """Tree mode can split the module, and the merged module computes what it does."""
    module, inputs = _built(case)
    module.eval()
    expected = case.call(nnx.clone(module), *inputs)
    with _refused("tree mode"):
        graphdef, state = nnx.split(module, graph=False)
        actual = case.call(nnx.merge(graphdef, state), *inputs)
    _require_same_function(actual, expected, "tree mode")


def check_eval_deterministic(case: Case) -> None:
    """After ``.eval()`` two calls agree exactly and no RNG stream advances."""
    module, inputs = _built(case)
    module.eval()
    counts = _rng_counts(module)
    first, second = case.call(module, *inputs), case.call(module, *inputs)
    pairs = zip(_arrays(first), _arrays(second), strict=True)
    _require(all(bool(jnp.array_equal(a, b)) for a, b in pairs), "two .eval() calls differ")
    _require(_rng_counts(module) == counts, ".eval() advanced an RNG stream")


def check_train_dropout_varies(case: Case) -> None:
    """After ``.train()`` a module with dropout draws a fresh mask per call."""
    module, inputs = _built(case)
    module.train()
    first, second = case.call(module, *inputs), case.call(module, *inputs)
    pairs = zip(_arrays(first), _arrays(second), strict=True)
    _require(any(not bool(jnp.array_equal(a, b)) for a, b in pairs), "two .train() calls agree")


def check_batch_stats(case: Case) -> None:
    """``.train()`` updates ``nnx.BatchStat``; ``.eval()`` reads it and leaves it alone."""
    module, inputs = _built(case)
    initial = _batch_stats(module)
    module.train()
    case.call(module, *inputs)
    trained = _batch_stats(module)
    module.eval()
    case.call(module, *inputs)
    moved = any(not bool(jnp.array_equal(a, b)) for a, b in zip(initial, trained, strict=True))
    _require(moved, ".train() left the running statistics unchanged")
    kept = all(
        bool(jnp.array_equal(a, b)) for a, b in zip(trained, _batch_stats(module), strict=True)
    )
    _require(kept, ".eval() changed the running statistics")


def check_view_matches_modes(case: Case) -> None:
    """``nnx.view`` computes what ``.eval()`` and ``.train()`` do, from the other mode."""
    module, inputs = _built(case)
    for mode, start, is_eval in (("eval", "train", True), ("train", "eval", False)):
        getattr(module, start)()
        switched = nnx.clone(module)
        getattr(switched, mode)()
        with _refused("nnx.view"):
            viewed = nnx.view(
                nnx.clone(module),
                deterministic=is_eval,
                use_running_average=is_eval,
                raise_if_not_found=False,
            )
        expected, actual = case.call(switched, *inputs), case.call(viewed, *inputs)
        what = f"nnx.view(..., deterministic={is_eval}) against .{mode}()"
        pairs = zip(_arrays(actual), _arrays(expected), strict=True)
        _require(all(bool(jnp.array_equal(a, b)) for a, b in pairs), f"{what}: outputs differ")
        _require(
            _rng_counts(viewed) == _rng_counts(switched), f"{what}: RNG streams advanced apart"
        )


def check_jit_matches_eager(case: Case) -> None:
    """``nnx.jit`` computes what an eager call does, and traces once for one input shape."""
    module, inputs = _built(case)
    module.eval()
    expected = case.call(nnx.clone(module), *inputs)
    counter = TraceCounter()
    compiled = nnx.jit(counter.wrap(case.call))
    first = counter.count
    with _refused("nnx.jit"):
        actual = compiled(module, *inputs)
        traced = counter.count
        compiled(module, *inputs)
    _require(traced - first == 1, f"the first call traced {traced - first} times")
    _require(counter.count == traced, "a second call with the same shapes traced again")
    _require_same_function(actual, expected, "nnx.jit against eager")


def check_rows_independent(case: Case) -> None:
    """In ``.eval()``, ``nnx.vmap`` over single rows reproduces the batched call."""
    module, inputs = _built(case)
    module.eval()
    expected = case.call(module, *inputs)

    def one_row(module: nnx.Module, *row: jax.Array) -> PyTree:
        return jax.tree.map(lambda leaf: leaf[0], case.call(module, *(x[None] for x in row)))

    with _refused("nnx.vmap"):
        actual = nnx.vmap(one_row, in_axes=(None, *[0] * len(inputs)))(module, *inputs)
    _require_same_function(actual, expected, "rows mapped one at a time")


def check_gradient_finite(case: Case) -> None:
    """The float32 parameter gradient is finite and not zero everywhere."""
    module, inputs = _built(case)
    module.eval()
    graphdef, params, rest = nnx.split(module, nnx.Param, ...)

    def loss(params: NnxState) -> jax.Array:
        return _scalar(case.call(nnx.merge(graphdef, params, rest), *inputs))

    with _refused("jax.grad"):
        leaves = jax.tree.leaves(jax.grad(loss)(params))
    _require(bool(leaves), "the module has no nnx.Param")
    _require(all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves), "a gradient is not finite")
    _require(any(bool(jnp.any(leaf != 0)) for leaf in leaves), "the gradient is zero everywhere")


def check_gradient_matches_finite_differences(case: Case) -> None:
    """In float64 the parameter gradient agrees with finite differences."""
    module, inputs = _built(case)
    module.eval()

    def loss(module: nnx.Module) -> jax.Array:
        return _scalar(case.call(module, *as_float64(inputs)))

    try:
        with _refused("the float64 gradient check"):
            check_parameter_gradients(module, loss, modes=("rev",))
    except ConformanceError:
        raise
    except AssertionError as error:
        raise ConformanceError(f"float64 gradient against finite differences: {error}") from error


def _always(case: Case) -> bool:  # noqa: ARG001  # every case owes the check
    return True


CHECKS: Mapping[str, tuple[Callable[[Case], None], Callable[[Case], bool]]] = {
    "state_structure": (check_state_structure, _always),
    "state_holds_arrays": (check_state_holds_arrays, _always),
    "tree_mode_split": (check_tree_mode_split, _always),
    "eval_deterministic": (check_eval_deterministic, _always),
    "train_dropout_varies": (check_train_dropout_varies, lambda case: case.has_dropout),
    "batch_stats": (check_batch_stats, lambda case: case.has_batch_stats),
    "view_matches_modes": (check_view_matches_modes, _always),
    "jit_matches_eager": (check_jit_matches_eager, _always),
    "rows_independent": (check_rows_independent, lambda case: case.is_rowwise),
    "gradient_finite": (check_gradient_finite, lambda case: case.is_differentiable),
    "gradient_matches_finite_differences": (
        check_gradient_matches_finite_differences,
        lambda case: case.is_differentiable,
    ),
}
"""Every check by name, with the predicate saying whether a case owes it."""
