"""Properties every ``nnx.Module`` owes under NNX's transforms, checked on a repository's cases.

A repository describes how to build and call each of its module classes as a :class:`Case`,
and runs every check of :data:`CHECKS` whose claim the case makes. :func:`module_classes` gives
the denominator: every concrete ``nnx.Module`` a package defines, so a class without a case is
found rather than silently skipped.

A module either holds its randomness (a forked stream, as ``nnx.Dropout(rngs=...)`` keeps) or
takes it at call time as ``rngs=``, the style flax recommends for new code
(``docs_nnx/guides/randomness.md``). A case of the second kind names its streams in
:attr:`Case.rng_streams`; every check then passes an ``nnx.Rngs`` of those streams built from a
key the check chooses, the same key to two computations it compares.

The checks:

Every check requires the case's call to return a pytree of arrays, the contract of
:attr:`Case.call`: a string, a host ``float`` or another object among the outputs cannot pass
through ``nnx.jit`` or ``nnx.vmap``, so each check reports it rather than failing on it.

- ``state_structure``: a forward pass leaves the state's paths, types, shapes and dtypes, and
  the graph definition, as ``__init__`` built them. A parameter created on the first call is
  missing from an ``nnx.Optimizer`` built before it. A Variable holding a Python value (which
  ``state_holds_arrays`` reports) is described by the value's type, so a forward pass that
  replaces a ``None`` with an ``int`` changes the structure.
- ``state_holds_arrays``: every Variable holds an array; a Python value in a Variable is traced,
  comes back from ``nnx.jit`` as an array and cannot be differentiated in tree mode.
- ``tree_mode_split``: ``nnx.split(graph=False)`` succeeds and the merged module computes what
  the module does. Tree mode, flax FLIP 5310's planned default, refuses shared Variables.
- ``eval_deterministic``: after ``.eval()`` two calls agree exactly and no RNG stream advances;
  with call-time streams the two calls get different keys, so the output must not depend on one.
  A case whose call draws (:attr:`Case.draws`) does not owe it: sampling is that call's job in
  every mode.
- ``output_follows_key``: for a case whose call draws, in ``.eval()`` and in ``.train()``
  different keys give different outputs and the same key the same output, the same-key run
  starting with every PRNG key the module holds changed. A draw from a stream or a key the
  module holds, which the caller's key does not control, is found, and so is a call that
  returns no sample in one of the modes.
- ``train_dropout_varies``: after ``.train()`` a module with dropout draws a fresh mask per call;
  with call-time streams, different keys give different outputs and the same key the same
  output. The same-key run starts with every PRNG key the module holds changed, so randomness
  the key does not control is found: a dropout leaf the key was not passed to, or noise from a
  stream or key the module holds.
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
key, a wrong mode) is of the order of ``scale``. Integer and boolean outputs (token ids, class
labels, masks) are compared exactly. Checks comparing two runs start both from cloned
state, RNG counts included, so a module that draws randomness in the wrong mode fails
``eval_deterministic`` alone rather than every check. A failed check raises
:class:`ConformanceError`, and so does a transform refusing the module: flax's errors (a Variable
mutated from another trace level, as dropout drawn or statistics written in ``.eval()`` are under
a gradient), JAX's tracer errors (Python control flow on a traced value), the ``ValueError`` of
a split or a mapping flax cannot do, and the ``OverflowError`` JAX raises when a transform turns
a Variable's Python ``int`` that does not fit the integer dtype into an array
(``jax/_src/abstract_arrays.py``, ``jax/_src/dtypes.py``). The refusal is chained as the error's
cause.
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
import numpy as np
from flax import nnx
from flax.errors import FlaxError
from flax.typing import Key

from substrax.nnx_typing import NnxState
from substrax.rng import is_legacy_key, rngs_from_seed, STANDARD_STREAMS
from substrax.testing.gradients import as_float64, check_parameter_gradients
from substrax.testing.traces import TraceCounter
from substrax.typing import PyTree


DEPTH = 1e4
"""A bound on the roundings any output of a small test module passes through."""

STREAMS = STANDARD_STREAMS
"""The streams a case's module is built with: :data:`substrax.rng.STANDARD_STREAMS`."""

_SEED = 0


class ConformanceError(AssertionError):
    """A module does not have a property a conformance check requires of it."""


_REFUSALS = (FlaxError, jax.errors.JAXTypeError, ValueError, OverflowError)
"""What flax and JAX raise when a transform refuses a module (see the module docstring)."""


@contextmanager
def _refused(transform: str) -> Generator[None]:
    """Raise a transform's refusal of the module as a :class:`ConformanceError`."""
    try:
        yield
    except _REFUSALS as error:
        message = f"{transform} refused the module: {type(error).__name__}: {error}"
        raise ConformanceError(message) from error


def _call(module: nnx.Module, *inputs: jax.Array, **keywords: nnx.Rngs) -> PyTree:
    return module(*inputs, **keywords)


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
        rng_streams: The streams the module takes at call time, as ``rngs=``; empty when it
            holds its randomness. When set, every check calls ``call(module, *inputs, rngs=...)``
            with an ``nnx.Rngs`` of these streams built from a key the check chooses: compared
            runs get the same key, and the dropout check also compares two keys.
        draws: The call draws samples (a distribution's ``sample``, a model's ``generate``), so
            its output depends on the key in every mode. Such a case names its ``rng_streams``,
            owes ``output_follows_key`` in place of ``eval_deterministic``, and does not claim
            ``is_rowwise``: one key drawn for a batch and the same key drawn for each row give
            different samples.

    Raises:
        ValueError: If ``draws`` is set without ``rng_streams``, or together with
            ``is_rowwise``.
    """

    build: Callable[[nnx.Rngs], nnx.Module]
    inputs: Callable[[jax.Array], tuple[jax.Array, ...]]
    call: Callable[..., PyTree] = _call
    has_dropout: bool = False
    has_batch_stats: bool = False
    is_rowwise: bool = True
    is_differentiable: bool = True
    rng_streams: tuple[str, ...] = ()
    draws: bool = False

    def __post_init__(self) -> None:
        """Refuse a drawing case that could not be checked as one."""
        if self.draws and not self.rng_streams:
            raise ValueError(
                "a case that draws names its rng_streams: a module that draws takes its key at"
                " call time, and the checks choose that key"
            )
        if self.draws and self.is_rowwise:
            raise ValueError(
                "a case that draws does not claim is_rowwise: one key drawn for a batch and the"
                " same key drawn for each row give different samples"
            )


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


def _inexact(tree: PyTree) -> list[jax.Array]:
    """The floating and complex leaves: the outputs a gradient can be taken of."""
    return [leaf for leaf in jax.tree.leaves(tree) if jnp.issubdtype(leaf.dtype, jnp.inexact)]


def _value(leaf: PyTree) -> PyTree:
    return leaf.get_value() if isinstance(leaf, nnx.Variable) else leaf


def _path(path: tuple[Key, ...]) -> str:
    return "/".join(map(str, path))


def _require_same_function(actual: PyTree, expected: PyTree, what: str) -> None:
    """Require one structure, floating outputs within the rounding bound, the rest exactly."""
    actual_structure, expected_structure = jax.tree.structure(actual), jax.tree.structure(expected)
    _require(
        actual_structure == expected_structure,
        f"{what}: the outputs differ in structure, {actual_structure} against {expected_structure}",
    )
    pairs = list(zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True))
    for a, b in pairs:
        if not jnp.issubdtype(b.dtype, jnp.inexact):
            _require(bool(jnp.array_equal(a, b)), f"{what}: integer outputs differ")
            continue
        scale = max(1.0, float(jnp.max(jnp.abs(b))))
        bound = DEPTH * float(jnp.finfo(b.dtype).eps) * scale
        error = float(jnp.max(jnp.abs(a - b))) if a.shape == b.shape else float("inf")
        _require(error <= bound, f"{what}: max |difference| {error:.3g} exceeds {bound:.3g}")


def _describe(value: PyTree) -> tuple[tuple[int, ...], str]:
    """An array's shape and dtype; a Python value, which has neither, by its type."""
    if isinstance(value, jax.Array | np.ndarray):
        return value.shape, str(value.dtype)
    return (), f"python {type(value).__name__}"


def _structure(module: nnx.Module) -> list[tuple[str, str, tuple[int, ...], str]]:
    return [
        (_path(path), type(leaf).__name__, *_describe(_value(leaf)))
        for path, leaf in nnx.to_flat_state(nnx.state(module))
    ]


def _rng_counts(module: nnx.Module) -> list[int]:
    return [int(count[...]) for _, count in nnx.to_flat_state(nnx.state(module, nnx.RngCount))]


def _batch_stats(module: nnx.Module) -> list[jax.Array]:
    return [leaf[...] for _, leaf in nnx.to_flat_state(nnx.state(module, nnx.BatchStat))]


def _scalar(output: PyTree) -> jax.Array:
    return sum((jnp.sum(jnp.sin(leaf)) for leaf in _inexact(output)), start=jnp.zeros(()))


def _call_keys() -> tuple[jax.Array, jax.Array]:
    """Two distinct keys the checks pass to a case's call-time streams."""
    rngs = rngs_from_seed(_SEED, streams=("calls",))
    return rngs.calls(), rngs.calls()


def _fold_once(keys: jax.Array) -> jax.Array:
    """Fold one more value into each key of a key array; ``fold_in`` takes one key at a time."""
    folded = jax.vmap(lambda key: jax.random.fold_in(key, 1))(keys.reshape(-1))
    return folded.reshape(keys.shape)


def _perturbed(module: nnx.Module) -> nnx.Module:
    """The module with every PRNG key it holds changed, so a draw from held state changes.

    That covers an ``nnx.Rngs`` stream, whose key is an ``nnx.RngKey``, a typed key in any
    Variable, and a legacy key, the ``uint32`` data ``jax.random.PRNGKey`` returns
    (:func:`substrax.rng.is_legacy_key`, whose shape follows the default PRNG implementation).
    Other ``uint32`` state of that shape would be changed too, which a module compared under one
    key would then show as a difference; nothing but a key has that dtype and shape in practice.
    """
    for _, node in nnx.iter_graph(module):
        if not isinstance(node, nnx.Variable):
            continue
        value = node.get_value()
        if isinstance(value, jax.Array) and jnp.issubdtype(value.dtype, jax.dtypes.prng_key):
            node.set_value(_fold_once(value))
        elif is_legacy_key(value):
            folded = _fold_once(jax.random.wrap_key_data(value))
            node.set_value(jax.random.key_data(folded))
    return module


def _run(case: Case, module: nnx.Module, inputs: tuple[jax.Array, ...], key: jax.Array) -> PyTree:
    """Call the case, giving its call-time streams an ``nnx.Rngs`` built from ``key``.

    The ``nnx.Rngs`` is built where the call happens, inside any transform, so no ``nnx.Rngs``
    crosses a transform boundary; the same key gives the same draws.
    """
    if case.rng_streams:
        streams = {name: jax.random.fold_in(key, i) for i, name in enumerate(case.rng_streams)}
        output = case.call(module, *inputs, rngs=nnx.Rngs(**streams))
    else:
        output = case.call(module, *inputs)
    _require_arrays(output)
    return output


def _where(path: tuple[Key, ...]) -> str:
    """A pytree path as text, naming the top level when the path is empty."""
    return jax.tree_util.keystr(path) or "the top level"


def _require_arrays(output: PyTree) -> None:
    """A case's call returns a pytree of arrays; a string, a host float or any other leaf fails.

    Inside ``jax.jit``, ``jax.vmap`` and ``jax.grad`` the leaves are tracers, which are
    ``jax.Array`` instances, so the requirement holds under every transform.
    """
    leaves = jax.tree_util.tree_leaves_with_path(output)
    _require(bool(leaves), "the call returned no arrays, so the check has no output to compare")
    for path, leaf in leaves:
        _require(
            isinstance(leaf, jax.Array),
            f"the call returned a {type(leaf).__name__}, not an array, at {_where(path)}",
        )


def _equal(first: PyTree, second: PyTree) -> bool:
    if jax.tree.structure(first) != jax.tree.structure(second):
        return False
    pairs = zip(jax.tree.leaves(first), jax.tree.leaves(second), strict=True)
    return all(bool(jnp.array_equal(a, b)) for a, b in pairs)


def check_state_structure(case: Case) -> None:
    """A forward pass leaves the state and graph definition as ``__init__`` built them."""
    module, inputs = _built(case)
    module.train()
    before, graph = _structure(module), nnx.graphdef(module)
    _run(case, module, inputs, _call_keys()[0])
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
    key = _call_keys()[0]
    expected = _run(case, nnx.clone(module), inputs, key)
    with _refused("tree mode"):
        graphdef, state = nnx.split(module, graph=False)
        actual = _run(case, nnx.merge(graphdef, state), inputs, key)
    _require_same_function(actual, expected, "tree mode")


def check_eval_deterministic(case: Case) -> None:
    """After ``.eval()`` two calls agree exactly and no RNG stream advances.

    With call-time streams the two calls get different keys, so the output must not depend on
    the key.
    """
    module, inputs = _built(case)
    module.eval()
    counts = _rng_counts(module)
    key_a, key_b = _call_keys()
    first, second = _run(case, module, inputs, key_a), _run(case, module, inputs, key_b)
    what = (
        "two .eval() calls with different keys differ"
        if case.rng_streams
        else "two .eval() calls differ"
    )
    _require(_equal(first, second), what)
    _require(_rng_counts(module) == counts, ".eval() advanced an RNG stream")


def check_train_dropout_varies(case: Case) -> None:
    """After ``.train()`` a module with dropout draws a fresh mask per call.

    With call-time streams the mask follows the key: different keys give different outputs, and
    the same key gives the same output. The same-key run starts with every PRNG key the module
    holds changed, so a draw from one of them, which the key does not control, changes its
    output; from cloned state alone it would repeat.
    """
    module, inputs = _built(case)
    module.train()
    if not case.rng_streams:
        key = _call_keys()[0]
        first, second = _run(case, module, inputs, key), _run(case, module, inputs, key)
        _require(not _equal(first, second), "two .train() calls agree")
        return
    _require_output_follows_key(case, module, inputs, "train")


def _require_output_follows_key(
    case: Case, module: nnx.Module, inputs: tuple[jax.Array, ...], mode: str
) -> None:
    """Different keys give different outputs, and the same key the same output.

    Each run starts from cloned state, and the same-key run from state whose every held PRNG key
    is changed, so a draw the caller's key does not control changes its output.
    """
    key_a, key_b = _call_keys()
    first = _run(case, nnx.clone(module), inputs, key_a)
    other_key = _run(case, nnx.clone(module), inputs, key_b)
    same_key = _run(case, _perturbed(nnx.clone(module)), inputs, key_a)
    _require(not _equal(first, other_key), f"two .{mode}() calls with different keys agree")
    _require(
        _equal(first, same_key),
        f"two .{mode}() calls with the same key differ: the module draws randomness the key does"
        " not control, from a stream it holds or outside any key",
    )


def check_output_follows_key(case: Case) -> None:
    """A call that draws follows the caller's key and nothing else, in both modes."""
    for mode in ("eval", "train"):
        module, inputs = _built(case)
        getattr(module, mode)()
        _require_output_follows_key(case, module, inputs, mode)


def check_batch_stats(case: Case) -> None:
    """``.train()`` updates ``nnx.BatchStat``; ``.eval()`` reads it and leaves it alone."""
    module, inputs = _built(case)
    key = _call_keys()[0]
    initial = _batch_stats(module)
    module.train()
    _run(case, module, inputs, key)
    trained = _batch_stats(module)
    module.eval()
    _run(case, module, inputs, key)
    moved = any(not bool(jnp.array_equal(a, b)) for a, b in zip(initial, trained, strict=True))
    _require(moved, ".train() left the running statistics unchanged")
    kept = all(
        bool(jnp.array_equal(a, b)) for a, b in zip(trained, _batch_stats(module), strict=True)
    )
    _require(kept, ".eval() changed the running statistics")


def check_view_matches_modes(case: Case) -> None:
    """``nnx.view`` computes what ``.eval()`` and ``.train()`` do, from the other mode."""
    module, inputs = _built(case)
    key = _call_keys()[0]
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
        expected, actual = _run(case, switched, inputs, key), _run(case, viewed, inputs, key)
        what = f"nnx.view(..., deterministic={is_eval}) against .{mode}()"
        _require(_equal(actual, expected), f"{what}: outputs differ")
        _require(
            _rng_counts(viewed) == _rng_counts(switched), f"{what}: RNG streams advanced apart"
        )


def check_jit_matches_eager(case: Case) -> None:
    """``nnx.jit`` computes what an eager call does, and traces once for one input shape."""
    module, inputs = _built(case)
    module.eval()
    key = _call_keys()[0]

    def keyed(module: nnx.Module, key: jax.Array, *inputs: jax.Array) -> PyTree:
        return _run(case, module, inputs, key)

    expected = keyed(nnx.clone(module), key, *inputs)
    counter = TraceCounter()
    compiled = nnx.jit(counter.wrap(keyed))
    first = counter.count
    with _refused("nnx.jit"):
        actual = compiled(module, key, *inputs)
        traced = counter.count
        compiled(module, key, *inputs)
    _require(traced - first == 1, f"the first call traced {traced - first} times")
    _require(counter.count == traced, "a second call with the same shapes traced again")
    _require_same_function(actual, expected, "nnx.jit against eager")


def check_rows_independent(case: Case) -> None:
    """In ``.eval()``, ``nnx.vmap`` over single rows reproduces the batched call."""
    module, inputs = _built(case)
    module.eval()
    key = _call_keys()[0]
    expected = _run(case, module, inputs, key)
    rows = inputs[0].shape[0]
    for path, leaf in jax.tree_util.tree_leaves_with_path(expected):
        _require(
            leaf.ndim >= 1 and leaf.shape[0] == rows,
            f"the output at {_where(path)} has no leading batch"
            f" axis of {rows} (shape {leaf.shape}); a case whose call reduces over the batch"
            " does not claim is_rowwise",
        )

    def one_row(module: nnx.Module, *row: jax.Array) -> PyTree:
        output = _run(case, module, tuple(x[None] for x in row), key)
        return jax.tree.map(lambda leaf: leaf[0], output)

    with _refused("nnx.vmap"):
        actual = nnx.vmap(one_row, in_axes=(None, *[0] * len(inputs)))(module, *inputs)
    _require_same_function(actual, expected, "rows mapped one at a time")


def check_gradient_finite(case: Case) -> None:
    """The float32 parameter gradient is finite and not zero everywhere."""
    module, inputs = _built(case)
    module.eval()
    key = _call_keys()[0]
    graphdef, params, rest = nnx.split(module, nnx.Param, ...)

    def loss(params: NnxState) -> jax.Array:
        return _scalar(_run(case, nnx.merge(graphdef, params, rest), inputs, key))

    with _refused("jax.grad"):
        leaves = jax.tree.leaves(jax.grad(loss)(params))
    _require(bool(leaves), "the module has no nnx.Param")
    _require(all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves), "a gradient is not finite")
    _require(any(bool(jnp.any(leaf != 0)) for leaf in leaves), "the gradient is zero everywhere")


def check_gradient_matches_finite_differences(case: Case) -> None:
    """In float64 the parameter gradient agrees with finite differences."""
    module, inputs = _built(case)
    module.eval()
    key = _call_keys()[0]

    def loss(module: nnx.Module) -> jax.Array:
        return _scalar(_run(case, module, tuple(as_float64(inputs)), key))

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
    "eval_deterministic": (check_eval_deterministic, lambda case: not case.draws),
    "output_follows_key": (check_output_follows_key, lambda case: case.draws),
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
