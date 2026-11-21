"""The framework facts the NNX source rules and conformance checks rest on, measured live.

Each :class:`Premise` measures one behaviour of the installed flax, jax or Python as a bool and
names the rules of :mod:`substrax.testing.nnx_rules`, or the checks, that depend on it.
:func:`check_premises` raises :class:`PremiseChangedError` when a measurement no longer matches,
naming the premise and what rests on it, so a release that changes one is found where it is
installed, before code written against the old behaviour misbehaves. A repository runs it in the
job that installs the newest jax and flax its bounds admit.

The tree-mode premises follow flax FLIP 5310 (``docs_nnx/flip/5310-tree-mode-nnx.md``), whose
goal is tree mode (``graph=False``) by default; ``graph-mode-is-default`` turning false is the
signal that it happened.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import flax
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from flax.configurations import config as flax_config

from substrax.testing.child_process import run_python


_STR_HASH = "from substrax.testing import report; report(hash('artifex'))"
_SEED = 0
_SALTS = ("1", "2")


@dataclass(frozen=True, slots=True, kw_only=True)
class Premise:
    """One measured framework fact.

    Attributes:
        name: The premise's name.
        why: What the fact means for code written against it.
        relied_on_by: Rule names of :data:`substrax.testing.nnx_rules.RULES`, or ``"tree-mode"``
            and ``"conformance"`` for the checks of :mod:`substrax.testing.nnx_conformance`.
        measure: Returns whether the fact holds on what is installed.
    """

    name: str
    why: str
    relied_on_by: tuple[str, ...]
    measure: Callable[[], bool]


class PremiseChangedError(AssertionError):
    """An installed framework no longer behaves as a rule or check assumes."""


def _membership_misreads_the_default_stream() -> bool:
    rngs = nnx.Rngs(_SEED)
    absent = "sample" not in rngs  # substrax-allow: rngs-probe: the premise measures the probe
    return absent and rngs.sample().shape == ()


def _trainable_metadata_is_ignored() -> bool:
    class Table(nnx.Module):
        def __init__(self) -> None:
            self.table = nnx.Param(
                jnp.ones(3), trainable=False
            )  # substrax-allow: trainable-metadata: the premise measures it

        def __call__(self) -> jax.Array:
            return jnp.sum(self.table[...])

    module = Table()
    optimizer = nnx.Optimizer(module, optax.sgd(1.0), wrt=nnx.Param)
    optimizer.update(module, nnx.grad(lambda m: m())(module))
    return bool(jnp.any(module.table[...] != 1.0))


def _call_time_flag_overrides_eval() -> bool:
    dropout = nnx.Dropout(0.5, rngs=nnx.Rngs(dropout=_SEED))
    dropout.eval()
    return bool(jnp.any(dropout(jnp.ones(64), deterministic=False) == 0.0))


_MODE_NAMES = ("deterministic", "use_running_average", "training", "is_training", "_deterministic")


class _ModeNames(nnx.Module):
    """Holds each candidate name for a mode, all ``False``."""

    def __init__(self) -> None:
        for name in _MODE_NAMES:
            setattr(self, name, False)


def _eval_sets_only_flax_mode_names() -> bool:
    module = _ModeNames()
    module.eval()
    switched = {name for name in _MODE_NAMES if getattr(module, name)}
    return switched == {"deterministic", "use_running_average"}


class _OwnMode(nnx.Module):
    """A leaf owning ``deterministic``, without ``set_view``."""

    def __init__(self) -> None:
        self.deterministic = False


class _OwnModeWithView(_OwnMode):
    """The same leaf with ``set_view``."""

    def set_view(self, deterministic: bool | None = None) -> None:
        if deterministic is not None:
            self.deterministic = deterministic


def _viewed_deterministic(module: _OwnMode) -> bool:
    return nnx.view(module, deterministic=True, raise_if_not_found=False).deterministic


def _view_reaches_only_set_view() -> bool:
    return not _viewed_deterministic(_OwnMode()) and _viewed_deterministic(_OwnModeWithView())


def _module_defines_no_init() -> bool:
    return "__init__" not in vars(nnx.Module)


def _tree_mode_rejects_aliases() -> bool:
    shared = nnx.Rngs(_SEED)

    class Child(nnx.Module):
        def __init__(self) -> None:
            self.rngs = shared

    class Parent(nnx.Module):
        def __init__(self) -> None:
            self.rngs = shared
            self.child = Child()

    try:
        nnx.split(Parent(), graph=False)
    except ValueError:
        return True
    return False


def _tree_mode_refuses_diff_state() -> bool:
    linear = nnx.Linear(2, 2, rngs=nnx.Rngs(_SEED))
    try:
        nnx.grad(
            lambda m: jnp.sum(m(jnp.ones((1, 2)))),
            argnums=nnx.DiffState(0, nnx.Param),
            graph=False,
        )(linear)
    except ValueError:
        return True
    return False


def _graph_mode_is_default() -> bool:
    return bool(flax_config.nnx_graph_mode)


def _str_hash_is_salted() -> bool:
    hashes = {
        run_python(_STR_HASH, env={"PYTHONHASHSEED": seed}, timeout=60).check().report_as(int)
        for seed in _SALTS
    }
    return len(hashes) == len(_SALTS)


PREMISES: tuple[Premise, ...] = (
    Premise(
        name="membership-misreads-the-default-stream",
        why="'sample' in nnx.Rngs(0) is False although rngs.sample() draws from default",
        relied_on_by=("rngs-probe",),
        measure=_membership_misreads_the_default_stream,
    ),
    Premise(
        name="trainable-metadata-is-ignored",
        why="an nnx.Param marked trainable=False still moves under Optimizer(wrt=nnx.Param)",
        relied_on_by=("trainable-metadata",),
        measure=_trainable_metadata_is_ignored,
    ),
    Premise(
        name="call-time-flag-overrides-eval",
        why="a deterministic=False passed at call time still drops units after .eval()",
        relied_on_by=("mode-override", "conformance"),
        measure=_call_time_flag_overrides_eval,
    ),
    Premise(
        name="eval-sets-only-flax-mode-names",
        why=".eval() sets deterministic and use_running_average, not training or a private name",
        relied_on_by=("unreachable-mode",),
        measure=_eval_sets_only_flax_mode_names,
    ),
    Premise(
        name="view-reaches-only-set-view",
        why="nnx.view changes a module's mode through set_view alone, unlike .eval()",
        relied_on_by=("conformance",),
        measure=_view_reaches_only_set_view,
    ),
    Premise(
        name="module-defines-no-init",
        why="nnx.Module defines no __init__, so super().__init__() in it reaches object",
        relied_on_by=("noop-super-init",),
        measure=_module_defines_no_init,
    ),
    Premise(
        name="tree-mode-rejects-aliases",
        why="nnx.split(graph=False) refuses a Variable two modules share, such as one Rngs",
        relied_on_by=("tree-mode", "conformance"),
        measure=_tree_mode_rejects_aliases,
    ),
    Premise(
        name="tree-mode-refuses-diff-state",
        why="nnx.grad with a DiffState prefix raises in tree mode; split into Param and rest",
        relied_on_by=("tree-mode", "conformance"),
        measure=_tree_mode_refuses_diff_state,
    ),
    Premise(
        name="graph-mode-is-default",
        why="graph mode is still the default; when false, tree mode (FLIP 5310) has arrived",
        relied_on_by=("tree-mode",),
        measure=_graph_mode_is_default,
    ),
    Premise(
        name="str-hash-is-salted",
        why="hash of a str differs between interpreters with different PYTHONHASHSEED values",
        relied_on_by=("salted-hash",),
        measure=_str_hash_is_salted,
    ),
)
"""Every premise, in the order :func:`check_premises` measures them."""


def measure_premises() -> dict[str, bool]:
    """Measure every premise on what is installed.

    Returns:
        ``{premise name: whether it holds}``.
    """
    return {premise.name: premise.measure() for premise in PREMISES}


def check_premises() -> None:
    """Raise if any premise no longer holds on the installed frameworks.

    Raises:
        PremiseChangedError: Naming each changed premise and the rules and checks resting on it.
    """
    changed = [premise for premise in PREMISES if not premise.measure()]
    if changed:
        details = "; ".join(
            f"{premise.name} ({premise.why}) no longer holds: "
            f"revisit {', '.join(premise.relied_on_by)}"
            for premise in changed
        )
        raise PremiseChangedError(f"flax {flax.__version__}, jax {jax.__version__}: {details}")
