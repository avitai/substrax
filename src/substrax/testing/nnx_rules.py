"""Source rules for Flax NNX library code, checked on the syntax tree without importing it.

Each rule names a defect class measured in the stack's libraries, and finds its sites:

- ``seed-literal``: a seed written into library code, ``nnx.Rngs(0)``, ``jax.random.key(0)``,
  ``np.random.default_rng(7)``, ``np.random.seed(3)``, ``seed = 0`` or a ``seed=0`` default.
  Every module built without ``rngs`` then draws the same numbers, and a stochastic call reading
  a fixed key draws the same noise on every call. Whoever builds a module names its streams;
  ``substrax.rng.rngs_from_seed(seed, ...)`` takes the seed as an argument.
- ``rngs-probe``: ``"sample" in rngs``, ``hasattr(rngs, ...)``, ``getattr(rngs, ...)`` or
  ``rngs.get(...)``. ``nnx.Rngs(0)`` holds only a ``default`` stream, so ``"sample" in rngs`` is
  ``False`` although ``rngs.sample()`` draws from it; the draw itself, or
  ``substrax.rng.key_from``, is the test.
- ``rngs-fallback``: ``rngs or nnx.Rngs(...)``, ``... if rngs is not None else nnx.Rngs(...)``,
  or ``if rngs is None: rngs = nnx.Rngs(...)``. A missing ``rngs`` is the caller's error.
- ``trainable-metadata``: ``nnx.Param(..., trainable=...)``. Flax reads no such metadata, so a
  table marked ``trainable=False`` still moves under ``nnx.Optimizer(..., wrt=nnx.Param)``; a
  table that must not train is not an ``nnx.Param``.
- ``salted-hash``: the builtin ``hash``, whose value for a ``str`` changes between interpreters
  (``PYTHONHASHSEED``), so a token id or seed derived from it differs on every run. A call
  inside a ``__hash__`` method implements Python's hash protocol and is not a finding.
- ``mode-flag-default``: a parameter named ``deterministic``, ``training``, ``train``,
  ``is_training`` or ``use_running_average`` defaulting to a bool. Forwarded to an NNX layer, a
  call-time flag overrides the mode ``.train()`` / ``.eval()`` set; flax's own layers default it
  to ``None``, meaning that mode.
- ``noop-super-init``: ``super().__init__()`` in a direct ``nnx.Module`` subclass, which reaches
  ``object.__init__``: ``nnx.Module`` defines no ``__init__``.

The facts the NNX rules rest on are measured on the installed flax by
:mod:`substrax.testing.nnx_premises`.

A site the rule cannot tell from a legitimate use, such as an in-process cache key built with
``hash`` or a fixed key that fills a shape-only input, is exempted on its own line by a trailing
comment naming the rule and the reason::

    key = jax.random.key(0)  # substrax-allow: seed-literal: shape-only dummy input

An exemption naming no known rule, or no reason, raises :class:`MalformedExemptionError`.
A corpus directory that does not exist raises :class:`FileNotFoundError`. Importing this module
imports no jax.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from substrax.testing.source_scans import python_files


@dataclass(frozen=True, slots=True)
class Site:
    """One site a rule found in a syntax tree.

    Attributes:
        line: The node's first line.
        end_line: The node's last line; an exemption on any line from ``line`` to here applies,
            so a formatter that wraps the node and moves its trailing comment keeps it exempt.
        source: ``ast.unparse`` of the node, or a description of it.
    """

    line: int
    end_line: int
    source: str


def _site(node: ast.AST, source: str | None = None) -> Site:
    line = getattr(node, "lineno", 0)
    return Site(line, getattr(node, "end_lineno", None) or line, source or ast.unparse(node))


_EXEMPTION = re.compile(r"#\s*substrax-allow\b(?P<body>.*)$")
_EXEMPTION_BODY = re.compile(r":\s*(?P<rule>[a-z-]+)\s*:(?P<reason>.*)$")
_RNGS = "Rngs"
_SEED_CALLS = frozenset(
    {
        "key",
        "PRNGKey",
        "default_rng",
        "RandomState",
        "Philox",
        "PCG64",
        "SFC64",
        "MT19937",
        "SeedSequence",
        "seed",
    }
)
_MODE_FLAGS = frozenset(
    {"deterministic", "training", "train", "is_training", "use_running_average"}
)


class MalformedExemptionError(ValueError):
    """An exemption comment names no known rule or gives no reason."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Finding:
    """One site a rule found.

    Attributes:
        path: The file, relative to the scanned root, in POSIX form.
        line: The line of the offending node.
        rule: The rule's name, a key of :data:`RULES`.
        source: ``ast.unparse`` of the offending node, or a description of it.
    """

    path: str
    line: int
    rule: str
    source: str


def _callee(call: ast.Call) -> str:
    function = call.func
    if isinstance(function, ast.Attribute):
        return function.attr
    return function.id if isinstance(function, ast.Name) else ""


def _calls(tree: ast.AST) -> Iterator[ast.Call]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            yield node


def _sites(tree: ast.AST, matches: Callable[[ast.AST], bool]) -> list[Site]:
    return [_site(node) for node in ast.walk(tree) if matches(node)]


def _is_int_literal(node: ast.expr | None) -> bool:
    return isinstance(node, ast.Constant) and type(node.value) is int


def _names_seed(target: ast.expr) -> bool:
    if isinstance(target, ast.Name):
        return target.id == "seed"
    return isinstance(target, ast.Attribute) and target.attr == "seed"


def _names_rngs(node: ast.AST) -> bool:
    return ast.unparse(node).endswith("rngs")


def _is_rngs_call(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Call) and _callee(node) == _RNGS


def _functions(tree: ast.AST) -> Iterator[ast.FunctionDef | ast.AsyncFunctionDef]:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def _defaults(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> Iterator[tuple[ast.arg, ast.expr]]:
    arguments = function.args
    positional = [*arguments.posonlyargs, *arguments.args]
    with_defaults = positional[len(positional) - len(arguments.defaults) :]
    yield from zip(with_defaults, arguments.defaults, strict=True)
    for argument, default in zip(arguments.kwonlyargs, arguments.kw_defaults, strict=True):
        if default is not None:
            yield argument, default


def _seeds_a_generator(node: ast.AST) -> bool:
    """A call seeding a generator or a stream with an integer literal."""
    if not isinstance(node, ast.Call):
        return False
    callee = _callee(node)
    if callee == _RNGS:
        return any(_is_int_literal(a) for a in [*node.args, *(k.value for k in node.keywords)])
    if callee in _SEED_CALLS and node.args and _is_int_literal(node.args[0]):
        return True
    return any(k.arg == "seed" and _is_int_literal(k.value) for k in node.keywords)


def _assigns_a_seed(node: ast.AST) -> bool:
    """``seed = <int>`` or ``seed: int = <int>``, to a name or an attribute."""
    if isinstance(node, ast.Assign):
        return _is_int_literal(node.value) and any(_names_seed(t) for t in node.targets)
    return (
        isinstance(node, ast.AnnAssign) and _is_int_literal(node.value) and _names_seed(node.target)
    )


def seed_literal(tree: ast.AST) -> list[Site]:
    """Seeds written into the code: seeding calls, ``seed=`` keywords, defaults and assignments."""
    defaults = [
        _site(argument, f"{function.name}({argument.arg}={ast.unparse(default)})")
        for function in _functions(tree)
        for argument, default in _defaults(function)
        if argument.arg == "seed" and _is_int_literal(default)
    ]
    return [*_sites(tree, lambda n: _seeds_a_generator(n) or _assigns_a_seed(n)), *defaults]


def _is_membership_probe(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Compare)
        and any(isinstance(op, (ast.In, ast.NotIn)) for op in node.ops)
        and any(_names_rngs(comparator) for comparator in node.comparators)
    )


def _is_attribute_probe(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    function = node.func
    if isinstance(function, ast.Name) and function.id in {"hasattr", "getattr"}:
        return bool(node.args) and _names_rngs(node.args[0])
    return (
        isinstance(function, ast.Attribute)
        and function.attr == "get"
        and _names_rngs(function.value)
    )


def rngs_probe(tree: ast.AST) -> list[Site]:
    """Membership, ``hasattr``, ``getattr`` and ``.get`` probes of an ``nnx.Rngs``."""
    return _sites(tree, lambda node: _is_membership_probe(node) or _is_attribute_probe(node))


def _substitutes_rngs(node: ast.AST) -> bool:
    """``rngs or nnx.Rngs(...)`` and ``... if ... else nnx.Rngs(...)``."""
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
        return any(_is_rngs_call(value) for value in node.values[1:])
    return isinstance(node, ast.IfExp) and _is_rngs_call(node.orelse)


def _tests_rngs_is_none(test: ast.expr) -> bool:
    return (
        isinstance(test, ast.Compare)
        and _names_rngs(test.left)
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Is)
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value is None
    )


def rngs_fallback(tree: ast.AST) -> list[Site]:
    """A fresh ``nnx.Rngs`` put in place of a missing one."""
    assigned = [
        _site(statement)
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and _tests_rngs_is_none(node.test)
        for statement in node.body
        if isinstance(statement, ast.Assign) and _is_rngs_call(statement.value)
    ]
    return [*_sites(tree, _substitutes_rngs), *assigned]


def trainable_metadata(tree: ast.AST) -> list[Site]:
    """``nnx.Param(..., trainable=...)``, metadata flax never reads."""
    return [
        _site(call)
        for call in _calls(tree)
        if _callee(call) == "Param" and any(k.arg == "trainable" for k in call.keywords)
    ]


def salted_hash(tree: ast.AST) -> list[Site]:
    """Calls of the builtin ``hash`` outside a ``__hash__`` method."""
    protocol = {
        id(call)
        for function in _functions(tree)
        if function.name == "__hash__"
        for call in _calls(function)
    }
    return [
        _site(call)
        for call in _calls(tree)
        if isinstance(call.func, ast.Name) and call.func.id == "hash" and id(call) not in protocol
    ]


def _is_bool_mode_default(argument: ast.arg, default: ast.expr) -> bool:
    return (
        argument.arg in _MODE_FLAGS
        and isinstance(default, ast.Constant)
        and isinstance(default.value, bool)
    )


def mode_flag_default(tree: ast.AST) -> list[Site]:
    """Mode-flag parameters defaulting to a bool rather than ``None``."""
    return [
        _site(argument, f"{function.name}({argument.arg}={ast.unparse(default)})")
        for function in _functions(tree)
        for argument, default in _defaults(function)
        if _is_bool_mode_default(argument, default)
    ]


def _direct_module_inits(tree: ast.AST) -> Iterator[tuple[ast.ClassDef, ast.FunctionDef]]:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and [ast.unparse(b) for b in node.bases] == [
            "nnx.Module"
        ]:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    yield node, item


def noop_super_init(tree: ast.AST) -> list[Site]:
    """``super().__init__()`` in a class whose only base is ``nnx.Module``."""
    return [
        _site(call, f"{cls.name}.__init__: super().__init__()")
        for cls, init in _direct_module_inits(tree)
        for call in _calls(init)
        if ast.unparse(call) == "super().__init__()"
    ]


RULES: Mapping[str, Callable[[ast.AST], list[Site]]] = {
    "seed-literal": seed_literal,
    "rngs-probe": rngs_probe,
    "rngs-fallback": rngs_fallback,
    "trainable-metadata": trainable_metadata,
    "salted-hash": salted_hash,
    "mode-flag-default": mode_flag_default,
    "noop-super-init": noop_super_init,
}
"""Every rule, by name."""


def exemptions(text: str) -> dict[int, tuple[str, str]]:
    """Map each line carrying a ``substrax-allow`` comment to the rule it exempts and the reason.

    Args:
        text: A Python source file's text.

    Returns:
        ``{line: (rule, reason)}``.

    Raises:
        MalformedExemptionError: If an exemption names no known rule or gives no reason.
    """
    found: dict[int, tuple[str, str]] = {}
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type != tokenize.COMMENT or not (marker := _EXEMPTION.search(token.string)):
            continue
        body = _EXEMPTION_BODY.match(marker.group("body"))
        reason = body.group("reason").strip() if body else ""
        if body is None or body.group("rule") not in RULES or not reason:
            raise MalformedExemptionError(
                f"line {token.start[0]}: an exemption reads '# substrax-allow: <rule>: <reason>' "
                f"with one of {sorted(RULES)}, got {token.string!r}"
            )
        found[token.start[0]] = (body.group("rule"), reason)
    return found


def _is_exempted(site: Site, rule: str, exempted: Mapping[int, tuple[str, str]]) -> bool:
    lines = range(site.line, site.end_line + 1)
    return any(exempted.get(line, ("", ""))[0] == rule for line in lines)


def findings(  # noqa: DOC503  # MalformedExemptionError is raised by exemptions
    root: Path,
    directories: Sequence[str],
    *,
    rules: Collection[str] | None = None,
    excluded_parts: Collection[str] = (),
) -> list[Finding]:
    """Every unexempted site of the chosen rules in the Python files under ``root / directory``.

    Args:
        root: The repository root; finding paths are relative to it.
        directories: Directories relative to ``root`` to search recursively.
        rules: Names of the rules to apply; every rule when ``None``.
        excluded_parts: Directory names whose files are skipped.

    Returns:
        The findings, ordered by path, line and rule.

    Raises:
        KeyError: If ``rules`` names an unknown rule.
        FileNotFoundError: If a directory does not exist.
        MalformedExemptionError: If an exemption comment is malformed.
    """
    chosen = list(RULES) if rules is None else list(rules)
    unknown = sorted(set(chosen) - set(RULES))
    if unknown:
        raise KeyError(f"unknown rules {unknown}; the rules are {sorted(RULES)}")
    found: list[Finding] = []
    for path in python_files(root, directories, excluded_parts=excluded_parts):
        text = path.read_text(encoding="utf-8")
        tree, exempted = ast.parse(text), exemptions(text)
        relative = path.relative_to(root).as_posix()
        for name in chosen:
            found += [
                Finding(path=relative, line=site.line, rule=name, source=site.source)
                for site in RULES[name](tree)
                if not _is_exempted(site, name, exempted)
            ]
    return sorted(found, key=lambda finding: (finding.path, finding.line, finding.rule))
