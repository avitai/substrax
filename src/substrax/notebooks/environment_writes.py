"""Which variables a statement writes into the process environment, read from its syntax tree.

One recogniser for the example guide's ``no-env-writes`` rule and the source scans' import-time
check. It reads an assignment or ``del`` of an item, ``|=``, the mapping methods that change
``os.environ`` and ``os.putenv``/``os.unsetenv``. Which names stand for the environment is the
caller's: by default the ``os.``-qualified ones, and a caller adds the names its module imports
from ``os``.
"""

from __future__ import annotations

import ast
from collections.abc import Collection


OS_ENVIRON = ("os.environ",)
"""The environment mapping, as a module that imports ``os`` names it."""

OS_MUTATORS = ("os.putenv", "os.unsetenv")
"""The functions that change the environment outside the mapping, as ``os`` names them."""

ENVIRON_METHODS = frozenset({"setdefault", "update", "pop", "popitem", "clear"})
"""The mapping methods that change ``os.environ``; ``get`` and the others only read it."""


def environment_writes(
    node: ast.AST,
    *,
    environ: Collection[str] = OS_ENVIRON,
    mutators: Collection[str] = OS_MUTATORS,
) -> list[str]:
    """What ``node`` writes into the environment: each variable by name, else ``os.environ``.

    Args:
        node: A syntax-tree node.
        environ: The names that stand for the environment mapping.
        mutators: The names of the functions that change the environment.

    Returns:
        The variables written, by name where the key is a string constant; ``os.environ`` for a
        write of the whole mapping or of a key that is not spelled out; empty for anything else.
    """
    if isinstance(node, ast.Assign | ast.Delete):
        return [key for target in node.targets for key in _environ_item(target, environ)]
    if isinstance(node, ast.AugAssign):
        return _augmented(node, environ)
    if isinstance(node, ast.Call):
        return _environ_call(node, environ, mutators)
    return []


def _augmented(node: ast.AugAssign, environ: Collection[str]) -> list[str]:
    """What an augmented assignment writes: the whole mapping (``|=``), or one of its items."""
    whole = ["os.environ"] if ast.unparse(node.target) in environ else []
    return whole + _environ_item(node.target, environ)


def _environ_item(target: ast.expr, environ: Collection[str]) -> list[str]:
    """The variable an assignment or ``del`` target names, when it is an item of the environment."""
    if isinstance(target, ast.Subscript) and ast.unparse(target.value) in environ:
        return [_variable(target.slice)]
    return []


def _environ_call(call: ast.Call, environ: Collection[str], mutators: Collection[str]) -> list[str]:
    """The variables a call that changes the environment names: its first argument or keywords."""
    if not _changes_environment(ast.unparse(call.func), environ, mutators):
        return []
    keywords = [keyword.arg for keyword in call.keywords if keyword.arg]
    return keywords or [_variable(call.args[0]) if call.args else "os.environ"]


def _changes_environment(
    function: str, environ: Collection[str], mutators: Collection[str]
) -> bool:
    """Whether calling ``function`` changes the environment: a mutating method or a mutator."""
    owner, _, method = function.rpartition(".")
    return (owner in environ and method in ENVIRON_METHODS) or function in mutators


def _variable(node: ast.expr) -> str:
    """The variable name a key expression spells, or ``os.environ`` when it is not a constant."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return "os.environ"
