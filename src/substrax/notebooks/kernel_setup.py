"""The setup a run gives an example's kernel, before the example's first cell.

The kernel never imports this module: the run sends its source in a hidden cell, which executes
it in a namespace of its own and calls :func:`setup_kernel` with the kernel's user namespace and
shell, so the example sees no name of substrax's but the reporter :func:`setup_kernel` binds.
The example then sees what ``python <script>`` gives a script: its directory first on
``sys.path`` and not the working directory IPython adds, its path in ``sys.argv``, its
``__file__``. IPython applies :class:`MainResult` to each cell's syntax tree as it runs, so the
top-level ``__main__`` guard keeps ``main()``'s value while the cell's source, and the page, stay
the author's. The reporter, ``_substrax_report``, is called after the example and writes the
devices, the versions and ``main()``'s value to the report's file, through references the example
cannot rebind: this namespace's, bound before the example runs. After every cell IPython calls
:func:`record_raised`, which writes the qualified class of an exception the cell raised, so the run
names the class itself where the kernel's error reply gives only its unqualified name.
"""

from __future__ import annotations

import ast
import copy
import json
import platform
import sys
from builtins import open as open_file
from collections.abc import Callable, MutableMapping
from contextlib import suppress
from functools import partial
from importlib import metadata
from typing import Protocol, TypeGuard


MAIN_RETURNED = "_substrax_main_returned"
"""The name, in the kernel's user namespace, that holds ``main()``'s value."""

REPORTER = "_substrax_report"
"""The name, in the kernel's user namespace, of the reporter the run calls after the example."""

# A script whose main() exits as Python exits with status 0 succeeded: a code of None or an
# integer 0 (False included). Any other code, 0.0 or a message, is printed and exits 1.
_KEEP_AND_EXIT_CLEANLY = ast.parse(
    "try:\n"
    f"    {MAIN_RETURNED} = main()\n"
    "except SystemExit as _substrax_exit:\n"
    "    _substrax_code = _substrax_exit.code\n"
    "    if not (\n"
    "        _substrax_code is None or (isinstance(_substrax_code, int) and _substrax_code == 0)\n"
    "    ):\n"
    "        raise\n"
).body[0]


class Events(Protocol):
    """The part of IPython's event manager the setup uses."""

    def register(self, event: str, function: Callable[..., object]) -> None:
        """Call ``function`` on every ``event``."""


class Shell(Protocol):
    """The part of IPython's interactive shell the setup uses."""

    ast_transformers: list[ast.NodeTransformer]

    @property
    def events(self) -> Events:
        """The shell's event manager, read and never replaced."""
        ...


class ExecutionResult(Protocol):
    """The part of IPython's result of running a cell that the recorder reads."""

    error_in_exec: BaseException | None


def setup_kernel(
    user_ns: MutableMapping[str, object],
    shell: Shell,
    *,
    directory: str,
    argv0: str,
    file: str,
    report_path: str,
    raised_path: str,
    distribution: str,
) -> None:
    """Give the kernel a script run's view, bind the reporter and register the recorder.

    Args:
        user_ns: The kernel's user namespace, the example's globals.
        shell: The kernel's interactive shell.
        directory: The script's directory, first on ``sys.path``.
        argv0: The script's path as the page's command names it, ``sys.argv[0]``.
        file: The script's absolute path, ``__file__``.
        report_path: Where the reporter writes the run's report.
        raised_path: Where the recorder writes the qualified class of a cell's exception.
        distribution: The distribution the repository builds, whose version the report
            holds.
    """
    sys.path[:] = [directory, *(entry for entry in sys.path if entry not in ("", directory))]
    sys.argv = [argv0]
    user_ns["__file__"] = file
    user_ns[REPORTER] = partial(write_report, user_ns, report_path, distribution=distribution)
    shell.ast_transformers.append(MainResult())
    shell.events.register("post_run_cell", partial(record_raised, raised_path))


def record_raised(raised_path: str, result: ExecutionResult | None) -> None:
    """Write the qualified class of the exception a cell raised, ``builtins.ValueError``.

    Args:
        raised_path: The file the class is written to; a cell that raised nothing writes nothing.
        result: IPython's result of running the cell, ``None`` when running it failed in IPython.
    """
    error = None if result is None else result.error_in_exec
    if error is None:
        return
    kind = type(error)
    # The open bound before the example ran, as the reporter's is: the example may rebind io.open.
    with open_file(raised_path, "w", encoding="utf-8") as out:  # noqa: PTH123
        json.dump(f"{kind.__module__}.{kind.__qualname__}", out)


def write_report(
    user_ns: MutableMapping[str, object], report_path: str, *, distribution: str
) -> None:
    """Write the devices, the versions and ``main()``'s value to the report's file.

    Args:
        user_ns: The kernel's user namespace, where the guard left ``main()``'s value.
        report_path: The report's file.
        distribution: The distribution the repository builds; its version is reported
            where installed, as jax's and substrax's are.
    """
    versions = {"python": platform.python_version()}
    for name in ("jax", "substrax", distribution):
        with suppress(metadata.PackageNotFoundError):
            versions[name] = metadata.version(name)
    result = {"device": _device(), "versions": versions, **_returned(user_ns.get(MAIN_RETURNED))}
    # The open bound before the example ran: Path.open looks io.open up when called, which the
    # example could have rebound.
    with open_file(report_path, "w", encoding="utf-8") as out:  # noqa: PTH123
        json.dump(result, out)


def _device() -> str:
    """The devices jax saw, or ``cpu`` when the example never imported jax."""
    jax_module = sys.modules.get("jax")
    if jax_module is None:
        return "cpu"
    devices = jax_module.devices()
    first = devices[0]
    kind = "" if first.device_kind == first.platform else f" ({first.device_kind})"
    return f"{len(devices)} {first.platform}{kind}"


def _returned(returned: object) -> dict[str, object]:
    """The report's ``returned`` entry, or its ``returned_error`` naming what JSON cannot hold.

    Args:
        returned: ``main()``'s value.

    Returns:
        ``{"returned": value}`` for a value JSON holds; otherwise ``{"returned_error": reason}``,
        naming the first entry of a dict JSON cannot hold.
    """
    for key, value in returned.items() if isinstance(returned, dict) else ():
        if (error := _json_error(value)) is not None:
            return {"returned_error": f"main() returned {key!r}, which JSON cannot hold: {error}"}
    if (error := _json_error(returned)) is not None:
        return {"returned_error": f"main() returned a value JSON cannot hold: {error}"}
    return {"returned": json.loads(json.dumps(returned, default=_scalar))}


def _json_error(value: object) -> str | None:
    """Why JSON cannot hold ``value``, zero-dimensional arrays read as their items, or ``None``."""
    try:
        json.dumps(value, default=_scalar)
    except (TypeError, ValueError) as error:
        return str(error)
    return None


def _scalar(value: object) -> object:
    """A zero-dimensional array's item, which JSON can hold.

    Args:
        value: What ``json.dumps`` met and cannot hold itself.

    Returns:
        The array's item.

    Raises:
        TypeError: If ``value`` is not a zero-dimensional array.
    """
    shape, item = getattr(value, "shape", None), getattr(value, "item", None)
    if shape == () and callable(item):
        return item()
    msg = f"{type(value).__name__} with shape {shape} is not a JSON value"
    raise TypeError(msg)


class MainResult(ast.NodeTransformer):
    """Rewrite a top-level ``__main__`` guard calling ``main()`` to keep its value."""

    def visit_Module(self, node: ast.Module) -> ast.Module:
        """Replace each guard's ``main()`` call with one that keeps its value and a clean exit.

        Args:
            node: A cell's syntax tree.

        Returns:
            The tree, each guard's call replaced.
        """
        for statement in node.body:
            if is_guarded_main_call(statement):
                call = statement.body[0]
                replacement = copy.deepcopy(_KEEP_AND_EXIT_CLEANLY)
                for child in ast.walk(replacement):
                    ast.copy_location(child, call)
                statement.body[0] = replacement
        return ast.fix_missing_locations(node)


def is_main_guard(node: ast.AST) -> bool:
    """Whether ``node`` is ``if __name__ == "__main__":`` (either operand order).

    Its body runs only when the module is run as a script. ``!=`` and chained comparisons are not
    guards: their bodies run at import.

    Args:
        node: A syntax-tree node.

    Returns:
        Whether it is the ``__main__`` guard.
    """
    if not (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)):
        return False
    test = node.test
    if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
        return False
    left, right = test.left, test.comparators[0]
    return (_is_name_variable(left) and _is_main_constant(right)) or (
        _is_main_constant(left) and _is_name_variable(right)
    )


def is_guarded_main_call(node: ast.AST) -> TypeGuard[ast.If]:
    """Whether ``node`` is the ``__main__`` guard whose only statement is the call ``main()``.

    Args:
        node: A syntax-tree node.

    Returns:
        Whether it is :func:`is_main_guard`'s guard with ``main()``, no arguments, its whole body.
    """
    return (
        is_main_guard(node)
        and isinstance(node, ast.If)
        and len(node.body) == 1
        and _is_call_to_main(node.body[0])
    )


def _is_main_constant(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value == "__main__"


def _is_name_variable(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and node.id == "__name__"


def _is_call_to_main(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == "main"
        and not statement.value.args
        and not statement.value.keywords
    )
