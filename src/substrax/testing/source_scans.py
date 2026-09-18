"""Source scans a repository's contract tests run over its own files.

Two families of check read source rather than run it:

- **Import-time side effects.** :func:`import_time_lines` walks a module and returns the lines of
  the nodes a predicate matches, among those that run when the module is imported: function,
  class and lambda bodies and the ``if __name__ == "__main__":`` block are skipped.
  :func:`writes_environment` matches a change to ``os.environ`` (assignment, deletion, ``|=``, a
  mutating method, ``os.putenv``/``os.unsetenv``); :func:`configures_logging` matches a
  ``logging.basicConfig`` call. A module that does either at import changes the process for
  everything imported after it, test collection included.
- **Documented meshes.** :func:`unguarded_make_mesh_calls` returns each ``make_mesh`` call whose
  arguments do not name ``axis_types``: from jax 0.11 ``jax.make_mesh`` builds
  ``AxisType.Explicit`` axes by default, and the backward pass of a batch-sharded step raises under
  them. :func:`documented_texts` collects what a reader copies from: every docstring in the
  package, the named pages, every Markdown page under the page directories and every Python file
  under the script directories.

A corpus location that does not exist raises :class:`FileNotFoundError`, so a mistyped directory
fails the check instead of shrinking what it reads. Importing this module imports no jax.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Collection, Sequence
from pathlib import Path


# The ``os.environ`` methods that change the environment.
_ENVIRONMENT_METHODS = frozenset({"update", "setdefault", "pop", "popitem", "clear"})
_MESH_CALL = "make_mesh("
_DOCUMENTED_NODES = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
_DEFERRED_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


def python_files(
    root: Path,
    directories: Sequence[str],
    *,
    pattern: str = "*.py",
    excluded_parts: Collection[str] = (),
) -> list[Path]:
    """Every file under ``root / directory`` matching ``pattern``, in directory then path order.

    Args:
        root: The repository root.
        directories: Directories relative to ``root`` to search recursively.
        pattern: The file-name glob.
        excluded_parts: Directory names whose files are skipped, such as ``"example_data"``.

    Returns:
        The matching files.

    Raises:
        FileNotFoundError: If a directory does not exist.
    """
    files: list[Path] = []
    for directory in directories:
        base = root / directory
        if not base.is_dir():
            raise FileNotFoundError(f"corpus directory {base} does not exist")
        files += [
            path
            for path in sorted(base.rglob(pattern))
            if not set(path.relative_to(root).parts) & set(excluded_parts)
        ]
    return files


def _is_os_environ(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "environ"
        and isinstance(node.value, ast.Name)
        and node.value.id == "os"
    )


def _is_environment_item(node: ast.AST) -> bool:
    return isinstance(node, ast.Subscript) and _is_os_environ(node.value)


def _assigns_environment(node: ast.AST) -> bool:
    if isinstance(node, (ast.Assign, ast.Delete)):
        return any(_is_environment_item(target) for target in node.targets)
    if isinstance(node, ast.AugAssign):
        return _is_os_environ(node.target) or _is_environment_item(node.target)
    return False


def _calls_environment_mutator(node: ast.AST) -> bool:
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
        return False
    function = node.func
    if _is_os_environ(function.value):
        return function.attr in _ENVIRONMENT_METHODS
    return (
        isinstance(function.value, ast.Name)
        and function.value.id == "os"
        and function.attr in {"putenv", "unsetenv"}
    )


def writes_environment(node: ast.AST) -> bool:
    """Whether ``node`` changes ``os.environ`` or calls ``os.putenv``/``os.unsetenv``."""
    return _assigns_environment(node) or _calls_environment_mutator(node)


def configures_logging(node: ast.AST) -> bool:
    """Whether ``node`` calls ``logging.basicConfig``, by module attribute or imported name."""
    if not isinstance(node, ast.Call):
        return False
    function = node.func
    if isinstance(function, ast.Attribute):
        return (
            function.attr == "basicConfig"
            and isinstance(function.value, ast.Name)
            and function.value.id == "logging"
        )
    return isinstance(function, ast.Name) and function.id == "basicConfig"


def _is_main_constant(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value == "__main__"


def _is_name_variable(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and node.id == "__name__"


def is_main_guard(node: ast.AST) -> bool:
    """Whether ``node`` is ``if __name__ == "__main__":`` (either operand order).

    Its body runs only when the module is run as a script. ``!=`` and chained comparisons are not
    guards: their bodies run at import.
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


def import_time_lines(path: Path, matches: Callable[[ast.AST], bool]) -> list[int]:
    """Line numbers of the nodes ``matches`` accepts that run when ``path`` is imported.

    Args:
        path: A Python source file.
        matches: A predicate over one AST node, such as :func:`writes_environment`.

    Returns:
        The line numbers in source order.
    """
    lines: list[int] = []

    def visit(node: ast.AST) -> None:
        if isinstance(node, _DEFERRED_NODES) or is_main_guard(node):
            return
        if isinstance(node, (ast.stmt, ast.expr)) and matches(node):
            lines.append(node.lineno)
        for child in ast.iter_child_nodes(node):
            visit(child)

    for statement in ast.parse(path.read_text(encoding="utf-8")).body:
        visit(statement)
    return lines


def unguarded_make_mesh_calls(text: str) -> list[str]:
    """Each ``make_mesh`` call in ``text`` whose arguments do not name ``axis_types``.

    A call whose parentheses never close is returned from its name to the end of the text.
    """
    calls: list[str] = []
    start = text.find(_MESH_CALL)
    while start != -1:
        depth = 0
        end = len(text)
        for index in range(start + len(_MESH_CALL) - 1, len(text)):
            depth += {"(": 1, ")": -1}.get(text[index], 0)
            if depth == 0:
                end = index + 1
                break
        call = text[start:end]
        if depth != 0 or "axis_types" not in call:
            calls.append(call)
        start = text.find(_MESH_CALL, start + len(_MESH_CALL))
    return calls


def documented_texts(
    root: Path,
    *,
    package: str,
    pages: Sequence[str] = ("README.md",),
    page_directories: Sequence[str] = ("docs",),
    script_directories: Sequence[str] = (),
) -> dict[str, str]:
    """Map every text a reader copies code from to its content, keyed by a repository path.

    Args:
        root: The repository root.
        package: The package under ``root / "src"`` whose docstrings are read.
        pages: Individual pages relative to ``root``.
        page_directories: Directories whose Markdown pages are read.
        script_directories: Directories whose Python files are read whole, such as ``"examples"``.

    Returns:
        Docstrings keyed ``"<path>:<line>"`` (line 0 for a module docstring), and pages and
        scripts keyed by their path, all relative to ``root``.

    Raises:
        FileNotFoundError: If the package, a page or a directory does not exist.
    """
    texts: dict[str, str] = {}
    for path in python_files(root, (f"src/{package}",)):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, _DOCUMENTED_NODES) and (docstring := ast.get_docstring(node)):
                line = 0 if isinstance(node, ast.Module) else node.lineno
                texts[f"{path.relative_to(root).as_posix()}:{line}"] = docstring
    for page in pages:
        if not (root / page).is_file():
            raise FileNotFoundError(f"corpus page {root / page} does not exist")
    files = [
        *(root / page for page in pages),
        *python_files(root, page_directories, pattern="*.md"),
        *python_files(root, script_directories),
    ]
    for path in files:
        texts[path.relative_to(root).as_posix()] = path.read_text(encoding="utf-8")
    return texts
