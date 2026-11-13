"""A run's outputs made fit to publish on a page, and the local paths a page must not hold.

Every known root in the outputs (the run's artifact directory, the repository, the environment,
the temporary directory, the home directory) becomes a placeholder where it starts a path,
terminal control sequences are removed, and ipykernel's per-process cell directory loses its
number. :func:`local_paths` finds what still names a person's machine.
"""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

import nbformat

from substrax.notebooks.terminal import strip_terminal_controls


_NOT_IN_A_URL = r"(?<![\w.~-])"
"""Where a local path may start: not inside a word or a URL's path (``example.org/home/...``)."""
_NAME = r"[^/\\\s'\"`]+"
"""One path component, up to a separator, a space or a quote; a trailing full stop included."""
LOCAL_PATH_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(rf"{_NOT_IN_A_URL}/media/[^/\s]+/{_NAME}"),
    re.compile(rf"{_NOT_IN_A_URL}/Users/{_NAME}"),
    re.compile(rf"{_NOT_IN_A_URL}/home/{_NAME}"),
    # A Windows path as printed, or as repr prints it, each backslash doubled.
    re.compile(rf"\b[A-Za-z]:\\{{1,2}}Users\\{{1,2}}{_NAME}"),
)
"""Paths naming a person's machine: artifex's root-hygiene patterns, matched whether or not a
separator follows the user's name (``/home/alice.`` ends a sentence), and Windows paths in either
spelling."""
_KERNEL_CELL_DIRECTORY = re.compile(r"<tmp>[/\\]ipykernel_\d+\b")
"""ipykernel's per-process directory for cell files under the temporary root's placeholder
(``ipykernel.compiler.get_tmp_directory``); the same name elsewhere is the example's own text."""


def publish_outputs(notebook: nbformat.NotebookNode, *, repo_root: Path, output_dir: Path) -> None:
    """Make every text output of an executed notebook fit to publish, in place.

    In each output a run printed, displayed or raised, terminal control sequences are removed, the
    run's known roots (its artifact directory, the repository, the environment, the temporary and
    home directories, each as given or resolved through symlinks) become placeholders where they
    start a path, and ipykernel's per-process cell directory loses its number. Code cells' sources
    are left as the script has them.

    Args:
        notebook: The executed notebook; its outputs are rewritten.
        repo_root: The repository root, shown as ``.``.
        output_dir: The run's artifact directory, shown as ``<outputs>``.
    """
    roots = _root_patterns(repo_root, output_dir=output_dir)
    for cell in notebook.cells:
        for output in cell.get("outputs", []):
            if output.output_type == "stream":
                output.text = _publishable(output.text, roots)
            elif output.output_type == "error":
                output.evalue = _publishable(output.evalue, roots)
                output.traceback = [_publishable(line, roots) for line in output.traceback]
            else:
                data = output.get("data", {})
                for mime, value in data.items():
                    if mime.startswith("text/") and isinstance(value, str):
                        data[mime] = _publishable(value, roots)


def _publishable(text: str, roots: list[tuple[re.Pattern[str], str]]) -> str:
    """An output's text as the page shows it, the same on every run that prints the same.

    Terminal control sequences are removed (:func:`strip_terminal_controls`); the run's roots
    become placeholders, the run's artifact directory as ``<outputs>``; and ipykernel's directory
    for cell files under the temporary root, named for the kernel's process (``ipykernel_<pid>``),
    loses the number. The file name inside it is a hash of the cell's code with a fixed seed, so
    a warning names the same file on every run.
    """
    return _KERNEL_CELL_DIRECTORY.sub(
        "<tmp>/ipykernel", _replace_roots(strip_terminal_controls(text), roots)
    )


def _root_patterns(repo_root: Path, *, output_dir: Path) -> list[tuple[re.Pattern[str], str]]:
    """The run's known roots, longest first, each matched only where it starts a whole path.

    Each root is matched as given and as resolved through symlinks, since a run prints either: a
    script's ``__file__`` keeps a symlinked checkout's path, ``Path.resolve()`` and the working
    directory give the real one, and macOS's temporary root ``/var`` is ``/private/var``.

    A root counts when no path or word character comes before it and none follows it except a
    separator or punctuation: ``/tmp`` in ``/tmp/x`` and ``saved to /tmp.``, but not in
    ``/tmp_demo``, ``/tmp.x`` or ``example.org/tmp``. A root of
    one character (a home directory of ``/``) is never replaced.
    """
    placeholders = {
        # The run's artifact directory, a new temporary name each run, under the temporary root.
        output_dir: "<outputs>",
        repo_root.absolute(): ".",
        Path(sys.prefix): "<env>",
        Path(sys.base_prefix): "<python>",
        Path(tempfile.gettempdir()): "<tmp>",
        Path.home(): "~",
    }
    roots = {
        str(spelling): placeholder
        for root, placeholder in placeholders.items()
        for spelling in (root, root.resolve())
    }
    return [
        (re.compile(rf"(?<![\w.~/\\-]){re.escape(root)}(?![\w~-]|\.\w)"), roots[root])
        for root in sorted(roots, key=len, reverse=True)
        if len(root) > 1
    ]


def _replace_roots(text: str, roots: list[tuple[re.Pattern[str], str]]) -> str:
    """``text`` with each root replaced by its placeholder."""
    for pattern, placeholder in roots:
        text = pattern.sub(placeholder, text)
    return text


def local_paths(text: str) -> set[str]:
    r"""The paths naming a person's machine that a text still holds.

    Args:
        text: The text to search, a page's or an output's.

    Returns:
        Each match of :data:`LOCAL_PATH_PATTERNS`, such as ``/home/alice`` or ``C:\Users\bob``.
    """
    return {match.group(0) for pattern in LOCAL_PATH_PATTERNS for match in pattern.finditer(text)}
