"""A repository's example scripts, and each one's path from the repository root.

An example is every ``*.py`` under the repository's ``examples`` directory that is not private
(:func:`discover_examples`). Its path from the root, ``examples/<path>.py``, is the one name its
page records, its budget is keyed by and a change names (:func:`example_source`), so every tool
asks the one function for it. Importing this module imports no jax.
"""

from __future__ import annotations

import os
from pathlib import Path


class ScriptPathError(ValueError):
    """A path is no example's or recipe's.

    It lies outside ``examples/`` (or ``recipes/``), or a symlink below the root leads to it, which
    would give the script a second name or put it outside the repository.
    """


def discover_examples(root: Path) -> list[Path]:
    """List the example scripts under ``root``, sorted.

    A file or directory whose name starts with ``_`` is private, as ``_common`` helpers,
    ``_templates`` and ``__init__.py`` are, and so is everything under it. Only the parts below
    ``root`` count, so ``root`` itself may have such a name.

    Args:
        root: The examples directory.

    Returns:
        The ``*.py`` files under ``root`` that are not private.
    """
    return [path for path in sorted(root.rglob("*.py")) if not is_private(path.relative_to(root))]


def is_private(relative: Path) -> bool:
    """Whether a part of a path below the examples directory starts with ``_``, as a helper's does.

    Args:
        relative: The path, from the examples directory.

    Returns:
        Whether the file is private, so no example, notebook or page.
    """
    return any(part.startswith("_") for part in relative.parts)


def example_source(script: Path, *, repo_root: Path) -> str:  # noqa: DOC502  # raised by source_below
    """Return an example's path from the repository root, as its page records it.

    Args:
        script: The example script.
        repo_root: The repository root.

    Returns:
        ``examples/<path>.py``, with ``/`` separators.

    Raises:
        ScriptPathError: If ``script`` is not under ``repo_root / "examples"``, or a symlink below
            the root leads to it (:func:`source_below`).
    """
    return source_below(script, repo_root=repo_root, directory="examples", noun="an example")


def source_below(script: Path, *, repo_root: Path, directory: str, noun: str) -> str:
    """Return a script's path from the repository root, refusing one outside ``directory``.

    The path is taken as written, from the working directory when relative, and starts below the
    first of its ancestors that resolves to the root, so a link above the root (one filesystem
    mounted at two paths) changes nothing. A link anywhere below the root is refused: the script
    would answer to a second name, or lie outside the repository.

    Args:
        script: The script.
        repo_root: The repository root.
        directory: The directory, below the root, the script must lie in: ``examples``,
            ``recipes``.
        noun: What such a script is, for the message: ``an example``, ``a recipe``.

    Returns:
        ``<directory>/<path>.py``, with ``/`` separators.

    Raises:
        ScriptPathError: If ``script`` is not under ``repo_root / directory``, or a symlink below
            the root leads to it.
    """
    root = repo_root.resolve()
    written = lexical_path(script)
    # The path from the root starts below the first ancestor that is the root, links above it
    # followed; any link below it then shows as a resolved path that differs.
    starts = (above for above in reversed(written.parents) if above.resolve() == root)
    relative = next((written.relative_to(above) for above in starts), None)
    if relative is None or relative.parts[:1] != (directory,):
        msg = f"{script} is not {noun} under {repo_root / directory}"
        raise ScriptPathError(msg)
    if (root / relative).resolve() != root / relative:
        msg = (
            f"{noun} is reached through a symlink, which pages and changes cannot follow: "
            f"{relative.as_posix()}"
        )
        raise ScriptPathError(msg)
    return relative.as_posix()


def lexical_path(path: Path) -> Path:
    """Return ``path`` made absolute and normalised as written: ``..`` removed, no link followed.

    ``Path.resolve`` follows links, which is what :func:`example_source` must see, not do, and a
    path named on a command line is compared with the scripts a directory lists as written.

    Args:
        path: A path, relative to the working directory or absolute.

    Returns:
        The absolute path, normalised without touching the filesystem.
    """
    return Path(os.path.normpath(path.absolute()))
