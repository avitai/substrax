"""A repository's example scripts, and each one's path from the repository root.

An example is every ``*.py`` under the repository's ``examples`` directory that is not private
(:func:`discover_examples`). Its path from the root, ``examples/<path>.py``, is the one name its
page records, its budget is keyed by and a change names (:func:`example_source`), so every tool
asks the one function for it. Importing this module imports no jax.
"""

from __future__ import annotations

import os
from pathlib import Path


class ExamplePathError(ValueError):
    """A path is no example's.

    It lies outside ``examples/``, or a symlink below the root leads to it, which would give an
    example a second name or put it outside the repository.
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
    return [path for path in sorted(root.rglob("*.py")) if not _is_private(path.relative_to(root))]


def _is_private(relative: Path) -> bool:
    """Whether a part of the path below the root starts with ``_``."""
    return any(part.startswith("_") for part in relative.parts)


def example_source(script: Path, *, repo_root: Path) -> str:
    """Return an example's path from the repository root, as its page records it.

    The path is taken as written, from the working directory when relative, and starts below the
    first of its ancestors that resolves to the root, so a link above the root (one filesystem
    mounted at two paths) changes nothing. A link anywhere below the root is refused: the example
    would answer to a second name, or lie outside the repository.

    Args:
        script: The example script.
        repo_root: The repository root.

    Returns:
        ``examples/<path>.py``, with ``/`` separators.

    Raises:
        ExamplePathError: If ``script`` is not under ``repo_root / "examples"``, or a symlink below
            the root leads to it.
    """
    root = repo_root.resolve()
    written = _lexical(script)
    # The path from the root starts below the first ancestor that is the root, links above it
    # followed; any link below it then shows as a resolved path that differs.
    starts = (above for above in reversed(written.parents) if above.resolve() == root)
    relative = next((written.relative_to(above) for above in starts), None)
    if relative is None or relative.parts[:1] != ("examples",):
        msg = f"{script} is not an example under {repo_root / 'examples'}"
        raise ExamplePathError(msg)
    if (root / relative).resolve() != root / relative:
        msg = (
            "an example is reached through a symlink, which pages and changes cannot follow: "
            f"{relative.as_posix()}"
        )
        raise ExamplePathError(msg)
    return relative.as_posix()


def _lexical(path: Path) -> Path:
    """``path`` made absolute and normalised as written: ``..`` removed, no symlink followed.

    ``Path.resolve`` follows links, which is what :func:`example_source` must see, not do.
    """
    return Path(os.path.normpath(path.absolute()))
