"""A repository's example scripts: every ``*.py`` under its examples directory that is not private.

Importing this module imports no jax.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path


def discover_examples(root: Path, *, include: Callable[[Path], bool] | None = None) -> list[Path]:
    """List the example scripts under ``root``, sorted.

    A file or directory whose name starts with ``_`` is private, as ``_common`` helpers,
    ``_templates`` and ``__init__.py`` are, and so is everything under it. Only the parts below
    ``root`` count, so ``root`` itself may have such a name.

    Args:
        root: The examples directory.
        include: Keeps a path when it returns ``True``, such as a repository's numbering rule.

    Returns:
        The ``*.py`` files under ``root`` that are not private and that ``include`` keeps.
    """
    return [
        path
        for path in sorted(root.rglob("*.py"))
        if not _is_private(path.relative_to(root)) and (include is None or include(path))
    ]


def _is_private(relative: Path) -> bool:
    """Whether a part of the path below the root starts with ``_``."""
    return any(part.startswith("_") for part in relative.parts)
