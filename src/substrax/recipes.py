"""A repository's full recipes, each the published-scale run of the example at its path.

An example runs at the size its page shows on every lane; when it ports a published reference, the
reference's own configuration at its published scale is the example's recipe. A recipe is every
``*.py`` under the repository's ``recipes`` directory that is not private, by the rule examples
follow (:func:`substrax.examples.discover_examples`). ``recipes/<path>.py`` is the recipe of
``examples/<path>.py`` (:func:`paired_example`), and its page is ``docs/examples/<path>.recipe.md``
(:func:`recipe_page_path`), beside the example's. Importing this module imports no jax.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from substrax.examples import discover_examples, source_below


RECIPES_DIRECTORY = "recipes"
"""The directory, under the repository root, that holds the recipes."""
RECIPE_PAGE_SUFFIX = ".recipe.md"
"""What a recipe page's name ends with; an example's name, snake_case, holds no dot."""


def discover_recipes(repo_root: Path) -> list[Path]:
    """List a repository's recipes, sorted.

    Args:
        repo_root: The repository root.

    Returns:
        The ``*.py`` files under ``recipes`` that are not private; none without the directory.
    """
    return discover_examples(repo_root / RECIPES_DIRECTORY)


def recipe_source(recipe: Path, *, repo_root: Path) -> str:  # noqa: DOC502  # raised by source_below
    """Return a recipe's path from the repository root, as its record and page name it.

    Args:
        recipe: The recipe.
        repo_root: The repository root.

    Returns:
        ``recipes/<path>.py``, with ``/`` separators.

    Raises:
        ScriptPathError: If ``recipe`` is not under ``repo_root / "recipes"``, or a symlink below
            the root leads to it (:func:`substrax.examples.source_below`).
    """
    return source_below(recipe, repo_root=repo_root, directory=RECIPES_DIRECTORY, noun="a recipe")


def paired_example(recipe: Path, *, repo_root: Path) -> Path:
    """Return the example a recipe is the full-scale run of.

    Args:
        recipe: The recipe, under ``repo_root / "recipes"``.
        repo_root: The repository root.

    Returns:
        ``examples/<the recipe's path below recipes>`` under ``repo_root``.
    """
    return repo_root / "examples" / recipe.relative_to(repo_root / RECIPES_DIRECTORY)


def recipe_page_path(recipe: Path, *, repo_root: Path) -> Path:
    """Return where a recipe's page lives: beside its example's page, below ``docs/examples``.

    Args:
        recipe: The recipe, under ``repo_root / "recipes"``.
        repo_root: The repository root.

    Returns:
        ``docs/examples/<the recipe's path below recipes>`` with :data:`RECIPE_PAGE_SUFFIX`.
    """
    relative = PurePosixPath(recipe.relative_to(repo_root / RECIPES_DIRECTORY).as_posix())
    return repo_root / "docs" / "examples" / relative.with_suffix(RECIPE_PAGE_SUFFIX)
