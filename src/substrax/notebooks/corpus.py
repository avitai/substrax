"""A repository's examples as a whole, and which of their runs a change reaches.

:func:`corpus_problems` holds the examples to the guide's corpus rules: there is at least one, every
page and figures directory under ``docs/examples`` belongs to one, a hand-written
``uv run python examples/...`` command in a tracked Markdown file names a file that exists, and
``mkdocs.yml`` lets no page leave the nav unnoticed. :func:`reaches_every_example` and
:func:`is_touched` decide which runs a change since a ref keeps.
"""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath

import yaml

from substrax.examples import example_source
from substrax.notebooks import repository
from substrax.notebooks.pages import page_path, PYTHON_COMMAND


LIBRARY_PATHS = ("src/", "pyproject.toml", "uv.lock")
"""A change to any of these is a change to what every example runs on: every example runs."""


_COMMAND = re.compile(rf"{re.escape(PYTHON_COMMAND)} (?P<path>examples/[^\s`'\")]+\.(?:py|ipynb))")


def corpus_problems(repo_root: Path, examples: frozenset[Path]) -> list[str]:
    """Every problem with a repository's examples as a whole.

    Args:
        repo_root: The repository root.
        examples: The repository's example scripts, resolved.

    Returns:
        One line per problem: no example, a page or figure directory with no example, a
        hand-written command naming a missing file, or a mkdocs nav that lets a page go unlisted.
    """
    if not examples:
        return [f"no example under {repo_root / 'examples'}"]
    return [
        *_orphan_pages(repo_root, examples),
        *_dead_commands(repo_root),
        *_unlisted_pages_allowed(repo_root),
    ]


def reaches_every_example(changed: set[str], examples: frozenset[Path], repo_root: Path) -> bool:
    """Whether a change reaches what any example may run on, so every example runs.

    That is the library (:data:`LIBRARY_PATHS`), or a file under ``examples/`` that is no
    example's own script, notebook or page, as a private helper or a data file is.

    Args:
        changed: The changed paths, from the repository root.
        examples: The repository's example scripts, resolved.
        repo_root: The repository root.

    Returns:
        Whether every example's run is kept.
    """
    if any(path.startswith(LIBRARY_PATHS) for path in changed):
        return True
    owned = {path for script in examples for path in _own_paths(script, repo_root)}
    return any(path.startswith("examples/") and path not in owned for path in changed)


def is_touched(script: Path, changed: set[str], repo_root: Path) -> bool:
    """Whether a change touched an example's script, notebook or page.

    Args:
        script: The example script.
        changed: The changed paths, from the repository root.
        repo_root: The repository root.

    Returns:
        Whether one of the example's own files is among ``changed``.
    """
    return not _own_paths(script, repo_root).isdisjoint(changed)


def _own_paths(script: Path, repo_root: Path) -> set[str]:
    """An example's script, notebook and page, as paths from the repository root."""
    source = PurePosixPath(example_source(script, repo_root=repo_root))
    return {
        source.as_posix(),
        source.with_suffix(".ipynb").as_posix(),
        ("docs" / source.with_suffix(".md")).as_posix(),
    }


def _orphan_pages(repo_root: Path, examples: frozenset[Path]) -> list[str]:
    """The pages and figure directories under ``docs/examples`` that no example makes."""
    pages_root = repo_root / "docs" / "examples"
    if not pages_root.is_dir():
        return []
    pages = {page_path(script, repo_root=repo_root).resolve() for script in examples}
    orphans = [path for path in sorted(pages_root.rglob("*.md")) if path.resolve() not in pages]
    orphans += _orphan_figures_directories(pages_root, pages)
    return [
        f"{path.relative_to(repo_root)}: no example makes this page; its example was renamed, "
        "moved or removed"
        for path in orphans
    ]


def _orphan_figures_directories(pages_root: Path, pages: set[Path]) -> list[Path]:
    """The ``*_files`` directories under ``pages_root`` that are no example's page's figures.

    A ``*_files`` directory is a page's figures directory unless an example's page lives in it:
    the page directory of ``examples/data_files/loading.py`` is ``docs/examples/data_files``.
    """
    return [
        path
        for path in sorted(pages_root.rglob("*_files"))
        if path.is_dir()
        and path.with_name(path.name.removesuffix("_files") + ".md").resolve() not in pages
        and not any(page.is_relative_to(path.resolve()) for page in pages)
    ]


def _dead_commands(repo_root: Path) -> list[str]:
    """Each hand-written ``uv run python examples/...`` naming a file that does not exist."""
    pages = repo_root / "docs" / "examples"
    problems = []
    for markdown in repository.tracked(repo_root, "*.md"):
        if markdown.is_relative_to(pages):
            continue
        for match in _COMMAND.finditer(markdown.read_text(encoding="utf-8")):
            if not (repo_root / match.group("path")).is_file():
                problems.append(
                    f"{markdown.relative_to(repo_root)}: `{PYTHON_COMMAND} {match.group('path')}` "
                    "names no file"
                )
    return problems


def _unlisted_pages_allowed(repo_root: Path) -> list[str]:
    """A problem when mkdocs.yml lets ``mkdocs build --strict`` pass with a page off the nav.

    mkdocs.yml holds tags (``!ENV``, ``!!python/name``) a safe loader refuses, so the file is
    composed into YAML nodes, which reads its structure and constructs no value.
    """
    mkdocs = repo_root / "mkdocs.yml"
    if not mkdocs.is_file():
        return []
    document = yaml.compose(mkdocs.read_text(encoding="utf-8"), Loader=yaml.SafeLoader)
    validation = _mapping_value(document, "validation")
    level = _mapping_value(_mapping_value(validation, "nav"), "omitted_files")
    level = level or _mapping_value(validation, "omitted_files")
    value = level.value if isinstance(level, yaml.ScalarNode) else None
    if value == "warn":
        return []
    return [
        f"mkdocs.yml: validation.nav.omitted_files is {value!r}; set it to 'warn' so "
        "mkdocs build --strict fails on a page left out of the nav"
    ]


def _mapping_value(node: yaml.Node | None, key: str) -> yaml.Node | None:
    """The value under ``key`` of a YAML mapping node, or ``None``."""
    if not isinstance(node, yaml.MappingNode):
        return None
    return next((value for name, value in node.value if name.value == key), None)
