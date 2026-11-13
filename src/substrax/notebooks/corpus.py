"""A repository's examples as a whole, and which of their runs a change reaches.

:func:`corpus_problems` holds the examples to the guide's corpus rules: there is at least one, every
file under ``docs/examples`` is an example's page or one of its figures, every notebook under
``examples`` has its script, a hand-written ``python examples/...`` or ``uv run python
examples/...`` command in a tracked Markdown file names a file that exists, and ``mkdocs.yml`` lets
no page leave the nav unnoticed. :func:`reaches_every_example` and
:func:`is_touched` decide which runs a change since a ref keeps.
"""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath

import yaml

from substrax.examples import example_source, is_private
from substrax.notebooks import repository
from substrax.notebooks.pages import figures_directory, page_path
from substrax.notebooks.text import NotUtf8Error, read_utf8


LIBRARY_PATHS = ("src/", "pyproject.toml", "uv.lock")
"""A change to any of these is a change to what every example runs on: every example runs."""


_COMMAND = re.compile(
    r"(?<![\w-])(?:uv run )?python (?P<path>examples/[^\s`'\")<>{}*]+\.(?:py|ipynb))"
)
"""A command running an example: ``python`` or ``uv run python``, then a path under ``examples``.

A path holding ``<``, ``>``, ``{``, ``}`` or ``*`` is a placeholder for a shape, not a file.
"""
OVERVIEW = "docs/examples.md"
"""Where a repository's hand-written examples overview lives.

``docs/examples`` holds generated pages and their figures only.
"""


def corpus_problems(repo_root: Path, examples: frozenset[Path]) -> list[str]:
    """Every problem with a repository's examples as a whole.

    Args:
        repo_root: The repository root.
        examples: The repository's example scripts, resolved.

    Returns:
        One line per problem: no example, a file under ``docs/examples`` that is no example's page
        or figure, a notebook with no script, a hand-written command naming a missing file, a
        mkdocs nav that lets a page go unlisted, or a Markdown file or ``mkdocs.yml`` that is not
        UTF-8.
    """
    if not examples:
        return [f"no example under {repo_root / 'examples'}"]
    return [
        *_strays_under_docs_examples(repo_root, examples),
        *_notebooks_without_scripts(repo_root, examples),
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


def _strays_under_docs_examples(repo_root: Path, examples: frozenset[Path]) -> list[str]:
    """The files under ``docs/examples`` that are neither an example's page nor one of its figures.

    The figures of ``examples/<path>.py`` live in its page's figures directory,
    ``docs/examples/<path>_files``; the page of ``examples/data_files/loading.py`` lives in
    ``docs/examples/data_files``, which is then a page directory, not a figures directory.
    """
    pages_root = repo_root / "docs" / "examples"
    if not pages_root.is_dir():
        return []
    pages = {page_path(script, repo_root=repo_root).resolve() for script in examples}
    figures = {figures_directory(page) for page in pages}
    strays = [
        path
        for path in sorted(pages_root.rglob("*"))
        if path.is_file()
        and path.resolve() not in pages
        and not any(path.resolve().is_relative_to(directory) for directory in figures)
    ]
    return [
        f"{path.relative_to(repo_root).as_posix()}: docs/examples holds generated pages and their "
        f"figures only, and no example makes this file; an examples overview lives at {OVERVIEW}"
        for path in strays
    ]


def _notebooks_without_scripts(repo_root: Path, examples: frozenset[Path]) -> list[str]:
    """The notebooks under ``examples`` whose script is no example, as a rename leaves behind."""
    root = repo_root / "examples"
    notebooks = {script.with_suffix(".ipynb") for script in examples}
    return [
        f"{path.relative_to(repo_root).as_posix()}: no example script makes this notebook; "
        "convert writes a notebook from its script, so remove it or restore "
        f"{path.with_suffix('.py').relative_to(repo_root).as_posix()}"
        for path in sorted(root.rglob("*.ipynb"))
        if not is_private(path.relative_to(root)) and path.resolve() not in notebooks
    ]


def _dead_commands(repo_root: Path) -> list[str]:
    """Each hand-written command (:data:`_COMMAND`) naming an example file that does not exist."""
    pages = repo_root / "docs" / "examples"
    problems = []
    for markdown in repository.tracked(repo_root, "*.md"):
        if markdown.is_relative_to(pages):
            continue
        try:
            text = read_utf8(markdown)
        except NotUtf8Error as error:
            problems.append(_not_utf8(error, repo_root))
            continue
        for match in _COMMAND.finditer(text):
            if not (repo_root / match.group("path")).is_file():
                problems.append(
                    f"{markdown.relative_to(repo_root)}: `{match.group(0)}` names no file"
                )
    return problems


def _not_utf8(error: NotUtf8Error, repo_root: Path) -> str:
    """The problem a file that is not UTF-8 is, naming it from the repository root."""
    return str(NotUtf8Error(error.path.relative_to(repo_root), error.byte))


def _unlisted_pages_allowed(repo_root: Path) -> list[str]:
    """A problem when mkdocs.yml lets ``mkdocs build --strict`` pass with a page off the nav.

    mkdocs.yml holds tags (``!ENV``, ``!!python/name``) a safe loader refuses, so the file is
    composed into YAML nodes, which reads its structure and constructs no value.
    """
    mkdocs = repo_root / "mkdocs.yml"
    if not mkdocs.is_file():
        return []
    try:
        text = read_utf8(mkdocs)
    except NotUtf8Error as error:
        return [_not_utf8(error, repo_root)]
    document = yaml.compose(text, Loader=yaml.SafeLoader)
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
