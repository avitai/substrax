"""A recipe's docs page, written from its run's record, and the check of a committed one.

A recipe runs on a GPU lane, so its page is written from the record the run left
(:class:`~substrax.notebooks.recipe_run.RecipeRecord`), fetched with the run's outputs, and never
from a run here. :func:`write_recipe_page` writes ``docs/examples/<path>.recipe.md`` beside the
example's page: the recipe's own one-line summary as the title, the header an example's page opens
with (:func:`~substrax.notebooks.page_records.run_header`), a link to the example's page, and what
``main()`` returned. It closes with the record and digests an example's page closes with
(:mod:`~substrax.notebooks.page_records`), so :func:`check_recipe_page` finds a page changed by
hand, or made from another version of the recipe, without running anything.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from substrax.notebooks.config import ExamplesConfig
from substrax.notebooks.outputs import local_paths
from substrax.notebooks.page_records import (
    edited,
    page_digests,
    PageDigests,
    read_page_record,
    record_comment,
    run_header,
)
from substrax.notebooks.pages import PageError, PageProblem
from substrax.notebooks.recipe_run import RecipeRecord
from substrax.notebooks.run import script_sha256
from substrax.notebooks.text import NotUtf8Error, read_utf8, write_utf8
from substrax.recipes import paired_example, recipe_page_path
from substrax.records import UNKNOWN_FIELDS_REFUSED
from substrax.typing import JsonValue


@dataclass(frozen=True, slots=True, kw_only=True)
class _RecipePageRecord:
    """What closes a recipe's page: its run's record and the digests of the page as written."""

    __pydantic_config__ = UNKNOWN_FIELDS_REFUSED

    provenance: RecipeRecord
    page: PageDigests


def write_recipe_page(record: RecipeRecord, *, repo_root: Path, config: ExamplesConfig) -> Path:
    """Write a recipe's page from its run's record.

    Args:
        record: The record a run of the recipe left.
        repo_root: The repository root.
        config: The repository's example settings, for the links' ``source_url``.

    Returns:
        The page written.

    Raises:
        PageError: If the record is of another version of the recipe than the one in the
            repository, or the page would hold a local path; nothing is written then.
    """
    recipe = repo_root / record.source
    if not recipe.is_file() or script_sha256(recipe) != record.sha256:
        msg = (
            f"the record is of another version of {record.source} than the repository holds; "
            "run the recipe as it is and write its page from that run"
        )
        raise PageError(msg)
    body = _body(record, recipe=recipe, repo_root=repo_root, source_url=config.source_url)
    digests = page_digests(body, {}, origin=record)
    text = body + record_comment(_RecipePageRecord(provenance=record, page=digests))
    leaks = local_paths(text)
    if leaks:
        msg = f"{record.source}: a returned value holds a local path: {', '.join(sorted(leaks))}"
        raise PageError(msg)
    page = recipe_page_path(recipe, repo_root=repo_root)
    page.parent.mkdir(parents=True, exist_ok=True)
    write_utf8(page, text)
    return page


def check_recipe_page(recipe: Path, *, repo_root: Path) -> list[PageProblem]:
    """Check a recipe's committed page without running anything.

    Args:
        recipe: The recipe.
        repo_root: The repository root.

    Returns:
        Every problem found: no page, a page or recipe that is not UTF-8 or a page with no
        record, a page made from another version of the recipe, a page changed since it was
        written, or a local path in it.
    """
    page = recipe_page_path(recipe, repo_root=repo_root)
    if not page.exists():
        return [PageProblem(script=recipe, kind="missing", detail=f"no {page}")]
    try:
        text = read_utf8(page)
        record, unrecorded = read_page_record(text, page=page, record_type=_RecipePageRecord)
        current = script_sha256(recipe)
    except (NotUtf8Error, ValueError) as error:
        return [PageProblem(script=recipe, kind="unreadable", detail=str(error))]
    problems = []
    if record.provenance.sha256 != current:
        problems.append(
            PageProblem(script=recipe, kind="stale", detail="made from another version")
        )
    found = edited(unrecorded, record.page, origin=record.provenance, directory=page.parent)
    if found:
        problems.append(PageProblem(script=recipe, kind="edited", detail=", ".join(found)))
    leaks = local_paths(text)
    if leaks:
        problems.append(
            PageProblem(script=recipe, kind="local-path", detail=", ".join(sorted(leaks)))
        )
    return problems


def _body(record: RecipeRecord, *, recipe: Path, repo_root: Path, source_url: str) -> str:
    """The page's text before its record: title, header, the example's link, and the results."""
    example = PurePosixPath(paired_example(recipe, repo_root=repo_root).relative_to(repo_root))
    title = _title(read_utf8(recipe)) or f"Full recipe: {record.source}"
    return (
        f"# {title}\n\n{run_header(record, source_url=source_url, notebook=False)}\n\n"
        f"This page records a run of the full recipe of [`{example.as_posix()}`]"
        f"({example.with_suffix('.md').name}), at the published configuration it states.\n\n"
        f"## Results\n\n{_results(record.returned)}\n"
    )


def _title(text: str) -> str | None:
    """The first line of a recipe's module docstring, or ``None`` when it has none."""
    docstring = ast.get_docstring(ast.parse(text))
    return docstring.strip().splitlines()[0] if docstring and docstring.strip() else None


def _results(returned: JsonValue) -> str:
    """What ``main()`` returned: a table of a flat mapping, else the value as JSON."""
    if returned is None:
        return "`main()` returned nothing."
    if isinstance(returned, dict) and all(
        isinstance(value, str | int | float | bool) for value in returned.values()
    ):
        rows = "\n".join(f"| `{name}` | {value} |" for name, value in returned.items())
        return f"| Value | Result |\n| --- | --- |\n{rows}"
    return f"```json\n{json.dumps(returned, indent=2)}\n```"
