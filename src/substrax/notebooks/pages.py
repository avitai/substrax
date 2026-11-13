"""Make an example's docs page from a run of a clean copy of it, and check a committed page.

:func:`generate_page` runs the example with :func:`substrax.notebooks.run.execute_example` and
exports the executed copy with nbconvert's Markdown exporter to
``docs/examples/<the example's path>.md``, its figures beside it. Before writing, every known root
in the outputs (the repository, the environment, the temporary directory, the home directory)
becomes a placeholder where it starts a path, and a page that still holds a local path is refused.
The code, the header and the provenance are published as they are.
The page opens with the example's title cell, as MkDocs takes a page's title from an h1 only when
it is the page's first element. A generated header follows it: where and how long the run took,
and the command that runs the example. The page ends with the run's provenance. Values in the
outputs depend on the machine, library versions and load, so a fresh run is compared with the page
on what does not: the script it came from and the kinds and number of outputs per cell. Standard
error is shown on the page and left out of that comparison, since warnings differ between machines.
The provenance also records the sha256 of the page's text and of each figure as written, so a page
or figure changed by hand afterwards is found without running anything.

A page is held to the repository's page-size settings as Documenter.jl holds an HTML page
(v1.20.0, `src/html/HTMLWriter.jl` `write_html`): above the warning size it is written with a
warning, above the limit it is refused, and an exempt example's page is not measured.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
import shlex
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal
from urllib.parse import quote

import nbformat
from markdown_it import MarkdownIt
from nbconvert import MarkdownExporter

from substrax.examples import discover_examples, example_source
from substrax.notebooks.config import ExamplesConfig
from substrax.notebooks.outputs import local_paths, publish_outputs
from substrax.notebooks.run import ExampleRun, execute_example, Provenance, script_sha256
from substrax.records import dump_record, read_record, UNKNOWN_FIELDS_REFUSED


logger = logging.getLogger(__name__)

PYTHON_COMMAND = "uv run python"
"""How an example is run from the repository root, in a page's header and in a hand-written
command the corpus check reads."""

RUN_COMMAND = f"source ./activate.sh\n{PYTHON_COMMAND} {{source}}"
"""The command a page's header gives, from the repository root, for the example at ``source``,
quoted for the shell (a space, ``$`` or a quote in its name)."""

_MARKER = "substrax-notebooks"
_PROVENANCE = re.compile(rf"<!-- {_MARKER}: (?P<json>\{{.*?\}}) -->")


@dataclass(frozen=True, slots=True, kw_only=True)
class _PageDigests:
    """The sha256 of a page's text before its record, of its provenance, and of each figure."""

    body: str
    record: str
    figures: Mapping[str, str]


@dataclass(frozen=True, slots=True, kw_only=True)
class _PageRecord:
    """What closes a page: the run's provenance and the digests of the page as written.

    Its refusal of undeclared fields governs the nested provenance and digests too: pydantic
    applies a record's config to a nested dataclass that sets none of its own.
    """

    __pydantic_config__ = UNKNOWN_FIELDS_REFUSED

    provenance: Provenance
    page: _PageDigests


"""``{"body": <sha256>, "record": <sha256>, "figures": {<path beside the page>: <sha256>}}``."""
_LINE_ENDING = re.compile(r"\r\n?")
"""A Windows or a lone carriage-return line ending."""
_UNREADABLE_RECORD = ValueError
"""What reading a page's record raises when the record is missing or malformed."""
type ProblemKind = Literal[
    "missing",
    "unreadable",
    "stale",
    "edited",
    "missing-figure",
    "stray-figure",
    "figures-directory-taken",
    "too-large",
    "over-budget",
    "local-path",
    "differs-from-run",
]


class PageError(RuntimeError):
    """An example's page cannot be made.

    An output holds a local path, the page is too large, or the page's figures directory is where
    other examples' pages live.
    """


@dataclass(frozen=True, slots=True, kw_only=True)
class PageProblem:
    """One way an example's committed page fails to show a current, real run.

    Attributes:
        script: The example script.
        kind: What is wrong (see :data:`ProblemKind`).
        detail: The specifics.
    """

    script: Path
    kind: ProblemKind
    detail: str

    def __str__(self) -> str:
        """Return the problem as one line naming the script."""
        return f"{self.script}: {self.kind}: {self.detail}"


def page_path(script: Path, *, repo_root: Path) -> Path:
    """Return where an example's page lives: its path under ``examples/``, below ``docs/examples``.

    Args:
        script: The example script, under ``repo_root / "examples"``.
        repo_root: The repository root.

    Returns:
        ``docs/examples/<the example's path>.md`` under ``repo_root``.
    """
    source = PurePosixPath(example_source(script, repo_root=repo_root))
    return repo_root / "docs" / source.with_suffix(".md")


def generate_page(  # noqa: DOC503  # RunError is raised by execute_example
    script: Path, *, repo_root: Path, config: ExamplesConfig
) -> Path:
    """Run a clean copy of an example and write its page with every output of the run.

    Args:
        script: The example script, under ``repo_root / "examples"``.
        repo_root: The repository root; the run's working directory.
        config: The repository's example settings; the run has the example's budget.

    Returns:
        The page written.

    Raises:
        RunError: If the run fails (see :func:`~substrax.notebooks.run.execute_example`); nothing
            is written then.
        PageError: If the page's figures directory is where other examples' pages live (checked
            before the run), an output holds a local path, or the page is larger than
            ``config.page_size_bytes``; nothing is written then.
    """
    taken = _figures_directory_taken(script, repo_root=repo_root)
    if taken:
        msg = f"{script}: {taken}"
        raise PageError(msg)
    run = execute_example(
        script,
        repo_root=repo_root,
        distribution=config.distribution,
        budget_seconds=config.budget_for(example_source(script, repo_root=repo_root)),
    )
    header = _header(run.provenance, source_url=config.source_url)
    paged = _with_header(run.notebook, header)
    publish_outputs(paged, repo_root=repo_root, output_dir=run.output_dir)
    body, figures = _export(paged, stem=script.stem)
    digests = _page_digests(body, figures, provenance=run.provenance)
    text = body + _provenance_comment(run.provenance, digests)
    leaks = local_paths(text)
    if leaks:
        msg = f"{script}: an output holds a local path: {', '.join(sorted(leaks))}"
        raise PageError(msg)
    size = _measured_size(text, source=run.provenance.source, config=config)
    if size is not None and size > config.page_size_bytes:
        msg = f"{script}: {_over_the_limit(size, config)}"
        raise PageError(msg)
    if size is not None and size > config.page_size_warn_bytes:
        logger.warning(
            "%s: the page is %d bytes, over the %d bytes page_size_warn_bytes allows",
            script,
            size,
            config.page_size_warn_bytes,
        )
    page = page_path(script, repo_root=repo_root)
    page.parent.mkdir(parents=True, exist_ok=True)
    _remove_written_figures(page)
    page.write_text(text, encoding="utf-8")
    for name, data in figures.items():
        (page.parent / name).parent.mkdir(parents=True, exist_ok=True)
        (page.parent / name).write_bytes(data)
    return page


def read_provenance(page: Path) -> Provenance:  # noqa: DOC502  # ValueError is raised by _read_record
    """Read the provenance a page records.

    Args:
        page: A page :func:`generate_page` wrote.

    Returns:
        Its provenance.

    Raises:
        ValueError: If the page records none.
    """
    provenance, _, _ = _read_record(page.read_text(encoding="utf-8"), page=page)
    return provenance


def _read_record(text: str, *, page: Path) -> tuple[Provenance, _PageDigests, str]:
    """A page's provenance, the digests of the page as written, and its text without the record.

    The record is the page's last marker, as generation writes it after the page's text; an
    output that prints a marker of its own comes before it. The record's comment and the line
    endings around it are what generation added, so the text without them is what the body
    digest covers.

    Args:
        text: The page's text.
        page: The page, for the error message.

    Returns:
        The provenance, the page digests, and the text without the record.

    Raises:
        ValueError: If the page records no provenance, or a record that is not JSON or not the
            record a page holds: a field missing, undeclared or of the wrong type.
    """
    match = next(reversed(list(_PROVENANCE.finditer(text))), None)
    if match is None:
        msg = f"{page} records no {_MARKER} provenance"
        raise ValueError(msg)
    record = read_record(_PageRecord, json.loads(match.group("json")))
    unrecorded = text[: match.start()] + text[match.end() :].lstrip("\r\n")
    return record.provenance, record.page, unrecorded


def check_page(script: Path, *, repo_root: Path, config: ExamplesConfig) -> list[PageProblem]:
    """Check an example's committed page without running anything.

    Args:
        script: The example script.
        repo_root: The repository root.
        config: The repository's example settings, for the page-size limit and the budgets.

    Returns:
        Every problem found: no page, no provenance, a page made from another version of the
        script, a page or figure changed since it was written, a figure its record lists that
        does not exist, a figure file the record does not list, a figures directory where other
        examples' pages live, a page over the size limit, a recorded run longer than the
        example's budget, or a local path in it.
    """
    problems, _ = _check_page(script, repo_root=repo_root, config=config)
    return problems


def _check_page(
    script: Path, *, repo_root: Path, config: ExamplesConfig
) -> tuple[list[PageProblem], Provenance | None]:
    """:func:`check_page`'s problems, and the provenance the page records (``None`` if none)."""
    page = page_path(script, repo_root=repo_root)
    if not page.exists():
        return [PageProblem(script=script, kind="missing", detail=f"no {page}")], None
    text = page.read_text(encoding="utf-8")
    try:
        provenance, digests, unrecorded = _read_record(text, page=page)
    except _UNREADABLE_RECORD as error:
        return [PageProblem(script=script, kind="unreadable", detail=str(error))], None
    problems = []
    current = script_sha256(script)
    if provenance.sha256 != current:
        problems.append(
            PageProblem(script=script, kind="stale", detail="made from another version")
        )
    edited = _edited(unrecorded, digests, provenance=provenance, directory=page.parent)
    if edited:
        problems.append(PageProblem(script=script, kind="edited", detail=", ".join(edited)))
    size = _measured_size(text, source=provenance.source, config=config)
    if size is not None and size > config.page_size_bytes:
        detail = _over_the_limit(size, config)
        problems.append(PageProblem(script=script, kind="too-large", detail=detail))
    budget = config.budget_for(provenance.source)
    if provenance.seconds > budget:
        detail = (
            f"the page records a run of {provenance.seconds} s, over the {budget} s budget the "
            "settings give the example; make the run shorter or give it a budgets entry"
        )
        problems.append(PageProblem(script=script, kind="over-budget", detail=detail))
    problems.extend(_figure_problems(script, page, recorded=digests.figures))
    taken = _figures_directory_taken(script, repo_root=repo_root)
    if taken:
        problems.append(PageProblem(script=script, kind="figures-directory-taken", detail=taken))
    leaks = local_paths(text)
    if leaks:
        problems.append(
            PageProblem(script=script, kind="local-path", detail=", ".join(sorted(leaks)))
        )
    return problems, provenance


def compare_with_run(
    script: Path, run: ExampleRun, *, repo_root: Path, config: ExamplesConfig
) -> list[PageProblem]:
    """Compare what a run of the example produced with what its committed page records.

    Args:
        script: The example script.
        run: A run of the example as it is now, from
            :func:`~substrax.notebooks.run.execute_example`.
        repo_root: The repository root.
        config: The repository's example settings.

    Returns:
        The problems :func:`check_page` finds, and ``differs-from-run`` when the run's cells
        produce other kinds or numbers of output than the page records; empty when the page
        shows the run the script makes now.
    """
    problems, provenance = _check_page(script, repo_root=repo_root, config=config)
    if provenance is None:
        return problems
    recorded = provenance.outputs
    if recorded != run.provenance.outputs:
        detail = f"page records {recorded}, the run produced {run.provenance.outputs}"
        problems.append(PageProblem(script=script, kind="differs-from-run", detail=detail))
    return problems


def _measured_size(text: str, *, source: str, config: ExamplesConfig) -> int | None:
    """A page's size as the limits measure it: its UTF-8 bytes, figures not counted.

    Args:
        text: The page's text.
        source: The example, as its provenance names it, for the exemption list.
        config: The repository's example settings.

    Returns:
        The size, or ``None`` when the example is in ``config.page_size_exempt``.
    """
    return None if source in config.page_size_exempt else len(text.encode("utf-8"))


def _over_the_limit(size: int, config: ExamplesConfig) -> str:
    """The problem with a page of ``size`` bytes over ``config.page_size_bytes``."""
    return (
        f"the page is {size} bytes, over the {config.page_size_bytes} bytes the settings allow; "
        "print less, or list the example in page_size_exempt"
    )


def _figure_problems(script: Path, page: Path, *, recorded: Mapping[str, str]) -> list[PageProblem]:
    """A figure the page's record lists that does not exist, and figure files it does not list.

    The record names exactly the figures the exporter wrote, by their path beside the page; image
    syntax in the page's text is not read, since an output or a fenced block may print it as text.
    An image an example links from its own prose is a link MkDocs checks (``mkdocs build
    --strict`` fails on a missing one).
    """
    problems = [
        PageProblem(script=script, kind="missing-figure", detail=name)
        for name in sorted(recorded)
        if not (page.parent / name).is_file()
    ]
    figures = figures_directory(page)
    stray = sorted(
        path.relative_to(page.parent).as_posix()
        for path in (figures.rglob("*") if figures.is_dir() else ())
        if path.is_file() and path.relative_to(page.parent).as_posix() not in recorded
    )
    if stray:
        problems.append(PageProblem(script=script, kind="stray-figure", detail=", ".join(stray)))
    return problems


def _remove_written_figures(page: Path) -> None:
    """Remove the figures the page's last generation wrote, before the page is written again.

    A figure an earlier run drew and this one does not must not stay beside the page. Only a file
    inside the page's figures directory whose bytes still match the sha256 its record holds is
    removed: the record is page text that a hand edit may change, and the directory may hold
    another example's page (``examples/data.py`` beside ``examples/data_files/``) or a file put
    there by hand, which ``check_page`` then reports as ``stray-figure``. The directory goes once
    nothing is left in it.
    """
    figures = figures_directory(page)
    if not (page.is_file() and figures.is_dir()):
        return
    try:
        _, digests, _ = _read_record(page.read_text(encoding="utf-8"), page=page)
    except _UNREADABLE_RECORD:
        return  # a page with no readable record names nothing it wrote
    inside = figures.resolve()
    for name, digest in digests.figures.items():
        figure = (page.parent / name).resolve()
        if (
            figure.is_relative_to(inside)
            and figure.is_file()
            and hashlib.sha256(figure.read_bytes()).hexdigest() == digest
        ):
            figure.unlink()
    if not any(figures.iterdir()):
        figures.rmdir()


def _figures_directory_taken(script: Path, *, repo_root: Path) -> str | None:
    """Why the example's figures directory cannot be its own, or ``None`` when it can.

    The figures of ``examples/<path>.py`` live in ``docs/examples/<path>_files/``, which is the
    page directory of every example under ``examples/<path>_files/``: the two would share one
    directory, and the page could never check clean.
    """
    figures = figures_directory(page_path(script, repo_root=repo_root))
    examples = (repo_root / "examples").resolve()
    directory = examples / figures.relative_to(repo_root / "docs" / "examples")
    inside = [
        path
        for path in (discover_examples(examples) if directory.is_dir() else [])
        if path.is_relative_to(directory)
    ]
    if not inside:
        return None
    names = ", ".join(path.relative_to(examples.parent).as_posix() for path in inside)
    return (
        f"its figures directory, {figures.relative_to(repo_root).as_posix()}, is the page "
        f"directory of {names}; rename the example or that directory"
    )


def figures_directory(page: Path) -> Path:
    """Where a page's figures live: ``<stem>_files`` beside it, as :func:`_export` names it."""
    return page.parent / f"{page.stem}_files"


def _export(notebook: nbformat.NotebookNode, *, stem: str) -> tuple[str, dict[str, bytes]]:
    r"""The executed notebook as Markdown, and its figures by their path beside the page.

    The Markdown's line endings are made ``\n``: an output can carry a lone ``\r`` (a displayed
    Markdown string), which CommonMark reads as a line ending and Python's universal newlines read
    back as ``\n``, so a page digested with it would not match the page read from disk.
    """
    resources = {"output_files_dir": f"{stem}_files", "unique_key": stem}
    body, written = MarkdownExporter().from_notebook_node(notebook, resources=resources)
    return _LINE_ENDING.sub("\n", body), dict(written.get("outputs", {}))


def _with_header(notebook: nbformat.NotebookNode, header: str) -> nbformat.NotebookNode:
    """A copy of the executed notebook with the generated header right after its title.

    The first markdown cell holds the title and, in a conforming example, the metadata table and
    the Overview after it, so the cell is split after its leading level-one heading, written with
    ``#`` or underlined, and the header goes between. A first cell that does not open with one is
    left whole, and the header goes before it.
    """
    paged = copy.deepcopy(notebook)
    header_cell = nbformat.v4.new_markdown_cell(header)
    first = paged.cells[0] if paged.cells else None
    end = _title_end(first.source) if first is not None and first.cell_type == "markdown" else None
    if first is None or end is None:
        paged.cells.insert(0, header_cell)
        return paged
    lines = first.source.splitlines(keepends=True)
    rest = "".join(lines[end:])
    first.source = "".join(lines[:end]).strip("\n")
    after = [header_cell]
    if rest.strip():
        after.append(nbformat.v4.new_markdown_cell(rest.strip("\n")))
    paged.cells[1:1] = after
    return paged


def _title_end(markdown: str) -> int | None:
    """The line after a Markdown text's leading level-one heading, or ``None`` when it has none."""
    blocks = [token for token in MarkdownIt("commonmark").parse(markdown) if token.level == 0]
    first = next((token for token in blocks if token.nesting >= 0), None)
    if first is None or first.type != "heading_open" or first.tag != "h1" or first.map is None:
        return None
    return first.map[1]


def _header(provenance: Provenance, *, source_url: str) -> str:
    """Where and how long the run took, links to the script and notebook, and the run command."""
    versions = ", ".join(f"{name} {version}" for name, version in provenance.versions.items())
    command = RUN_COMMAND.format(source=shlex.quote(provenance.source))
    script_url = f"{source_url}/{quote(provenance.source)}"
    notebook_url = script_url.removesuffix(".py") + ".ipynb"
    return (
        f"> Ran in {provenance.seconds} s on {provenance.device} ({versions}), from "
        f"[`{provenance.source}`]({script_url}) ([notebook]({notebook_url})). "
        f"Run it from the repository root:\n\n```bash\n{command}\n```"
    )


def _provenance_comment(provenance: Provenance, digests: _PageDigests) -> str:
    """The comment :func:`read_provenance` reads, closing the page: provenance and page digests."""
    record = dump_record(_PageRecord(provenance=provenance, page=digests))
    return f"\n<!-- {_MARKER}: {json.dumps(record, sort_keys=True)} -->\n"


def _page_digests(body: str, figures: dict[str, bytes], *, provenance: Provenance) -> _PageDigests:
    """The sha256 of the page's text before its record, of the record, and of each figure."""
    return _PageDigests(
        body=_text_sha256(body + "\n"),
        record=_record_sha256(provenance),
        figures={name: hashlib.sha256(data).hexdigest() for name, data in figures.items()},
    )


def _edited(
    unrecorded: str, digests: _PageDigests, *, provenance: Provenance, directory: Path
) -> list[str]:
    """What differs from the page as written: its text, its record, or a figure still present.

    ``unrecorded`` is the page's text without its provenance record. The record is checked as
    the provenance it holds, so an edited script hash, source or output list is found; a figure
    that is gone is reported as missing instead.
    """
    edited = []
    if _text_sha256(unrecorded) != digests.body:
        edited.append("the page's text")
    if _record_sha256(provenance) != digests.record:
        edited.append("the provenance record")
    for name, digest in sorted(digests.figures.items()):
        figure = directory / name
        if figure.is_file() and hashlib.sha256(figure.read_bytes()).hexdigest() != digest:
            edited.append(name)
    return edited


def _record_sha256(provenance: Provenance) -> str:
    """The sha256 of a provenance as the record writes it: ``dump_record``'s JSON, keys sorted."""
    return hashlib.sha256(json.dumps(dump_record(provenance), sort_keys=True).encode()).hexdigest()


def _text_sha256(text: str) -> str:
    """The sha256 of a text's UTF-8 bytes, Windows line endings made Unix ones as a checkout may."""
    return hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()
