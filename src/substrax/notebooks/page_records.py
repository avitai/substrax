"""The record that closes a generated page, and the header that opens it.

A generated page (an example's, or a recipe's) opens with a header naming where and how long its
run took and the command that runs the script, and closes with a record: a comment holding the
run's origin (:class:`~substrax.notebooks.run.RunOrigin`) and the sha256 of the page's text, of
that origin and of each figure as written, so a page or figure changed by hand afterwards is found
without running anything. The record's key for the origin is ``provenance``, the name every
committed page holds.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from substrax.notebooks.run import RunOrigin
from substrax.notebooks.text import text_sha256
from substrax.records import dump_record, read_record


PYTHON_COMMAND = "uv run python"
"""How a script is run from the repository root, in a page's header."""

RUN_COMMAND = f"source ./activate.sh\n{PYTHON_COMMAND} {{source}}"
"""The command a page's header gives, from the repository root, for the script at ``source``,
quoted for the shell (a space, ``$`` or a quote in its name)."""

MARKER = "substrax-notebooks"
"""What opens a page's record comment."""

_RECORD = re.compile(rf"<!-- {MARKER}: (?P<json>\{{.*?\}}) -->")


@dataclass(frozen=True, slots=True, kw_only=True)
class PageDigests:
    """The sha256 of a page's text before its record, of its origin, and of each figure.

    Attributes:
        body: The page's text before its record.
        record: The origin, as :func:`origin_sha256` gives it.
        figures: Each figure, by its path beside the page.
    """

    body: str
    record: str
    figures: Mapping[str, str]


def page_digests(body: str, figures: Mapping[str, bytes], *, origin: RunOrigin) -> PageDigests:
    """The digests a page records as it is written.

    Args:
        body: The page's text before its record.
        figures: Each figure's bytes, by its path beside the page.
        origin: The run the page shows.

    Returns:
        The digests.
    """
    return PageDigests(
        body=text_sha256(body + "\n"),
        record=origin_sha256(origin),
        figures={name: hashlib.sha256(data).hexdigest() for name, data in figures.items()},
    )


def record_comment(record: object) -> str:
    """The comment that closes a page: its record, as ``dump_record`` writes it, keys sorted.

    Args:
        record: The page's record, a dataclass with ``provenance`` and ``page`` fields.

    Returns:
        The comment, on a line of its own.
    """
    return f"\n<!-- {MARKER}: {json.dumps(dump_record(record), sort_keys=True)} -->\n"


def read_page_record[T](  # noqa: DOC503  # pydantic.ValidationError, a ValueError, is raised by read_record
    text: str, *, page: Path, record_type: type[T]
) -> tuple[T, str]:
    """A page's record, and its text without the record.

    The record is the page's last marker, as generation writes it after the page's text; an
    output that prints a marker of its own comes before it. The record's comment and the line
    endings around it are what generation added, so the text without them is what the body
    digest covers.

    Args:
        text: The page's text, as :func:`~substrax.notebooks.text.read_utf8` reads it.
        page: The page, for the error message.
        record_type: The record's dataclass.

    Returns:
        The record, and the text without it.

    Raises:
        ValueError: If the page records nothing, or a record that is not JSON or not of
            ``record_type``: a field missing, undeclared or of the wrong type.
    """
    match = next(reversed(list(_RECORD.finditer(text))), None)
    if match is None:
        msg = f"{page} records no {MARKER} provenance"
        raise ValueError(msg)
    record = read_record(record_type, json.loads(match.group("json")))
    return record, text[: match.start()] + text[match.end() :].lstrip("\n")


def edited(
    unrecorded: str, digests: PageDigests, *, origin: RunOrigin, directory: Path
) -> list[str]:
    """What differs from the page as written: its text, its record, or a figure still present.

    ``unrecorded`` is the page's text without its record. The record is checked as the origin it
    holds, so an edited script hash, source or output list is found; a figure that is gone is
    left to the caller to report as missing.

    Args:
        unrecorded: The page's text without its record.
        digests: The digests the page recorded.
        origin: The origin the page recorded.
        directory: The page's directory, which its figure paths are relative to.

    Returns:
        What was edited, by name; empty when nothing was.
    """
    found = []
    if text_sha256(unrecorded) != digests.body:
        found.append("the page's text")
    if origin_sha256(origin) != digests.record:
        found.append("the provenance record")
    for name, digest in sorted(digests.figures.items()):
        figure = directory / name
        if figure.is_file() and hashlib.sha256(figure.read_bytes()).hexdigest() != digest:
            found.append(name)
    return found


def origin_sha256(origin: RunOrigin) -> str:
    """Return the sha256 of a run's origin as the record writes it: its JSON, keys sorted.

    Args:
        origin: The origin.

    Returns:
        The hex digest.
    """
    return hashlib.sha256(json.dumps(dump_record(origin), sort_keys=True).encode()).hexdigest()


def run_header(origin: RunOrigin, *, source_url: str, notebook: bool) -> str:
    """Where and how long the run took, a link to the script, and the command that runs it.

    Args:
        origin: The run.
        source_url: Where the repository's files are browsed, ``[tool.substrax.examples]``'s.
        notebook: Whether the script has a notebook beside it to link to, as an example has.

    Returns:
        The header, as Markdown.
    """
    versions = ", ".join(f"{name} {version}" for name, version in origin.versions.items())
    command = RUN_COMMAND.format(source=shlex.quote(origin.source))
    script_url = f"{source_url}/{quote(origin.source)}"
    links = f"[`{origin.source}`]({script_url})"
    if notebook:
        links += f" ([notebook]({script_url.removesuffix('.py')}.ipynb))"
    return (
        f"> Ran in {origin.seconds} s on {origin.device} ({versions}), from {links}. "
        f"Run it from the repository root:\n\n```bash\n{command}\n```"
    )
