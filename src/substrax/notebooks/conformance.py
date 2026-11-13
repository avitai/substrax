"""The shared example guide's mechanical rules, checked on one example.

A repository's shared example tests (:mod:`substrax.testing.example_suite`) apply
:func:`check_example` to each example (:func:`substrax.examples.discover_examples`), and the
``example-conformance`` pre-commit hook applies :data:`SOURCE_RULES` to the examples a commit
changes. Each rule in :data:`RULES` has the id the guide (``docs/guides/writing-examples.md``)
names it by, so every rule the guide marks as checked has its assertion here. Rules about prose
(clarity, which figures help) are the guide's alone and reviewed by people.

The rules read the example's cells as jupytext reads them and its syntax tree with the markdown
cells (top-level string expressions) left out, so a path or a name in prose is never taken for
code. Installed with the ``notebooks`` extra; the package does not import this module.
"""

from __future__ import annotations

import ast
import codecs
import io
import itertools
import re
import tokenize
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import TypeGuard

import jupytext
from markdown_it import MarkdownIt

from substrax.examples import example_source
from substrax.notebooks.config import ExamplesConfig
from substrax.notebooks.environment_writes import environment_writes, OS_MUTATORS
from substrax.notebooks.kernel_setup import is_guarded_main_call
from substrax.notebooks.pages import check_page
from substrax.notebooks.pairs import check_pair
from substrax.notebooks.text import NotUtf8Error, read_utf8


class Tier(StrEnum):
    """The values of the Tier field."""

    QUICK_REFERENCE = "Quick Reference"
    TUTORIAL = "Tutorial"
    ADVANCED_GUIDE = "Advanced Guide"


class Devices(StrEnum):
    """The values of the Devices field: where an example runs."""

    CPU_COMPATIBLE = "CPU-compatible"
    GPU_OPTIONAL = "GPU-optional"
    GPU_REQUIRED = "GPU-required"


METADATA_FIELDS = ("Tier", "Devices", "Prerequisites")
"""The metadata fields an example writes, in this order; nothing else is authored."""

SECTIONS = (
    "Overview",
    "Learning Goals",
    "Setup",
    "Core Concepts?",
    "Implementation",
    "Results",
    "Troubleshooting?",
    "Next Steps",
)
"""The second-level sections, in order; a trailing ``?`` marks an optional one."""

_SECTION_LEVEL = 2
"""The heading level of the guide's sections (``##``); ``#`` is the title."""

_FIELD_VALUES: dict[str, type[StrEnum]] = {"Tier": Tier, "Devices": Devices}
_REQUIRE_GPU = frozenset({"substrax.devices.require_gpu", "substrax.devices.info.require_gpu"})
"""``require_gpu``'s qualified names: through the package exporting it, and its own module."""
_TIER_NAME = re.compile(rf"\b(?:{'|'.join(re.escape(tier) for tier in Tier)})\b", re.IGNORECASE)
"""A tier's name as words of a title; the metadata table says the tier."""
_NAMES = [section.rstrip("?") for section in SECTIONS]
_REQUIRED = {section for section in SECTIONS if not section.endswith("?")}
_MARKDOWN = MarkdownIt("commonmark")
"""The CommonMark reader the outline is built from, as the page generator reads a page."""
_CODE_TOKENS = ("fence", "code_block")
"""The tokens holding code: a fence of backticks or tildes, and an indented block."""
_MARKDOWN_MARKER = re.compile(r"^#\s*%%.*\[(?:markdown|md)\]")
"""jupytext's percent-format marker of a markdown cell, by its name or its ``md`` alias."""
_CELL_MARKER = re.compile(r"^\s*#\s*%%")
"""A line jupytext's percent format may read as a cell marker (``cell_reader.py``, 1.19)."""
_PLAIN_MARKER_METADATA = frozenset({"title", "region_name"})
"""What a marker with no option leaves in its cell's metadata: a title, and the ``md`` name."""
_TRIPLE_QUOTED = re.compile(r"^[rRuU]?(?:\"\"\"|''')")
"""A triple-quoted string, raw or not, opening a markdown cell's body."""
_STRING = re.compile(r"^[rRuU]?[\"']")
_TABLE_HEADER = re.compile(r"^\|\s*Metadata\s*\|\s*Value\s*\|\s*$")
_TABLE_DELIMITER = re.compile(r"^\|\s*:?-{3,}:?\s*\|\s*:?-{3,}:?\s*\|\s*$")
_ROW = re.compile(r"^\|\s*\*\*(?P<field>[^*|]+)\*\*\s*\|\s*(?P<value>[^|]*?)\s*\|\s*$")
_STEP = re.compile(r"^Step (?P<number>\d+): \S")
_SNAKE_CASE = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*")
"""A snake_case name: lowercase words of letters and digits joined by single underscores, opening
with a letter, so a number prefix (``01_intro``) is not one."""
_NUMBERED = re.compile(r"^\d+\.\s+\S")
_BULLET = re.compile(r"^[-*+]\s+\S")
_FORCING = (
    "JAX_PLATFORMS",
    "jax_platforms",
    "jax_platform_name",
    "jax_default_device",
    "CUDA_VISIBLE_DEVICES",
)
_FORCING_INSTRUCTION = re.compile(
    r"\b(JAX_PLATFORMS|CUDA_VISIBLE_DEVICES)="
    r"|\bunset\s+(JAX_PLATFORMS|CUDA_VISIBLE_DEVICES)\b"
    r"|[\"'](jax_platform_name|jax_platforms)[\"']"
)
"""Prose that sets or clears the backend: an assignment, an unset, a config key."""
_CONFIG_UPDATE = "jax.config.update"
"""How code sets one of jax's settings, the backend's among them."""
_LOADS_DOTENV = "dotenv.load_dotenv"
"""python-dotenv's function that writes a ``.env`` file's variables into the environment."""
_SETS_NONE = "it reads its settings from the environment and sets none"
_ARGUMENT_PARSERS = ("argparse", "optparse", "getopt", "click", "typer", "fire", "docopt")
"""The packages whose job is reading command-line arguments."""
_EXPECTED_OUTPUT = re.compile(
    r"^[#\s]*[^\w#\s]*\s*(?:(?:expected|sample|example)\s+output\b|expected\s*[:(])",
    re.IGNORECASE,
)
"""A line that opens by stating output: Expected, Sample or Example output, or Expected followed by
``:`` or ``(``; the ``#`` of a comment or a heading, and a symbol before the words, are skipped. A
bare ``Output`` names what code returns, a shape or a figure, and states no run's output."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Finding:
    """One rule an example breaks.

    Attributes:
        rule: The rule's id, as the guide names it.
        script: The example script.
        detail: What is wrong.
    """

    rule: str
    script: Path
    detail: str

    def __str__(self) -> str:
        """Return the finding as one line naming the script and the rule."""
        return f"{self.script}: {self.rule}: {self.detail}"


@dataclass(frozen=True, slots=True)
class _Outline:
    headings: list[tuple[int, str]]
    sections: dict[str, list[str]]


@dataclass(frozen=True, slots=True)
class _Example:
    script: Path
    repo_root: Path
    config: ExamplesConfig
    cells: list[tuple[str, str]]
    outline: _Outline
    tree: ast.Module | None
    source_problems: list[str]

    @property
    def code_nodes(self) -> list[ast.AST]:
        if self.tree is None:
            return []
        statements = [s for s in self.tree.body if not _is_prose(s)]
        return [node for statement in statements for node in ast.walk(statement)]


@dataclass(frozen=True, slots=True)
class Rule:
    """A mechanical rule of the example guide.

    Attributes:
        id: The id the guide names the rule by.
        title: The rule in one line.
        check: Returns a detail for each way the example breaks the rule.
    """

    id: str
    title: str
    check: Callable[[_Example], list[str]]


def check_example(
    script: Path,
    *,
    repo_root: Path,
    config: ExamplesConfig,
    rules: Iterable[str] | None = None,
) -> list[Finding]:
    """Check one example against the guide's rules.

    Args:
        script: The example script, under ``repo_root / "examples"``.
        repo_root: The repository root.
        config: The repository's example settings, which the ``page`` rule holds the page to;
            the caller reads them once, with
            :func:`~substrax.notebooks.config.read_examples_config`.
        rules: The ids of the rules to check; every rule in :data:`RULES` when ``None``.

    Returns:
        Every finding, in rule order; empty when the example follows every rule checked.

    Raises:
        KeyError: If a rule id is unknown.
    """
    chosen = set(RULES) if rules is None else set(rules)
    unknown = sorted(chosen - set(RULES))
    if unknown:
        msg = f"unknown rule {unknown[0]!r}; the rules are {', '.join(RULES)}"
        raise KeyError(msg)
    example = _read(script, repo_root=repo_root, config=config)
    return [
        Finding(rule=rule_id, script=script, detail=detail)
        for rule_id, rule in RULES.items()
        if rule_id in chosen
        for detail in rule.check(example)
    ]


def example_metadata(script: Path) -> dict[str, str]:
    """Read the metadata table of an example's first markdown cell.

    Args:
        script: The example script.

    Returns:
        Each ``| **Field** | value |`` row, field to value, in order; empty when there is none,
        as for a file that is not UTF-8, which the ``parses`` rule reports.
    """
    try:
        text = read_utf8(script)
    except NotUtf8Error:
        return {}
    return _metadata_table(_cells(text))


def _cells(text: str) -> list[tuple[str, str]]:
    """The kind and source of each cell of the example that holds anything."""
    return [
        (cell.cell_type, cell.source)
        for cell in jupytext.reads(text, fmt="py:percent").cells
        if cell.source.strip()
    ]


def _read(script: Path, *, repo_root: Path, config: ExamplesConfig) -> _Example:
    """The example's cells, outline and syntax tree, and what keeps its source from being read.

    A file that is not UTF-8 has no cells; a byte order mark, which Python accepts, is a problem of
    its own and is dropped before the other rules read the file.
    """
    try:
        text = read_utf8(script)
    except NotUtf8Error as error:
        return _Example(script, repo_root, config, [], _outline([]), None, [str(error)])
    problems = []
    if script.read_bytes().startswith(codecs.BOM_UTF8):
        problems.append("the file opens with a byte order mark; save it as UTF-8 without one")
    problems.extend(_markdown_form_problems(text))
    problems.extend(_cell_option_problems(text))
    cells = _cells(text)
    try:
        tree = ast.parse(text, filename=str(script))
    except SyntaxError as error:
        tree = None
        problems.append(
            f"line {error.lineno}: {error.msg} (a docstring may close a markdown cell early)"
        )
    outline = _outline([source for kind, source in cells if kind == "markdown"])
    return _Example(script, repo_root, config, cells, outline, tree, problems)


def _markdown_form_problems(text: str) -> list[str]:
    """Each markdown cell that is not written as a triple-quoted string, by its marker's line."""
    lines = text.splitlines()
    problems = []
    for number, line in enumerate(lines, start=1):
        if not _MARKDOWN_MARKER.match(line):
            continue
        body = next((later.strip() for later in lines[number:] if later.strip()), "")
        if not _TRIPLE_QUOTED.match(body):
            form = "a one-line string" if _STRING.match(body) else "comments"
            problems.append(
                f"line {number}: a markdown cell is written as {form}; write it as a "
                "triple-quoted string"
            )
    return problems


def _cell_option_problems(text: str) -> list[str]:
    """Each cell marker carrying an option, by its line, as jupytext reads the marker alone.

    An option (a tag such as ``skip-execution``, ``active``, a raw cell) changes what the notebook
    run executes and never what ``python <script>`` executes, so a marker carries none; its title,
    and the ``md`` name of a markdown cell, change nothing that runs.
    """
    problems = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not _CELL_MARKER.match(line):
            continue
        (cell,) = jupytext.reads(f"{line}\n", fmt="py:percent").cells
        options = sorted(set(cell.metadata) - _PLAIN_MARKER_METADATA)
        if cell.cell_type == "raw" or options:
            carried = ", ".join(options) if options else "a raw cell"
            problems.append(
                f"line {number}: the cell marker carries {carried}, which the notebook run obeys "
                "and python <script> does not; a marker carries a title at most"
            )
    return problems


def _is_prose(statement: ast.stmt) -> bool:
    """A top-level string expression: a markdown cell, or the module docstring."""
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, str)
    )


def _outline(markdown: list[str]) -> _Outline:
    """Every heading outside code, and the lines of each ``##`` section."""
    lines = list(_markdown_lines(markdown))
    headings = [line.heading for line in lines if line.heading is not None]
    return _Outline(headings=headings, sections=_sections(lines))


@dataclass(frozen=True, slots=True)
class _MarkdownLine:
    """One line of a markdown cell, read as CommonMark reads it."""

    text: str
    code: bool
    heading: tuple[int, str] | None


def _sections(lines: list[_MarkdownLine]) -> dict[str, list[str]]:
    """The lines under each ``##`` heading, up to the next heading of level 2 or above.

    A list or heading inside fenced code is an example of one, not the section's own, so fenced
    lines are left out.
    """
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in lines:
        heading = line.heading
        if heading is not None and heading[0] <= _SECTION_LEVEL:
            current = heading[1] if heading[0] == _SECTION_LEVEL else None
            if current is not None:
                sections.setdefault(current, [])
        elif current is not None and not line.code:
            sections[current].append(line.text)
    return sections


def _markdown_lines(markdown: list[str]) -> Iterator[_MarkdownLine]:
    """Each markdown line, with whether CommonMark reads it as code and the heading it opens."""
    for text in markdown:
        code, headings = _line_roles(text)
        for number, line in enumerate(text.splitlines()):
            yield _MarkdownLine(text=line, code=number in code, heading=headings.get(number))


def _line_roles(text: str) -> tuple[set[int], dict[int, tuple[int, str]]]:
    """The code lines of ``text`` and each heading's line: its level and text, by line number."""
    tokens = _MARKDOWN.parse(text)
    code = {
        number
        for token in tokens
        if token.type in _CODE_TOKENS and token.map is not None
        for number in range(*token.map)
    }
    headings = {
        token.map[0]: (int(token.tag[1:]), tokens[index + 1].content)
        for index, token in enumerate(tokens)
        if token.type == "heading_open" and token.map is not None
    }
    return code, headings


def _code_cells(example: _Example) -> list[tuple[str | None, str]]:
    """Each code cell with the ``##`` section it is in: the last opened before it, else ``None``."""
    cells: list[tuple[str | None, str]] = []
    current: str | None = None
    for kind, source in example.cells:
        if kind != "markdown":
            cells.append((current, source))
            continue
        for line in _markdown_lines([source]):
            heading = line.heading
            if heading is not None and heading[0] <= _SECTION_LEVEL:
                current = heading[1] if heading[0] == _SECTION_LEVEL else None
    return cells


def _code_by_section(example: _Example) -> dict[str, list[str]]:
    """Each ``##`` section's code cells, in order."""
    sections: dict[str, list[str]] = {}
    for section, source in _code_cells(example):
        if section is not None:
            sections.setdefault(section, []).append(source)
    return sections


def _subheadings(example: _Example, section: str) -> list[str]:
    """The ``###`` headings of a ``##`` section, outside code."""
    lines = "\n".join(example.outline.sections.get(section, []))
    return [
        line.heading[1]
        for line in _markdown_lines([lines])
        if line.heading is not None and line.heading[0] == _SECTION_LEVEL + 1
    ]


def _outlined(example: _Example) -> bool:
    """Whether the sections are the guide's, in its order: only then is each cell's section its own.

    A missing or misplaced section is the skeleton rule's one finding, not one per section rule.
    """
    return not _skeleton(example)


def _setup(example: _Example) -> list[str]:
    if not _outlined(example):
        return []
    setup = _code_by_section(example).get("Setup", [])
    problems = (
        [] if len(setup) <= 1 else [f"Setup holds {len(setup)} code cells; its imports are one"]
    )
    return problems + [
        f"line {statement.lineno} of a cell in {section or 'no section'} imports; "
        "every import is in Setup"
        for section, source in _code_cells(example)
        if section != "Setup"
        for statement in _module_statements(source)
        if isinstance(statement, ast.Import | ast.ImportFrom)
    ]


def _module_statements(source: str) -> list[ast.stmt]:
    """A code cell's top-level statements; none for a cell that does not parse (``parses``)."""
    try:
        return ast.parse(source).body
    except SyntaxError:
        return []


def _results(example: _Example) -> list[str]:
    if not _outlined(example) or _subheadings(example, "Results"):
        return []
    return ["Results has no ### title; what the run shows goes under concrete ones"]


def _troubleshooting(example: _Example) -> list[str]:
    absent = "Troubleshooting" not in example.outline.sections
    if not _outlined(example) or absent or _subheadings(example, "Troubleshooting"):
        return []
    return ["Troubleshooting has no ### entry; each symptom or error is one"]


def _next_steps(example: _Example) -> list[str]:
    if not _outlined(example):
        return []
    return [
        f"Next Steps has ### {title}; its only subsection is Experiments to Try"
        for title in _subheadings(example, "Next Steps")
        if title != "Experiments to Try"
    ]


def _path_names(example: _Example) -> list[str]:
    source = PurePosixPath(example_source(example.script, repo_root=example.repo_root))
    relative = source.relative_to("examples")
    parts = [*relative.parent.parts, relative.stem]
    return [
        f"{part!r} is not a snake_case name with no number prefix"
        for part in parts
        if not _SNAKE_CASE.fullmatch(part)
    ]


def _pair(example: _Example) -> list[str]:
    return [f"{problem.kind}: {problem.detail}" for problem in check_pair(example.script)]


def _parses(example: _Example) -> list[str]:
    return list(example.source_problems)


def _title(example: _Example) -> list[str]:
    opens = bool(example.cells) and example.cells[0][0] == "markdown"
    if not opens or not example.cells[0][1].lstrip().startswith("# "):
        return ["the example does not open with a markdown cell whose first line is its # title"]
    titles = [text for level, text in example.outline.headings if level == 1]
    problems = (
        [] if len(titles) == 1 else [f"{len(titles)} level-1 headings; the title is the only one"]
    )
    return problems + [
        f"the title {title!r} names the tier; it names the subject, and the table gives the tier"
        for title in titles[:1]
        if _TIER_NAME.search(title)
    ]


def _metadata(example: _Example) -> list[str]:
    fields = _metadata_table(example.cells)
    problems = [
        f"{name} is {fields.get(name)!r}; one of {[str(value) for value in allowed]}"
        for name, allowed in _FIELD_VALUES.items()
        if fields.get(name) not in allowed
    ]
    if tuple(fields) != METADATA_FIELDS:
        problems.insert(
            0, f"metadata fields are {list(fields)}; the table holds {list(METADATA_FIELDS)}"
        )
    if not fields.get("Prerequisites"):
        problems.append("Prerequisites is empty")
    return problems + _table_placement(example)


def _table_placement(example: _Example) -> list[str]:
    """Whether the table directly follows the title, opening with its header and delimiter rows."""
    lines = [line.strip() for line in _first_markdown(example.cells).strip().splitlines()]
    header, delimiter, *_ = [*(line for line in lines[1:] if line), "", ""]
    if _TABLE_HEADER.match(header) and _TABLE_DELIMITER.match(delimiter):
        return []
    return [
        "the metadata table directly follows the title, opening with `| Metadata | Value |` "
        "and its delimiter row"
    ]


def _metadata_table(cells: list[tuple[str, str]]) -> dict[str, str]:
    """The ``| **Field** | value |`` rows of the first markdown cell, in order."""
    rows = (_ROW.match(line) for line in _first_markdown(cells).splitlines())
    return {row.group("field").strip(): row.group("value").strip() for row in rows if row}


def _first_markdown(cells: list[tuple[str, str]]) -> str:
    """The first markdown cell, holding the title and the metadata table; ``""`` if none."""
    return next((text for kind, text in cells if kind == "markdown"), "")


def _skeleton(example: _Example) -> list[str]:
    found = [text for level, text in example.outline.headings if level == _SECTION_LEVEL]
    if _in_guide_order(found) and set(found) >= _REQUIRED:
        return []
    return [f"sections are {found}; the order is {list(SECTIONS)} (? optional)"]


def _in_guide_order(found: list[str]) -> bool:
    """Whether every section is one of the guide's, each at most once, in its order."""
    if any(section not in _NAMES for section in found):
        return False
    positions = [_NAMES.index(section) for section in found]
    return all(a < b for a, b in itertools.pairwise(positions))


def _goals(example: _Example) -> list[str]:
    # A missing section is the skeleton rule's finding; this rule judges the list in it.
    if "Learning Goals" not in example.outline.sections:
        return []
    items = [line.strip() for line in example.outline.sections["Learning Goals"]]
    if not any(_NUMBERED.match(line) for line in items):
        return ["Learning Goals has no numbered list"]
    if any(_BULLET.match(line) for line in items):
        return ["Learning Goals mixes bullets with its numbered list"]
    return []


def _steps(example: _Example) -> list[str]:
    if "Implementation" not in example.outline.sections:
        return []
    steps = _subheadings(example, "Implementation")
    if not steps:
        return ["Implementation has no '### Step 1: <action>'"]
    return [
        f"'### {step}' where 'Step {expected}: <action>' is due"
        for expected, step in enumerate(steps, start=1)
        if not _is_step(step, expected)
    ]


def _is_step(heading: str, number: int) -> bool:
    match = _STEP.match(heading)
    return match is not None and int(match.group("number")) == number


def _main(example: _Example) -> list[str]:
    if example.tree is None:
        return []
    problems = _main_definition_problems(example.tree)
    code = [text for kind, text in example.cells if kind == "code"]
    if not code or not _ends_with_main_guard(code[-1]):
        problems.append('the last code cell does not end with `if __name__ == "__main__": main()`')
    return problems


def _main_definition_problems(tree: ast.Module) -> list[str]:
    """A missing top-level ``main()``, or one that is ``async``."""
    mains = [
        statement
        for statement in tree.body
        if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef)
        and statement.name == "main"
    ]
    if not mains:
        return ["no top-level main()"]
    if isinstance(mains[-1], ast.AsyncFunctionDef):
        return ["main() is async; the run calls main(), and a coroutine is never run"]
    return []


def _ends_with_main_guard(source: str) -> bool:
    """Whether a cell's last statement is `if __name__ == "__main__": main()`."""
    body = _module_statements(source)
    return bool(body) and is_guarded_main_call(body[-1])


def _no_backend_forcing(example: _Example) -> list[str]:
    return (
        [f"the example sets {name}; JAX chooses the backend" for name in _forced_in_code(example)]
        + [
            f"the example chooses its device with {how}; JAX chooses the device"
            for how in _devices_chosen(example)
        ]
        + [
            f"the prose tells the reader to set {name}; JAX chooses the backend"
            for name in _forced_in_prose(example)
        ]
    )


def _forced_in_code(example: _Example) -> list[str]:
    """The backend settings the example's code writes, sorted: in the environment or jax's config.

    Reading one, ``os.environ.get("CUDA_VISIBLE_DEVICES")`` or ``jax.config.read``, chooses
    nothing.
    """
    names = _module_names(example)
    configured = {
        setting
        for call in example.code_nodes
        if isinstance(call, ast.Call) and _qualified(call.func, names) == _CONFIG_UPDATE
        for setting in _constant_strings(call.args[:1])
    }
    return sorted((_environment_written(example) | configured) & set(_FORCING))


def _constant_strings(nodes: list[ast.expr]) -> list[str]:
    """The string constants among ``nodes``."""
    return [
        node.value
        for node in nodes
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


def _module_names(example: _Example) -> dict[str, str]:
    """What the names the example's module-scope imports bind mean (:func:`_imported_names`)."""
    return _imported_names(example.tree.body if example.tree is not None else [])


def _environment_written(example: _Example) -> set[str]:
    """The variables the example's code writes into the process environment.

    ``os.environ`` and ``os.putenv``/``os.unsetenv`` count under any name the module's imports
    give them, and only then: a local mapping named ``environ`` is not the environment.
    """
    names = _module_names(example)
    environ = _spellings(names, "os.environ")
    mutators = {spelling for function in OS_MUTATORS for spelling in _spellings(names, function)}
    return {
        key
        for node in example.code_nodes
        for key in environment_writes(node, environ=environ, mutators=mutators)
    }


def _spellings(names: dict[str, str], qualified: str) -> set[str]:
    """How code spells a module's member, ``qualified``, through the names imports bind.

    The qualified name itself, a name bound to the member, and ``<name>.<member>`` for a name bound
    to its module: ``os.environ``, ``environ`` after ``from os import environ``, ``o.environ``
    after ``import os as o``.
    """
    module, _, member = qualified.rpartition(".")
    return {qualified} | {
        name if meaning == qualified else f"{name}.{member}"
        for name, meaning in names.items()
        if meaning in {qualified, module}
    }


_PLATFORM_POSITION = {"jax.devices": 0, "jax.local_devices": 1}
"""Where each device listing takes its platform positionally: ``devices(backend)``,
``local_devices(process_index, backend)``."""


def _devices_chosen(example: _Example) -> list[str]:
    """How the example's code picks a device or platform, sorted.

    Naming a platform for ``jax.devices`` or ``jax.local_devices``, a ``backend=`` keyword to any
    call, or ``jax.default_device``, under the names the module's imports give them.
    """
    names = _imported_names(example.tree.body if example.tree is not None else [])
    return sorted(
        {choice for node in example.code_nodes for choice in _device_choices(node, names)}
    )


def _device_choices(node: ast.AST, names: dict[str, str]) -> list[str]:
    """How ``node`` alone picks a device or platform, its names resolved through ``names``."""
    if isinstance(node, ast.Name | ast.Attribute):
        return ["jax.default_device"] if _qualified(node, names) == "jax.default_device" else []
    return _call_choices(node, names) if isinstance(node, ast.Call) else []


def _call_choices(call: ast.Call, names: dict[str, str]) -> list[str]:
    """How ``call`` picks a platform: positionally to a device listing, or by ``backend=``."""
    function = _qualified(call.func, names)
    position = _PLATFORM_POSITION.get(function or "")
    platform = call.args[position] if position is not None and len(call.args) > position else None
    choices = [f"{function}(<platform>)"] if _names_a_platform(platform) else []
    if any(
        keyword.arg == "backend" and _names_a_platform(keyword.value) for keyword in call.keywords
    ):
        choices.append(f"{function or ast.unparse(call.func)}(backend=...)")
    return choices


def _names_a_platform(argument: ast.expr | None) -> bool:
    """Whether ``argument`` names a platform: ``None`` asks for jax's default backend."""
    return argument is not None and not (
        isinstance(argument, ast.Constant) and argument.value is None
    )


_OWN_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
"""The bodies whose names are their own, not the module's."""


def _imported_names(statements: list[ast.stmt]) -> dict[str, str]:
    """What each name the module-scope imports among ``statements`` bind means.

    ``statements`` are a module's top-level ones. An import binds a module name at top level and
    in a module-level ``try``, ``if``, ``with``, loop or ``match``, never in a function's, class's
    or lambda's body; the last import of a name decides what it means.
    """
    names: dict[str, str] = {}
    for statement in statements:
        for imported in _module_scope_imports(statement):
            names.update(_bindings(imported))
    return names


def _module_scope_imports(node: ast.AST) -> Iterator[ast.Import | ast.ImportFrom]:
    """``node`` if it is an import, else the imports within it that bind module names, in order."""
    if isinstance(node, ast.Import | ast.ImportFrom):
        yield node
    elif not isinstance(node, _OWN_SCOPES):
        for child in ast.iter_child_nodes(node):
            yield from _module_scope_imports(child)


def _bindings(statement: ast.Import | ast.ImportFrom) -> list[tuple[str, str]]:
    """Each name an import binds, with the dotted name it binds it to.

    ``import a.b`` binds ``a`` to ``a``; ``import a.b as c`` binds ``c`` to ``a.b``; ``from a
    import b as c`` binds ``c`` to ``a.b``. A relative import binds names outside any package.
    """
    if isinstance(statement, ast.Import):
        return [
            (alias.asname, alias.name)
            if alias.asname
            else (alias.name.partition(".")[0], alias.name.partition(".")[0])
            for alias in statement.names
        ]
    module = "." * statement.level + (statement.module or "")
    return [(alias.asname or alias.name, f"{module}.{alias.name}") for alias in statement.names]


def _qualified(node: ast.expr, names: dict[str, str]) -> str | None:
    """The qualified name ``node`` reaches (``jax.devices`` for ``j.devices``), or ``None``."""
    if isinstance(node, ast.Name):
        return names.get(node.id)
    if isinstance(node, ast.Attribute):
        owner = _qualified(node.value, names)
        return None if owner is None else f"{owner}.{node.attr}"
    return None


def _forced_in_prose(example: _Example) -> list[str]:
    """The backend settings the example's markdown tells the reader to set or clear, sorted."""
    markdown = "\n".join(text for kind, text in example.cells if kind == "markdown")
    told = {
        next(name for name in match.groups() if name)
        for match in _FORCING_INSTRUCTION.finditer(markdown)
    }
    return sorted(told)


def _no_arguments(example: _Example) -> list[str]:
    nodes = example.code_nodes
    parsers = sorted(
        {name for node in nodes for name in _imported_packages(node) if name in _ARGUMENT_PARSERS}
    )
    problems = [
        f"imports {name}; the run passes no arguments, so a value a reader may change is a constant"
        for name in parsers
    ]
    return problems + [
        f"line {node.lineno} reads sys.argv beyond [0], the script's name; "
        "the run passes no arguments"
        for node in _argv_reads(nodes, _spellings(_module_names(example), "sys.argv"))
    ]


def _imported_packages(node: ast.AST) -> list[str]:
    """The top-level packages an import statement imports."""
    if isinstance(node, ast.Import):
        return [alias.name.partition(".")[0] for alias in node.names]
    if isinstance(node, ast.ImportFrom) and node.module and not node.level:
        return [node.module.partition(".")[0]]
    return []


def _argv_reads(nodes: list[ast.AST], spellings: set[str]) -> list[ast.expr]:
    """Each use of ``sys.argv`` other than ``argv[0]``.

    ``spellings`` are how the module's imports let code spell it (:func:`_spellings`); a
    function's own ``from sys import argv`` counts too.
    """
    names = spellings | {name for node in nodes for name in _argv_imported(node)}
    own_name = {id(node.value) for node in nodes if _is_index_zero(node)}
    return [
        node
        for node in nodes
        if isinstance(node, ast.Name | ast.Attribute)
        and ast.unparse(node) in names
        and id(node) not in own_name
    ]


def _argv_imported(node: ast.AST) -> list[str]:
    """The names a ``from sys import argv`` binds."""
    if not (isinstance(node, ast.ImportFrom) and node.module == "sys"):
        return []
    return [alias.asname or alias.name for alias in node.names if alias.name == "argv"]


def _is_index_zero(node: ast.AST) -> TypeGuard[ast.Subscript]:
    """Whether ``node`` is a subscript by the constant 0."""
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.Constant)
        and node.slice.value == 0
    )


def _no_env_writes(example: _Example) -> list[str]:
    # A backend setting written is no-backend-forcing's finding: one finding a write.
    written = _environment_written(example) - set(_FORCING)
    names = _module_names(example)
    loads = any(
        isinstance(node, ast.Call) and _qualified(node.func, names) == _LOADS_DOTENV
        for node in example.code_nodes
    )
    return [f"the example writes {key}; {_SETS_NONE}" for key in sorted(written)] + (
        [f"the example loads a .env file into the environment with {_LOADS_DOTENV}(); {_SETS_NONE}"]
        if loads
        else []
    )


def _no_expected_output(example: _Example) -> list[str]:
    stated = [
        *(line for kind, text in example.cells if kind == "markdown" for line in text.splitlines()),
        *(comment for kind, text in example.cells if kind == "code" for comment in _comments(text)),
    ]
    return [
        f"{line.strip()!r} states an output; the page shows what the run printed"
        for line in stated
        if _EXPECTED_OUTPUT.match(line)
    ]


def _comments(source: str) -> list[str]:
    """The comments of a code cell, as tokenize reads them: never a ``#`` inside a string."""
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, SyntaxError):
        return []  # the parses rule reports a cell that does not tokenize
    return [token.string for token in tokens if token.type == tokenize.COMMENT]


def _artifacts(example: _Example) -> list[str]:
    problems = [
        f"writes under {path!r}; artifacts go through resolve_output_dir"
        for path in _docs_paths(example.code_nodes)
    ]
    if _names(example.code_nodes, "savefig") and not _names(
        example.code_nodes, "resolve_output_dir"
    ):
        problems.append("savefig without resolve_output_dir; page figures come from plt.show()")
    return problems


def _docs_paths(nodes: list[ast.AST]) -> list[str]:
    """String constants in code that name a path under the docs tree."""
    return [
        node.value
        for node in nodes
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.startswith("docs/")
    ]


def _names(nodes: list[ast.AST], name: str) -> bool:
    """Whether code names ``name`` as a variable or an attribute."""
    return any(
        (isinstance(node, ast.Name) and node.id == name)
        or (isinstance(node, ast.Attribute) and node.attr == name)
        for node in nodes
    )


def _no_star_imports(example: _Example) -> list[str]:
    return [
        f"from {node.module} import *"
        for node in example.code_nodes
        if isinstance(node, ast.ImportFrom) and any(a.name == "*" for a in node.names)
    ]


def _typed_keys(example: _Example) -> list[str]:
    # Reached as an attribute (jax.random.PRNGKey) or imported by name, under any alias.
    if any(
        (isinstance(node, ast.Attribute) and node.attr == "PRNGKey")
        or (isinstance(node, ast.ImportFrom) and any(a.name == "PRNGKey" for a in node.names))
        for node in example.code_nodes
    ):
        return ["jax.random.PRNGKey; new keys come from jax.random.key"]
    return []


def _gpu_check(example: _Example) -> list[str]:
    devices = _metadata_table(example.cells).get("Devices")
    if devices != Devices.GPU_REQUIRED:
        return _require_gpu_without_needing_one(example, devices)
    statements = [
        statement
        for source in _code_by_section(example).get("Setup", [])
        for statement in _module_statements(source)
    ]
    if _calls_require_gpu(statements):
        return []
    return [
        "a GPU-required example's Setup calls substrax.devices.require_gpu() with no arguments "
        "right after its imports, as a statement of its own or the value of an assignment, so it "
        "stops before any work when there is no GPU"
    ]


def _require_gpu_without_needing_one(example: _Example, devices: str | None) -> list[str]:
    """A problem when an example not declared GPU-required calls ``require_gpu()`` anywhere.

    An example with no Devices value is the ``metadata`` rule's finding alone.
    """
    names = _module_names(example)
    if devices is None or not any(
        isinstance(node, ast.Call) and _qualified(node.func, names) in _REQUIRE_GPU
        for node in example.code_nodes
    ):
        return []
    return [
        f"a {devices} example calls require_gpu(), which stops every run without a GPU; "
        "declare it GPU-required or remove the call"
    ]


def _calls_require_gpu(statements: list[ast.stmt]) -> bool:
    """Whether the first statement after the imports calls ``require_gpu``, or assigns its value.

    Work before it could already touch a GPU, and a call inside a function, a condition or a
    ``try`` may never run or have its refusal caught, so the example could go on without a GPU. A
    docstring does no work, so it may come between. The call is resolved through the names the
    imports before it bind.
    """
    leading = 0
    while leading < len(statements) and _precedes_the_check(statements[leading]):
        leading += 1
    if leading == len(statements):
        return False
    first = statements[leading]
    names = _imported_names(statements[:leading])
    return (
        isinstance(first, ast.Expr | ast.Assign | ast.AnnAssign)
        and isinstance(first.value, ast.Call)
        and not first.value.args
        and not first.value.keywords
        and _qualified(first.value.func, names) in _REQUIRE_GPU
    )


def _precedes_the_check(statement: ast.stmt) -> bool:
    """Whether ``statement`` may come before ``require_gpu()``: an import, or a docstring."""
    return isinstance(statement, ast.Import | ast.ImportFrom) or (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, str)
    )


def _page(example: _Example) -> list[str]:
    return [
        f"{problem.kind}: {problem.detail}"
        for problem in check_page(
            example.script, repo_root=example.repo_root, config=example.config
        )
    ]


RULES: Mapping[str, Rule] = {
    rule.id: rule
    for rule in (
        Rule("pair", "The notebook is the clean pair of the script.", _pair),
        Rule("names", "Each part of the path is snake_case, no number prefix.", _path_names),
        Rule(
            "parses",
            "UTF-8 Python, markers without options, each markdown cell a triple-quoted string.",
            _parses,
        ),
        Rule("title", "The first cell opens with the example's only # title.", _title),
        Rule("metadata", "The first cell's table gives Tier, Devices, Prerequisites.", _metadata),
        Rule("skeleton", "The ## sections follow the guide's order.", _skeleton),
        Rule("goals", "Learning Goals is a numbered list.", _goals),
        Rule("setup", "Setup is one code cell holding every import.", _setup),
        Rule("steps", "Implementation is ### Step 1, Step 2, ...", _steps),
        Rule("results", "Results is under concrete ### titles.", _results),
        Rule("troubleshooting", "Troubleshooting is one ### entry per symptom.", _troubleshooting),
        Rule("next-steps", "Next Steps' only subsection is Experiments to Try.", _next_steps),
        Rule("main", "main() at top level, run by the last cell's __main__ guard.", _main),
        Rule("no-arguments", "The example reads no command-line arguments.", _no_arguments),
        Rule("no-backend-forcing", "The example never chooses the backend.", _no_backend_forcing),
        Rule(
            "no-env-writes",
            "The example reads the environment and never writes it.",
            _no_env_writes,
        ),
        Rule("no-expected-output", "No comment or prose states an output.", _no_expected_output),
        Rule("artifacts", "Artifacts go through resolve_output_dir, never docs/.", _artifacts),
        Rule("no-star-imports", "No `from x import *`.", _no_star_imports),
        Rule("typed-keys", "Keys come from jax.random.key, not PRNGKey.", _typed_keys),
        Rule(
            "gpu-check",
            "A GPU-required example's Setup calls require_gpu(), no arguments, after its imports.",
            _gpu_check,
        ),
        Rule("page", "The docs page shows a current run of this script.", _page),
    )
}
"""The guide's mechanical rules by id, in the order :func:`check_example` reports them."""

SOURCE_RULES = frozenset(RULES) - {"page"}
"""The rules that read only the example and its notebook, needing no run: what a commit checks."""
