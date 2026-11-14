"""A repository's example settings: the ``[tool.substrax.examples]`` table of its pyproject.toml.

```toml
[tool.substrax.examples]
source_url = "https://github.com/avitai/<repository>/blob/main"  # a page links its script under it
budget_seconds = 600                                  # the whole run of one example
budgets = { "examples/training/long_run.py" = 1800 }  # an example's own budget instead
# a run raising one of these messages is skipped:
unavailable = ["<the message an example fails with when its dataset is missing>"]
page_size_warn_bytes = 102400                         # a larger page is written with a warning
page_size_bytes = 204800                              # a larger page is refused
page_size_exempt = ["examples/gallery/overview.py"]   # examples whose pages are not measured
```

The page-size settings follow Documenter.jl's `size_threshold_warn`, `size_threshold` and
`size_threshold_ignore` (v1.20.0, `src/html/HTMLWriter.jl`), with its defaults of 100 and 200 KiB,
measured on the Markdown page substrax writes, its figures excluded; Documenter's documentation
prefers exempting a few pages to raising the limit for all.

A budget bounds one example's whole run, as pytest-timeout's per-item timeout and Bazel's per-target
``timeout`` do: ``budget_seconds`` is the repository default and an example's ``budgets`` entry,
keyed by its path as its page records it, replaces it for that example. An entry states the run it
was measured against; ``check_page`` reports a page whose recorded run is longer than its budget.

The command line, the page generator's callers, the shared example tests and compute's examples
jobs read the same table, so a repository states its settings once. A misspelled key is refused
rather than ignored.
"""

from __future__ import annotations

import math
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from substrax.examples import discover_examples, example_source
from substrax.records import read_record, UNKNOWN_FIELDS_REFUSED


_TABLE = "[tool.substrax.examples]"


@dataclass(frozen=True, slots=True, kw_only=True)
class ExamplesConfig:
    """How a repository's examples are run and checked.

    Attributes:
        distribution: The distribution the repository builds, its pyproject.toml's
            ``[project]`` ``name``, not a key of the table; each page records the version of it
            the run imported.
        source_url: Where the repository's files are published, with no trailing slash; a page
            links its script and notebook as ``<source_url>/examples/<path>``. A page lives under
            ``docs/``, so a relative link cannot reach ``examples/``.
        budget_seconds: How long one example's whole run may take, from its kernel's start to
            its last cell, unless ``budgets`` gives the example its own.
        unavailable: Messages that mean a dataset the example needs is not available here; a
            run whose raised exception, as ``<type>: <message>``, holds one is skipped with it
            rather than failed. A run that raised nothing, its budget passed or its kernel dead,
            is never skipped. A signature every raised exception holds, such as ``""`` or
            ``": "``, is refused.
        page_size_warn_bytes: A page larger than this, in bytes, is written with a warning.
        page_size_bytes: A page larger than this, in bytes, is refused.
        page_size_exempt: Examples whose pages are not measured, each as its page records it:
            ``examples/<path>.py``. An entry in another form is refused here, and one naming no
            example when the settings are read.
        budgets: Examples' own budgets, in seconds, keyed by each example's path as its page
            records it, ``examples/<path>.py``; an example without an entry has
            ``budget_seconds``. Keys are checked as ``page_size_exempt``'s entries are.
    """

    __pydantic_config__ = UNKNOWN_FIELDS_REFUSED

    distribution: str
    source_url: str
    budget_seconds: float
    unavailable: tuple[str, ...] = ()
    page_size_warn_bytes: int = 100 * 2**10
    page_size_bytes: int = 200 * 2**10
    page_size_exempt: tuple[str, ...] = ()
    budgets: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Refuse a source URL, a budget or page-size limits no page or run can be held to.

        Raises:
            ValueError: If ``source_url`` is not an https URL without a trailing slash,
                ``budget_seconds`` is not a finite number above zero, the page-size limits are
                not above zero with the warning size no larger than the limit, an
                ``unavailable`` signature is in every raised exception, a ``page_size_exempt``
                entry or ``budgets`` key is not an example's path as its page records it, or a
                ``budgets`` value is not a finite number above zero.
        """
        if not self.source_url.startswith("https://") or self.source_url.endswith("/"):
            msg = f"source_url must be an https URL with no trailing slash, got {self.source_url!r}"
            raise ValueError(msg)
        refuse_unusable_budget(self.budget_seconds)
        if not 0 < self.page_size_warn_bytes <= self.page_size_bytes:
            msg = (
                "page_size_warn_bytes and page_size_bytes must be above zero, the warning size no "
                f"larger than the limit, got {self.page_size_warn_bytes} and {self.page_size_bytes}"
            )
            raise ValueError(msg)
        _refuse_signatures_matching_every_failure(self.unavailable)
        _refuse_malformed_sources("page_size_exempt", self.page_size_exempt)
        _refuse_malformed_sources("budgets", tuple(self.budgets))
        for source, seconds in self.budgets.items():
            refuse_unusable_budget(seconds, setting=f"budgets[{source!r}]")

    def budget_for(self, source: str) -> float:
        """Return how long an example's whole run may take.

        Args:
            source: The example's path as its page records it, ``examples/<path>.py``.

        Returns:
            Its ``budgets`` entry, or ``budget_seconds`` when it has none.
        """
        return self.budgets.get(source, self.budget_seconds)


def _refuse_signatures_matching_every_failure(signatures: tuple[str, ...]) -> None:
    """Refuse an ``unavailable`` signature that every raised exception holds.

    A raised exception reads ``<type>: <message>``, so a signature the separator ``": "`` holds,
    the empty one included, would skip every failing run.

    Args:
        signatures: The ``unavailable`` signatures.

    Raises:
        ValueError: If a signature is ``""``, ``" "``, ``":"`` or ``": "``.
    """
    vacuous = [signature for signature in signatures if signature in ": "]
    if vacuous:
        msg = (
            "unavailable signatures must name what a run raises; "
            f"{', '.join(map(repr, vacuous))} is in every raised exception"
        )
        raise ValueError(msg)


def _refuse_malformed_sources(setting: str, entries: tuple[str, ...]) -> None:
    """Refuse an entry of ``setting`` that is not an example's path as its page records it.

    Args:
        setting: The setting the entries belong to, for the message.
        entries: Its entries.

    Raises:
        ValueError: If an entry is not ``examples/<path>.py`` in its normal form.
    """
    malformed = [entry for entry in entries if not _is_source_path(entry)]
    if malformed:
        msg = (
            f"{setting} entries name an example as its page records it, "
            f"examples/<path>.py; got {', '.join(map(repr, malformed))}"
        )
        raise ValueError(msg)


def _is_source_path(entry: str) -> bool:
    """Whether ``entry`` is an example's path as a page records it: ``examples/<path>.py``."""
    path = PurePosixPath(entry)
    return (
        str(path) == entry
        and path.parts[:1] == ("examples",)
        and ".." not in path.parts
        and path.suffix == ".py"
    )


def refuse_unusable_budget(budget_seconds: float, *, setting: str = "budget_seconds") -> None:
    """Refuse a budget no run can be held to: zero or less, infinite, or not a number.

    Args:
        budget_seconds: How long a run may take.
        setting: Where the budget was given, for the message.

    Raises:
        ValueError: If ``budget_seconds`` is not a finite number above zero.
    """
    if not (math.isfinite(budget_seconds) and budget_seconds > 0):
        msg = f"{setting} must be a finite number above zero, got {budget_seconds}"
        raise ValueError(msg)


def read_examples_config(repo_root: Path) -> ExamplesConfig:  # noqa: DOC503  # pydantic.ValidationError, a ValueError, is raised by read_record
    """Read the ``[tool.substrax.examples]`` table of the repository's pyproject.toml.

    Args:
        repo_root: The repository root.

    Returns:
        The settings.

    Raises:
        LookupError: If the repository has no pyproject.toml, it has no such table, or it names
            no ``[project]``.
        ValueError: If a key is unknown, missing or holds a value of the wrong type, the table
            sets ``distribution``, a ``page_size_exempt`` entry or ``budgets`` key names no
            example of the repository, or a symlink below the root leads to an example
            (:class:`~substrax.examples.ExamplePathError`).
    """
    pyproject = repo_root / "pyproject.toml"
    if not pyproject.is_file():
        msg = f"{repo_root} has no pyproject.toml"
        raise LookupError(msg)
    document = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    table = document.get("tool", {}).get("substrax", {}).get("examples")
    if table is None:
        msg = f"{pyproject} has no {_TABLE} table"
        raise LookupError(msg)
    distribution = document.get("project", {}).get("name")
    if distribution is None:
        msg = f"{pyproject} names no [project], the distribution each page records the version of"
        raise LookupError(msg)
    if "distribution" in table:
        msg = f"{_TABLE} sets distribution, which is read from [project] name"
        raise ValueError(msg)
    config = read_record(ExamplesConfig, {**table, "distribution": distribution})
    known = {
        example_source(path, repo_root=repo_root)
        for path in discover_examples(repo_root / "examples")
    }
    for setting, entries in (
        ("page_size_exempt", config.page_size_exempt),
        ("budgets", tuple(config.budgets)),
    ):
        _refuse_unknown_sources(setting, entries, known=known, repo_root=repo_root)
    return config


def _refuse_unknown_sources(
    setting: str, entries: tuple[str, ...], *, known: set[str], repo_root: Path
) -> None:
    """Refuse an entry of ``setting`` naming no example the repository's pages are made from.

    Args:
        setting: The setting the entries belong to, for the message.
        entries: Its entries.
        known: Every example's path as its page records it.
        repo_root: The repository root, for the message.

    Raises:
        ValueError: If :func:`~substrax.examples.discover_examples` does not find an entry.
    """
    unknown = [entry for entry in entries if entry not in known]
    if unknown:
        msg = f"{setting} names {', '.join(unknown)}, which no example of {repo_root} is"
        raise ValueError(msg)
