"""``substrax-notebooks``: write, check, page and test a repository's examples.

```text
substrax-notebooks convert EXAMPLE... [--repo-root DIR]
substrax-notebooks check EXAMPLE... [--repo-root DIR]
substrax-notebooks page EXAMPLE... [--repo-root DIR]
substrax-notebooks test [PATH...] [--changed-since REF] [--repo-root DIR]
```

``convert`` writes each example's clean notebook. ``check`` applies the guide's source rules
(:data:`substrax.notebooks.conformance.SOURCE_RULES`) and runs nothing, which is what the
``example-conformance`` pre-commit hook does on changed examples. ``page`` runs each example as
its clean notebook within the repository's ``[tool.substrax.examples]`` budget and writes its docs
page. An EXAMPLE is an existing ``.py`` under
``<repo root>/examples``: a private one (a path part starting with ``_``) is skipped, and any other
path is refused. The repository root is the working directory unless ``--repo-root`` names one.
Exit codes: 0 for success, 1 when a check finds a problem or a run fails, 2 for a refused command.

``test`` runs pytest on the examples and the tests of their results, ``examples`` and, where it
exists, ``tests/examples`` unless PATHs name others, apart from the repository's own tests: with
:func:`examples_pytest_options` and without the caller's ``PYTEST_ADDOPTS`` and
``PYTEST_PLUGINS``, so the session holds pytest and the example suite alone, reads none of the
repository's ini keys, and each kernel gets the environment ``uv run python examples/...`` gets.
``--changed-since`` is the example suite's ``--examples-changed-since``. A PATH is taken from the
working directory, as an EXAMPLE is, and one outside the repository root is refused. The
settings are read first, and refused as ``check`` and ``page`` refuse them. Its exit code is
otherwise pytest's. It needs pytest: ``substrax[notebooks,testing]``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from substrax.examples import discover_examples
from substrax.notebooks.config import ExamplesConfig, read_examples_config
from substrax.notebooks.conformance import check_example, SOURCE_RULES
from substrax.notebooks.pages import generate_page, PageError
from substrax.notebooks.pairs import convert
from substrax.notebooks.run import RunError


_SUCCESS, _UNSUCCESSFUL, _REFUSED = 0, 1, 2

EXAMPLES_PYTEST_CONFIG = Path(__file__).with_name("examples_pytest.ini")
"""The examples session's configuration file, which sets nothing."""

_CALLERS_PYTEST_SETTINGS = frozenset({"PYTEST_ADDOPTS", "PYTEST_PLUGINS"})
"""The variables through which pytest takes options and plugins from its caller's environment
(``pytest --help``); the examples' session runs without them."""

type _Command = Callable[[argparse.Namespace], int]
type _ExamplesCommand = Callable[[argparse.Namespace, list[Path]], int]


def main(argv: Sequence[str] | None = None) -> int:
    """Run one ``substrax-notebooks`` command.

    Args:
        argv: The arguments after the program name; the process's when ``None``.

    Returns:
        The exit code.
    """
    arguments = _parser().parse_args(argv)
    return arguments.command(arguments)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="substrax-notebooks", description=__doc__)
    commands = parser.add_subparsers(required=True)
    for name, command, help_text in (
        ("convert", _convert, "write each example's clean notebook"),
        ("check", _check, "apply the guide's source rules; runs nothing"),
        ("page", _page, "run each example as its clean notebook and write its docs page"),
    ):
        subparser = commands.add_parser(name, help=help_text)
        subparser.add_argument("examples", nargs="+", type=Path, metavar="EXAMPLE")
        subparser.add_argument("--repo-root", type=Path, default=Path.cwd())
        subparser.set_defaults(command=_on_examples(command))
    tester = commands.add_parser(
        "test", help="run the examples and the tests of their results apart from the other tests"
    )
    tester.add_argument("paths", nargs="*", type=Path, metavar="PATH")
    tester.add_argument("--changed-since", metavar="REF")
    tester.add_argument("--repo-root", type=Path, default=Path.cwd())
    tester.set_defaults(command=_test)
    return parser


def _on_examples(command: _ExamplesCommand) -> _Command:
    """``command`` given the examples its arguments name, once anything that is none is refused.

    A private example (a path part starting with ``_``) is skipped, as ``discover_examples`` skips
    it; a path that is not an existing file under ``<repo root>/examples`` is refused.
    """

    def run(arguments: argparse.Namespace) -> int:
        refused = _refused(arguments.examples, repo_root=arguments.repo_root)
        if refused:
            examples_root = arguments.repo_root / "examples"
            sys.stderr.write(
                f"substrax-notebooks: not an example under {examples_root}: {', '.join(refused)}\n"
            )
            return _REFUSED
        known = set(discover_examples((arguments.repo_root / "examples").resolve()))
        return command(arguments, [path for path in arguments.examples if path.resolve() in known])

    return run


def _refused(paths: list[Path], *, repo_root: Path) -> list[str]:
    """The paths that are not existing files under ``repo_root / "examples"``."""
    examples_root = (repo_root / "examples").resolve()
    return [
        str(path)
        for path in paths
        if not (path.is_file() and path.resolve().is_relative_to(examples_root))
    ]


def _convert(arguments: argparse.Namespace, examples: list[Path]) -> int:
    del arguments
    for script in examples:
        sys.stdout.write(f"{convert(script)}\n")
    return _SUCCESS


def _check(arguments: argparse.Namespace, examples: list[Path]) -> int:
    config = _read_settings(arguments.repo_root)
    if config is None:
        return _REFUSED
    findings = [
        finding
        for script in examples
        for finding in check_example(
            script, repo_root=arguments.repo_root, config=config, rules=SOURCE_RULES
        )
    ]
    for finding in findings:
        sys.stdout.write(f"{finding}\n")
    return _UNSUCCESSFUL if findings else _SUCCESS


def _page(arguments: argparse.Namespace, examples: list[Path]) -> int:
    config = _read_settings(arguments.repo_root)
    if config is None:
        return _REFUSED
    status = _SUCCESS
    for script in examples:
        try:
            page = generate_page(script, repo_root=arguments.repo_root, config=config)
            sys.stdout.write(f"{page}\n")
        except (RunError, PageError) as error:
            sys.stderr.write(f"substrax-notebooks: {error}\n")
            status = _UNSUCCESSFUL
    return status


def _test(arguments: argparse.Namespace) -> int:
    """Run pytest on the examples apart from the repository's own tests; return its exit code."""
    repo_root = arguments.repo_root.resolve()
    if _read_settings(repo_root) is None:
        return _REFUSED
    outside = [
        str(path) for path in arguments.paths if not path.resolve().is_relative_to(repo_root)
    ]
    if outside:
        sys.stderr.write(f"substrax-notebooks: not under {repo_root}: {', '.join(outside)}\n")
        return _REFUSED
    environment = {
        name: value for name, value in os.environ.items() if name not in _CALLERS_PYTEST_SETTINGS
    }
    completed = subprocess.run(  # noqa: S603  # nosec B603 - this interpreter, no shell
        _pytest_command(arguments, repo_root=repo_root), cwd=repo_root, env=environment, check=False
    )
    return completed.returncode


def examples_pytest_options(repo_root: Path) -> tuple[str, ...]:
    """The options that run pytest on ``repo_root``'s examples apart from its own tests.

    The repository's ini keys (``addopts`` with a unit-test timeout or a coverage floor,
    ``faulthandler_timeout``, ``required_plugins``, ``strict_config``), its conftests and every
    plugin installed with an entry point (pytest-timeout, pytest-cov, pytest-env, xdist, which
    read their own ini keys and write the environment every kernel inherits) would otherwise reach
    the session. So pytest reads :data:`EXAMPLES_PYTEST_CONFIG` instead of the repository's
    configuration, takes ``repo_root`` as its rootdir whatever directory holds it, and loads the
    example suite alone, by name.

    Args:
        repo_root: The repository whose examples the session runs.

    Returns:
        The options, before the paths to collect.
    """
    return (
        f"--rootdir={repo_root}",
        "-c",
        str(EXAMPLES_PYTEST_CONFIG),
        "--noconftest",
        "--disable-plugin-autoload",
        "-p",
        "substrax.testing.example_suite",
    )


def _pytest_command(arguments: argparse.Namespace, *, repo_root: Path) -> list[str]:
    """The examples' pytest command: this interpreter, the session's options and its paths."""
    paths = [str(path.resolve()) for path in arguments.paths] or _default_paths(repo_root)
    changed = (
        []
        if arguments.changed_since is None
        else [f"--examples-changed-since={arguments.changed_since}"]
    )
    return [
        sys.executable,
        "-m",
        "pytest",
        *examples_pytest_options(repo_root),
        *changed,
        *paths,
    ]


def _default_paths(repo_root: Path) -> list[str]:
    """``examples``, and ``tests/examples`` where it exists: pytest refuses a missing path."""
    return [
        "examples",
        *(["tests/examples"] if (repo_root / "tests" / "examples").is_dir() else []),
    ]


def _read_settings(repo_root: Path) -> ExamplesConfig | None:
    """The repository's example settings, or ``None`` once the reason they are unusable is written.

    A missing table or ``[project]`` ``name`` and settings no run can be held to
    (``read_examples_config``'s ``LookupError`` and ``ValueError``) are both the user's to fix, so
    neither is a traceback.
    """
    try:
        return read_examples_config(repo_root)
    except (LookupError, ValueError) as error:
        sys.stderr.write(f"substrax-notebooks: {error}\n")
        return None
