"""``substrax-notebooks``: write, check, page and test a repository's examples.

```text
substrax-notebooks convert EXAMPLE... [--repo-root DIR]
substrax-notebooks check SCRIPT... [--repo-root DIR]
substrax-notebooks page EXAMPLE... [--into-outputs] [--repo-root DIR]
substrax-notebooks recipe RECIPE [--repo-root DIR]
substrax-notebooks test [PATH...] [--changed-since REF] [--budget-seconds SECONDS]
                        [--repo-root DIR]
```

``convert`` writes each example's clean notebook. ``check`` applies the guide's source rules
(:data:`substrax.notebooks.conformance.SOURCE_RULES`) to each example and its rules about code
(:data:`~substrax.notebooks.conformance.RECIPE_RULES`) to each recipe, and runs nothing, which is
what the ``example-conformance`` pre-commit hook does on changed examples and recipes; a SCRIPT is
an EXAMPLE or a RECIPE, an existing ``.py`` under ``<repo root>/recipes`` that is not private.
``page`` runs each example as its clean notebook within the repository's
``[tool.substrax.examples]`` budget and writes its docs page; with ``--into-outputs`` it writes the
page and its figures under the run's outputs (``$AVITAI_OUTPUT_DIR/page/docs/examples/...``)
instead, as a compute task does. ``recipe`` runs a recipe as its script and writes what its
``main()`` returned, with where it came from, to ``$AVITAI_OUTPUT_DIR/recipe_record/record.json``
(:mod:`substrax.notebooks.recipe_run`), as a compute task does. An EXAMPLE is an existing ``.py``
under ``<repo root>/examples``, or its notebook, which names the script beside it: a private one (a
path part starting with ``_``) is skipped, and any other path is refused.
``page`` reports a GPU-required example run without a GPU in one line. The repository root is the
working directory unless ``--repo-root`` names one.
Exit codes: 0 for success, 1 when a check finds a problem or a run fails, 2 for a refused command.

``test`` runs pytest on the examples, the recipes and the tests of their results, ``examples``
and, where they exist, ``recipes`` and ``tests/examples`` unless PATHs name others, apart from the
repository's own tests: with :func:`examples_pytest_options` and without the caller's
``PYTEST_ADDOPTS`` and ``PYTEST_PLUGINS``, so the session holds pytest and the example suite
alone, reads none of the repository's ini keys, and each kernel gets the environment
``uv run python examples/...`` gets.
``--changed-since`` is the example suite's ``--examples-changed-since``, and ``--budget-seconds``
its ``--examples-budget-seconds``. A PATH is taken from the working directory, as an EXAMPLE
is, and one outside the repository root is refused. The settings are read first, and refused as
``check`` and ``page`` refuse them. Its exit code is otherwise pytest's. It needs pytest:
``substrax[notebooks,testing]``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

from substrax.artifacts import resolve_output_dir
from substrax.examples import discover_examples, example_source, lexical_path, ScriptPathError
from substrax.notebooks.config import ExamplesConfig, read_examples_config
from substrax.notebooks.conformance import (
    check_example,
    check_recipe,
    no_gpu_reason,
    SOURCE_RULES,
)
from substrax.notebooks.pages import generate_page, PageError
from substrax.notebooks.pairs import convert
from substrax.notebooks.recipe_run import RecipeRunError, RECORD_OUTPUT, run_recipe, write_record
from substrax.notebooks.run import RunError
from substrax.notebooks.text import NotUtf8Error
from substrax.recipes import discover_recipes, recipe_source, RECIPES_DIRECTORY


_SUCCESS, _UNSUCCESSFUL, _REFUSED = 0, 1, 2

EXAMPLES_PYTEST_CONFIG = Path(__file__).with_name("examples_pytest.ini")
"""The examples session's configuration file, which sets nothing."""

_PAGE_OUTPUT = "page"
"""The output, under the run's outputs, that ``page --into-outputs`` writes the pages into."""

_CALLERS_PYTEST_SETTINGS = frozenset({"PYTEST_ADDOPTS", "PYTEST_PLUGINS"})
"""The variables through which pytest takes options and plugins from its caller's environment
(``pytest --help``); the examples' session runs without them."""


class _UsageError(Exception):
    """A command refused before it does anything: :func:`main` writes why and returns 2."""


type _Command = Callable[[argparse.Namespace], int]
type _ExamplesCommand = Callable[[argparse.Namespace, list[Path]], int]
type _ScriptsCommand = Callable[[argparse.Namespace, list[Path], list[Path]], int]


class _SourceOf(Protocol):
    """A script's path from the root: ``example_source`` or ``recipe_source``."""

    def __call__(self, script: Path, /, *, repo_root: Path) -> str:
        """Return the script's path from ``repo_root``, or raise ``ScriptPathError``."""
        ...


def main(argv: Sequence[str] | None = None) -> int:
    """Run one ``substrax-notebooks`` command.

    Args:
        argv: The arguments after the program name; the process's when ``None``.

    Returns:
        The exit code.
    """
    arguments = _parser().parse_args(argv)
    try:
        return arguments.command(arguments)
    except _UsageError as refusal:
        sys.stderr.write(f"substrax-notebooks: {refusal}\n")
        return _REFUSED


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="substrax-notebooks", description=__doc__)
    commands = parser.add_subparsers(required=True)
    for name, command, metavar, help_text in (
        ("convert", _on_examples(_convert), "EXAMPLE", "write each example's clean notebook"),
        (
            "check",
            _on_examples_and_recipes(_check),
            "SCRIPT",
            "apply the guide's source rules to examples and recipes; runs nothing",
        ),
        (
            "page",
            _on_examples(_page),
            "EXAMPLE",
            "run each example as its clean notebook and write its docs page",
        ),
    ):
        subparser = commands.add_parser(name, help=help_text)
        subparser.add_argument("examples", nargs="+", type=Path, metavar=metavar)
        subparser.add_argument("--repo-root", type=Path, default=Path.cwd())
        subparser.set_defaults(command=command)
        if name == "page":
            subparser.add_argument(
                "--into-outputs",
                action="store_true",
                help=(
                    "write each page and its figures under the run's outputs "
                    f"($AVITAI_OUTPUT_DIR/{_PAGE_OUTPUT}/docs/examples/...), not into the "
                    "repository, as a compute task does so they come back to be committed"
                ),
            )
    runner = commands.add_parser(
        "recipe", help="run a recipe as its script and write its record under the run's outputs"
    )
    runner.add_argument("recipe", type=Path, metavar="RECIPE")
    runner.add_argument("--repo-root", type=Path, default=Path.cwd())
    runner.set_defaults(command=_recipe)
    tester = commands.add_parser(
        "test", help="run the examples and the tests of their results apart from the other tests"
    )
    tester.add_argument("paths", nargs="*", type=Path, metavar="PATH")
    tester.add_argument("--changed-since", metavar="REF")
    tester.add_argument("--budget-seconds", metavar="SECONDS", type=float)
    tester.add_argument("--repo-root", type=Path, default=Path.cwd())
    tester.set_defaults(command=_test)
    return parser


def _on_examples(command: _ExamplesCommand) -> _Command:
    """``command`` given the examples its arguments name, once anything that is none is refused.

    A notebook names the script beside it, so the hook checks a notebook edited by hand and
    ``convert`` writes it again; a script named with its notebook is taken once. A private example
    (a path part starting with ``_``) is skipped, as ``discover_examples`` skips it. A path that is
    neither a ``.py`` nor a ``.ipynb``, a notebook with no script, and a path that is not an
    existing file under ``<repo root>/examples`` or that a symlink below the root leads to (as
    :func:`substrax.examples.example_source` decides) are refused.
    """

    def run(arguments: argparse.Namespace) -> int:
        selected, refusals = _named_examples(arguments.examples, repo_root=arguments.repo_root)
        if refusals:
            raise _UsageError("; ".join(refusals))
        return command(arguments, selected)

    return run


def _on_examples_and_recipes(command: _ScriptsCommand) -> _Command:
    """``command`` given the examples and the recipes its arguments name, once all are accepted.

    A path below ``<repo root>/recipes`` names a recipe and any other an example, as
    :func:`_on_examples` takes it. A recipe is an existing ``.py`` that
    :func:`substrax.recipes.discover_recipes` lists; a private one is skipped, and any other path
    there is refused.
    """

    def run(arguments: argparse.Namespace) -> int:
        recipes_root = lexical_path(arguments.repo_root / RECIPES_DIRECTORY)
        named = [
            (path, lexical_path(path).is_relative_to(recipes_root)) for path in arguments.examples
        ]
        examples, refusals = _named_examples(
            [path for path, is_recipe in named if not is_recipe], repo_root=arguments.repo_root
        )
        recipes, recipe_refusals = _named_recipes(
            [path for path, is_recipe in named if is_recipe], repo_root=arguments.repo_root
        )
        if refusals or recipe_refusals:
            raise _UsageError("; ".join([*refusals, *recipe_refusals]))
        return command(arguments, examples, recipes)

    return run


def _named_recipes(paths: list[Path], *, repo_root: Path) -> tuple[list[Path], list[str]]:
    """The recipes ``paths`` name, each once and private ones left out, and the refusals."""
    root = repo_root / RECIPES_DIRECTORY
    return _named_scripts(
        paths,
        repo_root=repo_root,
        script_named=lambda path: (
            path
            if path.suffix == ".py" and path.is_file()
            else f"{path} is not a recipe script (.py) under {root}"
        ),
        source=recipe_source,
        discovered=set(discover_recipes(repo_root)),
    )


def _named_examples(paths: list[Path], *, repo_root: Path) -> tuple[list[Path], list[str]]:
    """The example scripts ``paths`` name, each once and private ones left out, and the refusals."""
    examples = repo_root / "examples"
    return _named_scripts(
        paths,
        repo_root=repo_root,
        script_named=lambda path: _script_named(path, examples=examples),
        source=example_source,
        discovered=set(discover_examples(examples)),
    )


def _named_scripts(
    paths: list[Path],
    *,
    repo_root: Path,
    script_named: Callable[[Path], Path | str],
    source: _SourceOf,
    discovered: set[Path],
) -> tuple[list[Path], list[str]]:
    """The scripts ``paths`` name, each once and private ones left out, and the refusals.

    ``script_named`` gives the script a path names or why it names none, ``source`` its path from
    the root or a :class:`~substrax.examples.ScriptPathError`, and a script not ``discovered``
    there is private, so skipped.
    """
    selected: list[Path] = []
    refusals: list[str] = []
    for path in paths:
        script = script_named(path)
        if isinstance(script, str):
            refusals.append(script)
            continue
        try:
            relative = source(script, repo_root=repo_root)
        except ScriptPathError as error:
            refusals.append(str(error))
            continue
        if repo_root / relative in discovered and script not in selected:
            selected.append(script)
    return selected, refusals


def _script_named(path: Path, *, examples: Path) -> Path | str:
    """The script ``path`` names (itself, or the one beside a notebook), or why it names none."""
    if path.suffix not in {".py", ".ipynb"}:
        return f"{path} is not an example script (.py) or its notebook (.ipynb)"
    if not path.is_file():
        return f"{path} is not an example under {examples}"
    script = path.with_suffix(".py")
    if not script.is_file():
        return (
            f"{path} is a notebook with no script {script}; convert writes a notebook from its "
            "script"
        )
    return script


def _convert(arguments: argparse.Namespace, examples: list[Path]) -> int:
    del arguments
    status = _SUCCESS
    for script in examples:
        try:
            sys.stdout.write(f"{convert(script)}\n")
        except NotUtf8Error as error:
            sys.stderr.write(f"substrax-notebooks: {error}\n")
            status = _UNSUCCESSFUL
    return status


def _check(arguments: argparse.Namespace, examples: list[Path], recipes: list[Path]) -> int:
    config = _read_settings(arguments.repo_root)
    findings = [
        *(
            finding
            for script in examples
            for finding in check_example(
                script, repo_root=arguments.repo_root, config=config, rules=SOURCE_RULES
            )
        ),
        *(
            finding
            for recipe in recipes
            for finding in check_recipe(recipe, repo_root=arguments.repo_root, config=config)
        ),
    ]
    for finding in findings:
        sys.stdout.write(f"{finding}\n")
    return _UNSUCCESSFUL if findings else _SUCCESS


def _page(arguments: argparse.Namespace, examples: list[Path]) -> int:
    config = _read_settings(arguments.repo_root)
    destination = resolve_output_dir(_PAGE_OUTPUT).path if arguments.into_outputs else None
    status = _SUCCESS
    for script in examples:
        try:
            page = generate_page(
                script, repo_root=arguments.repo_root, config=config, destination=destination
            )
            sys.stdout.write(f"{page}\n")
        except (RunError, PageError) as error:
            sys.stderr.write(f"substrax-notebooks: {_page_failure(script, error)}\n")
            status = _UNSUCCESSFUL
    return status


def _recipe(arguments: argparse.Namespace) -> int:
    """Run one recipe and write its record; refuse a path that is no recipe or a private one."""
    recipes, refusals = _named_recipes([arguments.recipe], repo_root=arguments.repo_root)
    if refusals or not recipes:
        raise _UsageError("; ".join(refusals) or f"{arguments.recipe} is private, so no recipe")
    config = _read_settings(arguments.repo_root)
    try:
        record = run_recipe(
            recipes[0], repo_root=arguments.repo_root, distribution=config.distribution
        )
    except RecipeRunError as error:
        sys.stderr.write(f"substrax-notebooks: {error}\n")
        return _UNSUCCESSFUL
    sys.stdout.write(f"{write_record(record, resolve_output_dir(RECORD_OUTPUT).path)}\n")
    return _SUCCESS


def _page_failure(script: Path, error: RunError | PageError) -> str:
    """What stopped an example's page, in one line when it is a GPU-required example's missing GPU.

    Without a GPU, a GPU-required example's run stops at ``require_gpu()``: the run did what it
    should, so its traceback would only bury the reason.
    """
    no_gpu = no_gpu_reason(script, error) if isinstance(error, RunError) else None
    return str(error) if no_gpu is None else f"{script}: {no_gpu}"


def _test(arguments: argparse.Namespace) -> int:
    """Run pytest on the examples apart from the repository's own tests; return its exit code."""
    repo_root = arguments.repo_root.resolve()
    _read_settings(repo_root)
    outside = [
        str(path) for path in arguments.paths if not path.resolve().is_relative_to(repo_root)
    ]
    if outside:
        msg = f"not under {repo_root}: {', '.join(outside)}"
        raise _UsageError(msg)
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
    forwarded = [
        f"{option}={value}"
        for option, value in (
            ("--examples-changed-since", arguments.changed_since),
            ("--examples-budget-seconds", arguments.budget_seconds),
        )
        if value is not None
    ]
    return [
        sys.executable,
        "-m",
        "pytest",
        *examples_pytest_options(repo_root),
        *forwarded,
        *paths,
    ]


def _default_paths(repo_root: Path) -> list[str]:
    """``examples``, and ``recipes`` and ``tests/examples`` where they exist.

    pytest refuses a missing path.
    """
    return [
        "examples",
        *(
            directory
            for directory in (RECIPES_DIRECTORY, "tests/examples")
            if (repo_root / directory).is_dir()
        ),
    ]


def _read_settings(repo_root: Path) -> ExamplesConfig:
    """The repository's example settings.

    A missing pyproject.toml, table or ``[project]`` ``name`` and settings no run can be held to
    (``read_examples_config``'s ``LookupError`` and ``ValueError``) are all the user's to fix, so
    each refuses the command rather than ending in a traceback.

    Args:
        repo_root: The repository root.

    Returns:
        The settings.

    Raises:
        _UsageError: If the settings cannot be read or no run can be held to them.
    """
    try:
        return read_examples_config(repo_root)
    except (LookupError, ValueError) as error:
        raise _UsageError(str(error)) from error
