"""The shared example tests: a pytest plugin a repository configures in pyproject only.

``substrax-notebooks test`` loads it by name and runs the examples, with the tests of their results
under ``tests/examples``, apart from the repository's own tests: without the repository's pytest
configuration or conftests, any other plugin, or the caller's ``PYTEST_ADDOPTS`` and
``PYTEST_PLUGINS`` (:func:`substrax.notebooks.cli.examples_pytest_options`). No conftest names it,
and the repository's own pytest leaves the tests of results out:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]                                 # a plain pytest leaves the examples out
addopts = "--ignore=tests/examples"

[tool.substrax.examples]                              # substrax.notebooks.config
source_url = "https://github.com/avitai/datarax/blob/main"
budget_seconds = 600
```

It collects, with no example-test code in the repository:

- ``examples::corpus``, once: there is at least one example; every page under ``docs/examples``
  belongs to one; a hand-written ``uv run python examples/...`` command in a tracked Markdown file
  names a file that exists; ``mkdocs.yml``, where there is one, sets ``validation.nav.omitted_files:
  warn``, so ``mkdocs build --strict`` fails on a page left out of the nav (mkdocs defaults it to
  ``info``).
- ``examples/<path>.py::follows-the-guide``: every rule of :mod:`substrax.notebooks.conformance`.
- ``examples/<path>.py::runs-as-its-page-shows``, marked ``example_run``: the example runs as its
  clean notebook (:func:`substrax.notebooks.run.execute_example`) and its page shows that run
  (:func:`substrax.notebooks.pages.compare_with_run`). A run whose raised exception, as
  :attr:`~substrax.notebooks.run.RunError.raised` names it, holds one of the configured
  ``unavailable`` signatures is skipped with it; so is the run of a ``GPU-required`` example that
  raised ``GPUUnavailableError``, as ``require_gpu()`` does without a GPU, by the qualified name
  :attr:`~substrax.notebooks.run.RunError.raised_class` records, and every test of its results. The
  plugin probes no device and never imports jax: the example's kernel reports the GPU's absence, and
  a GPU lane runs ``substrax-devices --require-gpu`` before the examples, so a lane without its GPU
  fails there instead of skipping.

``--examples-changed-since=REF`` keeps the run of each example whose script, notebook or page
changed since ``REF``, and every run when ``src/``, ``pyproject.toml`` or ``uv.lock`` changed, or a
file under ``examples/`` that is no example's own, as a private helper or data is. The
``example_run`` fixture gives a repository's own tests of an example's results the same run, made
once per session.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterator
from pathlib import Path, PurePosixPath

import pytest

from substrax.examples import discover_examples, example_source, ExamplePathError
from substrax.notebooks import repository
from substrax.notebooks.config import (
    ExamplesConfig,
    read_examples_config,
    refuse_unusable_budget,
)
from substrax.notebooks.conformance import check_example, no_gpu_reason
from substrax.notebooks.corpus import corpus_problems, is_touched, reaches_every_example
from substrax.notebooks.pages import compare_with_run
from substrax.notebooks.run import ExampleRun, execute_example, RunError


type ExampleRunner = Callable[[str], ExampleRun]
"""The ``example_run`` fixture's type, for a test of an example's results to annotate it with."""

RUN_MARKER = "example_run"
"""The marker on each example's run, which names the run items among the session's."""


_CONFIG = pytest.StashKey[ExamplesConfig]()
_EXAMPLES = pytest.StashKey[frozenset[Path]]()
_RUNS = pytest.StashKey[dict[Path, ExampleRun | RunError]]()
_COLLECTED = pytest.StashKey[frozenset[Path]]()
"""The examples whose runs this session collected, before any deselection."""
_SELECTED = pytest.StashKey[frozenset[Path]]()
_CHANGED = pytest.StashKey[frozenset[str]]()
"""The paths ``--examples-changed-since`` found changed, from the root; empty without it."""
"""The examples whose runs this session keeps after ``--examples-changed-since``."""


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add ``--examples-changed-since`` and ``--examples-budget-seconds``.

    Args:
        parser: pytest's option parser.
    """
    parser.addoption(
        "--examples-changed-since",
        metavar="REF",
        default=None,
        help="run only the examples changed since REF; every one when the library changed",
    )
    parser.addoption(
        "--examples-budget-seconds",
        metavar="SECONDS",
        type=float,
        default=None,
        help=(
            "this lane's budget for an example without a budgets entry, in place of the "
            "settings' budget_seconds; an example's own entry still applies"
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register the run marker and read the repository's ``[tool.substrax.examples]`` settings.

    Args:
        config: pytest's configuration.

    Raises:
        pytest.UsageError: If pyproject.toml has no ``[tool.substrax.examples]`` table or
            ``[project]`` ``name``, the table or ``--examples-budget-seconds`` holds settings no
            run can be held to, or an example is reached through a symlink.
    """
    config.addinivalue_line(
        "markers", f"{RUN_MARKER}: an example run as its clean notebook and compared with its page"
    )
    try:
        config.stash[_CONFIG] = _lane_settings(
            read_examples_config(config.rootpath),
            lane_budget=config.getoption("--examples-budget-seconds"),
        )
    except (LookupError, ValueError) as error:
        raise pytest.UsageError(str(error)) from error
    config.stash[_EXAMPLES] = _examples(config.rootpath)
    config.stash[_RUNS] = {}
    config.stash[_COLLECTED] = frozenset()
    config.stash[_SELECTED] = frozenset()
    config.stash[_CHANGED] = frozenset()


def _examples(repo_root: Path) -> frozenset[Path]:
    """The repository's examples, each reached under ``examples/`` without a symlink.

    An example's path from the root is what its page records and what a change names, so a link
    would give it a second name, or put it outside the repository.

    Args:
        repo_root: The repository root.

    Returns:
        The examples, resolved.

    Raises:
        pytest.UsageError: If an example's resolved path differs from its path under the root.
    """
    examples = repo_root / "examples"
    found = discover_examples(examples) if examples.is_dir() else []
    try:
        for path in found:
            example_source(path, repo_root=repo_root)
    except ExamplePathError as error:
        raise pytest.UsageError(str(error)) from error
    return frozenset(path.resolve() for path in found)


def _lane_settings(  # noqa: DOC502  # the ValueError is raised by refuse_unusable_budget
    settings: ExamplesConfig, *, lane_budget: float | None
) -> ExamplesConfig:
    """The settings with this lane's default budget, as ``--examples-budget-seconds`` gives it.

    A lane changes only the repository default: an example's ``budgets`` entry still applies, as
    pytest-timeout's command line yields to an item's own marker.

    Args:
        settings: The repository's settings.
        lane_budget: The lane's default budget, or ``None`` to keep ``budget_seconds``.

    Returns:
        The settings the session runs with.

    Raises:
        ValueError: If ``lane_budget`` is not a finite number above zero.
    """
    if lane_budget is None:
        return settings
    refuse_unusable_budget(lane_budget, setting="--examples-budget-seconds")
    return dataclasses.replace(settings, budget_seconds=lane_budget)


def pytest_collect_directory(path: Path, parent: pytest.Collector) -> pytest.Collector | None:
    """Collect the repository's ``examples`` directory with its corpus check.

    Args:
        path: A directory pytest is about to collect.
        parent: Its parent collector.

    Returns:
        The examples directory's collector, or ``None`` for any other directory.
    """
    if path.resolve() == (parent.config.rootpath / "examples").resolve():
        return ExamplesDirectory.from_parent(parent, path=path)
    return None


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Record which examples' runs the session holds; leave out those a change misses.

    ``example_run`` reads the record: a test of an example's results runs in the examples'
    own invocation, with the example's run, never in a session that did not collect it.

    The suite's own checks then go before every other item, in the order pytest gives them, so a
    test of an example's results reads the run its item made rather than making it. pytest-randomly
    shuffles in a ``tryfirst`` wrapper so that a plugin's stable sort applies on top of its order
    (``pytest_randomly`` 5.0.0, ``pytest_collection_modifyitems``).

    Args:
        config: pytest's configuration.
        items: The collected items, edited in place.
    """
    config.stash[_COLLECTED] = _run_scripts(items)
    _deselect_unchanged(config, items)
    config.stash[_SELECTED] = _run_scripts(items)
    items.sort(key=lambda item: not isinstance(item, _ExampleCheck))


def _run_scripts(items: list[pytest.Item]) -> frozenset[Path]:
    """The examples whose run items are among ``items``."""
    return frozenset(item.path.resolve() for item in items if isinstance(item, ExampleRunItem))


def _deselect_unchanged(config: pytest.Config, items: list[pytest.Item]) -> None:  # noqa: DOC502  # pytest.UsageError is raised by _changed_paths
    """With ``--examples-changed-since``, leave out the runs of the examples a change misses.

    Args:
        config: pytest's configuration.
        items: The collected items, edited in place.

    Raises:
        pytest.UsageError: If the option's value names no commit, or git cannot answer.
    """
    ref = config.getoption("--examples-changed-since")
    if ref is None:
        return
    changed = _changed_paths(config.rootpath, ref)
    config.stash[_CHANGED] = frozenset(changed)
    if reaches_every_example(changed, config.stash[_EXAMPLES], config.rootpath):
        return
    unchanged = [
        item
        for item in items
        if isinstance(item, ExampleRunItem) and not is_touched(item.path, changed, config.rootpath)
    ]
    if unchanged:
        config.hook.pytest_deselected(items=unchanged)
        items[:] = [item for item in items if item not in unchanged]


def _changed_paths(repo_root: Path, ref: str) -> set[str]:
    """The paths changed since ``ref``, a usage error when git cannot say which.

    Args:
        repo_root: The repository root.
        ref: The value of ``--examples-changed-since``.

    Returns:
        The changed paths, relative to the root.

    Raises:
        pytest.UsageError: If ``ref`` names no commit, or git cannot answer.
    """
    try:
        return repository.changed_since(repo_root, ref)
    except repository.RepositoryError as error:
        raise pytest.UsageError(f"--examples-changed-since: {error}") from error


@pytest.fixture
def example_run(request: pytest.FixtureRequest) -> ExampleRunner:
    """Give a test the session's run of an example, by its path below ``examples``.

    Args:
        request: The requesting test.

    Returns:
        A function from the example's path below ``examples`` to its run; it skips the test when
        the run was skipped (a ``GPU-required`` example with no GPU here, or a configured
        ``unavailable`` reason), or when ``--examples-changed-since`` left the example's run out
        and changed neither the test's own file nor a file beside the result tests that is no
        test module (a helper any of them may import), in which case the example runs for the
        test, as pytest-picked runs a changed test file; and fails it for any other failure, or
        when the session did not collect the example's run at all: a test of an example's results
        runs with its example, as ``uv run substrax-notebooks test`` runs them.
    """
    config = request.config

    def run(relative: str) -> ExampleRun:
        script = (config.rootpath / "examples" / relative).resolve()
        if script not in config.stash[_EXAMPLES]:
            pytest.fail(f"{relative} is not an example under {config.rootpath / 'examples'}")
        if script not in config.stash[_COLLECTED]:
            pytest.fail(
                f"{relative} does not run in this session; a test of an example's results runs "
                "with its example: uv run substrax-notebooks test",
                pytrace=False,
            )
        if script not in config.stash[_SELECTED] and not _changed_result_test(request.path, config):
            pytest.skip(
                f"{relative} is not run in this session: no change since the ref touched it"
            )
        return _run_or_skip(config, script)

    return run


class ExamplesDirectory(pytest.Directory):
    """The ``examples`` directory: its corpus check, then the examples below it.

    Only examples live under ``examples``, so the directory collects them itself, as
    :func:`substrax.examples.discover_examples` lists them, and a private file there is not one;
    no other plugin collects a file there, so an example named ``test_*.py`` is not imported as
    a test module.
    """

    def collect(self) -> Iterator[pytest.Item | pytest.Collector]:
        """Yield the corpus check, then the examples and example directories directly in it.

        Yields:
            pytest.Item | pytest.Collector: The corpus check, then the collectors below it.
        """
        yield CorpusItem.from_parent(self, name="corpus")
        yield from _collect_below(self)


class ExamplesSubdirectory(pytest.Directory):
    """A directory under ``examples`` holding examples, so a path or node id below it resolves."""

    def collect(self) -> Iterator[pytest.Collector]:
        """Yield the examples and example directories directly in it.

        Yields:
            pytest.Collector: Each example's collector, and each directory holding examples.
        """
        yield from _collect_below(self)


def _collect_below(directory: pytest.Directory) -> Iterator[pytest.Collector]:
    """Each example and each directory holding examples directly in ``directory``, by name.

    pytest matches a command-line path one directory at a time, so a nested example is reached
    through a collector for each directory above it, as pytest's own ``Dir`` collects; and as
    ``Dir`` does, an entry ``pytest_ignore_collect`` refuses (``--ignore``, ``--ignore-glob``) is
    left out.

    Args:
        directory: The examples directory, or one below it.

    Yields:
        pytest.Collector: Each example's collector, and each directory holding examples.
    """
    examples = directory.config.stash[_EXAMPLES]
    for entry in sorted(directory.path.iterdir()):
        if directory.ihook.pytest_ignore_collect(collection_path=entry, config=directory.config):
            continue
        resolved = entry.resolve()
        if entry.is_dir():
            if any(example.is_relative_to(resolved) for example in examples):
                yield ExamplesSubdirectory.from_parent(directory, path=entry)
        elif resolved in examples:
            yield ExampleFile.from_parent(directory, path=entry)


class ExampleFile(pytest.File):
    """One example script: its guide check and its run."""

    def collect(self) -> Iterator[pytest.Item]:
        """Yield the guide check and the run, the run marked ``example_run``.

        Yields:
            pytest.Item: The example's guide check, then its run.
        """
        yield GuideItem.from_parent(self, name="follows-the-guide")
        run = ExampleRunItem.from_parent(self, name="runs-as-its-page-shows")
        run.add_marker(RUN_MARKER)
        yield run


class _ExampleCheck(pytest.Item):
    """An example check, reported by its file and name."""

    def reportinfo(self) -> tuple[Path, int, str]:
        """Name the check by its file and id.

        Returns:
            The file, line 0 and the check's name.
        """
        return self.path, 0, self.name


class CorpusItem(_ExampleCheck):
    """The examples as a whole: they exist, their pages match them, commands name real files."""

    def runtest(self) -> None:
        """Fail with every problem in the corpus, one per line."""
        _fail_with(corpus_problems(self.config.rootpath, self.config.stash[_EXAMPLES]))


class GuideItem(_ExampleCheck):
    """One example against every rule of the shared guide."""

    def runtest(self) -> None:
        """Fail with every rule the example breaks, one per line."""
        _fail_with(
            [
                str(finding)
                for finding in check_example(
                    self.path, repo_root=self.config.rootpath, config=self.config.stash[_CONFIG]
                )
            ]
        )


class ExampleRunItem(_ExampleCheck):
    """One example run as its clean notebook and compared with its page."""

    def runtest(self) -> None:
        """Fail when the page does not show the run; skip for a configured unavailable reason."""
        run = _run_or_skip(self.config, self.path.resolve())
        problems = compare_with_run(
            self.path, run, repo_root=self.config.rootpath, config=self.config.stash[_CONFIG]
        )
        _fail_with([str(problem) for problem in problems])


def _changed_result_test(test_file: Path, config: pytest.Config) -> bool:
    """Whether ``--examples-changed-since`` changed a result test's own file or a helper of it.

    A helper is a file under ``tests/examples`` that is no test module by pytest's default naming
    (``test_*.py``, ``*_test.py``), such as a module of expected values; any result test may
    import it, so a change to it reaches every one.

    Args:
        test_file: The requesting test's file.
        config: pytest's configuration.

    Returns:
        Whether the test's file, or a helper beside the result tests, is among the changed paths.
    """
    changed = config.stash[_CHANGED]
    own = test_file.resolve().relative_to(config.rootpath.resolve()).as_posix()
    return own in changed or any(
        path.startswith("tests/examples/") and not _is_test_module(path) for path in changed
    )


def _is_test_module(path: str) -> bool:
    """Whether pytest's default naming collects ``path`` as a test module."""
    name = PurePosixPath(path).name
    return name.endswith(".py") and (name.startswith("test_") or name.endswith("_test.py"))


def _run_or_skip(config: pytest.Config, script: Path) -> ExampleRun:
    """The session's run of ``script``, made once; skipped for no GPU or an unavailable reason.

    A ``GPU-required`` example that raised ``GPUUnavailableError``, as ``require_gpu()`` does
    without a GPU, is skipped first, whatever the ``unavailable`` signatures say; any other example
    raising the same is held to them like any failure.
    """
    runs = config.stash[_RUNS]
    if script not in runs:
        settings = config.stash[_CONFIG]
        try:
            runs[script] = execute_example(
                script,
                repo_root=config.rootpath,
                distribution=settings.distribution,
                budget_seconds=settings.budget_for(
                    example_source(script, repo_root=config.rootpath)
                ),
            )
        except RunError as error:
            runs[script] = error
    outcome = runs[script]
    if isinstance(outcome, RunError):
        no_gpu = no_gpu_reason(script, outcome)
        if no_gpu is not None:
            pytest.skip(no_gpu)
        # Only the exception the example raised says why it failed: the message around it holds
        # the failing cell's source and traceback, which can name a signature the run never hit.
        raised = outcome.raised
        reason = next(
            (sig for sig in config.stash[_CONFIG].unavailable if raised and sig in raised), None
        )
        if reason is not None:
            pytest.skip(f"unavailable here: {reason}")
        pytest.fail(str(outcome), pytrace=False)
    return outcome


def _fail_with(problems: list[str]) -> None:
    """Fail the current test with its problems, one per line and no traceback, if there are any."""
    if problems:
        pytest.fail("\n".join(problems), pytrace=False)
