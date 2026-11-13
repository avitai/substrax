"""``substrax.testing.example_suite``: a repository's example tests from its settings alone.

Each case builds a repository under ``pytester``: pyproject settings and a conforming example
with its clean notebook and page, committed to git, and no conftest. It then runs pytest on it
in a separate process with the options ``substrax-notebooks test`` gives, as the
repository's CI would.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _example_texts import CONFIG, git, GOOD, GPU_REQUIRED

from substrax.notebooks.cli import examples_pytest_options
from substrax.notebooks.pages import generate_page
from substrax.notebooks.pairs import convert


PYPROJECT = f"""\
[project]
name = "demo"
version = "0"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--ignore=tests/examples"

[tool.substrax.examples]
source_url = "{CONFIG.source_url}"
budget_seconds = {CONFIG.budget_seconds}
unavailable = ["the dataset is not set"]
"""
MKDOCS = "site_name: demo\nvalidation:\n  nav:\n    omitted_files: warn\n"
ARGS = ("-p", "no:randomly", "-p", "no:cacheprovider", "--strict-markers", "-rs")


def _suite(pytester: pytest.Pytester) -> tuple[str, ...]:
    """pytest as ``substrax-notebooks test`` runs it on the pytester repository."""
    return (*ARGS, *examples_pytest_options(pytester.path))


@pytest.fixture(autouse=True)
def home_outside_the_repository(
    pytester: pytest.Pytester,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    # pytester makes its directory, the repository here, the home directory; a kernel then writes
    # its IPython profile into the repository, which no real repository is: home lives elsewhere.
    del pytester
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))


def _repository(pytester: pytest.Pytester, examples: dict[str, str] | None = None) -> Path:
    """A committed repository holding each example (path below ``examples``: text) and its page."""
    repo = pytester.path
    (repo / "pyproject.toml").write_text(PYPROJECT, encoding="utf-8")
    (repo / "mkdocs.yml").write_text(MKDOCS, encoding="utf-8")
    (repo / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")
    (repo / "examples").mkdir()
    git(repo, "init", "-q")
    for relative, text in (examples if examples is not None else {"sorting.py": GOOD}).items():
        script = repo / "examples" / relative
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(text, encoding="utf-8")
        convert(script)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "examples")
    for relative in examples if examples is not None else {"sorting.py": GOOD}:
        script = repo / "examples" / relative
        if not relative.startswith("_"):
            generate_page(script, repo_root=repo, config=CONFIG)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "--allow-empty", "-m", "pages")
    return repo


def _run(pytester: pytest.Pytester, *args: str) -> pytest.RunResult:
    # The examples' own invocation, as substrax-notebooks test runs it.
    return pytester.runpytest_subprocess(*_suite(pytester), "examples", *args)


def _result_tests(pytester: pytest.Pytester, test: str) -> None:
    """A repository's test of an example's results, where the guide puts it."""
    (pytester.path / "tests" / "examples").mkdir(parents=True)
    (pytester.path / "tests" / "examples" / "test_sorting.py").write_text(test, encoding="utf-8")


def test_a_conforming_example_passes_the_corpus_the_guide_and_its_run(
    pytester: pytest.Pytester,
) -> None:
    _repository(pytester)

    result = _run(pytester, "-v")

    result.assert_outcomes(passed=3)
    result.stdout.fnmatch_lines(
        [
            "*examples::corpus PASSED*",
            "*examples/sorting.py::follows-the-guide PASSED*",
            "*examples/sorting.py::runs-as-its-page-shows PASSED*",
        ]
    )


def test_an_example_breaking_a_rule_fails_naming_the_rule(pytester: pytest.Pytester) -> None:
    _repository(
        pytester, {"sorting.py": GOOD.replace("jax.random.key(0)", "jax.random.PRNGKey(0)")}
    )

    result = _run(pytester, "-m", "not example_run")

    result.assert_outcomes(passed=1, failed=1)
    result.stdout.fnmatch_lines(["*typed-keys*"])


def test_an_empty_corpus_fails(pytester: pytest.Pytester) -> None:
    _repository(pytester, {})

    result = _run(pytester)

    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*no example under*"])


def test_a_page_with_no_example_fails_the_corpus(pytester: pytest.Pytester) -> None:
    repo = _repository(pytester)
    orphan = repo / "docs" / "examples" / "renamed.md"
    orphan.write_text("# Renamed\n", encoding="utf-8")

    result = _run(pytester, "-m", "not example_run")

    result.assert_outcomes(passed=1, failed=1)
    result.stdout.fnmatch_lines(["*docs/examples/renamed.md*"])


def test_a_hand_written_command_naming_a_missing_example_fails_the_corpus(
    pytester: pytest.Pytester,
) -> None:
    repo = _repository(pytester)
    (repo / "README.md").write_text(
        "Run `uv run python examples/01_sorting.py`.\n", encoding="utf-8"
    )
    git(repo, "add", "README.md")

    result = _run(pytester, "-m", "not example_run")

    result.assert_outcomes(passed=1, failed=1)
    result.stdout.fnmatch_lines(["*README.md*examples/01_sorting.py*"])


def test_mkdocs_that_lets_a_page_leave_the_nav_fails_the_corpus(pytester: pytest.Pytester) -> None:
    repo = _repository(pytester)
    (repo / "mkdocs.yml").write_text("site_name: demo\n", encoding="utf-8")

    result = _run(pytester, "-m", "not example_run")

    result.assert_outcomes(passed=1, failed=1)
    result.stdout.fnmatch_lines(["*omitted_files*"])


def test_the_shorthand_validation_setting_holds_too(pytester: pytest.Pytester) -> None:
    # mkdocs propagates validation.omitted_files to validation.nav.omitted_files.
    repo = _repository(pytester)
    (repo / "mkdocs.yml").write_text(
        "site_name: demo\nvalidation:\n  omitted_files: warn\n", encoding="utf-8"
    )

    result = _run(pytester, "-m", "not example_run")

    result.assert_outcomes(passed=2)


def test_a_run_failing_for_a_configured_reason_is_skipped_with_it(
    pytester: pytest.Pytester,
) -> None:
    repo = _repository(pytester)
    script = repo / "examples" / "sorting.py"
    script.write_text(
        GOOD.replace(
            "    print(sort_records(make_records()))",
            '    raise RuntimeError("the dataset is not set")',
        ),
        encoding="utf-8",
    )
    convert(script)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "needs data")

    result = _run(pytester, "-m", "example_run")

    result.assert_outcomes(skipped=1)
    result.stdout.fnmatch_lines(["*the dataset is not set*"])


SLOW = GOOD.replace(
    '# %% [markdown]\n"""\n## Results',
    "# %%\nimport time\n\ntime.sleep(1.5)\n\n# %%\ntime.sleep(1.5)\n\n\n"
    '# %% [markdown]\n"""\n## Results',
)
"""The good example, two cells slower: each cell stays inside a 2.5 s budget, together they pass
it, and they stay well inside the repository default."""


def _slow_repository(pytester: pytest.Pytester, settings: str = "") -> Path:
    """A repository whose example is :data:`SLOW`, its page current, with ``settings`` added."""
    assert SLOW != GOOD
    assert CONFIG.budget_seconds > 3
    repo = _repository(pytester)
    script = repo / "examples" / "sorting.py"
    script.write_text(SLOW, encoding="utf-8")
    convert(script)
    generate_page(script, repo_root=repo, config=CONFIG)
    (repo / "pyproject.toml").write_text(PYPROJECT + settings, encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "a slow example")
    return repo


def test_each_example_runs_within_its_own_budget(pytester: pytest.Pytester) -> None:
    """An example's ``budgets`` entry, not the repository default, bounds its run."""
    _slow_repository(pytester, 'budgets = { "examples/sorting.py" = 2.5 }\n')

    result = _run(pytester, "-m", "example_run")

    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*budget of 2.5 s*"])


def test_a_lane_s_budget_replaces_the_repository_default(pytester: pytest.Pytester) -> None:
    _slow_repository(pytester)

    result = _run(pytester, "-m", "example_run", "--examples-budget-seconds=2.5")

    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*budget of 2.5 s*"])


def test_an_example_s_own_budget_outranks_the_lane_s(pytester: pytest.Pytester) -> None:
    """A lane changes only the default, as pytest-timeout's command line yields to a marker."""
    _slow_repository(pytester, 'budgets = { "examples/sorting.py" = 120 }\n')

    result = _run(pytester, "-m", "example_run", "--examples-budget-seconds=2.5")

    result.assert_outcomes(passed=1)


@pytest.mark.parametrize("seconds", ["0", "-1", "inf", "nan"])
def test_a_lane_budget_no_run_can_be_held_to_is_a_usage_error(
    pytester: pytest.Pytester, seconds: str
) -> None:
    _repository(pytester)

    result = _run(pytester, "-m", "example_run", f"--examples-budget-seconds={seconds}")

    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*--examples-budget-seconds must be a finite number*"])


@pytest.mark.parametrize(
    ("body", "failure"),
    [
        pytest.param(
            "    if not make_records():\n"
            '        raise RuntimeError("the dataset is not set")\n'
            '    raise ValueError("the model diverged")',
            "*ValueError: the model diverged*",
            id="a-guard-for-it-in-the-failing-cell",
        ),
        pytest.param(
            '    # Without data this fails with "the dataset is not set".\n'
            '    raise ValueError("the model diverged")',
            "*ValueError: the model diverged*",
            id="a-comment-naming-it-in-the-failing-cell",
        ),
        pytest.param(
            '    open("the dataset is not set.txt", "w").close()\n'
            "    print(sort_records(make_records()))",
            "*the repository changed while the run was in progress*",
            id="a-run-that-raised-nothing",
        ),
    ],
)
def test_a_run_failing_for_another_reason_fails_though_its_message_names_a_configured_one(
    pytester: pytest.Pytester, body: str, failure: str
) -> None:
    """Only the exception the example raised is matched, not the cell or the traceback around it."""
    repo = _repository(pytester)
    script = repo / "examples" / "sorting.py"
    script.write_text(
        GOOD.replace("    print(sort_records(make_records()))", body), encoding="utf-8"
    )
    convert(script)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "diverges")

    result = _run(pytester, "-m", "example_run")

    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines([failure])


def test_only_changed_examples_run_when_asked(pytester: pytest.Pytester) -> None:
    repo = _repository(pytester, {"sorting.py": GOOD, "other.py": GOOD})
    git(repo, "tag", "base")
    other = repo / "examples" / "other.py"
    other.write_text(other.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    result = _run(pytester, "-m", "example_run", "--examples-changed-since=base", "-v")

    result.stdout.fnmatch_lines(["*examples/other.py::runs-as-its-page-shows*"])
    result.stdout.no_fnmatch_line("*examples/sorting.py::runs-as-its-page-shows*")


@pytest.mark.parametrize("ref", ["--output=written.txt", "no-such-ref"])
def test_a_changed_since_ref_naming_no_commit_is_a_usage_error(
    pytester: pytest.Pytester, ref: str
) -> None:
    repo = _repository(pytester)

    result = _run(pytester, "-m", "example_run", f"--examples-changed-since={ref}")

    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines([f"*{ref}*names no commit*"])
    assert not (repo / "written.txt").exists()


@pytest.mark.parametrize("shared", ["_common/helpers.py", "data/records.csv"])
def test_a_change_under_examples_no_example_owns_runs_every_example(
    pytester: pytest.Pytester, shared: str
) -> None:
    """A private helper or a data file may feed any example, so a change to one runs them all."""
    repo = _repository(pytester, {"sorting.py": GOOD, "other.py": GOOD})
    changed = repo / "examples" / shared
    changed.parent.mkdir(parents=True, exist_ok=True)
    changed.write_text("VALUE = 1\n", encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "shared")
    git(repo, "tag", "base")
    changed.write_text("VALUE = 2\n", encoding="utf-8")

    result = _run(pytester, "-m", "example_run", "--examples-changed-since=base", "-v")

    result.assert_outcomes(passed=2)


@pytest.mark.parametrize(
    ("linked", "inside"),
    [
        pytest.param("examples", False, id="the-examples-directory-elsewhere"),
        pytest.param("examples/sorting.py", False, id="one-example-elsewhere"),
        pytest.param("examples/sorting.py", True, id="one-example-elsewhere-in-the-repository"),
    ],
)
def test_an_example_reached_through_a_symlink_is_a_usage_error(
    pytester: pytest.Pytester, linked: str, inside: bool
) -> None:
    """An example's path names it as its page and its changes do, so no link may stand between."""
    repo = _repository(pytester)
    holder = repo / "stash" if inside else repo.parent / f"{repo.name}-elsewhere"
    holder.mkdir()
    target = holder / Path(linked).name
    (repo / linked).rename(target)
    (repo / linked).symlink_to(target, target_is_directory=target.is_dir())

    result = _run(pytester, "-m", "example_run", "--examples-changed-since=HEAD")

    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*symlink*examples/sorting.py*"])


@pytest.mark.parametrize(
    ("setting", "named"),
    [
        pytest.param("budget_seconds = 0", "budget_seconds", id="a-budget-no-run-can-meet"),
        pytest.param("budget_secs = 600", "budget_secs", id="a-misspelled-key"),
        pytest.param(
            f'budget_seconds = {CONFIG.budget_seconds}\npage_size_exempt = ["examples/gone.py"]',
            "examples/gone.py",
            id="an-exemption-naming-no-example",
        ),
    ],
)
def test_settings_no_run_can_be_held_to_are_a_usage_error(
    pytester: pytest.Pytester, setting: str, named: str
) -> None:
    repo = _repository(pytester)
    budget = f"budget_seconds = {CONFIG.budget_seconds}"
    assert budget in PYPROJECT
    (repo / "pyproject.toml").write_text(PYPROJECT.replace(budget, setting), encoding="utf-8")

    result = _run(pytester, "-m", "example_run")

    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines([f"*{named}*"])


def test_a_repository_without_a_pyproject_is_a_usage_error(pytester: pytest.Pytester) -> None:
    """Not an internal error: pytest takes a ``pytest.ini`` directory as its root (R-8)."""
    repo = _repository(pytester)
    (repo / "pyproject.toml").unlink()
    (repo / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")

    result = _run(pytester, "-m", "example_run")

    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines([f"*{repo} has no pyproject.toml*"])


def test_a_repository_naming_no_distribution_is_a_usage_error(pytester: pytest.Pytester) -> None:
    repo = _repository(pytester)
    named = '[project]\nname = "demo"\n'
    assert PYPROJECT.startswith(named)
    (repo / "pyproject.toml").write_text(PYPROJECT.removeprefix(named), encoding="utf-8")

    result = _run(pytester, "-m", "example_run")

    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*names no ?project?*"])


def test_a_library_change_runs_every_example(pytester: pytest.Pytester) -> None:
    repo = _repository(pytester, {"sorting.py": GOOD, "other.py": GOOD})
    git(repo, "tag", "base")
    (repo / "src").mkdir()
    (repo / "src" / "library.py").write_text("VALUE = 1\n", encoding="utf-8")
    git(repo, "add", "-A")

    result = _run(pytester, "-m", "example_run", "--examples-changed-since=base", "-v")

    result.assert_outcomes(passed=2)


def test_a_domain_test_reads_the_same_run_through_the_fixture(pytester: pytest.Pytester) -> None:
    returning = GOOD.replace(
        "def main() -> None:\n    print(sort_records(make_records()))",
        "def main() -> dict[str, int]:\n"
        "    records = sort_records(make_records())\n"
        "    print(records)\n"
        '    return {"count": len(records)}',
    )
    _repository(pytester, {"sorting.py": returning})
    _result_tests(
        pytester,
        "def test_every_record_is_kept(example_run):\n"
        '    assert example_run("sorting.py").returned == {"count": 5}\n',
    )

    result = _run(pytester, "tests/examples")

    result.assert_outcomes(passed=4)


def test_a_private_file_under_examples_is_not_an_example(pytester: pytest.Pytester) -> None:
    _repository(pytester, {"sorting.py": GOOD, "_common/helpers.py": "VALUE = 1\n"})

    result = _run(pytester, "--collect-only", "-q")

    result.stdout.no_fnmatch_line("*_common*")
    result.stdout.fnmatch_lines(["*3 tests collected*"])


def test_the_example_suite_leaves_a_repositorys_own_tests_to_their_own_plugins(
    pytester: pytest.Pytester,
) -> None:
    # Enabling the shared example tests must not also enable substrax.testing.pytest_plugin's
    # per-test check of jax's configuration on the repository's own tests.
    repo = _repository(pytester)
    (repo / "tests").mkdir()
    (repo / "tests" / "test_own.py").write_text(
        "import jax\n\n\ndef test_turns_x64_on() -> None:\n"
        '    jax.config.update("jax_enable_x64", True)\n',
        encoding="utf-8",
    )

    result = pytester.runpytest_subprocess(*_suite(pytester), "tests")

    result.assert_outcomes(passed=1)


_EVERY_RECORD_KEPT = (
    "def test_every_record_is_kept(example_run):\n"
    '    assert example_run("sorting.py").returned is None\n'
)


def test_a_plain_pytest_run_collects_no_example_and_no_result_test(
    pytester: pytest.Pytester,
) -> None:
    _repository(pytester)
    _result_tests(pytester, _EVERY_RECORD_KEPT)
    (pytester.path / "tests" / "test_unit.py").write_text("def test_unit():\n    pass\n")

    result = pytester.runpytest_subprocess(*ARGS, "--collect-only", "-q")

    result.stdout.fnmatch_lines(["tests/test_unit.py::test_unit", "*1 test collected*"])
    result.stdout.no_fnmatch_line("*examples*")


def test_example_run_outside_the_examples_session_fails_naming_the_invocation(
    pytester: pytest.Pytester,
) -> None:
    _repository(pytester)
    _result_tests(pytester, _EVERY_RECORD_KEPT)

    result = pytester.runpytest_subprocess(*_suite(pytester), "tests/examples")

    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*runs with its example: uv run substrax-notebooks test*"])


def _committed_result_tests(pytester: pytest.Pytester, test: str) -> Path:
    """The repository with ``test`` committed as its result test, tagged ``base`` after it."""
    repo = _repository(pytester)
    _result_tests(pytester, test)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "result tests")
    git(repo, "tag", "base")
    return repo


def test_an_unchanged_result_test_of_an_example_a_change_left_out_is_skipped(
    pytester: pytest.Pytester,
) -> None:
    _committed_result_tests(pytester, _EVERY_RECORD_KEPT)

    result = _run(pytester, "tests/examples", "--examples-changed-since=base", "-rs")

    result.assert_outcomes(passed=2, skipped=1, deselected=1)
    result.stdout.fnmatch_lines(["*not run in this session*"])


def test_a_result_test_that_changed_runs_its_example(pytester: pytest.Pytester) -> None:
    """As pytest-picked runs a changed test file: the test asks for its example, which runs."""
    repo = _repository(pytester)
    git(repo, "tag", "base")
    _result_tests(pytester, _EVERY_RECORD_KEPT)

    result = _run(pytester, "tests/examples", "--examples-changed-since=base", "-rs")

    result.assert_outcomes(passed=3, deselected=1)


def test_another_changed_test_module_does_not_run_an_example_it_does_not_ask_for(
    pytester: pytest.Pytester,
) -> None:
    """A changed test module reaches the examples it asks for, not those its neighbours do."""
    repo = _committed_result_tests(pytester, _EVERY_RECORD_KEPT)
    (repo / "tests" / "examples" / "test_other.py").write_text(
        "def test_other():\n    assert True\n", encoding="utf-8"
    )

    result = _run(pytester, "tests/examples", "--examples-changed-since=base", "-rs")

    result.assert_outcomes(passed=3, skipped=1, deselected=1)
    result.stdout.fnmatch_lines(["*not run in this session*"])


def test_a_changed_helper_beside_the_result_tests_runs_the_examples_they_ask_for(
    pytester: pytest.Pytester,
) -> None:
    """A file under ``tests/examples`` that is no test module may be imported by any of them."""
    repo = _committed_result_tests(pytester, _EVERY_RECORD_KEPT)
    (repo / "tests" / "examples" / "records.py").write_text("EXPECTED = 3\n", encoding="utf-8")

    result = _run(pytester, "tests/examples", "--examples-changed-since=base", "-rs")

    result.assert_outcomes(passed=3, deselected=1)


def _collected(pytester: pytest.Pytester, *arguments: str) -> tuple[int, list[str]]:
    """The examples session's exit code and node ids for ``--collect-only`` over ``arguments``."""
    result = pytester.runpytest_subprocess(*_suite(pytester), "--collect-only", "-q", *arguments)
    return result.ret, [line for line in result.outlines if "::" in line]


NESTED = [
    "examples/topic/nested.py::follows-the-guide",
    "examples/topic/nested.py::runs-as-its-page-shows",
]
"""The items of the nested example of :func:`_nested_repository`."""


def _nested_repository(pytester: pytest.Pytester) -> Path:
    """A repository with a top-level example and one in a subdirectory, as the guide places them."""
    return _repository(pytester, {"sorting.py": GOOD, "topic/nested.py": GOOD})


@pytest.mark.parametrize(
    "selection",
    ["examples/topic/nested.py", "examples/topic"],
    ids=["by-its-path", "by-its-directory"],
)
def test_a_nested_example_is_selected_by_its_path_or_directory(
    pytester: pytest.Pytester, selection: str
) -> None:
    """pytest matches a command-line path one directory at a time (M-1)."""
    _nested_repository(pytester)

    assert _collected(pytester, selection) == (pytest.ExitCode.OK, NESTED)


def test_a_nested_example_s_node_id_selects_its_item(pytester: pytest.Pytester) -> None:
    """The node id pytest prints for a failure can be passed back to rerun it."""
    _nested_repository(pytester)

    assert _collected(pytester, NESTED[0]) == (pytest.ExitCode.OK, NESTED[:1])


def test_an_ignored_directory_under_examples_is_not_collected(pytester: pytest.Pytester) -> None:
    _nested_repository(pytester)

    code, collected = _collected(pytester, "examples", "--ignore=examples/topic")

    assert code == pytest.ExitCode.OK
    assert collected == [
        "examples::corpus",
        "examples/sorting.py::follows-the-guide",
        "examples/sorting.py::runs-as-its-page-shows",
    ]


def test_an_example_directory_named_like_a_figures_directory_is_not_an_orphan(
    pytester: pytest.Pytester,
) -> None:
    _repository(pytester, {"sorting.py": GOOD, "data_files/loading.py": GOOD})

    result = _run(pytester, "-m", "not example_run")

    result.assert_outcomes(passed=3)


def test_a_figures_directory_with_no_page_fails_the_corpus(pytester: pytest.Pytester) -> None:
    repo = _repository(pytester)
    orphan = repo / "docs" / "examples" / "renamed_files"
    orphan.mkdir()
    (orphan / "renamed_1_0.png").write_bytes(b"png")

    result = _run(pytester, "-m", "not example_run")

    result.assert_outcomes(passed=1, failed=1)
    result.stdout.fnmatch_lines(
        ["*docs/examples/renamed_files/renamed_1_0.png: docs/examples holds generated pages*"]
    )


def test_each_example_runs_before_any_test_of_its_results(pytester: pytest.Pytester) -> None:
    """A result test collected first would otherwise make the run itself, inside the test."""
    _repository(pytester)
    _result_tests(pytester, _EVERY_RECORD_KEPT)

    result = pytester.runpytest_subprocess(*_suite(pytester), "tests/examples", "examples", "-v")

    result.assert_outcomes(passed=4)
    result.stdout.fnmatch_lines(
        ["*sorting.py::runs-as-its-page-shows PASSED*", "*::test_every_record_is_kept PASSED*"]
    )


_CALLS_REQUIRE_GPU = (
    "import jax\n",
    "import jax\n\nfrom substrax.devices import require_gpu\n\nrequire_gpu()\n",
)
_REPORTS_JAX = """\
import sys


def pytest_terminal_summary(terminalreporter):
    terminalreporter.write_line(f"jax imported in the pytest process: {'jax' in sys.modules}")
"""


def _edit_example(repo: Path, text: str) -> None:
    """Give the committed example ``text``, with its notebook, after its page was made."""
    script = repo / "examples" / "sorting.py"
    script.write_text(text, encoding="utf-8")
    convert(script)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "the example changed after its page")


def _checking_its_gpu(text: str) -> str:
    """``text`` with ``require_gpu()`` called at the top level of its Setup."""
    old, new = _CALLS_REQUIRE_GPU
    assert old in text
    return text.replace(old, new, 1)


def test_a_gpu_required_example_s_own_refusal_skips_its_run_and_its_result_tests(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repository(pytester, {"sorting.py": GPU_REQUIRED})
    _edit_example(repo, _checking_its_gpu(GPU_REQUIRED))
    _result_tests(pytester, _EVERY_RECORD_KEPT)
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")

    result = _run(pytester, "tests/examples", "-k", "runs or every_record")

    result.assert_outcomes(skipped=2)
    result.stdout.fnmatch_lines(["*GPU-required*GPUUnavailableError: no GPU*"])


def test_the_gpu_skip_reads_the_class_whatever_raised_it(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``substrax.devices``' own class, raised by the example itself with a message of its own."""
    repo = _repository(pytester, {"sorting.py": GPU_REQUIRED})
    body = "    print(sort_records(make_records()))"
    assert body in GPU_REQUIRED
    raises_it = (
        "    from substrax.devices import GPUUnavailableError\n\n"
        '    raise GPUUnavailableError("this step needs an accelerator")'
    )
    _edit_example(repo, GPU_REQUIRED.replace(body, raises_it))
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")

    result = _run(pytester, "-m", "example_run")

    result.assert_outcomes(skipped=1)
    result.stdout.fnmatch_lines(["*GPUUnavailableError: this step needs an accelerator*"])


def test_a_gpu_optional_example_s_refusal_fails_its_run(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only an example that declares it needs a GPU is skipped for not having one."""
    optional = GOOD.replace("| **Devices** | CPU-compatible |", "| **Devices** | GPU-optional |")
    repo = _repository(pytester, {"sorting.py": optional})
    _edit_example(repo, _checking_its_gpu(optional))
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")

    result = _run(pytester, "-m", "example_run")

    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*GPUUnavailableError: no GPU*"])


def test_a_gpu_required_example_failing_for_another_reason_fails(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repository(pytester, {"sorting.py": GPU_REQUIRED})
    _edit_example(
        repo,
        GPU_REQUIRED.replace(
            "    print(sort_records(make_records()))", '    raise ValueError("the model diverged")'
        ),
    )
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")

    result = _run(pytester, "-m", "example_run")

    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*ValueError: the model diverged*"])


def test_the_gpu_refusal_is_read_before_the_unavailable_signatures(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repository(pytester, {"sorting.py": GPU_REQUIRED})
    pyproject = repo / "pyproject.toml"
    signature = 'unavailable = ["the dataset is not set"]'
    assert signature in pyproject.read_text(encoding="utf-8")
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8").replace(signature, 'unavailable = ["no GPU"]'),
        encoding="utf-8",
    )
    _edit_example(repo, _checking_its_gpu(GPU_REQUIRED))
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")

    result = _run(pytester, "-m", "example_run", "-rs")

    result.assert_outcomes(skipped=1)
    result.stdout.fnmatch_lines(["*GPU-required*GPUUnavailableError: no GPU*"])
    result.stdout.no_fnmatch_line("*unavailable here*")


def test_the_pytest_process_never_imports_jax_for_a_gpu_required_example(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A device probe in the parent would hold a CUDA context the example's kernel then lacks."""
    repo = _repository(pytester, {"sorting.py": GPU_REQUIRED})
    _edit_example(repo, _checking_its_gpu(GPU_REQUIRED))
    (pytester.path / "reports_jax.py").write_text(_REPORTS_JAX, encoding="utf-8")
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")

    result = _run(pytester, "-m", "example_run", "-p", "reports_jax")

    result.assert_outcomes(skipped=1)
    result.stdout.fnmatch_lines(["jax imported in the pytest process: False"])


def test_the_example_suite_adds_no_device_marker(pytester: pytest.Pytester) -> None:
    _repository(pytester)

    result = pytester.runpytest_subprocess(*_suite(pytester), "--markers")

    result.stdout.fnmatch_lines(["@pytest.mark.example_run*"])
    result.stdout.no_fnmatch_line("@pytest.mark.accelerator*")
    result.stdout.no_fnmatch_line("@pytest.mark.devices*")


@pytest.mark.parametrize(
    ("main_body", "raised"),
    [
        pytest.param(
            "    class GPUUnavailableError(Exception):\n        pass\n\n"
            '    raise GPUUnavailableError("the dataset is gone")',
            "*GPUUnavailableError: the dataset is gone*",
            id="a-class-of-its-own-with-the-name",
        ),
        pytest.param(
            "    class GPUUnavailableError(Exception):\n        pass\n\n"
            '    raise GPUUnavailableError("no GPU: the default backend is cpu")',
            "*GPUUnavailableError: no GPU*",
            id="a-class-of-its-own-with-the-name-and-the-message",
        ),
        pytest.param(
            "    from substrax.devices import GPUUnavailableError, require_gpu\n\n"
            "    try:\n        require_gpu()\n    except GPUUnavailableError as error:\n"
            "        raise RuntimeError(str(error)) from error",
            "*RuntimeError: no GPU*",
            id="the-refusal-wrapped",
        ),
        pytest.param(
            "    from substrax.devices import GPUUnavailableError\n\n"
            "    class NoAccelerator(GPUUnavailableError):\n        pass\n\n"
            '    raise NoAccelerator("no GPU: the default backend is cpu")',
            "*NoAccelerator: no GPU*",
            id="the-refusal-renamed",
        ),
    ],
)
def test_a_gpu_refusal_require_gpu_did_not_raise_fails_the_run(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, main_body: str, raised: str
) -> None:
    """The GPU skip is the qualified name of substrax's class, as the kernel recorded it."""
    repo = _repository(pytester, {"sorting.py": GPU_REQUIRED})
    body = "    print(sort_records(make_records()))"
    assert body in GPU_REQUIRED
    _edit_example(repo, GPU_REQUIRED.replace(body, main_body))
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")

    result = _run(pytester, "-m", "example_run")

    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines([raised])
    result.stdout.no_fnmatch_line("*no GPU here*")


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 6])
def test_each_example_runs_before_any_test_of_its_results_under_pytest_randomly(
    pytester: pytest.Pytester, seed: int
) -> None:
    """pytest-randomly shuffles first, in a tryfirst wrapper; the suite's stable sort follows."""
    _repository(pytester)
    _result_tests(pytester, _EVERY_RECORD_KEPT)
    options = examples_pytest_options(pytester.path)
    randomly = ("-p", "no:cacheprovider", "-rs", *options, "-p", "randomly")

    result = pytester.runpytest_subprocess(
        *randomly, f"--randomly-seed={seed}", "tests/examples", "examples", "-v"
    )

    result.assert_outcomes(passed=4)
    result.stdout.fnmatch_lines(
        ["*sorting.py::runs-as-its-page-shows PASSED*", "*::test_every_record_is_kept PASSED*"]
    )
