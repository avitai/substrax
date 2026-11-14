"""``substrax-notebooks``: convert, check, page and test examples from the command line."""

from __future__ import annotations

import configparser
import json
import re
import tomllib
from pathlib import Path

import pytest
import yaml
from _example_texts import (
    CONFIG,
    git,
    GOOD,
    GPU_REQUIRED,
    returning_environment,
    write_example,
)

from substrax.notebooks.cli import EXAMPLES_PYTEST_CONFIG, main
from substrax.notebooks.conformance import RULES, SOURCE_RULES
from substrax.notebooks.pages import generate_page, page_path


REPOSITORY = Path(__file__).resolve().parents[2]


def test_source_rules_are_every_rule_but_the_page() -> None:
    assert frozenset(RULES) - {"page"} == SOURCE_RULES


def test_convert_writes_each_clean_notebook(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo, script = write_example(tmp_path)
    notebook = script.with_suffix(".ipynb")
    notebook.unlink()

    assert main(["convert", "--repo-root", str(repo), str(script)]) == 0

    assert json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    assert str(notebook) in capsys.readouterr().out


def test_check_passes_a_conforming_example_without_running_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo, script = write_example(tmp_path)

    assert main(["check", "--repo-root", str(repo), str(script)]) == 0

    assert capsys.readouterr().out == ""
    assert not page_path(script, repo_root=repo).exists()


def test_check_prints_each_finding_and_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo, script = write_example(
        tmp_path, GOOD.replace("jax.random.key(0)", "jax.random.PRNGKey(0)")
    )

    assert main(["check", "--repo-root", str(repo), str(script)]) == 1

    assert f"{script}: typed-keys: " in capsys.readouterr().out


def test_check_finds_a_notebook_out_of_step_with_its_script(tmp_path: Path) -> None:
    repo, script = write_example(tmp_path)
    script.write_text(GOOD.replace("sorted(records)", "sorted(records, reverse=True)"))

    assert main(["check", "--repo-root", str(repo), str(script)]) == 1


def test_check_skips_a_private_file_under_examples(tmp_path: Path) -> None:
    repo, _ = write_example(tmp_path)
    helper = repo / "examples" / "_common" / "helpers.py"
    helper.parent.mkdir()
    helper.write_text("VALUE = 1\n", encoding="utf-8")

    assert main(["check", "--repo-root", str(repo), str(helper)]) == 0


def test_an_example_reached_through_a_linked_examples_directory_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The plugin refuses this layout, and every command decides an example's path as it does."""
    repo, script = write_example(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    (repo / "examples").rename(elsewhere)
    (repo / "examples").symlink_to(elsewhere, target_is_directory=True)

    assert main(["check", "--repo-root", str(repo), str(script)]) == 2

    assert "reached through a symlink" in capsys.readouterr().err


def test_a_path_outside_examples_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo, _ = write_example(tmp_path)
    outside = repo / "scripts" / "tool.py"
    outside.parent.mkdir()
    outside.write_text("VALUE = 1\n", encoding="utf-8")

    assert main(["check", "--repo-root", str(repo), str(outside)]) == 2

    assert "examples" in capsys.readouterr().err


def test_a_missing_example_is_refused(tmp_path: Path) -> None:
    repo, _ = write_example(tmp_path)

    assert main(["check", "--repo-root", str(repo), str(repo / "examples" / "typo.py")]) == 2


def test_check_given_a_notebook_checks_its_script(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo, script = write_example(
        tmp_path, GOOD.replace("jax.random.key(0)", "jax.random.PRNGKey(0)")
    )
    notebook = script.with_suffix(".ipynb")

    assert main(["check", "--repo-root", str(repo), str(notebook)]) == 1

    assert f"{script}: typed-keys: " in capsys.readouterr().out


def test_a_script_named_with_its_notebook_is_checked_once(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The hook passes both files of a pair that changed together."""
    repo, script = write_example(
        tmp_path, GOOD.replace("jax.random.key(0)", "jax.random.PRNGKey(0)")
    )
    notebook = script.with_suffix(".ipynb")

    assert main(["check", "--repo-root", str(repo), str(script), str(notebook)]) == 1

    assert capsys.readouterr().out.count(": typed-keys: ") == 1


def test_convert_given_a_notebook_writes_it_again_from_its_script(tmp_path: Path) -> None:
    repo, script = write_example(tmp_path)
    notebook = script.with_suffix(".ipynb")
    clean = notebook.read_bytes()
    notebook.write_text("{}", encoding="utf-8")

    assert main(["convert", "--repo-root", str(repo), str(notebook)]) == 0

    assert notebook.read_bytes() == clean


def test_a_path_that_is_neither_a_script_nor_a_notebook_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo, _ = write_example(tmp_path)
    notes = repo / "examples" / "notes.md"
    notes.write_text("# Notes\n", encoding="utf-8")

    assert main(["check", "--repo-root", str(repo), str(notes)]) == 2

    assert capsys.readouterr().err == (
        f"substrax-notebooks: {notes} is not an example script (.py) or its notebook (.ipynb)\n"
    )


def test_a_notebook_with_no_script_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo, script = write_example(tmp_path)
    orphan = script.with_name("renamed.ipynb")
    script.with_suffix(".ipynb").rename(orphan)

    assert main(["check", "--repo-root", str(repo), str(orphan)]) == 2

    assert capsys.readouterr().err == (
        f"substrax-notebooks: {orphan} is a notebook with no script "
        f"{orphan.with_suffix('.py')}; convert writes a notebook from its script\n"
    )


def test_page_reports_a_gpu_required_example_off_a_gpu_in_one_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The run's own refusal, without a device probe in this process (BR-10)."""
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    repo, script = write_example(
        tmp_path,
        GPU_REQUIRED.replace(
            "import jax\n",
            "import jax\n\nfrom substrax.devices import require_gpu\n\nrequire_gpu()\n",
            1,
        ),
    )

    assert main(["page", "--repo-root", str(repo), str(script)]) == 1

    assert capsys.readouterr().err == (
        f"substrax-notebooks: {script}: GPU-required and no GPU here: "
        "GPUUnavailableError: no GPU: the default backend is cpu\n"
    )
    assert not page_path(script, repo_root=repo).exists()


def test_page_writes_each_page(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo, script = write_example(tmp_path)

    assert main(["page", "--repo-root", str(repo), str(script)]) == 0

    page = page_path(script, repo_root=repo)
    assert page.is_file()
    assert str(page) in capsys.readouterr().out


def test_page_reports_a_failing_example_and_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo, script = write_example(
        tmp_path, GOOD.replace("return sorted(records)", 'raise RuntimeError("broken step")')
    )

    assert main(["page", "--repo-root", str(repo), str(script)]) == 1

    assert "broken step" in capsys.readouterr().err
    assert not page_path(script, repo_root=repo).exists()


@pytest.mark.parametrize("command", ["check", "page"])
def test_a_command_reading_the_settings_refuses_a_repository_without_them(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], command: str
) -> None:
    repo, script = write_example(tmp_path)
    (repo / "pyproject.toml").write_text('[project]\nname = "demo"\n', encoding="utf-8")

    assert main([command, "--repo-root", str(repo), str(script)]) == 2

    assert "[tool.substrax.examples]" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["check", "page"])
def test_a_command_refuses_a_repository_naming_no_distribution(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], command: str
) -> None:
    repo, script = write_example(tmp_path)
    pyproject = repo / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    assert text.startswith('[project]\nname = "demo"\nversion = "0"\n')
    pyproject.write_text(text.removeprefix('[project]\nname = "demo"\n'), encoding="utf-8")

    assert main([command, "--repo-root", str(repo), str(script)]) == 2

    assert "[project]" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["check", "page"])
def test_a_command_refuses_settings_no_run_can_be_held_to(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], command: str
) -> None:
    repo, script = write_example(tmp_path)
    pyproject = repo / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    assert "budget_seconds = " in text
    pyproject.write_text(
        re.sub(r"budget_seconds = \S+", "budget_seconds = 0", text), encoding="utf-8"
    )

    assert main([command, "--repo-root", str(repo), str(script)]) == 2

    assert "budget_seconds" in capsys.readouterr().err


def test_the_shared_hook_checks_changed_examples_in_the_repository_environment() -> None:
    hooks = yaml.safe_load((REPOSITORY / ".pre-commit-hooks.yaml").read_text(encoding="utf-8"))
    hook = next(hook for hook in hooks if hook["id"] == "example-conformance")

    assert hook["entry"] == "uv run --no-sync substrax-notebooks check"
    assert hook["language"] == "system"
    assert hook["files"] == r"^examples/.*\.(py|ipynb)$"


@pytest.mark.parametrize(
    ("path", "watched"),
    [
        ("examples/topic/sorting.py", True),
        ("examples/topic/sorting.ipynb", True),
        ("examples/topic/notes.md", False),
        ("docs/examples/topic/sorting.md", False),
    ],
)
def test_the_hook_watches_a_script_and_a_hand_edited_notebook(path: str, watched: bool) -> None:
    """A notebook edited by hand leaves its pair; the hook checks it as it checks the script."""
    hooks = yaml.safe_load((REPOSITORY / ".pre-commit-hooks.yaml").read_text(encoding="utf-8"))
    hook = next(hook for hook in hooks if hook["id"] == "example-conformance")

    assert (re.search(hook["files"], path) is not None) is watched


def test_the_command_is_installed_as_substrax_notebooks() -> None:
    scripts = tomllib.loads((REPOSITORY / "pyproject.toml").read_text(encoding="utf-8"))
    assert scripts["project"]["scripts"]["substrax-notebooks"] == "substrax.notebooks.cli:main"


_RETURNS_THE_PROBE = returning_environment("PROBE_VAR")

_READS_THE_SHELLS_PROBE = (
    "def test_the_kernel_had_the_shells_environment(example_run):\n"
    '    assert example_run("sorting.py").returned == {"PROBE_VAR": "from-shell"}\n'
)


def _examples_repository(tmp_path: Path, *, result_tests: bool = True) -> Path:
    """A committed repository whose example returns its kernel's PROBE_VAR, with its page."""
    assert "return {'PROBE_VAR'" in _RETURNS_THE_PROBE
    repo, script = write_example(tmp_path, _RETURNS_THE_PROBE)
    generate_page(script, repo_root=repo, config=CONFIG)
    pyproject = repo / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8")
        + '\n[tool.pytest.ini_options]\ntestpaths = ["tests"]\n'
        + 'addopts = "--ignore=tests/examples"\n',
        encoding="utf-8",
    )
    if result_tests:
        (repo / "tests" / "examples").mkdir(parents=True)
        (repo / "tests" / "examples" / "test_sorting.py").write_text(
            _READS_THE_SHELLS_PROBE, encoding="utf-8"
        )
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "page and settings")
    return repo


@pytest.mark.parametrize(
    ("path", "setting"),
    [
        pytest.param(
            "conftest.py", 'import os\nos.environ["PROBE_VAR"] = "from-conftest"\n', id="conftest"
        ),
        pytest.param("pyproject.toml", 'env = ["PROBE_VAR=from-pytest-env"]\n', id="pytest-env"),
        pytest.param(
            "pyproject.toml",
            'addopts = "--ignore=tests/examples -k no_test_has_this_name"\n',
            id="addopts",
        ),
        pytest.param("pyproject.toml", "timeout = 0.5\n", id="pytest-timeout-ini"),
        pytest.param(
            "pyproject.toml",
            "faulthandler_timeout = 0.5\nfaulthandler_exit_on_timeout = true\n",
            id="faulthandler-exit-ini",
        ),
        pytest.param(
            "pyproject.toml",
            'required_plugins = ["an-uninstalled-plugin"]\n',
            id="required-plugins",
        ),
        pytest.param(
            "pyproject.toml", "strict_config = true\ntimeout = 0.5\n", id="strict-config-ini"
        ),
    ],
)
def test_test_runs_the_examples_apart_from_the_repositorys_test_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path: str, setting: str
) -> None:
    """Each kernel gets the environment a script run gets, whatever the unit tests configure."""
    monkeypatch.setenv("PROBE_VAR", "from-shell")
    repo = _examples_repository(tmp_path)
    if path == "pyproject.toml":
        _set_in_the_unit_tests_settings(repo, setting)
    else:
        (repo / path).write_text(setting, encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "the unit tests' settings")

    assert main(["test", "--repo-root", str(repo)]) == 0


@pytest.mark.parametrize(
    ("setting", "result_test"),
    [
        pytest.param(
            'filterwarnings = ["error"]\n',
            "def test_a_library_warns():\n    import warnings\n\n"
            "    warnings.warn('a library deprecation', DeprecationWarning)\n",
            id="filterwarnings",
        ),
        pytest.param(
            'pythonpath = ["helpers"]\n',
            "def test_the_unit_tests_path_is_not_on_sys_path():\n    import importlib.util\n\n"
            "    assert importlib.util.find_spec('only_on_the_unit_tests_path') is None\n",
            id="pythonpath",
        ),
        pytest.param('usefixtures = ["no_such_fixture"]\n', "", id="usefixtures"),
        pytest.param(
            "xfail_strict = true\n",
            "import pytest\n\n\n@pytest.mark.xfail(reason='a known gap')\n"
            "def test_a_known_gap():\n    pass\n",
            id="xfail-strict",
        ),
    ],
)
def test_test_reads_none_of_the_repositorys_ini_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, setting: str, result_test: str
) -> None:
    """Each key would change how a test of the results runs, were it read."""
    monkeypatch.setenv("PROBE_VAR", "from-shell")
    repo = _examples_repository(tmp_path)
    _set_in_the_unit_tests_settings(repo, setting)
    (repo / "helpers").mkdir()
    (repo / "helpers" / "only_on_the_unit_tests_path.py").write_text("X = 1\n", encoding="utf-8")
    results = repo / "tests" / "examples" / "test_sorting.py"
    results.write_text(results.read_text(encoding="utf-8") + f"\n\n{result_test}", encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "the unit tests' settings and tests reacting to them")

    assert main(["test", "--repo-root", str(repo)]) == 0


@pytest.mark.parametrize(
    ("name", "status"),
    [
        pytest.param("sorting_test.py", 1, id="a-default-name-is-collected"),
        pytest.param("test_sorting_more.py", 1, id="the-other-default-name-is-collected"),
        pytest.param("check_sorting.py", 0, id="another-name-is-not"),
    ],
)
def test_test_collects_results_tests_by_pytest_s_default_naming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, status: int
) -> None:
    """The repository's ``python_files`` never reaches the session; a collected test fails it.

    The repository names ``check_*.py`` alone, so each case fails were its naming read.
    """
    monkeypatch.setenv("PROBE_VAR", "from-shell")
    repo = _examples_repository(tmp_path)
    _set_in_the_unit_tests_settings(repo, 'python_files = ["check_*.py"]\n')
    (repo / "tests" / "examples" / name).write_text(
        "def test_a_result_that_does_not_hold():\n    assert False\n", encoding="utf-8"
    )
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "a test of the results, named one way or another")

    assert main(["test", "--repo-root", str(repo)]) == status


def test_test_collects_a_results_test_named_as_a_path_whatever_its_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    """pytest collects a file its command line names; the default naming sorts files it walks to."""
    monkeypatch.setenv("PROBE_VAR", "from-shell")
    repo = _examples_repository(tmp_path)
    named = repo / "tests" / "examples" / "check_sorting.py"
    named.write_text(
        "def test_a_result_that_does_not_hold():\n    assert False\n", encoding="utf-8"
    )
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "a test of the results named otherwise")

    assert main(["test", "--repo-root", str(repo), str(named)]) == 1

    assert "check_sorting.py::test_a_result_that_does_not_hold" in capfd.readouterr().out


def _set_in_the_unit_tests_settings(repo: Path, setting: str) -> None:
    """Add ``setting`` to the repository's pytest table; an ``addopts`` replaces the table's."""
    pyproject = repo / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    line = 'addopts = "--ignore=tests/examples"\n'
    assert line in text
    pyproject.write_text(
        text.replace(line, setting if setting.startswith("addopts") else line + setting),
        encoding="utf-8",
    )


def test_the_examples_session_s_configuration_sets_nothing() -> None:
    parser = configparser.ConfigParser()
    assert parser.read(EXAMPLES_PYTEST_CONFIG, encoding="utf-8") == [str(EXAMPLES_PYTEST_CONFIG)]

    assert parser.sections() == ["pytest"]
    assert dict(parser["pytest"]) == {}


def test_test_runs_without_a_tests_examples_directory(tmp_path: Path) -> None:
    repo = _examples_repository(tmp_path, result_tests=False)

    assert main(["test", "--repo-root", str(repo)]) == 0


def test_test_passes_changed_since_to_the_example_suite(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    repo = _examples_repository(tmp_path)

    assert main(["test", "--repo-root", str(repo), "--changed-since", "no-such-ref"]) == 4

    err = capfd.readouterr().err
    assert "--examples-changed-since:" in err
    assert "no-such-ref" in err


def test_test_passes_budget_seconds_to_the_example_suite(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    repo = _examples_repository(tmp_path)

    assert main(["test", "--repo-root", str(repo), "--budget-seconds", "0"]) == 4

    assert "--examples-budget-seconds must be a finite number" in capfd.readouterr().err


def test_test_runs_only_the_paths_it_is_given(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    """A result test given without its example names the command that runs them together."""
    repo = _examples_repository(tmp_path)

    status = main(["test", "--repo-root", str(repo), str(repo / "tests" / "examples")])

    assert status == 1
    assert "runs with its example: uv run substrax-notebooks test" in capfd.readouterr().out


_WRITES_BELOW_SYS_STDOUT = GOOD.replace(
    "def main() -> None:\n    print(sort_records(make_records()))",
    "def main() -> None:\n    sort_records(make_records())\n"
    "    os.write(1, b'written to the file descriptor\\n')",
).replace("import jax\n", "import jax\nimport os\n", 1)


def test_test_runs_an_example_writing_below_sys_stdout_as_its_page_shows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ipykernel captures a file descriptor only outside pytest, where ``page`` makes the page."""
    assert "os.write(1" in _WRITES_BELOW_SYS_STDOUT
    monkeypatch.delenv("PYTEST_CURRENT_TEST")
    repo, script = write_example(tmp_path, _WRITES_BELOW_SYS_STDOUT)
    page = generate_page(script, repo_root=repo, config=CONFIG)
    assert "written to the file descriptor" in page.read_text(encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "page")

    assert main(["test", "--repo-root", str(repo)]) == 0


_WRITES_THE_PROBE = 'import os\n\nos.environ["PROBE_VAR"] = "from-a-plugin"\n'


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        pytest.param("PYTEST_ADDOPTS", "-k no_test_has_this_name", id="pytest-addopts"),
        pytest.param("PYTEST_TIMEOUT", "0.5", id="pytest-timeout"),
        pytest.param("PYTEST_PLUGINS", "writes_the_probe", id="pytest-plugins"),
    ],
)
def test_test_runs_the_examples_apart_from_the_callers_pytest_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variable: str, value: str
) -> None:
    """The caller's pytest variables configure the caller's tests, not the examples' session."""
    monkeypatch.setenv("PROBE_VAR", "from-shell")
    repo = _examples_repository(tmp_path)
    (repo / "writes_the_probe.py").write_text(_WRITES_THE_PROBE, encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "a plugin the caller may name")
    monkeypatch.setenv(variable, value)

    assert main(["test", "--repo-root", str(repo)]) == 0


def test_test_takes_the_repository_s_own_root_under_another_project(tmp_path: Path) -> None:
    """pytest's search skips a pyproject.toml with no pytest table and stops at an ancestor's."""
    outer = tmp_path / "outer"
    outer.mkdir()
    # Read as the session's configuration, this refuses to start: pytest checks required_plugins.
    (outer / "pytest.ini").write_text(
        "[pytest]\nrequired_plugins = an-uninstalled-plugin\n", encoding="utf-8"
    )
    repo = _examples_repository(outer, result_tests=False)
    pyproject = repo / "pyproject.toml"
    table = (
        '\n[tool.pytest.ini_options]\ntestpaths = ["tests"]\naddopts = "--ignore=tests/examples"\n'
    )
    assert table in pyproject.read_text(encoding="utf-8")
    pyproject.write_text(pyproject.read_text(encoding="utf-8").replace(table, ""), encoding="utf-8")
    git(repo, "commit", "-q", "-am", "no pytest table")

    assert main(["test", "--repo-root", str(repo)]) == 0


def test_test_refuses_a_path_outside_the_repository(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _examples_repository(tmp_path / "one", result_tests=False)
    elsewhere = _examples_repository(tmp_path / "two", result_tests=False)

    status = main(["test", "--repo-root", str(repo), str(elsewhere / "examples")])

    assert status == 2
    assert f"not under {repo.resolve()}" in capsys.readouterr().err


def test_test_takes_paths_from_the_working_directory_as_the_other_commands_do(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _examples_repository(tmp_path, result_tests=False)
    monkeypatch.chdir(repo / "examples")

    assert main(["test", "--repo-root", str(repo), "sorting.py"]) == 0


def test_test_refuses_a_repository_without_the_settings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _examples_repository(tmp_path, result_tests=False)
    pyproject = repo / "pyproject.toml"
    table = "[tool.substrax.examples]"
    assert table in pyproject.read_text(encoding="utf-8")
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8").replace(table, "[tool.other]"), encoding="utf-8"
    )

    assert main(["test", "--repo-root", str(repo)]) == 2
    assert table in capsys.readouterr().err


@pytest.mark.parametrize("command", ["check", "page", "test"])
def test_a_repository_without_a_pyproject_is_refused_not_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], command: str
) -> None:
    repo, script = write_example(tmp_path)
    (repo / "pyproject.toml").unlink()
    examples = [] if command == "test" else [str(script)]

    assert main([command, "--repo-root", str(repo), *examples]) == 2

    assert capsys.readouterr().err == f"substrax-notebooks: {repo} has no pyproject.toml\n"


def test_convert_reports_a_script_that_is_not_utf8_and_converts_the_others(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo, script = write_example(tmp_path)
    other = script.with_name("other.py")
    other.write_bytes(b"# caf\xe9\n")
    script.with_suffix(".ipynb").unlink()

    assert main(["convert", "--repo-root", str(repo), str(other), str(script)]) == 1

    captured = capsys.readouterr()
    assert captured.err == (
        f"substrax-notebooks: {other} is not UTF-8: byte 5 cannot be decoded; save it as UTF-8\n"
    )
    assert captured.out == f"{script.with_suffix('.ipynb')}\n"


def test_page_into_outputs_writes_the_page_under_the_runs_outputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Where a compute task's files come back from: ``$AVITAI_OUTPUT_DIR`` (W10)."""
    repo, script = write_example(tmp_path)
    outputs = tmp_path / "task-outputs"
    monkeypatch.setenv("AVITAI_OUTPUT_DIR", str(outputs))

    assert main(["page", "--into-outputs", "--repo-root", str(repo), str(script)]) == 0

    page = outputs / "page" / page_path(script, repo_root=repo).relative_to(repo)
    assert page.is_file()
    assert capsys.readouterr().out == f"{page}\n"
    assert not page_path(script, repo_root=repo).exists()
