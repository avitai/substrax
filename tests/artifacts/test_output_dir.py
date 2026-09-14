"""``resolve_output_dir`` chooses where a run writes its outputs, never inside the working tree."""

from __future__ import annotations

import json
import subprocess
import tempfile
import textwrap
from collections.abc import Callable
from pathlib import Path

import pytest

from substrax.artifacts import OUTPUT_DIR_ENV, resolve_output_dir


type Runner = Callable[[list[str], dict[str, str]], subprocess.CompletedProcess[str]]


@pytest.fixture
def temp_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fresh system temporary directory, so each test gets its own run directory."""
    root = tmp_path / "system-temp"
    root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(root))
    return root


@pytest.fixture
def working_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fresh working directory the test runs in."""
    cwd = tmp_path / "working"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    return cwd


@pytest.mark.usefixtures("temp_root", "working_dir")
class TestResolutionOrder:
    def test_an_explicit_directory_wins_over_the_environment(self, tmp_path: Path) -> None:
        location = resolve_output_dir(
            "run", explicit=tmp_path / "given", env={OUTPUT_DIR_ENV: str(tmp_path / "env")}
        )

        assert (location.path, location.source) == ((tmp_path / "given").resolve(), "argument")
        assert location.path.is_dir()

    def test_a_relative_explicit_directory_is_resolved_against_the_working_directory(
        self, working_dir: Path
    ) -> None:
        location = resolve_output_dir("run", explicit=Path("docs/assets/run"), env={})

        assert location.path == (working_dir / "docs" / "assets" / "run").resolve()
        assert location.path.is_dir()

    def test_the_environment_directory_holds_the_named_output(self, tmp_path: Path) -> None:
        outputs = tmp_path / "outputs"

        location = resolve_output_dir("fno_darcy/figures", env={OUTPUT_DIR_ENV: str(outputs)})

        assert (location.path, location.source) == (
            (outputs / "fno_darcy" / "figures").resolve(),
            "environment",
        )
        assert location.path.is_dir()

    def test_the_process_environment_is_read_by_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(OUTPUT_DIR_ENV, str(tmp_path / "outputs"))

        assert resolve_output_dir("run").source == "environment"

    def test_without_an_argument_or_variable_the_output_goes_under_the_system_temporary_directory(
        self, temp_root: Path, working_dir: Path
    ) -> None:
        location = resolve_output_dir("run", env={})

        assert location.source == "run_default"
        assert location.path.is_relative_to(temp_root.resolve())
        assert not location.path.is_relative_to(working_dir.resolve())
        assert location.path.is_dir()

    def test_an_empty_variable_counts_as_unset(self) -> None:
        assert resolve_output_dir("run", env={OUTPUT_DIR_ENV: ""}).source == "run_default"


@pytest.mark.usefixtures("working_dir")
class TestRunDirectory:
    def test_every_default_in_one_process_shares_one_run_directory(self, temp_root: Path) -> None:
        parents = {resolve_output_dir(name, env={}).path.parent for name in ("a", "b", "c")}

        assert len(parents) == 1
        assert parents.pop().parent == temp_root.resolve()

    def test_another_system_temporary_directory_gets_its_own_run_directory(
        self, temp_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        first = resolve_output_dir("a", env={}).path.parent
        other = tmp_path / "other-temp"
        other.mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(other))

        second = resolve_output_dir("a", env={}).path.parent

        assert first.parent == temp_root.resolve()
        assert second.parent == other.resolve()


@pytest.mark.parametrize("name", ["", ".", "/absolute/run", "../outside", "figures/../../outside"])
def test_a_name_that_leaves_the_output_directory_raises(name: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="name"):
        resolve_output_dir(name, env={OUTPUT_DIR_ENV: str(tmp_path)})


def test_a_relative_environment_directory_raises_and_names_the_variable() -> None:
    with pytest.raises(ValueError, match=OUTPUT_DIR_ENV):
        resolve_output_dir("run", env={OUTPUT_DIR_ENV: "relative/outputs"})


_IMPORT_PROBE = """
    import json, os
    os.chdir({cwd!r})
    import substrax.artifacts
    {action}
    print(json.dumps({{"temp": os.listdir({temp!r}), "cwd": os.listdir({cwd!r})}}))
"""


def _probe(run_interpreter: Runner, tmp_path: Path, action: str) -> dict[str, list[str]]:
    temp, cwd = tmp_path / "temp", tmp_path / "cwd"
    temp.mkdir()
    cwd.mkdir()
    program = textwrap.dedent(_IMPORT_PROBE.format(cwd=str(cwd), temp=str(temp), action=action))
    completed = run_interpreter(["-c", program], {"TMPDIR": str(temp)})
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_importing_the_package_creates_nothing(run_interpreter: Runner, tmp_path: Path) -> None:
    assert _probe(run_interpreter, tmp_path, "pass") == {"temp": [], "cwd": []}


def test_the_import_probe_sees_a_run_directory_being_created(
    run_interpreter: Runner, tmp_path: Path
) -> None:
    """Control: the same probe reports the directory a default resolution creates."""
    action = "substrax.artifacts.resolve_output_dir('run', env={})"

    observed = _probe(run_interpreter, tmp_path, action)

    assert (len(observed["temp"]), observed["cwd"]) == (1, [])
