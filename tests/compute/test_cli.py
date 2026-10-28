"""``substrax-compute``: run a configured job, then follow, fetch or cancel it by its run id."""

from __future__ import annotations

import json
import sys
import textwrap
import time
from collections.abc import Mapping
from pathlib import Path

import pytest

from substrax.compute import cli
from substrax.compute.backend import new_run, RunHandle, RunState, RunStore, STATE_DIR_ENV
from substrax.compute.cli import main
from substrax.compute.spec import JobSpec, Task
from substrax.typing import JsonValue


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A project whose ``local`` backend runs this interpreter, with state kept in the test."""
    monkeypatch.setenv(STATE_DIR_ENV, str(tmp_path / "state"))
    root = tmp_path / "project"
    (root / "examples").mkdir(parents=True)
    (root / "examples" / "hello.py").write_text("print('hello from the example')\n", "utf-8")
    (root / "examples" / "slow.py").write_text(
        "import time\nprint('sleeping', flush=True)\ntime.sleep(600)\n", "utf-8"
    )
    (root / "uv.lock").write_text('[[package]]\nname = "substrax"\nversion = "0.1.15"\n', "utf-8")
    (root / "pyproject.toml").write_text(
        textwrap.dedent(
            f"""
            [project]
            name = "demo"

            [tool.substrax.compute]
            backend = "local"

            [tool.substrax.compute.backends.local]
            python = "{sys.executable}"

            [tool.substrax.compute.jobs.hello]
            timeout_seconds = 120
            commands = {{ hello = ["python", "examples/hello.py"] }}

            [tool.substrax.compute.jobs.slow]
            timeout_seconds = 120
            commands = {{ slow = ["python", "examples/slow.py"] }}

            [tool.substrax.compute.jobs.failing]
            timeout_seconds = 120
            commands = {{ failing = ["python", "-c", "raise SystemExit(2)"] }}
            """
        ),
        "utf-8",
    )
    return root


def _run_id(output: str) -> str:
    return output.split("Submitted ", 1)[1].split(" ", 1)[0]


def test_run_follows_the_job_and_fetches_its_outputs(
    project: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["run", "hello", "--project", str(project), "--into", str(tmp_path / "out")])

    output = capsys.readouterr().out
    run_id = _run_id(output)
    assert code == 0
    assert "[hello] hello from the example" in output
    assert "succeeded" in output
    assert (tmp_path / "out" / run_id / "hello" / "stdout.log").is_file()


def test_a_failed_job_exits_non_zero(project: Path, tmp_path: Path) -> None:
    code = main(["run", "failing", "--project", str(project), "--into", str(tmp_path / "out")])

    assert code == 1


def test_a_detached_run_is_found_again_by_its_id(
    project: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["run", "hello", "--project", str(project), "--detach"]) == 0
    run_id = _run_id(capsys.readouterr().out)

    assert main(["logs", run_id, "--follow"]) == 0
    assert "[hello] hello from the example" in capsys.readouterr().out
    assert main(["status", run_id]) == 0
    assert "succeeded" in capsys.readouterr().out
    assert main(["fetch", run_id, "--into", str(tmp_path / "out")]) == 0
    assert (tmp_path / "out" / run_id / "manifest.json").is_file()
    assert main(["status"]) == 0
    assert run_id in capsys.readouterr().out


def test_cancel_stops_a_detached_run(project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main(["run", "slow", "--project", str(project), "--detach"])
    run_id = _run_id(capsys.readouterr().out)
    _wait_for_log_line(run_id, "[slow] sleeping", capsys)

    assert main(["cancel", run_id]) == 0
    capsys.readouterr()
    main(["status", run_id])
    assert "cancelled" in capsys.readouterr().out


def _wait_for_log_line(run_id: str, wanted: str, capsys: pytest.CaptureFixture[str]) -> None:
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        main(["logs", run_id])
        if wanted in capsys.readouterr().out.splitlines():
            return
        time.sleep(0.2)
    raise AssertionError(f"{run_id} never logged {wanted!r}")


def test_an_unknown_job_names_the_configured_ones(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["run", "train", "--project", str(project)])

    assert code == 2
    assert "configured: failing, hello, slow" in capsys.readouterr().err


def test_a_project_that_does_not_lock_substrax_is_refused_before_submitting(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (project / "uv.lock").write_text('[[package]]\nname = "jax"\nversion = "0.11.1"\n', "utf-8")

    code = main(["run", "hello", "--project", str(project)])

    assert code == 2
    assert "substrax is not in" in capsys.readouterr().err


_SAVED = JobSpec(
    name="saved", timeout_seconds=10, tasks=(Task(name="t", argv=("true",), timeout_seconds=5),)
)


class _RecordingBackend:
    """A backend the CLI rebuilds for a stored run; it records the settings it was given."""

    def __init__(self, name: str, settings: Mapping[str, JsonValue], built: list[object]) -> None:
        built.append((name, settings))

    def status(self, handle: RunHandle) -> RunState:
        del handle
        return RunState.SUCCEEDED


def _record_backends(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    built: list[object] = []
    monkeypatch.setattr(
        cli, "load_backend", lambda name, settings: _RecordingBackend(name, settings, built)
    )
    return built


@pytest.mark.parametrize("backend_name", ["skypilot", "local", "modal", "third-party"])
def test_a_stored_run_is_recovered_with_the_settings_it_was_submitted_with(
    project: Path, monkeypatch: pytest.MonkeyPatch, backend_name: str
) -> None:
    """Editing a project's provider settings must not redirect an existing run."""
    settings: dict[str, JsonValue] = {"state_dir": "/original/state"}
    handle = new_run(_SAVED, project, settings=settings).handle(backend_name, {"id": "owned"})
    RunStore().save(handle)
    (project / "pyproject.toml").write_text("[tool.substrax.compute]\n", "utf-8")
    built = _record_backends(monkeypatch)

    assert main(["status", handle.run_id]) == 0
    assert built == [(backend_name, settings)]


def test_listing_names_each_unsupported_record_and_still_lists_the_rest(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """One record an older substrax wrote does not hide the runs that can still be read."""
    handle = new_run(_SAVED, project, settings={}).handle("local", {"pid": "1"})
    store = RunStore()
    store.save(handle)
    old = store.root / "saved-20260901T000000Z-0000.json"
    old.write_text(json.dumps({"schema_version": 1, "run_id": "saved-old"}), "utf-8")
    _record_backends(monkeypatch)

    code = main(["status"])

    captured = capsys.readouterr()
    assert code == 1
    assert f"{handle.run_id}  local  saved  succeeded" in captured.out
    assert str(old) in captured.err
    assert "version 1" in captured.err


def test_cleanup_error_group_is_reported_without_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Every cleanup failure stays visible with a documented unsuccessful exit."""

    def cancel(arguments: object, store: object) -> None:
        del arguments, store
        raise ExceptionGroup(
            "cleanup",
            [
                RuntimeError("queue unavailable"),
                ExceptionGroup("teardown", [RuntimeError("down failed")]),
            ],
        )

    monkeypatch.setattr(cli, "_cancel", cancel)
    assert main(["cancel", "owned-run"]) == 1
    error = capsys.readouterr().err
    assert "queue unavailable" in error
    assert "down failed" in error
    assert "Traceback" not in error


def test_new_run_binds_typed_settings(project: Path) -> None:
    """Every provider uses the same required typed settings field."""
    run = new_run(_SAVED, project, settings={"state_dir": "/original"})
    handle = run.handle("local", {"pid": "123"})
    assert handle.settings == {"state_dir": "/original"}
    assert handle.schema_version == 2
