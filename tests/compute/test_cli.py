"""``substrax-compute``: run a configured job, then follow, fetch or cancel it by its run id."""

from __future__ import annotations

import sys
import textwrap
import time
from pathlib import Path

import pytest

from substrax.compute.backend import STATE_DIR_ENV
from substrax.compute.cli import main


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
