"""``examples.yml``: a stack repository's examples run as the shared example guide runs them.

One reusable workflow, apart from ``python-gates.yml`` so that the gates' callers keep their pins
(OWN-1113-W15-EXAMPLES-REUSABLE-WORKFLOW): on a pull request it runs the examples the change
reaches, counted from the fetched base branch; on any other event every example.
"""

from __future__ import annotations

import os
import shutil
import subprocess  # nosec B404
from pathlib import Path
from typing import Any

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "examples.yml"
GATES = ROOT / ".github" / "workflows" / "python-gates.yml"


def _load(path: Path) -> dict[str, Any]:
    # BaseLoader keeps every scalar a string, so the `on:` key stays `on` rather than True.
    return yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)  # noqa: S506


def _job() -> dict[str, Any]:
    (job,) = _load(WORKFLOW)["jobs"].values()
    return job


def _step(name: str) -> dict[str, Any]:
    (step,) = [step for step in _job()["steps"] if step.get("name") == name]
    return step


def test_it_runs_only_when_called_and_leaves_concurrency_to_the_caller() -> None:
    workflow = _load(WORKFLOW)

    assert set(workflow["on"]) == {"workflow_call"}
    assert "concurrency" not in workflow


def test_it_stands_down_where_the_caller_proved_the_tree_and_runs_on_cpu() -> None:
    job = _job()

    assert job["if"] == "${{ inputs.already-tested != 'true' }}"
    assert job["runs-on"] == "ubuntu-latest"
    assert job["env"]["JAX_PLATFORMS"] == "cpu"


def test_the_checkout_fetches_the_base_a_pull_request_counts_changes_from() -> None:
    """``actions/checkout`` fetches one commit by default, and the command refuses a ref that names
    no commit."""
    (checkout,) = [step for step in _job()["steps"] if "actions/checkout@" in step.get("uses", "")]

    assert checkout["with"]["fetch-depth"] == "0"


def test_it_installs_with_the_uv_and_the_cache_the_gates_use() -> None:
    gates_uv = _load(GATES)["on"]["workflow_call"]["inputs"]["uv-version"]["default"]
    inputs = _load(WORKFLOW)["on"]["workflow_call"]["inputs"]
    (setup,) = [step for step in _job()["steps"] if "astral-sh/setup-uv@" in step.get("uses", "")]

    assert inputs["uv-version"]["default"] == gates_uv
    assert setup["with"]["prune-cache"] == "true"
    assert "uv sync --locked $SYNC_ARGS" in _step("Install the environment from the lock")["run"]


def _command(tmp_path: Path, *, event: str, base: str = "", budget: str = "") -> list[str]:
    """The arguments the run step gives ``uv``, from a stand-in ``uv`` on PATH."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    record = tmp_path / "argv"
    stand_in = bin_dir / "uv"
    stand_in.write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" > {record}\n', encoding="utf-8")
    stand_in.chmod(0o755)
    step = _step("Run the examples")
    environment = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "EVENT": event,
        "BASE_REF": base,
        "BUDGET": budget,
    }
    bash = shutil.which("bash")
    assert bash is not None
    subprocess.run(  # noqa: S603  # nosec B603 - bash running the workflow's own step
        [bash, "-e", "-o", "pipefail", "-c", step["run"]], env=environment, check=True
    )
    return record.read_text(encoding="utf-8").splitlines()


_RUN = ["run", "--no-sync", "python", "-m", "substrax.notebooks", "test"]


def test_a_pull_request_runs_the_examples_its_change_reaches(tmp_path: Path) -> None:
    command = _command(tmp_path, event="pull_request", base="main")

    assert command == [*_RUN, "--changed-since", "origin/main"]


@pytest.mark.parametrize("event", ["push", "schedule", "workflow_dispatch"])
def test_any_other_event_runs_every_example(tmp_path: Path, event: str) -> None:
    assert _command(tmp_path, event=event, base="") == _RUN


def test_a_lane_budget_reaches_the_command(tmp_path: Path) -> None:
    command = _command(tmp_path, event="schedule", budget="900")

    assert command == [*_RUN, "--budget-seconds", "900"]


def test_the_run_step_reads_the_event_and_base_from_the_caller_s_run() -> None:
    env = _step("Run the examples")["env"]

    assert env["EVENT"] == "${{ github.event_name }}"
    assert env["BASE_REF"] == "${{ github.base_ref }}"
    assert env["BUDGET"] == "${{ inputs.budget-seconds }}"
