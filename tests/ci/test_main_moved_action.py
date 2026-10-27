"""The ``main-moved`` action tells a nightly workflow whether main moved since its last green night.

macOS runners are scarce and slow, so the macOS workflow runs nightly and stands down on a night when
main has not moved since the last scheduled run that succeeded. A manual run always runs. A lookup
that fails, or a first night with no green run before it, answers "moved", so the work runs rather
than being skipped on a guess. The decision lives in ``compare.sh``; these tests run it with ``gh``
replaced by a fake.
"""

from __future__ import annotations

import shutil
import subprocess  # nosec B404
from pathlib import Path
from typing import Any

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
ACTION = ROOT / ".github" / "actions" / "main-moved"
SHA = "a" * 40

FAKE_GH = """#!/usr/bin/env bash
printf 'gh %s\\n' "$*" >> "$MAIN_MOVED_LOG"
if [ -n "${FAKE_GH_EXIT:-}" ]; then exit "$FAKE_GH_EXIT"; fi
printf '%s\\n' "${FAKE_LAST_SHA:-}"
"""


@pytest.fixture
def run(tmp_path: Path) -> Any:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "gh").write_text(FAKE_GH)
    (bin_dir / "gh").chmod(0o755)
    log = tmp_path / "calls.log"
    output = tmp_path / "output"

    def _run(event: str, **env: str) -> tuple[str, list[str]]:
        output.write_text("")
        environment = {
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "MAIN_MOVED_LOG": str(log),
            "GITHUB_OUTPUT": str(output),
            "EVENT_NAME": event,
            "SHA": SHA,
            "REPOSITORY": "avitai/example",
            "WORKFLOW": "macos.yml",
            "BRANCH": "main",
            **env,
        }
        subprocess.run(  # noqa: S603  # nosec B603 - the tested script with fixed arguments
            [shutil.which("bash") or "bash", str(ACTION / "compare.sh")],
            env=environment,
            check=True,
            capture_output=True,
        )
        calls = log.read_text().splitlines() if log.exists() else []
        return output.read_text().strip(), calls

    return _run


class TestCompare:
    def test_a_night_on_the_last_green_commit_is_unchanged(self, run: Any) -> None:
        assert run("schedule", FAKE_LAST_SHA=SHA)[0] == "unchanged=true"

    def test_a_night_after_main_moved_runs(self, run: Any) -> None:
        assert run("schedule", FAKE_LAST_SHA="b" * 40)[0] == "unchanged=false"

    def test_a_first_night_with_no_green_run_runs(self, run: Any) -> None:
        assert run("schedule", FAKE_LAST_SHA="")[0] == "unchanged=false"

    def test_a_failed_lookup_runs(self, run: Any) -> None:
        assert run("schedule", FAKE_GH_EXIT="1")[0] == "unchanged=false"

    def test_a_manual_run_always_runs_without_asking(self, run: Any) -> None:
        output, calls = run("workflow_dispatch", FAKE_LAST_SHA=SHA)

        assert output == "unchanged=false"
        assert calls == []

    def test_it_asks_for_the_last_successful_scheduled_run_of_its_workflow_on_its_branch(
        self, run: Any
    ) -> None:
        _, calls = run("schedule", FAKE_LAST_SHA=SHA, WORKFLOW="nightly.yml", BRANCH="trunk")

        assert len(calls) == 1
        for part in (
            "run list",
            "--repo avitai/example",
            "--workflow nightly.yml",
            "--branch trunk",
            "--event schedule",
            "--status success",
            "--limit 1",
        ):
            assert part in calls[0]  # fmt: skip


@pytest.fixture(scope="module")
def action() -> dict[str, Any]:
    return yaml.safe_load((ACTION / "action.yml").read_text())


class TestActionContract:
    def test_it_runs_the_tested_script_with_the_run_s_own_facts(
        self, action: dict[str, Any]
    ) -> None:
        step = action["runs"]["steps"][0]

        assert action["runs"]["using"] == "composite"
        assert step["shell"] == "bash"
        assert 'bash "$GITHUB_ACTION_PATH/compare.sh"' in step["run"]
        assert step["env"]["EVENT_NAME"] == "${{ github.event_name }}"
        assert step["env"]["SHA"] == "${{ github.sha }}"
        assert step["env"]["WORKFLOW"] == "${{ inputs.workflow }}"
        assert "github.token" in step["env"]["GH_TOKEN"]

    def test_it_reports_unchanged_and_needs_the_workflow_named(
        self, action: dict[str, Any]
    ) -> None:
        assert action["inputs"]["workflow"]["required"] is True
        assert action["inputs"]["branch"]["default"] == "main"
        assert "steps.compare.outputs.unchanged" in action["outputs"]["unchanged"]["value"]

    def test_no_step_can_pass_on_a_failure(self) -> None:
        text = (ACTION / "action.yml").read_text() + (ACTION / "compare.sh").read_text()

        assert "continue-on-error" not in text
        assert "set -euo pipefail" in (ACTION / "compare.sh").read_text()


def test_the_macos_workflow_uses_the_action_and_carries_no_copy_of_its_logic() -> None:
    text = (ROOT / ".github" / "workflows" / "macos.yml").read_text()
    moved = yaml.safe_load(text)["jobs"]["main_moved"]

    assert [
        step.get("uses") for step in moved["steps"] if "main-moved" in str(step.get("uses"))
    ] == ["./.github/actions/main-moved"]
    assert "gh run list" not in text
