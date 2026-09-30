"""CI runs only what a change needs: no repeated merge runs, no macOS on pushes or pull requests.

Two rules, both about runner time. A squash merge onto an unmoved ``main`` carries the tree its pull
request already tested, so the jobs that would repeat it stand down when every check of that pull
request succeeded (the ``already-tested`` action). macOS runners are scarce and slow: no macOS job
runs on a push or a pull request; ``macos.yml`` runs nightly when ``main`` has moved, on demand, and
on the release commit before its tag (RELEASING.md).
"""

from __future__ import annotations

import json
import shutil
import subprocess  # nosec B404
from pathlib import Path
from typing import Any

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
ACTION = ROOT / ".github" / "actions" / "already-tested"
GATE_JOB = "already_tested"
GATE_CONDITION = f"needs.{GATE_JOB}.outputs.skip != 'true'"
GATED_WORKFLOWS = ("ci.yml", "quality-checks.yml")
MACOS_WORKFLOW = "macos.yml"


def _load(path: Path) -> dict[str, Any]:
    # BaseLoader keeps every scalar a string, so the `on:` key stays `on` rather than True.
    return yaml.load(path.read_text(), Loader=yaml.BaseLoader)  # noqa: S506


def _triggers(workflow: dict[str, Any]) -> set[str]:
    on = workflow["on"]
    return {on} if isinstance(on, str) else set(on)


def _text(job: dict[str, Any]) -> str:
    return yaml.safe_dump(job)


def _on_macos(job: dict[str, Any]) -> bool:
    """Whether the job runs on macOS: its runner, the matrix its runner reads, or the runner it
    passes a called workflow."""
    runner = [
        job.get("runs-on", ""),
        job.get("strategy", {}).get("matrix", {}),
        job.get("with", {}),
    ]
    return "macos" in yaml.safe_dump(runner).lower()


@pytest.mark.parametrize("path", sorted(WORKFLOWS.glob("*.yml")), ids=lambda p: p.name)
def test_no_workflow_run_by_a_push_or_pull_request_uses_macos(path: Path) -> None:
    workflow = _load(path)
    if not _triggers(workflow) & {"push", "pull_request"}:
        return
    on_macos = [name for name, job in workflow["jobs"].items() if _on_macos(job)]

    assert on_macos == [], f"{path.name} runs macOS on a push or pull request: {on_macos}"


def _run_jobs(job: dict[str, Any]) -> list[dict[str, Any]]:
    """The jobs that run for ``job``: itself, or those of the workflow it calls."""
    called = job.get("uses")
    if called is None:
        return [job]
    return list(_load(ROOT / called.removeprefix("./")).get("jobs", {}).values())


def test_the_macos_workflow_runs_nightly_on_demand_and_under_a_runner_cap() -> None:
    workflow = _load(WORKFLOWS / MACOS_WORKFLOW)
    macos_jobs = {name: job for name, job in workflow["jobs"].items() if _on_macos(job)}
    running = [run for job in macos_jobs.values() for run in _run_jobs(job)]

    assert _triggers(workflow) == {"schedule", "workflow_dispatch"}
    assert set(macos_jobs) == {"test", "build"}
    for job in running:
        assert int(job["strategy"]["max-parallel"]) <= 2, "a macOS job without a runner cap"
        assert job["strategy"]["matrix"]["python-version"] == ["3.12", "3.13"]
    commands = "\n".join(str(step.get("run", "")) for job in running for step in job["steps"])
    assert "pytest" in commands
    assert "uv build" in commands


def test_a_quiet_night_runs_no_macos_job() -> None:
    """A scheduled run stands down when main has not moved since the last green scheduled run.

    Only the schedule compares: a manual run, such as the release checklist's, always runs.
    """
    workflow = _load(WORKFLOWS / MACOS_WORKFLOW)
    moved = workflow["jobs"]["main_moved"]
    compare = next(step for step in moved["steps"] if step.get("id") == "compare")

    assert compare.get("if") == "github.event_name == 'schedule'"
    assert "--status success" in compare["run"]
    assert "--event schedule" in compare["run"]
    for name, job in workflow["jobs"].items():
        if name == "main_moved":
            continue
        assert job.get("if") == "needs.main_moved.outputs.unchanged != 'true'", name


def test_the_release_checklist_runs_macos_before_the_tag() -> None:
    releasing = (ROOT / "RELEASING.md").read_text()

    assert "gh workflow run macos.yml" in releasing
    assert releasing.index("gh workflow run macos.yml") < releasing.index("git tag -a")


@pytest.mark.parametrize("name", GATED_WORKFLOWS)
def test_the_gate_compares_only_on_a_push(name: str) -> None:
    """A schedule or a manual run re-measures on purpose; only a merge repeats a pull request."""
    gate = _load(WORKFLOWS / name)["jobs"][GATE_JOB]
    steps = [step for step in gate["steps"] if "already-tested" in str(step.get("uses", ""))]

    assert [step.get("if") for step in steps] == ["github.event_name == 'push'"]
    assert steps[0]["id"] in gate["outputs"]["skip"]


@pytest.mark.parametrize("name", GATED_WORKFLOWS)
def test_every_job_that_repeats_the_pull_request_consults_the_gate(name: str) -> None:
    jobs = _load(WORKFLOWS / name)["jobs"]
    ungated = sorted(
        job_name
        for job_name, job in jobs.items()
        if job_name != GATE_JOB and job.get("if") != GATE_CONDITION
    )

    assert ungated == [], f"{name}: these repeat the pull request without the gate: {ungated}"


@pytest.mark.parametrize("name", GATED_WORKFLOWS)
def test_an_unanswered_gate_leaves_the_work_running(name: str) -> None:
    """An empty output (no compare, or a lookup that failed) reads as "test it"."""
    for job_name, job in _load(WORKFLOWS / name)["jobs"].items():
        if job_name == GATE_JOB:
            continue
        text = _text(job)
        assert "outputs.skip == " not in text, f"{job_name} tests the gate for equality"
        assert "outputs.skip != 'false'" not in text, f"{job_name} runs only on an explicit false"


def _unproven(checks: list[dict[str, str]]) -> int:
    result = subprocess.run(  # noqa: S603  # nosec B603 - jq on a fixed program and fixture
        [shutil.which("jq") or "jq", "-f", str(ACTION / "unproven.jq")],
        input=json.dumps({"statusCheckRollup": checks}),
        capture_output=True,
        text=True,
        check=True,
    )
    return int(result.stdout)


@pytest.mark.skipif(shutil.which("jq") is None, reason="jq, present on GitHub's runners, is absent")
@pytest.mark.parametrize(
    ("checks", "unproven"),
    [
        ([{"conclusion": "SUCCESS"}, {"conclusion": "SKIPPED"}, {"conclusion": "NEUTRAL"}], 0),
        ([{"state": "SUCCESS"}], 0),  # a commit status carries state, not conclusion
        ([{"conclusion": "SUCCESS"}, {"conclusion": ""}], 1),  # still running
        ([{"conclusion": "SUCCESS"}, {"conclusion": "CANCELLED"}], 1),
        ([{"conclusion": "ACTION_REQUIRED"}], 1),
        ([{"conclusion": "FAILURE"}, {"conclusion": "TIMED_OUT"}], 2),
        ([{"state": "PENDING"}], 1),
    ],
)
def test_a_check_that_did_not_succeed_keeps_the_merge_tested(
    checks: list[dict[str, str]], unproven: int
) -> None:
    """Only SUCCESS, SKIPPED and NEUTRAL count as proven; pending and cancelled do not."""
    assert _unproven(checks) == unproven


def test_quality_checks_leave_the_suite_to_the_test_job() -> None:
    """The test job runs the suite on both Python versions; pre-commit's pytest hook would repeat it."""
    quality = _load(WORKFLOWS / "quality-checks.yml")["jobs"]["quality"]
    pre_commit = next(step for step in quality["steps"] if "pre-commit run" in str(step.get("run")))
    test = _load(WORKFLOWS / "ci.yml")["jobs"]["test"]

    assert "pytest" in pre_commit["env"]["SKIP"].split(",")
    assert (
        test["strategy"]["matrix"]["python-version"]
        == quality["strategy"]["matrix"]["python-version"]
    )
    assert any("pytest" in str(step.get("run", "")) for step in test["steps"])


def test_every_uv_cache_is_pruned_before_it_is_saved() -> None:
    """A saved uv cache holds only what uv built, not every wheel it downloaded.

    setup-uv prunes only when asked (``prune-cache`` defaults to false from v9); unpruned, the
    caches of the heavy extras grow to gigabytes each and evict the repository's other caches.
    """
    github = ROOT / ".github"
    documents = [
        *sorted(github.glob("workflows/*.yml")),
        *sorted(github.glob("actions/*/action.yml")),
    ]
    checked = 0
    unpruned: list[str] = []
    for path in documents:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        owners = {**document.get("jobs", {}), "runs": document.get("runs") or {}}
        for owner, body in owners.items():
            for step in body.get("steps", []):
                if not str(step.get("uses", "")).startswith("astral-sh/setup-uv@"):
                    continue
                checked += 1
                if (step.get("with") or {}).get("prune-cache") is not True:
                    unpruned.append(f"{path.relative_to(ROOT)}:{owner}")

    assert checked, "no setup-uv step found; the contract is reading the wrong files"
    assert unpruned == [], f"setup-uv steps saving an unpruned cache: {unpruned}"
