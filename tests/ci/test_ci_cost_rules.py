"""CI runs only what a change needs: no repeated merge runs, no macOS on pushes or pull requests.

Two rules, both about runner time. A merge onto an unmoved ``main``, squash or rebase, carries the
tree its pull request already tested, so the jobs that would repeat it stand down when every check of that pull
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
GATE_ANSWER = f"${{{{ needs.{GATE_JOB}.outputs.skip }}}}"
GATED_WORKFLOWS = ("ci.yml",)
MACOS_WORKFLOW = "macos.yml"
# The job that reports on every other job runs whatever they did.
REPORTING_JOB = "required"


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


def _called(job: dict[str, Any]) -> dict[str, Any]:
    """The workflow ``job`` calls."""
    return _load(ROOT / str(job["uses"]).removeprefix("./"))


def test_the_macos_workflow_runs_nightly_on_demand_and_under_a_runner_cap() -> None:
    """The night calls the shared gates on macOS: their tests and build take the runner it passes."""
    workflow = _load(WORKFLOWS / MACOS_WORKFLOW)
    macos_calls = [job for job in workflow["jobs"].values() if _on_macos(job)]

    assert _triggers(workflow) == {"schedule", "workflow_dispatch"}
    assert len(macos_calls) == 1, "one call runs the gates on macOS"
    called = _called(macos_calls[0])
    on_runner = {
        name: job
        for name, job in called["jobs"].items()
        if job.get("runs-on") == "${{ inputs.os }}"
    }
    assert set(on_runner) == {"test", "build"}
    versions = called["on"]["workflow_call"]["inputs"]["python-versions"]["default"]
    assert json.loads(versions) == ["3.12", "3.13"]
    assert "python-versions" not in macos_calls[0]["with"]
    for name, job in on_runner.items():
        assert int(job["strategy"]["max-parallel"]) <= 2, (
            f"{name}: a macOS job without a runner cap"
        )
        assert (
            job["strategy"]["matrix"]["python-version"] == "${{ fromJSON(inputs.python-versions) }}"
        )
    commands = "\n".join(
        str(step.get("run", "")) for job in on_runner.values() for step in job["steps"]
    )
    assert "pytest" in commands
    assert "uv build" in commands


def test_a_quiet_night_runs_no_macos_job() -> None:
    """A scheduled run stands down when main has not moved since the last green scheduled run.

    Only the schedule compares: a manual run, such as the release checklist's, always runs.
    """
    workflow = _load(WORKFLOWS / MACOS_WORKFLOW)
    moved = workflow["jobs"]["main_moved"]
    compare = next(step for step in moved["steps"] if step.get("id") == "compare")

    # The comparison is the shared action's, tested in test_main_moved_action.py.
    assert compare["uses"] == "./.github/actions/main-moved"
    assert compare["with"]["workflow"] == MACOS_WORKFLOW
    assert "steps.compare.outputs.unchanged" in moved["outputs"]["unchanged"]
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
    """A job stands down on the gate itself, or a call hands the gate's answer to the gates."""
    jobs = _load(WORKFLOWS / name)["jobs"]
    ungated = sorted(
        job_name
        for job_name, job in jobs.items()
        if job_name not in {GATE_JOB, REPORTING_JOB}
        and GATE_CONDITION not in str(job.get("if", ""))
        and job.get("with", {}).get("already-tested") != GATE_ANSWER
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


ROLLUPS = Path(__file__).parent / "fixtures" / "check_rollup"
NO_JQ = pytest.mark.skipif(
    shutil.which("jq") is None, reason="jq, present on GitHub's runners, is absent"
)


def _recorded(fixture: str) -> dict[str, Any]:
    """A response of the action's ``checks.graphql``, recorded from a real pull request."""
    return json.loads((ROLLUPS / f"{fixture}.json").read_text())["response"]


def _rollup(response: dict[str, Any]) -> dict[str, Any]:
    return response["data"]["repository"]["pullRequest"]["commits"]["nodes"][0]["commit"][
        "statusCheckRollup"
    ]


def _with_contexts(nodes: list[dict[str, str]], *, more: bool = False) -> dict[str, Any]:
    """The recorded private-repository response, holding ``nodes`` as its checks instead."""
    response = _recorded("private_repository_check_runs")
    contexts = _rollup(response)["contexts"]
    contexts["nodes"], contexts["pageInfo"]["hasNextPage"] = nodes, more
    return response


def _unproven(response: dict[str, Any]) -> int:
    result = subprocess.run(  # noqa: S603  # nosec B603 - jq on a fixed program and fixture
        [shutil.which("jq") or "jq", "-f", str(ACTION / "unproven.jq")],
        input=json.dumps(response),
        capture_output=True,
        text=True,
        check=True,
    )
    return int(result.stdout)


def _run(conclusion: str) -> dict[str, str]:
    return {
        "__typename": "CheckRun",
        "name": "a job",
        "status": "COMPLETED",
        "conclusion": conclusion,
    }


@NO_JQ
@pytest.mark.parametrize(
    "fixture", ["private_repository_check_runs", "check_runs_and_a_commit_status"]
)
def test_a_pull_request_whose_every_check_passed_is_proven(fixture: str) -> None:
    """Read from real responses: check runs, and a commit status, which carries a state."""
    assert _unproven(_recorded(fixture)) == 0


@NO_JQ
@pytest.mark.parametrize(
    ("nodes", "unproven"),
    [
        ([_run("SUCCESS"), _run("SKIPPED"), _run("NEUTRAL")], 0),
        ([{"__typename": "StatusContext", "context": "docs", "state": "SUCCESS"}], 0),
        ([_run("SUCCESS"), {**_run(""), "status": "IN_PROGRESS", "conclusion": None}], 1),
        ([_run("SUCCESS"), _run("CANCELLED")], 1),
        ([_run("ACTION_REQUIRED")], 1),
        ([_run("FAILURE"), _run("TIMED_OUT")], 2),
        ([{"__typename": "StatusContext", "context": "docs", "state": "PENDING"}], 1),
    ],
)
def test_a_check_that_did_not_succeed_keeps_the_merge_tested(
    nodes: list[dict[str, str]], unproven: int
) -> None:
    """Only SUCCESS, SKIPPED and NEUTRAL count as proven; pending and cancelled do not."""
    assert _unproven(_with_contexts(nodes)) == unproven


@NO_JQ
def test_checks_beyond_the_first_page_keep_the_merge_tested() -> None:
    """The query reads 100 checks; one it did not read is not proven."""
    assert _unproven(_with_contexts([_run("SUCCESS")], more=True)) > 0


@NO_JQ
def test_a_pull_request_with_no_check_proves_nothing() -> None:
    """A merge whose pull request ran no check repeats nothing, so its tree is tested on main."""
    assert _unproven(_with_contexts([])) > 0
    response = _with_contexts([])
    response["data"]["repository"]["pullRequest"]["commits"]["nodes"][0]["commit"][
        "statusCheckRollup"
    ] = None
    assert _unproven(response) > 0


def test_the_checks_query_reads_no_workflow_run() -> None:
    """A workflow run needs `actions: read` on a private repository; check results need only
    `checks` and `statuses`. `gh pr view --json statusCheckRollup` asks for the workflow run and
    fails there with "Resource not accessible by integration" (txpert-nnx, 2026-10-29)."""
    query = (ACTION / "checks.graphql").read_text()

    assert "conclusion" in query
    assert "state" in query
    assert "workflowRun" not in query
    assert "checkSuite" not in query


def test_the_action_reads_the_checks_through_its_own_query_and_shows_a_failure() -> None:
    step = yaml.safe_load((ACTION / "action.yml").read_text())["runs"]["steps"][0]
    script = step["run"]
    lookup = next(line for line in script.splitlines() if "gh api graphql" in line)

    assert "gh pr view" not in script
    assert step["env"]["CHECKS"].endswith("/checks.graphql")
    assert 'query=@"$CHECKS"' in lookup
    assert "2>/dev/null" not in lookup, "a failed lookup says why in the log"
    assert '-f "$UNPROVEN"' in script, "the count is the tested jq program"


PULLS = Path(__file__).parent / "fixtures" / "commit_pulls"


def _pull_request(fixture: str) -> str:
    """The pull request the action finds for a pushed commit, from a recorded API response.

    Each fixture is ``GET /repos/{owner}/{repo}/commits/{sha}/pulls`` for a real commit, reduced to
    the fields the selection reads.
    """
    recorded = json.loads((PULLS / f"{fixture}.json").read_text())
    result = subprocess.run(  # noqa: S603  # nosec B603 - jq on a fixed program and fixture
        [
            shutil.which("jq") or "jq",
            "-r",
            "--arg",
            "sha",
            recorded["sha"],
            "-f",
            str(ACTION / "pull_request.jq"),
        ],
        input=json.dumps(recorded["pulls"]),
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


@pytest.mark.skipif(shutil.which("jq") is None, reason="jq, present on GitHub's runners, is absent")
@pytest.mark.parametrize(
    ("fixture", "pull_request"),
    [
        ("rebase_merge", "47"),  # datarax bd92a3e: a rebase merge keeps the commit's own subject
        ("rebase_merge_tip", "37"),  # artifex 1613fb9: the last commit of a two-commit rebase
        ("rebase_merge_intermediate", ""),  # artifex 4ee4722: pushed with the tip, never alone
        ("squash_merge", "39"),  # datarax 27d1539
        ("open_pull_request", ""),  # datarax 80be69b: an open pull request carries a test merge
        ("stacked_pull_requests", "47"),  # one commit in an open and a merged pull request
    ],
)
def test_the_pushed_commit_names_the_pull_request_it_merged(
    fixture: str, pull_request: str
) -> None:
    """A commit belongs to the pull request whose merge made it, however the pull request merged.

    The subject names a pull request only after a squash merge; a rebase merge keeps each
    commit's own subject, so the action asks GitHub which merged pull request produced the commit.
    """
    assert _pull_request(fixture) == pull_request


def test_the_action_asks_github_which_pull_request_produced_the_commit() -> None:
    step = yaml.safe_load((ACTION / "action.yml").read_text())["runs"]["steps"][0]
    script = step["run"]

    assert "commits/$commit/pulls" in script
    assert step["env"]["PULL_REQUEST"].endswith("/pull_request.jq")
    assert '-f "$PULL_REQUEST"' in script, "the selection is the tested jq program"
    assert "--pretty=%s" not in script, "a subject names a pull request only after a squash merge"


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
