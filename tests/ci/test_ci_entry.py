"""``ci.yml`` is the one workflow a pull request runs, and ``required`` the one check main requires.

substrax runs its gates the way every stack repository does: ``ci.yml`` calls the shared
``python-gates.yml``, adds the checks only substrax has, and ends in a ``required`` job that fails
unless every other job succeeded or was skipped on purpose.
"""

from __future__ import annotations

import re
import shutil
import subprocess  # nosec B404
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
GATES = "./.github/workflows/python-gates.yml"
GATE_SKIP = "${{ needs.already_tested.outputs.skip }}"
WORKFLOW_FILES = {"ci.yml", "macos.yml", "publish.yml", "python-gates.yml"}


def _load(name: str) -> dict[str, Any]:
    # BaseLoader keeps every scalar a string, so the `on:` key stays `on` rather than True.
    return yaml.load((WORKFLOWS / name).read_text(encoding="utf-8"), Loader=yaml.BaseLoader)  # noqa: S506


def _ci_jobs() -> dict[str, Any]:
    return _load("ci.yml")["jobs"]


def test_the_folded_workflows_are_gone() -> None:
    """Quality checks, security, docs and the build all run inside ``ci.yml`` through the gates."""
    assert {path.name for path in WORKFLOWS.glob("*.yml")} == WORKFLOW_FILES


def test_ci_runs_on_every_change_the_weekly_audit_and_on_demand() -> None:
    on = _load("ci.yml")["on"]

    assert set(on) == {"push", "pull_request", "schedule", "workflow_dispatch"}
    assert on["push"] == {"branches": ["main"]}
    assert on["pull_request"] == {"branches": ["main"]}


def test_a_scheduled_run_and_a_merge_never_cancel_each_other() -> None:
    concurrency = _load("ci.yml")["concurrency"]

    assert "github.event_name" in concurrency["group"]
    assert "github.ref" in concurrency["group"]
    assert concurrency["cancel-in-progress"] == "true"


def test_the_gates_are_called_with_the_merge_gate_and_the_audit_switch() -> None:
    gates = _ci_jobs()["gates"]

    assert gates["uses"] == GATES
    assert gates["with"]["package"] == "substrax"
    assert gates["with"]["already-tested"] == GATE_SKIP
    assert gates["with"]["only-security"] == "${{ github.event_name == 'schedule' }}"
    assert "if" not in gates, "the called jobs decide what stands down"


def test_substraxs_own_checks_stand_down_on_a_proven_merge_and_on_the_schedule() -> None:
    for name, job in _ci_jobs().items():
        if name in {"already_tested", "gates", "required"}:
            continue
        assert "needs.already_tested.outputs.skip != 'true'" in job["if"], name
        assert "github.event_name != 'schedule'" in job["if"], name


def test_required_fails_unless_every_job_succeeded_or_was_skipped_on_purpose() -> None:
    jobs = _ci_jobs()
    required = jobs["required"]
    step = required["steps"][0]

    assert set(required["needs"]) == set(jobs) - {"required"}
    assert required["if"] == "always()"
    assert re.fullmatch(r"re-actors/alls-green@[0-9a-f]{40}", step["uses"])
    assert step["with"]["jobs"] == "${{ toJSON(needs) }}"


CONDITIONAL = re.compile(
    r"^\$\{\{ github\.event_name (?P<op>==|!=) '(?P<event>[a-z_]+)' && '(?P<then>[^']*)' \|\| '(?P<else>[^']*)' \}\}$"
)


def _evaluate(expression: str, event: str) -> str:
    """The value GitHub gives ``${{ github.event_name <op> '<e>' && '<a>' || '<b>' }}`` on ``event``.

    `&&` and `||` return an operand, and '' is falsy, so the expression yields `<a>` only when the
    comparison holds and `<a>` is non-empty: `cond && '' || 'list'` is `'list'` whatever `cond` is.
    """
    match = CONDITIONAL.match(expression)
    assert match, f"not a conditional this test can evaluate: {expression}"
    holds = (event == match["event"]) == (match["op"] == "==")
    return match["then"] if holds and match["then"] else match["else"]


def test_a_pull_request_may_skip_nothing() -> None:
    """Only a proven merge or the weekly audit stands jobs down, so a pull request allows no skip."""
    allowed = _ci_jobs()["required"]["steps"][0]["with"]["allowed-skips"]

    assert _evaluate(allowed, "pull_request") == ""
    for event in ("push", "schedule", "workflow_dispatch"):
        named = set(re.findall(r"[a-z_-]+", _evaluate(allowed, event)))
        assert named == set(_ci_jobs()) - {"already_tested", "required"}, event


def test_the_evaluator_reads_the_falsy_empty_string_as_github_does() -> None:
    """The pitfall the allowed-skips expression once had: an empty middle operand never wins."""
    pitfall = "${{ github.event_name == 'pull_request' && '' || 'gates' }}"

    assert _evaluate(pitfall, "pull_request") == "gates"
    assert _evaluate(pitfall, "push") == "gates"


def test_the_macos_night_calls_the_gates_on_macos_with_the_full_suite() -> None:
    gates = _load("macos.yml")["jobs"]["gates"]

    assert gates["uses"] == GATES
    assert gates["with"]["os"] == "macos-14"
    assert "only-security" not in gates["with"]
    shared = ("package", "pytest-args", "import-modules")
    assert {key: gates["with"][key] for key in shared} == {
        key: _ci_jobs()["gates"]["with"][key] for key in shared
    }


def test_every_job_in_ci_uses_the_uv_the_gates_pin() -> None:
    pinned = _load("python-gates.yml")["on"]["workflow_call"]["inputs"]["uv-version"]["default"]
    versions = [
        step["with"]["version"]
        for job in _ci_jobs().values()
        for step in job.get("steps", [])
        if str(step.get("uses", "")).startswith("astral-sh/setup-uv@")
    ]

    assert versions, "no setup-uv step in ci.yml; the contract is reading the wrong jobs"
    assert set(versions) == {pinned}


def test_the_gates_pin_the_audit_lock_action_this_tree_holds() -> None:
    """A self-pin to an older audit-lock would audit with code this repository has replaced."""
    security = _load("python-gates.yml")["jobs"]["security"]["steps"]
    pin = next(
        str(step["uses"]) for step in security if "/audit-lock@" in str(step.get("uses", ""))
    ).split("@")[1]

    def tree(ref: str) -> str:
        return subprocess.run(  # noqa: S603  # nosec B603 - git with fixed arguments
            [shutil.which("git") or "git", "rev-parse", f"{ref}:.github/actions/audit-lock"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    assert tree(pin) == tree("HEAD")
