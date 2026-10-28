"""The shared ``python-gates`` workflow every stack repository calls, defined once.

It runs the Required gates of the engineering guide: the lock check, ruff, pyright, pre-commit,
pytest with the coverage floor and the changed-line floor, the dependency audit with Bandit, the
build with ``twine check``, a clean-install import, and the strict docs build. Repositories call it
at the job level, pinned by commit, so every rule here holds for every caller.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml


WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "python-gates.yml"
INPUTS = {
    "package",
    "python-versions",
    "sync-args",
    "docs-sync-args",
    "pytest-args",
    "test-env",
    "import-modules",
    "os",
    "already-tested",
    "only-security",
    "uv-version",
}
JOBS = {"lint", "test", "security", "build", "docs"}
FULL_SHA = re.compile(r"@[0-9a-f]{40}$")
# The stack is a hierarchy built bottom-up from substrax: its shared workflow may use substrax's
# own actions and third-party actions, never another stack repository's.
STACK_OWNERS = ("avitai/", "mahdi-shafiei/")


def _workflow() -> dict[str, Any]:
    # BaseLoader keeps every scalar a string, so the `on:` key stays `on` rather than True.
    return yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)  # noqa: S506


def _jobs() -> dict[str, Any]:
    return _workflow()["jobs"]


def _steps(job: str) -> list[dict[str, Any]]:
    return list(_jobs()[job]["steps"])


def _runs(job: str) -> str:
    return "\n".join(str(step.get("run", "")) for step in _steps(job))


def _env(step: dict[str, Any]) -> dict[str, str]:
    return {str(key): str(value) for key, value in (step.get("env") or {}).items()}


def _uses() -> list[str]:
    return [
        str(step["uses"])
        for job in _jobs().values()
        for step in job.get("steps", [])
        if "uses" in step
    ]


def test_it_runs_only_when_called_with_the_declared_inputs() -> None:
    workflow = _workflow()

    assert set(workflow["on"]) == {"workflow_call"}
    assert set(workflow["on"]["workflow_call"]["inputs"]) == INPUTS
    assert workflow["on"]["workflow_call"]["inputs"]["package"]["required"] == "true"


def test_it_leaves_concurrency_and_env_to_the_caller() -> None:
    """A called workflow's `github.workflow` is the caller's, so a group here would cancel the caller."""
    workflow = _workflow()

    assert "concurrency" not in workflow
    assert "env" not in workflow
    assert [name for name, job in _jobs().items() if "concurrency" in job] == []


def test_it_runs_the_gate_jobs() -> None:
    assert set(_jobs()) == JOBS


def test_lint_runs_the_lock_check_pyright_and_every_hook() -> None:
    """pyright runs here itself: a repository's pre-commit file need not hold a pyright hook.

    ruff runs through the ruff and ruff-format hooks every stack repository's pre-commit file holds.
    A pathless `ruff format --check` would also format Python blocks inside Markdown, a gate no
    repository runs today.
    """
    runs = _runs("lint")

    for command in ("uv lock --check", "pyright --warnings", "pre-commit run --all-files"):
        assert command in runs, command


def test_pre_commit_leaves_the_suite_bandit_and_pyright_to_the_jobs_that_run_them() -> None:
    step = next(step for step in _steps("lint") if "pre-commit run" in str(step.get("run", "")))

    skipped = set(_env(step)["SKIP"].split(","))
    assert {"pytest", "bandit", "pyright"} <= skipped
    assert not {"ruff", "ruff-format"} & skipped


def test_lint_is_never_stood_down_for_an_already_tested_tree() -> None:
    """Lint saves the pre-commit cache main must hold, so a merge still runs it."""
    condition = _jobs()["lint"].get("if", "")

    assert "already-tested" not in condition
    cache = next(
        step for step in _steps("lint") if str(step.get("uses", "")).startswith("actions/cache@")
    )
    assert "hashFiles('.pre-commit-config.yaml')" in cache["with"]["key"]
    assert "runner.os" in cache["with"]["key"]


@pytest.mark.parametrize("job", ["test", "security", "build", "docs"])
def test_work_a_pull_request_already_proved_stands_down(job: str) -> None:
    assert "inputs.already-tested != 'true'" in _jobs()[job]["if"]


@pytest.mark.parametrize("job", ["lint", "test", "build", "docs"])
def test_the_scheduled_audit_runs_security_alone(job: str) -> None:
    assert "!inputs.only-security" in _jobs()[job]["if"]
    assert "only-security" not in _jobs()["security"].get("if", "")


@pytest.mark.parametrize("job", ["test", "build"])
def test_the_matrix_jobs_run_on_the_callers_runner_under_the_macos_cap(job: str) -> None:
    spec = _jobs()[job]

    assert spec["runs-on"] == "${{ inputs.os }}"
    assert int(spec["strategy"]["max-parallel"]) <= 2
    assert spec["strategy"]["fail-fast"] == "false"
    assert spec["strategy"]["matrix"]["python-version"] == "${{ fromJSON(inputs.python-versions) }}"


def test_the_tests_report_coverage_and_hold_new_lines_to_eighty_percent() -> None:
    runs = _runs("test")

    assert "pytest" in runs
    assert "--cov-report=xml" in runs
    diff = next(step for step in _steps("test") if "diff-cover" in str(step.get("run", "")))
    assert "--fail-under=80" in diff["run"]
    assert "--compare-branch=origin/main" in diff["run"]
    assert "github.event_name == 'pull_request'" in diff["if"]
    checkout = next(
        step for step in _steps("test") if str(step.get("uses", "")).startswith("actions/checkout@")
    )
    assert checkout["with"]["fetch-depth"] == "0"


def test_security_audits_the_lock_after_setup_uv_through_substrax_by_commit() -> None:
    """A called workflow resolves `./` in the caller's checkout, so the action is named in full."""
    uses = [str(step.get("uses", "")) for step in _steps("security")]

    audit = next(index for index, use in enumerate(uses) if "/.github/actions/audit-lock@" in use)
    assert re.fullmatch(r"avitai/substrax/\.github/actions/audit-lock@[0-9a-f]{40}", uses[audit])
    assert any(use.startswith("astral-sh/setup-uv@") for use in uses[:audit])


def test_bandit_runs_whatever_the_audit_found_and_its_failure_fails_the_job() -> None:
    bandit = next(step for step in _steps("security") if "bandit " in str(step.get("run", "")))

    assert bandit.get("if") == "${{ !cancelled() }}"
    assert "continue-on-error" not in bandit
    assert "||" not in bandit["run"]
    assert '"src/$PACKAGE"' in bandit["run"]
    assert _env(bandit)["PACKAGE"] == "${{ inputs.package }}"


def test_the_build_is_checked_and_imports_from_a_clean_environment() -> None:
    runs = _runs("build")

    assert "uv build" in runs
    assert "twine check --strict" in runs
    smoke = next(step for step in _steps("build") if "uv pip install" in str(step.get("run", "")))
    assert _env(smoke)["MODULES"] == "${{ inputs.import-modules }}"


def test_the_docs_build_strictly_with_no_path_filter() -> None:
    docs = next(step for step in _steps("docs") if "mkdocs build" in str(step.get("run", "")))

    assert "mkdocs build --strict" in docs["run"]
    assert _env(docs)["DOCS_SYNC_ARGS"] == "${{ inputs.docs-sync-args }}"


def test_no_script_interpolates_an_input() -> None:
    """An input reaches a script through `env`, never as `${{ }}` text the shell would run as code."""
    interpolated = [
        f"{name}: {step.get('name', step['run'][:40])}"
        for name, job in _jobs().items()
        for step in job.get("steps", [])
        if "run" in step and "${{" in str(step["run"])
    ]

    assert interpolated == []


def test_every_action_is_pinned_by_commit() -> None:
    assert [use for use in _uses() if not FULL_SHA.search(use)] == []


def test_it_uses_no_stack_repository_but_substrax() -> None:
    others = [
        use
        for use in _uses()
        if use.startswith(STACK_OWNERS) and not use.startswith("avitai/substrax/")
    ]

    assert others == []


def test_every_setup_uv_installs_the_pinned_uv_and_prunes_its_cache() -> None:
    inputs = _workflow()["on"]["workflow_call"]["inputs"]
    setups = [
        step["with"]
        for job in _jobs().values()
        for step in job.get("steps", [])
        if str(step.get("uses", "")).startswith("astral-sh/setup-uv@")
    ]

    assert re.fullmatch(r"\d+\.\d+\.\d+", inputs["uv-version"]["default"])
    assert len(setups) == len(JOBS)
    for spec in setups:
        assert spec["version"] == "${{ inputs.uv-version }}"
        assert spec["prune-cache"] == "true"
