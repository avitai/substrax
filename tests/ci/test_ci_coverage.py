"""CI fails when coverage drops below the floor, and uploads coverage nowhere."""

from __future__ import annotations

import re
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
CAP_OVERRIDES = ("--no-cov", "-o addopts", "--override-ini")
# The test job in python-gates.yml stands down only for a tree its pull request already proved,
# or on the weekly audit, which runs nothing but the dependency audit.
GATE_CONDITION = "${{ !inputs.only-security && inputs.already-tested != 'true' }}"


def _load_workflow(path: Path) -> dict[str, Any]:
    # BaseLoader keeps every scalar a string, so the `on:` key stays `on` rather than True.
    return yaml.load(path.read_text(), Loader=yaml.BaseLoader)  # noqa: S506


def _addopts(pyproject: dict[str, Any]) -> str:
    addopts = pyproject["tool"]["pytest"]["ini_options"].get("addopts", "")
    return " ".join(addopts) if isinstance(addopts, list) else addopts


def coverage_cap_violations(
    workflow: dict[str, Any], gates: dict[str, Any], pyproject: dict[str, Any]
) -> list[str]:
    """Return why CI would not fail below the coverage floor, if it would not.

    ``workflow`` is ci.yml, which calls ``gates`` (python-gates.yml) and passes its pytest
    arguments. The test job's shards each measure part of the suite, so the floor is held where
    they meet: the coverage job combines each Python's shards and reports them against
    ``[tool.coverage.report] fail_under``.
    """
    test, coverage = gates["jobs"]["test"], gates["jobs"]["coverage"]
    script = next(step["run"] for step in test["steps"] if "pytest" in step.get("run", ""))
    command = f"{script} {workflow['jobs']['gates']['with'].get('pytest-args', '')}"
    report = "\n".join(step.get("run", "") for step in coverage["steps"]).split("diff-cover")[0]
    floor = pyproject["tool"]["coverage"]["report"].get("fail_under", 0)

    problems = []
    if float(floor) < 80:
        problems.append(f"the configured coverage floor is {floor}, not at least 80")
    if "coverage report" not in report or "--fail-under" in report:
        problems.append("the coverage job does not report against the configured floor")
    if coverage.get("needs") != "test":
        problems.append(f"the coverage job needs {coverage.get('needs')}, not the test job")
    if not {"push", "pull_request"} <= set(workflow["on"]):
        problems.append(f"CI runs on {sorted(workflow['on'])}, not on both push and pull_request")
    # The one condition either job may carry is the already-tested gate: they stand down only
    # where the same tree already reported coverage on the pull request that produced it.
    problems += [
        f"the {name} job only runs when {job['if']}"
        for name, job in (("test", test), ("coverage", coverage))
        if job.get("if") not in (None, GATE_CONDITION)
    ]
    problems += [
        f"the pytest command overrides the floor with {override}"
        for override in CAP_OVERRIDES
        if override in command
    ]
    return problems


def test_ci_fails_below_the_coverage_floor() -> None:
    """Each Python's shards are held together to a floor of at least 80% on every push and pull
    request."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())

    workflow = _load_workflow(WORKFLOWS / "ci.yml")
    gates = _load_workflow(WORKFLOWS / "python-gates.yml")

    assert coverage_cap_violations(workflow, gates, pyproject) == []


def _lower_the_report(gates: dict[str, Any], pyproject: dict[str, Any]) -> None:
    del pyproject
    step = next(
        s for s in gates["jobs"]["coverage"]["steps"] if "coverage report" in s.get("run", "")
    )
    step["run"] = step["run"].replace("coverage report", "coverage report --fail-under=50")


def _unhook_the_coverage_job(gates: dict[str, Any], pyproject: dict[str, Any]) -> None:
    del pyproject
    gates["jobs"]["coverage"]["needs"] = "lint"


def _lower_the_configured_floor(gates: dict[str, Any], pyproject: dict[str, Any]) -> None:
    del gates
    pyproject["tool"]["coverage"]["report"]["fail_under"] = 70


def _drop_coverage_from_the_shards(gates: dict[str, Any], pyproject: dict[str, Any]) -> None:
    del pyproject
    step = next(s for s in gates["jobs"]["test"]["steps"] if "pytest" in s.get("run", ""))
    step["run"] = step["run"].replace("--cov-report=xml", "--no-cov")


@pytest.mark.parametrize(
    "lose_the_floor",
    [
        _lower_the_report,
        _unhook_the_coverage_job,
        _lower_the_configured_floor,
        _drop_coverage_from_the_shards,
    ],
)
def test_each_way_of_losing_the_floor_is_reported(lose_the_floor: Callable[..., None]) -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    workflow = _load_workflow(WORKFLOWS / "ci.yml")
    gates = _load_workflow(WORKFLOWS / "python-gates.yml")

    lose_the_floor(gates, pyproject)

    assert coverage_cap_violations(workflow, gates, pyproject) != []


def test_coverage_follows_child_processes() -> None:
    """Code the tests run in child interpreters is measured, not reported as missed.

    ``substrax.testing.run_python`` and ``pytester.runpytest_subprocess`` run code in child
    processes. pytest-cov 7 dropped its own subprocess support in favour of coverage's
    ``[run] patch = ["subprocess"]``, which hands the configuration to every child.
    """
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())

    assert "subprocess" in pyproject["tool"]["coverage"]["run"].get("patch", [])


def test_pytest_measures_coverage_by_source_directory() -> None:
    """Every ``--cov`` in addopts names a directory, so package files run as scripts are measured.

    coverage treats a ``--cov`` value that is not a directory as a package and matches executed
    code to it by module name. A file run as a script executes as ``__main__`` without a module
    spec, so under ``--cov=substrax`` it never counts toward the package and reports 0%; the
    SkyPilot runner ``substrax/compute/sky_sdk.py`` is such a file.
    """
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    sources = re.findall(r"--cov=(\S+)", _addopts(pyproject))

    assert sources, "pytest addopts measure no coverage source"
    assert [source for source in sources if not (ROOT / source).is_dir()] == []


def test_ci_uploads_no_coverage_to_codecov() -> None:
    """coverage.py in the test job is the coverage gate; no workflow uploads to Codecov."""
    uses = [
        step.get("uses", "")
        for path in sorted(WORKFLOWS.glob("*.yml"))
        for job in _load_workflow(path).get("jobs", {}).values()
        for step in job.get("steps", [])
    ]

    assert any(action.startswith("actions/checkout@") for action in uses)
    assert [action for action in uses if action.startswith("codecov/")] == []
