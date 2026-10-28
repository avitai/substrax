"""The publishing workflow releases the distributions CI built and checked, never its own build."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, cast

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
PUBLISH_WORKFLOW = ROOT / ".github" / "workflows" / "publish.yml"
RELEASING_DOC = ROOT / "RELEASING.md"
ARTIFACT = "python-package-distributions"
PUBLISHERS = ("publish-testpypi", "publish-pypi")
# What building distributions looks like in a step; the gates' Build job does it for publish.yml.
BUILDS = re.compile(
    r"\bpython -m build\b|\buv build\b|\bactions/setup-python@|\bactions/upload-artifact@"
)


def _load_publish_workflow() -> dict[str, Any]:
    # BaseLoader builds only strings, lists and dicts (no object construction), and it is
    # the one loader that keeps the `on:` key a string instead of the YAML 1.1 boolean.
    return cast(dict[str, Any], yaml.load(PUBLISH_WORKFLOW.read_text(), Loader=yaml.BaseLoader))  # noqa: S506


def _jobs() -> dict[str, Any]:
    return _load_publish_workflow()["jobs"]


def test_publish_workflow_does_not_create_releases_on_push() -> None:
    """No commit or tag push should create a GitHub release by itself."""
    workflow = _load_publish_workflow()

    assert "push" not in workflow["on"]


def test_a_release_runs_ci_on_the_released_commit_first() -> None:
    """The gates, the Orbax-floor tests and the derived status all run before anything publishes."""
    assert _jobs()["validate"]["uses"] == "./.github/workflows/ci.yml"


def test_the_publishing_workflow_builds_nothing_itself() -> None:
    """A release ships the distributions CI built and checked, not a second build of its own."""
    building = [
        f"{name}: {step.get('uses', step.get('run', ''))}"
        for name, job in _jobs().items()
        for step in job.get("steps", [])
        if BUILDS.search(f"{step.get('uses', '')} {step.get('run', '')}")
    ]

    assert building == []


@pytest.mark.parametrize("job", PUBLISHERS)
def test_each_index_receives_the_distributions_ci_built(job: str) -> None:
    spec = _jobs()[job]
    needs = spec["needs"] if isinstance(spec["needs"], list) else [spec["needs"]]
    download = next(
        step
        for step in spec["steps"]
        if str(step.get("uses", "")).startswith("actions/download-artifact@")
    )

    assert "validate" in needs
    assert download["with"]["name"] == ARTIFACT
    assert download["with"]["path"] == "dist/"


def test_manual_release_target_generates_release_notes() -> None:
    """Only an explicit manual release target should generate release notes."""
    workflow = _load_publish_workflow()

    workflow_inputs: dict[str, Any] = workflow["on"]["workflow_dispatch"]["inputs"]
    target_options: list[str] = workflow_inputs["target"]["options"]
    release_job: dict[str, Any] = workflow["jobs"]["create-github-release"]
    release_step: dict[str, Any] = release_job["steps"][-1]

    assert "github-release" in target_options
    assert "version_tag" in workflow_inputs
    assert release_job["needs"] == "validate"
    assert release_job["if"] == (
        "github.event_name == 'workflow_dispatch' && github.event.inputs.target == 'github-release'"
    )
    assert release_job["permissions"]["contents"] == "write"
    assert re.fullmatch(r"softprops/action-gh-release@[0-9a-f]{40}", release_step["uses"])
    assert release_step["with"]["generate_release_notes"] == "true"
    assert release_step["with"]["tag_name"] == "${{ github.event.inputs.version_tag }}"


def test_a_manual_release_runs_on_the_tag_it_releases() -> None:
    """CI builds the run's ref, so a run from any other ref would ship another commit's build."""
    steps = _jobs()["create-github-release"]["steps"]
    check = next(step for step in steps if step.get("name") == "Validate release tag")

    assert check["env"]["RUN_REF"] == "${{ github.ref }}"
    assert '"${RUN_REF}" != "refs/tags/${RELEASE_TAG}"' in check["run"]


def test_publish_pypi_waits_for_ci_and_manual_release_creation() -> None:
    """PyPI publishing needs a passing CI run, and a manual release waits for its GitHub Release."""
    publish_job: dict[str, Any] = _jobs()["publish-pypi"]
    condition: str = publish_job["if"]

    assert publish_job["needs"] == ["validate", "create-github-release"]
    assert "needs.validate.result == 'success'" in condition
    assert "github.event.inputs.target == 'github-release'" in condition
    assert "needs.create-github-release.result == 'success'" in condition


def test_releasing_docs_describe_generated_release_note_automation() -> None:
    """Human release docs should match the automated release path."""
    releasing = RELEASING_DOC.read_text()

    assert "No commit or tag push creates a release by itself" in releasing
    assert "softprops/action-gh-release" in releasing
    assert "generate_release_notes: true" in releasing
    assert "target=github-release" in releasing
    assert "runs CI" in releasing
