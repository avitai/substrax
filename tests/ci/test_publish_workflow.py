"""The publishing workflow releases the distributions CI built and checked, never its own build."""

from __future__ import annotations

import io
import os
import re
import shutil
import subprocess
import tarfile
import zipfile
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
    assert release_job["needs"] == "check-release"
    assert release_job["if"] == (
        "github.event_name == 'workflow_dispatch' && github.event.inputs.target == 'github-release'"
    )
    assert release_job["permissions"]["contents"] == "write"
    assert re.fullmatch(r"softprops/action-gh-release@[0-9a-f]{40}", release_step["uses"])
    assert release_step["with"]["generate_release_notes"] == "true"
    assert release_step["with"]["tag_name"] == "${{ github.event.inputs.version_tag }}"


def test_every_release_checks_the_distributions_ci_built_against_its_tag() -> None:
    """A release or a PyPI run checks the version first; a TestPyPI trial may run from any ref."""
    check = _jobs()["check-release"]
    download = next(
        step
        for step in check["steps"]
        if str(step.get("uses", "")).startswith("actions/download-artifact@")
    )

    assert check["needs"] == "validate"
    assert check["if"] == (
        "github.event_name == 'release' || github.event.inputs.target == 'pypi' || "
        "github.event.inputs.target == 'github-release'"
    )
    assert download["with"] == {"name": ARTIFACT, "path": "dist/"}


def _distributions(directory: Path, wheel: str, sdist: str) -> None:
    """Write a wheel and an sdist whose metadata hold the given versions, as CI's Build does."""
    dist = directory / "dist"
    dist.mkdir()
    with zipfile.ZipFile(dist / f"substrax-{wheel}-py3-none-any.whl", "w") as archive:
        archive.writestr(
            f"substrax-{wheel}.dist-info/METADATA", f"Name: substrax\nVersion: {wheel}\n"
        )
    info = f"Name: substrax\nVersion: {sdist}\n".encode()
    with tarfile.open(dist / f"substrax-{sdist}.tar.gz", "w:gz") as archive:
        member = tarfile.TarInfo(f"substrax-{sdist}/PKG-INFO")
        member.size = len(info)
        archive.addfile(member, io.BytesIO(info))


def _check_release(directory: Path, **env: str) -> subprocess.CompletedProcess[str]:
    """Run the version check's own command in ``directory`` with the run's context in ``env``."""
    steps = _jobs()["check-release"]["steps"]
    check = next(step for step in steps if step.get("name") == "The version matches the tag")
    assert set(check["env"]) == {"RUN_REF", "TARGET", "RELEASE_TAG"}
    return subprocess.run(  # noqa: S603  # nosec B603 - the workflow's own command on fixture files
        [shutil.which("bash") or "bash", "-e", "-o", "pipefail", "-c", check["run"]],
        cwd=directory,
        env={"PATH": os.environ["PATH"], "TARGET": "", "RELEASE_TAG": ""} | env,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("ref", "target", "release_tag"),
    [
        ("refs/tags/v0.1.23", "", ""),
        ("refs/tags/v0.1.23", "pypi", ""),
        ("refs/tags/v0.1.23", "github-release", "v0.1.23"),
    ],
)
def test_a_release_from_its_own_tag_passes(
    tmp_path: Path, ref: str, target: str, release_tag: str
) -> None:
    _distributions(tmp_path, "0.1.23", "0.1.23")

    result = _check_release(tmp_path, RUN_REF=ref, TARGET=target, RELEASE_TAG=release_tag)

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("ref", "target", "release_tag", "says"),
    [
        ("refs/tags/v0.1.22", "", "", "this run is on refs/tags/v0.1.22"),
        ("refs/heads/main", "pypi", "", "this run is on refs/heads/main"),
        ("refs/tags/v0.1.23", "github-release", "v0.1.24", "version_tag is 'v0.1.24'"),
        ("refs/tags/v0.1.23", "github-release", "", "version_tag is ''"),
    ],
)
def test_a_release_whose_tag_differs_from_its_version_stops(
    tmp_path: Path, ref: str, target: str, release_tag: str, says: str
) -> None:
    _distributions(tmp_path, "0.1.23", "0.1.23")

    result = _check_release(tmp_path, RUN_REF=ref, TARGET=target, RELEASE_TAG=release_tag)

    assert result.returncode != 0
    assert says in result.stderr


def test_distributions_that_disagree_on_their_version_stop(tmp_path: Path) -> None:
    _distributions(tmp_path, "0.1.23", "0.1.22")

    result = _check_release(tmp_path, RUN_REF="refs/tags/v0.1.23")

    assert result.returncode != 0
    assert "['0.1.22', '0.1.23']" in result.stderr


def test_a_release_with_no_distributions_stops(tmp_path: Path) -> None:
    (tmp_path / "dist").mkdir()

    result = _check_release(tmp_path, RUN_REF="refs/tags/v0.1.23")

    assert result.returncode != 0
    assert "[]" in result.stderr


def test_publish_pypi_waits_for_the_version_check_and_manual_release_creation() -> None:
    """PyPI publishing needs the checked tag, and a manual release waits for its GitHub Release."""
    publish_job: dict[str, Any] = _jobs()["publish-pypi"]

    assert publish_job["needs"] == ["validate", "check-release", "create-github-release"]
    assert publish_job["if"] == (
        "always() && needs.check-release.result == 'success' && "
        "(needs.create-github-release.result == 'success' || "
        "needs.create-github-release.result == 'skipped')"
    )


def test_releasing_docs_describe_generated_release_note_automation() -> None:
    """Human release docs should match the automated release path."""
    releasing = RELEASING_DOC.read_text()

    assert "No commit or tag push creates a release by itself" in releasing
    assert "softprops/action-gh-release" in releasing
    assert "generate_release_notes: true" in releasing
    assert "target=github-release" in releasing
    assert "runs CI" in releasing
    assert "refs/tags/v<version>" in releasing
