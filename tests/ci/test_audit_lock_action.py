"""The ``audit-lock`` action audits every extra a consumer's uv lockfile resolves.

A consumer that runs ``pip-audit`` on its environment audits only the extras installed there, and a
bare ``pip-audit`` run through ``uvx`` audits pip-audit's own environment. The action exports each
group of compatible extras from the lock and audits the export, with a fresh advisory cache, and
fails on an advisory the consumer's ``[tool.substrax.audit-lock.ignore]`` table does not name or on
an entry of that table no advisory matches any more.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess  # nosec B404
import sys
import tomllib
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
ACTION = ROOT / ".github" / "actions" / "audit-lock"
FIXTURE = Path(__file__).parent / "fixtures" / "audit_lock" / "pyproject.toml"
PIP_AUDIT_VERSION = "9.9.9"

FAKE_UV = """#!/usr/bin/env bash
set -euo pipefail
printf 'uv %s\\n' "$*" >> "$AUDIT_LOCK_LOG"
if [ -n "${FAKE_UV_EXIT:-}" ]; then exit "$FAKE_UV_EXIT"; fi
output=""
while [ $# -gt 0 ]; do
  if [ "$1" = "--output-file" ]; then output="$2"; shift; fi
  shift
done
printf 'pkg==1.0 \\\\\\n    --hash=sha256:00\\n' > "$output"
"""

FAKE_UVX = """#!/usr/bin/env bash
set -euo pipefail
printf 'uvx %s\\n' "$*" >> "$AUDIT_LOCK_LOG"
output=""
while [ $# -gt 0 ]; do
  if [ "$1" = "--output" ]; then output="$2"; shift; fi
  shift
done
cp "$FAKE_REPORT" "$output"
exit "${FAKE_PIP_AUDIT_EXIT:-0}"
"""


@pytest.fixture(scope="module")
def audit() -> ModuleType:
    spec = importlib.util.spec_from_file_location("audit_lock", ACTION / "audit_lock.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["audit_lock"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def action() -> dict[str, Any]:
    return yaml.safe_load((ACTION / "action.yml").read_text())


def _vuln(identifier: str, *aliases: str) -> dict[str, Any]:
    return {"id": identifier, "fix_versions": [], "aliases": list(aliases), "description": ""}


def _report(*dependencies: tuple[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {
        "dependencies": [
            {"name": name, "version": "1.0", "vulns": vulns} for name, vulns in dependencies
        ],
        "fixes": [],
    }


class TestExtrasGroups:
    """Conflicting extras cannot be exported together, so the audit covers them in groups."""

    def test_extras_without_conflicts_form_one_group(self, audit: ModuleType) -> None:
        assert audit.extras_groups(["a", "b", "c"], []) == [["a", "b", "c"]]

    def test_conflicting_extras_never_share_a_group(self, audit: ModuleType) -> None:
        conflicts = [{"a", "b"}, {"a", "c"}]

        groups = audit.extras_groups(["a", "b", "c", "d"], conflicts)

        assert all(not pair <= set(group) for group in groups for pair in conflicts)
        assert set().union(*groups) == {"a", "b", "c", "d"}

    def test_an_extra_compatible_with_every_group_joins_each(self, audit: ModuleType) -> None:
        groups = audit.extras_groups(["a", "b", "d"], [{"a", "b"}])

        assert all("d" in group for group in groups)

    def test_a_project_without_extras_audits_its_base_dependencies(self, audit: ModuleType) -> None:
        assert audit.extras_groups([], []) == [[]]

    def test_the_conflicts_are_read_from_tool_uv(self, audit: ModuleType) -> None:
        project = tomllib.loads(FIXTURE.read_text())

        assert audit.declared_conflicts(project) == [{"a", "b"}]
        assert audit.declared_conflicts({}) == []


class TestFindings:
    """An advisory fails the audit unless it is ignored, and an ignore must still match."""

    def test_an_unignored_advisory_is_a_finding(self, audit: ModuleType) -> None:
        report = _report(("pkg", [_vuln("PYSEC-1")]))

        assert audit.findings([report], {}) == [("pkg", "1.0", "PYSEC-1")]

    def test_an_advisory_ignored_by_its_id_or_an_alias_is_not(self, audit: ModuleType) -> None:
        report = _report(("pkg", [_vuln("PYSEC-1"), _vuln("PYSEC-2", "CVE-2")]))

        assert audit.findings([report], {"PYSEC-1": "why", "CVE-2": "why"}) == []

    def test_an_advisory_reported_twice_is_one_finding(self, audit: ModuleType) -> None:
        report = _report(("pkg", [_vuln("PYSEC-1"), _vuln("PYSEC-1")]))

        assert audit.findings([report, report], {}) == [("pkg", "1.0", "PYSEC-1")]

    def test_an_ignore_no_advisory_matches_is_stale(self, audit: ModuleType) -> None:
        report = _report(("pkg", [_vuln("PYSEC-1", "CVE-1")]))

        stale = audit.stale_ignores([report], {"CVE-1": "why", "PYSEC-9": "why"})

        assert stale == ["PYSEC-9"]


class TestIgnoreTable:
    """Each consumer keeps its ignores, with a reason for each, in its own pyproject."""

    def test_a_missing_table_ignores_nothing(self, audit: ModuleType) -> None:
        assert audit.load_ignored({"project": {"name": "x"}}) == {}
        assert audit.load_ignored({"tool": {"substrax": {}}}) == {}

    def test_a_table_maps_each_advisory_to_its_reason(self, audit: ModuleType) -> None:
        project = tomllib.loads(
            '[tool.substrax.audit-lock.ignore]\n"PYSEC-1" = "no fixed release"\n'
        )

        assert audit.load_ignored(project) == {"PYSEC-1": "no fixed release"}

    @pytest.mark.parametrize("reason", ['""', '"   "'], ids=["empty", "blank"])
    def test_an_empty_reason_is_refused(self, audit: ModuleType, reason: str) -> None:
        project = tomllib.loads(f'[tool.substrax.audit-lock.ignore]\n"PYSEC-1" = {reason}\n')

        with pytest.raises(ValueError, match="PYSEC-1"):
            audit.load_ignored(project)

    @pytest.mark.parametrize("reason", ["1", "true", '["why"]', '{why = "x"}'])
    def test_a_reason_that_is_not_a_string_is_refused(self, audit: ModuleType, reason: str) -> None:
        project = tomllib.loads(f'[tool.substrax.audit-lock.ignore]\n"PYSEC-1" = {reason}\n')

        with pytest.raises(TypeError, match="PYSEC-1"):
            audit.load_ignored(project)

    def test_an_ignore_entry_that_is_not_a_table_is_refused(self, audit: ModuleType) -> None:
        project = tomllib.loads('[tool.substrax.audit-lock]\nignore = ["PYSEC-1"]\n')

        with pytest.raises(TypeError, match="ignore"):
            audit.load_ignored(project)


class TestMain:
    """The script end to end, on a fixture project, with ``uv`` and ``uvx`` replaced by fakes."""

    @pytest.fixture
    def project(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        project = tmp_path / "project"
        project.mkdir()
        shutil.copy(FIXTURE, project / "pyproject.toml")
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        for name, body in (("uv", FAKE_UV), ("uvx", FAKE_UVX)):
            (bin_dir / name).write_text(body)
            (bin_dir / name).chmod(0o755)
        monkeypatch.chdir(project)
        monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")
        monkeypatch.setenv("AUDIT_LOCK_LOG", str(tmp_path / "calls.log"))
        monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
        monkeypatch.delenv("FAKE_UV_EXIT", raising=False)
        monkeypatch.delenv("FAKE_PIP_AUDIT_EXIT", raising=False)
        self.report(tmp_path, monkeypatch, _report(("pkg", [])))
        return project

    @staticmethod
    def report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, report: dict[str, Any]) -> None:
        path = tmp_path / "report.json"
        path.write_text(json.dumps(report))
        monkeypatch.setenv("FAKE_REPORT", str(path))

    @staticmethod
    def ignore(project: Path, *entries: tuple[str, str]) -> None:
        table = "".join(f'"{identifier}" = "{reason}"\n' for identifier, reason in entries)
        with (project / "pyproject.toml").open("a") as pyproject:
            pyproject.write(f"\n[tool.substrax.audit-lock.ignore]\n{table}")

    @staticmethod
    def calls(project: Path) -> list[str]:
        return (project.parent / "calls.log").read_text().splitlines()

    def run(self, audit: ModuleType) -> int:
        return audit.main(["--pip-audit-version", PIP_AUDIT_VERSION])

    def test_a_clean_lock_passes_after_auditing_every_group(
        self, audit: ModuleType, project: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert self.run(audit) == 0

        out = capsys.readouterr().out
        assert "audited extras: a, c" in out
        assert "audited extras: b, c" in out
        exports = [call for call in self.calls(project) if call.startswith("uv ")]
        assert len(exports) == 2
        assert all("export --frozen --all-groups --no-emit-project" in call for call in exports)
        assert "--extra a --extra c" in exports[0]
        assert "--extra b --extra c" in exports[1]

    def test_pip_audit_runs_pinned_and_isolated_on_the_export(
        self, audit: ModuleType, project: Path
    ) -> None:
        assert self.run(audit) == 0

        audits = [call for call in self.calls(project) if call.startswith("uvx ")]
        assert len(audits) == 2
        for call in audits:
            assert f"--from pip-audit=={PIP_AUDIT_VERSION} pip-audit" in call
            for flag in ("--requirement", "--disable-pip", "--no-deps", "--cache-dir"):
                assert flag in call
            assert "--format json" in call
        cache_dirs = [re.search(r"--cache-dir (\S+)", call) for call in audits]
        assert all(match is not None for match in cache_dirs)
        assert not any(Path(match.group(1)).exists() for match in cache_dirs if match)

    @pytest.mark.usefixtures("project")
    def test_an_unignored_advisory_fails_the_audit(
        self,
        audit: ModuleType,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self.report(tmp_path, monkeypatch, _report(("pkg", [_vuln("PYSEC-1", "CVE-1")])))
        monkeypatch.setenv("FAKE_PIP_AUDIT_EXIT", "1")
        summary = tmp_path / "summary.md"
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))

        assert self.run(audit) == 1

        assert "advisory: pkg 1.0 PYSEC-1" in capsys.readouterr().out
        written = summary.read_text()
        assert "a, c" in written
        assert "PYSEC-1" in written

    def test_an_ignored_advisory_passes(
        self,
        audit: ModuleType,
        project: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self.report(tmp_path, monkeypatch, _report(("pkg", [_vuln("PYSEC-1", "CVE-1")])))
        monkeypatch.setenv("FAKE_PIP_AUDIT_EXIT", "1")
        self.ignore(project, ("CVE-1", "no fixed release"))

        assert self.run(audit) == 0

        assert "advisory:" not in capsys.readouterr().out

    def test_a_stale_ignore_fails_the_audit(
        self, audit: ModuleType, project: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self.ignore(project, ("PYSEC-9", "fixed upstream long ago"))

        assert self.run(audit) == 1

        assert "stale ignore (no advisory matches it): PYSEC-9" in capsys.readouterr().out

    @pytest.mark.usefixtures("project")
    def test_a_failing_export_raises(
        self, audit: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FAKE_UV_EXIT", "2")

        with pytest.raises(subprocess.CalledProcessError):
            self.run(audit)

    @pytest.mark.usefixtures("project")
    def test_a_pip_audit_error_raises(
        self, audit: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FAKE_PIP_AUDIT_EXIT", "2")

        with pytest.raises(subprocess.CalledProcessError):
            self.run(audit)

    @pytest.mark.usefixtures("project")
    def test_a_missing_uv_names_setup_uv(
        self,
        audit: ModuleType,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        monkeypatch.setenv("PATH", str(empty))

        with pytest.raises(SystemExit, match="setup-uv"):
            self.run(audit)


class TestActionContract:
    """The composite action runs the tested script, pins pip-audit, and cannot pass on a failure."""

    def test_it_runs_the_script_from_the_action_path(self, action: dict[str, Any]) -> None:
        assert action["runs"]["using"] == "composite"
        steps = action["runs"]["steps"]
        runs = [step for step in steps if "run" in step]

        assert [step["shell"] for step in runs] == ["bash"]
        assert 'python3 "$GITHUB_ACTION_PATH/audit_lock.py"' in runs[0]["run"]
        assert runs[0]["working-directory"] == "${{ github.workspace }}"

    def test_it_pins_the_pip_audit_version_substrax_locks(self, action: dict[str, Any]) -> None:
        lock = tomllib.loads((ROOT / "uv.lock").read_text())
        locked = next(
            package["version"] for package in lock["package"] if package["name"] == "pip-audit"
        )
        pin = action["inputs"]["pip-audit-version"]["default"]
        step = action["runs"]["steps"][0]

        assert re.fullmatch(r"\d+(\.\d+)+", pin)
        assert pin == locked
        assert step["env"]["PIP_AUDIT_VERSION"] == "${{ inputs.pip-audit-version }}"
        assert '--pip-audit-version "$PIP_AUDIT_VERSION"' in step["run"]

    def test_no_step_can_pass_on_a_failure(self) -> None:
        text = (ACTION / "action.yml").read_text()

        assert "continue-on-error" not in text
        assert "|| true" not in text
        assert "|| :" not in text

    def test_it_says_uv_must_be_set_up_first(self, action: dict[str, Any]) -> None:
        assert "astral-sh/setup-uv" in action["description"]
