#!/usr/bin/env python3
"""Audit every extra a uv lockfile resolves, not only the ones an environment installs.

The audit reads an export of the lockfile, never an installed environment. ``pip-audit`` run on an
environment sees only the extras that environment installed, and a bare ``pip-audit`` run through
``uvx`` or ``pipx`` audits the tool's own isolated environment rather than the project. uv resolves
one lockfile across all extras, but extras declared as conflicting cannot be exported together, so
the extras are covered in groups that each hold no conflicting pair. Each group is exported from
the lock and audited by a pinned ``pip-audit``, run in isolation through ``uvx``, with a fresh
cache: a cached advisory database can hide an advisory published since it was filled.

The run fails on an advisory the project's ``[tool.substrax.audit-lock.ignore]`` table does not
name by id or alias, and on an entry of that table no advisory matches any more, so the table
cannot outlive its reasons.

Usage, from the project root (``uv`` and ``uvx`` on PATH):
    python3 audit_lock.py --pip-audit-version 2.10.1
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess  # nosec B404 - runs uv and uvx with fixed arguments
import sys
import tempfile
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


DESCRIPTION = "Audit every extra a uv lockfile resolves."
IGNORE_TABLE = "[tool.substrax.audit-lock.ignore]"
PIP_AUDIT_FINDINGS_EXIT = 1  # pip-audit's exit status when it reports an advisory

# (package, version, advisory id, aliases)
Advisory = tuple[str, str, str, frozenset[str]]


def declared_conflicts(project: Mapping[str, Any]) -> list[set[str]]:
    """The sets of extras ``[tool.uv] conflicts`` declares unable to resolve together.

    Args:
        project: The parsed ``pyproject.toml``.

    Returns:
        One set of extra names per declared conflict.
    """
    return [
        {item["extra"] for item in conflict if "extra" in item}
        for conflict in project.get("tool", {}).get("uv", {}).get("conflicts", [])
    ]


def extras_groups(extras: Sequence[str], conflicts: Sequence[set[str]]) -> list[list[str]]:
    """Groups of extras, none holding a conflicting pair, that together hold every extra.

    Each extra not yet covered seeds a group, which then takes every extra compatible with it, so
    an extra that conflicts with nothing is audited in every group's resolution. A project without
    extras is one empty group: its base dependencies.

    Args:
        extras: The declared extras, in declaration order.
        conflicts: The sets of extras that cannot resolve together.

    Returns:
        The groups, each in declaration order.
    """
    if not extras:
        return [[]]
    groups: list[list[str]] = []
    for seed in extras:
        if any(seed in group for group in groups):
            continue
        group = [seed]
        for extra in extras:
            if extra not in group and not any(c <= {*group, extra} for c in conflicts):
                group.append(extra)
        groups.append(sorted(group, key=list(extras).index))
    return groups


def load_ignored(project: Mapping[str, Any]) -> dict[str, str]:
    """The advisories the project ignores, each with its reason, from its ignore table.

    Args:
        project: The parsed ``pyproject.toml``.

    Returns:
        Advisory id or alias to reason; empty when the project has no ignore table.

    Raises:
        TypeError: The table is not a table, or a reason is not a string.
        ValueError: A reason is empty.
    """
    table = project.get("tool", {}).get("substrax", {}).get("audit-lock", {}).get("ignore", {})
    if not isinstance(table, dict):
        raise TypeError(f"{IGNORE_TABLE} must be a table of advisory id to reason")
    ignored: dict[str, str] = {}
    for identifier, reason in table.items():
        if not isinstance(reason, str):
            raise TypeError(f"{IGNORE_TABLE} {identifier}: the reason must be a string")
        if not reason.strip():
            raise ValueError(f"{IGNORE_TABLE} {identifier}: the reason is empty")
        ignored[identifier] = reason
    return ignored


def _advisories(reports: Iterable[Mapping[str, Any]]) -> set[Advisory]:
    """Every advisory the reports carry, once each."""
    return {
        (dependency["name"], dependency["version"], vuln["id"], frozenset(vuln["aliases"]))
        for report in reports
        for dependency in report["dependencies"]
        for vuln in dependency.get("vulns", [])
    }


def findings(
    reports: Iterable[Mapping[str, Any]], ignored: Mapping[str, str]
) -> list[tuple[str, str, str]]:
    """Advisories in pip-audit JSON reports that ``ignored`` names by neither id nor alias.

    Args:
        reports: pip-audit JSON reports.
        ignored: Advisory id or alias to reason.

    Returns:
        Sorted, de-duplicated (package, version, advisory id) triples.
    """
    return sorted(
        (name, version, identifier)
        for name, version, identifier, aliases in _advisories(reports)
        if identifier not in ignored and not aliases & ignored.keys()
    )


def stale_ignores(reports: Iterable[Mapping[str, Any]], ignored: Mapping[str, str]) -> list[str]:
    """Entries of ``ignored`` that no advisory in the reports matches by id or alias.

    Args:
        reports: pip-audit JSON reports.
        ignored: Advisory id or alias to reason.

    Returns:
        The stale entries, sorted.
    """
    named = {
        name for *_, identifier, aliases in _advisories(reports) for name in (identifier, *aliases)
    }
    return sorted(identifier for identifier in ignored if identifier not in named)


def _tool(name: str) -> str:
    """The path of ``name`` on PATH, or an exit that names the setup step it needs."""
    path = shutil.which(name)
    if path is None:
        raise SystemExit(
            f"audit-lock: `{name}` is not on PATH; run astral-sh/setup-uv before this action"
        )
    return path


def _label(group: Sequence[str]) -> str:
    """The extras of a group as the log and the summary name them."""
    return ", ".join(group) or "(none)"


def _audit_group(
    group: Sequence[str], project_dir: Path, work_dir: Path, pip_audit_version: str
) -> Mapping[str, Any]:
    """Export one group of extras from the lock and return pip-audit's JSON report on it."""
    requirements = work_dir / "requirements.txt"
    report = work_dir / "report.json"
    extras = [argument for extra in group for argument in ("--extra", extra)]
    subprocess.run(  # noqa: S603  # nosec B603 - uv with fixed arguments
        [_tool("uv"), "export", "--frozen", "--all-groups", "--no-emit-project", *extras,
         "--output-file", str(requirements)],
        cwd=project_dir, check=True, stdout=subprocess.DEVNULL,
    )  # fmt: skip
    with tempfile.TemporaryDirectory(dir=work_dir) as cache_dir:  # fresh: no stale advisories
        audited = subprocess.run(  # noqa: S603  # nosec B603 - uvx with fixed arguments
            [_tool("uvx"), "--from", f"pip-audit=={pip_audit_version}", "pip-audit",
             "--requirement", str(requirements), "--disable-pip", "--no-deps",
             "--cache-dir", cache_dir, "--format", "json", "--output", str(report)],
            cwd=project_dir, check=False,
        )  # fmt: skip
    if audited.returncode not in (0, PIP_AUDIT_FINDINGS_EXIT):  # findings: the caller judges
        raise subprocess.CalledProcessError(audited.returncode, audited.args)
    return json.loads(report.read_text())


def _summary(
    groups: Sequence[Sequence[str]], unignored: Sequence[tuple[str, str, str]], stale: Sequence[str]
) -> str:
    """The Markdown the run appends to the GitHub step summary."""
    lines = ["## Lockfile audit", ""]
    lines += [f"- audited extras: {_label(group)}" for group in groups]
    lines += ["", f"{len(unignored)} advisories not ignored, {len(stale)} stale ignores", ""]
    lines += [
        f"- advisory: {name} {version} {identifier}" for name, version, identifier in unignored
    ]
    lines += [f"- stale ignore: {identifier}" for identifier in stale]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Audit every group of extras of the project in the working directory.

    Args:
        argv: Command-line arguments; ``sys.argv`` when omitted.

    Returns:
        1 on an unignored advisory or a stale ignore, otherwise 0.
    """
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--pip-audit-version", required=True, help="the pip-audit release to run")
    arguments = parser.parse_args(argv)
    project_dir = Path.cwd()
    project = tomllib.loads((project_dir / "pyproject.toml").read_text(encoding="utf-8"))
    ignored = load_ignored(project)
    extras = list(project.get("project", {}).get("optional-dependencies", {}))
    groups = extras_groups(extras, declared_conflicts(project))
    reports: list[Mapping[str, Any]] = []
    for group in groups:
        with tempfile.TemporaryDirectory() as work_dir:
            reports.append(
                _audit_group(group, project_dir, Path(work_dir), arguments.pip_audit_version)
            )
    unignored = findings(reports, ignored)
    stale = stale_ignores(reports, ignored)
    for group in groups:
        print(f"audited extras: {_label(group)}")
    for name, version, identifier in unignored:
        print(f"advisory: {name} {version} {identifier}")
    for identifier in stale:
        print(f"stale ignore (no advisory matches it): {identifier}")
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with Path(summary_path).open("a", encoding="utf-8") as summary:
            summary.write(_summary(groups, unignored, stale))
    return 1 if unignored or stale else 0


if __name__ == "__main__":
    sys.exit(main())
