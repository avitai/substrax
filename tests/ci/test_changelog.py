"""The changelog's headings link to their changes, and its newest release is the package's version.

Each ``## [X.Y.Z]`` heading is a link reference, as Keep a Changelog writes them: a release
compares its tag with the one before it, the first release links to its tag, and ``[Unreleased]``
compares the newest tag with ``HEAD``. A heading with no definition renders as plain text, and a
definition left behind by a release links to the wrong range.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CHANGELOG = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
REPOSITORY = "https://github.com/avitai/substrax"
HEADINGS = re.findall(r"^## \[([^\]]+)\]", CHANGELOG, re.MULTILINE)
DEFINITIONS = dict(re.findall(r"^\[([^\]]+)\]: (\S+)$", CHANGELOG, re.MULTILINE))
RELEASES = HEADINGS[1:]


def test_the_changelog_opens_with_unreleased_then_releases_newest_first() -> None:
    def key(version: str) -> tuple[int, ...]:
        return tuple(int(part) for part in version.split("."))

    assert HEADINGS[0] == "Unreleased"
    assert RELEASES, "the changelog holds at least one release"
    assert sorted(RELEASES, key=key, reverse=True) == RELEASES


def test_every_heading_has_exactly_the_link_its_range_needs() -> None:
    previous = [*RELEASES[1:], None]
    expected = {"Unreleased": f"{REPOSITORY}/compare/v{RELEASES[0]}...HEAD"} | {
        release: f"{REPOSITORY}/compare/v{before}...v{release}"
        if before
        else f"{REPOSITORY}/releases/tag/v{release}"
        for release, before in zip(RELEASES, previous, strict=True)
    }

    assert expected == DEFINITIONS


def test_the_newest_release_is_the_version_pyproject_declares() -> None:
    """A release commit moves both together; between releases both name the last one."""
    with (ROOT / "pyproject.toml").open("rb") as pyproject:
        version = tomllib.load(pyproject)["project"]["version"]

    assert RELEASES[0] == version
