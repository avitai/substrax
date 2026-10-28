"""The release checklist works under main's ruleset: every change to main arrives by pull request.

main accepts only a rebase-merged pull request whose `required` check passed, with no bypass. The
release commit therefore reaches main through a pull request, and a rebase merge gives it a new
SHA, so the tag is made on main after a fast-forward pull, and only the tag is pushed.
"""

from __future__ import annotations

import re
from pathlib import Path


RELEASING = (Path(__file__).resolve().parents[2] / "RELEASING.md").read_text(encoding="utf-8")


def test_the_release_commit_reaches_main_through_a_pull_request() -> None:
    # The checklist names the action, not merely the words: "pull request" also appears in the
    # macOS step's note that no pull request runs on macOS.
    assert re.search(r"open a pull request", RELEASING, re.IGNORECASE)
    assert "--rebase" in RELEASING, "main accepts rebase merges only"


def test_the_tag_names_the_commit_main_holds_after_the_merge() -> None:
    tag = RELEASING.index("git tag -a")

    assert RELEASING.index("git switch main") < tag
    assert RELEASING.index("git pull --ff-only") < tag


def test_only_the_tag_is_pushed() -> None:
    pushes = re.findall(r"^\s*git push .*$", RELEASING, re.MULTILINE)

    assert pushes, "the checklist pushes the tag"
    assert all(re.fullmatch(r"\s*git push origin vX\.Y\.Z", push) for push in pushes), pushes
