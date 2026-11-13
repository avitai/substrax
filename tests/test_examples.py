"""``substrax.examples``: a repository's example scripts, and each one's path from the root."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from substrax.examples import discover_examples, example_source, ExamplePathError


def test_discovery_skips_private_parts_and_package_markers(tmp_path: Path) -> None:
    root = tmp_path / "_site" / "examples"
    for relative in (
        "a/01_first.py",
        "a/__init__.py",
        "_templates/template.py",
        "_private.py",
        "b/helpers/_shared.py",
        "b/02_second.py",
        "b/notes.txt",
    ):
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text("", encoding="utf-8")

    found = discover_examples(root)

    assert [path.relative_to(root).as_posix() for path in found] == [
        "a/01_first.py",
        "b/02_second.py",
    ]


def _repository(tmp_path: Path) -> Path:
    """A repository holding ``examples/topic/demo.py`` and ``scripts/tool.py``."""
    repo = tmp_path / "repo"
    for relative in ("examples/topic/demo.py", "scripts/tool.py"):
        (repo / relative).parent.mkdir(parents=True, exist_ok=True)
        (repo / relative).write_text("", encoding="utf-8")
    return repo


def test_an_example_s_source_is_its_path_from_the_root(tmp_path: Path) -> None:
    repo = _repository(tmp_path)

    source = example_source(repo / "examples/topic/demo.py", repo_root=repo)

    assert source == "examples/topic/demo.py"


def test_a_link_above_the_root_gives_the_same_source(tmp_path: Path) -> None:
    """One filesystem mounted at two paths: the root's own path may hold a link."""
    repo = _repository(tmp_path)
    alias = tmp_path / "alias"
    alias.symlink_to(repo, target_is_directory=True)
    script = "examples/topic/demo.py"

    assert example_source(alias / script, repo_root=repo) == script
    assert example_source(repo / script, repo_root=alias) == script
    assert example_source(alias / script, repo_root=alias) == script


def test_a_relative_script_path_is_read_from_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repository(tmp_path)
    monkeypatch.chdir(repo / "examples")

    assert example_source(Path("topic/demo.py"), repo_root=repo) == "examples/topic/demo.py"


def _linked_file(repo: Path) -> Path:
    link = repo / "examples" / "topic" / "second_name.py"
    link.symlink_to(repo / "examples" / "topic" / "demo.py")
    return link


def _linked_directory(repo: Path) -> Path:
    (repo / "examples" / "linked").symlink_to(repo / "examples" / "topic", target_is_directory=True)
    return repo / "examples" / "linked" / "demo.py"


def _linked_examples_directory(repo: Path) -> Path:
    elsewhere = repo.parent / "elsewhere"
    (repo / "examples").rename(elsewhere)
    (repo / "examples").symlink_to(elsewhere, target_is_directory=True)
    return repo / "examples" / "topic" / "demo.py"


@pytest.mark.parametrize(
    "link",
    [_linked_file, _linked_directory, _linked_examples_directory],
    ids=["a-linked-file", "a-linked-directory", "a-linked-examples-directory"],
)
def test_an_example_reached_through_a_link_below_the_root_is_refused(
    tmp_path: Path, link: Callable[[Path], Path]
) -> None:
    """A link would give an example a second name, or put it outside the repository."""
    repo = _repository(tmp_path)
    script = link(repo)

    with pytest.raises(ExamplePathError, match="reached through a symlink"):
        example_source(script, repo_root=repo)


def test_a_file_outside_the_examples_directory_is_refused(tmp_path: Path) -> None:
    repo = _repository(tmp_path)

    with pytest.raises(ExamplePathError, match="not an example under"):
        example_source(repo / "scripts" / "tool.py", repo_root=repo)


def test_a_file_outside_the_repository_is_refused(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    outside = tmp_path / "outside.py"
    outside.write_text("", encoding="utf-8")

    with pytest.raises(ExamplePathError, match="not an example under"):
        example_source(outside, repo_root=repo)
