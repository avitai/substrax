"""``substrax.notebooks.repository``: the git queries the example tools make."""

from __future__ import annotations

from pathlib import Path

import pytest
from _example_texts import git, write_example

from substrax.notebooks import repository


def test_unclean_files_name_each_changed_removed_and_untracked_file(tmp_path: Path) -> None:
    repo, script = write_example(tmp_path)
    assert repository.unclean_files(repo) == set()
    (repo / "pyproject.toml").write_text("changed\n", encoding="utf-8")
    script.with_suffix(".ipynb").unlink()
    (repo / "scratch" / "deeper").mkdir(parents=True)
    (repo / "scratch" / "deeper" / "new.txt").write_text("new\n", encoding="utf-8")
    (repo / "__pycache__").mkdir()
    (repo / "__pycache__" / "ignored.pyc").write_bytes(b"")

    assert repository.unclean_files(repo) == {
        "pyproject.toml",
        "examples/sorting.ipynb",
        "scratch/deeper/new.txt",
    }


def test_changed_since_names_edited_and_staged_paths(tmp_path: Path) -> None:
    repo, script = write_example(tmp_path)
    git(repo, "tag", "base")
    script.write_text(script.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    (repo / "added.py").write_text("VALUE = 1\n", encoding="utf-8")
    git(repo, "add", "added.py")

    assert repository.changed_since(repo, "base") == {"examples/sorting.py", "added.py"}


def test_changed_since_names_only_what_changed_since_the_merge_base(tmp_path: Path) -> None:
    """As a pull request's diff: what the ref gained after the branch point is no change here."""
    repo, _ = write_example(tmp_path)
    git(repo, "branch", "trunk")
    git(repo, "switch", "-q", "-c", "topic")
    (repo / "topic.py").write_text("TOPIC = 1\n", encoding="utf-8")
    git(repo, "add", "topic.py")
    git(repo, "commit", "-q", "-m", "topic")
    git(repo, "switch", "-q", "trunk")
    (repo / "trunk_only.py").write_text("TRUNK = 1\n", encoding="utf-8")
    git(repo, "add", "trunk_only.py")
    git(repo, "commit", "-q", "-m", "trunk moves on")
    git(repo, "switch", "-q", "topic")

    assert repository.changed_since(repo, "trunk") == {"topic.py"}


def test_changed_since_refuses_a_ref_sharing_no_history_with_head(tmp_path: Path) -> None:
    repo, _ = write_example(tmp_path)
    git(repo, "switch", "-q", "-c", "topic")
    git(repo, "switch", "-q", "--orphan", "elsewhere")
    git(repo, "commit", "-q", "--allow-empty", "-m", "unrelated")
    git(repo, "switch", "-q", "topic")

    with pytest.raises(repository.RepositoryError, match="no history in common"):
        repository.changed_since(repo, "elsewhere")


@pytest.mark.parametrize(
    "ref",
    [
        pytest.param("--output=written.txt", id="an-option-that-writes-a-file"),
        pytest.param("--cached", id="an-option-that-narrows-the-diff"),
        pytest.param("no-such-ref", id="a-missing-ref"),
        pytest.param("HEAD^{tree}", id="a-tree"),
    ],
)
def test_changed_since_refuses_what_names_no_commit_and_never_reads_it_as_an_option(
    tmp_path: Path, ref: str
) -> None:
    repo, _ = write_example(tmp_path)

    with pytest.raises(repository.RepositoryError, match="names no commit"):
        repository.changed_since(repo, ref)

    assert not (repo / "written.txt").exists()


def test_changed_since_names_a_projects_paths_when_it_sits_below_the_top_level(
    tmp_path: Path,
) -> None:
    """In a monorepo the paths are the project's own, and a change outside it is not one."""
    top = tmp_path / "top"
    project = top / "project"
    (project / "examples").mkdir(parents=True)
    (project / "examples" / "sorting.py").write_text("VALUE = 1\n", encoding="utf-8")
    (top / "elsewhere.py").write_text("VALUE = 1\n", encoding="utf-8")
    git(top, "init", "-q")
    git(top, "add", "-A")
    git(top, "commit", "-q", "-m", "base")
    git(top, "tag", "base")
    (project / "examples" / "sorting.py").write_text("VALUE = 2\n", encoding="utf-8")
    (top / "elsewhere.py").write_text("VALUE = 2\n", encoding="utf-8")

    assert repository.changed_since(project, "base") == {"examples/sorting.py"}


def test_changed_since_names_an_untracked_file_but_not_an_ignored_one(tmp_path: Path) -> None:
    repo, _ = write_example(tmp_path)
    git(repo, "tag", "base")
    (repo / "examples" / "new.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "__pycache__").mkdir()
    (repo / "__pycache__" / "ignored.pyc").write_bytes(b"")

    assert repository.changed_since(repo, "base") == {"examples/new.py"}


def test_tracked_lists_files_matching_a_pathspec(tmp_path: Path) -> None:
    repo, _ = write_example(tmp_path)
    (repo / "README.md").write_text("# Demo\n", encoding="utf-8")
    (repo / "untracked.md").write_text("# Not added\n", encoding="utf-8")
    git(repo, "add", "README.md")

    assert repository.tracked(repo, "*.md") == [repo / "README.md"]


def test_tracked_leaves_out_a_file_deleted_from_the_working_tree(tmp_path: Path) -> None:
    """The index still lists a file deleted and not staged; the working tree no longer holds it."""
    repo, _ = write_example(tmp_path)
    for name in ("README.md", "NOTES.md"):
        (repo / name).write_text(f"# {name}\n", encoding="utf-8")
    git(repo, "add", "README.md", "NOTES.md")
    (repo / "NOTES.md").unlink()

    assert repository.tracked(repo, "*.md") == [repo / "README.md"]


def test_a_directory_outside_git_is_a_repository_error(tmp_path: Path) -> None:
    with pytest.raises(repository.RepositoryError, match="ls-files"):
        repository.unclean_files(tmp_path)


def test_head_is_empty_before_the_first_commit_and_the_commit_after(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    git(empty, "init", "-q")
    repo, _ = write_example(tmp_path)

    assert repository.head(empty) == ""
    assert len(repository.head(repo)) == 40


def test_the_index_listing_changes_when_a_file_is_staged(tmp_path: Path) -> None:
    repo, _ = write_example(tmp_path)
    before = repository.index_listing(repo)
    (repo / "staged.txt").write_text("x\n", encoding="utf-8")
    git(repo, "add", "staged.txt")

    assert repository.index_listing(repo) != before
    assert "staged.txt" in repository.index_listing(repo)


def test_the_hooks_directory_follows_core_hooks_path(tmp_path: Path) -> None:
    repo, _ = write_example(tmp_path)
    assert repository.hooks_directory(repo) == repo / ".git" / "hooks"

    git(repo, "config", "core.hooksPath", "githooks")

    assert repository.hooks_directory(repo) == repo / "githooks"
