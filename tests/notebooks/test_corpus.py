"""``substrax.notebooks.corpus``: a repository's examples as a whole, and what a change reaches."""

from __future__ import annotations

from pathlib import Path

from _example_texts import git, write_example

from substrax.notebooks.corpus import corpus_problems, is_touched, reaches_every_example


def _examples(repo: Path) -> frozenset[Path]:
    return frozenset(path.resolve() for path in (repo / "examples").rglob("*.py"))


def test_a_repository_with_no_example_is_one_problem(tmp_path: Path) -> None:
    repo, _ = write_example(tmp_path)

    assert corpus_problems(repo, frozenset()) == [f"no example under {repo / 'examples'}"]


def test_a_page_no_example_makes_is_a_problem(tmp_path: Path) -> None:
    repo, _ = write_example(tmp_path)
    (repo / "docs" / "examples").mkdir(parents=True)
    (repo / "docs" / "examples" / "renamed.md").write_text("# Renamed\n", encoding="utf-8")

    problems = corpus_problems(repo, _examples(repo))

    assert len(problems) == 1
    assert problems[0].startswith("docs/examples/renamed.md: no example makes this page")


def test_a_hand_written_command_naming_no_file_is_a_problem(tmp_path: Path) -> None:
    repo, _ = write_example(tmp_path)
    (repo / "README.md").write_text("Run `uv run python examples/gone.py`.\n", encoding="utf-8")
    git(repo, "add", "README.md")

    assert corpus_problems(repo, _examples(repo)) == [
        "README.md: `uv run python examples/gone.py` names no file"
    ]


def test_a_document_deleted_and_not_staged_is_read_as_gone(tmp_path: Path) -> None:
    repo, _ = write_example(tmp_path)
    (repo / "NOTES.md").write_text("Run `uv run python examples/gone.py`.\n", encoding="utf-8")
    git(repo, "add", "NOTES.md")
    (repo / "NOTES.md").unlink()

    assert corpus_problems(repo, _examples(repo)) == []


def test_a_mkdocs_nav_that_lets_a_page_go_unlisted_is_a_problem(tmp_path: Path) -> None:
    repo, _ = write_example(tmp_path)
    (repo / "mkdocs.yml").write_text("site_name: demo\n", encoding="utf-8")

    problems = corpus_problems(repo, _examples(repo))

    assert len(problems) == 1
    assert problems[0].startswith("mkdocs.yml: validation.nav.omitted_files is None")


def test_a_change_to_an_examples_own_files_touches_it_alone(tmp_path: Path) -> None:
    repo, script = write_example(tmp_path)

    assert is_touched(script, {"examples/sorting.ipynb"}, repo)
    assert not is_touched(script, {"README.md"}, repo)
    assert not reaches_every_example({"examples/sorting.py"}, _examples(repo), repo)


def test_a_library_change_or_a_shared_file_reaches_every_example(tmp_path: Path) -> None:
    repo, _ = write_example(tmp_path)

    assert reaches_every_example({"src/library.py"}, _examples(repo), repo)
    assert reaches_every_example({"examples/_common/helpers.py"}, _examples(repo), repo)
