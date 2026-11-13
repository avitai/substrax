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

    assert problems == [
        "docs/examples/renamed.md: docs/examples holds generated pages and their figures only, "
        "and no example makes this file; an examples overview lives at docs/examples.md"
    ]


def _docs_examples_file(repo: Path, relative: str) -> None:
    (repo / "docs" / "examples" / relative).parent.mkdir(parents=True, exist_ok=True)
    (repo / "docs" / "examples" / relative).write_text("x\n", encoding="utf-8")


def test_any_file_under_docs_examples_no_example_makes_is_a_problem(tmp_path: Path) -> None:
    """Not only pages: an image, a script or a template there is no generated page either (C-7)."""
    repo, _ = write_example(tmp_path)
    for relative in ("diagram.png", "helper.py", "templates/example_template.py"):
        _docs_examples_file(repo, relative)

    problems = corpus_problems(repo, _examples(repo))

    assert [problem.split(":")[0] for problem in problems] == [
        "docs/examples/diagram.png",
        "docs/examples/helper.py",
        "docs/examples/templates/example_template.py",
    ]


def test_an_example_s_page_and_its_figures_are_not_a_problem(tmp_path: Path) -> None:
    repo, _ = write_example(tmp_path)
    for relative in ("sorting.md", "sorting_files/sorting_1_0.png"):
        _docs_examples_file(repo, relative)

    assert corpus_problems(repo, _examples(repo)) == []


def test_a_notebook_with_no_example_script_is_a_problem(tmp_path: Path) -> None:
    """A notebook left behind when its script was renamed or removed (A-6)."""
    repo, script = write_example(tmp_path)
    script.with_suffix(".ipynb").rename(repo / "examples" / "old_name.ipynb")
    (repo / "examples" / "_drafts").mkdir()
    (repo / "examples" / "_drafts" / "private.ipynb").write_text("{}", encoding="utf-8")

    assert corpus_problems(repo, _examples(repo)) == [
        "examples/old_name.ipynb: no example script makes this notebook; convert writes a "
        "notebook from its script, so remove it or restore examples/old_name.py"
    ]


def test_a_hand_written_command_naming_no_file_is_a_problem(tmp_path: Path) -> None:
    repo, _ = write_example(tmp_path)
    (repo / "README.md").write_text("Run `uv run python examples/gone.py`.\n", encoding="utf-8")
    git(repo, "add", "README.md")

    assert corpus_problems(repo, _examples(repo)) == [
        "README.md: `uv run python examples/gone.py` names no file"
    ]


def test_a_bare_python_command_naming_no_file_is_a_problem(tmp_path: Path) -> None:
    """A command a reader types without ``uv run`` names the file just as well (A-7)."""
    repo, _ = write_example(tmp_path)
    (repo / "README.md").write_text("Run `python examples/gone.py`.\n", encoding="utf-8")
    git(repo, "add", "README.md")

    assert corpus_problems(repo, _examples(repo)) == [
        "README.md: `python examples/gone.py` names no file"
    ]


def test_a_placeholder_path_is_no_command(tmp_path: Path) -> None:
    """A guide's ``examples/<topic>/<name>.py`` names a shape, not a file (T-9)."""
    repo, _ = write_example(tmp_path)
    (repo / "README.md").write_text(
        "Run `uv run python examples/<topic>/<name>.py` or `python examples/{name}.py`.\n",
        encoding="utf-8",
    )
    git(repo, "add", "README.md")

    assert corpus_problems(repo, _examples(repo)) == []


def test_a_document_that_is_not_utf8_is_a_problem_not_a_crash(tmp_path: Path) -> None:
    repo, _ = write_example(tmp_path)
    (repo / "README.md").write_bytes(b"# Caf\xe9\n")
    (repo / "mkdocs.yml").write_bytes(b"site_name: caf\xe9\n")
    git(repo, "add", "README.md", "mkdocs.yml")

    assert corpus_problems(repo, _examples(repo)) == [
        "README.md is not UTF-8: byte 5 cannot be decoded; save it as UTF-8",
        "mkdocs.yml is not UTF-8: byte 14 cannot be decoded; save it as UTF-8",
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
