"""``substrax.notebooks.pages``: an example's docs page, made by running a clean copy of it."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import re
import shutil
import subprocess
import sys
import tempfile
from importlib import metadata
from pathlib import Path

import pytest
from _example_texts import CONFIG, GOOD, write_example

from substrax.notebooks.config import ExamplesConfig
from substrax.notebooks.pages import (
    check_page,
    compare_with_run,
    generate_page,
    page_path,
    PageError,
    read_provenance,
)
from substrax.notebooks.pairs import check_pair, convert
from substrax.notebooks.run import ExampleRun, execute_example, RunError, script_sha256


SCRIPT = '''# %% [markdown]
"""
# Demo

What the demo shows.
"""

# %%
import hashlib
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt

print("interpreter:", hashlib.sha256(sys.executable.encode()).hexdigest()[:16])
print("artifacts:", os.environ["AVITAI_OUTPUT_DIR"])
Path(os.environ["AVITAI_OUTPUT_DIR"], "artifact.txt").write_text("x", encoding="utf-8")

# %%
values = [1, 4, 9]
print("total:", sum(values))

# %%
plt.plot(values)
plt.show()
'''


def _kinds(script: Path, repo: Path) -> list[str]:
    return [problem.kind for problem in check_page(script, repo_root=repo, config=CONFIG)]


def _repo(tmp_path: Path, text: str = SCRIPT) -> tuple[Path, Path]:
    return write_example(tmp_path, text, relative="topic/demo.py")


def test_the_page_mirrors_the_example_path(tmp_path: Path) -> None:
    repo, script = _repo(tmp_path)

    assert page_path(script, repo_root=repo) == repo / "docs" / "examples" / "topic" / "demo.md"


def test_the_page_carries_the_code_its_printed_output_and_its_figure(tmp_path: Path) -> None:
    repo, script = _repo(tmp_path)

    page = generate_page(script, repo_root=repo, config=CONFIG)

    text = page.read_text(encoding="utf-8")
    assert page == repo / "docs" / "examples" / "topic" / "demo.md"
    assert "values = [1, 4, 9]" in text
    assert "total: 14" in text
    figures = re.findall(r"!\[[^\]]*\]\(([^)]+\.png)\)", text)
    assert len(figures) == 1
    assert (page.parent / figures[0]).is_file()


def test_the_page_records_where_and_how_long_it_ran(tmp_path: Path) -> None:
    repo, script = _repo(tmp_path)

    page = generate_page(script, repo_root=repo, config=CONFIG)

    provenance = read_provenance(page)
    assert provenance.source == "examples/topic/demo.py"
    assert provenance.sha256 == hashlib.sha256(script.read_bytes()).hexdigest()
    assert provenance.seconds > 0
    assert provenance.device
    assert "Ran in " in page.read_text(encoding="utf-8")


def test_the_header_names_the_version_of_the_repository_s_distribution(tmp_path: Path) -> None:
    repo, script = _repo(tmp_path)
    config = dataclasses.replace(CONFIG, distribution="nbformat")

    page = generate_page(script, repo_root=repo, config=config)

    assert f"nbformat {metadata.version('nbformat')}" in page.read_text(encoding="utf-8")


def test_the_page_opens_with_the_example_title(tmp_path: Path) -> None:
    # MkDocs takes a page's title from an h1 only when it is the page's first element.
    repo, script = _repo(tmp_path)

    text = generate_page(script, repo_root=repo, config=CONFIG).read_text(encoding="utf-8")

    assert text.lstrip().splitlines()[0] == "# Demo"


def test_the_header_after_the_title_gives_the_run_command(tmp_path: Path) -> None:
    repo, script = _repo(tmp_path)

    text = generate_page(script, repo_root=repo, config=CONFIG).read_text(encoding="utf-8")

    command = "```bash\nsource ./activate.sh\nuv run python examples/topic/demo.py\n```"
    assert text.count(command) == 1
    assert text.index("# Demo") < text.index("Ran in ") < text.index(command)
    assert text.index(command) < text.index("import hashlib")


def test_the_header_links_the_script_and_its_notebook(tmp_path: Path) -> None:
    repo, script = _repo(tmp_path)

    text = generate_page(script, repo_root=repo, config=CONFIG).read_text(encoding="utf-8")

    base = f"{CONFIG.source_url}/examples/topic/demo"
    assert f"[`examples/topic/demo.py`]({base}.py)" in text
    assert f"[notebook]({base}.ipynb)" in text


def test_the_run_uses_this_interpreter_and_a_temporary_artifact_directory(tmp_path: Path) -> None:
    repo, script = _repo(tmp_path)

    text = generate_page(script, repo_root=repo, config=CONFIG).read_text(encoding="utf-8")

    assert f"interpreter: {hashlib.sha256(sys.executable.encode()).hexdigest()[:16]}" in text
    artifacts = re.search(r"artifacts: (\S+)", text)
    assert artifacts is not None
    assert not Path(artifacts.group(1)).is_relative_to(repo)
    assert not (repo / "artifact.txt").exists()


def test_generating_the_page_leaves_the_committed_pair_clean(tmp_path: Path) -> None:
    repo, script = _repo(tmp_path)
    before = script.with_suffix(".ipynb").read_bytes()

    generate_page(script, repo_root=repo, config=CONFIG)

    assert script.with_suffix(".ipynb").read_bytes() == before
    assert check_pair(script) == []


def test_a_failing_cell_names_the_cell_and_writes_nothing(tmp_path: Path) -> None:
    repo, script = _repo(tmp_path, SCRIPT + '\n# %%\nraise RuntimeError("broken step")\n')

    with pytest.raises(RunError, match="broken step"):
        generate_page(script, repo_root=repo, config=CONFIG)

    assert not page_path(script, repo_root=repo).exists()


def test_an_output_holding_a_local_path_is_refused(tmp_path: Path) -> None:
    repo, script = _repo(tmp_path, '# %%\nprint("/home/someone/private/data.csv")\n')

    with pytest.raises(PageError, match="local path"):
        generate_page(script, repo_root=repo, config=CONFIG)

    assert not page_path(script, repo_root=repo).exists()


def test_a_current_page_checks_clean(tmp_path: Path) -> None:
    repo, script = _repo(tmp_path)
    generate_page(script, repo_root=repo, config=CONFIG)

    assert check_page(script, repo_root=repo, config=CONFIG) == []


def test_a_missing_page_is_a_problem(tmp_path: Path) -> None:
    repo, script = _repo(tmp_path)

    assert _kinds(script, repo) == ["missing"]


def test_a_page_from_an_older_script_is_stale(tmp_path: Path) -> None:
    repo, script = _repo(tmp_path)
    generate_page(script, repo_root=repo, config=CONFIG)
    script.write_text(SCRIPT.replace("[1, 4, 9]", "[1, 2, 3]"), encoding="utf-8")

    assert _kinds(script, repo) == ["stale"]


def test_a_page_whose_figure_is_gone_is_a_problem(tmp_path: Path) -> None:
    repo, script = _repo(tmp_path)
    page = generate_page(script, repo_root=repo, config=CONFIG)
    for figure in page.parent.rglob("*.png"):
        figure.unlink()

    assert _kinds(script, repo) == ["missing-figure"]


def test_a_page_holding_a_local_path_is_a_problem(tmp_path: Path) -> None:
    repo, script = _repo(tmp_path)
    page = generate_page(script, repo_root=repo, config=CONFIG)
    page.write_text(page.read_text(encoding="utf-8") + "\n/Users/someone/notes\n", encoding="utf-8")

    # A line added by hand is an edit, and the path it holds is named too.
    kinds = _kinds(script, repo)
    assert kinds == ["edited", "local-path"]


def test_an_output_edited_by_hand_is_a_problem(tmp_path: Path) -> None:
    repo, script = _repo(tmp_path)
    page = generate_page(script, repo_root=repo, config=CONFIG)
    text = page.read_text(encoding="utf-8")
    assert "total: 14" in text
    page.write_text(text.replace("total: 14", "total: 999999"), encoding="utf-8")

    assert _kinds(script, repo) == ["edited"]


def test_a_figure_replaced_by_hand_is_a_problem(tmp_path: Path) -> None:
    repo, script = _repo(tmp_path)
    page = generate_page(script, repo_root=repo, config=CONFIG)
    (figure,) = page.parent.rglob("*.png")
    figure.write_bytes(figure.read_bytes() + b"\0")

    problems = check_page(script, repo_root=repo, config=CONFIG)

    assert [problem.kind for problem in problems] == ["edited"]
    assert figure.name in problems[0].detail


@pytest.mark.parametrize("ending", [b"\r\n", b"\r"], ids=["windows", "carriage-return"])
def test_a_page_with_other_line_endings_checks_clean(tmp_path: Path, ending: bytes) -> None:
    repo, script = _repo(tmp_path)
    page = generate_page(script, repo_root=repo, config=CONFIG)
    page.write_bytes(page.read_bytes().replace(b"\n", ending))

    assert check_page(script, repo_root=repo, config=CONFIG) == []


def test_a_page_that_is_not_utf8_is_unreadable_not_a_crash(tmp_path: Path) -> None:
    repo, script = _repo(tmp_path)
    page = generate_page(script, repo_root=repo, config=CONFIG)
    page.write_bytes(b"# Caf\xe9\n" + page.read_bytes())

    problems = check_page(script, repo_root=repo, config=CONFIG)

    assert [problem.kind for problem in problems] == ["unreadable"]
    assert problems[0].detail == (
        f"{page} is not UTF-8: byte 5 cannot be decoded; save it as UTF-8"
    )
    with pytest.raises(ValueError, match="is not UTF-8"):
        read_provenance(page)


def test_a_page_whose_script_is_not_utf8_is_unreadable_not_a_crash(tmp_path: Path) -> None:
    repo, script = _repo(tmp_path)
    generate_page(script, repo_root=repo, config=CONFIG)
    script.write_bytes(b"# caf\xe9\n" + script.read_bytes())

    problems = check_page(script, repo_root=repo, config=CONFIG)

    assert [problem.kind for problem in problems] == ["unreadable"]
    assert problems[0].detail == (
        f"{script} is not UTF-8: byte 5 cannot be decoded; save it as UTF-8"
    )


def test_a_page_whose_provenance_records_no_page_digest_is_unreadable(tmp_path: Path) -> None:
    repo, script = _repo(tmp_path)
    page = generate_page(script, repo_root=repo, config=CONFIG)
    text = page.read_text(encoding="utf-8")
    raw = re.findall(r"<!-- substrax-notebooks: (\{.*?\}) -->", text)[-1]
    record = json.loads(raw)
    del record["page"]
    page.write_text(text.replace(raw, json.dumps(record, sort_keys=True)), encoding="utf-8")

    assert _kinds(script, repo) == ["unreadable"]


def _run(script: Path, repo: Path) -> ExampleRun:
    return execute_example(
        script,
        repo_root=repo,
        distribution=CONFIG.distribution,
        budget_seconds=CONFIG.budget_seconds,
    )


def test_a_run_matching_the_page_is_clean(tmp_path: Path) -> None:
    repo, script = _repo(tmp_path)
    generate_page(script, repo_root=repo, config=CONFIG)

    assert compare_with_run(script, _run(script, repo), repo_root=repo, config=CONFIG) == []


_DRAWS_WHEN_ASKED = """# %%
import os

import matplotlib.pyplot as plt

print("values")
if os.environ.get("SUBSTRAX_TEST_DRAW"):
    plt.plot([1, 2])
    plt.show()
"""


def test_a_run_with_a_different_output_structure_differs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The same script, run where it draws a figure its page was made without.
    repo, script = _repo(tmp_path, _DRAWS_WHEN_ASKED)
    generate_page(script, repo_root=repo, config=CONFIG)
    monkeypatch.setenv("SUBSTRAX_TEST_DRAW", "1")

    problems = compare_with_run(script, _run(script, repo), repo_root=repo, config=CONFIG)

    assert [problem.kind for problem in problems] == ["differs-from-run"]


def test_known_roots_in_outputs_are_replaced_before_publishing(tmp_path: Path) -> None:
    repo, script = _repo(
        tmp_path,
        "# %%\nimport sys\nfrom pathlib import Path\n"
        'print(Path.cwd() / "examples" / "topic" / "demo.py")\n'
        "print(sys.prefix)\n",
    )

    text = generate_page(script, repo_root=repo, config=CONFIG).read_text(encoding="utf-8")

    assert str(repo) not in text
    assert "./examples/topic/demo.py" in text
    assert sys.prefix not in text
    assert "<env>" in text


def test_code_on_the_page_is_fenced_as_python(tmp_path: Path) -> None:
    repo, script = _repo(tmp_path)

    text = generate_page(script, repo_root=repo, config=CONFIG).read_text(encoding="utf-8")

    fences = re.findall(r"^```(\w*)$", text, flags=re.MULTILINE)
    assert "python" in fences
    assert "values = [1, 4, 9]" in text.split("```python", 1)[1]


@pytest.mark.skipif(sys.platform == "win32", reason="IPC transport needs Unix sockets")
def test_a_run_leaves_nothing_in_the_tree_but_the_page(tmp_path: Path) -> None:
    repo, script = _repo(tmp_path)
    before = {path for path in repo.rglob("*") if path.is_file() and ".git" not in path.parts}

    page = generate_page(script, repo_root=repo, config=CONFIG)

    after = {path for path in repo.rglob("*") if path.is_file() and ".git" not in path.parts}
    new = after - before
    assert page in new
    figures = {path for path in new if path.suffix == ".png" and path.is_relative_to(page.parent)}
    assert new == {page, *figures}


def test_a_page_without_provenance_is_unreadable(tmp_path: Path) -> None:
    repo, script = _repo(tmp_path)
    page = page_path(script, repo_root=repo)
    page.parent.mkdir(parents=True)
    page.write_text("# A page written by hand\n", encoding="utf-8")

    problems = check_page(script, repo_root=repo, config=CONFIG)

    assert [problem.kind for problem in problems] == ["unreadable"]
    assert str(script) in str(problems[0])


def test_comparing_without_a_page_reports_the_missing_page(tmp_path: Path) -> None:
    repo, script = _repo(tmp_path)

    problems = compare_with_run(script, _run(script, repo), repo_root=repo, config=CONFIG)

    assert [problem.kind for problem in problems] == ["missing"]


def test_a_root_is_replaced_in_the_outputs_only_where_it_starts_a_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    starts = f"{scratch}/archive.tar"
    inside = (f"https://data.example.org{scratch}/archive.tar", f"{scratch}_demo/notes.txt")
    repo, script = _repo(
        tmp_path, "# %%\n" + "".join(f"print({line!r})\n" for line in (starts, *inside))
    )

    text = generate_page(script, repo_root=repo, config=CONFIG).read_text(encoding="utf-8")

    assert "<tmp>/archive.tar" in text
    assert f"print({starts!r})" in text  # the code is published as the script has it
    assert [text.count(line) for line in inside] == [2, 2]  # in the code and in the output


def test_a_root_that_is_the_filesystem_root_is_never_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", "/")
    repo, script = _repo(tmp_path, '# %%\nprint("3/4 =", 3 / 4)\n')

    text = generate_page(script, repo_root=repo, config=CONFIG).read_text(encoding="utf-8")

    assert "~" not in text
    assert "3/4 = 0.75" in text
    assert f"({CONFIG.source_url}/examples/topic/demo.py)" in text


def test_a_page_for_a_name_with_a_space_checks_clean_and_links_its_script(tmp_path: Path) -> None:
    repo, script = write_example(tmp_path, SCRIPT, relative="topic/my demo.py")

    text = generate_page(script, repo_root=repo, config=CONFIG).read_text(encoding="utf-8")

    assert check_page(script, repo_root=repo, config=CONFIG) == []
    assert f"({CONFIG.source_url}/examples/topic/my%20demo.py)" in text
    assert f"({CONFIG.source_url}/examples/topic/my%20demo.ipynb)" in text


def test_a_remote_image_is_not_a_missing_figure(tmp_path: Path) -> None:
    repo, script = _repo(
        tmp_path,
        "# %%\nfrom IPython.display import Markdown, display\n\n"
        'display(Markdown("![logo](https://example.org/logo.png)"))\n',
    )
    generate_page(script, repo_root=repo, config=CONFIG)

    assert check_page(script, repo_root=repo, config=CONFIG) == []


def test_regenerating_a_page_removes_figures_the_run_no_longer_draws(tmp_path: Path) -> None:
    repo, script = _repo(tmp_path)
    page = generate_page(script, repo_root=repo, config=CONFIG)
    assert list(page.parent.rglob("*.png"))
    script.write_text(SCRIPT.replace("plt.plot(values)\nplt.show()\n", "print(values)\n"))

    generate_page(script, repo_root=repo, config=CONFIG)

    assert list(page.parent.rglob("*.png")) == []
    assert check_page(script, repo_root=repo, config=CONFIG) == []


def test_a_figure_the_page_does_not_link_is_a_problem(tmp_path: Path) -> None:
    repo, script = _repo(tmp_path)
    page = generate_page(script, repo_root=repo, config=CONFIG)
    (figure,) = page.parent.rglob("*.png")
    (figure.parent / "left_over.png").write_bytes(figure.read_bytes())

    problems = check_page(script, repo_root=repo, config=CONFIG)

    assert [problem.kind for problem in problems] == ["stray-figure"]
    assert "left_over.png" in problems[0].detail


_TITLE_THEN_METADATA = '''# %% [markdown]
"""
# Demo

| Metadata | Value |
| --- | --- |
| **Tier** | Quick Reference |

## Overview

What the demo shows.
"""

# %%
print("ran")
'''


def test_the_header_sits_right_after_the_title_line_of_a_longer_first_cell(
    tmp_path: Path,
) -> None:
    repo, script = _repo(tmp_path, _TITLE_THEN_METADATA)

    text = generate_page(script, repo_root=repo, config=CONFIG).read_text(encoding="utf-8")

    assert text.lstrip().splitlines()[0] == "# Demo"
    assert text.index("# Demo") < text.index("Ran in ") < text.index("| Metadata | Value |")
    assert text.index("| Metadata | Value |") < text.index("## Overview")


def test_terminal_colours_printed_by_the_run_are_not_published(tmp_path: Path) -> None:
    repo, script = _repo(tmp_path, '# %%\nprint("\\033[31mred\\033[0m text")\n')

    text = generate_page(script, repo_root=repo, config=CONFIG).read_text(encoding="utf-8")

    assert "\x1b" not in text
    assert "red text" in text


def _warning_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if "UserWarning: careful" in line]


def test_a_warning_names_the_same_cell_file_on_every_regeneration(tmp_path: Path) -> None:
    repo, script = _repo(tmp_path, '# %%\nimport warnings\n\nwarnings.warn("careful")\n')

    first = generate_page(script, repo_root=repo, config=CONFIG).read_text(encoding="utf-8")
    second = generate_page(script, repo_root=repo, config=CONFIG).read_text(encoding="utf-8")

    assert _warning_lines(first)
    assert _warning_lines(first) == _warning_lines(second)
    assert not re.search(r"ipykernel_\d", first)


_SMALL_PAGES = dataclasses.replace(CONFIG, page_size_warn_bytes=4_000, page_size_bytes=8_000)
"""Limits small enough that a few thousand printed characters cross them."""


def _printing(characters: int) -> str:
    return f'# %%\nprint("x" * {characters})\n'


def test_a_page_over_the_size_limit_is_refused_and_nothing_is_written(tmp_path: Path) -> None:
    repo, script = _repo(tmp_path, _printing(10_000))

    with pytest.raises(PageError, match=r"8000 bytes"):
        generate_page(script, repo_root=repo, config=_SMALL_PAGES)

    assert not page_path(script, repo_root=repo).exists()


def test_a_page_over_the_warning_size_is_written_with_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    repo, script = _repo(tmp_path, _printing(5_000))

    with caplog.at_level(logging.WARNING, logger="substrax.notebooks.pages"):
        page = generate_page(script, repo_root=repo, config=_SMALL_PAGES)

    assert page.exists()
    assert [record.levelno for record in caplog.records] == [logging.WARNING]
    assert "4000 bytes" in caplog.records[0].getMessage()


def test_an_exempt_example_is_not_held_to_the_size_limits(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    repo, script = _repo(tmp_path, _printing(10_000))
    exempt = dataclasses.replace(_SMALL_PAGES, page_size_exempt=("examples/topic/demo.py",))

    with caplog.at_level(logging.WARNING, logger="substrax.notebooks.pages"):
        generate_page(script, repo_root=repo, config=exempt)

    assert check_page(script, repo_root=repo, config=exempt) == []
    assert caplog.records == []


def test_a_committed_page_over_the_size_limit_is_a_problem(tmp_path: Path) -> None:
    repo, script = _repo(tmp_path, _printing(10_000))
    generate_page(script, repo_root=repo, config=CONFIG)

    problems = check_page(script, repo_root=repo, config=_SMALL_PAGES)

    assert [problem.kind for problem in problems] == ["too-large"]
    assert "8000 bytes" in problems[0].detail


def test_a_page_recording_a_run_longer_than_its_example_s_budget_is_a_problem(
    tmp_path: Path,
) -> None:
    """A budget lowered below the run the page records is found without running the example."""
    repo, script = _repo(tmp_path)
    page = generate_page(script, repo_root=repo, config=CONFIG)
    seconds = read_provenance(page).seconds
    tight = dataclasses.replace(CONFIG, budgets={"examples/topic/demo.py": seconds / 2})

    problems = check_page(script, repo_root=repo, config=tight)

    assert [problem.kind for problem in problems] == ["over-budget"]
    assert f"{seconds / 2} s budget" in problems[0].detail


def test_a_page_within_its_example_s_own_budget_checks_clean(tmp_path: Path) -> None:
    repo, script = _repo(tmp_path)
    page = generate_page(script, repo_root=repo, config=CONFIG)
    seconds = read_provenance(page).seconds
    other = dataclasses.replace(CONFIG, budgets={"examples/topic/demo.py": seconds * 2})

    assert check_page(script, repo_root=repo, config=other) == []


def test_a_page_is_made_within_its_example_s_own_budget(tmp_path: Path) -> None:
    """The example's entry, not the repository default, bounds the run that makes its page."""
    # Each cell stays inside the entry's budget; together they pass it.
    text = GOOD.replace(
        '# %% [markdown]\n"""\n## Results',
        "# %%\nimport time\n\ntime.sleep(1.5)\n\n# %%\ntime.sleep(1.5)\n\n\n"
        '# %% [markdown]\n"""\n## Results',
    )
    assert text != GOOD
    repo, script = write_example(tmp_path, text)
    entry = dataclasses.replace(CONFIG, budgets={"examples/sorting.py": 2.5})
    assert entry.budget_seconds > 3

    with pytest.raises(RunError, match=r"budget of 2\.5 s"):
        generate_page(script, repo_root=repo, config=entry)
    assert not page_path(script, repo_root=repo).exists()


def _rewrite_record(path: Path, **fields: object) -> None:
    text = path.read_text(encoding="utf-8")
    record = re.findall(r"<!-- substrax-notebooks: (\{.*?\}) -->", text)[-1]
    edited = json.dumps({**json.loads(record), **fields}, sort_keys=True)
    path.write_text(text.replace(record, edited), encoding="utf-8")


def _edit_provenance(path: Path, **fields: object) -> None:
    """Set fields of the record's provenance, as a person or a tool editing the page might."""
    text = path.read_text(encoding="utf-8")
    raw = re.findall(r"<!-- substrax-notebooks: (\{.*?\}) -->", text)[-1]
    record = json.loads(raw)
    record["provenance"].update(fields)
    path.write_text(text.replace(raw, json.dumps(record, sort_keys=True)), encoding="utf-8")


@pytest.mark.parametrize(
    "fields",
    [
        pytest.param({"sha256": 5}, id="a-number-for-the-hash"),
        pytest.param({"seconds": "abc"}, id="text-for-the-seconds"),
        pytest.param({"device": ["x"]}, id="a-list-for-the-device"),
        pytest.param({"outputs": "not a list"}, id="text-for-the-outputs"),
    ],
)
def test_a_record_holding_a_field_of_the_wrong_type_is_unreadable(
    tmp_path: Path, fields: dict[str, object]
) -> None:
    repo, script = _repo(tmp_path)
    page = generate_page(script, repo_root=repo, config=CONFIG)
    _edit_provenance(page, **fields)

    assert _kinds(script, repo) == ["unreadable"]


def test_a_record_holding_a_field_it_does_not_declare_is_a_value_error(tmp_path: Path) -> None:
    """``read_provenance`` documents one error for a record it cannot read."""
    repo, script = _repo(tmp_path)
    page = generate_page(script, repo_root=repo, config=CONFIG)
    _edit_provenance(page, extra=1)

    with pytest.raises(ValueError, match="extra"):
        read_provenance(page)


def test_a_record_edited_to_hide_a_stale_page_is_an_edit(tmp_path: Path) -> None:
    repo, script = _repo(tmp_path)
    page = generate_page(script, repo_root=repo, config=CONFIG)
    script.write_text(SCRIPT.replace("[1, 4, 9]", "[1, 2, 3]"), encoding="utf-8")
    _edit_provenance(page, sha256=script_sha256(script))

    problems = check_page(script, repo_root=repo, config=CONFIG)

    assert [problem.kind for problem in problems] == ["edited"]
    assert "provenance record" in problems[0].detail


def test_a_record_edited_to_exempt_a_large_page_is_an_edit(tmp_path: Path) -> None:
    repo, script = _repo(tmp_path, _printing(10_000))
    page = generate_page(script, repo_root=repo, config=CONFIG)
    exempt = dataclasses.replace(_SMALL_PAGES, page_size_exempt=("examples/topic/other.py",))
    _edit_provenance(page, source="examples/topic/other.py")

    assert _kinds_with(script, repo, exempt) == ["edited"]


def _kinds_with(script: Path, repo: Path, config: ExamplesConfig) -> list[str]:
    return [problem.kind for problem in check_page(script, repo_root=repo, config=config)]


def test_a_record_missing_a_digest_is_unreadable_not_an_error(tmp_path: Path) -> None:
    repo, script = _repo(tmp_path)
    page = generate_page(script, repo_root=repo, config=CONFIG)
    text = page.read_text(encoding="utf-8")
    record = json.loads(re.findall(r"<!-- substrax-notebooks: (\{.*?\}) -->", text)[-1])
    del record["page"]["body"]
    _rewrite_record(page, page=record["page"])

    assert _kinds(script, repo) == ["unreadable"]


def test_an_example_that_prints_the_record_marker_has_a_readable_page(tmp_path: Path) -> None:
    repo, script = _repo(
        tmp_path, '# %%\nprint(\'<!-- substrax-notebooks: {"sha256": "printed"} -->\')\n'
    )
    generate_page(script, repo_root=repo, config=CONFIG)

    assert _kinds(script, repo) == []


def test_a_root_at_the_end_of_a_sentence_is_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    repo, script = _repo(tmp_path, f'# %%\nprint("saved to {scratch}.")\n')

    text = generate_page(script, repo_root=repo, config=CONFIG).read_text(encoding="utf-8")

    assert "saved to <tmp>." in text


@pytest.mark.parametrize(
    "printed",
    [
        "'saved to /home/someone.'",
        # Built at run time, so only the output holds the path: repr doubles each backslash.
        "repr(chr(92).join(['C:', 'Users', 'someone', 'data.csv']))",
        "chr(92).join(['C:', 'Users', 'someone', 'data.csv'])",
    ],
)
def test_a_local_path_in_any_spelling_is_refused(tmp_path: Path, printed: str) -> None:
    repo, script = _repo(tmp_path, f"# %%\nprint({printed})\n")

    with pytest.raises(PageError, match="local path"):
        generate_page(script, repo_root=repo, config=CONFIG)

    assert not page_path(script, repo_root=repo).exists()


def test_a_url_holding_a_home_path_is_not_a_local_path(tmp_path: Path) -> None:
    repo, script = _repo(tmp_path, '# %%\nprint("see https://example.org/home/page")\n')

    page = generate_page(script, repo_root=repo, config=CONFIG)

    assert "https://example.org/home/page" in page.read_text(encoding="utf-8")


def _artifact_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if "artifacts:" in line and "print(" not in line]


def test_the_runs_artifact_directory_reads_the_same_on_every_regeneration(tmp_path: Path) -> None:
    repo, script = _repo(tmp_path)

    first = generate_page(script, repo_root=repo, config=CONFIG).read_text(encoding="utf-8")
    second = generate_page(script, repo_root=repo, config=CONFIG).read_text(encoding="utf-8")

    assert _artifact_lines(first) == _artifact_lines(second) == ["    artifacts: <outputs>"]


def test_an_examples_own_text_naming_ipykernel_is_published_as_printed(tmp_path: Path) -> None:
    repo, script = _repo(tmp_path, '# %%\nprint("ipykernel_12345 is a process name")\n')

    text = generate_page(script, repo_root=repo, config=CONFIG).read_text(encoding="utf-8")

    assert "    ipykernel_12345 is a process name" in text


def test_a_repository_reached_through_a_symlink_reads_as_through_its_real_path(
    tmp_path: Path,
) -> None:
    repo, script = _repo(tmp_path, '# %%\nprint("file:", __file__)\n')
    link = tmp_path / "link"
    link.symlink_to(repo, target_is_directory=True)

    page = generate_page(link / script.relative_to(repo), repo_root=link, config=CONFIG)

    assert "    file: ./examples/topic/demo.py" in page.read_text(encoding="utf-8")


def test_a_temporary_root_reached_through_a_symlink_reads_as_its_placeholder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (tmp_path / "scratch-link").symlink_to(scratch, target_is_directory=True)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "scratch-link"))  # macOS's /var
    repo, script = _repo(
        tmp_path,
        "# %%\nimport os\nfrom pathlib import Path\n\n"
        'resolved = Path(os.environ["AVITAI_OUTPUT_DIR"]).resolve()\n'
        'print("outputs:", resolved)\nprint("tmp:", resolved.parent.parent)\n',
    )

    text = generate_page(script, repo_root=repo, config=CONFIG).read_text(encoding="utf-8")

    assert "    outputs: <outputs>" in text
    assert "    tmp: <tmp>" in text


def test_a_hyperlink_an_example_prints_is_published_as_its_text(tmp_path: Path) -> None:
    link = r"\x1b]8;;https://example.org\x1b\\docs\x1b]8;;\x1b\\"
    repo, script = _repo(tmp_path, f'# %%\nprint("see {link} here")\n')

    text = generate_page(script, repo_root=repo, config=CONFIG).read_text(encoding="utf-8")

    assert "    see docs here" in text
    assert "\x1b" not in text


@pytest.mark.parametrize(
    ("cell", "shown"),
    [
        pytest.param(
            "from IPython.display import Markdown, display\n\n"
            'display(Markdown("step 1\\rstep 2"))\n',
            "\nstep 1\nstep 2\n",
            id="markdown-display",
        ),
        pytest.param(
            "class Steps:\n    def __repr__(self) -> str:\n"
            '        return "step 1\\rstep 2"\n\n\nSteps()\n',
            "\n    step 1\n    step 2\n",  # a plain output is indented, one line per line
            id="plain-result",
        ),
    ],
)
def test_a_lone_carriage_return_in_an_output_leaves_a_fresh_page_clean(
    tmp_path: Path, cell: str, shown: str
) -> None:
    repo, script = _repo(tmp_path, "# %%\n" + cell)

    page = generate_page(script, repo_root=repo, config=CONFIG)

    assert b"\r" not in page.read_bytes()
    assert shown in page.read_text(encoding="utf-8")
    assert check_page(script, repo_root=repo, config=CONFIG) == []


@pytest.mark.parametrize(
    "text",
    [
        pytest.param('# %%\nprint("the plot: ![loss](loss.png)")\n', id="printed"),
        pytest.param(
            '# %% [markdown]\n"""\n```markdown\n![loss](loss.png)\n```\n"""\n\n# %%\nprint(1)\n',
            id="fenced-in-prose",
        ),
    ],
)
def test_image_syntax_that_is_text_is_not_a_figure(tmp_path: Path, text: str) -> None:
    repo, script = _repo(tmp_path, text)

    generate_page(script, repo_root=repo, config=CONFIG)

    assert check_page(script, repo_root=repo, config=CONFIG) == []


def _example_in_figures_directory(script: Path) -> Path:
    """An example under ``examples/<path>_files/``, its page in the script's figures directory."""
    other = script.parent / f"{script.stem}_files" / "loading.py"
    other.parent.mkdir()
    other.write_text('# %% [markdown]\n"""\n# Loading\n"""\n\n# %%\nprint("loaded")\n', "utf-8")
    convert(other)
    return other


@pytest.mark.parametrize("relative", ["topic/demo.py", "demo.py"])
def test_an_example_whose_figures_directory_holds_examples_is_refused(
    tmp_path: Path, relative: str
) -> None:
    repo, script = write_example(tmp_path, SCRIPT, relative=relative)
    other = _example_in_figures_directory(script)
    other_page = generate_page(other, repo_root=repo, config=CONFIG)
    figures = f"docs/examples/{relative.removesuffix('.py')}_files"
    taken = f"examples/{relative.removesuffix('.py')}_files/loading.py"

    with pytest.raises(PageError, match=rf"{re.escape(figures)}, is the page directory of {taken}"):
        generate_page(script, repo_root=repo, config=CONFIG)

    assert not page_path(script, repo_root=repo).exists()
    assert other_page.is_file()
    assert check_page(other, repo_root=repo, config=CONFIG) == []


def test_a_page_whose_figures_directory_holds_examples_is_a_problem(tmp_path: Path) -> None:
    repo, script = _repo(tmp_path)
    generate_page(script, repo_root=repo, config=CONFIG)
    _example_in_figures_directory(script)

    problems = check_page(script, repo_root=repo, config=CONFIG)

    assert [problem.kind for problem in problems] == ["figures-directory-taken"]
    assert "examples/topic/demo_files/loading.py" in problems[0].detail


def test_regenerating_a_page_keeps_a_file_it_did_not_write_and_reports_it(tmp_path: Path) -> None:
    repo, script = _repo(tmp_path)
    page = generate_page(script, repo_root=repo, config=CONFIG)
    notes = page.parent / "demo_files" / "notes.txt"
    notes.write_text("kept", encoding="utf-8")

    generate_page(script, repo_root=repo, config=CONFIG)

    assert notes.read_text(encoding="utf-8") == "kept"
    assert [problem.kind for problem in check_page(script, repo_root=repo, config=CONFIG)] == [
        "stray-figure"
    ]


@pytest.mark.parametrize(
    "name", ["../../../pyproject.toml", "demo_files/../../../../pyproject.toml"]
)
def test_a_record_naming_a_file_outside_the_figures_directory_deletes_nothing(
    tmp_path: Path, name: str
) -> None:
    repo, script = _repo(tmp_path)
    page = generate_page(script, repo_root=repo, config=CONFIG)
    target = repo / "pyproject.toml"
    before = target.read_bytes()
    record = json.loads(re.findall(r"<!-- substrax-notebooks: (\{.*?\}) -->", page.read_text())[-1])
    record["page"]["figures"][name] = hashlib.sha256(before).hexdigest()
    _rewrite_record(page, page=record["page"])

    generate_page(script, repo_root=repo, config=CONFIG)

    assert target.read_bytes() == before


def test_regenerating_keeps_a_figure_changed_by_hand(tmp_path: Path) -> None:
    repo, script = _repo(tmp_path)
    page = generate_page(script, repo_root=repo, config=CONFIG)
    (figure,) = page.parent.rglob("*.png")
    figure.write_bytes(b"drawn by hand")
    script.write_text(SCRIPT.replace("plt.plot(values)\nplt.show()\n", "print(values)\n"))

    generate_page(script, repo_root=repo, config=CONFIG)

    assert figure.read_bytes() == b"drawn by hand"


def test_a_directory_of_the_figures_name_holding_no_example_is_not_refused(tmp_path: Path) -> None:
    repo, script = _repo(tmp_path)
    helpers = script.parent / "demo_files"
    helpers.mkdir()
    (helpers / "_shared.py").write_text("VALUE = 1\n", encoding="utf-8")
    (helpers / "table.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    generate_page(script, repo_root=repo, config=CONFIG)

    assert check_page(script, repo_root=repo, config=CONFIG) == []


@pytest.mark.parametrize("relative", ["topic/my demo.py", "topic/cost$5.py", "topic/it's.py"])
def test_the_header_run_command_names_the_example_as_the_shell_reads_it(
    tmp_path: Path, relative: str
) -> None:
    repo, script = write_example(tmp_path, SCRIPT, relative=relative)

    text = generate_page(script, repo_root=repo, config=CONFIG).read_text(encoding="utf-8")

    (line,) = [line for line in text.splitlines() if line.startswith("uv run python ")]
    argument = line.removeprefix("uv run python ")
    # The shell itself reads the argument: splitting, quotes and parameter expansion as a reader's.
    bash = shutil.which("bash")
    assert bash is not None
    read = subprocess.run(  # noqa: S603  # nosec B603 - bash reading the page's own command
        [bash, "-c", f"printf '%s\\0' {argument}"], capture_output=True, check=True, text=True
    )
    assert read.stdout == f"examples/{relative}\0"


def test_a_page_checks_clean_against_a_windows_checkout_of_its_script(tmp_path: Path) -> None:
    repo, script = _repo(tmp_path)
    generate_page(script, repo_root=repo, config=CONFIG)
    # What git's core.autocrlf=true writes on checkout.
    script.write_bytes(script.read_bytes().replace(b"\n", b"\r\n"))

    assert check_page(script, repo_root=repo, config=CONFIG) == []


def _opening(markdown: str) -> str:
    return SCRIPT.replace('"""\n# Demo\n\nWhat the demo shows.\n"""', f'"""\n{markdown}\n"""')


def test_the_header_follows_a_setext_title_whole(tmp_path: Path) -> None:
    repo, script = _repo(tmp_path, _opening("Demo\n====\n\nWhat the demo shows."))

    text = generate_page(script, repo_root=repo, config=CONFIG).read_text(encoding="utf-8")

    assert text.startswith("Demo\n====\n")
    assert text.index("> Ran in") < text.index("What the demo shows.")


@pytest.mark.parametrize(
    "markdown",
    [
        pytest.param("This example shows\nhow to sort.", id="paragraph"),
        pytest.param("<!-- a note -->\n# Demo\n\nWhat the demo shows.", id="comment-before-title"),
    ],
)
def test_the_header_goes_before_a_first_cell_that_does_not_open_with_a_title(
    tmp_path: Path, markdown: str
) -> None:
    repo, script = _repo(tmp_path, _opening(markdown))

    text = generate_page(script, repo_root=repo, config=CONFIG).read_text(encoding="utf-8")

    assert text.startswith("> Ran in")
    assert markdown in text
