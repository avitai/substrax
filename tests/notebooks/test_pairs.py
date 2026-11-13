"""``substrax.notebooks.pairs``: an example script and the clean notebook generated from it."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from substrax.notebooks.pairs import check_pair, convert, PairProblem


SCRIPT = '''# %% [markdown]
"""
# Title

Some text.
"""

# %%
x = 1
print(x)


# %%
def double(value: int) -> int:
    return 2 * value
'''


def _script(tmp_path: Path, text: str = SCRIPT) -> Path:
    path = tmp_path / "examples" / "demo.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _notebook(path: Path) -> dict[str, Any]:
    return json.loads(path.with_suffix(".ipynb").read_text(encoding="utf-8"))


def test_convert_writes_a_clean_notebook_beside_the_script(tmp_path: Path) -> None:
    script = _script(tmp_path)

    written = convert(script)

    notebook = _notebook(script)
    assert written == script.with_suffix(".ipynb")
    assert [cell["cell_type"] for cell in notebook["cells"]] == ["markdown", "code", "code"]
    assert all(cell.get("outputs", []) == [] for cell in notebook["cells"])
    assert all(cell.get("execution_count") is None for cell in notebook["cells"])
    assert all("id" not in cell for cell in notebook["cells"])
    assert notebook["metadata"]["jupytext"]["formats"] == "ipynb,py:percent"


def test_convert_is_deterministic(tmp_path: Path) -> None:
    script = _script(tmp_path)

    first = convert(script).read_bytes()
    second = convert(script).read_bytes()

    assert first == second


def test_a_converted_pair_checks_clean(tmp_path: Path) -> None:
    script = _script(tmp_path)
    convert(script)

    assert check_pair(script) == []


def test_a_missing_notebook_is_a_problem(tmp_path: Path) -> None:
    script = _script(tmp_path)

    problems = check_pair(script)

    assert [problem.kind for problem in problems] == ["missing"]


def test_an_edited_script_is_out_of_sync_and_names_the_cell(tmp_path: Path) -> None:
    script = _script(tmp_path)
    convert(script)
    script.write_text(SCRIPT.replace("x = 1", "x = 2"), encoding="utf-8")

    problems = check_pair(script)

    assert [problem.kind for problem in problems] == ["out-of-sync"]
    assert "code cell 2" in problems[0].detail
    assert "line 1" in problems[0].detail


def test_a_changed_cell_kind_is_out_of_sync(tmp_path: Path) -> None:
    script = _script(tmp_path)
    convert(script)
    notebook = _notebook(script)
    notebook["cells"][1]["cell_type"] = "markdown"
    script.with_suffix(".ipynb").write_text(json.dumps(notebook), encoding="utf-8")

    problems = check_pair(script)

    assert [problem.kind for problem in problems] == ["out-of-sync"]
    assert "cell 2" in problems[0].detail


@pytest.mark.parametrize("blank_lines", [0, 1, 2])
def test_the_blank_lines_a_formatter_leaves_before_a_cell_keep_the_pair_in_sync(
    tmp_path: Path, blank_lines: int
) -> None:
    # jupytext drops up to two blank lines before a cell marker, and ruff format leaves at most two.
    script = _script(tmp_path)
    convert(script)
    script.write_text(
        SCRIPT.replace("print(x)\n\n\n", "print(x)\n" + "\n" * blank_lines), encoding="utf-8"
    )

    assert check_pair(script) == []


@pytest.mark.parametrize(
    "edit",
    [
        lambda notebook: notebook["metadata"]["jupytext"].update(
            {"text_representation": {"jupytext_version": "1.19.6"}}
        ),
        lambda notebook: notebook["cells"][0].update({"id": "a1b2c3"}),
        lambda notebook: notebook["cells"][1].update({"metadata": {"tags": ["parameters"]}}),
    ],
    ids=["notebook-metadata", "cell-id", "cell-metadata"],
)
def test_a_notebook_convert_did_not_write_is_not_generated(
    tmp_path: Path, edit: Callable[[dict[str, Any]], None]
) -> None:
    script = _script(tmp_path)
    convert(script)
    notebook = _notebook(script)
    edit(notebook)
    script.with_suffix(".ipynb").write_text(json.dumps(notebook, indent=1), encoding="utf-8")

    problems = check_pair(script)

    assert [problem.kind for problem in problems] == ["not-generated"]
    assert "substrax-notebooks convert" in problems[0].detail


def test_a_notebook_written_with_other_json_layout_is_not_generated(tmp_path: Path) -> None:
    script = _script(tmp_path)
    convert(script)
    path = script.with_suffix(".ipynb")
    path.write_text(json.dumps(json.loads(path.read_text(encoding="utf-8"))), encoding="utf-8")

    assert [problem.kind for problem in check_pair(script)] == ["not-generated"]


def test_a_notebook_holding_outputs_is_not_clean(tmp_path: Path) -> None:
    script = _script(tmp_path)
    convert(script)
    notebook = _notebook(script)
    notebook["cells"][1]["outputs"] = [{"output_type": "stream", "name": "stdout", "text": "1\n"}]
    notebook["cells"][1]["execution_count"] = 1
    script.with_suffix(".ipynb").write_text(json.dumps(notebook), encoding="utf-8")

    problems = check_pair(script)

    assert [problem.kind for problem in problems] == ["not-clean"]
    assert "code cell 2" in problems[0].detail


def test_check_never_writes(tmp_path: Path) -> None:
    script = _script(tmp_path)
    convert(script)
    script.write_text(SCRIPT.replace("x = 1", "x = 3"), encoding="utf-8")
    notebook = script.with_suffix(".ipynb")
    before = (script.read_bytes(), notebook.read_bytes(), notebook.stat().st_mtime_ns)

    check_pair(script)

    assert (script.read_bytes(), notebook.read_bytes(), notebook.stat().st_mtime_ns) == before


def test_an_unreadable_notebook_is_a_problem_not_a_crash(tmp_path: Path) -> None:
    script = _script(tmp_path)
    script.with_suffix(".ipynb").write_text("{not json", encoding="utf-8")

    problems = check_pair(script)

    assert [problem.kind for problem in problems] == ["unreadable"]


def test_a_problem_names_its_script(tmp_path: Path) -> None:
    script = _script(tmp_path)

    problem = check_pair(script)[0]

    assert isinstance(problem, PairProblem)
    assert problem.script == script
    assert str(script) in str(problem)


def test_convert_refuses_a_path_that_is_not_a_python_script(tmp_path: Path) -> None:
    other = tmp_path / "notes.txt"
    other.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match=r"\.py"):
        convert(other)


def test_a_script_with_a_cell_the_notebook_lacks_is_out_of_sync(tmp_path: Path) -> None:
    script = _script(tmp_path)
    convert(script)
    script.write_text(SCRIPT + "\n\n# %%\nprint(double(x))\n", encoding="utf-8")

    problems = check_pair(script)

    assert [problem.kind for problem in problems] == ["out-of-sync"]
    assert problems[0].detail == "4 cells in the script, 3 in the notebook"


def test_a_script_that_is_not_utf8_is_an_unreadable_pair_not_a_crash(tmp_path: Path) -> None:
    script = tmp_path / "example.py"
    script.write_text(SCRIPT, encoding="utf-8")
    convert(script)
    script.write_bytes(script.read_bytes().replace(b"Some text", b"Som\xe9 text"))

    problems = check_pair(script)

    assert [problem.kind for problem in problems] == ["unreadable"]
    assert "UTF-8" in problems[0].detail


def test_convert_refuses_a_script_that_is_not_utf8_naming_the_encoding(tmp_path: Path) -> None:
    script = tmp_path / "example.py"
    script.write_bytes(SCRIPT.encode("utf-8").replace(b"Some text", b"Som\xe9 text"))

    with pytest.raises(ValueError, match="not UTF-8"):
        convert(script)


def test_a_byte_order_mark_does_not_change_the_generated_notebook(tmp_path: Path) -> None:
    plain, marked = tmp_path / "plain.py", tmp_path / "marked.py"
    plain.write_text(SCRIPT, encoding="utf-8")
    marked.write_bytes(b"\xef\xbb\xbf" + SCRIPT.encode("utf-8"))

    assert convert(plain).read_text(encoding="utf-8") == convert(marked).read_text(encoding="utf-8")


def test_a_notebook_that_is_not_utf8_is_unreadable_not_a_crash(tmp_path: Path) -> None:
    script = _script(tmp_path)
    notebook = convert(script)
    written = notebook.read_bytes()
    notebook.write_bytes(written.replace(b"Some text", b"Som\xe9 text"))
    byte = written.index(b"Some text") + len(b"Som")

    problems = check_pair(script)

    assert [problem.kind for problem in problems] == ["unreadable"]
    assert problems[0].detail == (
        f"{notebook} is not UTF-8: byte {byte} cannot be decoded; save it as UTF-8"
    )


def test_a_notebook_cell_with_no_kind_is_unreadable_not_a_crash(tmp_path: Path) -> None:
    script = _script(tmp_path)
    notebook = convert(script)
    written = json.loads(notebook.read_text(encoding="utf-8"))
    del written["cells"][0]["cell_type"]
    notebook.write_text(json.dumps(written), encoding="utf-8")

    problems = check_pair(script)

    assert [problem.kind for problem in problems] == ["unreadable"]
    assert problems[0].detail == "a cell has no 'cell_type' field"


@pytest.mark.parametrize("ending", [b"\r\n", b"\r"], ids=["windows", "carriage-return"])
def test_a_pair_checked_out_with_other_line_endings_checks_clean(
    tmp_path: Path, ending: bytes
) -> None:
    """What git's ``core.autocrlf`` writes on checkout is the same pair (BR-15)."""
    script = _script(tmp_path)
    notebook = convert(script)
    for path in (script, notebook):
        path.write_bytes(path.read_bytes().replace(b"\n", ending))

    assert check_pair(script) == []
