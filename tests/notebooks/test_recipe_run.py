"""``substrax.notebooks.recipe_run``: a recipe run as its script, ``main()``'s value recorded."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
from _example_texts import write_recipe

from substrax.notebooks.recipe_run import RecipeRecord, RecipeRunError, run_recipe, write_record
from substrax.records import read_record


_RETURNS = '''"""A recipe returning its result."""


def main() -> dict[str, float]:
    """Return the result."""
    return {"error": 0.25}


if __name__ == "__main__":
    main()
'''


def _run(tmp_path: Path, text: str) -> RecipeRecord:
    recipe = write_recipe(tmp_path, text)
    return run_recipe(recipe, repo_root=tmp_path, distribution="demo")


def test_a_run_records_main_s_value_with_where_it_came_from(tmp_path: Path) -> None:
    record = _run(tmp_path, _RETURNS)

    assert record.source == "recipes/sorting.py"
    assert record.sha256 == hashlib.sha256(_RETURNS.encode()).hexdigest()
    assert record.returned == {"error": 0.25}
    assert record.seconds >= 0
    assert record.device
    assert record.versions["python"] == sys.version.split()[0]


def test_a_written_record_reads_back_as_the_same_record(tmp_path: Path) -> None:
    record = _run(tmp_path, _RETURNS)

    path = write_record(record, tmp_path / "outputs")

    assert path == tmp_path / "outputs" / "record.json"
    assert read_record(RecipeRecord, json.loads(path.read_text(encoding="utf-8"))) == record


def test_the_recipe_sees_what_a_script_run_gives_it(tmp_path: Path) -> None:
    """Its directory first on ``sys.path``, its path as ``sys.argv[0]``, its ``__file__``.

    The helper beside it is named as a module the runner itself imports, ``nbformat``, so a run
    in the runner's process would import the runner's module in its place, as ``python`` does not.
    """
    assert "nbformat" in sys.modules
    write_recipe(tmp_path, "VALUE = 7\n", relative="nbformat.py")
    text = _RETURNS.replace(
        '    return {"error": 0.25}',
        '    import nbformat\n\n    return {"argv": sys.argv, "file": __file__, '
        '"helper": nbformat.VALUE, "name": __name__}',
    ).replace('"""A recipe returning its result."""\n', '"""A recipe."""\n\nimport sys\n')
    path, argv = list(sys.path), list(sys.argv)

    record = _run(tmp_path, text)

    assert record.returned == {
        "argv": ["recipes/sorting.py"],
        "file": str(tmp_path / "recipes" / "sorting.py"),
        "helper": 7,
        "name": "__main__",
    }
    assert (sys.path, sys.argv) == (path, argv)


def test_a_value_json_cannot_hold_is_refused_naming_it(tmp_path: Path) -> None:
    text = _RETURNS.replace('return {"error": 0.25}', 'return {"seen": {1, 2}}')

    with pytest.raises(RecipeRunError, match="'seen', which JSON cannot hold"):
        _run(tmp_path, text)


def test_a_recipe_whose_guard_never_ran_main_is_refused(tmp_path: Path) -> None:
    text = _RETURNS.replace('if __name__ == "__main__":\n    main()\n', "")

    with pytest.raises(RecipeRunError, match='no `if __name__ == "__main__": main\\(\\)` ran'):
        _run(tmp_path, text)


def test_a_clean_exit_is_a_run_that_returned_nothing(tmp_path: Path) -> None:
    text = _RETURNS.replace('    return {"error": 0.25}', "    raise SystemExit(0)").replace(
        "dict[str, float]", "None"
    )

    assert _run(tmp_path, text).returned is None


def test_any_other_exit_and_any_error_end_the_run_with_its_status(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    """The run leaves no record, says its status, and its traceback reaches standard error."""
    exits = _RETURNS.replace('    return {"error": 0.25}', "    raise SystemExit(3)")
    raises = _RETURNS.replace('    return {"error": 0.25}', '    raise ValueError("diverged")')

    with pytest.raises(RecipeRunError, match="exited with status 3"):
        _run(tmp_path, exits)
    with pytest.raises(RecipeRunError, match="exited with status 1"):
        _run(tmp_path / "second", raises)

    assert "ValueError: diverged" in capfd.readouterr().err
