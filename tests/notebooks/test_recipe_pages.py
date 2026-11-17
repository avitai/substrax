"""``substrax.notebooks.recipe_pages``: a recipe's page, written from its run's record."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from _example_texts import CONFIG, write_example, write_recipe

from substrax.notebooks.pages import PageError
from substrax.notebooks.recipe_pages import check_recipe_page, write_recipe_page
from substrax.notebooks.recipe_run import RecipeRecord, run_recipe


_RECIPE = '''"""Sorting at its published scale, as the reference sorts it."""


def main() -> dict[str, float]:
    """Return the measured error beside the published one."""
    return {"error": 0.25, "published_error": 0.24}


if __name__ == "__main__":
    main()
'''


def _recorded(tmp_path: Path, text: str = _RECIPE) -> tuple[Path, Path, RecipeRecord]:
    repo, _ = write_example(tmp_path)
    recipe = write_recipe(repo, text)
    return repo, recipe, run_recipe(recipe, repo_root=repo, distribution="demo")


def test_a_recipe_page_shows_its_run_beside_its_example(tmp_path: Path) -> None:
    repo, recipe, record = _recorded(tmp_path)

    page = write_recipe_page(record, repo_root=repo, config=CONFIG)

    assert page == repo / "docs" / "examples" / "sorting.recipe.md"
    text = page.read_text(encoding="utf-8")
    assert text.startswith("# Sorting at its published scale, as the reference sorts it.\n")
    assert f"> Ran in {record.seconds} s on {record.device} (" in text
    assert f"[`recipes/sorting.py`]({CONFIG.source_url}/recipes/sorting.py)" in text
    assert "uv run python recipes/sorting.py" in text
    assert "[`examples/sorting.py`](sorting.md)" in text
    assert "| `error` | 0.25 |" in text
    assert "| `published_error` | 0.24 |" in text
    assert check_recipe_page(recipe, repo_root=repo) == []


def test_a_value_that_is_no_flat_mapping_is_shown_as_json(tmp_path: Path) -> None:
    repo, _, record = _recorded(tmp_path)
    nested = dataclasses.replace(record, returned={"errors": [0.25, 0.26]})

    text = write_recipe_page(nested, repo_root=repo, config=CONFIG).read_text(encoding="utf-8")

    assert '```json\n{\n  "errors": [\n    0.25,\n    0.26\n  ]\n}\n```' in text


def test_a_page_with_no_page_edited_by_hand_or_of_another_version_is_a_problem(
    tmp_path: Path,
) -> None:
    repo, recipe, record = _recorded(tmp_path)
    assert [problem.kind for problem in check_recipe_page(recipe, repo_root=repo)] == ["missing"]

    page = write_recipe_page(record, repo_root=repo, config=CONFIG)
    page.write_text(page.read_text(encoding="utf-8").replace("0.25", "0.20"), encoding="utf-8")
    assert [problem.kind for problem in check_recipe_page(recipe, repo_root=repo)] == ["edited"]

    write_recipe_page(record, repo_root=repo, config=CONFIG)
    recipe.write_text(_RECIPE.replace("0.25", "0.30"), encoding="utf-8")
    assert [problem.kind for problem in check_recipe_page(recipe, repo_root=repo)] == ["stale"]


def test_a_record_of_another_version_of_the_recipe_is_refused(tmp_path: Path) -> None:
    repo, recipe, record = _recorded(tmp_path)
    recipe.write_text(_RECIPE.replace("0.25", "0.30"), encoding="utf-8")

    with pytest.raises(PageError, match=r"another version of recipes/sorting\.py"):
        write_recipe_page(record, repo_root=repo, config=CONFIG)

    assert not (repo / "docs" / "examples" / "sorting.recipe.md").exists()


def test_a_value_holding_a_local_path_is_refused(tmp_path: Path) -> None:
    """A path naming a person's machine, as ``local_paths`` reads one, never reaches a page."""
    repo, _, record = _recorded(tmp_path)
    leaking = dataclasses.replace(record, returned={"checkpoint": "/home/alice/runs/ckpt"})

    with pytest.raises(PageError, match="local path"):
        write_recipe_page(leaking, repo_root=repo, config=CONFIG)
