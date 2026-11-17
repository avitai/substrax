"""``substrax.recipes``: a repository's full recipes, each paired with its example by path."""

from __future__ import annotations

from pathlib import Path

from substrax.recipes import discover_recipes, paired_example, recipe_page_path


def _write(repo: Path, *relatives: str) -> None:
    for relative in relatives:
        (repo / relative).parent.mkdir(parents=True, exist_ok=True)
        (repo / relative).write_text("", encoding="utf-8")


def test_discovery_skips_private_parts_as_example_discovery_does(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "recipes/pinns/burgers.py",
        "recipes/pinns/_shared.py",
        "recipes/_common/config.py",
        "recipes/operators/fno_darcy.py",
        "recipes/README.md",
    )

    found = discover_recipes(tmp_path)

    assert [path.relative_to(tmp_path).as_posix() for path in found] == [
        "recipes/operators/fno_darcy.py",
        "recipes/pinns/burgers.py",
    ]


def test_a_repository_without_a_recipes_directory_has_no_recipe(tmp_path: Path) -> None:
    _write(tmp_path, "examples/pinns/burgers.py")

    assert discover_recipes(tmp_path) == []


def test_a_recipe_s_example_and_page_share_its_path(tmp_path: Path) -> None:
    recipe = tmp_path / "recipes" / "pinns" / "burgers.py"

    assert paired_example(recipe, repo_root=tmp_path) == (
        tmp_path / "examples" / "pinns" / "burgers.py"
    )
    assert recipe_page_path(recipe, repo_root=tmp_path) == (
        tmp_path / "docs" / "examples" / "pinns" / "burgers.recipe.md"
    )
