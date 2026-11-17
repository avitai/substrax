"""``substrax.notebooks.conformance.check_recipe``: the guide's code rules, held over a recipe."""

from __future__ import annotations

from pathlib import Path

import pytest
from _example_texts import CONFIG, RECIPE, write_example, write_recipe

from substrax.notebooks.conformance import check_recipe, Finding, RECIPE_RULES


_REQUIRE_GPU = "from substrax.devices import require_gpu\n"


def _findings(tmp_path: Path, text: str) -> list[Finding]:
    repo, _ = write_example(tmp_path)
    recipe = write_recipe(repo, text)
    return check_recipe(recipe, repo_root=repo, config=CONFIG)


def test_the_recipe_rules_are_the_guide_s_rules_about_code_not_about_notebooks() -> None:
    assert list(RECIPE_RULES) == [
        "parses",
        "main",
        "no-arguments",
        "no-backend-forcing",
        "no-env-writes",
        "artifacts",
        "no-star-imports",
        "typed-keys",
        "gpu-check",
    ]


def test_a_conforming_recipe_has_no_finding(tmp_path: Path) -> None:
    assert _findings(tmp_path, RECIPE) == []


@pytest.mark.parametrize(
    ("old", "new", "rule"),
    [
        pytest.param("jax.random.key(0)", "jax.random.PRNGKey(0)", "typed-keys", id="typed-keys"),
        pytest.param(
            "import jax\n", "import argparse\n\nimport jax\n", "no-arguments", id="no-arguments"
        ),
        pytest.param(
            "import jax\n",
            'import os\n\nimport jax\n\nos.environ["DATA"] = "here"\n',
            "no-env-writes",
            id="no-env-writes",
        ),
        pytest.param(
            "import jax\n",
            'import jax\n\njax.config.update("jax_platforms", "cpu")\n',
            "no-backend-forcing",
            id="no-backend-forcing",
        ),
        pytest.param(
            "import jax.numpy as jnp\n",
            "from jax.numpy import *\n",
            "no-star-imports",
            id="no-star-imports",
        ),
        pytest.param(
            '    return {"in_order"',
            '    open("docs/out.txt", "w").close()\n    return {"in_order"',
            "artifacts",
            id="artifacts",
        ),
        pytest.param("def main()", "def run()", "main", id="main-missing"),
        pytest.param('if __name__ == "__main__":\n    main()\n', "", "main", id="guard-missing"),
        pytest.param("SIZE = 1_048_576", "SIZE = = 1", "parses", id="parses"),
    ],
)
def test_each_rule_finds_its_own_breach(tmp_path: Path, old: str, new: str, rule: str) -> None:
    assert old in RECIPE
    findings = _findings(tmp_path, RECIPE.replace(old, new))

    assert [finding.rule for finding in findings] == [rule]


def test_a_recipe_without_its_guard_is_told_to_end_with_it(tmp_path: Path) -> None:
    (finding,) = _findings(tmp_path, RECIPE.replace('if __name__ == "__main__":\n    main()\n', ""))

    assert finding.detail == 'the script does not end with `if __name__ == "__main__": main()`'


def test_a_recipe_calling_require_gpu_right_after_its_imports_follows_gpu_check(
    tmp_path: Path,
) -> None:
    text = RECIPE.replace("import jax\n", f"import jax\n{_REQUIRE_GPU}\nrequire_gpu()\n", 1)

    assert _findings(tmp_path, text) == []


def test_a_recipe_calling_require_gpu_after_work_breaks_gpu_check(tmp_path: Path) -> None:
    """Work before the check could already run on the CPU, as a call inside ``main()`` can."""
    text = RECIPE.replace("import jax\n", f"import jax\n{_REQUIRE_GPU}", 1).replace(
        "    values = ", "    require_gpu()\n    values = "
    )

    (finding,) = _findings(tmp_path, text)

    assert finding.rule == "gpu-check"
    assert "right after its imports" in finding.detail


def test_an_unknown_rule_is_refused(tmp_path: Path) -> None:
    repo, _ = write_example(tmp_path)
    recipe = write_recipe(repo)

    with pytest.raises(KeyError, match="'title'"):
        check_recipe(recipe, repo_root=repo, config=CONFIG, rules=["title"])
