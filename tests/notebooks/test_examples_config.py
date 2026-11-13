"""``substrax.notebooks.config``: a repository's ``[tool.substrax.examples]`` settings."""

from __future__ import annotations

import re
from pathlib import Path

import pydantic
import pytest

from substrax.notebooks.config import ExamplesConfig, read_examples_config


SOURCE = "https://github.com/avitai/demo/blob/main"


def _pyproject(tmp_path: Path, table: str) -> Path:
    (tmp_path / "pyproject.toml").write_text(
        f'[project]\nname = "demo"\nversion = "0"\n\n{table}', encoding="utf-8"
    )
    return tmp_path


def test_the_table_is_read_with_its_signatures(tmp_path: Path) -> None:
    repo = _pyproject(
        tmp_path,
        "[tool.substrax.examples]\n"
        f'source_url = "{SOURCE}"\n'
        'budget_seconds = 600\nunavailable = ["dataset is not set"]\n',
    )

    config = read_examples_config(repo)

    assert config == ExamplesConfig(
        distribution="demo",
        source_url=SOURCE,
        budget_seconds=600.0,
        unavailable=("dataset is not set",),
    )


def test_signatures_are_optional(tmp_path: Path) -> None:
    repo = _pyproject(
        tmp_path, f'[tool.substrax.examples]\nsource_url = "{SOURCE}"\nbudget_seconds = 30\n'
    )

    assert read_examples_config(repo).unavailable == ()


def test_a_missing_table_names_the_table(tmp_path: Path) -> None:
    repo = _pyproject(tmp_path, "")

    with pytest.raises(LookupError, match=r"\[tool\.substrax\.examples\]"):
        read_examples_config(repo)


def test_a_misspelled_key_is_refused(tmp_path: Path) -> None:
    repo = _pyproject(
        tmp_path,
        f'[tool.substrax.examples]\nsource_url = "{SOURCE}"\nbudget_seconds = 30\nbudget = 30\n',
    )

    with pytest.raises(pydantic.ValidationError, match="budget"):
        read_examples_config(repo)


def test_a_missing_budget_is_refused(tmp_path: Path) -> None:
    repo = _pyproject(tmp_path, f'[tool.substrax.examples]\nsource_url = "{SOURCE}"\n')

    with pytest.raises(pydantic.ValidationError, match="budget_seconds"):
        read_examples_config(repo)


@pytest.mark.parametrize("budget", [0, -1, float("inf"), float("nan")])
def test_a_budget_that_is_not_positive_is_refused(budget: float) -> None:
    with pytest.raises(ValueError, match="budget_seconds"):
        ExamplesConfig(distribution="demo", source_url=SOURCE, budget_seconds=budget)


@pytest.mark.parametrize(
    "url",
    ["http://github.com/avitai/demo/blob/main", "https://github.com/avitai/demo/blob/main/", ""],
    ids=["not-https", "trailing-slash", "empty"],
)
def test_a_source_url_links_cannot_be_built_on_is_refused(url: str) -> None:
    with pytest.raises(ValueError, match="source_url"):
        ExamplesConfig(distribution="demo", source_url=url, budget_seconds=30)


def test_the_page_size_limits_default_to_documenters() -> None:
    config = ExamplesConfig(distribution="demo", source_url=SOURCE, budget_seconds=30)

    assert (config.page_size_warn_bytes, config.page_size_bytes) == (102_400, 204_800)
    assert config.page_size_exempt == ()


@pytest.mark.parametrize(("warn", "limit"), [(0, 100), (100, 0), (-1, 100), (200, 100)])
def test_page_size_limits_that_are_not_positive_or_out_of_order_are_refused(
    warn: int, limit: int
) -> None:
    with pytest.raises(ValueError, match="page_size"):
        ExamplesConfig(
            distribution="demo",
            source_url=SOURCE,
            budget_seconds=30,
            page_size_warn_bytes=warn,
            page_size_bytes=limit,
        )


@pytest.mark.parametrize(
    "entry",
    [
        pytest.param("./examples/sorting.py", id="a-dot-segment"),
        pytest.param("examples//sorting.py", id="a-double-slash"),
        pytest.param("examples/../examples/sorting.py", id="a-parent-segment"),
        pytest.param("/home/someone/repo/examples/sorting.py", id="absolute"),
        pytest.param("sorting.py", id="not-under-examples"),
        pytest.param("examples/sorting.ipynb", id="the-notebook"),
    ],
)
def test_a_page_size_exemption_not_in_a_pages_source_form_is_refused(entry: str) -> None:
    """An entry is compared with a page's source, ``examples/<path>.py``, as the page records it."""
    with pytest.raises(ValueError, match="page_size_exempt"):
        ExamplesConfig(
            distribution="demo", source_url=SOURCE, budget_seconds=60, page_size_exempt=(entry,)
        )


def _repository_with(tmp_path: Path, exempt: str) -> Path:
    repo = _pyproject(
        tmp_path,
        "[tool.substrax.examples]\n"
        f'source_url = "{SOURCE}"\nbudget_seconds = 600\npage_size_exempt = ["{exempt}"]\n',
    )
    for relative in ("sorting.py", "_shared.py"):
        (repo / "examples" / relative).parent.mkdir(parents=True, exist_ok=True)
        (repo / "examples" / relative).write_text("", encoding="utf-8")
    return repo


def test_a_page_size_exemption_naming_an_example_is_read(tmp_path: Path) -> None:
    config = read_examples_config(_repository_with(tmp_path, "examples/sorting.py"))

    assert config.page_size_exempt == ("examples/sorting.py",)


@pytest.mark.parametrize(
    "entry",
    [
        pytest.param("examples/sortng.py", id="misspelled"),
        pytest.param("examples/_shared.py", id="a-private-helper"),
    ],
)
def test_a_page_size_exemption_naming_no_example_is_refused(tmp_path: Path, entry: str) -> None:
    with pytest.raises(ValueError, match=re.escape(entry)):
        read_examples_config(_repository_with(tmp_path, entry))


@pytest.mark.parametrize(
    "signature",
    [
        pytest.param("", id="empty"),
        pytest.param(" ", id="a-space"),
        pytest.param(":", id="a-colon"),
        pytest.param(": ", id="the-separator"),
    ],
)
def test_an_unavailable_signature_every_raised_exception_holds_is_refused(signature: str) -> None:
    """A raised exception reads ``<type>: <message>``, so these would skip every failing run."""
    with pytest.raises(ValueError, match="unavailable"):
        ExamplesConfig(
            distribution="demo",
            source_url=SOURCE,
            budget_seconds=60,
            unavailable=("data", signature),
        )


def test_the_distribution_is_the_project_s_name(tmp_path: Path) -> None:
    repo = _pyproject(
        tmp_path, f'[tool.substrax.examples]\nsource_url = "{SOURCE}"\nbudget_seconds = 30\n'
    )

    assert read_examples_config(repo).distribution == "demo"


def test_a_pyproject_naming_no_project_is_refused(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        f'[tool.substrax.examples]\nsource_url = "{SOURCE}"\nbudget_seconds = 30\n',
        encoding="utf-8",
    )

    with pytest.raises(LookupError, match=r"\[project\]"):
        read_examples_config(tmp_path)


def test_the_table_cannot_set_the_distribution(tmp_path: Path) -> None:
    """The project declares what it builds; a second spelling in the table could disagree."""
    repo = _pyproject(
        tmp_path,
        "[tool.substrax.examples]\n"
        f'source_url = "{SOURCE}"\nbudget_seconds = 30\ndistribution = "other"\n',
    )

    with pytest.raises(ValueError, match="distribution"):
        read_examples_config(repo)
