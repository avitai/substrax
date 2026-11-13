"""``substrax.notebooks.environment_writes``: the variables a statement writes."""

from __future__ import annotations

import ast

import pytest

from substrax.notebooks.environment_writes import environment_writes


def _statement(source: str) -> ast.AST:
    return ast.parse(source).body[0]


def _writes(source: str, **names: tuple[str, ...]) -> list[str]:
    tree = _statement(source)
    return sorted({key for node in ast.walk(tree) for key in environment_writes(node, **names)})


@pytest.mark.parametrize(
    ("source", "written"),
    [
        pytest.param("os.environ['DATA'] = '1'", ["DATA"], id="an-item"),
        pytest.param("del os.environ['DATA']", ["DATA"], id="a-deletion"),
        pytest.param("os.environ |= {'DATA': '1'}", ["os.environ"], id="a-merge"),
        pytest.param("os.environ['DATA'] += '1'", ["DATA"], id="an-augmented-item"),
        pytest.param("os.environ.update(DATA='1')", ["DATA"], id="update-by-keyword"),
        pytest.param("os.environ.setdefault('DATA', '1')", ["DATA"], id="setdefault"),
        pytest.param("os.environ.pop('DATA')", ["DATA"], id="pop"),
        pytest.param("os.environ.clear()", ["os.environ"], id="clear"),
        pytest.param("os.environ[name] = '1'", ["os.environ"], id="a-key-not-spelled"),
        pytest.param("os.putenv('DATA', '1')", ["DATA"], id="putenv"),
        pytest.param("os.unsetenv('DATA')", ["DATA"], id="unsetenv"),
        pytest.param("print(os.environ.get('DATA'))", [], id="a-read"),
        pytest.param("value = os.environ['DATA']", [], id="an-item-read"),
    ],
)
def test_each_write_names_the_variable_it_writes(source: str, written: list[str]) -> None:
    assert _writes(source) == written


def test_only_the_names_given_count_as_the_environment() -> None:
    """By default only ``os.``-qualified names do; a caller adds the names its module imports."""
    assert _writes("environ['DATA'] = '1'") == []
    assert _writes("putenv('DATA', '1')") == []
    assert _writes(
        "environ['DATA'] = '1'", environ=("os.environ", "environ"), mutators=("putenv",)
    ) == ["DATA"]
    assert _writes("putenv('DATA', '1')", mutators=("putenv",)) == ["DATA"]
