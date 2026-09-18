"""Source scans a repository's contract tests run over its own files."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from substrax.testing import run_python
from substrax.testing.source_scans import (
    configures_logging,
    documented_texts,
    import_time_lines,
    is_main_guard,
    python_files,
    unguarded_make_mesh_calls,
    writes_environment,
)


def _module(tmp_path: Path, source: str) -> Path:
    path = tmp_path / "module.py"
    path.write_text(source, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "statement",
    [
        "os.environ['X'] = '1'",
        "os.environ['X'] += '1'",
        "del os.environ['X']",
        "os.environ |= {'X': '1'}",
        "os.environ.update({'X': '1'})",
        "os.environ.setdefault('X', '1')",
        "os.environ.pop('X', None)",
        "os.environ.popitem()",
        "os.environ.clear()",
        "os.putenv('X', '1')",
        "os.unsetenv('X')",
    ],
)
def test_each_way_of_changing_the_environment_is_a_write(tmp_path: Path, statement: str) -> None:
    module = _module(tmp_path, f"import os\n{statement}\n")

    assert import_time_lines(module, writes_environment) == [2]


@pytest.mark.parametrize(
    "statement",
    [
        "value = os.environ['X']",
        "value = os.environ.get('X')",
        "names = list(os.environ)",
        "settings = {}; settings['X'] = '1'",
        "settings = {}; settings.update(X='1')",
    ],
)
def test_reading_the_environment_or_writing_another_mapping_is_not_a_write(
    tmp_path: Path, statement: str
) -> None:
    module = _module(tmp_path, f"import os\n{statement}\n")

    assert import_time_lines(module, writes_environment) == []


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("import logging\nlogging.basicConfig()\n", [2]),
        ("from logging import basicConfig\nbasicConfig(level=10)\n", [2]),
        ("import logging\nlogging.getLogger(__name__)\n", []),
        ("import logging\nlogging.getLogger().setLevel(10)\n", []),
    ],
)
def test_configuring_logging_means_calling_basic_config(
    tmp_path: Path, source: str, expected: list[int]
) -> None:
    assert import_time_lines(_module(tmp_path, source), configures_logging) == expected


def test_only_code_that_runs_at_import_is_reported(tmp_path: Path) -> None:
    module = _module(
        tmp_path,
        "import logging\n"
        "import os\n"
        "logging.basicConfig()\n"
        "os.environ['X'] = '1'\n"
        "if os.name == 'posix':\n"
        "    os.environ['Y'] = '1'\n"
        "try:\n"
        "    logging.basicConfig()\n"
        "except ValueError:\n"
        "    pass\n"
        "def main():\n"
        "    logging.basicConfig()\n"
        "    os.environ['Z'] = '1'\n"
        "async def serve():\n"
        "    os.environ['Z'] = '1'\n"
        "class Settings:\n"
        "    def apply(self):\n"
        "        os.environ['Z'] = '1'\n"
        "reset = lambda: os.environ.clear()\n"
        "if __name__ != '__main__':\n"
        "    os.environ['W'] = '1'\n"
        "if __name__ == '__main__':\n"
        "    logging.basicConfig()\n"
        "    os.environ.update({})\n",
    )

    assert import_time_lines(module, configures_logging) == [3, 8]
    assert import_time_lines(module, writes_environment) == [4, 6, 21]


@pytest.mark.parametrize(
    ("condition", "expected"),
    [
        ("__name__ == '__main__'", True),
        ('__name__ == "__main__"', True),
        ("'__main__' == __name__", True),
        ("__name__ != '__main__'", False),
        ("__name__ == '__main__' == other", False),
        ("__name__ == 'other'", False),
        ("name == '__main__'", False),
    ],
)
def test_the_main_guard_is_recognised_by_its_comparison(condition: str, expected: bool) -> None:
    node = ast.parse(f"if {condition}:\n    pass\n").body[0]

    assert is_main_guard(node) is expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('mesh = jax.make_mesh((4,), ("data",))', ['make_mesh((4,), ("data",))']),
        ('jax.make_mesh((4, 2), ("data", "model"), axis_types=(AxisType.Auto,) * 2)', []),
        ('jax.make_mesh(\n    (4, 2),\n    ("data", "model"),\n    axis_types=types,\n)', []),
        ('DeviceMeshManager.create_device_mesh({"data": 4})', []),
        ('jax.sharding.Mesh(devices, ("data",))', []),
        (
            'a = jax.make_mesh((1,), ("x",), axis_types=t)\nb = jax.make_mesh((2,), ("y",))',
            ['make_mesh((2,), ("y",))'],
        ),
        ('jax.make_mesh((4,), ("data",)\n', ['make_mesh((4,), ("data",)\n']),
    ],
)
def test_a_make_mesh_call_without_axis_types_is_reported(text: str, expected: list[str]) -> None:
    assert unguarded_make_mesh_calls(text) == expected


def _repository(tmp_path: Path) -> Path:
    package = tmp_path / "src" / "demo"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('"""Package docstring."""\n', encoding="utf-8")
    (package / "mesh.py").write_text(
        '"""Module docstring."""\n\n\n'
        "class Mesh:\n"
        '    """Build it with ``jax.make_mesh((2,), ("data",))``."""\n\n'
        "    def build(self):\n"
        '        """Method docstring."""\n\n\n'
        "def undocumented():\n"
        "    return 1\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("# Demo\n", encoding="utf-8")
    (tmp_path / "docs" / "guide").mkdir(parents=True)
    (tmp_path / "docs" / "guide" / "mesh.md").write_text("Mesh guide\n", encoding="utf-8")
    (tmp_path / "examples" / "data").mkdir(parents=True)
    (tmp_path / "examples" / "train.py").write_text("print('train')\n", encoding="utf-8")
    (tmp_path / "examples" / "data" / "helper.py").write_text("x = 1\n", encoding="utf-8")
    return tmp_path


def test_documented_texts_reach_docstrings_pages_and_scripts(tmp_path: Path) -> None:
    root = _repository(tmp_path)

    texts = documented_texts(root, package="demo", script_directories=("examples",))

    assert texts == {
        "src/demo/__init__.py:0": "Package docstring.",
        "src/demo/mesh.py:0": "Module docstring.",
        "src/demo/mesh.py:4": 'Build it with ``jax.make_mesh((2,), ("data",))``.',
        "src/demo/mesh.py:7": "Method docstring.",
        "README.md": "# Demo\n",
        "docs/guide/mesh.md": "Mesh guide\n",
        "examples/data/helper.py": "x = 1\n",
        "examples/train.py": "print('train')\n",
    }


def test_documented_texts_default_to_docstrings_readme_and_docs(tmp_path: Path) -> None:
    root = _repository(tmp_path)

    texts = documented_texts(root, package="demo")

    assert not any(name.startswith("examples/") for name in texts)
    assert {"README.md", "docs/guide/mesh.md"} <= set(texts)


def test_a_missing_corpus_location_raises_instead_of_shrinking_the_corpus(tmp_path: Path) -> None:
    root = _repository(tmp_path)

    with pytest.raises(FileNotFoundError, match="src/absent"):
        documented_texts(root, package="absent")
    with pytest.raises(FileNotFoundError, match=r"CONTRIBUTING\.md"):
        documented_texts(root, package="demo", pages=("README.md", "CONTRIBUTING.md"))
    with pytest.raises(FileNotFoundError, match="scripts"):
        python_files(root, ("src", "scripts"))


def test_python_files_are_sorted_and_skip_excluded_directories(tmp_path: Path) -> None:
    root = _repository(tmp_path)

    files = python_files(root, ("examples", "src"), excluded_parts=("data",))

    assert [path.relative_to(root).as_posix() for path in files] == [
        "examples/train.py",
        "src/demo/__init__.py",
        "src/demo/mesh.py",
    ]
    assert python_files(root, ("examples",), pattern="train*.py") == [
        root / "examples" / "train.py"
    ]


def test_importing_the_scans_imports_no_jax() -> None:
    result = run_python(
        "import sys, substrax.testing.source_scans; print('jax' in sys.modules)", timeout=60
    )

    assert result.stdout.strip() == "False"
