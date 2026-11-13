"""``substrax.notebooks.kernel_setup``: what a run gives an example's kernel, tested in process."""

from __future__ import annotations

import ast
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import pytest
from _example_texts import CONFIG, GOOD, write_example

from substrax.notebooks.kernel_setup import (
    is_guarded_main_call,
    is_main_guard,
    MAIN_RETURNED,
    MainResult,
    REPORTER,
    setup_kernel,
    write_report,
)
from substrax.notebooks.run import execute_example


class _Events:
    def __init__(self) -> None:
        self.registered: dict[str, list[Callable[..., object]]] = {}

    def register(self, event: str, function: Callable[..., object]) -> None:
        self.registered.setdefault(event, []).append(function)


class _Shell:
    def __init__(self) -> None:
        self.ast_transformers: list[ast.NodeTransformer] = []
        self.events = _Events()


@dataclass(frozen=True)
class _Result:
    error_in_exec: BaseException | None


def test_the_kernel_sees_what_a_script_run_gives_the_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "path", ["", "/elsewhere", str(tmp_path)])
    monkeypatch.setattr(sys, "argv", ["ipykernel_launcher.py"])
    user_ns: dict[str, object] = {}
    shell = _Shell()

    setup_kernel(
        user_ns,
        shell,
        directory=str(tmp_path),
        argv0="examples/sorting.py",
        file=str(tmp_path / "sorting.py"),
        report_path=str(tmp_path / "report.json"),
        raised_path=str(tmp_path / "raised.json"),
        distribution=CONFIG.distribution,
    )

    assert sys.path == [str(tmp_path), "/elsewhere"]
    assert sys.argv == ["examples/sorting.py"]
    assert user_ns["__file__"] == str(tmp_path / "sorting.py")
    assert callable(user_ns[REPORTER])
    assert [type(transformer) for transformer in shell.ast_transformers] == [MainResult]


class _OwnError(Exception):
    pass


@pytest.mark.parametrize(
    ("error", "recorded"),
    [
        pytest.param(ValueError("bad"), "builtins.ValueError", id="a-builtin"),
        pytest.param(_OwnError(), f"{__name__}._OwnError", id="a-class-of-its-own"),
    ],
)
def test_a_cell_s_exception_is_recorded_by_its_qualified_class(
    tmp_path: Path, error: BaseException, recorded: str
) -> None:
    """The qualified name the class carries, which a same-named class elsewhere does not."""
    shell = _Shell()
    raised = tmp_path / "raised.json"
    setup_kernel(
        {},
        shell,
        directory=str(tmp_path),
        argv0="examples/sorting.py",
        file=str(tmp_path / "sorting.py"),
        report_path=str(tmp_path / "report.json"),
        raised_path=str(raised),
        distribution=CONFIG.distribution,
    )
    (after_each_cell,) = shell.events.registered["post_run_cell"]

    after_each_cell(_Result(error_in_exec=None))
    after_each_cell(None)  # IPython passes no result when running the cell itself failed
    assert not raised.exists()
    after_each_cell(_Result(error_in_exec=error))

    assert json.loads(raised.read_text(encoding="utf-8")) == recorded


def test_the_report_holds_the_versions_and_main_s_json_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An example that never imported jax ran on the CPU, whatever the host has.
    monkeypatch.delitem(sys.modules, "jax", raising=False)
    report = tmp_path / "report.json"

    write_report({MAIN_RETURNED: {"count": 5}}, str(report), distribution=CONFIG.distribution)

    written = json.loads(report.read_text(encoding="utf-8"))
    assert written["returned"] == {"count": 5}
    assert written["device"] == "cpu"
    assert "python" in written["versions"]


def test_a_value_json_cannot_hold_is_reported_naming_its_key(tmp_path: Path) -> None:
    report = tmp_path / "report.json"

    write_report(
        {MAIN_RETURNED: {"weights": object()}}, str(report), distribution=CONFIG.distribution
    )

    written = json.loads(report.read_text(encoding="utf-8"))
    assert "returned" not in written
    assert written["returned_error"].startswith("main() returned 'weights', which JSON cannot hold")


@pytest.mark.parametrize(
    ("source", "kept"),
    [
        pytest.param("if __name__ == '__main__':\n    main()\n", True, id="the-guard"),
        pytest.param("if '__main__' == __name__:\n    main()\n", True, id="the-guard-reversed"),
        pytest.param(
            "if __name__ == '__main__':\n    main(5)\n", False, id="a-call-with-arguments"
        ),
        pytest.param("main()\n", False, id="no-guard"),
    ],
)
def test_the_guard_s_call_keeps_main_s_value(source: str, kept: bool) -> None:
    module = MainResult().visit(ast.parse(source))
    namespace: dict[str, object] = {"__name__": "__main__", "main": lambda *args: {"seen": args}}

    # The transformed cell, as the kernel runs it.
    exec(compile(module, "<example>", "exec"), namespace)  # noqa: S102

    assert (MAIN_RETURNED in namespace) is kept


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
    ("body", "expected"),
    [
        pytest.param("main()", True, id="main-alone"),
        pytest.param("main(5)", False, id="an-argument"),
        pytest.param("main(steps=5)", False, id="a-keyword"),
        pytest.param("main()\n    print('done')", False, id="more-than-main"),
        pytest.param("run()", False, id="another-function"),
    ],
)
def test_the_guard_s_call_is_main_alone(body: str, expected: bool) -> None:
    node = ast.parse(f"if __name__ == '__main__':\n    {body}\n").body[0]

    assert is_guarded_main_call(node) is expected


_SEES_ITS_MODULES = GOOD.replace(
    "def main() -> None:\n    print(sort_records(make_records()))",
    "def main() -> dict[str, object]:\n"
    "    import sys\n\n"
    "    print(sort_records(make_records()))\n"
    "    loaded = (name for name in sys.modules if name.split('.')[0] == 'substrax')\n"
    '    return {"substrax": sorted(loaded)}',
)


def test_the_example_s_kernel_never_imports_substrax(tmp_path: Path) -> None:
    """The setup is sent as source, so the example's process holds a script run's modules."""
    assert _SEES_ITS_MODULES != GOOD
    repo, script = write_example(tmp_path, _SEES_ITS_MODULES)

    budget = CONFIG.budget_seconds

    run = execute_example(
        script, repo_root=repo, distribution=CONFIG.distribution, budget_seconds=budget
    )

    assert run.returned == {"substrax": []}


@pytest.mark.parametrize("declared", ["nbformat", "NBFormat"])
def test_the_report_holds_the_installed_version_of_the_repository_s_distribution(
    tmp_path: Path, declared: str
) -> None:
    """The name as ``[project]`` declares it; installed metadata matches it normalised."""
    report = tmp_path / "report.json"

    write_report({MAIN_RETURNED: None}, str(report), distribution=declared)

    written = json.loads(report.read_text(encoding="utf-8"))
    assert written["versions"][declared] == metadata.version("nbformat")


def test_a_distribution_not_installed_is_left_out_as_jax_would_be(tmp_path: Path) -> None:
    report = tmp_path / "report.json"

    write_report({MAIN_RETURNED: None}, str(report), distribution="not-installed-demo")

    written = json.loads(report.read_text(encoding="utf-8"))
    assert "not-installed-demo" not in written["versions"]
    assert "python" in written["versions"]
