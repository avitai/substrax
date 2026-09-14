"""``run_example`` runs one example per child interpreter, and ``discover_examples`` lists examples.

The examples written here import no jax, so each child starts in well under a second. Every run
states a generous budget except the timeout test's, whose example sleeps far past its own.
"""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path

import pytest

import substrax.testing.examples
from substrax.artifacts import OUTPUT_DIR_ENV
from substrax.testing import (
    ChildFailedError,
    discover_examples,
    ExampleTimeoutError,
    run_example,
    unavailable_reason,
)


_BUDGET = 120.0


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repository root holding an empty ``examples`` directory."""
    (tmp_path / "repo" / "examples").mkdir(parents=True)
    return tmp_path / "repo"


@pytest.fixture
def outputs(tmp_path: Path) -> Path:
    """The directory a run's outputs are redirected to."""
    return tmp_path / "outputs"


def _example(repo: Path, relative: str, source: str) -> Path:
    path = repo / "examples" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return path


def _files(root: Path) -> dict[str, bytes]:
    # Bytecode caches are left out: repositories ignore them, and importing a helper writes them.
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    }


def test_calling_main_returns_its_summary_with_array_scalars_as_python_numbers(
    repo: Path, outputs: Path
) -> None:
    path = _example(
        repo,
        "operators/fno.py",
        """
        import numpy as np

        def main():
            return {"loss": np.float32(0.25), "steps": np.int64(3), "label": "fno"}

        if __name__ == "__main__":
            raise SystemExit("loaded as __main__")
        """,
    )

    run = run_example(path, repo_root=repo, output_dir=outputs, timeout=_BUDGET, call_main=True)

    assert run.result.check().returncode == 0
    assert run.summary == {"loss": 0.25, "steps": 3, "label": "fno"}


def test_a_summary_that_is_not_a_mapping_is_decoded_too(repo: Path, outputs: Path) -> None:
    path = _example(
        repo,
        "sweep.py",
        """
        import numpy as np

        def main():
            return [np.float32(1.5), None]
        """,
    )

    run = run_example(path, repo_root=repo, output_dir=outputs, timeout=_BUDGET, call_main=True)

    assert run.summary == [1.5, None]


def test_running_as_a_script_executes_the_main_block_and_has_no_summary(
    repo: Path, outputs: Path
) -> None:
    path = _example(repo, "flat.py", 'print("ran as", __name__)\n')

    run = run_example(path, repo_root=repo, output_dir=outputs, timeout=_BUDGET, call_main=False)

    assert "ran as __main__" in run.result.check().stdout
    assert run.summary is None


def test_a_failing_example_is_returned_and_check_reports_its_stderr(
    repo: Path, outputs: Path
) -> None:
    path = _example(
        repo,
        "diverges.py",
        """
        import sys

        def main():
            sys.stderr.write("the solver diverged\\n")
            raise SystemExit(1)
        """,
    )

    run = run_example(path, repo_root=repo, output_dir=outputs, timeout=_BUDGET, call_main=True)

    assert (run.result.returncode, run.summary) == (1, None)
    with pytest.raises(ChildFailedError, match="the solver diverged"):
        run.result.check()


def test_an_example_past_its_budget_fails_with_the_budget_and_its_partial_stderr(
    repo: Path, outputs: Path
) -> None:
    path = _example(
        repo,
        "slow.py",
        """
        import sys
        import time

        def main():
            sys.stderr.write("started the long loop\\n")
            sys.stderr.flush()
            time.sleep(60)
        """,
    )

    with pytest.raises(ExampleTimeoutError, match=r"slow\.py exceeded its 3 s budget") as caught:
        run_example(path, repo_root=repo, output_dir=outputs, timeout=3.0, call_main=True)

    assert caught.value.timeout == 3.0
    assert "started the long loop" in caught.value.stderr_tail


def test_a_silent_example_past_its_budget_fails_with_an_empty_stderr_tail(
    repo: Path, outputs: Path
) -> None:
    """``subprocess`` reports no output at all, not empty bytes, for a child that wrote nothing."""
    path = _example(repo, "stalls.py", "import time\n\ntime.sleep(60)\n")

    with pytest.raises(ExampleTimeoutError, match=r"stalls\.py exceeded its 2 s budget") as caught:
        run_example(path, repo_root=repo, output_dir=outputs, timeout=2.0, call_main=False)

    assert caught.value.stderr_tail == ""


def test_logging_configured_at_import_stays_in_the_child(repo: Path, outputs: Path) -> None:
    path = _example(
        repo,
        "noisy.py",
        """
        import logging

        logging.basicConfig(level=logging.DEBUG, force=True)

        def main():
            return {"handlers": len(logging.getLogger().handlers)}
        """,
    )
    handlers = list(logging.getLogger().handlers)

    run = run_example(path, repo_root=repo, output_dir=outputs, timeout=_BUDGET, call_main=True)

    assert run.summary == {"handlers": 1}
    assert logging.getLogger().handlers == handlers


def test_outputs_land_in_the_output_directory_and_the_repository_is_unchanged(
    repo: Path, outputs: Path
) -> None:
    path = _example(
        repo,
        "plots.py",
        """
        from substrax.artifacts import resolve_output_dir

        def main():
            location = resolve_output_dir("figures")
            (location.path / "plot.txt").write_text("plot")
            return {"source": location.source}
        """,
    )
    before = _files(repo)

    run = run_example(path, repo_root=repo, output_dir=outputs, timeout=_BUDGET, call_main=True)

    assert run.summary == {"source": "environment"}
    assert (outputs / "figures" / "plot.txt").read_text() == "plot"
    assert _files(repo) == before


def test_the_repository_check_sees_an_example_writing_into_its_working_directory(
    repo: Path, outputs: Path
) -> None:
    """Control: the unchanged-repository check must catch a write into the repository root."""
    path = _example(
        repo,
        "writes_docs.py",
        """
        from pathlib import Path

        def main():
            target = Path("docs/assets/plot.txt")
            target.parent.mkdir(parents=True)
            target.write_text("plot")
            return {}
        """,
    )
    before = _files(repo)

    run_example(path, repo_root=repo, output_dir=outputs, timeout=_BUDGET, call_main=True)

    assert set(_files(repo)) - set(before) == {"docs/assets/plot.txt"}


@pytest.mark.parametrize("call_main", [True, False], ids=["main", "script"])
def test_an_example_imports_a_helper_beside_it_in_both_modes(
    repo: Path, outputs: Path, call_main: bool
) -> None:
    _example(repo, "pde/_shared.py", "SCALE = 3\n")
    path = _example(
        repo,
        "pde/uses_helper.py",
        """
        import json

        from _shared import SCALE

        def main():
            return {"scale": SCALE}

        if __name__ == "__main__":
            print(json.dumps(main()))
        """,
    )

    run = run_example(
        path, repo_root=repo, output_dir=outputs, timeout=_BUDGET, call_main=call_main
    )

    assert run.result.check().last_json() == {"scale": 3}


@pytest.mark.parametrize("call_main", [True, False], ids=["main", "script"])
def test_safe_path_hides_the_helper_in_both_modes(
    repo: Path, outputs: Path, call_main: bool
) -> None:
    _example(repo, "pde/_shared.py", "SCALE = 3\n")
    path = _example(
        repo,
        "pde/uses_helper.py",
        """
        from _shared import SCALE

        def main():
            return {"scale": SCALE}

        if __name__ == "__main__":
            main()
        """,
    )

    run = run_example(
        path,
        repo_root=repo,
        output_dir=outputs,
        timeout=_BUDGET,
        call_main=call_main,
        env={"PYTHONSAFEPATH": "1"},
    )

    assert run.result.returncode != 0
    assert "ModuleNotFoundError: No module named '_shared'" in run.result.stderr


@pytest.mark.parametrize("call_main", [True, False], ids=["main", "script"])
def test_an_example_sees_no_command_line_arguments_in_either_mode(
    repo: Path, outputs: Path, call_main: bool
) -> None:
    path = _example(
        repo,
        "reads_argv.py",
        """
        import json
        import sys

        def main():
            return {"arguments": sys.argv[1:]}

        if __name__ == "__main__":
            print(json.dumps(main()))
        """,
    )

    run = run_example(
        path, repo_root=repo, output_dir=outputs, timeout=_BUDGET, call_main=call_main
    )

    assert run.result.check().last_json() == {"arguments": []}


@pytest.mark.parametrize("call_main", [True, False], ids=["main", "script"])
def test_a_module_at_the_repository_root_is_not_importable_in_either_mode(
    repo: Path, outputs: Path, call_main: bool
) -> None:
    (repo / "at_root.py").write_text("VALUE = 1\n", encoding="utf-8")
    path = _example(
        repo,
        "uses_root.py",
        """
        import at_root

        def main():
            return {}

        if __name__ == "__main__":
            main()
        """,
    )

    run = run_example(
        path, repo_root=repo, output_dir=outputs, timeout=_BUDGET, call_main=call_main
    )

    assert "ModuleNotFoundError: No module named 'at_root'" in run.result.stderr


def test_the_runner_directory_is_replaced_by_the_example_directory_on_the_path(
    repo: Path, outputs: Path
) -> None:
    """A script run shows its own directory first; the loader's directory must not stay visible."""
    path = _example(
        repo,
        "pde/reads_path.py",
        """
        import sys

        def main():
            return {"path": sys.path}
        """,
    )

    run = run_example(path, repo_root=repo, output_dir=outputs, timeout=_BUDGET, call_main=True)

    assert run.summary["path"][0] == str(path.parent.resolve())
    assert str(Path(substrax.testing.examples.__file__).parent) not in run.summary["path"]


def test_calling_main_on_a_file_that_is_not_a_module_fails_naming_it(
    repo: Path, outputs: Path
) -> None:
    path = _example(repo, "notes.txt", "not python\n")

    run = run_example(path, repo_root=repo, output_dir=outputs, timeout=_BUDGET, call_main=True)

    with pytest.raises(ChildFailedError, match=r"cannot load .*notes\.txt as a module"):
        run.result.check()


def test_a_dataclass_with_string_annotations_loads(repo: Path, outputs: Path) -> None:
    """The loader registers the module in ``sys.modules``; without it ``dataclasses`` raises."""
    path = _example(
        repo,
        "config.py",
        """
        from __future__ import annotations

        from dataclasses import InitVar, dataclass

        @dataclass
        class Summary:
            loss: float
            scale: InitVar[float] = 1.0

        def main():
            return {"loss": Summary(0.5).loss}
        """,
    )

    run = run_example(path, repo_root=repo, output_dir=outputs, timeout=_BUDGET, call_main=True)

    assert run.summary == {"loss": 0.5}


def test_a_different_output_variable_in_env_is_refused(
    repo: Path, outputs: Path, tmp_path: Path
) -> None:
    path = _example(repo, "flat.py", "pass\n")

    with pytest.raises(ValueError, match=OUTPUT_DIR_ENV):
        run_example(
            path,
            repo_root=repo,
            output_dir=outputs,
            timeout=_BUDGET,
            call_main=False,
            env={OUTPUT_DIR_ENV: str(tmp_path / "elsewhere")},
        )


def test_a_summary_value_json_cannot_hold_fails_naming_its_key(repo: Path, outputs: Path) -> None:
    path = _example(
        repo,
        "arrays.py",
        """
        import numpy as np

        def main():
            return {"loss": 0.5, "weights": [np.ones(2), np.zeros(2)]}
        """,
    )

    run = run_example(path, repo_root=repo, output_dir=outputs, timeout=_BUDGET, call_main=True)

    assert run.summary is None
    with pytest.raises(ChildFailedError, match="'weights'"):
        run.result.check()


def test_discovery_skips_private_parts_and_package_markers(tmp_path: Path) -> None:
    root = tmp_path / "_site" / "examples"
    for relative in (
        "a/01_first.py",
        "a/__init__.py",
        "_templates/template.py",
        "_private.py",
        "b/helpers/_shared.py",
        "b/02_second.py",
        "b/notes.txt",
    ):
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text("", encoding="utf-8")

    found = discover_examples(root)
    narrowed = discover_examples(root, include=lambda path: path.name.startswith("02_"))

    assert [path.relative_to(root).as_posix() for path in found] == [
        "a/01_first.py",
        "b/02_second.py",
    ]
    assert [path.relative_to(root).as_posix() for path in narrowed] == ["b/02_second.py"]


def test_unavailable_reason_matches_only_the_stderr_of_a_failed_run(
    repo: Path, outputs: Path
) -> None:
    failing = _example(
        repo,
        "download.py",
        """
        import sys

        print("network unreachable")
        sys.stderr.write("HTTPError: 429 Client Error: Too Many Requests\\n")
        raise SystemExit(1)
        """,
    )
    succeeding = _example(
        repo,
        "warns.py",
        'import sys\nsys.stderr.write("429 Client Error, retried\\n")\n',
    )

    failed = run_example(
        failing, repo_root=repo, output_dir=outputs, timeout=_BUDGET, call_main=False
    )
    passed = run_example(
        succeeding, repo_root=repo, output_dir=outputs, timeout=_BUDGET, call_main=False
    )

    assert (
        unavailable_reason(failed, ["network unreachable", "429 client error"])
        == "429 client error"
    )
    assert unavailable_reason(failed, ["network unreachable"]) is None
    assert unavailable_reason(passed, ["429 client error"]) is None
