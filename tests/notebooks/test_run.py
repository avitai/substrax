"""``substrax.notebooks.run``: an example run as its clean notebook, and the run's provenance."""

from __future__ import annotations

import hashlib
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from importlib import metadata
from pathlib import Path
from typing import TypedDict

import nbformat
import pydantic
import pytest
from _example_texts import CONFIG, GOOD, returning_environment, write_example
from nbconvert.preprocessors import CoalesceStreamsPreprocessor

from substrax.notebooks.pairs import convert
from substrax.notebooks.run import (
    coalesce_streams,
    ExampleRun,
    execute_example,
    NativeKernelManager,
    Provenance,
    RunError,
    script_sha256,
)
from substrax.notebooks.text import NotUtf8Error
from substrax.typing import JsonValue


def test_the_kernel_talks_over_ipc_not_plain_tcp() -> None:
    assert NativeKernelManager().transport == "ipc"


def test_a_socket_path_stays_inside_the_unix_limit_under_a_deep_runtime_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    deep = tmp_path.joinpath(*["nested-runtime-directory"] * 6)
    monkeypatch.setenv("JUPYTER_RUNTIME_DIR", str(deep))
    assert len(str(deep)) > 107

    socket = NativeKernelManager().ip

    assert len(f"{socket}-9") <= 107
    assert Path(socket).is_absolute()


def test_an_example_runs_under_a_deep_runtime_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("JUPYTER_RUNTIME_DIR", str(tmp_path.joinpath(*["nested-runtime-dir"] * 6)))
    repo, script = write_example(tmp_path)

    run = execute_example(
        script,
        repo_root=repo,
        distribution=CONFIG.distribution,
        budget_seconds=CONFIG.budget_seconds,
    )

    assert run.provenance.outputs


def test_a_run_records_each_code_cells_output_kinds_and_the_script(tmp_path: Path) -> None:
    repo, script = write_example(tmp_path)

    run = execute_example(
        script,
        repo_root=repo,
        distribution=CONFIG.distribution,
        budget_seconds=CONFIG.budget_seconds,
    )

    code_cells = [cell for cell in run.notebook.cells if cell.cell_type == "code"]
    assert run.provenance.source == "examples/sorting.py"
    assert run.provenance.sha256 == hashlib.sha256(script.read_bytes()).hexdigest()
    assert run.provenance.sha256 == script_sha256(script)
    assert len(run.provenance.outputs) == len(code_cells)
    assert run.provenance.outputs[-1] == ("text",)


def test_a_failing_cell_is_a_run_error_naming_its_message(tmp_path: Path) -> None:
    repo, script = write_example(
        tmp_path, GOOD.replace("return sorted(records)", 'raise RuntimeError("broken step")')
    )

    with pytest.raises(RunError, match="broken step"):
        execute_example(
            script,
            repo_root=repo,
            distribution=CONFIG.distribution,
            budget_seconds=CONFIG.budget_seconds,
        )


def test_a_cell_tagged_to_raise_stops_the_run_as_a_script_run_stops(tmp_path: Path) -> None:
    """nbclient's ``raises-exception`` tag lets a cell fail and the run go on; a script cannot."""
    guard = '# %%\nif __name__ == "__main__":\n    main()\n'
    assert guard in GOOD
    tagged = '# %% tags=["raises-exception"]\nraise ValueError("a tagged failure")\n\n\n' + guard
    repo, script = write_example(tmp_path, GOOD.replace(guard, tagged))

    with pytest.raises(RunError) as raised:
        execute_example(
            script,
            repo_root=repo,
            distribution=CONFIG.distribution,
            budget_seconds=CONFIG.budget_seconds,
        )

    assert raised.value.raised == "ValueError: a tagged failure"
    assert raised.value.raised_class == "builtins.ValueError"


def test_a_run_past_its_budget_fails_naming_the_budget(tmp_path: Path) -> None:
    # Each cell stays inside the budget; together they pass it.
    text = GOOD.replace(
        '# %% [markdown]\n"""\n## Results',
        "# %%\nimport time\n\ntime.sleep(1.5)\n\n# %%\ntime.sleep(1.5)\n\n\n"
        '# %% [markdown]\n"""\n## Results',
    )
    assert text != GOOD
    repo, script = write_example(tmp_path, text)

    with pytest.raises(RunError, match=r"budget of 2\.5 s"):
        execute_example(
            script, repo_root=repo, distribution=CONFIG.distribution, budget_seconds=2.5
        )


def test_a_run_that_writes_into_the_repository_fails_naming_the_path(tmp_path: Path) -> None:
    text = GOOD.replace(
        "    print(sort_records(make_records()))",
        "    import pathlib\n\n"
        '    pathlib.Path("examples_output").mkdir()\n'
        '    pathlib.Path("examples_output", "x.txt").write_text("x")\n'
        "    print(sort_records(make_records()))",
    )
    assert text != GOOD
    repo, script = write_example(tmp_path, text)

    with pytest.raises(RunError, match="examples_output"):
        execute_example(
            script,
            repo_root=repo,
            distribution=CONFIG.distribution,
            budget_seconds=CONFIG.budget_seconds,
        )


def test_a_run_that_writes_into_a_tracked_file_fails(tmp_path: Path) -> None:
    text = GOOD.replace(
        "    print(sort_records(make_records()))",
        '    open("pyproject.toml", "a").write("# x\\n")\n    print(sort_records(make_records()))',
    )
    repo, script = write_example(tmp_path, text)

    with pytest.raises(RunError, match=r"pyproject\.toml"):
        execute_example(
            script,
            repo_root=repo,
            distribution=CONFIG.distribution,
            budget_seconds=CONFIG.budget_seconds,
        )


def test_a_write_into_a_file_already_changed_fails_naming_it(tmp_path: Path) -> None:
    repo, script = write_example(
        tmp_path, _endless('with open("pyproject.toml", "a") as out:\n        out.write("# x\\n")')
    )
    with (repo / "pyproject.toml").open("a", encoding="utf-8") as out:
        out.write("# changed before the run\n")

    with pytest.raises(RunError, match=r"changed while the run was in progress: pyproject\.toml"):
        execute_example(
            script,
            repo_root=repo,
            distribution=CONFIG.distribution,
            budget_seconds=CONFIG.budget_seconds,
        )


def test_a_new_file_in_a_directory_already_untracked_fails_naming_it(tmp_path: Path) -> None:
    repo, script = write_example(
        tmp_path, _endless('with open("scratch/new.txt", "w") as out:\n        out.write("x")')
    )
    (repo / "scratch").mkdir()
    (repo / "scratch" / "notes.txt").write_text("before the run\n", encoding="utf-8")

    with pytest.raises(RunError, match=r"changed while the run was in progress: scratch/new\.txt"):
        execute_example(
            script,
            repo_root=repo,
            distribution=CONFIG.distribution,
            budget_seconds=CONFIG.budget_seconds,
        )


def test_a_write_into_an_ignored_path_is_not_a_write_into_the_repository(tmp_path: Path) -> None:
    # A script run writes __pycache__ beside a helper it imports; git ignores it, and so does the
    # check.
    repo, script = write_example(
        tmp_path,
        _endless(
            'import os\n\n    os.makedirs("__pycache__", exist_ok=True)\n'
            '    with open("__pycache__/cache.bin", "w") as out:\n        out.write("x")'
        ),
    )

    run = execute_example(
        script,
        repo_root=repo,
        distribution=CONFIG.distribution,
        budget_seconds=CONFIG.budget_seconds,
    )

    assert run.provenance.outputs


def test_an_example_outside_a_git_repository_is_refused(tmp_path: Path) -> None:
    repo, script = write_example(tmp_path)
    shutil.rmtree(repo / ".git")

    with pytest.raises(RunError, match="git"):
        execute_example(
            script,
            repo_root=repo,
            distribution=CONFIG.distribution,
            budget_seconds=CONFIG.budget_seconds,
        )


_RETURNING = GOOD.replace(
    "def main() -> None:\n    print(sort_records(make_records()))",
    "def main() -> dict[str, object]:\n"
    "    import numpy as np\n\n"
    "    records = sort_records(make_records())\n"
    "    print(records)\n"
    '    return {"count": len(records), "mean": np.float32(sum(records) / len(records))}',
)


def test_the_run_reports_what_main_returned(tmp_path: Path) -> None:
    assert _RETURNING != GOOD
    repo, script = write_example(tmp_path, _RETURNING)

    run = execute_example(
        script,
        repo_root=repo,
        distribution=CONFIG.distribution,
        budget_seconds=CONFIG.budget_seconds,
    )

    assert run.returned == {"count": 5, "mean": 2.0}


def test_the_page_notebook_keeps_the_authored_guard(tmp_path: Path) -> None:
    repo, script = write_example(tmp_path, _RETURNING)

    run = execute_example(
        script,
        repo_root=repo,
        distribution=CONFIG.distribution,
        budget_seconds=CONFIG.budget_seconds,
    )

    sources = [cell.source for cell in run.notebook.cells]
    assert sources[-1] == 'if __name__ == "__main__":\n    main()'
    assert not any("_substrax" in source for source in sources)


def test_main_returning_nothing_reports_none(tmp_path: Path) -> None:
    repo, script = write_example(tmp_path)

    run = execute_example(
        script,
        repo_root=repo,
        distribution=CONFIG.distribution,
        budget_seconds=CONFIG.budget_seconds,
    )

    assert run.returned is None


def test_a_returned_value_json_cannot_hold_fails_naming_its_key(tmp_path: Path) -> None:
    text = _RETURNING.replace(
        '"mean": np.float32(sum(records) / len(records))', '"records": np.asarray(records)'
    )
    assert text != _RETURNING
    repo, script = write_example(tmp_path, text)

    with pytest.raises(RunError, match="'records'"):
        execute_example(
            script,
            repo_root=repo,
            distribution=CONFIG.distribution,
            budget_seconds=CONFIG.budget_seconds,
        )


def test_the_users_ipython_startup_files_do_not_run_in_the_example(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A startup file runs before the first cell, so its print reaches no cell; a name it
    # defines does.
    startup = tmp_path / "home" / ".ipython" / "profile_default" / "startup"
    startup.mkdir(parents=True)
    (startup / "00-flag.py").write_text("STARTUP_RAN = True\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("IPYTHONDIR", raising=False)
    text = GOOD.replace(
        "    print(sort_records(make_records()))",
        '    print("startup ran:", "STARTUP_RAN" in globals())\n'
        "    print(sort_records(make_records()))",
    )
    repo, script = write_example(tmp_path, text)

    run = execute_example(
        script,
        repo_root=repo,
        distribution=CONFIG.distribution,
        budget_seconds=CONFIG.budget_seconds,
    )

    printed = "".join(
        output.get("text", "")
        for cell in run.notebook.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
    )
    assert "startup ran: False" in printed


def _endless(statement: str) -> str:
    return GOOD.replace("    print(sort_records(make_records()))", f"    {statement}")


def test_a_cell_that_never_ends_is_stopped_at_the_budget(tmp_path: Path) -> None:
    repo, script = write_example(tmp_path, _endless("while True:\n        pass"))

    started = time.perf_counter()
    with pytest.raises(RunError, match=r"budget of 3\.0 s"):
        execute_example(
            script, repo_root=repo, distribution=CONFIG.distribution, budget_seconds=3.0
        )

    # Kernel shutdown comes on top of the budget; a run that waited for the loop would never end,
    # and pytest's own timeout would stop it instead.
    assert time.perf_counter() - started < 60


def test_a_kernel_that_dies_ends_the_run(tmp_path: Path) -> None:
    repo, script = write_example(tmp_path, _endless("import os\n\n    os._exit(3)"))

    started = time.perf_counter()
    with pytest.raises(RunError, match="kernel"):
        execute_example(
            script,
            repo_root=repo,
            distribution=CONFIG.distribution,
            budget_seconds=CONFIG.budget_seconds,
        )

    assert time.perf_counter() - started < 60


@pytest.mark.parametrize("budget", [math.inf, math.nan])
def test_a_budget_that_is_not_a_finite_number_is_refused(tmp_path: Path, budget: float) -> None:
    repo, script = write_example(tmp_path)

    with pytest.raises(ValueError, match="budget"):
        execute_example(
            script, repo_root=repo, distribution=CONFIG.distribution, budget_seconds=budget
        )


@pytest.mark.parametrize("budget", [1e12, 2.0**31 / 1000 + 1])
def test_a_huge_finite_budget_runs(tmp_path: Path, budget: float) -> None:
    """A finite budget above zero is usable however large; past 2**31 ms included."""
    repo, script = write_example(tmp_path)

    run = execute_example(
        script, repo_root=repo, distribution=CONFIG.distribution, budget_seconds=budget
    )

    assert run.provenance.seconds < budget


_RUN_IN_A_CHILD = """\
import sys
from pathlib import Path

from substrax.notebooks.run import execute_example

execute_example(
    Path(sys.argv[1]), repo_root=Path(sys.argv[2]), distribution="demo", budget_seconds=600.0
)
"""


def _wait_for(predicate: Callable[[], bool], seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.2)
    return predicate()


def _is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.mark.skipif(sys.platform == "win32", reason="a terminal interrupt is SIGINT on POSIX only")
def test_an_interrupt_stops_the_run_and_shuts_its_kernel_down(tmp_path: Path) -> None:
    marker = tmp_path / "kernel.pid"
    repo, script = write_example(
        tmp_path,
        _endless(
            f"import os\n    import time\n\n    with open({str(marker)!r}, 'w') as out:\n"
            "        out.write(str(os.getpid()))\n    time.sleep(600)"
        ),
    )
    child = subprocess.Popen(  # noqa: S603  # nosec B603 - this interpreter with a fixed program
        [sys.executable, "-c", _RUN_IN_A_CHILD, str(script), str(repo)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert _wait_for(lambda: marker.exists() and marker.read_text() != "", 120)
        kernel = int(marker.read_text())

        child.send_signal(signal.SIGINT)
        _, stderr = child.communicate(timeout=30)
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate()

    assert "KeyboardInterrupt" in stderr
    assert _wait_for(lambda: not _is_running(kernel), 15), f"kernel {kernel} outlived the run"


def test_text_printed_at_two_moments_is_one_output(tmp_path: Path) -> None:
    repo, script = write_example(
        tmp_path,
        _endless(
            'import time\n\n    print("first", flush=True)\n    time.sleep(1)\n    print("second")'
        ),
    )

    run = execute_example(
        script,
        repo_root=repo,
        distribution=CONFIG.distribution,
        budget_seconds=CONFIG.budget_seconds,
    )

    last = [cell for cell in run.notebook.cells if cell.cell_type == "code"][-1]
    assert run.provenance.outputs[-1] == ("text",)
    assert last.outputs[0].text == "first\nsecond\n"


def test_text_printed_around_a_warning_is_one_output_kind(tmp_path: Path) -> None:
    """A library's warning, which one machine prints and another does not, changes no kind.

    The page keeps the warning where it was written; the kinds the page check compares leave
    standard error out before joining the text on either side of it.
    """
    repo, script = write_example(
        tmp_path,
        _endless(
            "import sys\n    import time\n\n"
            '    print("first", flush=True)\n'
            '    print("a warning", file=sys.stderr, flush=True)\n'
            "    time.sleep(1)\n"
            '    print("second")'
        ),
    )

    run = execute_example(
        script,
        repo_root=repo,
        distribution=CONFIG.distribution,
        budget_seconds=CONFIG.budget_seconds,
    )

    last = [cell for cell in run.notebook.cells if cell.cell_type == "code"][-1]
    assert run.provenance.outputs[-1] == ("text",)
    assert [output.name for output in last.outputs] == ["stdout", "stderr", "stdout"]


_NOISY_JAX = """\
import sys
    import types

    def devices() -> list[types.SimpleNamespace]:
        print("a warning from the backend", file=sys.stderr)
        return [types.SimpleNamespace(platform="cpu", device_kind="cpu")]

    sys.modules["jax"] = types.SimpleNamespace(devices=devices)"""


def test_what_the_devices_print_to_standard_error_is_not_read_as_the_report(
    tmp_path: Path,
) -> None:
    repo, script = write_example(tmp_path, _endless(_NOISY_JAX))

    run = execute_example(
        script,
        repo_root=repo,
        distribution=CONFIG.distribution,
        budget_seconds=CONFIG.budget_seconds,
    )

    assert run.provenance.device == "1 cpu"


def test_an_example_that_redirects_stdout_still_has_its_run_reported(tmp_path: Path) -> None:
    # The report travels in a file the run owns, not the example's standard output.
    repo, script = write_example(
        tmp_path, _endless('import os\n    import sys\n\n    sys.stdout = open(os.devnull, "w")')
    )

    run = execute_example(
        script,
        repo_root=repo,
        distribution=CONFIG.distribution,
        budget_seconds=CONFIG.budget_seconds,
    )

    assert run.provenance.device


def test_an_unwritable_runtime_directory_falls_back_to_the_temporary_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    locked = tmp_path / "locked"
    locked.mkdir()
    monkeypatch.setenv("JUPYTER_RUNTIME_DIR", str(locked / "runtime"))
    locked.chmod(0o500)
    try:
        socket = NativeKernelManager().ip
    finally:
        locked.chmod(0o700)

    assert Path(socket).parent == Path(tempfile.gettempdir())


def test_a_run_error_carries_the_raised_exceptions_type_and_message_alone(tmp_path: Path) -> None:
    repo, script = write_example(
        tmp_path, GOOD.replace("return sorted(records)", 'raise RuntimeError("broken step")')
    )

    with pytest.raises(RunError) as raised:
        execute_example(
            script,
            repo_root=repo,
            distribution=CONFIG.distribution,
            budget_seconds=CONFIG.budget_seconds,
        )

    assert raised.value.raised == "RuntimeError: broken step"
    assert raised.value.raised_class == "builtins.RuntimeError"


_SPOILS_THE_RECORD = {
    "bytes-that-are-not-utf-8": (
        "    get_ipython().events.callbacks['post_run_cell'].clear()\n"
        "    with open(record, 'wb') as out:\n        out.write(b'\\xff\\xfe')\n"
    ),
    "a-directory-in-its-place": "    os.mkdir(record)\n",
}


@pytest.mark.parametrize("spoiling", list(_SPOILS_THE_RECORD.values()), ids=_SPOILS_THE_RECORD)
def test_a_record_the_example_spoiled_leaves_the_class_unknown(
    tmp_path: Path, spoiling: str
) -> None:
    """The run's directory is the example's to reach; an unreadable record names no class."""
    body = "    print(sort_records(make_records()))"
    assert body in GOOD
    reaches_the_record = (
        "    import os\n\n"
        "    record = os.path.join(os.environ['AVITAI_OUTPUT_DIR'], '..', 'raised.json')\n"
    )
    repo, script = write_example(
        tmp_path, GOOD.replace(body, reaches_the_record + spoiling + "    raise ValueError('boom')")
    )

    with pytest.raises(RunError) as raised:
        execute_example(
            script,
            repo_root=repo,
            distribution=CONFIG.distribution,
            budget_seconds=CONFIG.budget_seconds,
        )

    assert raised.value.raised == "ValueError: boom"
    assert raised.value.raised_class is None


def test_a_run_error_with_no_exception_from_the_example_carries_none(tmp_path: Path) -> None:
    repo, script = write_example(tmp_path, _endless("while True:\n        pass"))

    with pytest.raises(RunError) as raised:
        execute_example(
            script, repo_root=repo, distribution=CONFIG.distribution, budget_seconds=3.0
        )

    assert raised.value.raised is None
    assert raised.value.raised_class is None


def _printed(run: ExampleRun) -> str:
    last = [cell for cell in run.notebook.cells if cell.cell_type == "code"][-1]
    return "".join(output.get("text", "") for output in last.outputs)


def test_a_helper_beside_the_example_imports_as_it_does_in_a_script_run(tmp_path: Path) -> None:
    repo, script = write_example(
        tmp_path, _endless("from _shout import shout\n\n    print(shout('sorted'))")
    )
    (script.parent / "_shout.py").write_text(
        "def shout(word: str) -> str:\n    return word.upper()\n", encoding="utf-8"
    )

    run = execute_example(
        script,
        repo_root=repo,
        distribution=CONFIG.distribution,
        budget_seconds=CONFIG.budget_seconds,
    )

    assert _printed(run) == "SORTED\n"


def test_an_import_a_script_run_cannot_make_fails_the_run_too(tmp_path: Path) -> None:
    # As a script, the repository root is not on sys.path, so `examples` is not importable.
    repo, script = write_example(tmp_path, _endless("import examples\n\n    print(examples)"))

    with pytest.raises(RunError) as raised:
        execute_example(
            script,
            repo_root=repo,
            distribution=CONFIG.distribution,
            budget_seconds=CONFIG.budget_seconds,
        )

    assert raised.value.raised == "ModuleNotFoundError: No module named 'examples'"


def test_the_example_sees_the_arguments_and_file_a_script_run_gives_it(tmp_path: Path) -> None:
    repo, script = write_example(
        tmp_path, _endless("import sys\n\n    print(sys.argv)\n    print(__file__)")
    )

    run = execute_example(
        script,
        repo_root=repo,
        distribution=CONFIG.distribution,
        budget_seconds=CONFIG.budget_seconds,
    )

    assert _printed(run) == f"['examples/sorting.py']\n{script.absolute()}\n"


def test_an_argument_parser_with_defaults_runs_as_it_does_in_a_script_run(tmp_path: Path) -> None:
    repo, script = write_example(
        tmp_path,
        _endless(
            "import argparse\n\n    parser = argparse.ArgumentParser()\n"
            '    parser.add_argument("--steps", type=int, default=3)\n'
            "    print(parser.parse_args().steps)"
        ),
    )

    run = execute_example(
        script,
        repo_root=repo,
        distribution=CONFIG.distribution,
        budget_seconds=CONFIG.budget_seconds,
    )

    assert _printed(run) == "3\n"


def test_main_exiting_with_status_zero_is_a_successful_run(tmp_path: Path) -> None:
    repo, script = write_example(tmp_path, _endless('print("done early")\n    raise SystemExit(0)'))

    run = execute_example(
        script,
        repo_root=repo,
        distribution=CONFIG.distribution,
        budget_seconds=CONFIG.budget_seconds,
    )

    assert _printed(run) == "done early\n"
    assert run.returned is None


def test_main_exiting_with_another_status_fails_the_run(tmp_path: Path) -> None:
    repo, script = write_example(tmp_path, _endless("raise SystemExit(2)"))

    with pytest.raises(RunError) as raised:
        execute_example(
            script,
            repo_root=repo,
            distribution=CONFIG.distribution,
            budget_seconds=CONFIG.budget_seconds,
        )

    assert raised.value.raised == "SystemExit: 2"


def test_a_run_error_message_carries_no_terminal_colours(tmp_path: Path) -> None:
    repo, script = write_example(
        tmp_path, GOOD.replace("return sorted(records)", 'raise RuntimeError("broken step")')
    )

    with pytest.raises(RunError) as raised:
        execute_example(
            script,
            repo_root=repo,
            distribution=CONFIG.distribution,
            budget_seconds=CONFIG.budget_seconds,
        )

    assert "\x1b" not in str(raised.value)
    assert "RuntimeError: broken step" in str(raised.value)


def test_a_run_error_message_carries_no_terminal_hyperlink(tmp_path: Path) -> None:
    link = r"\x1b]8;;https://example.org\x1b\\the docs\x1b]8;;\x1b\\"
    repo, script = write_example(
        tmp_path, GOOD.replace("return sorted(records)", f'raise RuntimeError("see {link}")')
    )

    with pytest.raises(RunError) as raised:
        execute_example(
            script,
            repo_root=repo,
            distribution=CONFIG.distribution,
            budget_seconds=CONFIG.budget_seconds,
        )

    assert "\x1b" not in str(raised.value)
    assert "RuntimeError: see the docs" in str(raised.value)
    assert raised.value.raised == "RuntimeError: see the docs"


def test_text_printed_after_a_figure_stays_after_it(tmp_path: Path) -> None:
    repo, script = write_example(
        tmp_path,
        _endless(
            "import matplotlib.pyplot as plt\n\n    print('before')\n"
            "    plt.plot([1, 2])\n    plt.show()\n    print('after')"
        ),
    )

    run = execute_example(
        script,
        repo_root=repo,
        distribution=CONFIG.distribution,
        budget_seconds=CONFIG.budget_seconds,
    )

    assert run.provenance.outputs[-1] == ("text", "figure", "text")


_CARRIAGE_RETURN_CASES = [
    "plain\n",
    "progress 10%\rprogress 50%\rdone\n",
    "a\r\nb\n",
    "first\rsecond\nthird\rfourth",
    "ends with a return\r",
    "\r\r\rx\n",
    "".join(f"\r{i}/100" for i in range(100)) + "\n",
]


@pytest.mark.parametrize("text", _CARRIAGE_RETURN_CASES)
def test_carriage_returns_resolve_as_nbconvert_resolves_them(text: str) -> None:
    reference = nbformat.v4.new_code_cell()
    reference.outputs = [nbformat.v4.new_output("stream", name="stdout", text=text)]
    expected, _ = CoalesceStreamsPreprocessor().preprocess_cell(reference, {}, 0)

    cell = nbformat.v4.new_code_cell()
    cell.outputs = [nbformat.v4.new_output("stream", name="stdout", text=text)]
    coalesce_streams(cell)

    assert cell.outputs[0].text == expected.outputs[0].text


@pytest.mark.timeout(20)
def test_a_long_line_after_a_carriage_return_resolves_in_linear_time() -> None:
    # nbconvert's pattern takes quadratic time here: 80,000 characters cost 1.6 s, so 2,000,000
    # would take some 1,000 s; a linear pass takes milliseconds.
    cell = nbformat.v4.new_code_cell()
    cell.outputs = [
        nbformat.v4.new_output("stream", name="stdout", text="progress\r" + "y" * 2_000_000)
    ]

    coalesce_streams(cell)

    assert cell.outputs[0].text == "y" * 2_000_000


def test_only_consecutive_outputs_of_one_stream_merge() -> None:
    cell = nbformat.v4.new_code_cell()
    cell.outputs = [
        nbformat.v4.new_output("stream", name="stdout", text="a"),
        nbformat.v4.new_output("stream", name="stdout", text="b"),
        nbformat.v4.new_output("display_data", data={"text/plain": "figure"}),
        nbformat.v4.new_output("stream", name="stdout", text="c"),
        nbformat.v4.new_output("stream", name="stderr", text="d"),
    ]

    coalesce_streams(cell)

    assert [(o.output_type, o.get("text")) for o in cell.outputs] == [
        ("stream", "ab"),
        ("display_data", None),
        ("stream", "c"),
        ("stream", "d"),
    ]


def _kernel_starts_with(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, code: str) -> None:
    """Run ``code`` in every new Python process, the kernel's included, through sitecustomize."""
    site = tmp_path / "site"
    site.mkdir()
    (site / "sitecustomize.py").write_text(code, encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(site), os.environ.get("PYTHONPATH", "")]))


def test_a_kernel_slower_to_start_than_the_budget_is_a_run_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, script = write_example(tmp_path)
    _kernel_starts_with(monkeypatch, tmp_path, "import time\n\ntime.sleep(4)\n")

    with pytest.raises(RunError, match="the kernel did not start"):
        execute_example(
            script, repo_root=repo, distribution=CONFIG.distribution, budget_seconds=2.0
        )


def test_a_kernel_that_dies_while_starting_is_a_run_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, script = write_example(tmp_path)
    # The kernel is the only Python process started once PYTHONPATH carries the hook.
    _kernel_starts_with(monkeypatch, tmp_path, "raise SystemExit(3)\n")

    with pytest.raises(RunError, match="the kernel did not start"):
        execute_example(
            script,
            repo_root=repo,
            distribution=CONFIG.distribution,
            budget_seconds=CONFIG.budget_seconds,
        )


def test_the_kernels_start_counts_against_the_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A 3 s start and a 2 s cell pass a 4 s budget only if the start is left out of it.
    repo, script = write_example(tmp_path, _endless("import time\n\n    time.sleep(2)"))
    _kernel_starts_with(monkeypatch, tmp_path, "import time\n\ntime.sleep(3)\n")

    with pytest.raises(RunError, match=r"budget of 4\.0 s"):
        execute_example(
            script, repo_root=repo, distribution=CONFIG.distribution, budget_seconds=4.0
        )


_GIT_IN_THE_EXAMPLE = (
    "import os\n    import subprocess\n\n"
    "    # Without the GIT_* variables a commit hook running these tests sets.\n"
    "    env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}\n"
    "    def git(*args):\n"
    "        subprocess.run(['git', '-c', 'user.name=t', '-c', 'user.email=t@e.invalid', *args],\n"
    "                       check=True, env=env)\n"
)


def test_a_run_that_deletes_a_clean_tracked_file_fails_naming_it(tmp_path: Path) -> None:
    repo, script = write_example(tmp_path, _endless('import os\n\n    os.remove("pyproject.toml")'))

    with pytest.raises(RunError, match=r"changed while the run was in progress: pyproject\.toml"):
        execute_example(
            script,
            repo_root=repo,
            distribution=CONFIG.distribution,
            budget_seconds=CONFIG.budget_seconds,
        )


def test_a_change_during_the_run_is_not_blamed_on_the_example(tmp_path: Path) -> None:
    """Another writer (an editor, a commit, a second run) changes the checkout too (M-4)."""
    repo, script = write_example(tmp_path, _endless('import os\n\n    os.remove("pyproject.toml")'))

    with pytest.raises(RunError) as raised:
        execute_example(
            script,
            repo_root=repo,
            distribution=CONFIG.distribution,
            budget_seconds=CONFIG.budget_seconds,
        )

    assert str(raised.value) == (
        f"{script}: the repository changed while the run was in progress: pyproject.toml; the "
        "check cannot tell the example's writes from another writer's, so leave the checkout "
        "unchanged while an example runs"
    )


def test_a_run_that_stages_a_new_file_fails_naming_the_index(tmp_path: Path) -> None:
    repo, script = write_example(
        tmp_path,
        _endless(
            _GIT_IN_THE_EXAMPLE
            + "    with open('staged.txt', 'w') as out:\n        out.write('x')\n"
            "    git('add', 'staged.txt')\n    os.remove('staged.txt')"
        ),
    )

    with pytest.raises(RunError, match="the index"):
        execute_example(
            script,
            repo_root=repo,
            distribution=CONFIG.distribution,
            budget_seconds=CONFIG.budget_seconds,
        )


def test_a_run_that_commits_fails_naming_head(tmp_path: Path) -> None:
    repo, script = write_example(
        tmp_path, _endless(_GIT_IN_THE_EXAMPLE + "    git('commit', '--allow-empty', '-qm', 'x')")
    )

    with pytest.raises(RunError, match="HEAD"):
        execute_example(
            script,
            repo_root=repo,
            distribution=CONFIG.distribution,
            budget_seconds=CONFIG.budget_seconds,
        )


def test_a_run_that_plants_a_git_hook_fails_naming_the_hooks(tmp_path: Path) -> None:
    repo, script = write_example(
        tmp_path,
        _endless(
            "with open('.git/hooks/pre-commit', 'w') as out:\n        out.write('#!/bin/sh\\n')"
        ),
    )

    with pytest.raises(RunError, match="hooks"):
        execute_example(
            script,
            repo_root=repo,
            distribution=CONFIG.distribution,
            budget_seconds=CONFIG.budget_seconds,
        )


def test_an_example_that_rebinds_print_still_has_its_run_reported(tmp_path: Path) -> None:
    repo, script = write_example(
        tmp_path, GOOD.replace("# %%\nimport jax\n", "# %%\nimport jax\n\nprint = repr\n")
    )

    run = execute_example(
        script,
        repo_root=repo,
        distribution=CONFIG.distribution,
        budget_seconds=CONFIG.budget_seconds,
    )

    assert run.provenance.device


def test_a_thread_still_printing_does_not_reach_the_report(tmp_path: Path) -> None:
    repo, script = write_example(
        tmp_path,
        _endless(
            "import threading\n    import time\n\n"
            "    def chatter():\n        while True:\n            print('noise', flush=True)\n"
            "            time.sleep(0.001)\n\n"
            "    threading.Thread(target=chatter, daemon=True).start()\n    time.sleep(0.05)"
        ),
    )

    run = execute_example(
        script,
        repo_root=repo,
        distribution=CONFIG.distribution,
        budget_seconds=CONFIG.budget_seconds,
    )

    assert run.provenance.device


def test_main_returning_what_json_cannot_hold_is_reported_as_the_report_not_the_example(
    tmp_path: Path,
) -> None:
    repo, script = write_example(
        tmp_path,
        GOOD.replace(
            "def main() -> None:\n    print(sort_records(make_records()))",
            "def main() -> dict[str, object]:\n    loop: dict[str, object] = {}\n"
            '    loop["self"] = loop\n    return loop',
        ),
    )

    with pytest.raises(RunError, match=r"main\(\) returned") as raised:
        execute_example(
            script,
            repo_root=repo,
            distribution=CONFIG.distribution,
            budget_seconds=CONFIG.budget_seconds,
        )

    assert raised.value.raised is None


_JAX_THAT_CANNOT_LIST_DEVICES = """\
import sys
    import types

    def devices() -> list[object]:
        raise RuntimeError("no backend")

    sys.modules["jax"] = types.SimpleNamespace(devices=devices)"""


def test_a_failure_while_reporting_the_run_is_substraxs_not_the_examples(tmp_path: Path) -> None:
    repo, script = write_example(tmp_path, _endless(_JAX_THAT_CANNOT_LIST_DEVICES))

    with pytest.raises(RunError, match="report") as raised:
        execute_example(
            script,
            repo_root=repo,
            distribution=CONFIG.distribution,
            budget_seconds=CONFIG.budget_seconds,
        )

    assert raised.value.raised is None


_JAX_SLOW_TO_LIST_DEVICES = """\
import sys
    import time
    import types

    def devices() -> list[object]:
        time.sleep(6.0)
        return [types.SimpleNamespace(platform="cpu", device_kind="cpu")]

    sys.modules["jax"] = types.SimpleNamespace(devices=devices)
    time.sleep(3.0)"""


def test_reporting_the_run_counts_against_its_one_budget(tmp_path: Path) -> None:
    # The cells take 3 s past the kernel's start and listing the devices 6 s more. Within a budget
    # of 8 s the cells end (a kernel start under 5 s), and no more than 5 s are left for the report,
    # under its 6 s; a report given a budget of its own (8 s) would end and pass.
    repo, script = write_example(tmp_path, _endless(_JAX_SLOW_TO_LIST_DEVICES))

    with pytest.raises(RunError, match=r"budget of 8\.0 s at substrax's own cell"):
        execute_example(
            script, repo_root=repo, distribution=CONFIG.distribution, budget_seconds=8.0
        )


@pytest.mark.parametrize(
    ("code", "clean"),
    [
        pytest.param("0", True, id="zero"),
        pytest.param("None", True, id="none"),
        pytest.param("False", True, id="false-is-the-integer-zero"),
        pytest.param("0.0", False, id="a-float-zero"),
        pytest.param('"done"', False, id="a-message"),
        pytest.param("1", False, id="one"),
    ],
)
def test_main_exits_cleanly_exactly_when_python_would_exit_with_status_zero(
    tmp_path: Path, code: str, clean: bool
) -> None:
    """Python exits 0 for None or an integer 0; any other code is printed and exits 1."""
    repo, script = write_example(tmp_path, _endless(f"raise SystemExit({code})"))

    if clean:
        execute_example(
            script,
            repo_root=repo,
            distribution=CONFIG.distribution,
            budget_seconds=CONFIG.budget_seconds,
        )
    else:
        with pytest.raises(RunError, match="SystemExit"):
            execute_example(
                script,
                repo_root=repo,
                distribution=CONFIG.distribution,
                budget_seconds=CONFIG.budget_seconds,
            )


def test_the_provenance_holds_the_version_of_the_repository_s_distribution(
    tmp_path: Path,
) -> None:
    repo, script = write_example(tmp_path)

    run = execute_example(
        script, repo_root=repo, distribution="nbformat", budget_seconds=CONFIG.budget_seconds
    )

    assert run.provenance.versions["nbformat"] == metadata.version("nbformat")


def test_the_kernel_has_none_of_the_variables_pytest_writes(tmp_path: Path) -> None:
    """A script run has neither, and ipykernel stops capturing a file descriptor when it sees one.

    pytest writes ``PYTEST_CURRENT_TEST`` (``_pytest/runner.py``) and ``PYTEST_VERSION``
    (``_pytest/config/__init__.py``) into its own process, which the example suite runs in.
    """
    assert "PYTEST_CURRENT_TEST" in os.environ
    assert "PYTEST_VERSION" in os.environ
    repo, script = write_example(
        tmp_path, returning_environment("PYTEST_CURRENT_TEST", "PYTEST_VERSION")
    )

    run = execute_example(
        script,
        repo_root=repo,
        distribution=CONFIG.distribution,
        budget_seconds=CONFIG.budget_seconds,
    )

    assert run.returned == {"PYTEST_CURRENT_TEST": None, "PYTEST_VERSION": None}


def _run(script: Path, repo: Path) -> ExampleRun:
    return execute_example(
        script,
        repo_root=repo,
        distribution=CONFIG.distribution,
        budget_seconds=CONFIG.budget_seconds,
    )


def test_an_example_that_is_not_utf8_is_a_run_error_naming_the_encoding(tmp_path: Path) -> None:
    repo, script = write_example(tmp_path)
    script.write_bytes(b"# caf\xe9\n" + script.read_bytes())

    with pytest.raises(RunError, match="is not UTF-8: byte 5 cannot be decoded"):
        _run(script, repo)


def test_a_run_executes_the_cells_the_clean_notebook_holds(tmp_path: Path) -> None:
    """A byte order mark is dropped as the pair drops it, so the opening cell stays prose (P-1)."""
    repo, script = write_example(tmp_path)
    script.write_bytes(b"\xef\xbb\xbf" + script.read_bytes())
    convert(script)
    paired = nbformat.read(script.with_suffix(".ipynb"), as_version=4)

    run = _run(script, repo)

    assert [cell.cell_type for cell in run.notebook.cells] == [
        cell.cell_type for cell in paired.cells
    ]
    assert run.notebook.cells[0].source == paired.cells[0].source


@pytest.mark.parametrize(
    "encode",
    [
        lambda text: text.encode("utf-8").replace(b"\n", b"\r\n"),
        lambda text: text.encode("utf-8").replace(b"\n", b"\r"),
        lambda text: b"\xef\xbb\xbf" + text.encode("utf-8"),
    ],
    ids=["windows", "carriage-return", "byte-order-mark"],
)
def test_a_script_s_identity_is_its_text_as_the_tools_read_it(
    tmp_path: Path, encode: Callable[[str], bytes]
) -> None:
    plain, other = tmp_path / "plain.py", tmp_path / "other.py"
    plain.write_text(GOOD, encoding="utf-8")
    other.write_bytes(encode(GOOD))

    assert script_sha256(other) == script_sha256(plain)


def test_a_script_that_is_not_utf8_has_no_identity(tmp_path: Path) -> None:
    script = tmp_path / "example.py"
    script.write_bytes(b"# caf\xe9\n")

    with pytest.raises(NotUtf8Error):
        script_sha256(script)


class _Normalised(TypedDict):
    count: int
    mean_error: float


def _returning(value: JsonValue) -> ExampleRun:
    provenance = Provenance(source="examples/x.py", sha256="0" * 64, seconds=1.0, device="cpu")
    return ExampleRun(
        notebook=nbformat.v4.new_notebook(),
        provenance=provenance,
        output_dir=Path("/unused"),
        returned=value,
    )


def test_returned_reads_as_the_shape_a_result_test_declares() -> None:
    """A result test types what ``main()`` returned, as ``report_as`` types a report (T-5)."""
    run = _returning({"count": 128, "mean_error": 1})

    normalised = run.returned_as(_Normalised)

    assert normalised == {"count": 128, "mean_error": 1}


def test_returned_of_another_shape_is_refused_naming_the_field() -> None:
    run = _returning({"count": "128", "mean_error": 0.5})

    with pytest.raises(pydantic.ValidationError, match="count"):
        run.returned_as(_Normalised)
