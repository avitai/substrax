"""``run_python`` runs code in a fresh interpreter whose JAX configuration the test chooses."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import textwrap
from pathlib import Path
from typing import TypedDict

import pydantic
import pytest

from substrax.runtime import JaxRuntime, XlaFlagConflictError
from substrax.testing import (
    ChildFailedError,
    ChildResult,
    cuda_is_visible,
    report,
    REPORT_PATH_ENV,
    run_python,
)


TIMEOUT = 180.0


class _Device(TypedDict):
    platform: str
    id: int


class _Report(TypedDict):
    devices: list[_Device]
    loss: float


def _result(
    report_text: str | None = None, *, returncode: int = 0, stderr: str = ""
) -> ChildResult:
    return ChildResult(
        argv=("python", "-c", "pass"),
        returncode=returncode,
        stdout="",
        stderr=stderr,
        report=report_text,
        seconds=0.1,
    )


class TestChildResult:
    def test_report_as_returns_the_value_in_the_shape_asked_for(self) -> None:
        shape = _result('{"devices": [{"platform": "cpu", "id": 0}], "loss": 0.5}').report_as(
            _Report
        )

        assert shape["devices"][0]["id"] == 0
        assert shape["loss"] == 0.5

    def test_report_as_refuses_a_value_of_another_shape_naming_the_field(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="loss"):
            _result('{"devices": [], "loss": "high"}').report_as(_Report)

    def test_report_as_refuses_a_child_that_wrote_no_report(self) -> None:
        with pytest.raises(ValueError, match="no report"):
            _result(None).report_as(int)

    def test_report_json_decodes_the_report_without_a_shape(self) -> None:
        assert _result('{"a": [1, 2]}').report_json() == {"a": [1, 2]}

    def test_report_json_refuses_a_child_that_wrote_no_report(self) -> None:
        with pytest.raises(ValueError, match="no report"):
            _result(None).report_json()

    def test_check_returns_a_successful_result(self) -> None:
        result = _result("ok\n")

        assert result.check() is result

    def test_check_raises_with_the_exit_code_and_the_end_of_stderr(self) -> None:
        stderr = "\n".join(f"line {index}" for index in range(100))

        with pytest.raises(ChildFailedError, match="exit code 3") as caught:
            _result("", returncode=3, stderr=stderr).check()

        message = str(caught.value)
        assert "line 60" in message
        assert "line 99" in message
        assert "line 59" not in message
        assert caught.value.result.returncode == 3


class TestRunPython:
    def test_code_runs_and_its_output_comes_back(self) -> None:
        result = run_python("print('hello')", timeout=TIMEOUT).check()

        assert result.stdout == "hello\n"
        assert result.report is None
        assert result.argv[1] == "-c"
        assert result.seconds > 0

    def test_a_script_runs_with_its_arguments_in_the_given_directory(self, tmp_path: Path) -> None:
        script = tmp_path / "script.py"
        script.write_text(
            "import os, sys\nfrom substrax.testing import report\n"
            "report({'args': sys.argv[1:], 'cwd': os.getcwd()})\n"
        )

        result = run_python(script, "one", "two", cwd=tmp_path, timeout=TIMEOUT).check()

        assert result.report_as(dict[str, object]) == {
            "args": ["one", "two"],
            "cwd": str(tmp_path.resolve()),
        }

    def test_a_failing_child_is_returned_rather_than_raised(self) -> None:
        assert run_python("import sys; sys.exit(3)", timeout=TIMEOUT).returncode == 3

    def test_a_child_that_outlives_its_timeout_raises(self) -> None:
        with pytest.raises(subprocess.TimeoutExpired):
            run_python("import time; time.sleep(30)", timeout=1.0)

    def test_inherited_jax_and_xla_settings_do_not_reach_the_child(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
        monkeypatch.setenv("JAX_NUM_CPU_DEVICES", "4")
        monkeypatch.setenv("JAX_PLATFORMS", "cuda")
        monkeypatch.setenv("SUBSTRAX_TEST_SENTINEL", "kept")
        code = textwrap.dedent(
            """
            import os
            import jax
            from substrax.testing import report
            report({
                "devices": jax.device_count(),
                "platforms": os.environ.get("JAX_PLATFORMS"),
                "flags": os.environ.get("XLA_FLAGS"),
                "preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
                "sentinel": os.environ.get("SUBSTRAX_TEST_SENTINEL"),
            })
            """
        )

        assert run_python(code, timeout=TIMEOUT).check().report_as(dict[str, object]) == {
            "devices": 1,
            "platforms": "cpu",
            "flags": None,
            "preallocate": "false",
            "sentinel": "kept",
        }

    def test_the_runtime_configures_the_child(self) -> None:
        runtime = JaxRuntime(platforms=("cpu",), cpu_devices=2, enable_x64=True)
        code = (
            "import jax, jax.numpy as jnp; from substrax.testing import report; "
            "report([jax.device_count(), str(jnp.ones(1).dtype)])"
        )

        assert run_python(code, runtime=runtime, timeout=TIMEOUT).check().report_as(
            tuple[int, str]
        ) == (
            2,
            "float64",
        )

    def test_a_runtime_without_platforms_still_runs_the_child_on_the_cpu(self) -> None:
        code = "import os; from substrax.testing import report; report(os.environ.get('JAX_PLATFORMS'))"

        result = run_python(code, runtime=JaxRuntime(cpu_devices=2), timeout=TIMEOUT)

        assert result.check().report_as(str) == "cpu"

    def test_env_applies_on_top_and_runtime_flags_merge_into_it(self) -> None:
        runtime = JaxRuntime(xla_flags=("--xla_cpu_multi_thread_eigen=false",))
        code = (
            "import os; from substrax.testing import report; "
            "report([os.environ['XLA_FLAGS'], os.environ['MARK']])"
        )

        result = run_python(
            code,
            runtime=runtime,
            env={"XLA_FLAGS": "--xla_cpu_enable_fast_math=false", "MARK": "on"},
            timeout=TIMEOUT,
        )

        assert result.check().report_as(list[str]) == [
            "--xla_cpu_enable_fast_math=false --xla_cpu_multi_thread_eigen=false",
            "on",
        ]

    def test_a_report_survives_output_written_after_it(self) -> None:
        code = textwrap.dedent(
            """
            import os, sys
            from substrax.testing import report
            report({"loss": 0.5})
            print("[Gloo] Rank 1 is connected to 1 peer ranks.")
            sys.stdout.flush()
            os.write(1, b"written to descriptor 1 by a library\\n")
            """
        )

        result = run_python(code, timeout=TIMEOUT).check()

        assert result.report_as(dict[str, float]) == {"loss": 0.5}
        assert result.stdout.splitlines()[-1] == "written to descriptor 1 by a library"

    def test_a_child_that_reports_twice_is_refused_naming_the_first_report(self) -> None:
        code = "from substrax.testing import report; report(1); report(2)"

        result = run_python(code, timeout=TIMEOUT)

        assert result.returncode != 0
        assert "already reported" in result.stderr
        assert result.report_as(int) == 1

    def test_the_report_path_is_one_the_caller_cannot_choose(self, tmp_path: Path) -> None:
        chosen = tmp_path / "chosen.json"
        code = "from substrax.testing import report; report(7)"

        result = run_python(code, env={REPORT_PATH_ENV: str(chosen)}, timeout=TIMEOUT).check()

        assert result.report_as(int) == 7
        assert not chosen.exists()

    def test_a_runtime_flag_conflicting_with_env_raises_before_the_child_starts(
        self, tmp_path: Path
    ) -> None:
        started = tmp_path / "started"

        with pytest.raises(XlaFlagConflictError):
            run_python(
                f"open({str(started)!r}, 'w').close()",
                runtime=JaxRuntime(xla_flags=("--xla_force_host_platform_device_count=1",)),
                env={"XLA_FLAGS": "--xla_force_host_platform_device_count=2"},
                timeout=TIMEOUT,
            )

        assert not started.exists()


@pytest.mark.skipif(
    any(importlib.util.find_spec(name) for name in ("jax_cuda12_plugin", "jax_cuda13_plugin")),
    reason="a JAX CUDA plugin is installed, so the probe would start a GPU backend",
)
def test_cuda_is_not_visible_without_a_cuda_plugin() -> None:
    assert cuda_is_visible(timeout=TIMEOUT) is False


def test_report_outside_a_child_of_run_python_is_refused_naming_the_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(REPORT_PATH_ENV, raising=False)

    with pytest.raises(RuntimeError, match=REPORT_PATH_ENV):
        report({"a": 1})


def test_report_writes_json_the_parent_reads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "report.json"
    monkeypatch.setenv(REPORT_PATH_ENV, str(path))

    report({"a": [1, 2]})

    assert json.loads(path.read_text(encoding="utf-8")) == {"a": [1, 2]}
