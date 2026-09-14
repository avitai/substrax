"""``run_python`` runs code in a fresh interpreter whose JAX configuration the test chooses."""

from __future__ import annotations

import importlib.util
import subprocess
import textwrap
from pathlib import Path

import pytest

from substrax.runtime import JaxRuntime, XlaFlagConflictError
from substrax.testing import ChildFailedError, ChildResult, cuda_is_visible, run_python


TIMEOUT = 180.0


def _result(stdout: str, *, returncode: int = 0, stderr: str = "") -> ChildResult:
    return ChildResult(
        argv=("python", "-c", "pass"),
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        seconds=0.1,
    )


class TestChildResult:
    def test_last_json_reads_the_last_non_empty_line(self) -> None:
        assert _result('progress\n{"a": 1}\n\n').last_json() == {"a": 1}

    def test_last_json_without_output_raises(self) -> None:
        with pytest.raises(ValueError, match="no output"):
            _result("\n").last_json()

    def test_last_json_on_a_line_that_is_not_json_raises_and_shows_it(self) -> None:
        with pytest.raises(ValueError, match="done"):
            _result('{"a": 1}\ndone\n').last_json()

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
        assert result.argv[1] == "-c"
        assert result.seconds > 0

    def test_a_script_runs_with_its_arguments_in_the_given_directory(self, tmp_path: Path) -> None:
        script = tmp_path / "script.py"
        script.write_text(
            "import json, os, sys\nprint(json.dumps({'args': sys.argv[1:], 'cwd': os.getcwd()}))\n"
        )

        result = run_python(script, "one", "two", cwd=tmp_path, timeout=TIMEOUT).check()

        assert result.last_json() == {"args": ["one", "two"], "cwd": str(tmp_path.resolve())}

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
            import json, os
            import jax
            print(json.dumps({
                "devices": jax.device_count(),
                "platforms": os.environ.get("JAX_PLATFORMS"),
                "flags": os.environ.get("XLA_FLAGS"),
                "preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
                "sentinel": os.environ.get("SUBSTRAX_TEST_SENTINEL"),
            }))
            """
        )

        assert run_python(code, timeout=TIMEOUT).check().last_json() == {
            "devices": 1,
            "platforms": "cpu",
            "flags": None,
            "preallocate": "false",
            "sentinel": "kept",
        }

    def test_the_runtime_configures_the_child(self) -> None:
        runtime = JaxRuntime(platforms=("cpu",), cpu_devices=2, enable_x64=True)
        code = (
            "import json, jax, jax.numpy as jnp; "
            "print(json.dumps([jax.device_count(), str(jnp.ones(1).dtype)]))"
        )

        assert run_python(code, runtime=runtime, timeout=TIMEOUT).check().last_json() == [
            2,
            "float64",
        ]

    def test_a_runtime_without_platforms_still_runs_the_child_on_the_cpu(self) -> None:
        code = "import json, os; print(json.dumps(os.environ.get('JAX_PLATFORMS')))"

        result = run_python(code, runtime=JaxRuntime(cpu_devices=2), timeout=TIMEOUT)

        assert result.check().last_json() == "cpu"

    def test_env_applies_on_top_and_runtime_flags_merge_into_it(self) -> None:
        runtime = JaxRuntime(xla_flags=("--xla_cpu_multi_thread_eigen=false",))
        code = "import json, os; print(json.dumps([os.environ['XLA_FLAGS'], os.environ['MARK']]))"

        result = run_python(
            code,
            runtime=runtime,
            env={"XLA_FLAGS": "--xla_cpu_enable_fast_math=false", "MARK": "on"},
            timeout=TIMEOUT,
        )

        assert result.check().last_json() == [
            "--xla_cpu_enable_fast_math=false --xla_cpu_multi_thread_eigen=false",
            "on",
        ]

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
