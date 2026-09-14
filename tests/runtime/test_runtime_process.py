"""``substrax.runtime`` in fresh interpreters: device topology, late updates and import purity.

jax fixes its device set when its backends start, so every case runs in its own interpreter
through the ``run_interpreter`` fixture.
"""

from __future__ import annotations

import json
import subprocess
import textwrap
from collections.abc import Callable
from pathlib import Path

import pytest

from substrax.runtime import JaxRuntime, runtime_environment


type Runner = Callable[[list[str], dict[str, str]], subprocess.CompletedProcess[str]]


def _program(source: str) -> list[str]:
    return ["-c", textwrap.dedent(source)]


def _last_line(completed: subprocess.CompletedProcess[str]) -> str:
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip().splitlines()[-1]


@pytest.mark.parametrize("count", [1, 2, 8])
def test_the_rendered_environment_sets_the_cpu_device_count(
    run_interpreter: Runner, count: int
) -> None:
    env = runtime_environment(JaxRuntime(platforms=("cpu",), cpu_devices=count), {})

    completed = run_interpreter(_program("import jax; print(jax.device_count())"), env)

    assert _last_line(completed) == str(count)


def test_apply_runtime_before_import_configures_the_process(run_interpreter: Runner) -> None:
    completed = run_interpreter(
        _program(
            """
            from substrax.runtime import JaxRuntime, apply_runtime

            apply_runtime(JaxRuntime(cpu_devices=3, enable_x64=True, matmul_precision="high"))

            import jax
            import jax.numpy as jnp

            print(jax.device_count(), jnp.ones(1).dtype, jax.config.jax_default_matmul_precision)
            """
        ),
        {},
    )

    assert _last_line(completed) == "3 float64 high"


def test_the_device_count_applies_after_import_until_the_first_operation(
    run_interpreter: Runner,
) -> None:
    completed = run_interpreter(
        _program(
            """
            import jax

            from substrax.runtime import JaxRuntime, apply_runtime

            apply_runtime(JaxRuntime(cpu_devices=3))
            print(jax.device_count())
            """
        ),
        {},
    )

    assert _last_line(completed) == "3"


def test_a_device_count_after_the_first_operation_raises_and_names_the_field(
    run_interpreter: Runner,
) -> None:
    completed = run_interpreter(
        _program(
            """
            import jax

            from substrax.runtime import JaxRuntime, RuntimeConfigurationError, apply_runtime

            jax.devices()
            try:
                apply_runtime(JaxRuntime(cpu_devices=3))
            except RuntimeConfigurationError as error:
                print(type(error.__cause__).__name__, "cpu_devices" in str(error), jax.device_count())
            """
        ),
        {},
    )

    assert _last_line(completed) == "RuntimeError True 1"


def test_a_compilation_cache_directory_applies_after_import(
    run_interpreter: Runner, tmp_path: Path
) -> None:
    completed = run_interpreter(
        _program(
            f"""
            from pathlib import Path

            import jax

            from substrax.runtime import JaxRuntime, apply_runtime

            apply_runtime(JaxRuntime(compilation_cache_dir=Path({str(tmp_path)!r})))
            print(jax.config.jax_compilation_cache_dir)
            """
        ),
        {},
    )

    assert _last_line(completed) == str(tmp_path)


_IMPORT_PROBE = """
    import json, logging, os, sys

    environment = dict(os.environ)
    handlers = list(logging.getLogger().handlers)
    import {module}
    print(json.dumps({{
        "jax_imported": "jax" in sys.modules,
        "environment_changed": dict(os.environ) != environment,
        "handlers_changed": logging.getLogger().handlers != handlers,
    }}))
"""


def test_importing_the_runtime_changes_no_process_state(run_interpreter: Runner) -> None:
    completed = run_interpreter(_program(_IMPORT_PROBE.format(module="substrax.runtime")), {})

    assert json.loads(_last_line(completed)) == {
        "jax_imported": False,
        "environment_changed": False,
        "handlers_changed": False,
    }


def test_the_import_probe_detects_a_module_that_imports_jax(run_interpreter: Runner) -> None:
    """Control: the same probe reports jax for a subpackage that imports it."""
    completed = run_interpreter(_program(_IMPORT_PROBE.format(module="substrax.mesh")), {})

    assert json.loads(_last_line(completed))["jax_imported"] is True
