"""``substrax.runtime`` in fresh interpreters: device topology, late updates and import purity.

jax fixes its device set when its backends start, so every case runs in its own interpreter
through ``substrax.testing.run_python``.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import pytest

from substrax.runtime import JaxRuntime
from substrax.testing import run_python


TIMEOUT = 180.0


def _run(source: str, runtime: JaxRuntime | None = None) -> Any:
    """Run ``source`` in a fresh interpreter and return the JSON its last line printed."""
    result = run_python(textwrap.dedent(source), runtime=runtime, timeout=TIMEOUT)
    return result.check().last_json()


@pytest.mark.parametrize("count", [1, 2, 8])
def test_the_rendered_environment_sets_the_cpu_device_count(count: int) -> None:
    runtime = JaxRuntime(platforms=("cpu",), cpu_devices=count)

    assert _run("import json, jax; print(json.dumps(jax.device_count()))", runtime) == count


def test_apply_runtime_before_import_configures_the_process() -> None:
    source = """
        import json

        from substrax.runtime import JaxRuntime, apply_runtime

        apply_runtime(JaxRuntime(cpu_devices=3, enable_x64=True, matmul_precision="high"))

        import jax
        import jax.numpy as jnp

        print(json.dumps(
            [jax.device_count(), str(jnp.ones(1).dtype), jax.config.jax_default_matmul_precision]
        ))
    """

    assert _run(source) == [3, "float64", "high"]


def test_the_device_count_applies_after_import_until_the_first_operation() -> None:
    source = """
        import json

        import jax

        from substrax.runtime import JaxRuntime, apply_runtime

        apply_runtime(JaxRuntime(cpu_devices=3))
        print(json.dumps(jax.device_count()))
    """

    assert _run(source) == 3


def test_a_device_count_after_the_first_operation_raises_and_names_the_field() -> None:
    source = """
        import json

        import jax

        from substrax.runtime import JaxRuntime, RuntimeConfigurationError, apply_runtime

        jax.devices()
        try:
            apply_runtime(JaxRuntime(cpu_devices=3))
        except RuntimeConfigurationError as error:
            cause = type(error.__cause__).__name__
            print(json.dumps([cause, "cpu_devices" in str(error), jax.device_count()]))
    """

    assert _run(source) == ["RuntimeError", True, 1]


def test_a_compilation_cache_directory_applies_after_import(tmp_path: Path) -> None:
    source = f"""
        import json
        from pathlib import Path

        import jax

        from substrax.runtime import JaxRuntime, apply_runtime

        apply_runtime(JaxRuntime(compilation_cache_dir=Path({str(tmp_path)!r})))
        print(json.dumps(jax.config.jax_compilation_cache_dir))
    """

    assert _run(source) == str(tmp_path)


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


def test_importing_the_runtime_changes_no_process_state() -> None:
    assert _run(_IMPORT_PROBE.format(module="substrax.runtime")) == {
        "jax_imported": False,
        "environment_changed": False,
        "handlers_changed": False,
    }


def test_the_import_probe_detects_a_module_that_imports_jax() -> None:
    """Control: the same probe reports jax for a subpackage that imports it."""
    assert _run(_IMPORT_PROBE.format(module="substrax.mesh"))["jax_imported"] is True
