"""``apply_runtime`` writes the environment before jax is imported, and configures jax after.

Settings jax reads only when its backends start (platforms, XLA flags, client memory) are refused
once jax is imported, because jax offers no public signal of backend initialisation and a late
write would silently change nothing. The device count, the compilation cache and the paths that
need a fresh interpreter are covered in ``test_runtime_process.py``. The substrax pytest plugin
fails a test that changes jax's global configuration, so the tests that apply settings on purpose
set them back through ``restored_jax_config``.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import jax
import pytest

from substrax.runtime import apply_runtime, JaxRuntime, RuntimeConfigurationError
from substrax.testing import restored_jax_config


@pytest.fixture
def jax_not_imported(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hide the imported jax module, as in a process that has not imported it yet."""
    monkeypatch.delitem(sys.modules, "jax")


@pytest.mark.usefixtures("jax_not_imported")
class TestBeforeJaxIsImported:
    def test_every_field_is_written_to_the_given_environment(self) -> None:
        environ = {"XLA_FLAGS": "--a=1"}

        apply_runtime(
            JaxRuntime(cpu_devices=2, xla_flags=("--b=2",), preallocate=False), environ=environ
        )

        assert environ == {
            "XLA_FLAGS": "--a=1 --b=2",
            "JAX_NUM_CPU_DEVICES": "2",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }

    def test_the_process_environment_is_the_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("JAX_ENABLE_X64", "true")

        apply_runtime(JaxRuntime(enable_x64=False))

        assert os.environ["JAX_ENABLE_X64"] == "false"


@pytest.mark.parametrize(
    ("fields", "name"),
    [
        ({"platforms": ("cpu",)}, "platforms"),
        ({"xla_flags": ("--xla_gpu_deterministic_ops=true",)}, "xla_flags"),
        ({"preallocate": False}, "preallocate"),
        ({"memory_fraction": 0.5}, "memory_fraction"),
    ],
)
def test_a_backend_start_setting_raises_once_jax_is_imported(
    fields: dict[str, Any], name: str
) -> None:
    environ: dict[str, str] = {}

    with pytest.raises(RuntimeConfigurationError, match=name):
        apply_runtime(JaxRuntime(**fields), environ=environ)

    assert environ == {}


def test_the_refusal_names_every_backend_start_setting_given() -> None:
    runtime = JaxRuntime(platforms=("cpu",), preallocate=False, enable_x64=False)

    with pytest.raises(RuntimeConfigurationError, match=r"platforms.*preallocate"):
        apply_runtime(runtime, environ={})


def test_a_refused_runtime_changes_no_jax_setting() -> None:
    previous = jax.config.jax_enable_x64

    with pytest.raises(RuntimeConfigurationError):
        apply_runtime(JaxRuntime(enable_x64=not previous, preallocate=False), environ={})

    assert jax.config.jax_enable_x64 is previous


def test_x64_goes_through_jax_config_once_jax_is_imported() -> None:
    requested = not jax.config.jax_enable_x64
    environ: dict[str, str] = {}

    with restored_jax_config() as changed:
        apply_runtime(JaxRuntime(enable_x64=requested), environ=environ)
        applied = jax.config.jax_enable_x64

    assert applied is requested
    assert changed == ["jax_enable_x64"]
    assert environ == {}


def test_matmul_precision_goes_through_jax_config_once_jax_is_imported() -> None:
    requested = "highest" if jax.config.jax_default_matmul_precision != "highest" else "high"

    with restored_jax_config() as changed:
        apply_runtime(JaxRuntime(matmul_precision=requested), environ={})
        applied = jax.config.jax_default_matmul_precision

    assert applied == requested
    assert changed == ["jax_default_matmul_precision"]
