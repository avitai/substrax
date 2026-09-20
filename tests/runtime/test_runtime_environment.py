"""The environment a ``JaxRuntime`` needs before jax is imported, and the test-run resolver."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import pytest

from substrax.runtime import (
    child_environment,
    JaxRuntime,
    resolve_test_runtime,
    runtime_environment,
    RuntimeConfigurationError,
    XlaFlagConflictError,
)


DEVICE_COUNT = "--xla_force_host_platform_device_count"
PREFIX = "DATARAX_TEST_"


class TestRuntimeEnvironment:
    def test_an_empty_runtime_sets_nothing(self) -> None:
        assert runtime_environment(JaxRuntime(), {"XLA_FLAGS": "--a=1"}) == {}

    def test_every_field_becomes_the_variable_jax_or_xla_reads(self, tmp_path: Path) -> None:
        runtime = JaxRuntime(
            platforms=("cuda", "cpu"),
            cpu_devices=8,
            enable_x64=True,
            matmul_precision="high",
            compilation_cache_dir=tmp_path,
            xla_flags=("--xla_gpu_deterministic_ops=true",),
            preallocate=False,
            memory_fraction=0.75,
        )

        assert runtime_environment(runtime, {}) == {
            "JAX_PLATFORMS": "cuda,cpu",
            "JAX_NUM_CPU_DEVICES": "8",
            "JAX_ENABLE_X64": "true",
            "JAX_DEFAULT_MATMUL_PRECISION": "high",
            "JAX_COMPILATION_CACHE_DIR": str(tmp_path),
            "XLA_FLAGS": "--xla_gpu_deterministic_ops=true",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_CLIENT_MEM_FRACTION": "0.75",
        }

    def test_false_booleans_are_written_as_false(self) -> None:
        environment = runtime_environment(JaxRuntime(enable_x64=False, preallocate=True), {})

        assert (environment["JAX_ENABLE_X64"], environment["XLA_PYTHON_CLIENT_PREALLOCATE"]) == (
            "false",
            "true",
        )

    def test_a_memory_fraction_refuses_the_deprecated_variable_left_in_the_environment(
        self,
    ) -> None:
        """jaxlib raises at GPU start when both memory-fraction variables are set."""
        with pytest.raises(RuntimeConfigurationError, match="XLA_PYTHON_CLIENT_MEM_FRACTION"):
            runtime_environment(
                JaxRuntime(memory_fraction=0.5), {"XLA_PYTHON_CLIENT_MEM_FRACTION": ".75"}
            )

    def test_the_deprecated_variable_alone_is_left_to_jax(self) -> None:
        environment = runtime_environment(
            JaxRuntime(cpu_devices=2), {"XLA_PYTHON_CLIENT_MEM_FRACTION": ".75"}
        )

        assert environment == {"JAX_NUM_CPU_DEVICES": "2"}

    def test_requested_flags_merge_into_the_inherited_ones(self) -> None:
        runtime = JaxRuntime(xla_flags=("--c=3",))

        assert runtime_environment(runtime, {"XLA_FLAGS": "--a=1 --b=2"}) == {
            "XLA_FLAGS": "--a=1 --b=2 --c=3"
        }

    def test_a_requested_flag_conflicting_with_an_inherited_one_raises(self) -> None:
        runtime = JaxRuntime(xla_flags=(f"{DEVICE_COUNT}=1",))

        with pytest.raises(XlaFlagConflictError, match=DEVICE_COUNT):
            runtime_environment(runtime, {"XLA_FLAGS": f"{DEVICE_COUNT}=8"})

    def test_an_explicit_field_replaces_the_inherited_variable(self) -> None:
        environment = runtime_environment(JaxRuntime(cpu_devices=8), {"JAX_NUM_CPU_DEVICES": "2"})

        assert environment == {"JAX_NUM_CPU_DEVICES": "8"}

    def test_the_base_environment_is_not_modified(self) -> None:
        base = {"XLA_FLAGS": "--a=1", "JAX_PLATFORMS": "cuda"}

        runtime_environment(JaxRuntime(platforms=("cpu",), xla_flags=("--b=2",)), base)

        assert base == {"XLA_FLAGS": "--a=1", "JAX_PLATFORMS": "cuda"}


class TestResolveTestRuntime:
    """The datarax test-run rules, with datarax's variable names passed as a prefix."""

    def _resolve(self, env: dict[str, str], *, plugin: bool = True) -> tuple[object, object]:
        runtime = resolve_test_runtime(env, prefix=PREFIX, cuda_plugin_available=plugin)
        return runtime.platforms, runtime.cpu_devices

    def test_tests_default_to_eight_emulated_cpu_devices(self) -> None:
        assert self._resolve({}) == (("cpu",), 8)

    def test_an_inherited_cuda_selection_still_runs_tests_on_emulated_cpus(self) -> None:
        assert self._resolve({"JAX_PLATFORMS": "cuda,cpu"}) == (("cpu",), 8)

    def test_an_explicit_accelerator_request_disables_cpu_emulation(self) -> None:
        env = {f"{PREFIX}JAX_PLATFORMS": "cuda,cpu", "XLA_FLAGS": "--xla_gpu_autotune_level=2"}

        assert self._resolve(env) == (("cuda", "cpu"), None)

    def test_an_explicit_cpu_request_keeps_emulation(self) -> None:
        assert self._resolve({f"{PREFIX}JAX_PLATFORMS": "cpu"}) == (("cpu",), 8)

    def test_a_device_count_of_zero_disables_emulation(self) -> None:
        assert self._resolve({f"{PREFIX}DEVICE_COUNT": "0"}, plugin=False) == (("cpu",), None)

    def test_a_requested_device_count_is_used(self) -> None:
        assert self._resolve({f"{PREFIX}DEVICE_COUNT": "4"}, plugin=False) == (("cpu",), 4)

    def test_a_device_count_already_in_the_flags_is_kept(self) -> None:
        assert self._resolve({"XLA_FLAGS": f"--a=1 {DEVICE_COUNT}=2"}, plugin=False) == (
            ("cpu",),
            None,
        )

    def test_a_device_count_already_in_the_jax_variable_is_kept(self) -> None:
        assert self._resolve({"JAX_NUM_CPU_DEVICES": "2"}, plugin=False) == (("cpu",), None)

    def test_asking_for_cuda_without_the_plugin_fails(self) -> None:
        with pytest.raises(RuntimeConfigurationError, match="CUDA plugin"):
            self._resolve({f"{PREFIX}JAX_PLATFORMS": "cuda"}, plugin=False)

    def test_a_device_count_that_is_not_an_integer_raises_and_names_the_variable(self) -> None:
        with pytest.raises(ValueError, match=f"{PREFIX}DEVICE_COUNT"):
            self._resolve({f"{PREFIX}DEVICE_COUNT": "eight"})

    def test_variables_of_another_prefix_are_ignored(self) -> None:
        assert self._resolve({"OTHER_TEST_DEVICE_COUNT": "4"}) == (("cpu",), 8)

    def test_the_default_device_count_can_be_chosen(self) -> None:
        runtime = resolve_test_runtime(
            {}, prefix=PREFIX, cuda_plugin_available=False, default_cpu_devices=2
        )

        assert runtime.cpu_devices == 2

    def test_the_resolved_runtime_renders_to_a_cpu_environment(self) -> None:
        env = {"JAX_PLATFORMS": "cuda,cpu", "XLA_FLAGS": "--a=1"}
        runtime = resolve_test_runtime(env, prefix=PREFIX, cuda_plugin_available=True)

        assert runtime_environment(runtime, env) == {
            "JAX_PLATFORMS": "cpu",
            "JAX_NUM_CPU_DEVICES": "8",
        }


PARENT = MappingProxyType(
    {
        "PATH": "/usr/bin",
        "JAX_PLATFORMS": "cuda",
        "XLA_FLAGS": "--xla_dump_to=/tmp/dump",
        "JAX_ENABLE_X64": "1",
    }
)


class TestChildEnvironment:
    def test_the_parents_jax_and_xla_settings_are_dropped_and_the_rest_kept(self) -> None:
        assert child_environment(JaxRuntime(), parent=PARENT) == {"PATH": "/usr/bin"}

    def test_env_applies_over_the_parent_and_the_runtime_over_env(self) -> None:
        runtime = JaxRuntime(platforms=("cuda",), xla_flags=("--xla_gpu_deterministic_ops=true",))

        child = child_environment(
            runtime,
            {"XLA_FLAGS": "--xla_gpu_autotune_level=0", "PATH": "/opt/bin"},
            parent=PARENT,
        )

        assert child == {
            "PATH": "/opt/bin",
            "JAX_PLATFORMS": "cuda",
            "XLA_FLAGS": "--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true",
        }

    def test_env_may_set_jax_variables_the_parent_could_not_pass_on(self) -> None:
        child = child_environment(JaxRuntime(), {"JAX_ENABLE_X64": "0"}, parent=PARENT)

        assert child["JAX_ENABLE_X64"] == "0"

    def test_the_parent_defaults_to_this_process(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SUBSTRAX_CHILD_ENVIRONMENT_PROBE", "kept")
        monkeypatch.setenv("JAX_CHILD_ENVIRONMENT_PROBE", "dropped")

        child = child_environment(JaxRuntime())

        assert child["SUBSTRAX_CHILD_ENVIRONMENT_PROBE"] == "kept"
        assert "JAX_CHILD_ENVIRONMENT_PROBE" not in child

    def test_a_runtime_flag_conflicting_with_env_raises(self) -> None:
        runtime = JaxRuntime(xla_flags=("--xla_gpu_autotune_level=4",))

        with pytest.raises(XlaFlagConflictError):
            child_environment(runtime, {"XLA_FLAGS": "--xla_gpu_autotune_level=0"}, parent={})
