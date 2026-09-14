"""The opt-in pytest plugin: JAX numeric state per test, the x64 collection guard, device markers.

Each case runs a separate pytest process through ``pytester``, because the plugin's subject is
process-wide JAX state.
"""

from __future__ import annotations

import pytest

from substrax.testing import run_python


PLUGIN = (
    "-p",
    "substrax.testing.pytest_plugin",
    "-p",
    "no:randomly",
    "-p",
    "no:cacheprovider",
    "--strict-markers",
)


def _run(pytester: pytest.Pytester, *args: str) -> pytest.RunResult:
    return pytester.runpytest_subprocess(*PLUGIN, *args)


def test_an_x64_marked_test_runs_in_float64_and_the_next_test_does_not(
    pytester: pytest.Pytester,
) -> None:
    pytester.makepyfile(
        test_dtypes="""
        import jax.numpy as jnp
        import pytest

        @pytest.mark.x64
        def test_marked():
            assert jnp.ones(1).dtype == jnp.float64

        def test_unmarked():
            assert jnp.ones(1).dtype == jnp.float32
        """
    )

    _run(pytester).assert_outcomes(passed=2)


_LEAKING_TESTS = """
    import jax
    import jax.numpy as jnp

    ORIGINAL_PRECISION = jax.config.jax_default_matmul_precision

    def test_flips_global_state():
        jax.config.update("jax_enable_x64", True)
        jax.config.update("jax_default_matmul_precision", "highest")

    def test_sees_the_original_state():
        assert jnp.ones(1).dtype == jnp.float32
        assert jax.config.jax_default_matmul_precision == ORIGINAL_PRECISION
"""


def test_a_test_that_changes_global_config_fails_and_the_next_test_sees_the_original(
    pytester: pytest.Pytester,
) -> None:
    pytester.makepyfile(test_leak=_LEAKING_TESTS)

    result = _run(pytester)

    result.assert_outcomes(passed=2, errors=1)
    result.stdout.fnmatch_lines(
        ["*changed global jax configuration*jax_default_matmul_precision*jax_enable_x64*"]
    )


def test_a_test_that_sets_back_its_own_change_passes(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        test_restoring="""
        import jax

        from substrax.testing import restored_jax_config

        def test_changes_and_restores():
            with restored_jax_config():
                jax.config.update("jax_enable_x64", True)
        """
    )

    _run(pytester).assert_outcomes(passed=1)


def test_without_the_plugin_the_same_change_leaks_into_the_next_test(
    pytester: pytest.Pytester,
) -> None:
    """Control: the restoration above is the plugin's doing."""
    pytester.makepyfile(test_leak=_LEAKING_TESTS)

    pytester.runpytest_subprocess("-p", "no:randomly", "-p", "no:cacheprovider").assert_outcomes(
        passed=1, failed=1
    )


def test_x64_left_on_by_a_module_at_import_fails_the_session(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        test_module_level="""
        import jax

        jax.config.update("jax_enable_x64", True)

        def test_anything():
            pass
        """
    )

    result = _run(pytester)

    assert result.ret != pytest.ExitCode.OK
    assert "jax_enable_x64 is on after collection" in result.stdout.str() + result.stderr.str()


def test_a_session_started_in_x64_is_not_refused(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("JAX_ENABLE_X64", "1")
    pytester.makepyfile(
        test_x64_lane="""
        import jax.numpy as jnp

        def test_float64_by_default():
            assert jnp.ones(1).dtype == jnp.float64
        """
    )

    _run(pytester).assert_outcomes(passed=1)


_DEVICE_TESTS = """
    import jax
    import pytest

    @pytest.mark.devices(2)
    def test_needs_two_devices():
        assert jax.device_count() >= 2

    @pytest.mark.devices(1, kind="gpu")
    def test_needs_a_gpu():
        pass
"""


@pytest.mark.parametrize(("visible", "passed"), [("1", 0), ("2", 1), ("8", 1)])
def test_a_devices_marker_runs_only_with_enough_devices_of_its_kind(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, visible: str, passed: int
) -> None:
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("JAX_NUM_CPU_DEVICES", visible)
    pytester.makepyfile(test_devices=_DEVICE_TESTS)

    result = _run(pytester, "-rs")

    result.assert_outcomes(passed=passed, skipped=2 - passed)
    result.stdout.fnmatch_lines([f"*needs 1 gpu device*{visible} cpu*"])


def test_an_accelerator_marker_skips_on_the_cpu(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    pytester.makepyfile(
        test_accelerator="""
        import pytest

        @pytest.mark.accelerator
        def test_any_accelerator():
            pass

        @pytest.mark.accelerator(kind="gpu")
        def test_a_gpu():
            pass
        """
    )

    _run(pytester).assert_outcomes(skipped=2)


@pytest.mark.parametrize(
    "marker",
    [
        "devices()",
        "devices('two')",
        "devices(0)",
        "devices(2, kind='quantum')",
        "devices(2, 3)",
        "accelerator(kind='quantum')",
        "accelerator('gpu', 'tpu')",
    ],
)
def test_a_malformed_device_marker_is_a_usage_error(pytester: pytest.Pytester, marker: str) -> None:
    pytester.makepyfile(
        test_marker=f"""
        import pytest

        @pytest.mark.{marker}
        def test_marked():
            pass
        """
    )

    assert _run(pytester).ret == pytest.ExitCode.USAGE_ERROR


_IMPORT_PROBE = "import json, sys; import {module}; print(json.dumps('jax' in sys.modules))"


@pytest.mark.parametrize("module", ["substrax.testing", "substrax.testing.pytest_plugin"])
def test_importing_the_testing_package_imports_no_jax(module: str) -> None:
    result = run_python(_IMPORT_PROBE.format(module=module), timeout=180.0)

    assert result.check().last_json() is False


def test_the_import_probe_detects_jax() -> None:
    """Control: the same probe reports jax for a subpackage that imports it."""
    result = run_python(_IMPORT_PROBE.format(module="substrax.devices"), timeout=180.0)

    assert result.check().last_json() is True
