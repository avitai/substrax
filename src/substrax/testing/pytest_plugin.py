"""Opt-in pytest plugin for JAX test suites.

Enable it with ``pytest_plugins = ["substrax.testing.pytest_plugin"]`` in the top-level
``conftest.py``. No entry point loads it, so installing substrax changes no other project's tests.
It provides:

* ``@pytest.mark.x64``, which runs one test with jax 64-bit types enabled;
* a test failure, after the values are set back, when a test changes jax's global configuration,
  as jax's own test harness does;
* a session failure when a module leaves x64 on at import, since every module collected after it
  would build float64 constants;
* ``@pytest.mark.devices(count, kind=None)`` and ``@pytest.mark.accelerator(kind=None)``, which skip
  a test the visible devices cannot run;
* an ``output_dir`` fixture: a fresh directory with ``AVITAI_OUTPUT_DIR`` pointing at it.

Importing the plugin imports neither jax nor ``substrax.devices``; both load when a test runs.
"""

from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Iterator
from typing import TYPE_CHECKING

import pytest

from substrax.artifacts import OUTPUT_DIR_ENV
from substrax.testing.jax_config import restored_jax_config


if TYPE_CHECKING:
    from pathlib import Path

    from substrax.devices import DeviceInfo, DeviceKind


_MARKERS = (
    "x64: run the test with jax 64-bit types enabled",
    "devices(count, kind=None): skip unless at least `count` devices, of `kind` when given, are visible",
    "accelerator(kind=None): skip unless jax's default backend is an accelerator, of `kind` when given",
)
_X64_VARIABLE = "JAX_ENABLE_X64"
# The values jax reads as true for a boolean variable (jax/_src/config.py, bool_env).
_TRUE_VALUES = frozenset({"y", "yes", "t", "true", "on", "1"})


def pytest_configure(config: pytest.Config) -> None:
    """Register the plugin's markers, so ``--strict-markers`` accepts them.

    Args:
        config: The pytest configuration.
    """
    for line in _MARKERS:
        config.addinivalue_line("markers", line)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:  # noqa: DOC502
    """Refuse a malformed ``devices`` or ``accelerator`` marker before any test runs.

    Args:
        items: The collected tests.

    Raises:
        pytest.UsageError: If a marker's arguments do not describe a device requirement.
    """
    for item in items:
        for marker in item.iter_markers("devices"):
            _device_requirement(marker)
        for marker in item.iter_markers("accelerator"):
            _accelerator_kind(marker)


def pytest_collection_finish(session: pytest.Session) -> None:
    """Fail the session when collection left x64 on although the environment did not ask for it.

    Args:
        session: The pytest session.
    """
    jax = sys.modules.get("jax")
    if jax is None:
        return
    requested = os.environ.get(_X64_VARIABLE, "").strip().lower() in _TRUE_VALUES
    if bool(jax.config.jax_enable_x64) and not requested:
        session.shouldfail = (
            "jax_enable_x64 is on after collection: a test module enabled it at import time; "
            "mark the tests that need it with @pytest.mark.x64 instead"
        )


@pytest.fixture(autouse=True)
def substrax_jax_config_isolation(request: pytest.FixtureRequest) -> Iterator[None]:
    """Fail a test that changes jax's global configuration, after setting the values back.

    This follows jax's own test harness, which snapshots ``jax.config.values`` around each test and
    fails a test that changed a global value. ``x64``-marked tests run inside
    ``jax.enable_x64(True)``, a thread-local change that leaves the global configuration alone.

    Args:
        request: The requesting test.

    Yields:
        None: control passes to the test.
    """
    with restored_jax_config() as changed:
        if request.node.get_closest_marker("x64") is None:
            yield
        else:
            with importlib.import_module("jax").enable_x64(True):
                yield
    if changed:
        pytest.fail(
            f"the test changed global jax configuration {changed}; the values were set back. "
            "Wrap intended changes in substrax.testing.restored_jax_config(), and mark tests "
            "that need 64-bit types with @pytest.mark.x64",
            pytrace=False,
        )


@pytest.fixture
def output_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fresh directory for a test's outputs, with ``AVITAI_OUTPUT_DIR`` pointing at it.

    ``resolve_output_dir`` calls in the test, and in every child interpreter it starts, write under
    this directory. The variable is set back after the test.

    Args:
        tmp_path: The test's temporary directory.
        monkeypatch: Sets the variable for the test and restores it afterwards.

    Returns:
        The created directory.
    """
    directory = tmp_path / "outputs"
    directory.mkdir()
    monkeypatch.setenv(OUTPUT_DIR_ENV, str(directory))
    return directory


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip a test whose ``devices`` or ``accelerator`` marker the visible devices cannot satisfy.

    Args:
        item: The test about to run.
    """
    devices_markers = list(item.iter_markers("devices"))
    accelerator_markers = list(item.iter_markers("accelerator"))
    if not devices_markers and not accelerator_markers:
        return
    info: DeviceInfo = importlib.import_module("substrax.devices").detect_devices()
    for marker in devices_markers:
        _skip_unless_devices(info, *_device_requirement(marker))
    for marker in accelerator_markers:
        _skip_unless_accelerator(info, _accelerator_kind(marker))


def _skip_unless_devices(info: DeviceInfo, count: int, kind: DeviceKind | None) -> None:
    """Skip when fewer than ``count`` devices, of ``kind`` when given, are visible."""
    kinds = importlib.import_module("substrax.devices").DeviceKind
    visible = (
        info.count
        if kind is None
        else sum(kinds.from_platform(name) is kind for name in info.device_kinds)
    )
    if visible < count:
        needed = f"{count} device(s)" if kind is None else f"{count} {kind} device(s)"
        pytest.skip(f"needs {needed}; visible: {info.count} {info.kind}")


def _skip_unless_accelerator(info: DeviceInfo, kind: DeviceKind | None) -> None:
    """Skip unless the default backend is an accelerator, of ``kind`` when given."""
    if not info.has_accelerator or (kind is not None and info.kind is not kind):
        needed = "an accelerator" if kind is None else f"a {kind} backend"
        pytest.skip(f"needs {needed}; default backend: {info.platform}")


def _device_requirement(marker: pytest.Mark) -> tuple[int, DeviceKind | None]:
    """Read ``devices(count, kind=None)``, refusing anything else."""
    count = marker.args[0] if len(marker.args) == 1 else None
    if (
        set(marker.kwargs) - {"kind"}
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 1
    ):
        raise pytest.UsageError(
            f"use @pytest.mark.devices(count, kind=None) with count >= 1, got {marker}"
        )
    return count, _device_kind(marker.kwargs.get("kind"), marker)


def _accelerator_kind(marker: pytest.Mark) -> DeviceKind | None:
    """Read ``accelerator(kind=None)``, refusing anything else."""
    if marker.args or set(marker.kwargs) - {"kind"}:
        raise pytest.UsageError(f"use @pytest.mark.accelerator(kind=None), got {marker}")
    return _device_kind(marker.kwargs.get("kind"), marker)


def _device_kind(value: object, marker: pytest.Mark) -> DeviceKind | None:
    """Read a marker's ``kind`` as a ``DeviceKind``, refusing a name substrax does not know."""
    if value is None:
        return None
    kinds = importlib.import_module("substrax.devices").DeviceKind
    try:
        return kinds(value)
    except ValueError as error:
        known = ", ".join(kind.value for kind in kinds)
        raise pytest.UsageError(
            f"unknown device kind {value!r} in {marker}; use one of {known}"
        ) from error
