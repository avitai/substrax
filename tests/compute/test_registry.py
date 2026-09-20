"""Backends load by name from the ``substrax.compute.backends`` entry-point group."""

from __future__ import annotations

from collections.abc import Callable
from importlib.metadata import EntryPoint, EntryPoints
from pathlib import Path

import pytest

from substrax.compute.local import LocalBackend
from substrax.compute.registry import (
    backend_names,
    ENTRY_POINT_GROUP,
    load_backend,
    UnknownBackendError,
)


def _discover(**points: str) -> EntryPoints:
    return EntryPoints(
        EntryPoint(name=name, value=value, group=ENTRY_POINT_GROUP)
        for name, value in points.items()
    )


def _only(**points: str) -> Callable[..., EntryPoints]:
    def discover(*, group: str) -> EntryPoints:
        assert group == ENTRY_POINT_GROUP
        return _discover(**points)

    return discover


def test_substrax_registers_its_own_backends_in_the_installed_metadata() -> None:
    assert "local" in backend_names()


def test_a_backend_loads_by_name_with_its_settings(tmp_path: Path) -> None:
    backend = load_backend("local", {"state_dir": str(tmp_path)})

    assert isinstance(backend, LocalBackend)


def test_a_third_party_backend_loads_the_same_way() -> None:
    discover = _only(thirdparty="substrax.compute.local:LocalBackend")

    assert backend_names(discover=discover) == ("thirdparty",)
    assert isinstance(load_backend("thirdparty", {}, discover=discover), LocalBackend)


def test_an_unknown_name_is_refused_with_the_registered_names() -> None:
    discover = _only(alpha="substrax.compute.local:LocalBackend", beta="x:y")

    with pytest.raises(UnknownBackendError, match="'gamma'; registered: alpha, beta"):
        load_backend("gamma", discover=discover)


def test_an_entry_point_that_builds_something_else_is_refused() -> None:
    discover = _only(wrong="builtins:dict")

    with pytest.raises(TypeError, match="builtins:dict"):
        load_backend("wrong", discover=discover)
