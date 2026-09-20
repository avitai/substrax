"""The ``local`` backend passes the backend contract and reads its settings strictly."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from substrax.compute.backend import ComputeBackend
from substrax.compute.local import LocalBackend
from substrax.testing.compute import BackendContract


class TestLocalBackend(BackendContract):
    def make_backend(self, state_dir: Path) -> ComputeBackend:
        return LocalBackend({"python": sys.executable, "state_dir": str(state_dir)})


def test_an_unknown_setting_is_refused() -> None:
    with pytest.raises(ValueError, match="gpu"):
        LocalBackend({"gpu": "L4"})


def test_a_setting_of_the_wrong_type_is_refused() -> None:
    with pytest.raises(ValueError, match="python"):
        LocalBackend({"python": 3})
