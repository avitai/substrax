"""Tests for the optional-dependency loader."""

from __future__ import annotations

import json
import sys

import pytest

from substrax.tracking._optional import import_optional


def test_returns_the_module_when_installed() -> None:
    assert import_optional("json", extra="anything") is json


def test_names_the_extra_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "substrax_no_such_module", None)
    with pytest.raises(ImportError, match=r"uv add 'substrax\[tracking-x\]'") as info:
        import_optional("substrax_no_such_module", extra="tracking-x")
    assert isinstance(info.value.__cause__, ImportError)
