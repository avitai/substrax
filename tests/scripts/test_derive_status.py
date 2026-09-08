"""The derived-status check fails when the README's subpackage table drifts."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def derive_status() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "derive_status", REPO_ROOT / "scripts" / "derive_status.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["derive_status"] = module
    spec.loader.exec_module(module)
    return module


def test_the_tree_matches_the_readme(derive_status: ModuleType) -> None:
    metrics = derive_status.collect_metrics(REPO_ROOT, "substrax")
    drifted = [metric.label for metric in metrics if metric.is_drifted]
    assert drifted == []
    subpackages = next(metric for metric in metrics if metric.label == "subpackages")
    assert subpackages.asserted == subpackages.measured
    assert "tracking" in subpackages.measured.split(",")


def test_a_missing_readme_row_is_drift(derive_status: ModuleType, tmp_path: Path) -> None:
    readme = REPO_ROOT / "README.md"
    lines = [line for line in readme.read_text().splitlines() if "`substrax.tracking`" not in line]
    edited = tmp_path / "README.md"
    edited.write_text("\n".join(lines) + "\n")

    metrics = derive_status.collect_metrics(REPO_ROOT, "substrax", readme=edited)

    assert [metric.label for metric in metrics if metric.is_drifted] == ["subpackages"]
