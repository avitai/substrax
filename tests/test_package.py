"""Package-level contracts: version, public surface, and import hygiene."""

from __future__ import annotations

import importlib
import sys

import substrax


def test_version_is_a_release_string() -> None:
    """The version follows MAJOR.MINOR.PATCH so hatch and PyPI agree on it."""
    major, minor, patch = substrax.__version__.split(".")
    assert major.isdigit()
    assert minor.isdigit()
    assert patch.isdigit()


def test_top_level_module_exposes_only_the_version() -> None:
    """Consumers import from the subpackages; the root carries nothing else."""
    assert substrax.__all__ == ["__version__"]


def test_importing_the_root_does_not_import_optional_backends() -> None:
    """Weights & Biases and MLflow are extras and must not load on a plain import."""
    importlib.reload(substrax)

    assert "wandb" not in sys.modules
    assert "mlflow" not in sys.modules
