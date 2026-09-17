"""Fixtures for the checkpoint tests."""

from __future__ import annotations

import pytest
from _helpers import SimpleModel
from flax import nnx


@pytest.fixture
def model() -> SimpleModel:
    """A small deterministic model."""
    return SimpleModel(rngs=nnx.Rngs(0))
