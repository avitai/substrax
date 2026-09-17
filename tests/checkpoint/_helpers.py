"""Shared helpers for the checkpoint tests: a small model and the format-2 fixtures on disk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
from flax import nnx


FORMAT2_FIXTURES = Path(__file__).with_name("fixtures") / "format2"
FIXTURE_STEP = 7


class SimpleModel(nnx.Module):
    """Two linear layers: enough real array state to place and compare."""

    def __init__(self, features: int = 4, *, rngs: nnx.Rngs) -> None:
        super().__init__()
        self.dense1 = nnx.Linear(features, features, rngs=rngs)
        self.dense2 = nnx.Linear(features, features, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.dense2(nnx.relu(self.dense1(x)))


def metadata_file(checkpoint_dir: Path, step: int) -> Path:
    """The JSON file Orbax's ``JsonSave`` wrote for the ``metadata`` item of ``step``."""
    path = checkpoint_dir / str(step) / "metadata" / "metadata"
    assert path.is_file(), path
    return path


def raw_metadata(checkpoint_dir: Path, step: int) -> dict[str, Any]:
    """The metadata JSON Orbax wrote for ``step``, read from disk without the store."""
    return json.loads(metadata_file(checkpoint_dir, step).read_text(encoding="utf-8"))


def write_raw_metadata(checkpoint_dir: Path, step: int, payload: dict[str, Any]) -> None:
    """Overwrite the metadata JSON of ``step`` on disk, as a producer of another format would."""
    metadata_file(checkpoint_dir, step).write_text(json.dumps(payload), encoding="utf-8")
