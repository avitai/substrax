"""Shared helpers for the checkpoint tests: a small model and the format-2 fixtures on disk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
from flax import nnx


FORMAT2_FIXTURES = Path(__file__).with_name("fixtures") / "format2"
FIXTURE_STEP = 7
FORMAT2_LAYOUTS = (
    "substrax_module",
    "opifex_module",
    "artifex_trainer",
    "pertrax_phase",
    "diffav_model",
    "diffav_model_optimizer",
    "datarax_iterator",
)
_MAKE_FIXTURES = (
    'uv run --no-project --with "substrax==0.1.5" --with "optax==0.2.8" '
    "python scripts/make_format2_fixtures.py tests/checkpoint/fixtures/format2"
)


def require_format2_fixtures() -> None:
    """Fail loudly when the generated format-2 roots are missing, naming the command that writes them.

    The roots are written by the release that produced the format, never committed, and
    CI writes them before the tests run; a developer runs the same command once.
    """
    missing = [layout for layout in FORMAT2_LAYOUTS if not (FORMAT2_FIXTURES / layout).is_dir()]
    if missing:
        raise RuntimeError(
            f"format-2 checkpoint fixtures are missing under {FORMAT2_FIXTURES}: {missing}. "
            f"Write them first: {_MAKE_FIXTURES}"
        )


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
