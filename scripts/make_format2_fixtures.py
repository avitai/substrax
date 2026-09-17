"""Write the format-2 checkpoint fixtures the migration tests read.

Format 2 is what substrax 0.1.5 through 0.1.9 wrote: one Orbax ``model`` item holding the
payload pytree and a ``metadata`` JSON item stamped ``checkpoint_version = "2.0"``. Each
producer in the ecosystem laid its payload out differently, and the migration to format 3
has to recognise every layout, so this script writes one small checkpoint per layout with
the substrax release that produced it. Run it in isolation, never from the project venv::

    uv run --no-project --with "substrax==0.1.5" --with "optax==0.2.8" \
        python scripts/make_format2_fixtures.py tests/checkpoint/fixtures/format2

Every fixture is a few kilobytes: the models are two-parameter linear layers.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import jax
import numpy as np
import optax
from flax import nnx

from substrax.checkpoint import OrbaxCheckpointStore


def _model() -> nnx.Module:
    return nnx.Linear(in_features=4, out_features=2, rngs=nnx.Rngs(0))


def _artifex_trainer_tree(model: nnx.Module) -> dict[str, object]:
    """Artifex ``Trainer.checkpoint_state()``: model, optax state, raw key, extension states."""
    params = nnx.state(model, nnx.Param)
    optimizer = optax.adam(1e-3)
    return {
        "model": nnx.state(model),
        "opt_state": optimizer.init(params),
        "rng": jax.random.PRNGKey(0),
        "extensions": {"regulariser": nnx.state(_model())},
    }


def _datarax_iterator_state() -> dict[str, object]:
    """Datarax 0.1.9 ``Pipeline`` iterator state: position, epoch, per-stream fork counts."""
    return {"position": np.int64(96), "epoch": 1, "rng_counts": [3, 0]}


def _write(
    root: Path, name: str, payload: object, *, loss: float | None = None, **kwargs: object
) -> None:
    directory = root / name
    if directory.exists():
        shutil.rmtree(directory)
    with OrbaxCheckpointStore(directory, max_to_keep=None) as store:
        store.save(payload, 7, loss, **kwargs)  # type: ignore[arg-type]
    print(
        f"{name}: {sum(p.stat().st_size for p in directory.rglob('*') if p.is_file()) // 1024} KB"
    )


def main(argv: list[str] | None = None) -> int:
    """Write every fixture under the root named on the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path, help="Directory the fixtures are written under")
    root = parser.parse_args(argv).root
    root.mkdir(parents=True, exist_ok=True)

    model = _model()
    trainer_tree = _artifex_trainer_tree(model)

    # substrax itself, opifex, cellifex and the mlflow backend: a module's state alone.
    _write(root, "substrax_module", model, loss=0.25)
    _write(
        root,
        "opifex_module",
        model,
        loss=0.25,
        physics_metadata={"pde": "burgers"},
        additional_metadata={"step": 7, "epoch": 2},
    )
    # artifex: the trainer's tree.
    _write(root, "artifex_trainer", trainer_tree)
    # pertrax: a phase tree wrapping the trainer's tree and a datarax iterator state.
    _write(
        root,
        "pertrax_phase",
        {"trainer": trainer_tree, "step": 7, "iterator": _datarax_iterator_state()},
    )
    # DiffAV: the model alone, and the model with its NNX optimizer, both with the sidecar.
    sidecar = {"epoch": 2, "metrics": {"val_loss": 0.5}, "architecture_version": 3}
    _write(
        root, "diffav_model", {"model": nnx.state(model)}, loss=0.25, additional_metadata=sidecar
    )
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    _write(
        root,
        "diffav_model_optimizer",
        {"model": nnx.state(model), "optimizer": nnx.state(optimizer)},
        loss=0.25,
        additional_metadata=sidecar,
    )
    # datarax: an iterator state with the pipeline's own metadata.
    _write(
        root,
        "datarax_iterator",
        _datarax_iterator_state(),
        additional_metadata={"pipeline": "train"},
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
