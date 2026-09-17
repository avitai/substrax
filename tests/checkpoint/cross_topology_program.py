"""Child program for the cross-topology checkpoint tests.

jax fixes its device set when it initialises, so a save on one topology and a
restore on another need two interpreters. The test runs this file twice with
different environments (``JAX_NUM_CPU_DEVICES``, ``JAX_PLATFORMS``)::

    python cross_topology_program.py save <checkpoint_dir>
    python cross_topology_program.py restore <checkpoint_dir> [--without-templates]

``save`` places every array on the last visible device and writes one checkpoint
whose ``model`` item is an NNX module's state and whose ``data_iterator`` item is a
mixed-leaf dictionary. ``restore`` reads both back, onto templates built in this
process by default. Each mode prints one JSON object as the last line of stdout;
the parent test compares the two sides.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from substrax.checkpoint import OrbaxCheckpointStore


STEP = 1
LOSS = 0.5
EXTRA = {"run": "topology"}


class Model(nnx.Module):
    """One linear layer: enough real array state to place and compare."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        super().__init__()
        self.dense = nnx.Linear(2, 2, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.dense(x)


def iterator_state() -> dict[str, Any]:
    """A dictionary item with every leaf kind the store carries."""
    return {
        "half": jnp.arange(3, dtype=jnp.float16),
        "counts": np.array([1, 2], dtype=np.int32),
        "key": jax.random.key(3),
        "position": 7,
        "name": "sampler",
        "shuffle": True,
        "history": [0.5, 0.25],
    }


def _device(device: Any) -> dict[str, Any]:
    return {"platform": device.platform, "id": device.id}


def _placements(tree: Any) -> list[dict[str, Any]]:
    """Every device any jax array leaf of ``tree`` lives on, without duplicates."""
    seen: dict[tuple[str, int], dict[str, Any]] = {}
    for leaf in jax.tree.leaves(tree):
        if isinstance(leaf, jax.Array):
            for device in leaf.devices():
                seen[(device.platform, device.id)] = _device(device)
    return [seen[key] for key in sorted(seen)]


def _flat_values(state: Any) -> dict[str, list[Any]]:
    return {
        "/".join(str(part) for part in path): np.asarray(variable[...]).tolist()
        for path, variable in nnx.to_flat_state(state)
    }


def _put(tree: Any, device: Any) -> Any:
    return jax.tree.map(
        lambda leaf: jax.device_put(leaf, device) if isinstance(leaf, jax.Array) else leaf,
        tree,
    )


def save(directory: Path) -> dict[str, Any]:
    """Write the module state and the iterator state with every array on the last device."""
    device = jax.devices()[-1]
    with jax.default_device(device):
        model = Model(rngs=nnx.Rngs(0))
        nnx.update(model, _put(nnx.state(model), device))
        with OrbaxCheckpointStore(directory) as store:
            store.save(
                STEP,
                {"model": nnx.state(model), "data_iterator": _put(iterator_state(), device)},
                metrics={"loss": LOSS},
                extra=EXTRA,
            )
    return {
        "devices": [_device(d) for d in jax.devices()],
        "saved_on": _placements(nnx.state(model)),
        "state": _flat_values(nnx.state(model)),
    }


def _describe_array(value: Any) -> dict[str, Any]:
    return {
        "kind": type(value).__name__,
        "dtype": str(value.dtype),
        "values": np.asarray(value).tolist(),
        "devices": _placements(value),
    }


def restore(directory: Path, *, with_templates: bool) -> dict[str, Any]:
    """Read both items back, onto templates built in this process."""
    with OrbaxCheckpointStore(directory) as store:
        if not with_templates:
            checkpoint = store.restore(STEP)
            return {
                "devices": [_device(d) for d in jax.devices()],
                "metadata": checkpoint.metadata.to_dict(),
            }
        model = Model(rngs=nnx.Rngs(1))
        checkpoint = store.restore(
            STEP, templates={"model": nnx.state(model), "data_iterator": iterator_state()}
        )
    nnx.update(model, checkpoint.items["model"])
    payload = checkpoint.items["data_iterator"]
    return {
        "devices": [_device(d) for d in jax.devices()],
        "restored_on": _placements(nnx.state(model)),
        "state": _flat_values(nnx.state(model)),
        "metadata": checkpoint.metadata.to_dict(),
        "payload": {
            "half": _describe_array(payload["half"]),
            "counts": _describe_array(payload["counts"]),
            "key": {
                "dtype": str(payload["key"].dtype),
                "data": np.asarray(jax.random.key_data(payload["key"])).tolist(),
                "devices": _placements(payload["key"]),
            },
            "position": payload["position"],
            "name": payload["name"],
            "shuffle": payload["shuffle"],
            "history": payload["history"],
        },
    }


def main(argv: list[str]) -> None:
    """Dispatch on the mode argument and print the JSON result."""
    mode, directory = argv[1], Path(argv[2])
    if mode == "save":
        result = save(directory)
    elif mode == "restore":
        result = restore(directory, with_templates="--without-templates" not in argv[3:])
    else:
        raise SystemExit(f"unknown mode {mode!r}; expected 'save' or 'restore'")
    sys.stdout.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main(sys.argv)
