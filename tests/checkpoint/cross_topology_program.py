"""Child program for the cross-topology checkpoint tests.

jax fixes its device set when it initialises, so a save on one topology and a
restore on another need two interpreters. The test runs this file twice with
different environments (``JAX_NUM_CPU_DEVICES``, ``JAX_PLATFORMS``)::

    python cross_topology_program.py save <checkpoint_dir>
    python cross_topology_program.py restore <checkpoint_dir> [--without-target]

``save`` places every array on the last visible device and writes an NNX module
and a mixed-leaf dictionary. ``restore`` reads both back, onto freshly built
targets by default. Each mode prints one JSON object as the last line of stdout;
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


MODEL_STEP = 1
PAYLOAD_STEP = 2
LOSS = 0.5
ADDITIONAL_METADATA = {"run": "topology"}


class Model(nnx.Module):
    """One linear layer: enough real array state to place and compare."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        super().__init__()
        self.dense = nnx.Linear(2, 2, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.dense(x)


def payload() -> dict[str, Any]:
    """A dictionary payload with every leaf kind the store carries."""
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


def _flat_values(module: nnx.Module) -> dict[str, list[Any]]:
    return {
        "/".join(str(part) for part in path): np.asarray(variable[...]).tolist()
        for path, variable in nnx.to_flat_state(nnx.state(module))
    }


def _put(tree: Any, device: Any) -> Any:
    return jax.tree.map(
        lambda leaf: jax.device_put(leaf, device) if isinstance(leaf, jax.Array) else leaf,
        tree,
    )


def save(directory: Path) -> dict[str, Any]:
    """Write the module and the payload with every array on the last device."""
    device = jax.devices()[-1]
    with jax.default_device(device):
        model = Model(rngs=nnx.Rngs(0))
        nnx.update(model, _put(nnx.state(model), device))
        with OrbaxCheckpointStore(directory) as store:
            store.save(model, step=MODEL_STEP, loss=LOSS, additional_metadata=ADDITIONAL_METADATA)
            store.save(_put(payload(), device), step=PAYLOAD_STEP)
    return {
        "devices": [_device(d) for d in jax.devices()],
        "saved_on": _placements(nnx.state(model)),
        "state": _flat_values(model),
    }


def _describe_array(value: Any) -> dict[str, Any]:
    return {
        "kind": type(value).__name__,
        "dtype": str(value.dtype),
        "values": np.asarray(value).tolist(),
        "devices": _placements(value),
    }


def restore(directory: Path, *, with_target: bool) -> dict[str, Any]:
    """Read the module and the payload back, onto targets built in this process."""
    with OrbaxCheckpointStore(directory) as store:
        if not with_target:
            _stored, metadata = store.restore(step=MODEL_STEP)
            return {"devices": [_device(d) for d in jax.devices()], "metadata": metadata}
        model = Model(rngs=nnx.Rngs(1))
        restored_model, metadata = store.restore(model, step=MODEL_STEP)
        restored_payload, _ = store.restore(payload(), step=PAYLOAD_STEP)
    assert isinstance(restored_model, Model)
    assert isinstance(restored_payload, dict)
    return {
        "devices": [_device(d) for d in jax.devices()],
        "restored_on": _placements(nnx.state(restored_model)),
        "state": _flat_values(restored_model),
        "metadata": metadata,
        "payload": {
            "half": _describe_array(restored_payload["half"]),
            "counts": _describe_array(restored_payload["counts"]),
            "key": {
                "dtype": str(restored_payload["key"].dtype),
                "data": np.asarray(jax.random.key_data(restored_payload["key"])).tolist(),
                "devices": _placements(restored_payload["key"]),
            },
            "position": restored_payload["position"],
            "name": restored_payload["name"],
            "shuffle": restored_payload["shuffle"],
            "history": restored_payload["history"],
        },
    }


def main(argv: list[str]) -> None:
    """Dispatch on the mode argument and print the JSON result."""
    mode, directory = argv[1], Path(argv[2])
    if mode == "save":
        result = save(directory)
    elif mode == "restore":
        result = restore(directory, with_target="--without-target" not in argv[3:])
    else:
        raise SystemExit(f"unknown mode {mode!r}; expected 'save' or 'restore'")
    sys.stdout.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main(sys.argv)
