"""Tests for the unified :mod:`checkpoint_store` abstraction.

These tests pin the single, orbax-backed :class:`CheckpointStore` that
replaces the previously-duplicated ``OrbaxCheckpointManager`` (step-int
addressed) and ``CheckpointManager`` (path-string addressed) classes.

The store uses Orbax's step-int addressing as the canonical disk contract.
Behaviour preserved from the deleted managers:

* lossless save/restore round-trip of an ``nnx.Module`` parameter pytree,
* rich JSON metadata (physics + additional) round-trip,
* ``max_to_keep`` retention handled natively by Orbax,
* best-metric checkpoint selection (carried over from the path-string
  manager's ``get_best_checkpoint``),
* pickle-free serialization (Orbax ``PyTreeSave`` + ``JsonSave``).
"""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx
from flax.training import train_state

import substrax.checkpoint.checkpoint_store as checkpoint_store_module
from substrax.checkpoint.checkpoint_store import (
    CheckpointStore,
    OrbaxCheckpointStore,
)


class _SimpleModel(nnx.Module):
    """Minimal two-layer NNX model providing real array state."""

    def __init__(self, features: int = 4, *, rngs: nnx.Rngs) -> None:
        self.dense1 = nnx.Linear(features, features, rngs=rngs)
        self.dense2 = nnx.Linear(features, features, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.dense2(nnx.relu(self.dense1(x)))


@pytest.fixture
def model() -> _SimpleModel:
    """Construct a small deterministic model."""
    return _SimpleModel(features=4, rngs=nnx.Rngs(0))


_CROSS_TOPOLOGY_PROGRAM = Path(__file__).with_name("cross_topology_program.py")
_SHARDING_FILE_WARNING = "Sharding info not provided"
_CPU_0 = {"platform": "cpu", "id": 0}


def _cpu_devices(count: int) -> dict[str, str]:
    return {"JAX_PLATFORMS": "cpu", "JAX_NUM_CPU_DEVICES": str(count)}


def _run_interpreter(argv: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """Run this test's own interpreter on ``argv`` in a fresh process with ``env`` overrides."""
    child_env = {**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false", **env}
    return subprocess.run(  # noqa: S603 - fixed argv: the test's interpreter and its own files
        [sys.executable, *argv],
        env=child_env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


def _run_child(
    mode: str, directory: Path, *, env: dict[str, str], extra: tuple[str, ...] = ()
) -> subprocess.CompletedProcess[str]:
    """Run the cross-topology program in ``mode`` against ``directory``."""
    return _run_interpreter([str(_CROSS_TOPOLOGY_PROGRAM), mode, str(directory), *extra], env)


def _child_result(completed: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    """The JSON object a successful child printed on its last stdout line."""
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.splitlines()[-1])


def _cuda_is_visible() -> bool:
    """Whether a fresh interpreter can initialise jax's CUDA backend."""
    probe = _run_interpreter(["-c", "import jax; jax.devices('gpu')"], {"JAX_PLATFORMS": "cuda"})
    return probe.returncode == 0


class TestOrbaxCheckpointStoreInit:
    """Initialization and validation."""

    def test_creates_directory(self, tmp_path: Path) -> None:
        """Creates the checkpoint directory on construction."""
        ckpt_dir = tmp_path / "ckpt"
        store = OrbaxCheckpointStore(ckpt_dir)
        assert ckpt_dir.exists()
        assert store.max_to_keep == 5

    def test_custom_max_to_keep(self, tmp_path: Path) -> None:
        """Respects a custom ``max_to_keep``."""
        store = OrbaxCheckpointStore(tmp_path / "ckpt", max_to_keep=3)
        assert store.max_to_keep == 3

    def test_empty_dir_raises(self) -> None:
        """An empty directory string is rejected."""
        with pytest.raises(ValueError, match="cannot be empty"):
            OrbaxCheckpointStore("")

    def test_whitespace_dir_raises(self) -> None:
        """A whitespace-only directory string is rejected."""
        with pytest.raises(ValueError, match="cannot be empty"):
            OrbaxCheckpointStore("   ")

    def test_directory_is_resolved(self, tmp_path: Path) -> None:
        """The checkpoint directory is resolved to an absolute path."""
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        assert store.checkpoint_dir.is_absolute()


class TestProtocolConformance:
    """The concrete store satisfies the abstract protocol."""

    def test_store_is_checkpoint_store(self, tmp_path: Path) -> None:
        """``OrbaxCheckpointStore`` is a structural ``CheckpointStore``."""
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        assert isinstance(store, CheckpointStore)


class TestSaveRestoreRoundTrip:
    """Lossless save/restore of model state and metadata."""

    def test_roundtrip_nnx_module_lossless(self, tmp_path: Path, model: _SimpleModel) -> None:
        """Restored parameters match the originals exactly."""
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        original_state = nnx.state(model)

        path = store.save(model, step=100, loss=0.05)
        assert path.endswith("100")

        restored_model, metadata = store.restore(model, step=100)
        assert restored_model is not None
        restored_state = nnx.state(restored_model)

        original_leaves = nnx.to_flat_state(original_state)
        restored_leaves = nnx.to_flat_state(restored_state)
        assert [p for p, _ in restored_leaves] == [p for p, _ in original_leaves]
        for (_, original), (_, restored) in zip(original_leaves, restored_leaves, strict=True):
            assert jnp.allclose(original[...], restored[...])

        assert metadata["step"] == 100
        assert metadata["loss"] == 0.05

    def test_metadata_roundtrip(self, tmp_path: Path, model: _SimpleModel) -> None:
        """Physics and additional metadata survive the round-trip."""
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        physics = {"chemical_accuracy": 0.001, "scf_iterations": 15}
        additional = {"experiment_id": "exp_001"}

        store.save(
            model,
            step=10,
            loss=0.2,
            physics_metadata=physics,
            additional_metadata=additional,
        )
        _, metadata = store.restore(model, step=10)

        assert metadata["physics_metadata"]["chemical_accuracy"] == 0.001
        assert metadata["physics_metadata"]["scf_iterations"] == 15
        assert metadata["experiment_id"] == "exp_001"
        assert metadata["checkpoint_version"] == "2.0"

    def test_restore_missing_returns_original(self, tmp_path: Path, model: _SimpleModel) -> None:
        """Restoring a never-saved step returns the original model."""
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        restored, metadata = store.restore(model, step=999)
        assert restored is model
        assert metadata == {}

    def test_restore_without_target_returns_the_stored_payload(
        self, tmp_path: Path, model: _SimpleModel
    ) -> None:
        """With no target the checkpoint describes itself: the State tree and metadata come back.

        A module is stored as its ``nnx.State``, so each Variable is a ``{"value": ...}`` node.
        """
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        store.save(model, step=5, loss=0.1)

        restored, metadata = store.restore(step=5)

        assert isinstance(restored, dict)
        assert jnp.array_equal(
            restored["dense1"]["kernel"]["value"],
            nnx.to_pure_dict(nnx.state(model))["dense1"]["kernel"],
        )
        assert metadata["step"] == 5

    def test_restore_without_target_keeps_python_leaves_and_shapes(self, tmp_path: Path) -> None:
        """A template-free restore keeps list lengths and plain leaves as saved."""
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        store.save({"history": [0.5, 0.25], "epoch": 2, "w": jnp.ones(3)}, step=1)

        restored, _ = store.restore(step=1)

        assert isinstance(restored, dict)
        assert restored["history"] == [0.5, 0.25]
        assert restored["epoch"] == 2
        assert jnp.array_equal(restored["w"], jnp.ones(3))

    def test_restore_into_a_mismatched_target_raises(self, tmp_path: Path) -> None:
        """A target whose tree differs from the checkpoint is an error, not a missing step."""
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        store.save({"history": [0.5], "w": jnp.ones(3)}, step=1)

        with pytest.raises(ValueError, match="do not match"):
            store.restore({"history": [], "w": jnp.ones(3)}, step=1)

    def test_invalid_model_type_raises(self, tmp_path: Path) -> None:
        """Saving an unsupported model type raises ``TypeError``."""
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        with pytest.raises(TypeError, match=r"nnx\.Module"):
            store.save("not-a-model", step=1, loss=0.1)  # type: ignore[arg-type]

    def test_negative_step_raises(self, tmp_path: Path, model: _SimpleModel) -> None:
        """Saving a negative step raises ``ValueError``."""
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        with pytest.raises(ValueError, match="non-negative"):
            store.save(model, step=-1, loss=0.1)


class TestTargetPlacement:
    """With a target, placement and dtype come from the target's leaves, not the checkpoint."""

    def test_restore_with_target_does_not_read_the_saved_sharding(
        self, tmp_path: Path, model: _SimpleModel
    ) -> None:
        """The target supplies every array's sharding, so Orbax never opens the sharding file.

        The sharding file names the devices the checkpoint was written on; consulting it is
        what breaks a restore on a different topology.
        """
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        store.save(model, step=1, loss=0.1)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            store.restore(model, step=1)

        assert not [w for w in caught if _SHARDING_FILE_WARNING in str(w.message)]

    def test_restore_without_target_reads_the_saved_sharding(
        self, tmp_path: Path, model: _SimpleModel
    ) -> None:
        """Control: the template-free restore keeps the saved placement, via the sharding file.

        This is the warning ``filterwarnings`` in pyproject.toml ignores, and the reason it
        must stay: the documented target-free behaviour is to come back as stored.
        """
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        store.save(model, step=1, loss=0.1)

        with pytest.warns(UserWarning, match=_SHARDING_FILE_WARNING):
            store.restore(step=1)

    def test_restore_with_target_keeps_dtypes_kinds_and_placement(self, tmp_path: Path) -> None:
        """Each leaf comes back with its target's dtype, array kind and device."""
        store = OrbaxCheckpointStore(tmp_path / "ckpt")

        def payload() -> dict[str, Any]:
            return {
                "half": jnp.arange(3, dtype=jnp.float16),
                "brain": jnp.ones(2, dtype=jnp.bfloat16),
                "ints": jnp.arange(2, dtype=jnp.int32),
                "host": np.array([1, 2], dtype=np.int32),
            }

        store.save(payload(), step=1)
        restored, _ = store.restore(payload(), step=1)

        assert isinstance(restored, dict)
        for name, expected in payload().items():
            assert restored[name].dtype == expected.dtype, name
            assert type(restored[name]) is type(expected), name
            np.testing.assert_array_equal(np.asarray(restored[name]), np.asarray(expected))
        for name in ("half", "brain", "ints"):
            assert restored[name].devices() == {jax.devices()[0]}, name


class TestCrossTopologyRestore:
    """A checkpoint written on one device set restores onto a target in another.

    Each side runs in its own interpreter because jax fixes its device set at
    initialisation. The save side exposes two CPU devices and places every array on
    the second; the restore side exposes one. Restoring the module, the dictionary
    payload and the metadata onto that one device is the contract.
    """

    def test_two_device_save_restores_onto_a_one_device_target(self, tmp_path: Path) -> None:
        saved = _child_result(_run_child("save", tmp_path, env=_cpu_devices(2)))
        assert saved["saved_on"] == [{"platform": "cpu", "id": 1}]

        restored = _child_result(_run_child("restore", tmp_path, env=_cpu_devices(1)))

        assert restored["devices"] == [_CPU_0]
        assert restored["restored_on"] == [_CPU_0]
        assert restored["state"] == saved["state"]
        assert restored["metadata"]["loss"] == 0.5
        assert restored["metadata"]["run"] == "topology"
        assert restored["metadata"]["model_type"] == "nnx_module"

        payload = restored["payload"]
        assert payload["half"] == {
            "kind": "ArrayImpl",
            "dtype": "float16",
            "values": [0.0, 1.0, 2.0],
            "devices": [_CPU_0],
        }
        assert payload["counts"] == {
            "kind": "ndarray",
            "dtype": "int32",
            "values": [1, 2],
            "devices": [],
        }
        assert payload["key"]["dtype"] == "key<fry>"
        assert payload["key"]["data"] == jax.random.key_data(jax.random.key(3)).tolist()
        assert payload["key"]["devices"] == [_CPU_0]
        assert payload["position"] == 7
        assert payload["name"] == "sampler"
        assert payload["shuffle"] is True
        assert payload["history"] == [0.5, 0.25]

    def test_two_device_save_without_a_target_cannot_resolve_the_saved_device(
        self, tmp_path: Path
    ) -> None:
        """Control: template-free, the same checkpoint fails on the device it was saved on.

        This proves the two interpreters really differ in topology, so the with-target
        pass above measures placement rather than a shared device set. It also pins
        the documented target-free behaviour: arrays come back as stored, or not at all.
        """
        _child_result(_run_child("save", tmp_path, env=_cpu_devices(2)))

        completed = _run_child(
            "restore", tmp_path, env=_cpu_devices(1), extra=("--without-target",)
        )

        assert completed.returncode != 0
        assert "Device cpu:1 was not found in jax.local_devices()" in completed.stderr

    @pytest.mark.gpu
    def test_gpu_save_restores_onto_a_cpu_target(self, tmp_path: Path) -> None:
        """A checkpoint written on an accelerator restores onto a CPU-only target.

        The CPU-to-CPU case above does not prove this: an accelerator checkpoint
        records a different platform, not just a different device id.
        """
        if not _cuda_is_visible():
            pytest.skip("no CUDA device visible to jax")

        saved = _child_result(_run_child("save", tmp_path, env={"JAX_PLATFORMS": "cuda"}))
        assert [d["platform"] for d in saved["saved_on"]] == ["gpu"]

        restored = _child_result(_run_child("restore", tmp_path, env=_cpu_devices(1)))

        assert restored["restored_on"] == [_CPU_0]
        assert restored["state"] == saved["state"]
        assert restored["payload"]["half"]["devices"] == [_CPU_0]
        assert restored["payload"]["key"]["devices"] == [_CPU_0]


class TestTrainStateRoundTrip:
    """TrainState helpers preserve optimizer state and parameters."""

    def test_create_train_state_structure(self, tmp_path: Path, model: _SimpleModel) -> None:
        """``create_train_state`` wires apply_fn/tx/params/step correctly."""
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        optimizer = optax.adam(1e-3)
        state = store.create_train_state(model, optimizer, step=100)

        assert isinstance(state, train_state.TrainState)
        assert state.step == 100
        assert state.apply_fn is model
        assert state.tx is optimizer
        assert state.params is not None

    def test_create_train_state_invalid_model_raises(self, tmp_path: Path) -> None:
        """A non-NNX model raises ``TypeError``."""
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        with pytest.raises(TypeError, match=r"nnx\.Module"):
            store.create_train_state("not-a-model", None, step=0)  # type: ignore[arg-type]

    def test_train_state_roundtrip(self, tmp_path: Path, model: _SimpleModel) -> None:
        """Save/restore a ``TrainState`` losslessly."""
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        optimizer = optax.adam(1e-3)
        state = store.create_train_state(model, optimizer, step=0)

        store.save_train_state(state, step=50, loss=0.03)
        restored, metadata = store.restore_train_state(state, step=50)

        assert isinstance(restored, train_state.TrainState)
        assert restored.step == 50
        assert restored.apply_fn is state.apply_fn
        assert restored.tx is state.tx
        assert metadata["loss"] == 0.03


class TestListingAndRetention:
    """Step listing, latest-step, retention and deletion."""

    def test_list_and_latest_step(self, tmp_path: Path, model: _SimpleModel) -> None:
        """List returns saved steps and latest reports the newest."""
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        for step in (100, 200, 300):
            store.save(model, step=step, loss=0.1)
        assert sorted(store.list_steps()) == [100, 200, 300]
        assert store.latest_step() == 300

    def test_restore_specific_step_metadata(self, tmp_path: Path, model: _SimpleModel) -> None:
        """Restoring a specific step returns that step's metadata."""
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        for step in (100, 200, 300):
            store.save(model, step=step, loss=0.1)
        _, metadata = store.restore(model, step=200)
        assert metadata["step"] == 200

    def test_max_to_keep_retention(self, tmp_path: Path, model: _SimpleModel) -> None:
        """Only the most recent ``max_to_keep`` checkpoints are retained."""
        store = OrbaxCheckpointStore(tmp_path / "ckpt", max_to_keep=2)
        for step in (1, 2, 3, 4):
            store.save(model, step=step, loss=0.1)
        kept = sorted(store.list_steps())
        assert len(kept) == 2
        assert kept == [3, 4]

    def test_max_to_keep_none_keeps_every_step(self, tmp_path: Path, model: _SimpleModel) -> None:
        """``max_to_keep=None`` disables pruning: a keep-every-checkpoint policy."""
        store = OrbaxCheckpointStore(tmp_path / "ckpt", max_to_keep=None)
        for step in range(8):
            store.save(model, step=step, loss=1.0)
        assert store.list_steps() == list(range(8))
        store.close()

    def test_delete_step(self, tmp_path: Path, model: _SimpleModel) -> None:
        """Deleting a step removes it from the listing."""
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        store.save(model, step=10, loss=0.1)
        store.save(model, step=20, loss=0.1)
        assert store.delete(10) is True
        assert 10 not in store.list_steps()


class TestBestMetricSelection:
    """Best-metric checkpoint selection carried over from the old manager."""

    def test_best_step_minimize_loss(self, tmp_path: Path, model: _SimpleModel) -> None:
        """The lowest-loss step is selected when minimizing."""
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        store.save(model, step=1, loss=0.9)
        store.save(model, step=2, loss=0.2)
        store.save(model, step=3, loss=0.5)
        assert store.best_step(metric="loss", minimize=True) == 2

    def test_best_step_maximize_metric(self, tmp_path: Path, model: _SimpleModel) -> None:
        """A custom metadata metric is honoured when maximizing."""
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        store.save(model, step=1, loss=0.5, additional_metadata={"accuracy": 0.7})
        store.save(model, step=2, loss=0.5, additional_metadata={"accuracy": 0.9})
        assert store.best_step(metric="accuracy", minimize=False) == 2

    def test_best_step_empty_returns_none(self, tmp_path: Path) -> None:
        """No checkpoints yields ``None``."""
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        assert store.best_step() is None


class TestResourceManagement:
    """Context-manager support and explicit close."""

    def test_context_manager(self, tmp_path: Path, model: _SimpleModel) -> None:
        """The store works as a context manager and closes cleanly."""
        with OrbaxCheckpointStore(tmp_path / "ckpt") as store:
            store.save(model, step=1, loss=0.1)
            assert store.latest_step() == 1


class TestSerializationSafety:
    """No pickle anywhere in the persisted artifacts or module source."""

    def test_module_does_not_use_pickle(self) -> None:
        """The store module never imports or references pickle."""
        source = inspect.getsource(checkpoint_store_module)
        assert "import pickle" not in source
        assert "pickle." not in source

    def test_saved_artifacts_are_not_pickle(self, tmp_path: Path, model: _SimpleModel) -> None:
        """Persisted files are not pickle streams."""
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        path = store.save(model, step=1, loss=0.1)

        for file in Path(path).rglob("*"):
            if file.is_file():
                head = file.read_bytes()[:2]
                assert not head.startswith(b"\x80"), f"pickle stream in {file}"


class TestPlainPayloads:
    """Payloads that are not model arrays: iterator state, counters, identity strings."""

    @staticmethod
    def _iterator_state() -> dict[str, object]:
        return {
            "position": 7,
            "rng_counts": [1, 2],
            "sampler_repr": "SequentialSampler(seed=42)",
            "shuffle": True,
            "indices": jnp.arange(3),
            "key": jax.random.key(0),
        }

    def test_roundtrip_python_string_and_key_leaves(self, tmp_path: Path) -> None:
        """Ints, lists, strings, booleans and typed PRNG keys come back as themselves."""
        store = OrbaxCheckpointStore(tmp_path)
        store.save(self._iterator_state(), step=3)

        restored, _ = store.restore(self._iterator_state(), step=3)

        assert isinstance(restored, dict)
        assert restored["position"] == 7
        assert restored["rng_counts"] == [1, 2]
        assert restored["sampler_repr"] == "SequentialSampler(seed=42)"
        assert restored["shuffle"] is True
        assert jnp.array_equal(restored["indices"], jnp.arange(3))
        assert jnp.array_equal(
            jax.random.key_data(restored["key"]), jax.random.key_data(jax.random.key(0))
        )

    def test_save_without_loss_records_none(self, tmp_path: Path) -> None:
        """A payload with no training loss carries no loss in its metadata."""
        store = OrbaxCheckpointStore(tmp_path)
        store.save({"position": 1}, step=0)

        _, metadata = store.restore(step=0)

        assert "loss" not in metadata
        assert metadata["step"] == 0
        assert metadata["model_type"] == "dict"

    def test_best_step_skips_checkpoints_without_the_metric(
        self, tmp_path: Path, model: _SimpleModel
    ) -> None:
        store = OrbaxCheckpointStore(tmp_path)
        store.save({"position": 1}, step=1)
        store.save(model, step=2, loss=0.4)
        store.save(model, step=3, loss=0.9)

        assert store.best_step(metric="loss", minimize=True) == 2
