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
from pathlib import Path

import jax
import jax.numpy as jnp
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
