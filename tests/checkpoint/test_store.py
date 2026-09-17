"""The format-3 checkpoint store: named items, one metadata record, strict writes.

A checkpoint is a step holding named items (``model``, ``optimizer``, ``rng``,
``data_iterator``, ``extensions``), each an Orbax ``PyTreeSave`` item, beside one
``CheckpointMetadata`` record. Writes never silently replace or reorder steps, a restore
onto templates places every array on the template's device, and nothing touches the
filesystem before the first save.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import warnings
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from _helpers import raw_metadata, SimpleModel, write_raw_metadata
from flax import nnx

import substrax.checkpoint
from substrax.checkpoint import (
    Checkpoint,
    CheckpointNotFoundError,
    CheckpointNotWrittenError,
    CheckpointStore,
    CURRENT_FORMAT_VERSION,
    OrbaxCheckpointStore,
    Producer,
    UnsupportedCheckpointError,
)
from substrax.runtime import JaxRuntime
from substrax.testing import ChildResult, cuda_is_visible, run_python


_CROSS_TOPOLOGY_PROGRAM = Path(__file__).with_name("cross_topology_program.py")
_SHARDING_FILE_WARNING = "Sharding info not provided"
_CPU_0 = {"platform": "cpu", "id": 0}


def _items(model: nnx.Module) -> dict[str, Any]:
    """The three items a trainer holds: module state, optimizer state and a key."""
    params = nnx.state(model, nnx.Param)
    return {
        "model": nnx.state(model),
        "optimizer": optax.adam(1e-3).init(params),
        "rng": jax.random.key(3),
    }


def _flat(state: Any) -> list[tuple[str, np.ndarray]]:
    return [
        ("/".join(str(p) for p in path), np.asarray(leaf[...]))
        for path, leaf in nnx.to_flat_state(state)
    ]


def _cpu_devices(count: int) -> JaxRuntime:
    return JaxRuntime(platforms=("cpu",), cpu_devices=count)


def _run_child(
    mode: str, directory: Path, *, runtime: JaxRuntime, extra: tuple[str, ...] = ()
) -> ChildResult:
    """Run the cross-topology program in ``mode`` against ``directory`` in a fresh interpreter."""
    return run_python(
        _CROSS_TOPOLOGY_PROGRAM, mode, str(directory), *extra, runtime=runtime, timeout=180.0
    )


class TestConstruction:
    def test_nothing_is_written_before_the_first_save(self, tmp_path: Path) -> None:
        directory = tmp_path / "ckpt"
        store = OrbaxCheckpointStore(directory)

        assert store.list_steps() == []
        assert store.latest_step() is None
        assert store.best_step("loss") is None
        assert not directory.exists()

    def test_the_directory_is_resolved_and_the_retention_kept(self, tmp_path: Path) -> None:
        store = OrbaxCheckpointStore(tmp_path / "ckpt", max_to_keep=3)
        assert store.directory == (tmp_path / "ckpt").resolve()
        assert store.directory.is_absolute()
        assert store.max_to_keep == 3
        assert OrbaxCheckpointStore(tmp_path / "other").max_to_keep == 5

    @pytest.mark.parametrize("directory", ["", "   "])
    def test_an_empty_directory_is_refused(self, directory: str) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            OrbaxCheckpointStore(directory)

    def test_the_orbax_store_is_a_checkpoint_store(self, tmp_path: Path) -> None:
        assert isinstance(OrbaxCheckpointStore(tmp_path / "ckpt"), CheckpointStore)


class TestSaveAndRestore:
    def test_items_round_trip_onto_templates(self, tmp_path: Path, model: SimpleModel) -> None:
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        items = _items(model)

        path = store.save(100, items, epoch=2, metrics={"loss": 0.05})
        assert path == store.directory / "100"
        assert path.is_dir()

        fresh = SimpleModel(rngs=nnx.Rngs(1))
        checkpoint = store.restore(100, templates=_items(fresh))

        assert isinstance(checkpoint, Checkpoint)
        assert checkpoint.step == 100
        assert set(checkpoint.items) == {"model", "optimizer", "rng"}
        for (path_a, a), (path_b, b) in zip(
            _flat(nnx.state(model)), _flat(checkpoint.items["model"]), strict=True
        ):
            assert path_a == path_b
            np.testing.assert_array_equal(a, b)
        assert jnp.array_equal(
            jax.random.key_data(checkpoint.items["rng"]), jax.random.key_data(items["rng"])
        )
        assert checkpoint.metadata.step == 100
        assert checkpoint.metadata.epoch == 2
        assert checkpoint.metadata.metrics == {"loss": 0.05}
        assert checkpoint.metadata.items == ("model", "optimizer", "rng")
        assert checkpoint.metadata.format_version == CURRENT_FORMAT_VERSION
        assert "jax" in checkpoint.metadata.libraries

    def test_a_restore_without_templates_returns_the_items_as_stored(
        self, tmp_path: Path, model: SimpleModel
    ) -> None:
        """Without templates the checkpoint describes itself: a module's Variables are nodes."""
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        store.save(
            5, {"model": nnx.state(model), "data_iterator": {"history": [0.5, 0.25], "epoch": 2}}
        )

        checkpoint = store.restore(5)

        assert jnp.array_equal(
            checkpoint.items["model"]["dense1"]["kernel"]["value"],
            nnx.to_pure_dict(nnx.state(model))["dense1"]["kernel"],
        )
        assert checkpoint.items["data_iterator"] == {"history": [0.5, 0.25], "epoch": 2}

    def test_templates_may_cover_a_subset_of_the_items(
        self, tmp_path: Path, model: SimpleModel
    ) -> None:
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        store.save(1, _items(model))

        checkpoint = store.restore(1, templates={"model": nnx.state(SimpleModel(rngs=nnx.Rngs(1)))})

        assert isinstance(checkpoint.items["model"], nnx.State)
        assert set(checkpoint.items) == {"model", "optimizer", "rng"}

    def test_a_template_that_does_not_fit_raises(self, tmp_path: Path) -> None:
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        store.save(1, {"data_iterator": {"history": [0.5], "w": jnp.ones(3)}})

        with pytest.raises(ValueError, match="do not match"):
            store.restore(1, templates={"data_iterator": {"history": [], "w": jnp.ones(3)}})

    def test_a_template_for_an_item_the_checkpoint_lacks_raises(
        self, tmp_path: Path, model: SimpleModel
    ) -> None:
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        store.save(1, {"model": nnx.state(model)})

        with pytest.raises(ValueError, match="optimizer"):
            store.restore(1, templates={"optimizer": {"count": jnp.zeros(())}})

    def test_an_unknown_item_name_is_refused(self, tmp_path: Path, model: SimpleModel) -> None:
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        with pytest.raises(ValueError, match="weights"):
            store.save(1, {"weights": nnx.state(model)})
        assert not store.directory.exists()

    def test_no_items_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="at least one item"):
            OrbaxCheckpointStore(tmp_path / "ckpt").save(1, {})

    def test_a_negative_step_is_refused(self, tmp_path: Path, model: SimpleModel) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            OrbaxCheckpointStore(tmp_path / "ckpt").save(-1, {"model": nnx.state(model)})

    def test_a_missing_step_raises(self, tmp_path: Path, model: SimpleModel) -> None:
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        with pytest.raises(CheckpointNotFoundError, match="999"):
            store.restore(999)
        store.save(1, {"model": nnx.state(model)})
        with pytest.raises(CheckpointNotFoundError, match="999"):
            store.restore(999)
        with pytest.raises(CheckpointNotFoundError, match="999"):
            store.read_metadata(999)


class TestWriteRules:
    def test_an_existing_step_is_not_rewritten(self, tmp_path: Path, model: SimpleModel) -> None:
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        store.save(10, {"model": nnx.state(model)}, metrics={"loss": 0.9})

        with pytest.raises(CheckpointNotWrittenError) as caught:
            store.save(10, {"model": nnx.state(model)}, metrics={"loss": 0.1})

        assert caught.value.step == 10
        assert caught.value.latest_step == 10
        assert caught.value.reason == "exists"
        assert store.read_metadata(10).metrics == {"loss": 0.9}

    def test_overwrite_rewrites_the_step(self, tmp_path: Path, model: SimpleModel) -> None:
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        store.save(10, {"model": nnx.state(model)}, metrics={"loss": 0.9})

        store.save(10, {"model": nnx.state(model)}, metrics={"loss": 0.1}, overwrite=True)

        assert store.list_steps() == [10]
        assert store.read_metadata(10).metrics == {"loss": 0.1}

    def test_a_step_below_the_latest_is_refused(self, tmp_path: Path, model: SimpleModel) -> None:
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        store.save(20, {"model": nnx.state(model)})

        with pytest.raises(CheckpointNotWrittenError) as caught:
            store.save(10, {"model": nnx.state(model)})

        assert (caught.value.step, caught.value.latest_step) == (10, 20)
        assert caught.value.reason == "below_latest"
        assert store.list_steps() == [20]

    def test_a_refusal_names_the_steps(self, tmp_path: Path, model: SimpleModel) -> None:
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        store.save(20, {"model": nnx.state(model)})
        with pytest.raises(CheckpointNotWrittenError, match=r"step 10.*latest.*20"):
            store.save(10, {"model": nnx.state(model)})


class TestMetadata:
    def test_extra_cannot_shadow_a_reserved_key(self, tmp_path: Path, model: SimpleModel) -> None:
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        with pytest.raises(ValueError, match="step"):
            store.save(1, {"model": nnx.state(model)}, extra={"step": 3})
        assert not store.directory.exists()

    def test_producer_extra_and_epoch_round_trip(self, tmp_path: Path, model: SimpleModel) -> None:
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        producer = Producer(name="artifex", version="0.1.10")
        store.save(
            3,
            {"model": nnx.state(model)},
            epoch=1,
            producer=producer,
            extra={"physics": {"pde": "burgers"}, "run": "demo"},
        )

        metadata = store.read_metadata(3)

        assert metadata.producer == producer
        assert metadata.epoch == 1
        assert metadata.extra == {"physics": {"pde": "burgers"}, "run": "demo"}
        assert metadata.items == ("model",)
        assert metadata.created_at.endswith("+00:00")
        assert store.restore(3).metadata == metadata

    def test_the_metadata_on_disk_is_the_record(self, tmp_path: Path, model: SimpleModel) -> None:
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        store.save(3, {"model": nnx.state(model)}, metrics={"loss": 0.5})

        assert raw_metadata(store.directory, 3) == store.read_metadata(3).to_dict()

    def test_a_newer_format_is_refused(self, tmp_path: Path, model: SimpleModel) -> None:
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        store.save(3, {"model": nnx.state(model)})
        payload = raw_metadata(store.directory, 3)
        payload["format_version"] = CURRENT_FORMAT_VERSION + 1
        write_raw_metadata(store.directory, 3, payload)

        with pytest.raises(UnsupportedCheckpointError, match=str(CURRENT_FORMAT_VERSION + 1)):
            store.restore(3)
        with pytest.raises(UnsupportedCheckpointError):
            store.read_metadata(3)


class TestTemplatePlacement:
    """With templates, placement and dtype come from the template's leaves, not the checkpoint."""

    def test_a_restore_onto_templates_does_not_read_the_saved_sharding(
        self, tmp_path: Path, model: SimpleModel
    ) -> None:
        """The template supplies every array's sharding, so Orbax never opens the sharding file.

        The sharding file names the devices the checkpoint was written on; consulting it is
        what breaks a restore on a different topology.
        """
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        store.save(1, {"model": nnx.state(model)})

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            store.restore(1, templates={"model": nnx.state(model)})

        assert not [w for w in caught if _SHARDING_FILE_WARNING in str(w.message)]

    def test_a_restore_without_templates_reads_the_saved_sharding(
        self, tmp_path: Path, model: SimpleModel
    ) -> None:
        """Control: the template-free restore keeps the saved placement, via the sharding file.

        This is the warning ``filterwarnings`` in pyproject.toml ignores, and the reason it
        must stay: the documented template-free behaviour is to come back as stored.
        """
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        store.save(1, {"model": nnx.state(model)})

        with pytest.warns(UserWarning, match=_SHARDING_FILE_WARNING):
            store.restore(1)

    def test_a_restore_onto_templates_keeps_dtypes_kinds_and_placement(
        self, tmp_path: Path
    ) -> None:
        store = OrbaxCheckpointStore(tmp_path / "ckpt")

        def payload() -> dict[str, Any]:
            return {
                "half": jnp.arange(3, dtype=jnp.float16),
                "brain": jnp.ones(2, dtype=jnp.bfloat16),
                "ints": jnp.arange(2, dtype=jnp.int32),
                "host": np.array([1, 2], dtype=np.int32),
            }

        store.save(1, {"data_iterator": payload()})
        restored = store.restore(1, templates={"data_iterator": payload()}).items["data_iterator"]

        for name, expected in payload().items():
            assert restored[name].dtype == expected.dtype, name
            assert type(restored[name]) is type(expected), name
            np.testing.assert_array_equal(np.asarray(restored[name]), np.asarray(expected))
        for name in ("half", "brain", "ints"):
            assert restored[name].devices() == {jax.devices()[0]}, name


class TestCrossTopologyRestore:
    """A checkpoint written on one device set restores onto templates in another.

    Each side runs in its own interpreter because jax fixes its device set at
    initialisation. The save side exposes two CPU devices and places every array on
    the second; the restore side exposes one. Restoring the module state, the
    iterator state and the metadata onto that one device is the contract.
    """

    def test_a_two_device_save_restores_onto_one_device_templates(self, tmp_path: Path) -> None:
        saved = _run_child("save", tmp_path, runtime=_cpu_devices(2)).check().last_json()
        assert saved["saved_on"] == [{"platform": "cpu", "id": 1}]

        restored = _run_child("restore", tmp_path, runtime=_cpu_devices(1)).check().last_json()

        assert restored["devices"] == [_CPU_0]
        assert restored["restored_on"] == [_CPU_0]
        assert restored["state"] == saved["state"]
        assert restored["metadata"]["metrics"] == {"loss": 0.5}
        assert restored["metadata"]["extra"] == {"run": "topology"}
        assert restored["metadata"]["items"] == ["model", "data_iterator"]

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

    def test_a_two_device_save_without_templates_cannot_resolve_the_saved_device(
        self, tmp_path: Path
    ) -> None:
        """Control: template-free, the same checkpoint fails on the device it was saved on.

        This proves the two interpreters really differ in topology, so the pass above
        measures placement rather than a shared device set. It also pins the documented
        template-free behaviour: arrays come back as stored, or not at all.
        """
        _run_child("save", tmp_path, runtime=_cpu_devices(2)).check().last_json()

        completed = _run_child(
            "restore", tmp_path, runtime=_cpu_devices(1), extra=("--without-templates",)
        )

        assert completed.returncode != 0
        assert "Device cpu:1 was not found in jax.local_devices()" in completed.stderr

    @pytest.mark.gpu
    def test_a_gpu_save_restores_onto_cpu_templates(self, tmp_path: Path) -> None:
        """A checkpoint written on an accelerator restores onto CPU-only templates.

        The CPU-to-CPU case above does not prove this: an accelerator checkpoint
        records a different platform, not just a different device id.
        """
        if not cuda_is_visible():
            pytest.skip("no CUDA device visible to jax")

        cuda = JaxRuntime(platforms=("cuda",))
        saved = _run_child("save", tmp_path, runtime=cuda).check().last_json()
        assert [d["platform"] for d in saved["saved_on"]] == ["gpu"]

        restored = _run_child("restore", tmp_path, runtime=_cpu_devices(1)).check().last_json()

        assert restored["restored_on"] == [_CPU_0]
        assert restored["state"] == saved["state"]
        assert restored["payload"]["half"]["devices"] == [_CPU_0]
        assert restored["payload"]["key"]["devices"] == [_CPU_0]


class TestListingAndRetention:
    def test_steps_are_listed_in_order_and_the_latest_is_the_newest(
        self, tmp_path: Path, model: SimpleModel
    ) -> None:
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        for step in (100, 200, 300):
            store.save(step, {"model": nnx.state(model)})
        assert store.list_steps() == [100, 200, 300]
        assert store.latest_step() == 300
        assert store.read_metadata(200).step == 200

    def test_max_to_keep_prunes_the_oldest_steps(self, tmp_path: Path, model: SimpleModel) -> None:
        store = OrbaxCheckpointStore(tmp_path / "ckpt", max_to_keep=2)
        for step in (1, 2, 3, 4):
            store.save(step, {"model": nnx.state(model)})
        assert store.list_steps() == [3, 4]

    def test_max_to_keep_none_keeps_every_step(self, tmp_path: Path, model: SimpleModel) -> None:
        with OrbaxCheckpointStore(tmp_path / "ckpt", max_to_keep=None) as store:
            for step in range(8):
                store.save(step, {"model": nnx.state(model)})
            assert store.list_steps() == list(range(8))

    def test_delete_removes_the_step(self, tmp_path: Path, model: SimpleModel) -> None:
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        store.save(10, {"model": nnx.state(model)})
        store.save(20, {"model": nnx.state(model)})

        store.delete(10)

        assert store.list_steps() == [20]
        with pytest.raises(CheckpointNotFoundError, match="10"):
            store.delete(10)


class TestBestStep:
    def test_the_minimum_of_a_metric(self, tmp_path: Path, model: SimpleModel) -> None:
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        for step, loss in ((1, 0.9), (2, 0.2), (3, 0.5)):
            store.save(step, {"model": nnx.state(model)}, metrics={"loss": loss})
        assert store.best_step("loss") == 2
        assert store.best_step("loss", mode="min") == 2

    def test_the_maximum_of_a_metric(self, tmp_path: Path, model: SimpleModel) -> None:
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        store.save(1, {"model": nnx.state(model)}, metrics={"accuracy": 0.7})
        store.save(2, {"model": nnx.state(model)}, metrics={"accuracy": 0.9})
        assert store.best_step("accuracy", mode="max") == 2

    def test_steps_without_the_metric_are_skipped(self, tmp_path: Path, model: SimpleModel) -> None:
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        store.save(1, {"data_iterator": {"position": 1}})
        store.save(2, {"model": nnx.state(model)}, metrics={"loss": 0.4})
        store.save(3, {"model": nnx.state(model)}, metrics={"loss": 0.9})
        assert store.best_step("loss") == 2
        assert store.best_step("accuracy") is None

    def test_an_unknown_mode_is_refused(self, tmp_path: Path, model: SimpleModel) -> None:
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        store.save(1, {"model": nnx.state(model)}, metrics={"loss": 0.4})
        with pytest.raises(ValueError, match="mode"):
            store.best_step("loss", mode="best")  # type: ignore[arg-type]


class TestResourceManagement:
    def test_the_store_is_a_context_manager(self, tmp_path: Path, model: SimpleModel) -> None:
        with OrbaxCheckpointStore(tmp_path / "ckpt") as store:
            store.save(1, {"model": nnx.state(model)})
            assert store.latest_step() == 1

    def test_close_before_any_save_is_harmless(self, tmp_path: Path) -> None:
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        store.close()
        assert not store.directory.exists()


class TestSerializationSafety:
    def test_the_package_never_uses_pickle(self) -> None:
        for module_info in pkgutil.iter_modules(substrax.checkpoint.__path__):
            module = importlib.import_module(f"substrax.checkpoint.{module_info.name}")
            source = inspect.getsource(module)
            assert "import pickle" not in source, module_info.name
            assert "pickle." not in source, module_info.name

    def test_the_saved_artifacts_are_not_pickle(self, tmp_path: Path, model: SimpleModel) -> None:
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        path = store.save(1, {"model": nnx.state(model)})

        for file in path.rglob("*"):
            if file.is_file():
                assert not file.read_bytes()[:2].startswith(b"\x80"), f"pickle stream in {file}"


class TestOnDiskLayout:
    """Each item is one Orbax item holding the pytree under a single ``tree`` node.

    The node is what lets an item that is a bare array, such as the ``rng`` key, pass
    Orbax 0.11.33's ``if not item`` check; the floor lane in CI runs this file on it.
    """

    def test_each_item_holds_its_tree_under_one_node(
        self, tmp_path: Path, model: SimpleModel
    ) -> None:
        import orbax.checkpoint as ocp  # type: ignore[import-untyped]  # noqa: PLC0415

        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        store.save(1, {"model": nnx.state(model), "rng": jax.random.key(3)})
        store.close()

        with ocp.CheckpointManager(store.directory) as manager:
            raw = manager.restore(
                1,
                args=ocp.args.Composite(  # type: ignore[reportCallIssue]
                    model=ocp.args.PyTreeRestore(),  # type: ignore[reportCallIssue]
                    rng=ocp.args.PyTreeRestore(),  # type: ignore[reportCallIssue]
                ),
            )
        assert set(raw["model"]) == {"tree"}
        assert set(raw["rng"]) == {"tree"}
        assert jnp.array_equal(
            jax.random.key_data(raw["rng"]["tree"]), jax.random.key_data(jax.random.key(3))
        )

    def test_a_bare_array_item_round_trips_onto_a_template(self, tmp_path: Path) -> None:
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        store.save(1, {"rng": jax.random.PRNGKey(0), "data_iterator": jnp.asarray(2.0)})

        restored = store.restore(
            1, templates={"rng": jax.random.PRNGKey(1), "data_iterator": jnp.asarray(0.0)}
        ).items

        assert jnp.array_equal(restored["rng"], jax.random.PRNGKey(0))
        assert float(restored["data_iterator"]) == 2.0


class TestPlainLeaves:
    """A ``data_iterator`` item carries ints, lists, strings, booleans and typed keys."""

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

    def test_plain_leaves_and_keys_round_trip_onto_a_template(self, tmp_path: Path) -> None:
        store = OrbaxCheckpointStore(tmp_path)
        store.save(3, {"data_iterator": self._iterator_state()})

        restored = store.restore(3, templates={"data_iterator": self._iterator_state()})
        state = restored.items["data_iterator"]

        assert state["position"] == 7
        assert state["rng_counts"] == [1, 2]
        assert state["sampler_repr"] == "SequentialSampler(seed=42)"
        assert state["shuffle"] is True
        assert jnp.array_equal(state["indices"], jnp.arange(3))
        assert jnp.array_equal(
            jax.random.key_data(state["key"]), jax.random.key_data(jax.random.key(0))
        )
        assert restored.metadata.metrics == {}
