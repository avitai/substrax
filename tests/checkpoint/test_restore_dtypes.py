"""Restoring onto a template refuses to change a saved array's dtype unless asked to.

Orbax casts each array to its template leaf's dtype on restore, so a checkpoint written in
float64 comes back as float32 onto a float32 template, and a float32 model onto a bfloat16 one,
without a word. ``save`` writes every array's dtype as a ``dtypes`` item beside the metadata
record; ``restore`` compares its templates against it, or against Orbax's per-array metadata for a
checkpoint written without it, and raises ``CheckpointDtypeMismatchError`` naming every differing
leaf.
``cast_dtypes=True`` is the explicit request for the cast.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import pytest
from _helpers import raw_metadata, SimpleModel
from flax import nnx

from substrax.checkpoint import (
    CheckpointDtypeMismatchError,
    DtypeMismatch,
    OrbaxCheckpointStore,
)
from substrax.runtime import JaxRuntime
from substrax.testing import run_python


def _saved(tmp_path: Path, items: dict[str, Any]) -> OrbaxCheckpointStore:
    store = OrbaxCheckpointStore(tmp_path / "ckpt")
    store.save(1, items)
    store.close()
    # A fresh store, as a restoring process has: nothing is known about the items beforehand.
    return OrbaxCheckpointStore(tmp_path / "ckpt")


def _model_state(dtype: Any = jnp.float32) -> Any:
    """A two-layer model's state with every array in ``dtype``."""
    state = nnx.state(SimpleModel(rngs=nnx.Rngs(0)))
    return jax.tree.map(lambda leaf: leaf.astype(dtype), state)


def _key_data(key: jax.Array) -> jax.Array:
    return jax.random.key_data(key) if jnp.issubdtype(key.dtype, jax.dtypes.prng_key) else key


@pytest.fixture
def orbax_metadata_reads(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Every item directory whose dtypes are read through Orbax's per-array metadata."""
    reads: list[Path] = []
    original = ocp.PyTreeCheckpointHandler.metadata

    def counted(self: Any, directory: Any) -> Any:
        reads.append(Path(directory))
        return original(self, directory)

    monkeypatch.setattr(ocp.PyTreeCheckpointHandler, "metadata", counted)
    return reads


def _recorded_dtypes(store: OrbaxCheckpointStore, step: int) -> dict[str, Any]:
    """The ``dtypes`` item Orbax's ``JsonSave`` wrote for ``step``, read from disk.

    Orbax's JSON handler names its file ``metadata`` whatever the item is called.
    """
    path = store.directory / str(step) / "dtypes" / "metadata"
    return json.loads(path.read_text(encoding="utf-8"))


def _drop_recorded_dtypes(store: OrbaxCheckpointStore, step: int) -> None:
    """Remove the ``dtypes`` item, as substrax 0.1.10 and 0.1.11 wrote a checkpoint."""
    shutil.rmtree(store.directory / str(step) / "dtypes")


def test_save_records_every_array_dtype(tmp_path: Path) -> None:
    store = _saved(
        tmp_path,
        {
            "model": _model_state(),
            "extensions": {
                "table": np.ones((2,), np.float64),
                "half": jnp.ones((2,), jnp.bfloat16),
            },
            "rng": jax.random.key(0),
            "data_iterator": {"position": 7, "name": "shard-1"},
        },
    )

    recorded = _recorded_dtypes(store, 1)

    assert recorded["model"] == {
        "dense1/bias/value": "float32",
        "dense1/kernel/value": "float32",
        "dense2/bias/value": "float32",
        "dense2/kernel/value": "float32",
    }
    assert recorded["extensions"] == {"half": "bfloat16", "table": "float64"}
    # A typed key is recorded by its data, as Orbax stores it.
    assert recorded["rng"] == {"": "uint32"}
    assert recorded["data_iterator"] == {}
    # The metadata record, which best_step reads for every step, stays free of the map.
    assert "dtypes" not in raw_metadata(store.directory, 1)


def test_a_recorded_checkpoint_is_checked_without_reading_array_metadata(
    tmp_path: Path, orbax_metadata_reads: list[Path]
) -> None:
    store = _saved(tmp_path, {"extensions": {"table": np.ones((2,), np.float64)}})

    with pytest.raises(CheckpointDtypeMismatchError):
        store.restore(1, templates={"extensions": {"table": jnp.zeros((2,), jnp.float32)}})
    store.restore(1, templates={"extensions": {"table": np.zeros((2,), np.float64)}})

    assert orbax_metadata_reads == []


def test_a_checkpoint_without_recorded_dtypes_is_checked_from_orbax_metadata(
    tmp_path: Path, orbax_metadata_reads: list[Path]
) -> None:
    store = _saved(
        tmp_path, {"model": _model_state(), "extensions": {"table": np.ones((2,), np.float64)}}
    )
    _drop_recorded_dtypes(store, 1)

    with pytest.raises(CheckpointDtypeMismatchError) as raised:
        store.restore(1, templates={"extensions": {"table": jnp.zeros((2,), jnp.float32)}})

    assert raised.value.mismatches == (
        DtypeMismatch(item="extensions", leaf="table", saved="float64", restored="float32"),
    )
    # Only the templated item is read.
    assert orbax_metadata_reads == [store.directory / "1" / "extensions"]


def test_every_item_without_recorded_dtypes_is_read_once(
    tmp_path: Path, orbax_metadata_reads: list[Path]
) -> None:
    store = _saved(
        tmp_path, {"model": _model_state(), "extensions": {"table": np.ones((2,), np.float64)}}
    )
    _drop_recorded_dtypes(store, 1)

    with pytest.raises(CheckpointDtypeMismatchError) as raised:
        store.restore(
            1,
            templates={
                "model": _model_state(jnp.bfloat16),
                "extensions": {"table": jnp.zeros((2,), jnp.float32)},
            },
        )

    assert {mismatch.item for mismatch in raised.value.mismatches} == {"model", "extensions"}
    assert sorted(orbax_metadata_reads) == sorted(
        [store.directory / "1" / "model", store.directory / "1" / "extensions"]
    )


def test_cast_dtypes_reads_no_array_metadata(
    tmp_path: Path, orbax_metadata_reads: list[Path]
) -> None:
    store = _saved(tmp_path, {"extensions": {"table": np.ones((2,), np.float64)}})
    _drop_recorded_dtypes(store, 1)

    store.restore(
        1, templates={"extensions": {"table": jnp.zeros((2,), jnp.float32)}}, cast_dtypes=True
    )

    assert orbax_metadata_reads == []


def test_matching_dtypes_restore_as_before(tmp_path: Path) -> None:
    store = _saved(tmp_path, {"model": _model_state()})

    checkpoint = store.restore(1, templates={"model": _model_state()})

    assert checkpoint.items["model"]["dense1"]["kernel"][...].dtype == jnp.float32


def test_a_float64_array_onto_a_float32_template_is_refused(tmp_path: Path) -> None:
    store = _saved(tmp_path, {"extensions": {"table": np.ones((2, 2), np.float64)}})

    with pytest.raises(CheckpointDtypeMismatchError, match="cast_dtypes=True") as raised:
        store.restore(1, templates={"extensions": {"table": jnp.zeros((2, 2), jnp.float32)}})

    assert raised.value.step == 1
    assert raised.value.mismatches == (
        DtypeMismatch(item="extensions", leaf="table", saved="float64", restored="float32"),
    )


def test_every_differing_leaf_of_every_item_is_named(tmp_path: Path) -> None:
    store = _saved(
        tmp_path,
        {"model": _model_state(), "extensions": {"table": np.ones((2,), np.float64)}},
    )

    with pytest.raises(CheckpointDtypeMismatchError) as raised:
        store.restore(
            1,
            templates={
                "model": _model_state(jnp.bfloat16),
                "extensions": {"table": jnp.zeros((2,), jnp.float32)},
            },
        )

    named = {(mismatch.item, mismatch.leaf) for mismatch in raised.value.mismatches}
    assert named == {
        ("model", "dense1/bias/value"),
        ("model", "dense1/kernel/value"),
        ("model", "dense2/bias/value"),
        ("model", "dense2/kernel/value"),
        ("extensions", "table"),
    }
    assert "model dense1/kernel/value: saved float32, restored as bfloat16" in str(raised.value)


def test_cast_dtypes_restores_onto_the_template_dtype(tmp_path: Path) -> None:
    values = np.arange(4, dtype=np.float64).reshape(2, 2) / 3
    store = _saved(tmp_path, {"extensions": {"table": values}, "model": _model_state()})

    checkpoint = store.restore(
        1,
        templates={
            "extensions": {"table": jnp.zeros((2, 2), jnp.float32)},
            "model": _model_state(jnp.bfloat16),
        },
        cast_dtypes=True,
    )

    table = checkpoint.items["extensions"]["table"]
    assert table.dtype == jnp.float32
    np.testing.assert_allclose(np.asarray(table), values.astype(np.float32))
    assert checkpoint.items["model"]["dense1"]["kernel"][...].dtype == jnp.bfloat16


@pytest.mark.parametrize(
    "key", [jax.random.key(0), jax.random.key(0, impl="rbg"), jax.random.PRNGKey(0)]
)
def test_a_prng_key_restores_onto_a_key_template(tmp_path: Path, key: jax.Array) -> None:
    store = _saved(tmp_path, {"rng": key})

    checkpoint = store.restore(1, templates={"rng": jax.random.split(key)[0]})

    assert checkpoint.items["rng"].dtype == key.dtype
    np.testing.assert_array_equal(_key_data(checkpoint.items["rng"]), _key_data(key))


def test_plain_python_leaves_are_not_compared(tmp_path: Path) -> None:
    store = _saved(
        tmp_path, {"data_iterator": {"position": 7, "name": "shard-1", "counts": [1, 2]}}
    )

    checkpoint = store.restore(
        1, templates={"data_iterator": {"position": 0, "name": "", "counts": [0, 0]}}
    )

    assert checkpoint.items["data_iterator"] == {"position": 7, "name": "shard-1", "counts": [1, 2]}


def test_an_abstract_template_is_checked_by_its_dtype(tmp_path: Path) -> None:
    store = _saved(tmp_path, {"model": _model_state()})
    abstract = jax.eval_shape(lambda: _model_state(jnp.bfloat16))

    with pytest.raises(CheckpointDtypeMismatchError, match="bfloat16"):
        store.restore(1, templates={"model": abstract})

    matching = jax.eval_shape(_model_state)
    checkpoint = store.restore(1, templates={"model": matching})
    assert checkpoint.items["model"]["dense1"]["kernel"][...].dtype == jnp.float32


def test_a_restore_without_templates_returns_the_saved_arrays(tmp_path: Path) -> None:
    store = _saved(tmp_path, {"model": _model_state()})

    checkpoint = store.restore(1)

    assert checkpoint.items["model"]["dense1"]["kernel"]["value"].dtype == jnp.float32


def test_an_item_without_a_template_is_not_compared(tmp_path: Path) -> None:
    store = _saved(
        tmp_path, {"model": _model_state(), "extensions": {"table": np.ones((2,), np.float64)}}
    )

    checkpoint = store.restore(1, templates={"model": _model_state()})

    assert set(checkpoint.items) == {"model", "extensions"}


_WRITE_WITH_X64 = """
import sys
import jax.numpy as jnp
import numpy as np
from substrax.checkpoint import OrbaxCheckpointStore

with OrbaxCheckpointStore(sys.argv[1]) as store:
    store.save(
        1,
        {"extensions": {"table": jnp.arange(3, dtype=jnp.float64) / 3,
                        "host": np.arange(3, dtype=np.float64) / 3}},
    )
"""


@pytest.fixture(scope="module")
def x64_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A root holding a float64 jax.Array and a float64 numpy array, written with x64 enabled."""
    root = tmp_path_factory.mktemp("x64") / "ckpt"
    run_python(
        _WRITE_WITH_X64,
        str(root),
        runtime=JaxRuntime(platforms=("cpu",), enable_x64=True),
        timeout=120,
    ).check()
    return root


class TestWithoutTemplates:
    """Without x64 a float64 jax.Array comes back at 32 bits; that is refused like a template's cast."""

    def test_a_64_bit_array_restored_at_32_bits_is_refused(self, x64_root: Path) -> None:
        with pytest.raises(CheckpointDtypeMismatchError, match="JAX_ENABLE_X64") as raised:
            OrbaxCheckpointStore(x64_root).restore(1)

        # The numpy leaf keeps float64 and is not named.
        assert raised.value.mismatches == (
            DtypeMismatch(item="extensions", leaf="table", saved="float64", restored="float32"),
        )

    def test_cast_dtypes_accepts_the_32_bit_array(self, x64_root: Path) -> None:
        items = OrbaxCheckpointStore(x64_root).restore(1, cast_dtypes=True).items

        assert items["extensions"]["table"].dtype == jnp.float32
        assert items["extensions"]["host"].dtype == np.float64

    @pytest.mark.x64
    def test_with_x64_the_array_keeps_64_bits(self, x64_root: Path) -> None:
        items = OrbaxCheckpointStore(x64_root).restore(1).items

        assert items["extensions"]["table"].dtype == jnp.float64

    def test_without_the_record_the_orbax_metadata_catches_it(
        self, tmp_path: Path, x64_root: Path
    ) -> None:
        root = tmp_path / "copy"
        shutil.copytree(x64_root, root)
        store = OrbaxCheckpointStore(root)
        _drop_recorded_dtypes(store, 1)

        with pytest.raises(CheckpointDtypeMismatchError, match="table"):
            store.restore(1)
