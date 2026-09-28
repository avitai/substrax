"""Tests for data parallelism functions.

Tests both SPMD-based and legacy pmap-based data parallel utilities.
"""

import dataclasses
import json
import socket
import subprocess  # nosec B404
import sys
from pathlib import Path
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx
from jax.sharding import NamedSharding, PartitionSpec

from substrax.mesh import DeviceMeshManager
from substrax.runtime import child_environment, JaxRuntime
from substrax.spmd import (
    create_data_parallel_sharding,
    place_batch_on_shards,
    spmd_train_step,
)


class TestCreateDataParallelSharding:
    """Tests for create_data_parallel_sharding function."""

    def test_single_device_sharding(self) -> None:
        """Test creating sharding with single device mesh."""
        mesh = DeviceMeshManager.create_data_parallel_mesh(num_devices=1)
        sharding = create_data_parallel_sharding(mesh)
        assert sharding.mesh == mesh
        assert sharding.spec == jax.sharding.PartitionSpec("data")

    def test_custom_data_axis(self) -> None:
        """Test creating sharding with custom axis name."""
        mesh = DeviceMeshManager.create_device_mesh([("batch", 1)])
        sharding = create_data_parallel_sharding(mesh, data_axis="batch")
        assert sharding.spec == jax.sharding.PartitionSpec("batch")

    @pytest.mark.devices(2)
    def test_multi_device_sharding(self) -> None:
        """Test creating sharding across multiple devices."""
        mesh = DeviceMeshManager.create_data_parallel_mesh(num_devices=2)
        sharding = create_data_parallel_sharding(mesh)
        assert sharding.spec == jax.sharding.PartitionSpec("data")


class TestShardBatch:
    """Tests for place_batch_on_shards function."""

    def test_shards_array_values(self) -> None:
        """Test that jax.Array values are sharded."""
        mesh = DeviceMeshManager.create_data_parallel_mesh(num_devices=1)
        sharding = create_data_parallel_sharding(mesh)
        batch = {"inputs": jnp.ones((4, 2)), "targets": jnp.zeros((4,))}

        result = place_batch_on_shards(batch, sharding)

        assert result["inputs"].shape == (4, 2)
        assert result["targets"].shape == (4,)

    def test_non_array_values_unchanged(self) -> None:
        """Test that non-array values pass through unchanged."""
        mesh = DeviceMeshManager.create_data_parallel_mesh(num_devices=1)
        sharding = create_data_parallel_sharding(mesh)
        batch = {"data": jnp.ones((4, 2)), "label": "test_string"}

        result = place_batch_on_shards(batch, sharding)

        assert result["label"] == "test_string"

    def test_numpy_array_values_are_placed(self) -> None:
        """A host batch, as a data loader yields it, is placed like a jax.Array batch."""
        mesh = DeviceMeshManager.create_data_parallel_mesh(num_devices=1)
        sharding = create_data_parallel_sharding(mesh)
        batch = {"inputs": np.ones((4, 2), dtype=np.float32), "label": "test_string"}

        result = place_batch_on_shards(batch, sharding)

        assert isinstance(result["inputs"], jax.Array)
        assert result["inputs"].sharding == sharding
        assert result["label"] == "test_string"

    def test_placement_keeps_values_dtypes_and_nesting(self) -> None:
        """Placement moves every array leaf of a nested batch and changes nothing else."""
        mesh = DeviceMeshManager.create_data_parallel_mesh(num_devices=1)
        sharding = create_data_parallel_sharding(mesh)
        batch = {
            "features": {
                "images": np.arange(8, dtype=np.float16).reshape(4, 2),
                "mask": np.array([True, False, True, True]),
            },
            "targets": {"labels": jnp.arange(4, dtype=jnp.int32)},
        }

        result = place_batch_on_shards(batch, sharding)

        assert jax.tree.structure(result) == jax.tree.structure(batch)
        for placed, original in zip(jax.tree.leaves(result), jax.tree.leaves(batch), strict=True):
            assert placed.sharding == sharding
            assert placed.dtype == original.dtype
            np.testing.assert_array_equal(np.asarray(placed), np.asarray(original))

    def test_empty_batch_is_returned_empty(self) -> None:
        mesh = DeviceMeshManager.create_data_parallel_mesh(num_devices=1)

        assert place_batch_on_shards({}, create_data_parallel_sharding(mesh)) == {}

    @pytest.mark.devices(2)
    def test_an_array_already_on_another_sharding_is_moved(self) -> None:
        """A replicated array lands split over the data axis with its values unchanged."""
        mesh = DeviceMeshManager.create_data_parallel_mesh(num_devices=2)
        sharding = create_data_parallel_sharding(mesh)
        replicated = jax.device_put(
            jnp.arange(4.0), jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        )

        result = place_batch_on_shards({"x": replicated}, sharding)

        assert result["x"].sharding == sharding
        np.testing.assert_array_equal(np.asarray(result["x"]), np.arange(4.0))

    def test_array_leaves_become_global_arrays_in_one_call(self) -> None:
        """Each process passes its local batch; jax assembles the global batch in one call.

        ``jax.make_array_from_process_local_data`` is jax's data-loading primitive: on one
        process it is a single batched ``device_put``, and on several it stitches every
        process's local slice into one global array.
        """
        mesh = DeviceMeshManager.create_data_parallel_mesh(num_devices=1)
        sharding = create_data_parallel_sharding(mesh)
        batch = {
            "inputs": np.ones((4, 2), dtype=np.float32),
            "targets": jnp.zeros((4,)),
            "label": "test_string",
        }

        with mock.patch.object(
            jax,
            "make_array_from_process_local_data",
            wraps=jax.make_array_from_process_local_data,
        ) as assemble:
            result = place_batch_on_shards(batch, sharding)

        assert assemble.call_count == 1
        placed_sharding, local_leaves = assemble.call_args.args
        assert placed_sharding == sharding
        assert [leaf.shape for leaf in local_leaves] == [(4, 2), (4,)]
        assert result["targets"].sharding == sharding
        assert result["label"] == "test_string"

    @pytest.mark.devices(2)
    def test_multi_device_shard(self) -> None:
        """Test sharding across multiple devices."""
        mesh = DeviceMeshManager.create_data_parallel_mesh(num_devices=2)
        sharding = create_data_parallel_sharding(mesh)
        batch = {"inputs": jnp.ones((4, 2)), "targets": jnp.zeros((4,))}

        result = place_batch_on_shards(batch, sharding)

        assert result["inputs"].shape == (4, 2)
        assert result["targets"].shape == (4,)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class _Rows[T]:
    """A batch shaped like datarax's: per-row fields and one batch-level field without a row axis.

    The same type, holding shardings, is the batch's sharding prefix.
    """

    data: T
    batch_state: T


class TestPlacementByPrefix:
    """``place_batch_on_shards`` with a pytree prefix of shardings, one per subtree."""

    @pytest.mark.devices(2)
    def test_a_prefix_places_each_subtree_on_its_own_sharding(self) -> None:
        mesh = DeviceMeshManager.create_data_parallel_mesh(num_devices=2)
        row = create_data_parallel_sharding(mesh)
        replicated = NamedSharding(mesh, PartitionSpec())
        batch = _Rows(
            data={"x": np.arange(8, dtype=np.float32).reshape(4, 2), "i": np.arange(4)},
            batch_state={"lam": np.float32(0.25)},
        )

        result = place_batch_on_shards(batch, _Rows(data=row, batch_state=replicated))

        assert isinstance(result, _Rows)
        assert result.data["x"].sharding == row
        assert result.data["i"].sharding == row
        assert isinstance(result.batch_state["lam"], jax.Array)
        assert result.batch_state["lam"].sharding == replicated
        assert float(result.batch_state["lam"]) == 0.25
        np.testing.assert_array_equal(np.asarray(result.data["x"]), batch.data["x"])

    def test_a_numpy_scalar_under_a_replicated_sharding_is_placed(self) -> None:
        mesh = DeviceMeshManager.create_data_parallel_mesh(num_devices=1)
        replicated = NamedSharding(mesh, PartitionSpec())

        result = place_batch_on_shards({"lam": np.float32(0.5)}, replicated)

        assert isinstance(result["lam"], jax.Array)
        assert result["lam"].sharding == replicated
        assert float(result["lam"]) == 0.5

    def test_a_leaf_its_sharding_cannot_hold_is_refused_by_path(self) -> None:
        mesh = DeviceMeshManager.create_data_parallel_mesh(num_devices=1)
        row = create_data_parallel_sharding(mesh)
        batch = {"x": np.ones((4, 2), np.float32), "batch_state": {"lam": np.float32(0.5)}}

        with pytest.raises(ValueError, match=r"\['batch_state'\]\['lam'\]"):
            place_batch_on_shards(batch, row)

    def test_non_array_leaves_under_a_prefix_are_returned_as_they_are(self) -> None:
        mesh = DeviceMeshManager.create_data_parallel_mesh(num_devices=1)
        row = create_data_parallel_sharding(mesh)
        replicated = NamedSharding(mesh, PartitionSpec())

        result = place_batch_on_shards(
            {"x": np.ones((4,), np.float32), "meta": {"name": "a"}},
            {"x": row, "meta": replicated},
        )

        assert result["meta"] == {"name": "a"}
        assert result["x"].sharding == row

    def test_two_processes_assemble_one_global_batch(self, tmp_path: Path) -> None:
        """Each process passes the rows it loaded; the batch-level leaf is replicated and equal."""
        worker = tmp_path / "worker.py"
        worker.write_text(_TWO_PROCESS_WORKER)
        with socket.socket() as probe:
            probe.bind(("localhost", 0))
            port = probe.getsockname()[1]
        env = child_environment(JaxRuntime(platforms=("cpu",), preallocate=False), {})
        workers = [
            subprocess.Popen(  # noqa: S603  # nosec B603  # this interpreter and a test script
                [sys.executable, str(worker), str(port), str(process)],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for process in (0, 1)
        ]
        outputs = [child.communicate(timeout=240) for child in workers]

        for child, (_, stderr) in zip(workers, outputs, strict=True):
            assert child.returncode == 0, stderr[-3000:]
        reports = [json.loads(stdout.strip().splitlines()[-1]) for stdout, _ in outputs]
        for report in reports:
            assert report["shapes"] == [[8, 2], [8], []]
            assert report["specs"] == [["data"], ["data"], []]
            assert report["rows"] == list(range(8))
            assert report["lam"] == 0.25


_TWO_PROCESS_WORKER = """
import json, sys

import jax
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding, PartitionSpec

from substrax.spmd import place_batch_on_shards

port, process = int(sys.argv[1]), int(sys.argv[2])
jax.distributed.initialize(f"localhost:{port}", num_processes=2, process_id=process)
mesh = jax.make_mesh((2,), ("data",), axis_types=(jax.sharding.AxisType.Auto,))
row, replicated = NamedSharding(mesh, PartitionSpec("data")), NamedSharding(mesh, PartitionSpec())
local = {
    "x": np.full((4, 2), process, np.float32),
    "i": np.arange(4) + 4 * process,
    "batch_state": {"lam": np.float32(0.25)},
}
placed = place_batch_on_shards(local, {"x": row, "i": row, "batch_state": replicated})
leaves = [placed["x"], placed["i"], placed["batch_state"]["lam"]]
rows = multihost_utils.process_allgather(placed["i"], tiled=True)
print(json.dumps({
    "shapes": [list(leaf.shape) for leaf in leaves],
    "specs": [list(leaf.sharding.spec) for leaf in leaves],
    "rows": np.asarray(rows).tolist(),
    "lam": float(np.asarray(placed["batch_state"]["lam"].addressable_data(0))),
}))
jax.distributed.shutdown()
"""


class TestDataParallelGradients:
    """Where a data-parallel gradient is reduced: by the compiler under jit, on the loss in shard_map.

    There is no gradient to reduce after differentiating: under ``jax.jit`` the gradient of a
    sharded batch's loss is already the full one, and inside ``jax.shard_map`` the gradient of a
    replicated parameter comes back summed over the axis, so averaging it afterwards leaves it
    wrong by the axis size. The loss is averaged before differentiating instead.
    """

    @staticmethod
    def _problem() -> tuple[jax.Array, dict[str, jax.Array]]:
        weights = jnp.arange(12.0).reshape(3, 4) / 10
        batch = {
            "x": jax.random.normal(jax.random.key(0), (64, 3)),
            "y": jax.random.normal(jax.random.key(1), (64, 4)),
        }
        return weights, batch

    @staticmethod
    def _loss(weights: jax.Array, batch: dict[str, jax.Array]) -> jax.Array:
        return jnp.mean((batch["x"] @ weights - batch["y"]) ** 2)

    @staticmethod
    def _assert_gradients_match(actual: jax.Array, expected: jax.Array) -> None:
        # A sharded gradient sums its terms in another order, so it differs from the
        # single-device one by round-off at the array's scale: measured 0.74, 0.74 and 1.49 ULP
        # of the largest entry on 2, 4 and 8 devices (jit and shard_map alike). A relative check
        # fails on entries near zero; four ULP of the largest entry bounds it.
        atol = 4 * float(jnp.finfo(expected.dtype).eps) * float(jnp.max(jnp.abs(expected)))
        np.testing.assert_allclose(actual, expected, rtol=0, atol=atol)

    @pytest.mark.devices(2)
    def test_under_jit_the_gradient_of_a_sharded_batch_is_already_the_full_one(self) -> None:
        weights, batch = self._problem()
        mesh = DeviceMeshManager.create_data_parallel_mesh()
        sharded = place_batch_on_shards(batch, create_data_parallel_sharding(mesh))

        gradient = jax.jit(jax.grad(self._loss))(weights, sharded)

        self._assert_gradients_match(gradient, jax.grad(self._loss)(weights, batch))

    @pytest.mark.devices(2)
    def test_in_shard_map_the_loss_is_averaged_before_differentiating(self) -> None:
        weights, batch = self._problem()
        mesh = DeviceMeshManager.create_data_parallel_mesh()
        spec = jax.sharding.PartitionSpec("data")

        def per_shard(w: jax.Array, shard: dict[str, jax.Array]) -> jax.Array:
            return jax.grad(lambda w: jax.lax.pmean(self._loss(w, shard), "data"))(w)

        gradient = jax.jit(
            jax.shard_map(
                per_shard,
                mesh=mesh,
                in_specs=(jax.sharding.PartitionSpec(), {"x": spec, "y": spec}),
                out_specs=jax.sharding.PartitionSpec(),
            )
        )(weights, batch)

        self._assert_gradients_match(gradient, jax.grad(self._loss)(weights, batch))

    @pytest.mark.devices(2)
    def test_in_shard_map_a_gradient_averaged_afterwards_is_off_by_the_axis_size(self) -> None:
        weights, batch = self._problem()
        mesh = DeviceMeshManager.create_data_parallel_mesh()
        spec = jax.sharding.PartitionSpec("data")

        def per_shard(w: jax.Array, shard: dict[str, jax.Array]) -> jax.Array:
            return jax.lax.pmean(jax.grad(self._loss)(w, shard), "data")

        gradient = jax.jit(
            jax.shard_map(
                per_shard,
                mesh=mesh,
                in_specs=(jax.sharding.PartitionSpec(), {"x": spec, "y": spec}),
                out_specs=jax.sharding.PartitionSpec(),
            )
        )(weights, batch)

        full = jax.grad(self._loss)(weights, batch)
        self._assert_gradients_match(gradient, mesh.shape["data"] * full)


class TestSpmdTrainStep:
    """Tests for spmd_train_step function."""

    def _make_model_and_optimizer(self) -> tuple[nnx.Linear, nnx.Optimizer[nnx.Linear]]:
        """Create a minimal NNX model and optimizer for testing."""
        model = nnx.Linear(2, 1, rngs=nnx.Rngs(0))
        optimizer = nnx.Optimizer(model, optax.sgd(0.01), wrt=nnx.Param)
        return model, optimizer

    def test_reduces_loss(self) -> None:
        """Test that a single training step produces a finite loss."""
        model, optimizer = self._make_model_and_optimizer()
        batch = {"x": jnp.ones((4, 2)), "y": jnp.zeros((4, 1))}

        def loss_fn(m: nnx.Linear, b: dict[str, jax.Array]) -> jax.Array:
            return jnp.mean((m(b["x"]) - b["y"]) ** 2)

        loss = spmd_train_step(model, optimizer, loss_fn, batch)
        assert jnp.isfinite(loss)

    def test_updates_parameters(self) -> None:
        """Test that parameters change after a training step."""
        model, optimizer = self._make_model_and_optimizer()
        params_before = jax.tree.map(jnp.copy, nnx.state(model, nnx.Param))
        batch = {"x": jnp.ones((4, 2)), "y": jnp.zeros((4, 1))}

        def loss_fn(m: nnx.Linear, b: dict[str, jax.Array]) -> jax.Array:
            return jnp.mean((m(b["x"]) - b["y"]) ** 2)

        spmd_train_step(model, optimizer, loss_fn, batch)
        params_after = nnx.state(model, nnx.Param)

        # At least one parameter leaf must have changed
        leaves_before = jax.tree.leaves(params_before)
        leaves_after = jax.tree.leaves(params_after)
        any_changed = any(
            not jnp.array_equal(b, a) for b, a in zip(leaves_before, leaves_after, strict=True)
        )
        assert any_changed, "Parameters should change after a training step"

    def test_loss_decreases_over_steps(self) -> None:
        """Test that loss decreases over multiple training steps."""
        model, optimizer = self._make_model_and_optimizer()
        batch = {"x": jnp.ones((4, 2)), "y": jnp.zeros((4, 1))}

        def loss_fn(m: nnx.Linear, b: dict[str, jax.Array]) -> jax.Array:
            return jnp.mean((m(b["x"]) - b["y"]) ** 2)

        loss_first = spmd_train_step(model, optimizer, loss_fn, batch)
        loss_last = loss_first
        for _ in range(10):
            loss_last = spmd_train_step(model, optimizer, loss_fn, batch)

        assert float(loss_last) < float(loss_first)
