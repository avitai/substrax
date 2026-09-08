"""Tests for metric reductions and collectives."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from hypothesis import given, settings, strategies as st
from hypothesis.extra import numpy as hnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from substrax.spmd import (
    all_gather,
    collect_from_devices,
    reduce_custom,
    reduce_max,
    reduce_mean,
    reduce_mean_collective,
    reduce_min,
    reduce_sum,
    reduce_sum_collective,
)


# XLA's CPU backend flushes subnormal float32 values to zero, so a reduction cannot
# return them; the property holds on the normal range, which is what training produces.
_VECTORS = hnp.arrays(
    np.float32,
    hnp.array_shapes(min_dims=1, max_dims=3, max_side=5),
    elements=st.floats(-1e3, 1e3, width=32, allow_subnormal=False),
)


class TestReductions:
    def test_mean_reduces_arrays_and_leaves_the_rest(self) -> None:
        result = reduce_mean({"loss": jnp.array([1.0, 2.0, 3.0]), "step": 10, "tag": "x"})
        assert float(result["loss"]) == 2.0
        assert result["step"] == 10
        assert result["tag"] == "x"

    def test_scalars_pass_through_unchanged(self) -> None:
        assert float(reduce_mean({"loss": jnp.array(3.0)})["loss"]) == 3.0

    @given(_VECTORS)
    @settings(deadline=None)
    def test_each_reduction_matches_numpy(self, values: np.ndarray) -> None:
        metrics = {"v": jnp.asarray(values)}
        assert np.isclose(float(reduce_mean(metrics)["v"]), values.mean(), rtol=1e-4)
        assert np.isclose(float(reduce_sum(metrics)["v"]), values.sum(), rtol=1e-4, atol=1e-2)
        assert float(reduce_max(metrics)["v"]) == values.max()
        assert float(reduce_min(metrics)["v"]) == values.min()


class TestReduceCustom:
    def test_applies_a_reduction_per_key(self) -> None:
        metrics = {
            "loss": jnp.array([1.0, 2.0, 3.0]),
            "count": jnp.array([1.0, 2.0, 3.0]),
            "peak": jnp.array([1.0, 5.0, 3.0]),
        }
        result = reduce_custom(metrics, {"loss": "mean", "count": "sum", "peak": "max"})
        assert (float(result["loss"]), float(result["count"]), float(result["peak"])) == (
            2.0,
            6.0,
            5.0,
        )

    def test_missing_keys_default_to_mean(self) -> None:
        assert float(reduce_custom({"loss": jnp.array([1.0, 3.0])}, {})["loss"]) == 2.0

    def test_none_leaves_a_metric_unreduced(self) -> None:
        result = reduce_custom({"raw": jnp.array([5.0, 10.0])}, {"raw": None})
        assert result["raw"].shape == (2,)


def _on_data_axis(mesh: Mesh, values: list[float]) -> jax.Array:
    """Place a vector sharded on the data axis, as shard_map's in_specs require."""
    return jax.device_put(jnp.array(values), NamedSharding(mesh, PartitionSpec("data")))


class TestCollectives:
    @staticmethod
    def _mesh() -> Mesh:
        return jax.make_mesh((1,), ("data",))

    def test_positive_control_mean_collective_inside_shard_map(self) -> None:
        """A real shard_map on a one-device mesh exercises lax.pmean end to end."""
        mesh = self._mesh()

        def body(x: jax.Array) -> jax.Array:
            return reduce_mean_collective({"x": x}, axis_name="data")["x"]

        with jax.set_mesh(mesh):
            out = jax.shard_map(
                body, mesh=mesh, in_specs=PartitionSpec("data"), out_specs=PartitionSpec()
            )(_on_data_axis(mesh, [2.0, 4.0]))
        assert out.tolist() == [2.0, 4.0]

    def test_sum_collective_and_all_gather_inside_shard_map(self) -> None:
        mesh = self._mesh()

        def body(x: jax.Array) -> tuple[jax.Array, jax.Array]:
            total = reduce_sum_collective({"x": x}, axis_name="data")["x"]
            gathered = all_gather({"x": x}, axis_name="data")["x"]
            return total, gathered

        with jax.set_mesh(mesh):
            total, gathered = jax.shard_map(
                body,
                mesh=mesh,
                in_specs=PartitionSpec("data"),
                # The gathered value carries a leading device axis, so it is emitted along
                # the data axis; the summed value is replicated.
                out_specs=(PartitionSpec(), PartitionSpec("data")),
            )(_on_data_axis(mesh, [1.0, 2.0]))
        assert total.tolist() == [1.0, 2.0]
        assert gathered.shape == (1, 2)

    def test_non_arrays_pass_through_collectives(self) -> None:
        mesh = self._mesh()

        def body(x: jax.Array) -> jax.Array:
            return reduce_mean_collective({"x": x, "step": 5}, axis_name="data")["x"]

        with jax.set_mesh(mesh):
            jax.shard_map(
                body, mesh=mesh, in_specs=PartitionSpec("data"), out_specs=PartitionSpec()
            )(_on_data_axis(mesh, [1.0]))


class TestCollectFromDevices:
    def test_splits_the_leading_device_axis(self) -> None:
        result = collect_from_devices({"loss": jnp.array([1.0, 2.0, 3.0]), "accuracy": 0.95})
        assert [float(v) for v in result["loss"]] == [1.0, 2.0, 3.0]
        assert result["accuracy"] == 0.95

    def test_scalars_and_non_arrays_are_kept(self) -> None:
        result = collect_from_devices({"scalar": jnp.array(1.0), "label": "test"})
        scalar = result["scalar"]
        assert isinstance(scalar, jax.Array)
        assert float(scalar) == 1.0
        assert result["label"] == "test"
