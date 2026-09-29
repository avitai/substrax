"""Tests for distributed metrics functions.

Tests both SPMD-compatible reductions (jnp.*) and collective reductions (lax.p*).
"""

from collections.abc import Callable
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

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


# ---------------------------------------------------------------------------
# SPMD-compatible reductions (jnp.* on global arrays)
# ---------------------------------------------------------------------------


def _nested() -> dict[str, jax.Array | dict[str, jax.Array]]:
    """A metric beside a gradient tree, the shape the distributed-training guides pass."""
    return {
        "loss": jnp.array([1.0, 3.0]),
        "grads": {"w": jnp.array([2.0, 4.0]), "b": jnp.array([0.0, 2.0])},
    }


@pytest.mark.parametrize(
    ("reduce", "numpy_op"),
    [(reduce_mean, np.mean), (reduce_sum, np.sum), (reduce_max, np.max), (reduce_min, np.min)],
)
def test_every_array_leaf_of_a_nested_tree_is_reduced(
    reduce: Callable[[dict[str, jax.Array | dict[str, jax.Array]]], object],
    numpy_op: Callable[[np.ndarray], np.floating],
) -> None:
    result = reduce(_nested())

    expected = jax.tree.map(lambda leaf: numpy_op(np.asarray(leaf)), _nested())
    assert jax.tree.structure(result) == jax.tree.structure(expected)
    for got, want in zip(jax.tree.leaves(result), jax.tree.leaves(expected), strict=True):
        assert float(got) == float(want)


def test_a_nested_reduction_is_the_same_under_jit() -> None:
    assert jax.tree.map(float, jax.jit(reduce_mean)(_nested())) == jax.tree.map(
        float, reduce_mean(_nested())
    )


@pytest.mark.parametrize(
    ("collective", "combine"),
    [(reduce_mean_collective, np.mean), (reduce_sum_collective, np.sum)],
)
def test_a_collective_reduces_every_leaf_of_a_nested_tree_across_the_axis(
    collective: Callable[..., object], combine: Callable[..., np.ndarray]
) -> None:
    """Under ``vmap`` with a named axis, as inside ``pmap`` or ``shard_map``."""
    reduced = jax.vmap(lambda tree: collective(tree, axis_name="batch"), axis_name="batch")(
        _nested()
    )

    expected = jax.tree.map(lambda leaf: combine(np.asarray(leaf), axis=0), _nested())
    for got, want in zip(jax.tree.leaves(reduced), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(np.asarray(got), np.broadcast_to(want, np.shape(got)))


def test_all_gather_gathers_every_leaf_of_a_nested_tree() -> None:
    gathered = jax.vmap(lambda tree: all_gather(tree, axis_name="batch"), axis_name="batch")(
        _nested()
    )

    for got, leaf in zip(jax.tree.leaves(gathered), jax.tree.leaves(_nested()), strict=True):
        np.testing.assert_array_equal(np.asarray(got)[0], np.asarray(leaf))


class TestReduceMean:
    """Tests for reduce_mean (SPMD-compatible)."""

    def test_reduces_multidim_arrays(self) -> None:
        """Test that multi-dimensional arrays are reduced with jnp.mean."""
        metrics = {"loss": jnp.array([1.0, 2.0, 3.0])}
        result = reduce_mean(metrics)
        assert float(result["loss"]) == 2.0

    def test_scalar_arrays_unchanged(self) -> None:
        """Test that scalar arrays pass through unchanged."""
        metrics = {"loss": jnp.array(3.0)}
        result = reduce_mean(metrics)
        assert float(result["loss"]) == 3.0

    def test_non_arrays_unchanged(self) -> None:
        """Test that non-array values pass through unchanged."""
        metrics = {"loss": jnp.array([1.0, 2.0]), "step": 10, "label": "test"}
        result = reduce_mean(metrics)
        assert result["step"] == 10
        assert result["label"] == "test"


class TestReduceSum:
    """Tests for reduce_sum (SPMD-compatible)."""

    def test_reduces_arrays(self) -> None:
        """Test that arrays are summed."""
        metrics = {"count": jnp.array([1.0, 2.0, 3.0])}
        result = reduce_sum(metrics)
        assert float(result["count"]) == 6.0


class TestReduceMax:
    """Tests for reduce_max (SPMD-compatible)."""

    def test_reduces_arrays(self) -> None:
        """Test that arrays are reduced with max."""
        metrics = {"peak": jnp.array([1.0, 5.0, 3.0])}
        result = reduce_max(metrics)
        assert float(result["peak"]) == 5.0


class TestReduceMin:
    """Tests for reduce_min (SPMD-compatible)."""

    def test_reduces_arrays(self) -> None:
        """Test that arrays are reduced with min."""
        metrics = {"low": jnp.array([1.0, 5.0, 3.0])}
        result = reduce_min(metrics)
        assert float(result["low"]) == 1.0


class TestReduceCustom:
    """Tests for reduce_custom (SPMD-compatible)."""

    def test_custom_reductions(self) -> None:
        """Test applying different reductions per metric."""
        metrics = {
            "loss": jnp.array([1.0, 2.0, 3.0]),
            "count": jnp.array([1.0, 2.0, 3.0]),
            "peak": jnp.array([1.0, 5.0, 3.0]),
        }
        result = reduce_custom(
            metrics,
            reduce_fn={"loss": "mean", "count": "sum", "peak": "max"},
        )
        assert float(result["loss"]) == 2.0
        assert float(result["count"]) == 6.0
        assert float(result["peak"]) == 5.0

    def test_default_reduction_is_mean(self) -> None:
        """Test that default reduction is mean when reduce_fn is None."""
        metrics = {"loss": jnp.array([1.0, 2.0, 3.0])}
        result = reduce_custom(metrics)
        assert float(result["loss"]) == 2.0

    def test_unknown_operation_passthrough(self) -> None:
        """Test that unknown reduction operations pass value through."""
        metrics = {"val": jnp.array([5.0, 10.0])}
        result = reduce_custom(metrics, reduce_fn={"val": "unknown_op"})
        # Unknown op doesn't reduce, value passes through
        assert result["val"].shape == (2,)


# ---------------------------------------------------------------------------
# Collective reductions (lax.p* — pmap/shard_map only)
# ---------------------------------------------------------------------------


class TestReduceMeanCollective:
    """Tests for reduce_mean_collective (pmap/shard_map context)."""

    def test_reduces_arrays_with_pmean(self) -> None:
        """Test that arrays are reduced with lax.pmean."""
        with mock.patch("jax.lax.pmean", return_value=jnp.array(2.0)):
            metrics = {"loss": jnp.array(3.0), "step": 10}
            result = reduce_mean_collective(metrics)
            assert float(result["loss"]) == 2.0
            assert result["step"] == 10


class TestReduceSumCollective:
    """Tests for reduce_sum_collective (pmap/shard_map context)."""

    def test_reduces_arrays_with_psum(self) -> None:
        """Test that arrays are reduced with lax.psum."""
        with mock.patch("jax.lax.psum", return_value=jnp.array(6.0)):
            metrics = {"loss": jnp.array(3.0)}
            result = reduce_sum_collective(metrics)
            assert float(result["loss"]) == 6.0


# ---------------------------------------------------------------------------
# Gather and collect utilities
# ---------------------------------------------------------------------------


class TestAllGather:
    """Tests for all_gather (pmap/shard_map context)."""

    def test_gathers_arrays(self) -> None:
        """Test that arrays are gathered."""
        with mock.patch("jax.lax.all_gather", return_value=jnp.array([1.0, 2.0])):
            metrics = {"loss": jnp.array(1.0), "step": 5}
            result = all_gather(metrics)
            gathered = result["loss"]
            assert isinstance(gathered, jax.Array)
            assert gathered.tolist() == [1.0, 2.0]
            assert result["step"] == 5


class TestCollectFromDevices:
    """Tests for collect_from_devices."""

    def test_collects_multi_dim_arrays(self) -> None:
        """Test collecting values from multi-dimensional arrays."""
        metrics = {"loss": jnp.array([1.0, 2.0, 3.0]), "accuracy": 0.95}
        result = collect_from_devices(metrics)

        per_device = result["loss"]
        assert isinstance(per_device, list)
        assert [float(value) for value in per_device] == [1.0, 2.0, 3.0]
        assert result["accuracy"] == 0.95

    def test_scalar_arrays_unchanged(self) -> None:
        """Test that scalar arrays are kept as-is."""
        metrics = {"scalar": jnp.array(1.0)}
        result = collect_from_devices(metrics)
        assert float(result["scalar"]) == 1.0  # type: ignore[reportArgumentType]

    def test_non_array_unchanged(self) -> None:
        """Test that non-array values pass through."""
        metrics = {"label": "test", "count": 42}
        result = collect_from_devices(metrics)
        assert result["label"] == "test"
        assert result["count"] == 42
