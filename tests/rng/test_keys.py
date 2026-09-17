"""``substrax.rng``: keys from explicit owners, seeds folded by stream name, no silent fallback."""

from __future__ import annotations

import hashlib
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from substrax.rng import (
    fold_in_name,
    key_from,
    MissingRngStreamError,
    rngs_from_seed,
    split_key,
)
from substrax.testing import run_python


def _data(key: jax.Array) -> np.ndarray:
    return np.asarray(jax.random.key_data(key))


class TestKeyFrom:
    def test_the_first_present_stream_wins_in_the_order_given(self) -> None:
        rngs = nnx.Rngs(sample=1, default=2)

        key = key_from(rngs, streams=("dropout", "sample", "default"), context="sampling")

        assert np.array_equal(_data(key), _data(nnx.Rngs(sample=1).sample()))

    def test_a_missing_stream_raises_naming_the_context_and_the_streams(self) -> None:
        with pytest.raises(MissingRngStreamError, match=r"sampling .* 'sample' or 'default'"):
            key_from(nnx.Rngs(params=0), streams=("sample", "default"), context="sampling")

    def test_the_error_is_a_value_error(self) -> None:
        assert issubclass(MissingRngStreamError, ValueError)

    def test_a_typed_key_is_returned_as_given(self) -> None:
        key = jax.random.key(3)

        assert key_from(key, streams=("sample",), context="sampling") is key

    def test_a_raw_key_is_returned_as_given(self) -> None:
        key = jax.random.PRNGKey(3)

        assert key_from(key, streams=("sample",), context="sampling") is key

    @pytest.mark.parametrize(
        "value", [7, jnp.ones(2, dtype=jnp.float32), None, jnp.zeros((3, 2), dtype=jnp.uint32)]
    )
    def test_anything_but_a_key_or_rngs_is_a_type_error(self, value: object) -> None:
        with pytest.raises(TypeError, match="sampling"):
            key_from(value, streams=("sample",), context="sampling")  # pyright: ignore[reportArgumentType]

    def test_works_under_nnx_jit(self) -> None:
        @nnx.jit
        def draw(rngs: nnx.Rngs) -> jax.Array:
            return jax.random.normal(key_from(rngs, streams=("sample",), context="draw"), (3,))

        first = draw(nnx.Rngs(sample=0))
        second = draw(nnx.Rngs(sample=0))

        assert first.shape == (3,)
        np.testing.assert_array_equal(np.asarray(first), np.asarray(second))


class TestRngsFromSeed:
    def test_each_stream_is_the_seed_folded_by_its_name(self) -> None:
        rngs = rngs_from_seed(5, streams=("params", "dropout"))

        for name in ("params", "dropout"):
            expected = nnx.Rngs(**{name: fold_in_name(jax.random.key(5), name)})
            assert np.array_equal(_data(getattr(rngs, name)()), _data(getattr(expected, name)()))

    def test_adding_a_stream_does_not_move_the_others(self) -> None:
        two = rngs_from_seed(5, streams=("params", "dropout"))
        three = rngs_from_seed(5, streams=("params", "dropout", "sample"))

        for name in ("params", "dropout"):
            assert np.array_equal(_data(getattr(two, name)()), _data(getattr(three, name)()))

    def test_the_default_stream_is_default(self) -> None:
        rngs = rngs_from_seed(0)

        assert "default" in rngs
        assert rngs.default().shape == ()

    def test_streams_differ_from_each_other_and_from_the_seed_key(self) -> None:
        rngs = rngs_from_seed(0, streams=("params", "dropout"))
        params = _data(rngs.params())
        dropout = _data(rngs.dropout())

        assert not np.array_equal(params, dropout)
        assert not np.array_equal(params, _data(nnx.Rngs(params=jax.random.key(0)).params()))

    def test_a_negative_seed_or_an_empty_stream_list_raises(self) -> None:
        with pytest.raises(ValueError, match="seed"):
            rngs_from_seed(-1)
        with pytest.raises(ValueError, match="streams"):
            rngs_from_seed(0, streams=())


class TestSplitKey:
    def test_returns_one_key_per_requested_split(self) -> None:
        keys = split_key(jax.random.key(0), 3)

        assert keys.shape == (3,)
        assert jnp.issubdtype(keys.dtype, jax.dtypes.prng_key)

    def test_one_split_keeps_the_leading_axis(self) -> None:
        assert split_key(jax.random.key(0), 1).shape == (1,)

    def test_zero_or_fewer_splits_raise(self) -> None:
        with pytest.raises(ValueError, match="num"):
            split_key(jax.random.key(0), 0)


class TestFoldInName:
    def test_folds_in_the_blake2b_digest_of_the_name(self) -> None:
        key = jax.random.key(0)
        digest = int.from_bytes(hashlib.blake2b(b"params", digest_size=4).digest(), "big")
        assert digest < 2**31

        expected = jax.random.fold_in(key, jnp.uint32(digest))

        assert np.array_equal(_data(fold_in_name(key, "params")), _data(expected))

    def test_a_digest_above_the_signed_range_is_folded_in_unchanged(self) -> None:
        # "dropout" digests above 2**31, where a Python int would overflow int32.
        digest = int.from_bytes(hashlib.blake2b(b"dropout", digest_size=4).digest(), "big")
        assert digest >= 2**31

        key = fold_in_name(jax.random.key(0), "dropout")

        assert np.array_equal(
            _data(key), _data(jax.random.fold_in(jax.random.key(0), jnp.uint32(digest)))
        )

    def test_different_names_give_different_keys(self) -> None:
        key = jax.random.key(0)

        assert not np.array_equal(_data(fold_in_name(key, "a")), _data(fold_in_name(key, "b")))

    def test_an_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="name"):
            fold_in_name(jax.random.key(0), "")

    def test_is_stable_across_interpreters_with_different_hash_seeds(self) -> None:
        program = textwrap.dedent(
            """
            import json, jax
            from substrax.rng import fold_in_name
            key = fold_in_name(jax.random.key(0), "dropout")
            print(json.dumps([int(v) for v in jax.random.key_data(key).tolist()]))
            """
        )
        results = [
            run_python(program, timeout=120, env={"PYTHONHASHSEED": seed}).check().last_json()
            for seed in ("0", "12345")
        ]

        assert results[0] == results[1]
        assert results[0] == [int(v) for v in _data(fold_in_name(jax.random.key(0), "dropout"))]

    def test_works_under_jit(self) -> None:
        folded = jax.jit(lambda key: fold_in_name(key, "sample"))(jax.random.key(0))

        assert np.array_equal(_data(folded), _data(fold_in_name(jax.random.key(0), "sample")))
