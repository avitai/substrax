"""Keys from an explicit owner, streams derived from a seed by name, splits and named folds."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Final

import jax
import jax.numpy as jnp
from flax import nnx


_DIGEST_BYTES: Final = 4
_RAW_KEY_SHAPE: Final = (2,)


class MissingRngStreamError(ValueError):
    """An ``nnx.Rngs`` holds none of the streams a call accepts.

    Attributes:
        context: What needed the key, as the caller named it.
        streams: The stream names the caller accepts, in order of preference.
    """

    context: str
    streams: tuple[str, ...]

    def __init__(self, context: str, streams: Sequence[str]) -> None:
        """Name what needed the key and the streams it accepts.

        Args:
            context: What needed the key, as the caller named it.
            streams: The stream names the caller accepts, in order of preference.
        """
        self.context = context
        self.streams = tuple(streams)
        super().__init__(
            f"{context} needs an nnx.Rngs with a {_phrase(self.streams)} stream; none is present"
        )


def key_from(rng: jax.Array | nnx.Rngs, *, streams: Sequence[str], context: str) -> jax.Array:
    """Return a key from an explicit owner: an ``nnx.Rngs`` stream or a key given directly.

    With an ``nnx.Rngs``, the first of ``streams`` it holds is called, in the order given. A
    typed key (``jax.random.key``) or a raw ``uint32`` key of shape ``(2,)`` is returned as
    given. There is no fallback: a missing stream, ``None``, a Python integer or any other array
    raises, so two callers never share a key drawn from a default seed.

    Args:
        rng: The key's owner.
        streams: Stream names accepted, most preferred first.
        context: What needs the key, named in the errors (``"sampling"``, ``"dropout"``).

    Returns:
        The key.

    Raises:
        MissingRngStreamError: If ``rng`` is an ``nnx.Rngs`` holding none of ``streams``.
        TypeError: If ``rng`` is neither an ``nnx.Rngs`` nor a single key.
    """
    if isinstance(rng, nnx.Rngs):
        for name in streams:
            if name in rng:
                return getattr(rng, name)()
        raise MissingRngStreamError(context, streams)
    if not _is_single_key(rng):
        raise TypeError(
            f"{context} needs a JAX key (jax.random.key, or a raw uint32 key of shape (2,)) or "
            f"an nnx.Rngs, got {_describe(rng)}"
        )
    return rng


def rngs_from_seed(seed: int, streams: Sequence[str] = ("default",)) -> nnx.Rngs:
    """Build an ``nnx.Rngs`` whose every stream is the seed's key folded by the stream's name.

    Each stream depends on the seed and its own name only, so adding or reordering streams
    leaves the others' keys unchanged, which a split of the seed key into ``len(streams)``
    parts does not.

    Args:
        seed: A non-negative integer.
        streams: Distinct stream names; at least one.

    Returns:
        The ``nnx.Rngs`` with one stream per name.

    Raises:
        TypeError: If ``seed`` is not an integer.
        ValueError: If ``seed`` is negative, or ``streams`` is empty or repeats a name.
    """
    if isinstance(seed, bool):
        raise TypeError("seed must be an int, got a bool")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    names = tuple(streams)
    if not names:
        raise ValueError("streams must name at least one stream")
    if len(set(names)) != len(names):
        raise ValueError(f"streams must be distinct, got {names}")
    base = jax.random.key(seed)
    return nnx.Rngs(**{name: fold_in_name(base, name) for name in names})


def split_key(key: jax.Array, num: int) -> jax.Array:
    """Split ``key`` into ``num`` keys along a new leading axis.

    Args:
        key: The key to split.
        num: How many keys to return; at least one, so a caller asking for none is a defect.

    Returns:
        Keys of shape ``(num,)``.

    Raises:
        ValueError: If ``num`` is below one.
    """
    if num < 1:
        raise ValueError(f"num must be at least 1, got {num}")
    return jax.random.split(key, num)


def fold_in_name(key: jax.Array, name: str) -> jax.Array:
    """Fold a name into ``key`` through a stable digest.

    The name's BLAKE2b digest, truncated to 32 bits, is folded in, so the derived key is the
    same in every interpreter, whatever ``PYTHONHASHSEED`` is; Python's ``hash`` is salted per
    process and is never used.

    Args:
        key: The key to derive from.
        name: A non-empty name, such as a stream's.

    Returns:
        The derived key.

    Raises:
        ValueError: If ``name`` is empty.
    """
    if not name:
        raise ValueError("name must not be empty")
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=_DIGEST_BYTES).digest()
    return jax.random.fold_in(key, jnp.uint32(int.from_bytes(digest, "big")))


def _is_single_key(value: object) -> bool:
    """Whether ``value`` is one typed key or one raw ``uint32`` key."""
    if not isinstance(value, jax.Array):
        return False
    if jnp.issubdtype(value.dtype, jax.dtypes.prng_key):
        return value.ndim == 0
    return value.dtype == jnp.uint32 and value.shape == _RAW_KEY_SHAPE


def _describe(value: object) -> str:
    if isinstance(value, jax.Array):
        return f"an array of dtype {value.dtype} and shape {value.shape}"
    return type(value).__name__


def _phrase(streams: Sequence[str]) -> str:
    quoted = [f"{name!r}" for name in streams]
    if len(quoted) <= 1:
        return quoted[0] if quoted else "(no stream named)"
    return ", ".join(quoted[:-1]) + f" or {quoted[-1]}"
