"""Keys from an explicit owner, streams derived from a seed by name, splits and named folds.

``key_from`` takes a key from an ``nnx.Rngs`` stream or a key given directly and never falls
back to a default seed; ``rngs_from_seed`` derives each stream from the seed and the stream's
name, so streams are independent of each other; ``split_key`` and ``fold_in_name`` are the
two derivations every package needs, with a digest that is stable across interpreters.
"""

from substrax.rng.keys import (
    fold_in_name,
    key_from,
    MissingRngStreamError,
    rngs_from_seed,
    split_key,
)


__all__ = [
    "MissingRngStreamError",
    "fold_in_name",
    "key_from",
    "rngs_from_seed",
    "split_key",
]
