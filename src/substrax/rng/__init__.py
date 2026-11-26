"""Keys from an explicit owner, streams derived from a seed by name, splits and named folds.

The stream names the stack uses are defined once (``STANDARD_STREAMS`` and its four members),
and ``stream_key`` draws one stream's next key from an ``nnx.Rngs`` given at call time,
refusing anything else. ``key_from`` takes a key from an ``nnx.Rngs`` stream or a key given
directly and never falls back to a default seed; ``rngs_from_seed`` derives each stream from
the seed and the stream's name, so streams are independent of each other; ``split_key`` and
``fold_in_name`` are the two derivations every package needs, with a digest that is stable
across interpreters.
"""

from substrax.rng.keys import (
    DEFAULT_STREAM,
    DROPOUT_STREAM,
    fold_in_name,
    key_from,
    MissingRngStreamError,
    PARAMS_STREAM,
    rngs_from_seed,
    SAMPLE_STREAM,
    split_key,
    STANDARD_STREAMS,
    stream_key,
)


__all__ = [
    "DEFAULT_STREAM",
    "DROPOUT_STREAM",
    "PARAMS_STREAM",
    "SAMPLE_STREAM",
    "STANDARD_STREAMS",
    "MissingRngStreamError",
    "fold_in_name",
    "key_from",
    "rngs_from_seed",
    "split_key",
    "stream_key",
]
