# Rng

`substrax.rng` takes keys from an explicit owner and derives streams from a seed by name.
Nothing in it falls back to a default seed: two callers never share a key they did not ask
for.

```python
from flax import nnx

from substrax.rng import (
    SAMPLE_STREAM,
    STANDARD_STREAMS,
    fold_in_name,
    key_from,
    rngs_from_seed,
    split_key,
    stream_key,
)

rngs = rngs_from_seed(42, streams=STANDARD_STREAMS)

# A method that draws takes the caller's Rngs and asks one stream for its next key.
draw_key = stream_key(rngs, SAMPLE_STREAM)

# A sampler accepts an Rngs with a `sample` stream, or a key given directly.
key = key_from(rngs, streams=("sample", "default"), context="sampling")
per_walker = split_key(key, 1024)

# A sub-component derives its own key from a name, the same in every interpreter.
noise_key = fold_in_name(key, "observation_noise")
```

| Function | Behaviour |
| --- | --- |
| `key_from(rng, *, streams, context)` | With an `nnx.Rngs`, calls the first of `streams` it holds, in the order given; a typed key or a raw `uint32` key of shape `(2,)` is returned as given. A missing stream raises `MissingRngStreamError` naming `context` and the streams; `None`, a Python integer or any other array raises `TypeError` |
| `stream_key(rngs, stream)` | The next key of one stream of an `nnx.Rngs`, by flax's own lookup `rngs[stream]()`: a name the `nnx.Rngs` does not hold falls back to its `default` stream, and with neither flax raises `KeyError`. Anything that is not an `nnx.Rngs` (a typed or raw key, an `nnx.RngStream`, `None`, an integer) raises `TypeError` naming what it got |
| `legacy_key_shape()`, `is_legacy_key(value)` | The shape of one legacy `uint32` key under the default PRNG implementation (`(2,)` for threefry, `(4,)` under `rbg`), read from the implementation each time; and whether a value is `uint32` data ending in that shape, one key or a batch |
| `rngs_from_seed(seed, streams=("default",))` | One stream per name, each the seed's key folded by the name, so adding or reordering streams leaves the others unchanged |
| `split_key(key, num)` | `jax.random.split` with `num` at least one; asking for no keys is a defect |
| `fold_in_name(key, name)` | Folds in the name's BLAKE2b digest truncated to 32 bits, stable across interpreters whatever `PYTHONHASHSEED` is |

`rngs_from_seed` differs from splitting the seed key into one part per stream, which every
package used to do: there, appending a stream moved every existing stream's key, so a
recorded run stopped reproducing when a model gained a `dropout` stream.

## Stream names and the call-time contract

The stack's stream names are defined once: `DEFAULT_STREAM`, `PARAMS_STREAM`, `DROPOUT_STREAM`
and `SAMPLE_STREAM`, together `STANDARD_STREAMS`. A module that draws keeps its stream's *name*,
never a key or a stream, as flax's `nnx.Dropout(rng_collection="dropout")` does, and takes the
`nnx.Rngs` on each call:

```python
class Prior(nnx.Module):
    def __init__(self, *, rng_collection: str = SAMPLE_STREAM) -> None:
        self.rng_collection = rng_collection

    def sample(self, shape: tuple[int, ...], *, rngs: nnx.Rngs) -> jax.Array:
        return jax.random.normal(stream_key(rngs, self.rng_collection), shape)
```

An `nnx.Rngs` advances only where NNX can see it. Passed into a plain JAX transform
(`jax.jit`, `jax.vmap`, `jax.lax.scan`), or broadcast by `nnx.vmap` or `nnx.scan`, it repeats
its keys without an error. Carry or split it inside NNX transforms, or build one per step from
a key folded with the step.

The pytest plugin (`substrax.testing.pytest_plugin`) provides `rng_key`, a typed key from
seed 42, and `rngs`, an `nnx.Rngs` from the same seed with the `STANDARD_STREAMS`.

::: substrax.rng
