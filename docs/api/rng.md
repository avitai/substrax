# Rng

`substrax.rng` takes keys from an explicit owner and derives streams from a seed by name.
Nothing in it falls back to a default seed: two callers never share a key they did not ask
for.

```python
from flax import nnx

from substrax.rng import fold_in_name, key_from, rngs_from_seed, split_key

rngs = rngs_from_seed(42, streams=("params", "dropout", "sample"))

# A sampler accepts an Rngs with a `sample` stream, or a key given directly.
key = key_from(rngs, streams=("sample", "default"), context="sampling")
per_walker = split_key(key, 1024)

# A sub-component derives its own key from a name, the same in every interpreter.
noise_key = fold_in_name(key, "observation_noise")
```

| Function | Behaviour |
| --- | --- |
| `key_from(rng, *, streams, context)` | With an `nnx.Rngs`, calls the first of `streams` it holds, in the order given; a typed key or a raw `uint32` key of shape `(2,)` is returned as given. A missing stream raises `MissingRngStreamError` naming `context` and the streams; `None`, a Python integer or any other array raises `TypeError` |
| `rngs_from_seed(seed, streams=("default",))` | One stream per name, each the seed's key folded by the name, so adding or reordering streams leaves the others unchanged |
| `split_key(key, num)` | `jax.random.split` with `num` at least one; asking for no keys is a defect |
| `fold_in_name(key, name)` | Folds in the name's BLAKE2b digest truncated to 32 bits, stable across interpreters whatever `PYTHONHASHSEED` is |

`rngs_from_seed` differs from splitting the seed key into one part per stream, which every
package used to do: there, appending a stream moved every existing stream's key, so a
recorded run stopped reproducing when a model gained a `dropout` stream.

The pytest plugin (`substrax.testing.pytest_plugin`) provides `rng_key`, a typed key from
seed 42, and `rngs`, an `nnx.Rngs` from the same seed with `default`, `params`, `dropout`
and `sample` streams.

::: substrax.rng
