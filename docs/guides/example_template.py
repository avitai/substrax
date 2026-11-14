# %% [markdown]
"""
# Normalising a Batch of Measurements

| Metadata | Value |
| --- | --- |
| **Tier** | Quick Reference |
| **Devices** | CPU-compatible |
| **Prerequisites** | Basic JAX arrays |

## Overview

One short paragraph: what the example demonstrates and why it matters to a user of this
repository. Name the feature, API or capability it shows. This template normalises a batch of
synthetic measurements, so every line of it runs as written.

## Learning Goals

1. Generate a reproducible batch of measurements from a typed key
2. Normalise the batch to zero mean and unit variance
3. Check the result's statistics
"""

# %% [markdown]
"""
## Setup

The imports, in one cell. A GPU-required example then calls `require_gpu()` from
`substrax.devices` here, right after the imports and with no arguments, as a statement of its own
or the value of an assignment, which raises unless jax's default backend is a GPU.
"""

# %%
import jax
import jax.numpy as jnp


# %% [markdown]
"""
## Core Concepts

Optional. The idea a reader needs before the code: here, that normalising subtracts the mean and
divides by the standard deviation, per feature.

## Implementation

### Step 1: Generate the measurements
"""


# %%
def make_measurements(key: jax.Array, count: int = 128, features: int = 4) -> jax.Array:
    """Draw a reproducible batch of measurements.

    Args:
        key: The typed key the draw uses.
        count: The number of measurements.
        features: The number of features per measurement.

    Returns:
        The batch, shaped ``(count, features)``.
    """
    return 3.0 + 2.0 * jax.random.normal(key, (count, features))


# %% [markdown]
"""
### Step 2: Normalise them
"""


# %%
def normalise(batch: jax.Array) -> jax.Array:
    """Shift each feature to zero mean and scale it to unit variance.

    Args:
        batch: The measurements, shaped ``(count, features)``.

    Returns:
        The normalised batch.
    """
    return (batch - batch.mean(axis=0)) / batch.std(axis=0)


# %% [markdown]
"""
## Results

### Statistics after normalising

`main()` prints what lets a reader check the step: the shape and each feature's mean and standard
deviation. The page shows the run's real output below the next cell.
"""


# %%
def main() -> dict[str, float | int]:
    """Run the example and return its results.

    Returns:
        The batch size, the largest absolute mean and the largest distance from unit standard
        deviation.
    """
    batch = normalise(make_measurements(jax.random.key(0)))
    mean_error = float(jnp.abs(batch.mean(axis=0)).max())
    std_error = float(jnp.abs(batch.std(axis=0) - 1.0).max())
    print(f"shape: {batch.shape}")
    print(f"largest |mean|: {mean_error:.2e}")
    print(f"largest |std - 1|: {std_error:.2e}")
    return {"count": batch.shape[0], "mean_error": mean_error, "std_error": std_error}


if __name__ == "__main__":
    main()


# %% [markdown]
"""
## Troubleshooting

### A feature's standard deviation is zero

Symptom: the normalised feature is `nan`. Cause: the feature is constant across the batch.
Solution: leave constant features out before normalising.

## Next Steps

- [The shared example guide](https://github.com/avitai/substrax/blob/main/docs/guides/writing-examples.md)

### Experiments to Try

- Draw a larger batch and watch the statistics approach zero and one.
"""
