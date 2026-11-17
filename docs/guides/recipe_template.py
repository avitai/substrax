"""Normalising a batch of measurements at its reference's scale.

The recipe of ``examples/statistics/normalising.py``: the reference's own configuration, run on a
GPU lane, never in CPU CI. Replace this paragraph with the reference: the paper, the repository
and the commit, the table or figure whose value the recipe reproduces, and the margin it must
reproduce it within, derived from the spread the reference reports across its runs. Each
constant's docstring cites the line of the reference it copies. Call the functions the example
calls, from the repository's package; this template calls ``jax.nn.standardize`` in their place.
"""

import jax
import jax.numpy as jnp


COUNT = 1_000_000
"""Measurements per batch, as the reference draws them (its table or line)."""

FEATURES = 64
"""Features per measurement, as the reference draws them (its table or line)."""

PUBLISHED_MEAN_ERROR = 1e-6
"""The largest per-feature mean the reference reports after normalising (its table or line)."""


def main() -> dict[str, float]:
    """Normalise the reference's batch; return the measured error beside the published one.

    Returns:
        Each measured value, and the published value it reproduces under ``published_<name>``.
    """
    batch = 3.0 + 2.0 * jax.random.normal(jax.random.key(0), (COUNT, FEATURES))
    normalised = jax.nn.standardize(batch, axis=0)
    mean_error = float(jnp.abs(normalised.mean(axis=0)).max())
    return {"mean_error": mean_error, "published_mean_error": PUBLISHED_MEAN_ERROR}


if __name__ == "__main__":
    main()
