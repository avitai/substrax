"""Type aliases over Flax NNX, shared across the avitai packages; this module imports flax.

The aliases that import no jax are in :mod:`substrax.typing`.
"""

from __future__ import annotations

import jax
from flax import nnx
from flax.typing import Key


type NnxState = nnx.State[Key, NnxState | nnx.Variable[jax.Array]]
"""A module's state, as ``nnx.split`` and ``nnx.state`` return it: keyed by attribute name or
list index (flax's ``Key``), each value a nested state or a ``Variable`` holding an array. A gradient
taken with respect to a state has the same structure."""
