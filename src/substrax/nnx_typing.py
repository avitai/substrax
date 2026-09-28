"""Type aliases over JAX and Flax NNX, shared across the avitai packages; this module imports both.

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


PathEntry = (
    jax.tree_util.DictKey
    | jax.tree_util.SequenceKey
    | jax.tree_util.GetAttrKey
    | jax.tree_util.FlattenedIndexKey
)
"""An entry of a key path, as ``jax.tree.flatten_with_path`` and ``tree_map_with_path`` give it;
a path is ``jax.tree_util.KeyPath[PathEntry]``. A plain assignment, not a ``type`` statement: jax
types the entry classes ``Any`` (``jax/_src/tree_util.py``), which a lazily evaluated alias turns
into an unknown type under strict checking."""
