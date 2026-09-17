"""Which parameters an optimizer decays: NNX filters turned into optax masks."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Final

import jax
from flax import nnx

from substrax.optim.config import OptimizerConfig


_UNDECAYED_LEAVES: Final = frozenset({"bias", "scale"})


def _is_neither_bias_nor_scale(path: tuple[Any, ...], value: object) -> bool:
    del value
    return bool(path) and path[-1] not in _UNDECAYED_LEAVES


EXCLUDE_BIAS_AND_NORM_SCALE: Final[nnx.filterlib.Filter] = _is_neither_bias_nor_scale
"""Decay every parameter except biases and normalisation scales, as Hugging Face and timm do."""


def weight_decay_mask(model: nnx.Module, config: OptimizerConfig) -> Any:
    """Return the optax mask for ``config.weight_decay_filter`` over ``nnx.state(model, wrt)``.

    The mask is the parameters' pure tree (nested dicts, as ``nnx.to_pure_dict`` gives it)
    with a static Python boolean at every leaf, ``True`` where the filter selects the
    parameter. ``None`` selects every leaf. The builder passes optax the callable form,
    ``mask_callable``, which maps the same filter over whatever parameter tree optax hands it.

    Args:
        model: The model the optimizer updates.
        config: The specification holding ``wrt`` and ``weight_decay_filter``.

    Returns:
        The mask tree.

    Raises:
        ValueError: If the filter selects no parameter, which optax would accept as a silent
            no-op.
    """
    mask = mask_from_filter(nnx.to_pure_dict(nnx.state(model, config.wrt)), _chosen(config))
    if not any(jax.tree.leaves(mask)):
        raise ValueError(
            f"weight_decay_filter {config.weight_decay_filter!r} selects no parameter of "
            f"wrt {config.wrt!r}"
        )
    return mask


def mask_callable(config: OptimizerConfig) -> Callable[[Any], Any]:
    """Return the callable optax evaluates on the parameter tree to get the decay mask.

    Args:
        config: The specification holding ``weight_decay_filter``.

    Returns:
        A function from a parameter tree to a tree of Python booleans of the same structure.
    """
    chosen = _chosen(config)
    return lambda params: mask_from_filter(params, chosen)


def _chosen(config: OptimizerConfig) -> nnx.filterlib.Filter:
    return True if config.weight_decay_filter is None else config.weight_decay_filter


def mask_from_filter(params: Any, chosen: nnx.filterlib.Filter) -> Any:
    """Map an NNX filter over a parameter tree, giving a Python boolean per leaf.

    The tree may be an ``nnx.State`` or the pure tree optax hands a callable mask; a trailing
    ``value`` attribute key of a Variable is not part of the parameter's path.

    Args:
        params: The parameter tree.
        chosen: The filter, evaluated on each leaf's path and value.

    Returns:
        A tree of the same structure with a ``bool`` at every leaf.
    """
    predicate = nnx.filterlib.to_predicate(chosen)

    def decayed(keypath: tuple[Any, ...], value: object) -> bool:
        entries = [_path_key(entry) for entry in keypath]
        if entries and entries[-1] == "value" and isinstance(keypath[-1], jax.tree_util.GetAttrKey):
            entries = entries[:-1]
        return bool(predicate(tuple(entries), value))

    return jax.tree_util.tree_map_with_path(decayed, params)


def _path_key(entry: Any) -> Any:
    if isinstance(entry, jax.tree_util.DictKey):
        return entry.key
    if isinstance(entry, jax.tree_util.SequenceKey):
        return entry.idx
    if isinstance(entry, jax.tree_util.GetAttrKey):
        return entry.name
    return str(entry)
