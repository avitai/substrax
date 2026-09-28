"""Type aliases shared across the avitai packages; importing this module imports no jax."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


PyTree = Any
"""Any JAX pytree: nested tuples, lists and dicts of arrays and leaves."""

type JsonValue = str | int | float | bool | list[JsonValue] | dict[str, JsonValue] | None
"""A value JSON can hold: a scalar, or a list or object of JSON values (as Orbax's tree types)."""

type CheckpointState = dict[str, Any]
"""The state a :class:`Checkpointable` hands a checkpoint: arrays, plain-Python leaves and nested
dictionaries, laid out by each implementer. Its values stay ``Any``, the type Grain's
``DatasetIterator.get_state``, PyTorch's ``Stateful.state_dict`` and flax's ``nnx.to_pure_dict``
give the same object: each implementer reads its own layout back."""


@runtime_checkable
class Checkpointable(Protocol):
    """An object a checkpoint restores: it hands over its state and takes it back.

    The state is a dictionary a checkpoint store can write (arrays and plain-Python leaves),
    and ``set_state`` on an object built the same way continues exactly where the saved one
    stood. A training loop checkpoints its data iterator, callbacks and extensions through
    this protocol without importing their packages.
    """

    def get_state(self) -> CheckpointState:
        """The state to checkpoint."""
        ...

    def set_state(self, state: CheckpointState, /) -> None:
        """Take back a state ``get_state`` returned."""
        ...
