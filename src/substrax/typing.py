"""Type aliases shared across the avitai packages; importing this module imports no jax."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


PyTree = Any
"""Any JAX pytree: nested tuples, lists and dicts of arrays and leaves."""

type JsonValue = str | int | float | bool | list[JsonValue] | dict[str, JsonValue] | None
"""A value JSON can hold: a scalar, or a list or object of JSON values (as Orbax's tree types)."""


@runtime_checkable
class Checkpointable(Protocol):
    """An object a checkpoint restores: it hands over its state and takes it back.

    The state is a dictionary a checkpoint store can write (arrays and plain-Python leaves),
    and ``set_state`` on an object built the same way continues exactly where the saved one
    stood. A training loop checkpoints its data iterator, callbacks and extensions through
    this protocol without importing their packages.
    """

    def get_state(self) -> dict[str, Any]:
        """The state to checkpoint."""
        ...

    def set_state(self, state: dict[str, Any], /) -> None:
        """Take back a state ``get_state`` returned."""
        ...
