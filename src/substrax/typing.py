"""Type aliases shared across the avitai packages; importing this module imports no jax."""

from __future__ import annotations

from typing import Any


PyTree = Any
"""Any JAX pytree: nested tuples, lists and dicts of arrays and leaves."""

type JsonValue = str | int | float | bool | list[JsonValue] | dict[str, JsonValue] | None
"""A value JSON can hold: a scalar, or a list or object of JSON values (as Orbax's tree types)."""
