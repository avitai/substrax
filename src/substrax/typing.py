"""Type aliases shared across the package."""

from __future__ import annotations

from typing import Any


PyTree = Any
"""Any JAX pytree: nested tuples, lists and dicts of arrays and leaves."""

Metrics = dict[str, Any]
"""A flat mapping from metric name to value, the shape training loops log."""
