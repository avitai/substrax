"""Errors raised when a JAX process cannot be configured as asked."""

from __future__ import annotations


class RuntimeConfigurationError(RuntimeError):
    """A JAX process setting cannot be applied, or conflicts with one already in place."""


class XlaFlagConflictError(RuntimeConfigurationError):
    """One XLA flag was given two different values.

    Attributes:
        flag: The flag name, with its leading dashes.
        existing: The value already in place; ``None`` for a flag given without a value.
        requested: The conflicting value; ``None`` for a flag given without a value.
    """

    flag: str
    existing: str | None
    requested: str | None

    def __init__(self, flag: str, existing: str | None, requested: str | None) -> None:
        """Record the flag and both of its values.

        Args:
            flag: The flag name, with its leading dashes.
            existing: The value already in place.
            requested: The conflicting value.
        """
        super().__init__(
            f"XLA flag {flag} is already set to {existing!r} and was requested as {requested!r}"
        )
        self.flag = flag
        self.existing = existing
        self.requested = requested
