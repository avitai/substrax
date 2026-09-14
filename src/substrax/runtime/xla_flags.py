"""``XLA_FLAGS`` parsed and merged by flag name."""

from __future__ import annotations

from collections.abc import Iterable

from substrax.runtime.errors import XlaFlagConflictError


_FLAG_PREFIX = "--"


def parse_xla_flags(value: str) -> dict[str, str | None]:  # noqa: DOC502
    """Parse an ``XLA_FLAGS`` value into flag names mapped to their values, in order.

    Flags are separated by whitespace. ``--name=value`` maps ``--name`` to ``value``, and a bare
    ``--name`` maps it to ``None``. A flag given twice with the same value counts once.

    Args:
        value: The ``XLA_FLAGS`` string.

    Returns:
        Each flag name, with its leading dashes, mapped to its value.

    Raises:
        ValueError: If a token is not a flag.
        XlaFlagConflictError: If one flag is given two different values.
    """
    return _with_flags({}, value.split())


def merge_xla_flags(existing: str, requested: Iterable[str]) -> str:  # noqa: DOC502
    """Merge requested flags into an ``XLA_FLAGS`` value by flag name.

    Flags already present keep their position, and new flags follow in the order requested. A
    requested flag already set to the same value changes nothing, and one set to a different value
    raises, so a value already in the environment is never silently replaced.

    Args:
        existing: The current ``XLA_FLAGS`` value.
        requested: Flags to add, each ``--name`` or ``--name=value``.

    Returns:
        The merged ``XLA_FLAGS`` value, with flags separated by single spaces.

    Raises:
        ValueError: If a token is not a flag.
        XlaFlagConflictError: If a requested flag has a different value from an existing one.
    """
    flags = _with_flags(parse_xla_flags(existing), requested)
    return " ".join(name if value is None else f"{name}={value}" for name, value in flags.items())


def _with_flags(flags: dict[str, str | None], tokens: Iterable[str]) -> dict[str, str | None]:
    """Return ``flags`` with each token added, raising on a conflicting value."""
    merged = dict(flags)
    for token in tokens:
        name, value = _split_flag(token)
        if name in merged and merged[name] != value:
            raise XlaFlagConflictError(name, merged[name], value)
        merged[name] = value
    return merged


def _split_flag(token: str) -> tuple[str, str | None]:
    """Split ``--name=value`` into its name and value; a bare ``--name`` has no value."""
    name, separator, value = token.partition("=")
    if not name.startswith(_FLAG_PREFIX) or name == _FLAG_PREFIX or any(c.isspace() for c in token):
        raise ValueError(f"{token!r} is not an XLA flag of the form --name or --name=value")
    return name, value if separator else None
