"""The check every callback state passes before it is taken back."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any


def check_state_keys(state: Mapping[str, Any], expected: Collection[str], *, owner: str) -> None:
    """Refuse a state whose keys are not the ones ``owner`` writes.

    Args:
        state: The state to take back.
        expected: The keys ``owner.get_state`` writes.
        owner: The class taking the state back, named in the error.

    Raises:
        ValueError: If a key is missing or unknown.
    """
    missing = sorted(set(expected) - set(state))
    unknown = sorted(set(state) - set(expected))
    if missing or unknown:
        msg = f"{owner} state has missing keys {missing} and unknown keys {unknown}"
        raise ValueError(msg)
