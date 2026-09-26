"""The shared type aliases live in one light module."""

from __future__ import annotations

import substrax.checkpoint.metadata
import substrax.typing
from substrax.testing import run_python


def test_the_checkpoint_record_uses_the_one_json_alias() -> None:
    assert substrax.checkpoint.metadata.JsonValue is substrax.typing.JsonValue


def test_importing_the_aliases_loads_no_jax_or_orbax() -> None:
    result = run_python(
        "import sys, substrax.typing; "
        "print(any(name in sys.modules for name in ('jax', 'orbax', 'flax')))",
        timeout=60,
    )

    assert result.stdout.strip() == "False"


class _Checkpointable:
    def get_state(self) -> dict[str, object]:
        return {}

    def set_state(self, state: dict[str, object]) -> None:
        del state


class _ReadOnly:
    def get_state(self) -> dict[str, object]:
        return {}


def test_checkpointable_is_whatever_can_hand_over_and_take_back_its_state() -> None:
    assert isinstance(_Checkpointable(), substrax.typing.Checkpointable)
    assert not isinstance(_ReadOnly(), substrax.typing.Checkpointable)


def test_json_value_names_the_json_types() -> None:
    assert str(substrax.typing.JsonValue.__value__) == (
        "str | int | float | bool | list[JsonValue] | dict[str, JsonValue] | None"
    )
