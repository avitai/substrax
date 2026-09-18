"""Records read from JSON: frozen dataclasses validated by pydantic's strict JSON mode.

A record is a frozen dataclass whose ``to_dict`` writes JSON values; ``read_record`` reads
one back from the parsed JSON object. Each field is checked against its annotation as JSON
defines the types: a number is never read from a string, a ``bool`` is not a number, an
integer is accepted where a float is expected (JSON has one number type), nested dataclasses,
``datetime`` (ISO 8601 text), enums (by value) and tuples (from arrays) are read by their
annotations, and a missing field takes its default. A refused record raises
``pydantic.ValidationError``, a ``ValueError`` listing each field's path through the record.

The object is re-serialised to JSON text and validated in pydantic's JSON mode, because its
strict Python mode reads a dataclass only from an instance of it; a record is small and read
at the edge of a program, never inside a traced function.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Mapping

from pydantic import TypeAdapter

from substrax.typing import JsonValue


@functools.cache
def _adapter(record_type: type[object]) -> TypeAdapter[object]:
    """One validator per record type, built on first use (about 15 ms each)."""
    return TypeAdapter(record_type)


def read_record[T](  # noqa: DOC503  # pydantic.ValidationError is raised by validate_json
    record_type: type[T], data: Mapping[str, JsonValue]
) -> T:
    """Read a record of ``record_type`` from a parsed JSON object.

    Args:
        record_type: The record's dataclass.
        data: The JSON object, as ``json.loads`` or Orbax's ``JsonRestore`` returns it.

    Returns:
        The record.

    Raises:
        pydantic.ValidationError: If a field is missing or holds a value its annotation does not
            admit; ``errors()`` gives each field's path as ``loc``.
        TypeError: If the validator returns something other than the record type, which
            pydantic does not do for a dataclass.
    """
    record = _adapter(record_type).validate_json(json.dumps(data), strict=True)
    if not isinstance(record, record_type):
        msg = f"validating {record_type.__name__} returned {type(record).__name__}"
        raise TypeError(msg)
    return record
