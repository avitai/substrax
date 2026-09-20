"""Records read from JSON: pydantic's strict JSON validation of a frozen dataclass."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

import pytest
from pydantic import ValidationError

from substrax.records import dump_record, read_record
from substrax.testing import run_python
from substrax.typing import JsonValue


class Direction(StrEnum):
    HIGHER = "higher"
    LOWER = "lower"


@dataclass(frozen=True, slots=True, kw_only=True)
class Sample:
    value: float
    spread: tuple[float, ...] | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class Series:
    name: str
    direction: Direction
    started: datetime
    samples: Mapping[str, Sample] = field(default_factory=dict)
    extra: Mapping[str, JsonValue] = field(default_factory=dict)


RECORD: dict[str, JsonValue] = json.loads(
    '{"name": "loss", "direction": "lower", "started": "2026-09-18T12:00:00+00:00",'
    ' "samples": {"a": {"value": 1, "spread": [0.5, 2]}}, "extra": {"k": [1, {"n": null}]}}'
)


def test_a_record_reads_with_its_nested_records_enums_times_and_tuples() -> None:
    series = read_record(Series, RECORD)

    assert series.name == "loss"
    assert series.direction is Direction.LOWER
    assert series.started == datetime.fromisoformat("2026-09-18T12:00:00+00:00")
    assert series.samples["a"] == Sample(value=1.0, spread=(0.5, 2.0))
    assert type(series.samples["a"].value) is float
    assert series.extra == {"k": [1, {"n": None}]}


def test_missing_optional_fields_take_their_defaults() -> None:
    series = read_record(
        Series, {"name": "x", "direction": "higher", "started": "2026-09-18T12:00:00"}
    )

    assert series.samples == {}
    assert series.extra == {}


@pytest.mark.parametrize(
    ("change", "location"),
    [
        ({"name": 100}, ("name",)),
        ({"direction": "sideways"}, ("direction",)),
        ({"started": 5}, ("started",)),
        ({"samples": {"a": {"value": "0.25"}}}, ("samples", "a", "value")),
        ({"samples": {"a": {"value": True}}}, ("samples", "a", "value")),
        ({"samples": {"a": {"value": 1.0, "spread": [1.0, "b"]}}}, ("samples", "a", "spread", 1)),
    ],
)
def test_a_field_of_another_json_type_is_refused_at_its_path(
    change: dict[str, JsonValue], location: tuple[str | int, ...]
) -> None:
    with pytest.raises(ValidationError) as refused:
        read_record(Series, {**RECORD, **change})

    assert refused.value.errors()[0]["loc"] == location


def test_a_missing_required_field_is_refused() -> None:
    record = dict(RECORD)
    del record["name"]

    with pytest.raises(ValidationError) as refused:
        read_record(Series, record)

    assert refused.value.errors()[0]["type"] == "missing"


def test_the_error_is_a_value_error() -> None:
    assert issubclass(ValidationError, ValueError)


def test_importing_the_module_loads_no_jax_or_orbax() -> None:
    result = run_python(
        "import sys, substrax.records; "
        "print(any(name in sys.modules for name in ('jax', 'orbax', 'flax')))",
        timeout=60,
    )

    assert result.stdout.strip() == "False"


def test_a_dumped_record_is_json_and_reads_back_equal() -> None:
    series = read_record(Series, RECORD)

    dumped = dump_record(series)

    assert json.loads(json.dumps(dumped)) == dumped
    assert dumped["direction"] == "lower"
    assert dumped["samples"] == {"a": {"value": 1.0, "spread": [0.5, 2.0]}}
    assert read_record(Series, dumped) == series


def test_dumping_something_other_than_a_dataclass_is_refused() -> None:
    with pytest.raises(TypeError, match="dict"):
        dump_record({"name": "loss"})
