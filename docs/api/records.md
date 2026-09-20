# Records

A record is a frozen dataclass written to JSON by `substrax.records.dump_record` and read back
by `substrax.records.read_record`, which validates every field against its annotation with
pydantic's strict JSON mode. `substrax.typing` holds the aliases records are typed with; it
imports nothing beyond the standard library.

```python
from dataclasses import dataclass

from substrax.records import dump_record, read_record, UNKNOWN_FIELDS_REFUSED


@dataclass(frozen=True, slots=True, kw_only=True)
class Measurement:
    __pydantic_config__ = UNKNOWN_FIELDS_REFUSED

    name: str
    value: float
    samples: tuple[float, ...] | None = None


dump_record(Measurement(name="loss", value=0.25, samples=(0.2, 0.3)))
# {'name': 'loss', 'value': 0.25, 'samples': [0.2, 0.3]}

read_record(Measurement, {"name": "loss", "value": "0.25"})
# pydantic.ValidationError: 1 validation error for Measurement
# value
#   Input should be a valid number [type=float_type, input_value='0.25', input_type=str]
```

| Annotation | Reads |
| --- | --- |
| `str`, `bool` | The JSON value of that type only |
| `int` | A JSON integer; a float or a `bool` is refused |
| `float` | A JSON number, an integer included; a string or a `bool` is refused |
| `datetime` | ISO 8601 text |
| An enum | One of its values |
| `tuple[T, ...]`, `list[T]` | A JSON array whose elements read as `T` |
| `Mapping[str, T]`, `dict[str, T]` | A JSON object whose values read as `T` |
| A dataclass | A nested JSON object, read by the same rules |
| `JsonValue` | Any JSON value, unchecked below it |

A missing field takes the dataclass default; a missing field without one is refused. A field
the record does not declare is ignored, unless the record sets `__pydantic_config__ =
UNKNOWN_FIELDS_REFUSED`, as settings and versioned files do, so a misspelled key is refused
rather than dropped. Every
refusal is collected into one `pydantic.ValidationError`, a `ValueError` whose `errors()`
give each field's path as `loc`, such as `("metrics", "loss", "value")`.

The object is written back to JSON text and validated in pydantic's JSON mode, because strict
Python-mode validation accepts a dataclass only as an instance. One validator per record type
is built on first use and kept. Records are read at the edges of a program, never inside a
function JAX traces.

::: substrax.records

::: substrax.typing
