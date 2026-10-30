"""The child side of a ``run_python`` report: one JSON value, written to a file only it knows.

A child interpreter shares its standard output with every library it loads, and a C or C++
library can write there at any moment, before or after the child's own last line. The report
therefore travels on its own channel: ``run_python`` names a file in the child's environment,
the child writes its value there with :func:`report`, and the parent reads it with
``ChildResult.report_as``. Reporting imports no jax, like the rest of ``substrax.testing``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from substrax.typing import JsonValue


REPORT_PATH_ENV = "SUBSTRAX_CHILD_REPORT"


def report(value: JsonValue) -> None:
    """Send ``value`` to the parent that started this interpreter with ``run_python``.

    A child reports once. The value is written to a temporary file in the report's directory and
    moved into place, so the parent reads either the whole value or none of it.

    Args:
        value: The child's result, a value JSON can hold.

    Raises:
        RuntimeError: If this interpreter was not started by ``run_python``, or has reported
            already.
    """
    target = os.environ.get(REPORT_PATH_ENV)
    if not target:
        raise RuntimeError(
            f"report() runs only in a child started by run_python, which sets {REPORT_PATH_ENV}"
        )
    path = Path(target)
    if path.exists():
        raise RuntimeError(f"this child already reported: {path.read_text(encoding='utf-8')}")
    staging = path.with_name(f"{path.name}.partial")
    staging.write_text(json.dumps(value), encoding="utf-8")
    staging.replace(path)
