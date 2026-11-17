"""A recipe run as its script, with what ``main()`` returned recorded beside where it came from.

:func:`run_recipe` runs a recipe in a fresh interpreter from the repository root, as
``python recipes/<path>.py`` runs it, so no module this process imported stands in for one of the
recipe's own: the interpreter executes the source of :mod:`~substrax.notebooks.kernel_setup`, as a
notebook run's kernel does, and its :func:`~substrax.notebooks.kernel_setup.run_as_script` runs
the recipe with its ``__main__`` guard keeping ``main()``'s value and writes the run's report. The
recipe's output and errors go to this process's streams. :class:`RecipeRecord` holds the report
with the recipe's path and digest. A compute task runs it as
``python -m substrax.notebooks recipe recipes/<path>.py``, whose record :func:`write_record` writes
under the task's outputs.
"""

from __future__ import annotations

import ast
import inspect
import json
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from substrax.examples import lexical_path
from substrax.notebooks import kernel_setup
from substrax.notebooks.kernel_setup import is_guarded_main_call
from substrax.notebooks.run import RunOrigin, script_sha256
from substrax.notebooks.text import read_utf8
from substrax.recipes import recipe_source
from substrax.records import dump_record, read_record
from substrax.typing import JsonValue


RECORD_OUTPUT = "recipe_record"
"""The output, under the run's outputs, that a recipe's record is written to."""
RECORD_FILE = "record.json"
"""The record's file in that output."""

_KERNEL_SETUP_SOURCE = inspect.getsource(kernel_setup)
_REPORT_FILE = "report.json"


class RecipeRunError(Exception):
    """A recipe's run left no record: it failed, no guard ran ``main()``, or JSON cannot hold it."""


@dataclass(frozen=True, slots=True, kw_only=True)
class RecipeRecord(RunOrigin):
    """What a recipe's run returned, and where it came from (:class:`RunOrigin`).

    Attributes:
        returned: What ``main()`` returned, decoded from JSON; ``None`` when it returned nothing.
    """

    returned: JsonValue = None


@dataclass(frozen=True, slots=True, kw_only=True)
class _ScriptReport:
    """The report :func:`~substrax.notebooks.kernel_setup.run_as_script` writes."""

    device: str
    seconds: float
    versions: Mapping[str, str] = field(default_factory=dict)
    returned: JsonValue = None
    returned_error: str | None = None


def run_recipe(recipe: Path, *, repo_root: Path, distribution: str) -> RecipeRecord:
    """Run a recipe as its script in a fresh interpreter and record what ``main()`` returned.

    Args:
        recipe: The recipe, under ``repo_root / "recipes"``.
        repo_root: The repository root, which the run is taken from.
        distribution: The distribution the repository builds, whose version the record holds.

    Returns:
        The run's record.

    Raises:
        RecipeRunError: If the recipe has no top-level ``__main__`` guard calling ``main()``, its
            run exited with a status other than 0, or ``main()`` returned a value JSON cannot hold.
    """
    source = recipe_source(recipe, repo_root=repo_root)
    tree = ast.parse(read_utf8(recipe), filename=str(recipe))
    if not any(is_guarded_main_call(statement) for statement in tree.body):
        msg = (
            f'{source}: no `if __name__ == "__main__": main()` ran main(); a recipe ends with it, '
            "and its run records what main() returns"
        )
        raise RecipeRunError(msg)
    script = lexical_path(recipe)
    with tempfile.TemporaryDirectory(prefix="substrax-recipe-") as scratch:
        report_path = Path(scratch) / _REPORT_FILE
        completed = subprocess.run(  # noqa: S603  # nosec B603 - this interpreter, no shell
            [
                sys.executable,
                "-c",
                _program(script, argv0=source, report_path=report_path, distribution=distribution),
            ],
            cwd=repo_root,
            check=False,
        )
        if completed.returncode != 0:
            msg = (
                f"{source}: its run exited with status {completed.returncode}; what it printed, "
                "and any traceback, is on standard error"
            )
            raise RecipeRunError(msg)
        reported = read_record(_ScriptReport, json.loads(report_path.read_text(encoding="utf-8")))
    if reported.returned_error is not None:
        msg = f"{source}: {reported.returned_error}"
        raise RecipeRunError(msg)
    return RecipeRecord(
        source=source,
        sha256=script_sha256(recipe),
        seconds=round(reported.seconds, 1),
        device=reported.device,
        versions=reported.versions,
        returned=reported.returned,
    )


def write_record(record: RecipeRecord, directory: Path) -> Path:
    """Write a recipe's record as JSON into ``directory``, creating it.

    Args:
        record: The record.
        directory: Where it is written, such as the run's :data:`RECORD_OUTPUT` output.

    Returns:
        The record's file, :data:`RECORD_FILE` in ``directory``.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / RECORD_FILE
    path.write_text(json.dumps(dump_record(record), indent=2) + "\n", encoding="utf-8")
    return path


def _program(script: Path, *, argv0: str, report_path: Path, distribution: str) -> str:
    """The fresh interpreter's program: the kernel setup's source, then the script's run."""
    arguments = {
        "directory": str(script.parent),
        "argv0": argv0,
        "report_path": str(report_path),
        "distribution": distribution,
    }
    keywords = ", ".join(f"{name}={value!r}" for name, value in arguments.items())
    run = kernel_setup.run_as_script.__name__
    return (
        "_substrax_setup = {}\n"
        f"exec(compile({_KERNEL_SETUP_SOURCE!r}, {kernel_setup.__name__!r}, 'exec'), "
        "_substrax_setup)\n"
        f"_substrax_setup[{run!r}]({str(script)!r}, {keywords})\n"
    )
