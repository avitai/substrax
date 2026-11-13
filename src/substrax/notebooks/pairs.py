"""Generate an example's clean notebook from its script, and check that the pair agrees.

The script is the source: :func:`convert` reads it as jupytext's percent format and writes the
``.ipynb`` beside it with no outputs, no execution counts, no cell ids and fixed metadata, so the
same script always gives the same bytes. :func:`check_pair` never writes: the committed notebook
must be exactly what :func:`convert` would write now, and when it is not, the problem says why
(outputs, cells that differ from the script's, or metadata, ids or layout another writer left), each
fixed by ``substrax-notebooks convert``. A comparison by content holds after a fresh checkout, where
file times say nothing, and does not take the newest file as the source, as ``jupytext --sync``
does.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import jupytext
import nbformat

from substrax.notebooks.source import ExampleSourceError, read_source


FORMATS = "ipynb,py:percent"
"""The pairing every example declares: the notebook and the percent-format script."""

_CONVERT = "substrax-notebooks convert"
_KERNELSPEC = {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"}

type Cell = tuple[str, str]
"""A cell as its kind (``code`` or ``markdown``) and its text."""

type ProblemKind = Literal["missing", "out-of-sync", "not-clean", "not-generated", "unreadable"]


@dataclass(frozen=True, slots=True, kw_only=True)
class PairProblem:
    """One way an example's script and notebook fail to form a clean pair.

    Attributes:
        script: The example script.
        kind: ``missing`` (no notebook), ``out-of-sync`` (the cells differ from the script's),
            ``not-clean`` (the notebook holds outputs or execution counts), ``not-generated`` (the
            cells agree but the bytes are not what :func:`convert` writes) or ``unreadable`` (it
            is not a notebook).
        detail: What differs, naming the cell and line where there is one.
    """

    script: Path
    kind: ProblemKind
    detail: str

    def __str__(self) -> str:
        """Return the problem as one line naming the script."""
        return f"{self.script}: {self.kind}: {self.detail}"


def convert(script: Path) -> Path:
    """Write the clean notebook for an example script beside it.

    Args:
        script: The example's ``.py`` file, in jupytext's percent format.

    Returns:
        The notebook's path: the script's, with ``.ipynb``.

    Raises:
        ValueError: If ``script`` is not a ``.py`` file, or not UTF-8
            (:class:`~substrax.notebooks.source.ExampleSourceError`).
    """
    if script.suffix != ".py":
        msg = f"{script} is not a .py example script"
        raise ValueError(msg)
    target = script.with_suffix(".ipynb")
    target.write_text(_generated(script), encoding="utf-8")
    return target


def check_pair(script: Path) -> list[PairProblem]:
    """Check that an example's notebook is exactly what :func:`convert` writes, without writing.

    Args:
        script: The example's ``.py`` file.

    Returns:
        Every problem found; empty when the committed notebook equals the generated one.
    """
    notebook_path = script.with_suffix(".ipynb")
    if not notebook_path.exists():
        return [PairProblem(script=script, kind="missing", detail=f"no {notebook_path.name}")]
    committed = notebook_path.read_text(encoding="utf-8")
    try:
        generated = _generated(script)
    except ExampleSourceError as error:
        return [PairProblem(script=script, kind="unreadable", detail=str(error))]
    if committed == generated:
        return []
    try:
        notebook: dict[str, Any] = json.loads(committed)
        notebook_cells = _cells(notebook)
    except (json.JSONDecodeError, AttributeError, TypeError) as error:
        return [PairProblem(script=script, kind="unreadable", detail=str(error))]
    problems = [
        PairProblem(script=script, kind="not-clean", detail=detail)
        for detail in _unclean_cells(notebook)
    ]
    difference = _first_difference(_cells(json.loads(generated)), notebook_cells)
    if difference is not None:
        problems.append(PairProblem(script=script, kind="out-of-sync", detail=difference))
    if not problems:
        detail = f"metadata, cell ids or layout differ from what {_CONVERT} writes; run it"
        problems.append(PairProblem(script=script, kind="not-generated", detail=detail))
    return problems


def _generated(script: Path) -> str:
    """The notebook :func:`convert` writes for ``script``, as text."""
    notebook = jupytext.reads(read_source(script), fmt="py:percent")
    notebook.metadata = {
        "jupytext": {"formats": FORMATS},
        "kernelspec": dict(_KERNELSPEC),
        "language_info": {"name": "python"},
    }
    notebook.nbformat_minor = 4
    for cell in notebook.cells:
        cell.pop("id", None)
        cell.metadata = {}
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    return nbformat.writes(notebook) + "\n"


def _cells(notebook: dict[str, Any]) -> list[Cell]:
    """Every cell of a notebook's JSON, as its kind and its exact text."""
    cells: list[Cell] = []
    for cell in notebook.get("cells", []):
        source = cell.get("source", "")
        cells.append((cell["cell_type"], "".join(source) if isinstance(source, list) else source))
    return cells


def _unclean_cells(notebook: dict[str, Any]) -> list[str]:
    """Describe each code cell holding outputs or an execution count."""
    found = []
    for index, cell in enumerate(notebook.get("cells", []), start=1):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs") or cell.get("execution_count") is not None:
            found.append(f"code cell {index} holds outputs or an execution count")
    return found


def _first_difference(generated: list[Cell], committed: list[Cell]) -> str | None:
    """Describe the first cell where the notebook differs from the script's, or ``None``."""
    if len(generated) != len(committed):
        return f"{len(generated)} cells in the script, {len(committed)} in the notebook"
    for index, ((script_kind, script_text), (notebook_kind, notebook_text)) in enumerate(
        zip(generated, committed, strict=True), start=1
    ):
        if script_kind != notebook_kind:
            return f"cell {index} is {script_kind} in the script, {notebook_kind} in the notebook"
        if script_text != notebook_text:
            script_lines, notebook_lines = script_text.splitlines(), notebook_text.splitlines()
            line = next(
                (
                    n
                    for n, (a, b) in enumerate(zip(script_lines, notebook_lines, strict=False), 1)
                    if a != b
                ),
                min(len(script_lines), len(notebook_lines)) + 1,
            )
            return f"{script_kind} cell {index} differs at line {line}"
    return None
