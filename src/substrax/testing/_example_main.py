"""The child side of ``run_example``: load an example by path, call its ``main()``, print the result.

``run_example`` runs this file as a script with the example's path as its only argument. The
example then sees what a script run would show it, apart from ``__name__``: its own directory first
on ``sys.path`` (unless ``PYTHONSAFEPATH`` is set), no command-line arguments, and a module
registered in ``sys.modules``, which ``dataclasses`` reads. The value ``main()`` returns is printed
as JSON on the last line of standard output.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import cast


_MODULE_NAME = "_substrax_example"


def load_example(path: Path) -> ModuleType:
    """Execute an example's module body under a name other than ``__main__``.

    The module is registered in ``sys.modules`` before it runs, as an import would register it.
    Without that, ``dataclasses`` fails on a dataclass with string annotations.

    Args:
        path: The example.

    Returns:
        The executed module.

    Raises:
        ImportError: If ``path`` cannot be loaded as a Python module.
    """
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path} as a module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def to_json(summary: object) -> str:
    """Encode what ``main()`` returned as one line of JSON.

    A 0-d numpy or jax value becomes its Python scalar. JSON has no other representation for
    arrays, so anything else it cannot hold is refused.

    Args:
        summary: What ``main()`` returned.

    Returns:
        The JSON text.

    Raises:
        TypeError: If a value is neither JSON nor a 0-d array; for a mapping, the message names the
            key holding it.
    """
    if isinstance(summary, Mapping):
        for key, value in cast("Mapping[object, object]", summary).items():
            try:
                json.dumps(value, default=_scalar)
            except TypeError as error:
                raise TypeError(
                    f"main() returned {key!r}, which JSON cannot hold: {error}"
                ) from error
    return json.dumps(summary, default=_scalar)


def main(argv: list[str]) -> None:
    """Run the example named by ``argv`` and print what its ``main()`` returns.

    Args:
        argv: The example's path, alone.
    """
    [example] = argv
    path = Path(example).resolve()
    if not sys.flags.safe_path:
        # A script run puts the script's own directory first; this file's directory is there now.
        sys.path[0] = str(path.parent)
    sys.argv = [str(path)]
    summary = load_example(path).main()
    # A leading newline keeps the JSON on its own line after output without a trailing newline.
    sys.stdout.write(f"\n{to_json(summary)}\n")


def _scalar(value: object) -> object:
    """Return a 0-d array's Python scalar for ``json.dumps``, and refuse anything else."""
    shape = getattr(value, "shape", None)
    item = getattr(value, "item", None)
    if shape == () and callable(item):
        return item()
    raise TypeError(f"{type(value).__name__} with shape {shape} is not a JSON value")


if __name__ == "__main__":
    main(sys.argv[1:])
