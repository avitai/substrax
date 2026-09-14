"""Where a run writes its outputs: an argument, ``AVITAI_OUTPUT_DIR``, or a per-run directory."""

from __future__ import annotations

import functools
import logging
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Final, Literal


logger = logging.getLogger(__name__)

OUTPUT_DIR_ENV: Final = "AVITAI_OUTPUT_DIR"
_RUN_DIRECTORY_PREFIX: Final = "avitai-run-"


@dataclass(frozen=True, slots=True, kw_only=True)
class OutputLocation:
    """A resolved output directory and where the choice came from.

    Attributes:
        path: The absolute directory, which exists.
        source: ``"argument"`` for a directory the caller passed, ``"environment"`` for one under
            ``AVITAI_OUTPUT_DIR``, ``"run_default"`` for one under the per-run directory.
    """

    path: Path
    source: Literal["argument", "environment", "run_default"]


def resolve_output_dir(  # noqa: DOC502
    name: str, *, explicit: Path | None = None, env: Mapping[str, str] | None = None
) -> OutputLocation:
    """Return the directory a run writes the output called ``name`` to, creating it.

    The directory is ``explicit`` when given, with a relative path resolved against the working
    directory. Otherwise it is ``$AVITAI_OUTPUT_DIR/<name>``, and otherwise ``<name>`` inside a
    directory created once per process under the system temporary directory. It never defaults
    into the working tree, so running an example or a test cannot overwrite tracked files, and
    writing into ``docs/assets`` is always an explicit choice.

    Args:
        name: The output's relative path, such as ``"fno_darcy"``.
        explicit: A directory chosen by the caller, typically from an ``--output-dir`` option.
        env: The environment ``AVITAI_OUTPUT_DIR`` is read from; ``os.environ`` when ``None``.

    Returns:
        The created directory and the source it was chosen from.

    Raises:
        ValueError: If ``name`` is empty, absolute or leaves the output directory, or
            ``AVITAI_OUTPUT_DIR`` holds a relative path.
    """
    relative = _checked_name(name)
    variable = (os.environ if env is None else env).get(OUTPUT_DIR_ENV, "")
    if explicit is not None:
        location = OutputLocation(path=explicit.resolve(), source="argument")
    elif variable:
        root = _absolute_variable(variable)
        location = OutputLocation(path=(root / relative).resolve(), source="environment")
    else:
        root = _run_directory(tempfile.gettempdir())
        location = OutputLocation(path=root / relative, source="run_default")
    location.path.mkdir(parents=True, exist_ok=True)
    logger.info("Writing %s outputs to %s (%s)", name, location.path, location.source)
    return location


def _checked_name(name: str) -> PurePath:
    """Return ``name`` as a relative path, refusing one that is empty or leaves its parent."""
    path = PurePath(name)
    if not path.parts or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"name must be a relative path inside the output directory, got {name!r}")
    return path


def _absolute_variable(value: str) -> Path:
    """Return the ``AVITAI_OUTPUT_DIR`` value as a path, refusing a relative one."""
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{OUTPUT_DIR_ENV} must be an absolute path, got {value!r}")
    return path


@functools.cache
def _run_directory(temp_dir: str) -> Path:
    """Create the directory that default outputs share, once per process and temporary directory.

    ``tempfile.mkdtemp`` creates it atomically with owner-only permissions, so another user of a
    shared temporary directory cannot claim or read it.
    """
    return Path(tempfile.mkdtemp(prefix=_RUN_DIRECTORY_PREFIX, dir=temp_dir)).resolve()
