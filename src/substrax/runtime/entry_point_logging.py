"""Root-logger set-up for an application, script or example entry point."""

from __future__ import annotations

import logging
from typing import TextIO


_HANDLER_NAME = "substrax.runtime.entry_point"


def configure_entry_point_logging(
    level: int = logging.INFO, *, stream: TextIO | None = None, fmt: str = "%(message)s"
) -> None:
    """Send log records to ``stream`` without replacing any other handler.

    Call it inside ``main()``, never at import. A second call replaces only the handler the first
    one installed and leaves every other handler, pytest's log capture included, in place, so it
    needs no ``force=True``.

    Args:
        level: The level set on the root logger.
        stream: Where records are written; standard error when ``None``.
        fmt: The ``logging.Formatter`` format string.
    """
    root = logging.getLogger()
    for previous in [handler for handler in root.handlers if handler.get_name() == _HANDLER_NAME]:
        root.removeHandler(previous)
        previous.close()
    handler = logging.StreamHandler(stream)
    handler.set_name(_HANDLER_NAME)
    handler.setFormatter(logging.Formatter(fmt))
    root.addHandler(handler)
    root.setLevel(level)
