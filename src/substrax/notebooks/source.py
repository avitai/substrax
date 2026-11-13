"""Read an example script's source the one way every example tool reads it.

The pair check, the converter and the conformance rules all start from the script's text, so they
read it here: UTF-8, a byte order mark dropped (Python accepts one), and a clear refusal for any
other encoding rather than a ``UnicodeDecodeError`` from deep inside a library.
"""

from __future__ import annotations

from pathlib import Path


class ExampleSourceError(ValueError):
    """An example script that is not UTF-8."""


def read_source(script: Path) -> str:
    """Return an example script's text, decoded as UTF-8 with any byte order mark dropped.

    Args:
        script: The example script.

    Returns:
        The script's text.

    Raises:
        ExampleSourceError: If the script's bytes are not UTF-8.
    """
    try:
        return script.read_bytes().decode("utf-8-sig")
    except UnicodeDecodeError as error:
        msg = f"{script} is not UTF-8: byte {error.start} cannot be decoded; save it as UTF-8"
        raise ExampleSourceError(msg) from error
