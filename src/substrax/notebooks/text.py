r"""Read a file the example tools read the one way they all read it.

The pair check, the converter, the run, the page check and the corpus check all start from a
file's text: an example's script, its notebook, its page, a hand-written document. They read it
here: UTF-8, a byte order mark dropped (Python accepts one), every line ending made ``\n``, and
:class:`NotUtf8Error` for any other encoding rather than a ``UnicodeDecodeError`` from deep inside
a library. A checkout with Windows line endings (git's ``core.autocrlf``) therefore holds the same
text as the commit it came from.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path, PurePath


_LINE_ENDING = re.compile(r"\r\n?")
"""A Windows or a lone carriage-return line ending."""


class NotUtf8Error(ValueError):
    """A file the example tools read that is not UTF-8.

    Attributes:
        path: The file, as the message names it.
        byte: The offset of the first byte that cannot be decoded.
    """

    path: PurePath
    byte: int

    def __init__(self, path: PurePath, byte: int) -> None:
        """Name the file and the first byte that cannot be decoded.

        Args:
            path: The file, as the message names it.
            byte: The offset of the first byte that cannot be decoded.
        """
        super().__init__(f"{path} is not UTF-8: byte {byte} cannot be decoded; save it as UTF-8")
        self.path = path
        self.byte = byte


def read_utf8(path: Path) -> str:
    r"""Return a file's text: UTF-8, any byte order mark dropped, every line ending ``\n``.

    Args:
        path: The file.

    Returns:
        Its text.

    Raises:
        NotUtf8Error: If its bytes are not UTF-8.
    """
    try:
        text = path.read_bytes().decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise NotUtf8Error(path, error.start) from error
    return unix_line_endings(text)


def unix_line_endings(text: str) -> str:
    r"""Return ``text`` with each Windows (``\r\n``) or lone ``\r`` line ending made ``\n``.

    Args:
        text: Any text.

    Returns:
        The text with Unix line endings.
    """
    return _LINE_ENDING.sub("\n", text)


def text_sha256(text: str) -> str:
    """Return the sha256 of a text's UTF-8 bytes, as a page records its script and its own text.

    Args:
        text: A text as :func:`read_utf8` returns it, or one made with Unix line endings.

    Returns:
        The hex digest.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
