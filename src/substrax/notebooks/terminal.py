"""Terminal control sequences removed from the text a page or an error message shows.

A search for a maintained package that removes every ECMA-48 control came back empty: nbconvert's
``strip_ansi`` removes control sequences (CSI) only, so a hyperlink or a window title kept its raw
escape bytes; rich's decoder prints a BEL-terminated OSC's payload as text and applies carriage
returns; and ``strip-ansi``, ``stransi`` and ``pyte`` have had no release since 2020 to 2023. The
grammar below is ECMA-48's (5th edition, 1991) escape sequences, control sequences and control
strings, in 7-bit and 8-bit form, with xterm's BEL as an alternative string terminator.
"""

from __future__ import annotations

import re


__all__ = ["strip_terminal_controls"]

_CONTROL = re.compile(
    "|".join(
        (
            # A control string (OSC, DCS, SOS, PM or APC, 7-bit or 8-bit) up to its terminator:
            # ST, or BEL as xterm accepts; else up to the next escape or the end, as a terminal
            # reads it.
            r"(?:\x1b[\]PX^_]|[\x90\x98\x9d\x9e\x9f])[^\x07\x1b\x9c]*(?:\x07|\x1b\\|\x9c)?",
            # A control sequence (CSI): parameter bytes, intermediate bytes and a final byte.
            r"(?:\x1b\[|\x9b)[0-?]*[ -/]*[@-~]",
            # An escape sequence: intermediate bytes (a character set) and a final byte, or a
            # final byte alone (keypad modes, cursor save and restore, reset).
            r"\x1b[ -/]*[0-~]",
            # An escape the text ends on, or a string terminator with no string.
            r"\x1b|\x9c",
        )
    )
)


def strip_terminal_controls(text: str) -> str:
    """``text`` without its terminal control sequences, as a reader of a page should see it.

    Colours and cursor movement, hyperlinks (OSC 8) and window titles, device control strings and
    character-set selection all go; a hyperlink keeps its visible text. Tabs, newlines and carriage
    returns are text and stay. The scan is linear in the text's length.

    Args:
        text: Text as a terminal program wrote it.

    Returns:
        The text with every control sequence removed.
    """
    return _CONTROL.sub("", text)
