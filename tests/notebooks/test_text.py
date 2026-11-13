"""``substrax.notebooks.text``: the one way the example tools read a script, notebook or page."""

from __future__ import annotations

from pathlib import Path

import pytest

from substrax.notebooks.text import NotUtf8Error, read_utf8, unix_line_endings


def _file(tmp_path: Path, data: bytes) -> Path:
    path = tmp_path / "example.py"
    path.write_bytes(data)
    return path


def test_a_utf8_file_reads_as_its_text(tmp_path: Path) -> None:
    assert read_utf8(_file(tmp_path, "x = 'é'\n".encode())) == "x = 'é'\n"


def test_a_byte_order_mark_is_dropped(tmp_path: Path) -> None:
    assert read_utf8(_file(tmp_path, b"\xef\xbb\xbfx = 1\n")) == "x = 1\n"


@pytest.mark.parametrize("ending", [b"\r\n", b"\r"], ids=["windows", "carriage-return"])
def test_every_line_ending_reads_as_a_newline(tmp_path: Path, ending: bytes) -> None:
    """What git's ``core.autocrlf`` checks out, or an old editor saves, is the same text (BR-15)."""
    assert read_utf8(_file(tmp_path, b"a = 1" + ending + b"b = 2" + ending)) == "a = 1\nb = 2\n"


def test_a_file_that_is_not_utf8_is_refused_naming_the_file_and_the_byte(tmp_path: Path) -> None:
    path = _file(tmp_path, b"# caf\xe9\n")

    with pytest.raises(NotUtf8Error) as raised:
        read_utf8(path)

    assert str(raised.value) == f"{path} is not UTF-8: byte 5 cannot be decoded; save it as UTF-8"
    assert isinstance(raised.value, ValueError)


def test_unix_line_endings_leave_newlines_and_other_text_alone() -> None:
    assert unix_line_endings("a\r\nb\rc\nd\r\r\n") == "a\nb\nc\nd\n\n"
