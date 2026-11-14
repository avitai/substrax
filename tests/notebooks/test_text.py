"""``substrax.notebooks.text``: the one way the example tools read a script, notebook or page."""

from __future__ import annotations

from pathlib import Path
from typing import Any, IO

import pytest

from substrax.notebooks.text import NotUtf8Error, read_utf8, unix_line_endings, write_utf8


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


def test_a_written_file_holds_the_text_s_own_bytes(tmp_path: Path) -> None:
    path = tmp_path / "example.ipynb"

    write_utf8(path, "{\n  'é': 1\n}\n")

    assert path.read_bytes() == "{\n  'é': 1\n}\n".encode()


def test_a_file_is_written_with_unix_line_endings_on_every_platform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows' text mode writes ``\\r\\n`` unless told otherwise, which no Linux run can show, so
    the writer is held to opening its file without translation (W9)."""
    opened: list[tuple[str | None, str | None]] = []
    real_open = Path.open

    def recording_open(
        self: Path,
        mode: str = "r",
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> IO[Any]:
        opened.append((newline, encoding))
        return real_open(self, mode, buffering, encoding, errors, newline)

    monkeypatch.setattr(Path, "open", recording_open)

    write_utf8(tmp_path / "page.md", "a\n")

    assert opened == [("\n", "utf-8")]
