"""Tests for removing terminal control sequences from text a page or an error message shows."""

from __future__ import annotations

import time

import pytest

from substrax.notebooks.terminal import strip_terminal_controls


ESC = "\x1b"
ST = ESC + "\\"
BEL = "\x07"


@pytest.mark.parametrize(
    ("text", "shown"),
    [
        pytest.param(f"{ESC}[1;31mred{ESC}[0m", "red", id="csi-colour"),
        pytest.param(f"{ESC}[2K{ESC}[1Gdone", "done", id="csi-erase-and-move"),
        pytest.param(f"{ESC}[?25lhidden cursor{ESC}[?25h", "hidden cursor", id="csi-private"),
        pytest.param(f"{ESC}]8;;https://example.org{ST}docs{ESC}]8;;{ST}", "docs", id="osc8-st"),
        pytest.param(f"{ESC}]8;;https://example.org{BEL}docs{ESC}]8;;{BEL}", "docs", id="osc8-bel"),
        pytest.param(f"{ESC}]0;window title{BEL}text", "text", id="osc-title"),
        pytest.param(f"{ESC}Pq#0;2;0;0;0{ST}after", "after", id="dcs"),
        pytest.param(f"{ESC}_application{ST}after", "after", id="apc"),
        pytest.param(f"{ESC}(Bplain", "plain", id="nf-charset"),
        pytest.param(f"{ESC}=keypad{ESC}>", "keypad", id="fp-keypad"),
        pytest.param(f"{ESC}7saved{ESC}8", "saved", id="fp-cursor-save"),
        pytest.param("\x9b31mred\x9b0m", "red", id="c1-csi"),
        pytest.param("\x9d8;;https://example.org\x9cdocs\x9d8;;\x9c", "docs", id="c1-osc"),
        pytest.param(f"before{ESC}]8;;https://example.org", "before", id="osc-unterminated"),
        pytest.param(f"a{ESC}", "a", id="lone-escape"),
    ],
)
def test_every_kind_of_terminal_control_is_removed(text: str, shown: str) -> None:
    assert strip_terminal_controls(text) == shown


def test_text_without_controls_is_unchanged() -> None:
    text = "tabs\tand\nnewlines\r\nand a carriage return\rstay, as do é and 日本"

    assert strip_terminal_controls(text) == text


def test_a_string_command_ends_at_the_next_escape_sequence() -> None:
    text = f"{ESC}]8;;https://example.org{ESC}[31mred{ESC}[0m"

    assert strip_terminal_controls(text) == "red"


def test_long_text_with_many_unterminated_commands_is_linear() -> None:
    text = f"{ESC}]" * 200_000 + "x" * 2_000_000

    start = time.perf_counter()
    stripped = strip_terminal_controls(text)
    elapsed = time.perf_counter() - start

    assert stripped == ""
    # The bound separates linear from quadratic, not one machine's speed: 200,000 commands over
    # 2,000,000 characters scanned once each is milliseconds, rescanned per command hours. The same
    # bound holds the stream merge's 2,000,000-character line (test_run.py).
    assert elapsed < 20.0
