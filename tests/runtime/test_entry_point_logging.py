"""``configure_entry_point_logging`` sets up the root logger once, without ``force=True``."""

from __future__ import annotations

import io
import logging
from collections.abc import Iterator

import pytest

from substrax.runtime import configure_entry_point_logging


@pytest.fixture
def root_logger() -> Iterator[logging.Logger]:
    """The root logger, with its handlers and level restored after the test."""
    root = logging.getLogger()
    handlers = list(root.handlers)
    level = root.level
    yield root
    for handler in list(root.handlers):
        if handler not in handlers:
            root.removeHandler(handler)
    root.setLevel(level)


def test_messages_reach_the_given_stream(root_logger: logging.Logger) -> None:
    stream = io.StringIO()

    configure_entry_point_logging(stream=stream)
    logging.getLogger("an.entry.point").info("hello")

    assert stream.getvalue() == "hello\n"
    assert root_logger.level == logging.INFO


def test_a_second_call_replaces_only_its_own_handler(root_logger: logging.Logger) -> None:
    before = len(root_logger.handlers)
    first, second = io.StringIO(), io.StringIO()

    configure_entry_point_logging(stream=first)
    configure_entry_point_logging(stream=second)
    logging.getLogger("an.entry.point").info("hello")

    assert len(root_logger.handlers) == before + 1
    assert (first.getvalue(), second.getvalue()) == ("", "hello\n")


def test_a_foreign_handler_survives(root_logger: logging.Logger) -> None:
    foreign = logging.StreamHandler(io.StringIO())
    root_logger.addHandler(foreign)

    configure_entry_point_logging(stream=io.StringIO())
    configure_entry_point_logging(stream=io.StringIO())

    assert foreign in root_logger.handlers


def test_the_level_applies_to_the_root_logger(root_logger: logging.Logger) -> None:
    stream = io.StringIO()

    configure_entry_point_logging(logging.WARNING, stream=stream)
    logging.getLogger("an.entry.point").info("dropped")
    logging.getLogger("an.entry.point").warning("kept")

    assert root_logger.level == logging.WARNING
    assert stream.getvalue() == "kept\n"


def test_the_format_is_applied(root_logger: logging.Logger) -> None:
    stream = io.StringIO()

    configure_entry_point_logging(stream=stream, fmt="%(levelname)s:%(name)s:%(message)s")
    logging.getLogger("an.entry.point").info("hello")

    assert stream.getvalue() == "INFO:an.entry.point:hello\n"
    assert root_logger.handlers
