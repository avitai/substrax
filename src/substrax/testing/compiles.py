"""The XLA executables jax builds inside a block, for tests that a repeated call compiles nothing.

jax records the event ``/jax/core/compile/backend_compile_duration`` around
``compile_or_get_cached``, which runs only when its in-process executable cache misses, and passes
the program's name, ``jit(<function>)``, as ``fun_name`` (``jax/_src/interpreters/pxla.py`` and
``jax/_src/dispatch.py``, jax 0.11.1). Eager operations are compiled programs too. A block therefore sees every executable built for it,
whether XLA compiled it or the persistent compilation cache supplied it. A trace is a different
event: :class:`~substrax.testing.TraceCounter` counts those.

The listener is registered with ``jax.monitoring`` for the block and removed after it. Listeners
are process-wide, so a compile on another thread during the block is seen too.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

import jax


BACKEND_COMPILE_EVENT = "/jax/core/compile/backend_compile_duration"
"""The ``jax.monitoring`` duration event jax records for each executable it builds."""


class CompileCountError(AssertionError):
    """A block built a different number of executables than the test expected.

    Attributes:
        expected: The number of compiles the test expected.
        names: The names of the functions compiled in the block, in order.
    """

    expected: int
    names: tuple[str, ...]

    def __init__(self, expected: int, names: tuple[str, ...]) -> None:
        """Record the expectation and what was compiled, and name both in the message.

        Args:
            expected: The number of compiles the test expected.
            names: The names of the functions compiled in the block.
        """
        super().__init__(
            f"expected {expected} compile(s) in the block, observed {len(names)}: {list(names)}"
        )
        self.expected = expected
        self.names = names


@contextmanager
def compiled_programs() -> Generator[list[str]]:
    """Collect the names of the functions jax compiles inside the block.

    Yields:
        list[str]: The name of each program compiled in the block, in order; the list stops
        growing when the block ends, also when the block raises.
    """
    names: list[str] = []

    def record(event: str, duration_secs: float, **kwargs: str | int) -> None:
        del duration_secs
        if event == BACKEND_COMPILE_EVENT:
            names.append(str(kwargs.get("fun_name", "")))

    jax.monitoring.register_event_duration_secs_listener(record)
    try:
        yield names
    finally:
        jax.monitoring.unregister_event_duration_listener(record)


@contextmanager
def expect_compiles(count: int) -> Generator[None]:
    """Require exactly ``count`` compiles inside the block.

    An exception raised inside the block propagates unchanged, without a count check.

    Args:
        count: How many executables the block must build.

    Yields:
        None: Control to the block.

    Raises:
        ValueError: If ``count`` is negative.
        CompileCountError: If the block built a different number of executables.
    """
    if count < 0:
        msg = f"count must be at least 0, got {count}"
        raise ValueError(msg)
    with compiled_programs() as names:
        yield
    if len(names) != count:
        raise CompileCountError(count, tuple(names))
