"""Count the traces a jitted function takes, and fail a block that caused an unexpected number."""

from __future__ import annotations

import functools
from collections.abc import Callable, Generator
from contextlib import contextmanager


class RetraceError(AssertionError):
    """A block caused a different number of traces than the test expected.

    Attributes:
        expected: The number of new traces the test expected.
        observed: The number of new traces the block caused.
    """

    expected: int
    observed: int

    def __init__(self, expected: int, observed: int) -> None:
        """Record both counts and name them in the message.

        Args:
            expected: The number of new traces the test expected.
            observed: The number of new traces the block caused.
        """
        super().__init__(f"expected {expected} new trace(s) in the block, observed {observed}")
        self.expected = expected
        self.observed = observed


class TraceCounter:
    """Count how often wrapped functions run their Python body.

    ``jax.jit`` runs a function's body once per trace and reuses the cached trace for later calls
    with the same static signature, and ``nnx.jit`` calls the user function once inside the
    function jax traces. The count of body executions is therefore the count of traces. An eager
    call also runs the body and also counts.

    Each counter keeps its own count, so tests share no state to clear.
    """

    def __init__(self) -> None:
        """Start at zero."""
        self._count = 0

    @property
    def count(self) -> int:
        """The number of body executions of every function this counter wrapped."""
        return self._count

    def wrap[**P, R](self, function: Callable[P, R]) -> Callable[P, R]:
        """Return ``function`` counting each run of its body; jit the result, not the original.

        The wrapper keeps the function's name, docstring and signature, so ``static_argnames``
        and other signature-based arguments of a transform still resolve.

        Args:
            function: The Python function to count.

        Returns:
            A function that adds one to the count, then calls ``function``. The count rises
            before the call, so a trace that raises still counts.
        """

        @functools.wraps(function)
        def counted(*args: P.args, **kwargs: P.kwargs) -> R:
            self._count += 1
            return function(*args, **kwargs)

        return counted

    @contextmanager
    def expect(self, *, new_traces: int) -> Generator[None, None, None]:
        """Require exactly ``new_traces`` body executions inside the block.

        An exception raised inside the block propagates unchanged, without a count check.

        Args:
            new_traces: How many new traces the block must cause.

        Yields:
            None: Control to the block.

        Raises:
            ValueError: If ``new_traces`` is negative.
            RetraceError: If the block caused a different number of traces.
        """
        if new_traces < 0:
            raise ValueError(f"new_traces must be at least 0, got {new_traces}")
        before = self._count
        yield
        observed = self._count - before
        if observed != new_traces:
            raise RetraceError(new_traces, observed)
