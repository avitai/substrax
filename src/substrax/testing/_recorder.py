"""The ``sys.monitoring`` callbacks behind :func:`substrax.testing.counted_calls`.

coverage.py's tracer does not see code that runs inside a ``sys.monitoring`` callback, so the
callbacks live in their own class, which the tests also drive directly.
"""

from __future__ import annotations

import sys
import threading
from collections import Counter
from collections.abc import Callable, Mapping
from types import CodeType


type CodeKeys = Callable[[CodeType], tuple[str, ...]]
"""The keys one start of a code object adds one to; an empty tuple leaves it out."""

type CallWeight = Callable[[object, object], int]
"""How much one call adds, from the callable and the first argument the call passes.

The first argument is the first value the call site passes, positional or keyword, in the order
written (``f(y=1, x=2)`` passes 1 first), and ``sys.monitoring.MISSING`` when it passes none.
"""


class CallRecorder:
    """The ``sys.monitoring`` callbacks of one counted block and the counts they add to."""

    def __init__(self, keys: CodeKeys, weights: Mapping[str, CallWeight]) -> None:
        """Start with no counts.

        Args:
            keys: The keys of each code object's starts.
            weights: Per key, how much each call adds.
        """
        self.counts: Counter[str] = Counter()
        self.failures: list[Exception] = []
        self._keys = keys
        self._weights = dict(weights)
        self._lock = threading.Lock()  # callbacks run on several threads at once without a GIL
        # Keyed by identity: code objects compare equal across files when their bytecode and
        # names match, and holding each one keeps its id from being reused inside the block.
        self._known: dict[int, tuple[CodeType, tuple[str, ...]]] = {}

    @property
    def events(self) -> int:
        """The events the block needs: function starts, and calls when there are weights."""
        events = sys.monitoring.events
        return events.PY_START | (events.CALL if self._weights else events.NO_EVENTS)

    def started(self, code: CodeType, offset: int) -> None:
        """Add one to each key of ``code`` (the ``PY_START`` callback).

        Args:
            code: The code object that started.
            offset: The instruction offset (unused).

        Raises:
            Exception: Whatever the key function raised, at the call it was counting; nothing is
                counted after it.
        """
        del offset
        entry = self._known.get(id(code))
        if entry is None:
            if self.failures:
                return
            try:
                entry = self._known[id(code)] = (code, self._keys(code))
            except Exception as error:  # re-raised at the call; recorded to stop counting
                self.failures.append(error)
                raise
        self._add([(key, 1) for key in entry[1]])

    def called(self, code: CodeType, offset: int, callee: object, first: object) -> None:
        """Add each weight's amount for one call (the ``CALL`` callback).

        Args:
            code: The code object making the call (unused).
            offset: The instruction offset (unused).
            callee: The called object.
            first: The first argument the call passes, or ``sys.monitoring.MISSING``.

        Raises:
            Exception: Whatever a weight raised, at the call it was counting; nothing is counted
                after it.
        """
        del code, offset
        if self.failures:
            return
        try:
            added = [
                (key, n) for key, weigh in self._weights.items() if (n := weigh(callee, first))
            ]
        except Exception as error:  # re-raised at the call; recorded to stop counting
            self.failures.append(error)
            raise
        self._add(added)

    def _add(self, amounts: list[tuple[str, int]]) -> None:
        if amounts:
            with self._lock:
                for key, amount in amounts:
                    self.counts[key] += amount
