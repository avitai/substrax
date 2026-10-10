"""The ``sys.monitoring`` callbacks behind :func:`substrax.testing.counted_calls`.

coverage.py's tracer does not see code that runs inside a ``sys.monitoring`` callback, so the
callbacks live in their own class, which the tests also drive directly.
"""

from __future__ import annotations

import inspect
import sys
import threading
from collections import Counter
from collections.abc import Callable, Mapping
from types import CodeType, FrameType


type CodeKeys = Callable[[CodeType], tuple[str, ...]]
"""The keys one start of a code object adds one to; an empty tuple leaves it out."""

type CallWeight = Callable[[object, object], int]
"""How much one call adds, from the callable and the first argument the call passes.

The first argument is the first value the call site passes, positional or keyword, in the order
written (``f(y=1, x=2)`` passes 1 first), and ``sys.monitoring.MISSING`` when it passes none.
"""


def _outermost(frame: FrameType) -> FrameType:
    """The frame a thread's Python execution started in, below ``frame``."""
    while frame.f_back is not None:
        frame = frame.f_back
    return frame


def _current_outermost() -> FrameType:
    """The frame the calling thread's Python execution started in.

    Returns:
        The calling thread's outermost frame.

    Raises:
        RuntimeError: If the interpreter gives no access to frames.
    """
    frame = inspect.currentframe()
    if frame is None:
        raise RuntimeError("counting by thread needs an interpreter that exposes stack frames")
    return _outermost(frame)


class BlockThreads:
    """The threads a counted block owns: the one that opened it and every thread started after.

    A thread already running Python when the block opens (a library's monitor or heartbeat, a
    pool an earlier test left) is not the block's work. A thread is told apart by its outermost
    frame, the frame its Python execution started in, which lives exactly as long as the thread;
    this holds the running threads' outermost frames until the block ends, so none is freed and
    replaced. Neither ``threading.get_ident()`` nor ``threading.current_thread()`` would do:
    CPython gives the id of a thread that ended to the next one it starts, and until a new
    thread registers itself ``current_thread()`` returns the placeholder an earlier thread on
    that id left (placeholders are never removed on 3.12), or creates one. The running threads'
    frames come from ``sys._current_frames()``, so a thread started without the ``threading``
    module is seen too.
    A thread running no Python when the block opens is not seen then, and counts as started after.
    Each thread's answer is kept in a thread-local, so it is worked out once per thread.
    """

    def __init__(self) -> None:
        """Take the opening thread and the threads running Python now."""
        self._opener = _current_outermost()
        # sys._current_frames is the sys module's documented way to read other threads' frames
        # (profilers such as Scalene read it for the same reason); no public name exposes them.
        running = sys._current_frames()  # pyright: ignore[reportPrivateUsage]
        self._running = frozenset(_outermost(frame) for frame in running.values())
        self._answer = threading.local()

    def has_current(self) -> bool:
        """Whether the thread calling this belongs to the block.

        Returns:
            ``True`` for the opening thread and for a thread that was not running Python when
            the block opened.
        """
        answer: bool | None = getattr(self._answer, "value", None)
        if answer is None:
            outermost = _current_outermost()
            answer = self._answer.value = (
                outermost is self._opener or outermost not in self._running
            )
        return answer


class CallRecorder:
    """The ``sys.monitoring`` callbacks of one counted block and the counts they add to."""

    def __init__(
        self,
        keys: CodeKeys,
        weights: Mapping[str, CallWeight],
        threads: BlockThreads | None = None,
    ) -> None:
        """Start with no counts.

        Args:
            keys: The keys of each code object's starts.
            weights: Per key, how much each call adds.
            threads: The threads whose starts and calls count; by default the thread creating
                the recorder and the threads started after it.
        """
        self.threads = BlockThreads() if threads is None else threads
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
        """Add one to each key of ``code`` on one of the block's threads (``PY_START``).

        Args:
            code: The code object that started.
            offset: The instruction offset (unused).

        Raises:
            Exception: Whatever the key function raised, at the call it was counting; nothing is
                counted after it.
        """
        del offset
        if not self.threads.has_current():
            return
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
        """Add each weight's amount for one call on one of the block's threads (``CALL``).

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
        if self.failures or not self.threads.has_current():
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
