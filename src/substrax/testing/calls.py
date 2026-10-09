"""Count the Python functions a block starts, on every thread, for tests that assert work by count.

A test that bounds how much work a code path does by timing it measures the machine as much as the
code. The number of Python functions the path starts, and the calls it makes to chosen callables,
are the same on every machine. :func:`counted_calls` counts them with ``sys.monitoring``
(PEP 669, Python 3.12+), whose events are process-wide, so work a block hands to another thread is
counted too. :func:`by_package` groups the starts by package and, for chosen packages, by qualified
name. :func:`per_iteration` turns two pass lengths into an exact per-iteration count, so work done
once per pass cancels.

The counts are deterministic for deterministic code:

- Automatic garbage collection is off inside the block, and one collection runs at its end, inside
  the count. The collector runs by allocation count, so finalizers and weak-reference callbacks
  would otherwise run at a point that depends on the allocator's history.
- ``by_package`` leaves out ``threading`` and ``queue`` by default. How often a thread waits on a
  condition, and so how often those modules' functions run, depends on thread timing.
- A ``PY_START`` event fires when a Python function or generator starts, not when a generator
  resumes, and C functions fire none, so neither is counted.

Code whose calls depend on object addresses is not deterministic, and no counter makes it so: an
``__eq__`` of a key hashed by ``id()`` runs only when a new object reuses a dead one's address.
:func:`per_iteration` names such a count; a keys function that wraps :func:`by_package` can leave
the code object out.

On a free-threaded build, callbacks run on several threads at once; a lock guards the counts, so
they stay exact. The standard library's ``cProfile`` uses ``sys.monitoring`` too, but documents
one thread: across threads on 3.12 its call stack interleaves and calls are attributed as
recursive.
"""

from __future__ import annotations

import gc
import importlib.util
import sys
from collections import Counter
from collections.abc import Callable, Collection, Generator, Mapping
from contextlib import contextmanager
from pathlib import Path
from types import CodeType

from substrax.testing._recorder import CallRecorder, CallWeight, CodeKeys


TOOL_ID_ORDER = (3, 4, 5, 2, 0, 1)
"""The ``sys.monitoring`` tool ids :func:`counted_calls` tries, in order.

CPython assigns 0 to debuggers, 1 to coverage (coverage.py's ``sysmon`` core uses it), 2 to
profilers (``cProfile`` uses it) and 5 to optimizers; 3 and 4 are unassigned, so they come first.
"""

DEFAULT_EXCLUDED = ("threading", "queue")
"""Modules :func:`by_package` leaves out: how often their functions run depends on thread timing."""

_TOOL_NAME = "substrax.testing.calls"


class ToolIdsInUseError(RuntimeError):
    """Every ``sys.monitoring`` tool id is held by another tool.

    Attributes:
        owners: The name of the tool holding each id.
    """

    owners: dict[int, str]

    def __init__(self, owners: dict[int, str]) -> None:
        """Record and name the holder of each id.

        Args:
            owners: The name of the tool holding each id.
        """
        super().__init__(f"every sys.monitoring tool id is in use: {owners}")
        self.owners = owners


class CountingFailedError(RuntimeError):
    """A key or weight function raised inside the block, so the counts are incomplete.

    The function's exception propagates from the call it was counting; this error, chained to
    it, ends a block whose code caught that exception.
    """

    def __init__(self) -> None:
        """Name the incomplete counts in the message."""
        super().__init__(
            "a key or weight function raised inside the block; the counts are incomplete"
        )


class UnevenCountError(AssertionError):
    """A count did not grow by a whole number per iteration between two passes.

    Attributes:
        deltas: Each such key's count in the long pass minus its count in the short pass.
        iterations: How many more iterations the long pass ran.
    """

    deltas: dict[str, int]
    iterations: int

    def __init__(self, deltas: dict[str, int], iterations: int) -> None:
        """Record the uneven deltas and name them in the message.

        Args:
            deltas: Each uneven key's count in the long pass minus the short pass's.
            iterations: How many more iterations the long pass ran.
        """
        super().__init__(f"counts not a whole number per iteration over {iterations}: {deltas}")
        self.deltas = deltas
        self.iterations = iterations


def _locations(module: str) -> tuple[str, ...]:
    """Return the directories of a package, or the file of a module, as path prefixes.

    Args:
        module: An importable module or package name.

    Returns:
        Each directory of the package with a trailing separator, or the module's file.

    Raises:
        ValueError: If ``module`` cannot be found or has no file.
    """
    spec = importlib.util.find_spec(module)
    if spec is None:
        raise ValueError(f"no module or package named {module!r} can be found")
    if spec.submodule_search_locations:
        return tuple(str(Path(location)) + "/" for location in spec.submodule_search_locations)
    if spec.origin is None or not spec.has_location:
        raise ValueError(f"{module!r} has no file to match code against")
    return (spec.origin,)


def by_package(
    *packages: str,
    named: Collection[str] = (),
    excluded: Collection[str] = DEFAULT_EXCLUDED,
    other: str = "other",
) -> CodeKeys:
    """Group function starts by the package whose files hold the code.

    A start in a file under one of ``packages`` adds one to that package's name; one in a package
    of ``named`` also adds one to ``"<package>:<qualified name>"``. A start in a file of an
    ``excluded`` module or package adds nothing, and any other start adds one to ``other``.
    Packages are matched by the directories ``importlib`` finds for them, the longest first, so a
    subpackage named alongside its parent takes its own files.

    Args:
        *packages: Importable names of the packages to count apart.
        named: Packages among ``packages`` whose starts are also counted by qualified name.
        excluded: Modules or packages whose starts are not counted.
        other: The key of a start in no listed package.

    Returns:
        The keys of each code object, for :func:`counted_calls`.

    Raises:
        ValueError: If a package or excluded module cannot be found, or a ``named`` package is not
            one of ``packages``.
    """
    unknown = set(named) - set(packages)
    if unknown:
        raise ValueError(f"named packages {sorted(unknown)} are not among {list(packages)}")
    roots = sorted(
        ((location, package) for package in packages for location in _locations(package)),
        key=lambda root: len(root[0]),
        reverse=True,
    )
    skipped = tuple(location for module in excluded for location in _locations(module))
    by_name = frozenset(named)

    def keys(code: CodeType) -> tuple[str, ...]:
        filename = code.co_filename
        if filename.startswith(skipped):
            return ()
        for location, package in roots:
            if filename.startswith(location):
                if package in by_name:
                    return (package, f"{package}:{code.co_qualname}")
                return (package,)
        return (other,)

    return keys


def _claim_tool_id() -> int:
    """Take the first free tool id of :data:`TOOL_ID_ORDER`.

    Returns:
        The id taken, under this module's tool name.

    Raises:
        ToolIdsInUseError: If every id is held.
    """
    monitoring = sys.monitoring
    for tool in TOOL_ID_ORDER:
        if monitoring.get_tool(tool) is None:
            try:
                monitoring.use_tool_id(tool, _TOOL_NAME)
            except ValueError:  # taken between the check and the claim
                continue
            return tool
    raise ToolIdsInUseError({tool: str(monitoring.get_tool(tool)) for tool in TOOL_ID_ORDER})


@contextmanager
def _collector_paused() -> Generator[None]:
    """Collect once, keep automatic collection off for the block, then restore its state."""
    collecting = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        yield
    finally:
        if collecting:
            gc.enable()


@contextmanager
def _monitored(recorder: CallRecorder) -> Generator[None]:
    """Deliver the block's events to ``recorder`` under a tool id freed when the block ends."""
    monitoring = sys.monitoring
    tool = _claim_tool_id()
    try:
        monitoring.register_callback(tool, monitoring.events.PY_START, recorder.started)
        monitoring.register_callback(tool, monitoring.events.CALL, recorder.called)
        monitoring.set_events(tool, recorder.events)
        try:
            yield
        finally:
            monitoring.set_events(tool, monitoring.events.NO_EVENTS)
    finally:
        monitoring.register_callback(tool, monitoring.events.PY_START, None)
        monitoring.register_callback(tool, monitoring.events.CALL, None)
        monitoring.free_tool_id(tool)


@contextmanager
def counted_calls(  # noqa: DOC503  # ToolIdsInUseError is raised by _claim_tool_id
    keys: CodeKeys, *, weights: Mapping[str, CallWeight] | None = None
) -> Generator[Counter[str]]:
    """Count the Python functions started, and weighted calls made, inside the block.

    Every thread's work is counted. ``keys`` decides which counts each function start adds one to
    (it runs once per code object). Each of ``weights`` sees every call made from Python code, with
    the callable and the first argument it passes, and adds what it returns to its own key.
    Events do not fire while a callback runs, so the weights' own calls are not counted.

    Automatic garbage collection is off inside the block; one collection runs at its end, still
    counted, and the collector's previous state is restored after it.

    Args:
        keys: The keys of each code object's starts, usually from :func:`by_package`.
        weights: Per key, how much each call adds.

    Yields:
        Counter[str]: The counts, which stop growing when the block ends, also when it raises.

    Raises:
        ToolIdsInUseError: If every ``sys.monitoring`` tool id is held by another tool.
        CountingFailedError: If ``keys`` or a weight raised inside the block and the block's code
            caught the exception. Once one has raised, nothing more is counted.
    """
    recorder = CallRecorder(keys, weights or {})
    with _collector_paused(), _monitored(recorder):
        try:
            yield recorder.counts
            if recorder.failures:
                raise CountingFailedError from recorder.failures[0]
        finally:
            gc.collect()  # counted: the block's garbage is freed at a fixed point


def per_iteration(
    count: Callable[[int], Mapping[str, int]], *, short: int, long: int
) -> dict[str, int]:
    """Return the work one iteration adds: the slope between a short and a long pass.

    ``count(n)`` runs a pass of ``n`` iterations and returns its counts. Work done once per pass
    (setup, a thread's start and close) is the same in both passes and cancels. Each count must
    grow by a whole number per iteration, so a count that varies between passes fails instead
    of averaging.

    Args:
        count: Runs a pass of the given number of iterations and returns its counts.
        short: Iterations of the first pass, at least 0.
        long: Iterations of the second pass, more than ``short``.

    Returns:
        Each key's count added per iteration; keys that add nothing are left out.

    Raises:
        ValueError: If ``short`` is negative or ``long`` is not more than ``short``.
        UnevenCountError: If a count does not grow by a whole number per iteration.
    """
    if short < 0 or long <= short:
        raise ValueError(f"need 0 <= short < long, got short={short}, long={long}")
    first, second = count(short), count(long)
    span = long - short
    deltas = {key: second.get(key, 0) - first.get(key, 0) for key in {*first, *second}}
    uneven = {key: delta for key, delta in deltas.items() if delta % span}
    if uneven:
        raise UnevenCountError(uneven, span)
    return {key: delta // span for key, delta in deltas.items() if delta}
