"""``counted_calls`` counts the Python functions a block starts, on every thread, by key.

Each test imports a small package written to ``tmp_path``, so the counts it asserts belong to code
no other test, thread or plugin runs; work elsewhere in the process lands under other keys.
"""

from __future__ import annotations

import contextlib
import gc
import importlib
import json
import sys
import threading
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from types import CodeType, ModuleType

import pytest

from substrax.testing import (
    by_package,
    counted_calls,
    CountingFailedError,
    per_iteration,
    run_python,
    ToolIdsInUseError,
    UnevenCountError,
)
from substrax.testing._recorder import CallRecorder
from substrax.testing.calls import TOOL_ID_ORDER


_SOURCE = '''
def leaf():
    return 1


def branch(n):
    for _ in range(n):
        leaf()

    def inner():
        return leaf()

    return inner()


class Holder:
    def method(self):
        return leaf()


class Finalized:
    """A reference cycle whose finalizer runs when the collector frees it."""

    def __init__(self):
        self.self = self

    def __del__(self):
        leaf()
'''

_TOOL_IDS = range(6)


@pytest.fixture
def make_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Callable[[], ModuleType]]:
    """Write and import a fresh package with ``_SOURCE`` under a unique name."""
    monkeypatch.syspath_prepend(str(tmp_path))
    names: list[str] = []

    def make() -> ModuleType:
        name = f"work_{uuid.uuid4().hex}"
        (tmp_path / name).mkdir()
        (tmp_path / name / "__init__.py").write_text(_SOURCE)
        names.append(name)
        importlib.invalidate_caches()  # the path finder caches the directory's listing
        return importlib.import_module(name)

    yield make
    for name in names:
        sys.modules.pop(name, None)


def _owners() -> list[str | None]:
    return [sys.monitoring.get_tool(tool) for tool in _TOOL_IDS]


def test_each_started_function_is_counted_under_its_package_and_name(
    make_package: Callable[[], ModuleType],
) -> None:
    package = make_package()

    with counted_calls(by_package(package.__name__, named=[package.__name__])) as counts:
        package.branch(3)
        package.Holder().method()

    name = package.__name__
    assert counts[name] == 8  # branch, 3 leaf, inner, leaf, method, leaf
    assert counts[f"{name}:branch"] == 1
    assert counts[f"{name}:leaf"] == 5
    assert counts[f"{name}:branch.<locals>.inner"] == 1
    assert counts[f"{name}:Holder.method"] == 1


def test_one_extra_call_is_one_extra_count(make_package: Callable[[], ModuleType]) -> None:
    package = make_package()
    keys = by_package(package.__name__, named=[package.__name__])

    with counted_calls(keys) as without:
        package.branch(4)
    with counted_calls(keys) as with_extra:
        package.branch(4)
        package.leaf()

    assert with_extra[package.__name__] - without[package.__name__] == 1
    assert with_extra[f"{package.__name__}:leaf"] - without[f"{package.__name__}:leaf"] == 1


def test_packages_are_counted_apart_and_unnamed_packages_only_in_total(
    make_package: Callable[[], ModuleType],
) -> None:
    first, second = make_package(), make_package()

    with counted_calls(
        by_package(first.__name__, second.__name__, named=[first.__name__])
    ) as counts:
        first.branch(2)
        second.leaf()

    assert first.leaf.__code__ == second.leaf.__code__  # equal code, so keyed by identity
    assert counts[first.__name__] == 5
    assert counts[f"{first.__name__}:leaf"] == 3
    assert counts[second.__name__] == 1
    assert not any(key.startswith(f"{second.__name__}:") for key in counts)


def test_code_outside_the_packages_is_counted_as_other() -> None:
    def outside() -> None:
        pass

    with counted_calls(by_package("json", other="elsewhere")) as counts:
        outside()
        json.dumps([1])

    assert counts["elsewhere"] >= 1
    assert counts["json"] >= 1


def test_threading_and_queue_are_left_out_by_default() -> None:
    event = threading.Event()

    with counted_calls(by_package("threading", other="other")) as default:
        event.is_set()
        threading.current_thread()
    with counted_calls(by_package("json", excluded=())) as kept:
        threading.current_thread()

    assert "threading" not in default
    assert kept["other"] >= 1


def test_every_thread_is_counted_and_repeats_are_identical(
    make_package: Callable[[], ModuleType],
) -> None:
    package = make_package()
    keys = by_package(package.__name__, named=[package.__name__])

    def run() -> dict[str, int]:
        with counted_calls(keys) as counts:
            workers = [threading.Thread(target=package.branch, args=(500,)) for _ in range(4)]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join()
        return dict(counts)

    runs = [run() for _ in range(5)]

    assert runs[0][f"{package.__name__}:leaf"] == 4 * 501
    assert all(counts == runs[0] for counts in runs)


def test_the_collector_is_off_inside_and_frees_the_blocks_garbage_before_it_ends(
    make_package: Callable[[], ModuleType],
) -> None:
    package = make_package()
    was_enabled = gc.isenabled()

    with counted_calls(by_package(package.__name__, named=[package.__name__])) as counts:
        assert not gc.isenabled()
        for _ in range(50):
            package.Finalized()

    assert gc.isenabled() == was_enabled
    assert counts[f"{package.__name__}:Finalized.__del__"] == 50


def test_a_disabled_collector_stays_disabled() -> None:
    gc.disable()
    try:
        with counted_calls(by_package("json")):
            pass
        assert not gc.isenabled()
    finally:
        gc.enable()


def test_nothing_is_counted_after_the_block(make_package: Callable[[], ModuleType]) -> None:
    package = make_package()
    with counted_calls(by_package(package.__name__)) as counts:
        package.leaf()
    package.branch(10)

    assert counts[package.__name__] == 1


def test_the_tool_id_is_released_when_the_block_ends_and_when_it_raises() -> None:
    before = _owners()

    with counted_calls(by_package("json")):
        assert _owners() != before
    assert _owners() == before

    with pytest.raises(RuntimeError, match="inside"), counted_calls(by_package("json")):
        raise RuntimeError("inside the block")
    assert _owners() == before


def test_ids_other_tools_hold_are_left_to_them(make_package: Callable[[], ModuleType]) -> None:
    package = make_package()
    taken = [tool for tool in (3, 4) if sys.monitoring.get_tool(tool) is None]
    for tool in taken:
        sys.monitoring.use_tool_id(tool, "another tool")
    try:
        with counted_calls(by_package(package.__name__)) as counts:
            package.leaf()
        assert [sys.monitoring.get_tool(tool) for tool in taken] == ["another tool"] * len(taken)
    finally:
        for tool in taken:
            sys.monitoring.free_tool_id(tool)

    assert counts[package.__name__] == 1


def test_no_free_tool_id_is_refused_naming_the_holders() -> None:
    free = [tool for tool in _TOOL_IDS if sys.monitoring.get_tool(tool) is None]
    for tool in free:
        sys.monitoring.use_tool_id(tool, f"holder {tool}")
    try:
        with (
            pytest.raises(ToolIdsInUseError, match="holder") as refused,
            counted_calls(by_package("json")),
        ):
            pass
    finally:
        for tool in free:
            sys.monitoring.free_tool_id(tool)

    assert set(refused.value.owners) == set(_TOOL_IDS)


def test_weights_add_what_they_return_for_each_call_site() -> None:
    def target(*args: object) -> None:
        del args

    def first_length(callee: object, first: object) -> int:
        return len(first) if callee is target and isinstance(first, list) else 0

    def called_without_arguments(callee: object, first: object) -> int:
        return int(callee is target and first is sys.monitoring.MISSING)

    weights = {"items": first_length, "bare": called_without_arguments}
    with counted_calls(by_package("json"), weights=weights) as counts:
        target([1, 2, 3])
        target([4])
        target()
        target("not a list")

    assert counts["items"] == 4
    assert counts["bare"] == 1


def test_a_weight_that_raises_propagates_and_releases_the_tool_id() -> None:
    before = _owners()

    def broken(callee: object, first: object) -> int:
        del callee, first
        raise ZeroDivisionError

    with (
        pytest.raises(ZeroDivisionError),
        counted_calls(by_package("json"), weights={"broken": broken}),
    ):
        len([])

    assert _owners() == before


def test_a_weight_failure_the_block_caught_still_fails_the_block() -> None:
    before = _owners()

    def broken(callee: object, first: object) -> int:
        del first
        if callee is len:
            raise ZeroDivisionError
        return 0

    def caught() -> None:
        with contextlib.suppress(ZeroDivisionError):
            len([])

    with (
        pytest.raises(CountingFailedError) as refused,
        counted_calls(by_package("json"), weights={"broken": broken}),
    ):
        caught()

    assert isinstance(refused.value.__cause__, ZeroDivisionError)
    assert _owners() == before


def test_a_package_that_cannot_be_found_is_refused() -> None:
    with pytest.raises(ValueError, match="no_such_package_here"):
        by_package("no_such_package_here")


def test_a_named_package_must_be_one_of_the_packages() -> None:
    with pytest.raises(ValueError, match="named"):
        by_package("json", named=["email"])


def test_per_iteration_is_the_exact_slope_of_two_passes() -> None:
    def count(iterations: int) -> dict[str, int]:
        return {"setup": 7, "step": 3 * iterations, "flat": 2}

    assert per_iteration(count, short=10, long=30) == {"step": 3}


def test_per_iteration_refuses_a_count_that_does_not_divide() -> None:
    def count(iterations: int) -> dict[str, int]:
        return {"step": iterations, "drift": 5 if iterations == 30 else 0}

    with pytest.raises(UnevenCountError, match="drift") as refused:
        per_iteration(count, short=10, long=30)

    assert refused.value.deltas == {"drift": 5}
    assert refused.value.iterations == 20


@pytest.mark.parametrize(("short", "long"), [(-1, 3), (5, 5), (6, 2)])
def test_per_iteration_needs_a_longer_second_pass(short: int, long: int) -> None:
    with pytest.raises(ValueError, match="short"):
        per_iteration(lambda iterations: {"step": iterations}, short=short, long=long)


def test_counts_agree_while_coverage_measures_through_sys_monitoring(tmp_path: Path) -> None:
    (tmp_path / "pkg_under_cover").mkdir()
    (tmp_path / "pkg_under_cover" / "__init__.py").write_text(_SOURCE)
    program = f"""
import json, sys
sys.path.insert(0, {str(tmp_path)!r})
import coverage
# coverage measures through sys.monitoring only without branch measurement before 3.14, and this
# repository's configuration measures branches, so the child configures its own instance.
cov = coverage.Coverage(data_file=None, config_file=False, branch=False)
cov.start()
from substrax.testing import by_package, counted_calls
import pkg_under_cover
with counted_calls(by_package("pkg_under_cover", named=["pkg_under_cover"])) as counts:
    pkg_under_cover.branch(3)
holder = sys.monitoring.get_tool(sys.monitoring.COVERAGE_ID)
cov.stop()
print(json.dumps({{"holder": holder, "leaf": counts["pkg_under_cover:leaf"]}}))
"""

    report = run_python(program, timeout=120, env={"COVERAGE_CORE": "sysmon"}).check().last_json()

    assert report == {"holder": "coverage.py", "leaf": 4}


def test_the_keys_of_a_code_object_name_its_package_and_qualified_name(
    make_package: Callable[[], ModuleType],
) -> None:
    package, plain = make_package(), make_package()
    keys = by_package(package.__name__, plain.__name__, named=[package.__name__], other="rest")

    assert keys(package.Holder.method.__code__) == (
        package.__name__,
        f"{package.__name__}:Holder.method",
    )
    assert keys(plain.leaf.__code__) == (plain.__name__,)
    assert keys(_owners.__code__) == ("rest",)
    assert keys(threading.Thread.start.__code__) == ()


def test_a_module_without_a_file_is_refused() -> None:
    with pytest.raises(ValueError, match="no file"):
        by_package("json", excluded=["sys"])


def test_an_id_taken_between_the_check_and_the_claim_is_skipped(
    monkeypatch: pytest.MonkeyPatch, make_package: Callable[[], ModuleType]
) -> None:
    package = make_package()
    first = next(tool for tool in TOOL_ID_ORDER if sys.monitoring.get_tool(tool) is None)
    sys.monitoring.use_tool_id(first, "a racing tool")
    looked_up = sys.monitoring.get_tool
    monkeypatch.setattr(
        sys.monitoring, "get_tool", lambda tool: None if tool == first else looked_up(tool)
    )
    try:
        with counted_calls(by_package(package.__name__)) as counts:
            package.leaf()
        assert looked_up(first) == "a racing tool"
    finally:
        sys.monitoring.free_tool_id(first)

    assert counts[package.__name__] == 1


# coverage.py's tracer does not see code that runs inside a sys.monitoring callback, so the
# recorder's callbacks are also driven directly here; the tests above drive them through events.


def test_the_recorder_adds_each_start_and_each_weight() -> None:
    def weigh(callee: object, first: object) -> int:
        return 2 if callee is len and first == "x" else 0

    recorder = CallRecorder(lambda code: ("all", code.co_name), {"weighed": weigh})
    code = _owners.__code__

    recorder.started(code, 0)
    recorder.started(code, 0)
    recorder.called(code, 0, len, "x")
    recorder.called(code, 0, len, "y")

    assert recorder.counts == {"all": 2, "_owners": 2, "weighed": 2}
    events = sys.monitoring.events
    assert recorder.events == events.PY_START | events.CALL
    assert CallRecorder(lambda _: (), {}).events == events.PY_START


def test_the_recorder_stops_counting_after_a_key_function_raises() -> None:
    def keys(code: CodeType) -> tuple[str, ...]:
        if code is _owners.__code__:
            raise LookupError(code.co_name)
        return ("counted",)

    recorder = CallRecorder(keys, {"calls": lambda _callee, _first: 1})

    with pytest.raises(LookupError):
        recorder.started(_owners.__code__, 0)
    recorder.started(test_the_recorder_adds_each_start_and_each_weight.__code__, 0)
    recorder.called(_owners.__code__, 0, len, None)

    assert recorder.counts == {}
    assert [type(error) for error in recorder.failures] == [LookupError]


def test_the_recorder_stops_counting_after_a_weight_raises() -> None:
    def weigh(callee: object, first: object) -> int:
        del callee, first
        raise LookupError

    recorder = CallRecorder(lambda _: ("counted",), {"broken": weigh})

    with pytest.raises(LookupError):
        recorder.called(_owners.__code__, 0, len, None)
    recorder.called(_owners.__code__, 0, len, None)
    recorder.started(_owners.__code__, 0)

    assert recorder.counts == {}
    assert len(recorder.failures) == 1
