"""The behaviour every compute backend owes, as pytest tests a backend's own suite inherits.

Subclass :class:`BackendContract` in a test module and define ``make_backend``; the inherited tests
submit small jobs of plain Python scripts, wait for them, and check the states, logs, fetched
outputs and cancellation the :class:`~substrax.compute.backend.ComputeBackend` protocol promises.
The scripts import no jax, so a backend that runs them locally finishes each test in seconds.

```python
class TestLocalBackend(BackendContract):
    def make_backend(self, state_dir: Path) -> ComputeBackend:
        return LocalBackend({"python": sys.executable, "state_dir": str(state_dir)})
```
"""

from __future__ import annotations

import json
import textwrap
import time
from pathlib import Path

import pytest

from substrax.compute.backend import ComputeBackend, RunHandle, RunState, RunStore
from substrax.compute.spec import JobSpec, Task
from substrax.compute.worker import MANIFEST_NAME, read_manifest, STDOUT_LOG, TaskStatus


_TASK_BUDGET = 120.0
_WAIT_SECONDS = 180.0
_POLL_SECONDS = 0.2


def wait_until_final(
    backend: ComputeBackend, handle: RunHandle, *, timeout: float = _WAIT_SECONDS
) -> RunState:
    """Poll a run until it ends.

    Args:
        backend: The backend that runs it.
        handle: The run.
        timeout: Seconds to wait.

    Returns:
        Its final state.

    Raises:
        TimeoutError: If the run has not ended after ``timeout`` seconds.
    """
    deadline = time.monotonic() + timeout
    while (state := backend.status(handle)) and not state.is_final:
        if time.monotonic() > deadline:
            msg = f"run {handle.run_id} still {state} after {timeout:g} s"
            raise TimeoutError(msg)
        time.sleep(_POLL_SECONDS)
    return state


class BackendContract:
    """Tests every backend must pass; subclass and define :meth:`make_backend`."""

    def make_backend(self, state_dir: Path) -> ComputeBackend:
        """Build a fresh backend instance keeping whatever it stores under ``state_dir``.

        Args:
            state_dir: A directory private to the test.

        Returns:
            The backend.

        Raises:
            NotImplementedError: Always, until a subclass defines it.
        """
        raise NotImplementedError

    @pytest.fixture
    def project(self, tmp_path: Path) -> Path:
        """A project holding the contract's scripts.

        Args:
            tmp_path: The test's directory.

        Returns:
            The project root.
        """
        root = tmp_path / "project"
        scripts = {
            "writes": """
                import os
                from pathlib import Path
                print("hello from the task")
                target = Path(os.environ["AVITAI_OUTPUT_DIR"]) / "result"
                target.mkdir(parents=True)
                (target / "value.txt").write_text("42")
            """,
            "fails": "raise SystemExit(7)",
            "sleeps": "import time\nprint('sleeping', flush=True)\ntime.sleep(600)",
        }
        (root / "scripts").mkdir(parents=True)
        for name, body in scripts.items():
            (root / "scripts" / f"{name}.py").write_text(textwrap.dedent(body), encoding="utf-8")
        return root

    @pytest.fixture
    def state_dir(self, tmp_path: Path) -> Path:
        """The directory the backend under test keeps its state in.

        Args:
            tmp_path: The test's directory.

        Returns:
            The directory.
        """
        return tmp_path / "state"

    def test_a_job_succeeds_and_its_outputs_come_back(
        self, project: Path, state_dir: Path, tmp_path: Path
    ) -> None:
        """A succeeding job ends ``succeeded`` and fetches its logs, outputs and manifest."""
        backend = self.make_backend(state_dir)
        handle = backend.submit(_job("writes"), project=project)

        state = wait_until_final(backend, handle)
        fetched = backend.fetch(handle, tmp_path / "fetched")

        _expect_equal("final state", state, RunState.SUCCEEDED)
        _expect_equal("fetched directory", fetched, tmp_path / "fetched" / handle.run_id)
        manifest = read_manifest(json.loads((fetched / MANIFEST_NAME).read_text("utf-8")))
        _expect_equal("manifest finished", manifest.finished, True)
        _expect_equal(
            "task states", [task.status for task in manifest.tasks], [TaskStatus.SUCCEEDED]
        )
        _expect_equal(
            "output file", (fetched / "writes" / "result" / "value.txt").read_text("utf-8"), "42"
        )
        _expect_in(
            "task stdout",
            "hello from the task",
            (fetched / "writes" / STDOUT_LOG).read_text("utf-8"),
        )

    def test_a_failed_task_fails_the_run_and_the_tasks_after_it_still_run(
        self, project: Path, state_dir: Path, tmp_path: Path
    ) -> None:
        """A failed task makes the run ``failed`` without skipping the rest."""
        backend = self.make_backend(state_dir)
        handle = backend.submit(_job("fails", "writes"), project=project)

        state = wait_until_final(backend, handle)
        fetched = backend.fetch(handle, tmp_path / "fetched")

        _expect_equal("final state", state, RunState.FAILED)
        manifest = read_manifest(json.loads((fetched / MANIFEST_NAME).read_text("utf-8")))
        _expect_equal(
            "task outcomes",
            [(task.status, task.returncode) for task in manifest.tasks],
            [(TaskStatus.FAILED, 7), (TaskStatus.SUCCEEDED, 0)],
        )

    def test_following_the_logs_ends_with_the_run_and_names_each_task(
        self, project: Path, state_dir: Path
    ) -> None:
        """Followed logs carry each task's lines, prefixed with its name, and stop at the end."""
        backend = self.make_backend(state_dir)
        handle = backend.submit(_job("writes"), project=project)

        lines = list(backend.logs(handle, follow=True))

        _expect_in("followed log lines", "[writes] hello from the task", lines)
        _expect_equal("run ended when the logs did", backend.status(handle).is_final, True)

    def test_a_cancelled_run_ends_cancelled(self, project: Path, state_dir: Path) -> None:
        """Cancelling a running job ends it as ``cancelled``; cancelling again changes nothing."""
        backend = self.make_backend(state_dir)
        handle = backend.submit(_job("sleeps"), project=project)
        _wait_for_line(backend, handle, "[sleeps] sleeping")

        backend.cancel(handle)
        state = wait_until_final(backend, handle)
        backend.cancel(handle)

        _expect_equal("final state", state, RunState.CANCELLED)
        _expect_equal("state after a second cancel", backend.status(handle), RunState.CANCELLED)

    def test_a_stored_handle_finds_the_run_from_a_new_backend(
        self, project: Path, state_dir: Path, tmp_path: Path
    ) -> None:
        """A handle read back from a ``RunStore`` is enough for another backend instance."""
        store = RunStore(tmp_path / "handles")
        submitted = self.make_backend(state_dir).submit(_job("writes"), project=project)
        store.save(submitted)

        handle = store.load(submitted.run_id)
        state = wait_until_final(self.make_backend(state_dir), handle)

        _expect_equal("stored handle", handle, submitted)
        _expect_equal("final state", state, RunState.SUCCEEDED)


def _expect_equal(what: str, actual: object, expected: object) -> None:
    """Raise ``AssertionError`` naming ``what`` unless the two are equal.

    The contract ships in the package, where pytest does not rewrite ``assert`` statements, so
    each check states its own message.
    """
    if actual != expected:
        msg = f"{what}: expected {expected!r}, got {actual!r}"
        raise AssertionError(msg)


def _expect_in(what: str, member: str, container: str | list[str]) -> None:
    """Raise ``AssertionError`` naming ``what`` unless ``member`` is in ``container``."""
    if member not in container:
        msg = f"{what}: {member!r} not found in {container!r}"
        raise AssertionError(msg)


def _job(*scripts: str) -> JobSpec:
    return JobSpec(
        name="contract",
        tasks=tuple(
            Task(name=name, argv=("python", f"scripts/{name}.py"), timeout_seconds=_TASK_BUDGET)
            for name in scripts
        ),
        timeout_seconds=_WAIT_SECONDS,
    )


def _wait_for_line(backend: ComputeBackend, handle: RunHandle, wanted: str) -> None:
    deadline = time.monotonic() + _WAIT_SECONDS
    while wanted not in backend.logs(handle, follow=False):
        if time.monotonic() > deadline:
            msg = f"run {handle.run_id} never logged {wanted!r}"
            raise TimeoutError(msg)
        time.sleep(_POLL_SECONDS)
