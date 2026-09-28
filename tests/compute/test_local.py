"""The ``local`` backend passes the backend contract and reads its settings strictly."""

from __future__ import annotations

import json
import sys
from datetime import datetime, UTC
from pathlib import Path

import pytest

from substrax.compute import local
from substrax.compute.backend import ComputeBackend, RunHandle, RunState
from substrax.compute.local import LocalBackend
from substrax.compute.worker import JobManifest, MANIFEST_NAME
from substrax.records import dump_record
from substrax.testing.compute import BackendContract


class TestLocalBackend(BackendContract):
    def make_backend(self, state_dir: Path) -> ComputeBackend:
        return LocalBackend({"python": sys.executable, "state_dir": str(state_dir)})


def test_an_unknown_setting_is_refused() -> None:
    with pytest.raises(ValueError, match="gpu"):
        LocalBackend({"gpu": "L4"})


def test_a_setting_of_the_wrong_type_is_refused() -> None:
    with pytest.raises(ValueError, match="python"):
        LocalBackend({"python": 3})


def test_a_run_that_finishes_between_the_status_checks_reads_as_finished(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The worker writes its final manifest and exits while ``status`` is checking.

    The stand-in for the liveness check does exactly that before reporting the worker gone, so
    ``status`` must read the manifest after observing liveness to see the finished run.
    """
    run_dir = tmp_path / "run"
    (run_dir / "outputs").mkdir(parents=True)
    handle = RunHandle(
        run_id="run",
        backend="local",
        job="job",
        project=str(tmp_path),
        submitted_at=datetime.now(UTC),
        details={"pid": "1", "run_dir": str(run_dir)},
    )

    def the_worker_finishes_then_exits(pid: int, run_id: str) -> bool:
        del pid, run_id
        finished = dump_record(JobManifest(job="job", finished=True))
        (run_dir / "outputs" / MANIFEST_NAME).write_text(json.dumps(finished), "utf-8")
        return False

    monkeypatch.setattr(local, "_is_worker", the_worker_finishes_then_exits)

    assert LocalBackend({"state_dir": str(tmp_path / "state")}).status(handle) is RunState.SUCCEEDED
