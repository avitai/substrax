"""A task timeout must not leave descendants holding the worker's pipes."""

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from substrax.compute import JobSpec, Task
from substrax.compute.local import LocalBackend


def test_timeout_reaps_the_task_process_group(tmp_path: Path) -> None:
    """An inherited stdout pipe must not extend the task's lifetime."""
    if os.name != "posix":
        pytest.skip("POSIX process groups")
    pid_file = tmp_path / "descendant.pid"
    script = tmp_path / "check.py"
    ready = tmp_path / "ready"
    script.write_text(
        "from pathlib import Path\n"
        "from substrax.compute import JobSpec, Task\n"
        "from substrax.compute.worker import run_job\n"
        "from substrax.runtime import JaxRuntime\n"
        f"root=Path({str(tmp_path)!r})\n"
        'body="import subprocess,sys,time; from pathlib import Path; '
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        f'Path({str(pid_file)!r}).write_text(str(p.pid)); time.sleep(60)"\n'
        "job=JobSpec(name='tree',tasks=(Task(name='child',argv=('python','-c',body),"
        "timeout_seconds=0.5),),timeout_seconds=20,runtime=JaxRuntime(platforms=('cpu',)))\n"
        f"Path({str(ready)!r}).write_text('ready')\n"
        "result=run_job(job,project=root,outputs=root/'outputs',echo=False)\n"
        "assert result.tasks[0].status.value=='timed-out'\n",
        encoding="utf-8",
    )
    process = subprocess.Popen(  # noqa: S603 - fixed local regression program
        [sys.executable, str(script)], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    try:
        # Import/instrumentation startup has its own bound; the task still gets
        # 0.5 seconds and the pipe-drain assertion still gets 12 seconds.
        deadline = time.monotonic() + 30
        while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert ready.exists(), "worker imports did not complete within the startup budget"
        _stdout, stderr = process.communicate(timeout=12)
        assert process.returncode == 0, stderr.decode()
        assert pid_file.exists(), "timeout must be exercised after the descendant starts"
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate()
        if pid_file.exists():
            with contextlib.suppress(ProcessLookupError):
                os.kill(int(pid_file.read_text()), signal.SIGKILL)


@pytest.mark.parametrize("ignore_term", [False, True])
def test_local_cancel_stops_task_and_descendant(tmp_path: Path, ignore_term: bool) -> None:
    """Cancellation must stop actual processes, including TERM-resistant children."""

    if os.name != "posix":
        pytest.skip("POSIX local backend")
    marker = tmp_path / "pids.json"
    body = (
        "import json,os,signal,subprocess,sys,time; from pathlib import Path; "
        + ("signal.signal(signal.SIGTERM,signal.SIG_IGN); " if ignore_term else "")
        + "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        + f"Path({str(marker)!r}).write_text(json.dumps([os.getpid(),child.pid])); time.sleep(60)"
    )
    backend = LocalBackend({"python": sys.executable, "state_dir": str(tmp_path / "state")})
    spec = JobSpec(
        name="cancel-tree",
        timeout_seconds=30,
        tasks=(Task(name="tree", argv=("python", "-c", body), timeout_seconds=25),),
    )
    handle = backend.submit(spec, project=tmp_path)
    pids = []
    try:
        deadline = time.monotonic() + 10
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert marker.exists(), "task did not start"
        pids = json.loads(marker.read_text())
        backend.cancel(handle)
        deadline = time.monotonic() + 3
        while any(_running(pid) for pid in pids) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not any(_running(pid) for pid in pids), "task tree survived cancellation"
    finally:
        for pid in [*pids, int(handle.details["pid"])]:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)


def _running(pid: int) -> bool:
    state = subprocess.run(  # noqa: S603 - numeric PID of this test's child
        ["ps", "-o", "stat=", "-p", str(pid)],  # noqa: S607 - POSIX ps
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return bool(state) and not state.startswith("Z")
