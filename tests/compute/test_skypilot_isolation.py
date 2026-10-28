"""Exercise the generated mount boundary with the real local bubblewrap binary."""

import errno
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from substrax.compute import JobSpec, Mount, MountAccess, Task
from substrax.compute.backend import backend_state_dir, new_run, stage_run
from substrax.compute.skypilot_backend import SkyPilotBackend
from substrax.compute.skypilot_transports import SPEC_PATH


def test_generated_read_only_boundary(tmp_path: Path) -> None:
    """The worker sees read-only inputs, writable outputs and a new network namespace."""
    if sys.platform != "linux" or shutil.which("bwrap") is None:
        pytest.skip("Linux bubblewrap integration")
    inputs, outputs = tmp_path / "input", tmp_path / "output"
    inputs.mkdir()
    outputs.mkdir()
    backend = SkyPilotBackend(
        {
            "state_dir": str(tmp_path / "state"),
            "copy": {"sources": {"data": str(inputs)}, "network_isolation": True},
        }
    )
    spec = JobSpec(
        name="isolation",
        timeout_seconds=10,
        tasks=(Task(name="probe", argv=("true",), timeout_seconds=5),),
        mounts=(Mount(name="data", path=str(inputs), access=MountAccess.READ_ONLY),),
    )
    run = new_run(spec, Path.cwd(), settings={})
    run_dir = stage_run(backend_state_dir(str(tmp_path / "state"), "skypilot"), run)
    task = backend.task_for(spec, project=run.project, run_id=run.run_id, run_dir=run_dir)
    boundary = str(task["run"]).splitlines()[-1].split(" -- ", 1)[0]
    boundary = boundary.replace("/outputs", str(outputs)).replace(
        SPEC_PATH, str(run_dir / "spec.json")
    )
    body = (
        "import errno,os; from pathlib import Path; "
        f"root=Path({str(inputs)!r}); out=Path({str(outputs)!r})\n"
        "try: (root/'forbidden').write_text('bad')\n"
        "except OSError as e: assert e.errno in (errno.EROFS,errno.EACCES,errno.EPERM)\n"
        "else: raise AssertionError('input writable')\n"
        "(out/'proof').write_text(os.readlink('/proc/self/ns/net'))\n"
    )
    command = boundary + " -- " + shlex.join([sys.executable, "-c", body])
    result = subprocess.run(  # noqa: S603 - generated boundary with quoted fixture arguments
        ["/bin/bash", "-c", command], capture_output=True, timeout=10, check=False
    )
    assert result.returncode == 0, json.dumps(
        {"stderr": result.stderr.decode(), "errno": errno.EPERM}
    )
    assert (outputs / "proof").read_text() != str(Path("/proc/self/ns/net").readlink())
    assert not (inputs / "forbidden").exists()
