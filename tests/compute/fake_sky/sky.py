"""A stand-in for SkyPilot's public SDK, running each task's ``run`` script locally.

State lives under ``$FAKE_SKY_ROOT``: a bucket ``gs://NAME`` is ``buckets/NAME``, and a cluster is
``clusters/<name>`` holding its job's pid, log and exit code. Paths the task mounts are rewritten
to their local directories in the script. The launch ``down`` flag removes the cluster once its job ends, as
SkyPilot's autodown does, so a finished run is known only from its bucket.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shlex
import shutil
import signal
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(os.environ["FAKE_SKY_ROOT"])


def bucket_dir(uri: str) -> Path:
    return ROOT / "buckets" / uri.removeprefix("gs://")


def cluster_dir(name: str) -> Path:
    return ROOT / "clusters" / name


def launch_command(arguments: argparse.Namespace) -> int:
    task = json.loads(Path(arguments.task).read_text())
    local: dict[str, str] = {}
    for remote, source in task["file_mounts"].items():
        if isinstance(source, dict):
            local[remote] = str(bucket_dir(source["source"]))
            Path(local[remote]).mkdir(parents=True, exist_ok=True)
        else:
            local[remote] = source
    script = task["run"]
    for remote in sorted(local, key=len, reverse=True):
        script = script.replace(remote, local[remote])
    cluster = cluster_dir(arguments.cluster)
    cluster.mkdir(parents=True)
    (cluster / "task.json").write_text(json.dumps({"down": arguments.down}))
    wrapped = f"cd {shlex.quote(task['workdir'])}\n{script}\necho $? > {cluster / 'exit'}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    pid = os.posix_spawn(
        "/bin/bash",
        ["/bin/bash", "-c", wrapped],
        dict(os.environ),
        file_actions=[
            (os.POSIX_SPAWN_OPEN, 1, str(cluster / "job.log"), flags, 0o644),
            (os.POSIX_SPAWN_DUP2, 1, 2),
        ],
        setpgroup=0,
    )
    (cluster / "pid").write_text(str(pid))
    return 0


class Task:
    @staticmethod
    def from_yaml(path: str) -> str:
        return path


def launch(task: str, *, cluster_name: str, retry_until_up: bool, down: bool) -> str:
    assert not retry_until_up
    launch_command(argparse.Namespace(task=task, cluster=cluster_name, down=down))
    return _request("launch", [1, None])


def _ended(cluster: Path) -> bool:
    return (cluster / "exit").exists() or (cluster / "cancelled").exists()


def _gone(name: str) -> bool:
    cluster = cluster_dir(name)
    if not cluster.exists():
        return True
    down = json.loads((cluster / "task.json").read_text())["down"]
    return down and _ended(cluster) and (cluster / "logged").exists()


def queue_command(arguments: argparse.Namespace) -> int:
    records = {}
    for name in arguments.clusters:
        cluster = cluster_dir(name)
        if _gone(name):
            sys.stdout.write(f"Failed to get the job queue for cluster {name!r}.\n")
            continue
        if (cluster / "cancelled").exists():
            status = "CANCELLED"
        elif (cluster / "exit").exists():
            status = "SUCCEEDED" if (cluster / "exit").read_text().strip() == "0" else "FAILED"
        else:
            status = "RUNNING"
        records[name] = [{"job_id": 1, "status": status}]
    sys.stdout.write(json.dumps(records, indent=2) + "\n")
    return 0


def queue(cluster: str) -> str:
    """Return native SDK-style records for the adapter contract."""
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        queue_command(argparse.Namespace(clusters=[cluster]))
    text = output.getvalue()
    start = text.index("{")
    jobs = json.loads(text[start:]).get(cluster, [])
    return _request("queue", jobs)


def _request(operation: str, result: object) -> str:
    requests = ROOT / "requests"
    requests.mkdir(parents=True, exist_ok=True)
    identity = "request-" + uuid.uuid4().hex
    (requests / identity).write_text(json.dumps({"operation": operation, "result": result}))
    return identity


def api_cancel(*, request_ids: list[str]) -> str:
    for identity in request_ids:
        assert (ROOT / "requests" / identity).is_file()
    return _request("api_cancel", request_ids)


def get(request: str) -> object:
    """Resolve a persistent request identifier, never the result itself."""
    record = json.loads((ROOT / "requests" / request).read_text())
    result = record["result"]
    if record["operation"] == "queue":
        for job in result:
            job["status"] = SimpleNamespace(value=job["status"])
    return result


def tail_logs(cluster_name: str, job_id: int, follow: bool, *, preload_content: bool) -> object:
    assert not preload_content
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        logs(argparse.Namespace(cluster=cluster_name, job_id=job_id, follow=follow))
    return iter([output.getvalue(), None])


def cancel(cluster_name: str, *, job_ids: list[int]) -> str:
    assert job_ids == [1]
    cancel_command(argparse.Namespace(cluster=cluster_name))
    return _request("cancel", None)


def down(cluster_name: str) -> str:
    down_command(argparse.Namespace(cluster=cluster_name))
    return _request("down", None)


def logs(arguments: argparse.Namespace) -> int:
    cluster = cluster_dir(arguments.cluster)
    with (cluster / "job.log").open() as log:
        while True:
            line = log.readline()
            if line:
                sys.stdout.write(line)
                sys.stdout.flush()
            elif arguments.follow and not _ended(cluster):
                time.sleep(0.1)
            else:
                sys.stdout.write(log.read())
                break
    (cluster / "logged").touch()
    return 0


def cancel_command(arguments: argparse.Namespace) -> int:
    cluster = cluster_dir(arguments.cluster)
    (cluster / "cancelled").touch()
    with contextlib.suppress(ProcessLookupError):
        os.killpg(int((cluster / "pid").read_text()), signal.SIGTERM)
    return 0


def down_command(arguments: argparse.Namespace) -> int:
    shutil.rmtree(cluster_dir(arguments.cluster), ignore_errors=True)
    return 0
