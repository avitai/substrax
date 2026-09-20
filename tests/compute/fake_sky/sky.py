"""A stand-in for SkyPilot's ``sky`` command, running each task's ``run`` script locally.

State lives under ``$FAKE_SKY_ROOT``: a bucket ``gs://NAME`` is ``buckets/NAME``, and a cluster is
``clusters/<name>`` holding its job's pid, log and exit code. Paths the task mounts are rewritten
to their local directories in the script. ``--down`` removes the cluster once its job ends, as
SkyPilot's autodown does, so a finished run is known only from its bucket.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shlex
import shutil
import signal
import sys
import time
from pathlib import Path


ROOT = Path(os.environ["FAKE_SKY_ROOT"])


def bucket_dir(uri: str) -> Path:
    return ROOT / "buckets" / uri.removeprefix("gs://")


def cluster_dir(name: str) -> Path:
    return ROOT / "clusters" / name


def launch(arguments: argparse.Namespace) -> int:
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


def _ended(cluster: Path) -> bool:
    return (cluster / "exit").exists() or (cluster / "cancelled").exists()


def _gone(name: str) -> bool:
    cluster = cluster_dir(name)
    if not cluster.exists():
        return True
    down = json.loads((cluster / "task.json").read_text())["down"]
    return down and _ended(cluster) and (cluster / "logged").exists()


def queue(arguments: argparse.Namespace) -> int:
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


def cancel(arguments: argparse.Namespace) -> int:
    cluster = cluster_dir(arguments.cluster)
    (cluster / "cancelled").touch()
    with contextlib.suppress(ProcessLookupError):
        os.killpg(int((cluster / "pid").read_text()), signal.SIGTERM)
    return 0


def down(arguments: argparse.Namespace) -> int:
    shutil.rmtree(cluster_dir(arguments.cluster), ignore_errors=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="sky")
    commands = parser.add_subparsers(required=True)
    p = commands.add_parser("launch")
    p.add_argument("task")
    p.add_argument("-c", "--cluster", required=True)
    p.add_argument("-d", "--detach-run", action="store_true")
    p.add_argument("--down", action="store_true")
    p.add_argument("-y", "--yes", action="store_true")
    p.set_defaults(command=launch)
    p = commands.add_parser("queue")
    p.add_argument("clusters", nargs="+")
    p.add_argument("-o", "--output", choices=["json"], required=True)
    p.set_defaults(command=queue)
    p = commands.add_parser("logs")
    p.add_argument("cluster")
    p.add_argument("job_id")
    p.add_argument("--follow", action="store_true")
    p.add_argument("--no-follow", dest="follow", action="store_false")
    p.set_defaults(command=logs)
    p = commands.add_parser("cancel")
    p.add_argument("cluster")
    p.add_argument("-a", "--all", action="store_true")
    p.add_argument("-y", "--yes", action="store_true")
    p.set_defaults(command=cancel)
    p = commands.add_parser("down")
    p.add_argument("cluster")
    p.add_argument("-y", "--yes", action="store_true")
    p.set_defaults(command=down)
    arguments = parser.parse_args()
    return arguments.command(arguments)


if __name__ == "__main__":
    sys.exit(main())
