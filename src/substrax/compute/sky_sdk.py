"""Small public-SDK bridge executed in the separate SkyPilot tool environment.

This file intentionally imports only the standard library until its entry point
runs. The scientific environment does not need SkyPilot installed.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import TextIO


def queue_snapshot(api: ModuleType, cluster: str) -> list[dict[str, str]]:
    """Read native queue records through the public asynchronous SDK.

    Args:
        api: The SkyPilot module, or an SDK-contract test double.
        cluster: The exact owned cluster name.

    Returns:
        Job identifiers and status names, without provider-specific objects.
    """
    jobs = api.get(api.queue(cluster))
    return [{"job_id": str(job["job_id"]), "status": job["status"].value} for job in jobs]


def launch_recorded(
    api: ModuleType, task: str, cluster: str, receipt: Path, *, down: bool
) -> dict[str, str]:
    """Persist the provisioning request before waiting for its result.

    Args:
        api: The public SkyPilot SDK module.
        task: The staged task file.
        cluster: The owned cluster name.
        receipt: New receipt path; existing attempts cannot be overwritten.
        down: Whether SkyPilot should remove the cluster when the job ends.

    Returns:
        Exact request and job identifiers.
    """
    # Reserve the local receipt before any request. A crash leaves an explicit
    # unresolved attempt; callers must reconcile it rather than submit again.
    with receipt.open("x", encoding="utf-8") as stream:
        request = api.launch(
            api.Task.from_yaml(task), cluster_name=cluster, retry_until_up=False, down=down
        )
        identity = {"request_id": str(request), "cluster": cluster}
        json.dump(identity, stream)
        stream.flush()
        os.fsync(stream.fileno())
    job_id, _handle = api.get(request)
    return {**identity, "job_id": str(job_id)}


def cancel_request(api: ModuleType, request_id: str) -> None:
    """Wait for cancellation of one owned server request.

    Args:
        api: The public SkyPilot SDK module.
        request_id: The exact durable launch request identity.
    """
    api.get(api.api_cancel(request_ids=[request_id]))


def cancel_job(api: ModuleType, cluster: str, job_id: int) -> None:
    """Cancel one job through the same SDK used for submission.

    Args:
        api: The public SkyPilot SDK module.
        cluster: The owned cluster.
        job_id: The exact job identifier.
    """
    api.get(api.cancel(cluster, job_ids=[job_id]))


def down_cluster(api: ModuleType, cluster: str) -> None:
    """Wait for SDK teardown without purging failed provider state.

    Args:
        api: The public SkyPilot SDK module.
        cluster: The owned cluster.
    """
    api.get(api.down(cluster))


def stream_logs(
    api: ModuleType, cluster: str, job_id: int, *, follow: bool, output: TextIO
) -> None:
    """Stream native log chunks, separate from SDK diagnostics.

    Args:
        api: The public SkyPilot SDK module.
        cluster: The owned cluster.
        job_id: The exact job identifier.
        follow: Wait for subsequent log output.
        output: Destination for job log text.
    """
    for chunk in api.tail_logs(cluster, job_id, follow, preload_content=False):
        if chunk is not None:
            output.write(chunk)
            output.flush()


def main() -> None:
    """Execute a public SDK operation; diagnostics go to standard error."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="operation", required=True)
    queue = sub.add_parser("queue")
    queue.add_argument("cluster")
    launch = sub.add_parser("launch")
    launch.add_argument("cluster")
    launch.add_argument("task")
    launch.add_argument("receipt", type=Path)
    launch.add_argument("--down", action="store_true")
    cancel = sub.add_parser("cancel-request")
    cancel.add_argument("request_id")
    cancel_job_parser = sub.add_parser("cancel-job")
    cancel_job_parser.add_argument("cluster")
    cancel_job_parser.add_argument("job_id", type=int)
    down = sub.add_parser("down")
    down.add_argument("cluster")
    logs = sub.add_parser("logs")
    logs.add_argument("cluster")
    logs.add_argument("job_id", type=int)
    logs.add_argument("--follow", action="store_true")
    args = parser.parse_args()
    output = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        api = importlib.import_module("sky")
        if args.operation == "queue":
            records = queue_snapshot(api, args.cluster)
        elif args.operation == "cancel-request":
            cancel_request(api, args.request_id)
            records = {"cancelled_request": args.request_id}
        elif args.operation == "cancel-job":
            cancel_job(api, args.cluster, args.job_id)
            records = {"cancelled_job": str(args.job_id)}
        elif args.operation == "down":
            down_cluster(api, args.cluster)
            records = {"down": args.cluster}
        elif args.operation == "logs":
            stream_logs(api, args.cluster, args.job_id, follow=args.follow, output=output)
            return
        else:
            records = launch_recorded(api, args.task, args.cluster, args.receipt, down=args.down)
    sys.stdout.write(json.dumps(records) + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # External process boundary reports every SDK failure.
        sys.stderr.write(f"{type(error).__name__}: {error}\n")
        sys.exit(1)
