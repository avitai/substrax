"""The bridge calls SkyPilot's public SDK and reads its native records, never its CLI's text."""

from enum import Enum
from pathlib import Path
from types import ModuleType

import pytest

from substrax.compute.sky_sdk import (
    cancel_job,
    cancel_request,
    down_cluster,
    launch_recorded,
    queue_snapshot,
)


class Status(Enum):
    """Provider record spelling."""

    RUNNING = "RUNNING"


def _sdk(**members: object) -> ModuleType:
    """A stand-in for the ``sky`` module holding exactly ``members``."""
    api = ModuleType("sky")
    vars(api).update(members)
    return api


def test_queue_uses_sdk_get_and_serializes_native_status() -> None:
    """SDK queue returns a request, whose result is a list of records."""
    calls: list[str] = []

    def queue(cluster: str) -> str:
        calls.append(cluster)
        return "request-id"

    def get(request: str) -> list[dict[str, object]]:
        assert request == "request-id"
        return [{"job_id": 4, "status": Status.RUNNING}]

    api = _sdk(queue=queue, get=get)
    assert queue_snapshot(api, "owned-cluster") == [{"job_id": "4", "status": "RUNNING"}]
    assert calls == ["owned-cluster"]


def test_queue_errors_are_not_reported_as_job_failure() -> None:
    """A transport error must remain visible to the caller."""

    def queue(cluster: str) -> str:
        raise ConnectionError(cluster)

    with pytest.raises(ConnectionError, match="owned-cluster"):
        queue_snapshot(_sdk(queue=queue, get=lambda request: request), "owned-cluster")


def test_launch_persists_request_before_waiting(tmp_path: Path) -> None:
    """A failed wait must leave the exact cancellable server request on disk."""
    receipt = tmp_path / "request.json"

    class Task:
        @staticmethod
        def from_yaml(path: str) -> str:
            return path

    def launch(task: str, **kwargs: object) -> str:
        assert task == "task.yaml"
        assert kwargs == {"cluster_name": "owned", "retry_until_up": False, "down": False}
        return "exact-request"

    def get(request: str) -> None:
        assert request == "exact-request"
        assert '"request_id": "exact-request"' in receipt.read_text()
        raise ConnectionError("client disconnected")

    api = _sdk(Task=Task, launch=launch, get=get)
    with pytest.raises(ConnectionError, match="disconnected"):
        launch_recorded(api, "task.yaml", "owned", receipt, down=False)
    assert receipt.exists()


def test_cancel_targets_only_the_recorded_request() -> None:
    """Cancelling a client process is insufficient to stop server provisioning."""
    calls: list[object] = []

    def api_cancel(**kwargs: object) -> str:
        calls.append(kwargs)
        return "cancel-operation"

    cancel_request(_sdk(api_cancel=api_cancel, get=calls.append), "owned-request")
    assert calls == [{"request_ids": ["owned-request"]}, "cancel-operation"]


def test_job_cleanup_uses_one_sdk_and_exact_job_id() -> None:
    """Cleanup uses public request APIs without switching to CLI display commands."""
    calls: list[object] = []

    def cancel(cluster: str, *, job_ids: list[int]) -> str:
        calls.append((cluster, job_ids))
        return "cancel-id"

    def down(cluster: str) -> str:
        calls.append(cluster)
        return "down-id"

    api = _sdk(cancel=cancel, down=down, get=calls.append)
    cancel_job(api, "owned", 7)
    down_cluster(api, "owned")
    assert calls == [("owned", [7]), "cancel-id", "owned", "down-id"]
