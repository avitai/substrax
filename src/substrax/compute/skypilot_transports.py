"""How a SkyPilot run's inputs reach its cluster and its outputs come back.

The ``skypilot`` backend runs every job the same way and leaves this to its transport. The one
transport, :class:`BucketTransport`, keeps inputs and outputs on object-store buckets SkyPilot
mounts (``mode: MOUNT``), so the outputs outlive the cluster, which SkyPilot tears down when the
job ends. Outputs are fetched with the bucket's own tool: ``gcloud storage`` for ``gs://``,
``aws s3`` for ``s3://``.
"""

from __future__ import annotations

import json
import shlex
import subprocess  # nosec B404 - runs the bucket tools, no shell
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from substrax.compute.backend import RunHandle, RunState
from substrax.compute.spec import JobSpec
from substrax.compute.worker import JobManifest, MANIFEST_NAME, read_manifest
from substrax.records import UNKNOWN_FIELDS_REFUSED
from substrax.typing import JsonValue


OUTPUTS_PATH = "/outputs"
SPEC_PATH = "/tmp/substrax-job-spec.json"  # noqa: S108  # nosec B108 - a path on the run's own cluster
UV_ON_PATH = 'export PATH="$HOME/.local/bin:$PATH"'
_DOWNLOADERS = {
    "gs://": ("gcloud", "storage", "rsync", "--recursive"),
    "s3://": ("aws", "s3", "sync"),
}


@dataclass(frozen=True, slots=True, kw_only=True)
class BucketSettings:
    """``[tool.substrax.compute.backends.skypilot.bucket]``: buckets SkyPilot mounts.

    Attributes:
        outputs_bucket: The ``gs://`` or ``s3://`` URI the runs' outputs go to.
        buckets: The bucket URI behind each of a job's mounts, by mount name.
    """

    __pydantic_config__ = UNKNOWN_FIELDS_REFUSED

    outputs_bucket: str
    buckets: Mapping[str, str] = field(default_factory=dict)


class Transport(Protocol):
    """The part of a SkyPilot run that depends on how its files move."""

    @property
    def down_with_job(self) -> bool:
        """Whether SkyPilot removes the cluster as soon as its job ends."""
        ...

    @property
    def recovers_without_queue(self) -> bool:
        """Whether the outputs answer for the run when its cluster's queue cannot be read."""
        ...

    def add_to_task(
        self, spec: JobSpec, resources: dict[str, JsonValue], mounts: dict[str, JsonValue]
    ) -> None:
        """Add the transport's resources and the job's input and output mounts to the task."""
        ...

    def wrap_worker(self, worker: str, spec: JobSpec) -> str:
        """The command that runs ``worker`` on the cluster."""
        ...

    def fetch(self, handle: RunHandle, target: Path) -> None:
        """Copy the run's outputs into ``target``."""
        ...

    def state_from_outputs(self, handle: RunHandle) -> RunState:
        """The run's state once its cluster no longer reports it, from what it left behind."""
        ...


class BucketTransport:
    """Inputs and outputs on buckets SkyPilot mounts; see the module documentation."""

    down_with_job = True
    # The cluster may still be running the job while its queue cannot be read.
    recovers_without_queue = False

    def __init__(self, settings: BucketSettings) -> None:  # noqa: DOC502  # the ValueError is raised by downloader
        """Check the settings.

        Args:
            settings: The bucket settings.

        Raises:
            ValueError: If the outputs bucket's scheme has no download tool.
        """
        downloader(settings.outputs_bucket)
        self._settings = settings

    def add_to_task(
        self, spec: JobSpec, resources: dict[str, JsonValue], mounts: dict[str, JsonValue]
    ) -> None:
        """Mount the outputs bucket and each input mount's bucket.

        Args:
            spec: The job.
            resources: The task's resources; unchanged.
            mounts: The task's file mounts.

        Raises:
            LookupError: If a mount has no configured bucket.
        """
        del resources
        mounts[OUTPUTS_PATH] = {"source": self._settings.outputs_bucket, "mode": "MOUNT"}
        for mount in spec.mounts:
            if mount.name not in self._settings.buckets:
                msg = f"mount {mount.name!r} has no bucket in the backend's `buckets`"
                raise LookupError(msg)
            mounts[mount.path] = {"source": self._settings.buckets[mount.name], "mode": "MOUNT"}

    def wrap_worker(self, worker: str, spec: JobSpec) -> str:
        """The worker runs as it is.

        Args:
            worker: The worker command.
            spec: The job.

        Returns:
            ``worker``.
        """
        del spec
        return worker

    def fetch(self, handle: RunHandle, target: Path) -> None:
        """Download the run's directory of the outputs bucket.

        Args:
            handle: The run.
            target: The directory to copy into.
        """
        source = self._outputs(handle)
        run_tool([*downloader(source), source, str(target)])

    def state_from_outputs(self, handle: RunHandle) -> RunState:
        """The state the run's manifest in the outputs bucket records.

        Args:
            handle: The run.

        Returns:
            Its state; ``failed`` when the run left no finished manifest.
        """
        with tempfile.TemporaryDirectory() as scratch:
            source = self._outputs(handle)
            completed = subprocess.run(  # noqa: S603  # nosec B603 - the bucket tool, no shell
                [*downloader(source), source, scratch], capture_output=True, check=False
            )
            manifest_path = Path(scratch) / MANIFEST_NAME
            if completed.returncode != 0 or not manifest_path.is_file():
                return RunState.FAILED
            return _finished_state(_read(manifest_path)) or RunState.FAILED

    def _outputs(self, handle: RunHandle) -> str:
        return f"{self._settings.outputs_bucket.rstrip('/')}/{handle.run_id}"


def run_tool(command: Sequence[str]) -> str:
    """Run a command line tool and return its standard output.

    Args:
        command: The command.

    Returns:
        Its standard output.

    Raises:
        RuntimeError: If it exits non-zero; the message holds its standard error.
    """
    completed = subprocess.run(  # noqa: S603  # nosec B603 - a configured tool, no shell
        list(command), capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        msg = f"{shlex.join(command)} failed ({completed.returncode}): {completed.stderr.strip()}"
        raise RuntimeError(msg)
    return completed.stdout


def downloader(uri: str) -> tuple[str, ...]:
    """The command that downloads a bucket URI's directory.

    Args:
        uri: A ``gs://`` or ``s3://`` URI.

    Returns:
        The command, to which the source and the destination are appended.

    Raises:
        ValueError: If the scheme has no download tool.
    """
    for scheme, command in _DOWNLOADERS.items():
        if uri.startswith(scheme):
            return command
    msg = f"no download tool for {uri!r}; use a gs:// or s3:// bucket"
    raise ValueError(msg)


def _read(path: Path) -> JobManifest:
    return read_manifest(json.loads(path.read_text(encoding="utf-8")))


def _finished_state(manifest: JobManifest) -> RunState | None:
    """``succeeded`` or ``failed`` for a finished job's manifest, ``None`` for an unfinished one."""
    if not manifest.finished:
        return None
    return RunState.SUCCEEDED if manifest.succeeded else RunState.FAILED
