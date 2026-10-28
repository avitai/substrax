"""How a SkyPilot run's inputs reach its cluster and its outputs come back.

The ``skypilot`` backend runs every job the same way and leaves this to its transport, one of:

- :class:`BucketTransport`: inputs and outputs are object-store buckets SkyPilot mounts
  (``mode: MOUNT``), so the outputs outlive the cluster, which SkyPilot tears down when the job
  ends. Outputs are fetched with the bucket's own tool: ``gcloud storage`` for ``gs://``,
  ``aws s3`` for ``s3://``.
- :class:`CopyTransport`: SkyPilot copies explicit local inputs onto the cluster, read-only, and
  the outputs are fetched over SSH and rsync, verified against complete manifests of the remote
  directory taken before and after the copy. The cluster has to stay up until then, so it is
  removed by an explicit teardown, or by its autostop timer, which starts after setup and is not
  a provisioning deadline or a cost cap. The worker runs under bubblewrap, which the image has to
  provide with working user namespaces. Providers without bucket mounts (RunPod, Vast) use it.
"""

from __future__ import annotations

import json
import shlex
import subprocess  # nosec B404 - runs ssh, rsync and the bucket tools, no shell
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from substrax.compute import sky_transfer
from substrax.compute.backend import RunHandle, RunState
from substrax.compute.spec import JobSpec, MountAccess
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
# Providers SkyPilot cannot mount a bucket on.
_COPY_ONLY_PROVIDERS = frozenset({"runpod", "vast"})
_RECOVERED = "recovered.json"


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


@dataclass(frozen=True, slots=True, kw_only=True)
class CopySettings:
    """``[tool.substrax.compute.backends.skypilot.copy]``: local inputs copied to the cluster.

    Attributes:
        sources: The local path behind each of a job's mounts, by mount name; read-only.
        disk_size: The disk the cluster is asked for, in GB.
        lifetime_minutes: How long the cluster may idle after setup before it removes itself.
        network_isolation: Run the worker without network access.
    """

    __pydantic_config__ = UNKNOWN_FIELDS_REFUSED

    sources: Mapping[str, str] = field(default_factory=dict)
    disk_size: int = 60
    lifetime_minutes: int = 60
    network_isolation: bool = False


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

    def __init__(self, settings: BucketSettings, *, infra: str | None) -> None:
        """Check the settings against where the run goes.

        Args:
            settings: The bucket settings.
            infra: The backend's ``infra``.

        Raises:
            ValueError: If the provider cannot mount buckets, or the outputs bucket's scheme has
                no download tool.
        """
        if (infra or "").split("/")[0] in _COPY_ONLY_PROVIDERS:
            msg = f"{infra} cannot mount buckets; configure copy transport instead"
            raise ValueError(msg)
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


class CopyTransport:
    """Local inputs copied to the cluster, outputs fetched over SSH; see the module documentation."""

    down_with_job = False
    # A verified copy of the outputs answers for the run once it is taken.
    recovers_without_queue = True

    def __init__(self, settings: CopySettings, *, worker_python: str, runs: Path) -> None:
        """Check the settings.

        Args:
            settings: The copy settings.
            worker_python: The command that starts Python in the cluster's synced environment.
            runs: Where the backend keeps its runs' local files.

        Raises:
            ValueError: If the disk size or the lifetime is not positive.
        """
        if settings.disk_size <= 0 or settings.lifetime_minutes <= 0:
            msg = "copy transport's disk_size and lifetime_minutes must be positive"
            raise ValueError(msg)
        self._settings = settings
        self._worker_python = worker_python
        self._runs = runs

    def add_to_task(
        self, spec: JobSpec, resources: dict[str, JsonValue], mounts: dict[str, JsonValue]
    ) -> None:
        """Ask for the disk and the autostop timer, and copy each input mount's local source.

        Args:
            spec: The job.
            resources: The task's resources.
            mounts: The task's file mounts.

        Raises:
            ValueError: If a mount is writable; a copy cannot be written back.
            LookupError: If a mount has no configured source.
        """
        resources["disk_size"] = self._settings.disk_size
        resources["autostop"] = {
            "idle_minutes": self._settings.lifetime_minutes,
            "down": True,
            "wait_for": "none",
        }
        for mount in spec.mounts:
            if mount.access != MountAccess.READ_ONLY:
                msg = f"mount {mount.name!r} is writable; copy transport copies inputs read-only"
                raise ValueError(msg)
            if mount.name not in self._settings.sources:
                msg = f"mount {mount.name!r} has no local source in the backend's `sources`"
                raise LookupError(msg)
            mounts[mount.path] = str(Path(self._settings.sources[mount.name]).resolve(strict=True))

    def wrap_worker(self, worker: str, spec: JobSpec) -> str:
        """Run the worker under bubblewrap: the system, the project and the inputs read-only.

        Args:
            worker: The worker command.
            spec: The job; each mount is bound read-only.

        Returns:
            The bubblewrap command around ``worker``.
        """
        flags = ["--die-with-parent", "--unshare-user", "--unshare-pid"]
        if self._settings.network_isolation:
            flags.append("--unshare-net")
        binds = [
            ["--ro-bind", "/", "/"],
            ["--dev-bind", "/dev", "/dev"],
            ["--proc", "/proc"],
            ["--bind", OUTPUTS_PATH, OUTPUTS_PATH],
            ["--bind", "/tmp", "/tmp"],  # noqa: S108  # nosec B108 - the sandbox's scratch space
            *(["--ro-bind", mount.path, mount.path] for mount in spec.mounts),
            ["--ro-bind", SPEC_PATH, SPEC_PATH],
        ]
        bound = " ".join([shlex.join(["bwrap", *flags]), *(shlex.join(b) for b in binds)])
        # The project is the cluster's working directory, which only the shell knows.
        return f'{bound} --ro-bind "$PWD" "$PWD" -- {worker}'

    def fetch(self, handle: RunHandle, target: Path) -> None:
        """Copy the outputs over rsync and verify them against the remote directory's manifest.

        Args:
            handle: The run.
            target: The directory to copy into.

        Raises:
            ValueError: If the remote outputs changed during the copy, or the copy differs from
                them; the cluster is left up.
        """
        cluster, remote = handle.details["cluster"], f"{OUTPUTS_PATH}/{handle.run_id}"
        before = self._remote_manifest(cluster, remote)
        run_tool(
            ["rsync", "--archive", "--protect-args", "--", f"{cluster}:{remote}/", str(target)]
        )
        if self._remote_manifest(cluster, remote) != before:
            msg = f"the outputs of {handle.run_id} changed during the copy; the cluster is left up"
            raise ValueError(msg)
        sky_transfer.verify(target, before)
        receipt = {"path": str(target.resolve()), "manifest": before}
        (self._runs / handle.run_id / _RECOVERED).write_text(json.dumps(receipt), encoding="utf-8")

    def state_from_outputs(self, handle: RunHandle) -> RunState:
        """The state of the run's verified local copy of its outputs.

        Args:
            handle: The run.

        Returns:
            Its state.

        Raises:
            RuntimeError: If no verified copy exists, or it records an unfinished job.
        """
        receipt = self._runs / handle.run_id / _RECOVERED
        if not receipt.is_file():
            msg = f"{handle.run_id} has no live status and no verified copy of its outputs"
            raise RuntimeError(msg)
        recovered = json.loads(receipt.read_text(encoding="utf-8"))
        root = Path(recovered["path"])
        sky_transfer.verify(root, recovered["manifest"])
        state = _finished_state(_read(root / MANIFEST_NAME))
        if state is None:
            msg = f"the verified copy of {handle.run_id} records an unfinished job"
            raise RuntimeError(msg)
        return state

    def _remote_manifest(self, cluster: str, remote: str) -> sky_transfer.Manifest:
        program = [*shlex.split(self._worker_python), "-m", "substrax.compute.sky_transfer", remote]
        command = f"{UV_ON_PATH}; cd ~/sky_workdir && {shlex.join(program)}"
        return json.loads(run_tool(["ssh", cluster, command]))


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
