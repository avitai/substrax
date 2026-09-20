"""The ``skypilot`` backend: runs a job through SkyPilot (https://docs.skypilot.ai) on its clouds.

SkyPilot is a separate tool, installed the way its documentation says, in an environment of its
own (``uv tool install --with pip "skypilot[gcp]"``). This backend never imports it. It writes a
SkyPilot task file and runs the ``sky`` command:

- ``sky launch -c <cluster> -d --down -y`` provisions a cluster for the run. The cluster syncs the
  project as its workdir and installs the locked environment (``uv sync --frozen``). It runs the
  worker, then tears itself down.
- ``sky queue -o json`` gives the job's state while the cluster is up.
- ``sky logs`` streams its output, and ``sky cancel`` then ``sky down`` stop it.

The outputs go to a bucket mounted at ``/outputs``, so they outlive the cluster. They are fetched
with the bucket's own command line tool: ``gcloud storage`` for ``gs://``, ``aws s3`` for ``s3://``.

Settings (``[tool.substrax.compute.backends.skypilot]``):

- ``outputs_bucket`` (required): the bucket URI the runs' outputs go to, such as
  ``gs://avitai-runs``.
- ``buckets``: the bucket URI behind each of a job's mounts, by mount name.
- ``infra``: where to run, in SkyPilot's ``infra`` syntax, such as ``gcp`` or ``gcp/us-central1``;
  any enabled cloud when unset.
- ``use_spot``: run on spot instances.
- ``sky``: the ``sky`` command; ``sky`` on ``PATH`` by default.
- ``worker_python``: the command that starts Python in the synced environment;
  ``uv run --no-sync python`` by default.
- ``state_dir``: where task files and cancellations are kept; the default state directory when
  unset.
"""

from __future__ import annotations

import json
import shlex
import subprocess  # nosec B404 - runs the sky and bucket command line tools, no shell
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from substrax.compute.backend import (
    backend_state_dir,
    new_run,
    RunHandle,
    RunState,
    SPEC_NAME,
    stage_run,
)
from substrax.compute.spec import Accelerator, JobSpec
from substrax.compute.worker import MANIFEST_NAME, read_manifest
from substrax.records import read_record, UNKNOWN_FIELDS_REFUSED
from substrax.typing import JsonValue


NAME = "skypilot"
OUTPUTS_PATH = "/outputs"
SPEC_PATH = "/tmp/substrax-job-spec.json"  # noqa: S108  # nosec B108 - a path on the run's own cluster
_JOB_ID = "1"  # the first and only job on a cluster made for the run
_UV_ON_PATH = 'export PATH="$HOME/.local/bin:$PATH"'
_STATES = {
    "INIT": RunState.PENDING,
    "PENDING": RunState.PENDING,
    "SETTING_UP": RunState.PENDING,
    "RUNNING": RunState.RUNNING,
    "SUCCEEDED": RunState.SUCCEEDED,
    "CANCELLED": RunState.CANCELLED,
}
_DOWNLOADERS = {
    "gs://": ("gcloud", "storage", "rsync", "--recursive"),
    "s3://": ("aws", "s3", "sync"),
}


@dataclass(frozen=True, slots=True, kw_only=True)
class SkyPilotSettings:
    """The ``skypilot`` backend's settings; see the module documentation.

    Attributes:
        outputs_bucket: The bucket URI the runs' outputs go to.
        buckets: The bucket URI behind each mount, by mount name.
        infra: Where to run, in SkyPilot's ``infra`` syntax.
        use_spot: Run on spot instances.
        sky: The ``sky`` command.
        worker_python: The command that starts Python in the synced environment.
        state_dir: Where task files and cancellations are kept.
    """

    __pydantic_config__ = UNKNOWN_FIELDS_REFUSED

    outputs_bucket: str
    buckets: Mapping[str, str] = field(default_factory=dict)
    infra: str | None = None
    use_spot: bool = False
    sky: str = "sky"
    worker_python: str = "uv run --no-sync python"
    state_dir: str | None = None


class SkyPilotBackend:
    """Runs jobs through SkyPilot's ``sky`` command."""

    name = NAME

    def __init__(self, settings: Mapping[str, JsonValue]) -> None:  # noqa: DOC502  # pydantic.ValidationError, a ValueError, is raised by read_record
        """Read the backend's settings.

        Args:
            settings: The settings table; ``outputs_bucket`` is required.

        Raises:
            ValueError: If a setting is missing, unknown or of the wrong type, or the outputs
                bucket's scheme has no known download tool.
        """
        self._settings = read_record(SkyPilotSettings, settings)
        _downloader(self._settings.outputs_bucket)
        self._runs = backend_state_dir(self._settings.state_dir, NAME)

    def submit(self, spec: JobSpec, *, project: Path) -> RunHandle:
        """Write the task file and launch it on a cluster of its own.

        Args:
            spec: The job.
            project: The project root.

        Returns:
            The run's handle, once the job is submitted on the cluster.
        """
        run = new_run(spec, project)
        root, run_id, run_dir = run.project, run.run_id, stage_run(self._runs, run)
        task = run_dir / "task.yaml"  # YAML is a superset of JSON
        task.write_text(json.dumps(self._task(spec, root, run_id, run_dir), indent=2), "utf-8")
        cluster = _cluster_name(run_id)
        self._sky("launch", "-c", cluster, "-d", "--down", "-y", str(task))
        return run.handle(NAME, {"cluster": cluster, "job_id": _JOB_ID})

    def status(self, handle: RunHandle) -> RunState:
        """The job's state from ``sky queue`` while its cluster is up, else from its manifest.

        Args:
            handle: The run.

        Returns:
            Its state.
        """
        if (self._runs / handle.run_id / "cancelled").exists():
            return RunState.CANCELLED
        queue = _json_value(self._sky("queue", handle.details["cluster"], "-o", "json"))
        jobs = queue.get(handle.details["cluster"], []) if isinstance(queue, dict) else []
        for job in jobs if isinstance(jobs, list) else []:
            if isinstance(job, dict) and str(job.get("job_id")) == handle.details["job_id"]:
                return _STATES.get(str(job.get("status")), RunState.FAILED)
        return self._state_from_manifest(handle)

    def logs(self, handle: RunHandle, *, follow: bool) -> Iterator[str]:
        """The job's output lines, from ``sky logs``.

        Args:
            handle: The run.
            follow: Keep reading until the job ends.

        Yields:
            str: Each line, without its line ending.
        """
        flag = "--follow" if follow else "--no-follow"
        command = [*self._command(), "logs", handle.details["cluster"], handle.details["job_id"]]
        with subprocess.Popen(  # noqa: S603  # nosec B603 - the sky command, no shell
            [*command, flag], stdout=subprocess.PIPE, text=True
        ) as streamer:
            if streamer.stdout is not None:
                yield from (line.rstrip("\n") for line in streamer.stdout)

    def fetch(self, handle: RunHandle, destination: Path) -> Path:
        """Download the run's directory of the outputs bucket to ``destination / handle.run_id``.

        Args:
            handle: The run.
            destination: The directory to copy into.

        Returns:
            The copy.
        """
        target = destination / handle.run_id
        target.mkdir(parents=True, exist_ok=True)
        source = f"{self._settings.outputs_bucket.rstrip('/')}/{handle.run_id}"
        _run([*_downloader(source), source, str(target)])
        return target

    def cancel(self, handle: RunHandle) -> None:
        """Cancel the job and tear its cluster down; a run that already ended is left as it is.

        Args:
            handle: The run.
        """
        if self.status(handle).is_final:
            return
        (self._runs / handle.run_id / "cancelled").touch()
        self._sky("cancel", handle.details["cluster"], "-a", "-y")
        self._sky("down", handle.details["cluster"], "-y")

    def _task(
        self, spec: JobSpec, project: Path, run_id: str, run_dir: Path
    ) -> dict[str, JsonValue]:
        resources: dict[str, JsonValue] = {"use_spot": self._settings.use_spot}
        if spec.accelerator is not None:
            resources["accelerators"] = _accelerators(spec.accelerator)
        if self._settings.infra is not None:
            resources["infra"] = self._settings.infra
        mounts: dict[str, JsonValue] = {
            OUTPUTS_PATH: {"source": self._settings.outputs_bucket, "mode": "MOUNT"},
            SPEC_PATH: str(run_dir / SPEC_NAME),
        }
        for mount in spec.mounts:
            if mount.name not in self._settings.buckets:
                msg = f"mount {mount.name!r} has no bucket in the backend's `buckets`"
                raise LookupError(msg)
            mounts[mount.path] = {"source": self._settings.buckets[mount.name], "mode": "MOUNT"}
        extras = " ".join(f"--extra {shlex.quote(extra)}" for extra in spec.extras)
        worker = shlex.join(
            [
                *shlex.split(self._settings.worker_python),
                "-m",
                "substrax.compute.worker",
                "--spec",
                SPEC_PATH,
                "--project",
                ".",
                "--outputs",
                f"{OUTPUTS_PATH}/{run_id}",
            ]
        )
        return {
            "name": _cluster_name(run_id),
            "workdir": str(project),
            "resources": resources,
            "file_mounts": mounts,
            "setup": "\n".join(
                [
                    "set -e",
                    "command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh",
                    _UV_ON_PATH,
                    f"uv sync --frozen {extras}".rstrip(),
                ]
            ),
            "run": "\n".join([_UV_ON_PATH, worker]),
        }

    def _state_from_manifest(self, handle: RunHandle) -> RunState:
        """The state of a run whose cluster is gone, from the manifest it left in the bucket."""
        with tempfile.TemporaryDirectory() as scratch:
            source = f"{self._settings.outputs_bucket.rstrip('/')}/{handle.run_id}"
            completed = subprocess.run(  # noqa: S603  # nosec B603 - the bucket tool, no shell
                [*_downloader(source), source, scratch], capture_output=True, check=False
            )
            manifest_path = Path(scratch) / MANIFEST_NAME
            if completed.returncode != 0 or not manifest_path.is_file():
                return RunState.FAILED
            manifest = read_manifest(json.loads(manifest_path.read_text(encoding="utf-8")))
        if not manifest.finished:
            return RunState.FAILED
        return RunState.SUCCEEDED if manifest.succeeded else RunState.FAILED

    def _command(self) -> list[str]:
        return shlex.split(self._settings.sky)

    def _sky(self, *arguments: str) -> str:
        return _run([*self._command(), *arguments])


def _run(command: Sequence[str]) -> str:
    """Run a command line tool and return its standard output; a failure raises with its stderr."""
    completed = subprocess.run(  # noqa: S603  # nosec B603 - a configured tool, no shell
        list(command), capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        msg = f"{shlex.join(command)} failed ({completed.returncode}): {completed.stderr.strip()}"
        raise RuntimeError(msg)
    return completed.stdout


def _json_value(output: str) -> JsonValue:
    """Decode the JSON a ``sky ... -o json`` command prints after any notices on the same stream."""
    lines = output.splitlines()
    for index, line in enumerate(lines):
        if line.startswith(("{", "[")):
            return json.loads("\n".join(lines[index:]))
    return None


def _downloader(uri: str) -> tuple[str, ...]:
    for scheme, command in _DOWNLOADERS.items():
        if uri.startswith(scheme):
            return command
    msg = f"no download tool for {uri!r}; use a gs:// or s3:// bucket"
    raise ValueError(msg)


def _cluster_name(run_id: str) -> str:
    """A cluster name the clouds accept: lower case letters, digits and hyphens."""
    return "sx-" + "".join(c if c.isalnum() else "-" for c in run_id.lower())


def _accelerators(accelerator: Accelerator) -> str:
    return f"{accelerator.kind}:{accelerator.count}"
