"""The ``skypilot`` backend: runs a job through SkyPilot (https://docs.skypilot.ai) on its clouds.

SkyPilot is a separate tool, installed the way its documentation says, in an environment of its
own (``uv tool install --with pip "skypilot[gcp]"``). The scientific environment never imports it:
every operation runs ``substrax/compute/sky_sdk.py`` in SkyPilot's own interpreter, which calls
SkyPilot's public SDK and reads its native request and queue records.

Each run gets a cluster of its own. The task syncs the project as its workdir, installs the locked
environment (``uv sync --frozen``) and runs the worker. Before it waits for provisioning, a launch
records the run's intent and SkyPilot's request id beside the task file, so an interrupted launch
is reconciled from them rather than submitted again. Cancelling cancels that request and the job,
then tears the cluster down, keeping every failure.

How inputs reach the cluster and outputs come back is the transport's, configured as exactly one
of ``[...skypilot.bucket]`` or ``[...skypilot.copy]``; see :mod:`substrax.compute.skypilot_transports`.

Settings (``[tool.substrax.compute.backends.skypilot]``):

- ``bucket`` or ``copy``: the transport's table.
- ``setup_commands``: commands that prepare the image before the locked environment is synced.
- ``infra``: where to run, in SkyPilot's ``infra`` syntax, such as ``gcp`` or ``runpod``; any
  enabled cloud when unset.
- ``use_spot``: run on spot instances.
- ``sky``: the ``sky`` command, used to find SkyPilot's interpreter; ``sky`` on ``PATH``.
- ``sky_python``: SkyPilot's interpreter, when it is not beside the ``sky`` command.
- ``worker_python``: the command that starts Python in the synced environment;
  ``uv run --no-sync python`` by default.
- ``state_dir``: where task files, launch records and cancellations are kept; the default state
  directory when unset.
"""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess  # nosec B404 - runs SkyPilot's interpreter on the bridge, no shell
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from substrax.compute import skypilot_transports as transports
from substrax.compute.backend import (
    backend_state_dir,
    new_run,
    RunHandle,
    RunState,
    SPEC_NAME,
    stage_run,
)
from substrax.compute.skypilot_transports import (
    BucketSettings,
    BucketTransport,
    CopySettings,
    CopyTransport,
    OUTPUTS_PATH,
    SPEC_PATH,
    Transport,
    UV_ON_PATH,
)
from substrax.compute.spec import Accelerator, JobSpec
from substrax.records import dump_record, read_record, UNKNOWN_FIELDS_REFUSED
from substrax.typing import JsonValue


NAME = "skypilot"
_JOB_ID = "1"  # the first and only job on a cluster made for the run
_STATES = {
    "INIT": RunState.PENDING,
    "PENDING": RunState.PENDING,
    "SETTING_UP": RunState.PENDING,
    "RUNNING": RunState.RUNNING,
    "SUCCEEDED": RunState.SUCCEEDED,
    "CANCELLED": RunState.CANCELLED,
    "FAILED": RunState.FAILED,
    "FAILED_SETUP": RunState.FAILED,
    "FAILED_DRIVER": RunState.FAILED,
}
_CLEANUP_ERRORS = (RuntimeError, OSError, ValueError, KeyError)


@dataclass(frozen=True, slots=True, kw_only=True)
class SkyPilotSettings:
    """The ``skypilot`` backend's settings; see the module documentation.

    Attributes:
        bucket: Bucket transport's settings; set exactly one of ``bucket`` and ``copy``.
        copy: Copy transport's settings.
        setup_commands: Commands run before the locked environment is synced.
        infra: Where to run, in SkyPilot's ``infra`` syntax.
        use_spot: Run on spot instances.
        sky: The ``sky`` command, used to find SkyPilot's interpreter.
        sky_python: SkyPilot's interpreter; the ``python`` beside ``sky`` when unset.
        worker_python: The command that starts Python in the synced environment.
        state_dir: Where task files, launch records and cancellations are kept.
    """

    __pydantic_config__ = UNKNOWN_FIELDS_REFUSED

    bucket: BucketSettings | None = None
    copy: CopySettings | None = None
    setup_commands: tuple[str, ...] = ()
    infra: str | None = None
    use_spot: bool = False
    sky: str = "sky"
    sky_python: str | None = None
    worker_python: str = "uv run --no-sync python"
    state_dir: str | None = None


class SkyPilotBackend:
    """Runs jobs through SkyPilot's public SDK, in SkyPilot's own interpreter."""

    name = NAME

    def __init__(self, settings: Mapping[str, JsonValue]) -> None:  # noqa: DOC502  # pydantic.ValidationError, a ValueError, is raised by read_record
        """Read the backend's settings.

        Args:
            settings: The settings table, with exactly one of ``bucket`` and ``copy``.

        Raises:
            ValueError: If a setting is missing, unknown or of the wrong type, or the transport
                is not exactly one of ``bucket`` and ``copy``, or its own checks fail.
        """
        chosen = read_record(SkyPilotSettings, settings)
        self._runs = backend_state_dir(chosen.state_dir, NAME).resolve()
        self._settings = replace(chosen, state_dir=str(self._runs.parent.parent))
        self._transport = _transport(self._settings, self._runs)

    def submit(self, spec: JobSpec, *, project: Path) -> RunHandle:
        """Write the task file, record the launch, and launch it on a cluster of its own.

        Args:
            spec: The job.
            project: The project root.

        Returns:
            The run's handle, once the job is submitted on the cluster.
        """
        run = new_run(spec, project, settings=dump_record(self._settings))
        run_dir = stage_run(self._runs, run)
        task = run_dir / "task.yaml"  # YAML is a superset of JSON
        content = self.task_for(spec, project=run.project, run_id=run.run_id, run_dir=run_dir)
        task.write_text(json.dumps(content, indent=2), "utf-8")
        details = {"cluster": _cluster_name(run.run_id), "job_id": _JOB_ID}
        intent = dump_record(run.handle(NAME, details))
        (run_dir / "intent.json").write_text(json.dumps(intent), encoding="utf-8")
        launch = [details["cluster"], str(task), str(run_dir / "request.json")]
        down = ["--down"] if self._transport.down_with_job else []
        launched = json.loads(self._sdk("launch", *launch, *down))
        return run.handle(
            NAME, {**details, "job_id": launched["job_id"], "request_id": launched["request_id"]}
        )

    def status(self, handle: RunHandle) -> RunState:  # noqa: DOC503  # the ValueError is raised by _state
        """The job's state from the cluster's queue while it is up, else from its outputs.

        Args:
            handle: The run.

        Returns:
            Its state.

        Raises:
            RuntimeError: If the queue cannot be read and the outputs cannot answer for the run.
            ValueError: If SkyPilot reports a job status this backend does not know.
        """
        if (self._runs / handle.run_id / "cancelled").exists():
            return RunState.CANCELLED
        try:
            jobs = json.loads(self._sdk("queue", handle.details["cluster"]))
        except RuntimeError:
            if not self._transport.recovers_without_queue:
                raise
            return self._transport.state_from_outputs(handle)
        for job in jobs:
            if str(job["job_id"]) == handle.details["job_id"]:
                return _state(str(job["status"]))
        return self._transport.state_from_outputs(handle)

    def logs(self, handle: RunHandle, *, follow: bool) -> Iterator[str]:
        """The job's output lines, from SkyPilot's log stream.

        Args:
            handle: The run.
            follow: Keep reading until the job ends.

        Yields:
            str: Each line, without its line ending.

        Raises:
            RuntimeError: If the log stream fails.
        """
        command = [*self._bridge(), "logs", handle.details["cluster"], handle.details["job_id"]]
        with subprocess.Popen(  # noqa: S603  # nosec B603 - SkyPilot's interpreter, no shell
            [*command, *(["--follow"] if follow else [])], stdout=subprocess.PIPE, text=True
        ) as streamer:
            if streamer.stdout is not None:
                yield from (line.rstrip("\n") for line in streamer.stdout)
            if streamer.wait() != 0:
                msg = f"the log stream of {handle.run_id} failed"
                raise RuntimeError(msg)

    def fetch(self, handle: RunHandle, destination: Path) -> Path:
        """Copy the run's outputs to ``destination / handle.run_id``.

        Args:
            handle: The run.
            destination: The directory to copy into.

        Returns:
            The copy.
        """
        target = destination / handle.run_id
        target.mkdir(parents=True, exist_ok=True)
        self._transport.fetch(handle, target)
        return target

    def cancel(self, handle: RunHandle) -> None:
        """Cancel unfinished work and tear its cluster down; a run that already ended is left.

        Every step is attempted, whatever the one before it did.

        Args:
            handle: The run.

        Raises:
            ExceptionGroup: If any step failed; it holds each failure.
        """
        errors: list[Exception] = []
        state = _attempt(errors, self.status, handle)
        if state is not None and state.is_final:
            return
        _attempt(errors, self._cancel_request, handle)
        _attempt(errors, self._cancel_job, handle)
        _attempt(errors, self.teardown, handle)
        if errors:
            msg = f"the cleanup of {handle.run_id} is incomplete"
            raise ExceptionGroup(msg, errors)
        if state is not None:
            (self._runs / handle.run_id / "cancelled").touch()

    def teardown(self, handle: RunHandle) -> None:
        """Remove the run's cluster, and with it anything left on it.

        Unlike :meth:`cancel`, this removes the cluster of a run that already ended too; fetch a
        copy transport's outputs first.

        Args:
            handle: The run.
        """
        self._sdk("down", handle.details["cluster"])

    def task_for(
        self, spec: JobSpec, *, project: Path, run_id: str, run_dir: Path
    ) -> dict[str, JsonValue]:
        """The SkyPilot task that runs ``spec``, as its task file holds it.

        Args:
            spec: The job.
            project: The project root, synced as the task's workdir.
            run_id: The run.
            run_dir: The run's staged directory, which holds its spec.

        Returns:
            The task.
        """
        resources = _resources(spec, self._settings)
        mounts: dict[str, JsonValue] = {SPEC_PATH: str(run_dir / SPEC_NAME)}
        self._transport.add_to_task(spec, resources, mounts)
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
        extras = " ".join(f"--extra {shlex.quote(extra)}" for extra in spec.extras)
        setup = [
            "set -e",
            *self._settings.setup_commands,
            f"mkdir -p {OUTPUTS_PATH}",
            "command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh",
            UV_ON_PATH,
            f"uv sync --frozen {extras}".rstrip(),
        ]
        run = [
            UV_ON_PATH,
            "export PYTHONDONTWRITEBYTECODE=1",
            self._transport.wrap_worker(worker, spec),
        ]
        return {
            "name": _cluster_name(run_id),
            "workdir": str(project),
            "resources": resources,
            "file_mounts": mounts,
            "setup": "\n".join(setup),
            "run": "\n".join(run),
        }

    def _cancel_request(self, handle: RunHandle) -> None:
        """Cancel the launch's request with SkyPilot's API server, when one was recorded."""
        request_id = handle.details.get("request_id")
        receipt = self._runs / handle.run_id / "request.json"
        if request_id is None and receipt.exists():
            request_id = json.loads(receipt.read_text(encoding="utf-8"))["request_id"]
        if request_id is not None:
            self._sdk("cancel-request", request_id)

    def _cancel_job(self, handle: RunHandle) -> None:
        self._sdk("cancel-job", handle.details["cluster"], handle.details["job_id"])

    def _bridge(self) -> list[str]:
        """The command that runs ``sky_sdk.py`` in SkyPilot's interpreter."""
        python = self._settings.sky_python
        if python is None:
            sky = shutil.which(shlex.split(self._settings.sky)[0])
            if sky is None:
                msg = f"{self._settings.sky!r} is not on PATH; set `sky` or `sky_python`"
                raise FileNotFoundError(msg)
            python = str(Path(sky).resolve().with_name("python"))
        return [python, str(Path(__file__).with_name("sky_sdk.py"))]

    def _sdk(self, *arguments: str) -> str:
        return transports.run_tool([*self._bridge(), *arguments])


def _transport(settings: SkyPilotSettings, runs: Path) -> Transport:
    """The configured transport; exactly one of ``bucket`` and ``copy`` is set."""
    if settings.bucket is not None and settings.copy is None:
        return BucketTransport(settings.bucket, infra=settings.infra)
    if settings.copy is not None and settings.bucket is None:
        return CopyTransport(settings.copy, worker_python=settings.worker_python, runs=runs)
    msg = "configure exactly one transport: [...skypilot.bucket] or [...skypilot.copy]"
    raise ValueError(msg)


def _resources(spec: JobSpec, settings: SkyPilotSettings) -> dict[str, JsonValue]:
    resources: dict[str, JsonValue] = {"use_spot": settings.use_spot}
    requested = spec.resources
    if requested.accelerator is not None:
        resources["accelerators"] = _accelerators(requested.accelerator)
    # SkyPilot reads "<n>+" as at least n and the unit "MB" as 2**20 bytes
    # (sky/resources.py, sky/utils/resources_utils.py parse_memory_resource).
    if requested.cpu is not None:
        resources["cpus"] = f"{requested.cpu:g}+"
    if requested.memory_mib is not None:
        resources["memory"] = f"{requested.memory_mib}MB+"
    if settings.infra is not None:
        resources["infra"] = settings.infra
    return resources


def _state(status: str) -> RunState:
    """A SkyPilot job status as a run state; an unknown one is an error, never a failed job."""
    if status not in _STATES:
        msg = f"Unknown SkyPilot job status: {status}"
        raise ValueError(msg)
    return _STATES[status]


def _attempt[T](
    errors: list[Exception], operation: Callable[..., T], *arguments: object
) -> T | None:
    """Run one cleanup step, keeping its failure instead of stopping the ones after it."""
    try:
        return operation(*arguments)
    except _CLEANUP_ERRORS as error:
        errors.append(error)
        return None


def _cluster_name(run_id: str) -> str:
    """A cluster name the clouds accept: lower case letters, digits and hyphens."""
    return "sx-" + "".join(c if c.isalnum() else "-" for c in run_id.lower())


def _accelerators(accelerator: Accelerator) -> str:
    return f"{accelerator.kind}:{accelerator.count}"
