"""The ``modal`` backend: runs a job on Modal (https://modal.com); needs ``substrax[modal]``.

The image is built in layers keyed on what they depend on, so editing the project's source rebuilds
only the last two:

1. ``Image.uv_sync`` installs the locked dependencies and the job's extras, from ``pyproject.toml``
   and ``uv.lock`` alone, on this interpreter's Python version, which the serialized worker
   needs;
2. the project's source is copied in;
3. the project itself is installed into that environment without dependencies.

``Image.uv_sync`` needs a Modal image builder of 2025.06 or later, which is a workspace setting.

The worker runs in a Modal function, detached from the submitting shell, with the outputs volume
mounted at ``/outputs`` and each of the job's mounts at its path. Logs come from ``modal app logs``,
the provider's own command, run by this interpreter.

Settings (``[tool.substrax.compute.backends.modal]``):

- ``outputs_volume`` (required): the Modal Volume the runs' outputs go to, created when missing.
- ``volumes``: the Modal Volume behind each of a job's mounts, by mount name.
- ``environment``: the Modal environment; the profile's default when unset.
- ``ignore``: patterns of project files left out of the image, ``.dockerignore`` style.
- ``state_dir``: where cancellations are recorded; the default state directory when unset.
"""

from __future__ import annotations

import json
import math
import queue
import subprocess  # nosec B404 - runs `python -m modal app logs`, no shell
import sys
import threading
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath


try:
    import modal
    import modal.exception
    import modal.types
except ImportError as error:  # pragma: no cover - exercised only without the extra
    msg = "the modal backend needs Modal; install it with `uv add 'substrax[modal]'`"
    raise ImportError(msg) from error

from substrax.compute.backend import backend_state_dir, new_run, RunHandle, RunState
from substrax.compute.spec import Accelerator, JobSpec, MountAccess, read_job_spec
from substrax.compute.worker import JobManifest, MANIFEST_NAME, read_manifest, run_job
from substrax.records import dump_record, read_record, UNKNOWN_FIELDS_REFUSED
from substrax.typing import JsonValue


NAME = "modal"
PROJECT_PATH = "/project"
OUTPUTS_PATH = "/outputs"
_UV = "/.uv/uv"
# The worker is sent to Modal serialized, which needs the image's Python to be this one. It is
# also named to `uv sync`, so a project's own uv settings (such as `python-preference`) cannot
# pick another.
_PYTHON = f"{sys.version_info.major}.{sys.version_info.minor}"
_VENV_PYTHON = "/.uv/.venv/bin/python"
_DEFAULT_IGNORE = (
    ".git",
    ".venv",
    "**/__pycache__",
    "**/*.pyc",
    ".pytest_cache",
    ".ruff_cache",
    "build",
    "dist",
    "site",
    "temp",
)
_DRAIN_SECONDS = 5.0
_POLL_SECONDS = 2.0


@dataclass(frozen=True, slots=True, kw_only=True)
class ModalSettings:
    """The ``modal`` backend's settings; see the module documentation.

    Attributes:
        outputs_volume: The Volume the runs' outputs go to.
        volumes: The Volume behind each mount, by mount name.
        environment: The Modal environment; the profile's default when ``None``.
        ignore: Project files left out of the image.
        state_dir: Where cancellations are recorded.
    """

    __pydantic_config__ = UNKNOWN_FIELDS_REFUSED

    outputs_volume: str
    volumes: Mapping[str, str] = field(default_factory=dict)
    environment: str | None = None
    ignore: tuple[str, ...] = _DEFAULT_IGNORE
    state_dir: str | None = None


def run_worker(spec_json: str, run_id: str, project: str, outputs: str, volume: str) -> bool:
    """Run the job inside the Modal function and commit its outputs.

    Args:
        spec_json: The job spec as JSON text.
        run_id: The run, whose directory under ``outputs`` the files go to.
        project: The project's path in the container.
        outputs: The outputs volume's path in the container.
        volume: The outputs volume's name.

    Returns:
        Whether every task succeeded.
    """
    spec = read_job_spec(json.loads(spec_json))
    manifest = run_job(spec, project=Path(project), outputs=Path(outputs) / run_id)
    modal.Volume.from_name(volume).commit()
    return manifest.succeeded


class ModalBackend:
    """Runs jobs on Modal."""

    name = NAME

    def __init__(self, settings: Mapping[str, JsonValue]) -> None:  # noqa: DOC502  # pydantic.ValidationError, a ValueError, is raised by read_record
        """Read the backend's settings.

        Args:
            settings: The settings table; ``outputs_volume`` is required.

        Raises:
            ValueError: If a setting is missing, unknown or of the wrong type.
        """
        self._settings = read_record(ModalSettings, settings)
        self._cancelled = backend_state_dir(self._settings.state_dir, NAME) / "cancelled"

    def submit(self, spec: JobSpec, *, project: Path) -> RunHandle:
        """Build the image and spawn the worker in a detached app.

        Args:
            spec: The job.
            project: The project root.

        Returns:
            The run's handle.
        """
        run = new_run(spec, project)
        root, run_id = run.project, run.run_id
        app = modal.App(f"{root.name}-{spec.name}")
        worker = app.function(
            image=self._image(spec, root),
            gpu=_gpu(spec.accelerator),
            timeout=math.ceil(spec.timeout_seconds),
            volumes=self._volumes(spec),
            serialized=True,
        )(run_worker)
        with app.run(detach=True, environment_name=self._settings.environment):
            call = worker.spawn(
                json.dumps(dump_record(spec)),
                run_id,
                PROJECT_PATH,
                OUTPUTS_PATH,
                self._settings.outputs_volume,
            )
        return run.handle(NAME, {"app_id": app.app_id or "", "call_id": call.object_id})

    def status(self, handle: RunHandle) -> RunState:
        """Where the run is: running while its call has no result, then from its manifest.

        Args:
            handle: The run.

        Returns:
            Its state.
        """
        if (self._cancelled / handle.run_id).exists():
            return RunState.CANCELLED
        if _is_running(modal.FunctionCall.from_id(handle.details["call_id"])):
            return RunState.RUNNING
        manifest = self._manifest(handle)
        if manifest is not None and manifest.finished:
            return RunState.SUCCEEDED if manifest.succeeded else RunState.FAILED
        return RunState.FAILED

    def logs(self, handle: RunHandle, *, follow: bool) -> Iterator[str]:
        """The app's log lines, from ``modal app logs``.

        Args:
            handle: The run.
            follow: Keep reading until the run ends.

        Yields:
            str: Each line, without its line ending.
        """
        command = [sys.executable, "-m", "modal", "app", "logs", handle.details["app_id"]]
        if self._settings.environment:
            command += ["--env", self._settings.environment]
        if not follow:
            since = ["--since", handle.submitted_at.isoformat()]
            completed = subprocess.run(  # noqa: S603  # nosec B603 - this interpreter, no shell
                command + since, capture_output=True, text=True, check=True
            )
            yield from completed.stdout.splitlines()
            return
        yield from self._follow(handle, [*command, "--follow"])

    def fetch(self, handle: RunHandle, destination: Path) -> Path:
        """Copy the run's directory of the outputs volume to ``destination / handle.run_id``.

        Args:
            handle: The run.
            destination: The directory to copy into.

        Returns:
            The copy.
        """
        volume = self._outputs_volume()
        target = destination / handle.run_id
        for entry in volume.iterdir(handle.run_id, recursive=True):
            if entry.type != modal.types.FileEntryType.FILE:
                continue
            local = target / Path(entry.path).relative_to(handle.run_id)
            local.parent.mkdir(parents=True, exist_ok=True)
            with local.open("wb") as copy:
                for chunk in volume.read_file(entry.path):
                    copy.write(chunk)
        return target

    def cancel(self, handle: RunHandle) -> None:
        """Cancel the call and stop its container; a run that already ended is left as it is.

        Args:
            handle: The run.
        """
        if self.status(handle).is_final:
            return
        self._cancelled.mkdir(parents=True, exist_ok=True)
        (self._cancelled / handle.run_id).touch()
        modal.FunctionCall.from_id(handle.details["call_id"]).cancel(terminate_containers=True)

    def _image(self, spec: JobSpec, project: Path) -> modal.Image:
        return (
            modal.Image.debian_slim(python_version=_PYTHON)
            .uv_sync(
                uv_project_dir=str(project),
                extras=list(spec.extras),
                frozen=True,
                extra_options=f"--python {_PYTHON}",
            )
            .add_local_dir(
                project, remote_path=PROJECT_PATH, copy=True, ignore=list(self._settings.ignore)
            )
            .run_commands(f"{_UV} pip install --python {_VENV_PYTHON} --no-deps {PROJECT_PATH}")
        )

    def _volumes(
        self, spec: JobSpec
    ) -> dict[str | PurePosixPath, modal.Volume | modal.CloudBucketMount]:
        volumes: dict[str | PurePosixPath, modal.Volume | modal.CloudBucketMount] = {
            OUTPUTS_PATH: self._outputs_volume()
        }
        for mount in spec.mounts:
            if mount.name not in self._settings.volumes:
                msg = f"mount {mount.name!r} has no Modal Volume in the backend's `volumes`"
                raise LookupError(msg)
            volume = modal.Volume.from_name(
                self._settings.volumes[mount.name],
                environment_name=self._settings.environment,
                create_if_missing=True,
            )
            volumes[mount.path] = (
                volume.read_only() if mount.access is MountAccess.READ_ONLY else volume
            )
        return volumes

    def _outputs_volume(self) -> modal.Volume:
        return modal.Volume.from_name(
            self._settings.outputs_volume,
            environment_name=self._settings.environment,
            create_if_missing=True,
        )

    def _manifest(self, handle: RunHandle) -> JobManifest | None:
        path = f"{handle.run_id}/{MANIFEST_NAME}"
        try:
            payload = b"".join(self._outputs_volume().read_file(path))
        except modal.exception.NotFoundError:
            return None
        return read_manifest(json.loads(payload))

    def _follow(self, handle: RunHandle, command: list[str]) -> Iterator[str]:
        """Stream ``modal app logs --follow`` until the run ends, then drain for a few seconds."""
        lines: queue.Queue[str | None] = queue.Queue()
        with subprocess.Popen(  # noqa: S603  # nosec B603 - this interpreter, no shell
            command, stdout=subprocess.PIPE, text=True
        ) as streamer:
            threading.Thread(target=_pump, args=(streamer, lines), daemon=True).start()
            drain_until: float | None = None
            while drain_until is None or time.monotonic() < drain_until:
                try:
                    line = lines.get(timeout=_POLL_SECONDS)
                except queue.Empty:
                    if drain_until is None and self.status(handle).is_final:
                        drain_until = time.monotonic() + _DRAIN_SECONDS
                    continue
                if line is None:
                    return
                yield line
            streamer.terminate()


def _pump(streamer: subprocess.Popen[str], lines: queue.Queue[str | None]) -> None:
    """Copy the streamer's lines into ``lines``, then ``None`` when it closes."""
    if streamer.stdout is not None:
        for line in streamer.stdout:
            lines.put(line.rstrip("\n"))
    lines.put(None)


def _is_running(call: modal.FunctionCall[bool]) -> bool:
    """Whether the call has no result yet; a failed or expired call has ended."""
    try:
        call.get(timeout=0)
    except TimeoutError:
        return True
    except modal.exception.Error:
        # Failed, timed out or expired: the run's manifest tells how far it got.
        return False
    return False


def _gpu(accelerator: Accelerator | None) -> str | None:
    """Modal's GPU request: ``KIND`` or ``KIND:COUNT``."""
    if accelerator is None:
        return None
    return accelerator.kind if accelerator.count == 1 else f"{accelerator.kind}:{accelerator.count}"
