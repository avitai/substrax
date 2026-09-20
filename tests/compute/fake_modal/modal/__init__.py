"""A stand-in for the parts of Modal the ``modal`` backend uses, running everything locally.

``Function.spawn`` starts the function in a child interpreter whose arguments naming container
paths (the project copied into the image, a volume's mount point) are replaced by local
directories. Volumes are directories under ``$FAKE_MODAL_ROOT/volumes``, an app's log is
``$FAKE_MODAL_ROOT/apps/<app_id>.log`` and a call's record is
``$FAKE_MODAL_ROOT/calls/<call_id>.json``, so a second backend instance finds a run from its
handle, as it would on Modal. The tests put this package first on ``sys.path`` and ``PYTHONPATH``.
"""

from __future__ import annotations

import json
import os
import secrets
import signal
import sys
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from pathlib import Path

from . import exception, types, volume
from .types import FileEntry, FileEntryType


ROOT_ENV = "FAKE_MODAL_ROOT"
LAST: dict[str, object] = {}
"""The last image, function and spawn arguments, for tests that inspect what was built."""


def _root() -> Path:
    return Path(os.environ[ROOT_ENV])


class Image:
    """Records the build steps; ``add_local_dir`` says where the project lives locally."""

    def __init__(self, steps: tuple[tuple[str, object], ...], local_dirs: dict[str, str]) -> None:
        self.steps = steps
        self.local_dirs = local_dirs

    @classmethod
    def debian_slim(cls, python_version: str | None = None) -> Image:
        return cls((("debian_slim", python_version),), {})

    def uv_sync(self, **options: object) -> Image:
        return Image((*self.steps, ("uv_sync", options)), self.local_dirs)

    def add_local_dir(self, local_path: str | Path, remote_path: str, **options: object) -> Image:
        dirs = {**self.local_dirs, remote_path: str(local_path)}
        return Image(
            (*self.steps, ("add_local_dir", {"remote_path": remote_path, **options})), dirs
        )

    def run_commands(self, *commands: str) -> Image:
        return Image((*self.steps, ("run_commands", commands)), self.local_dirs)


class Volume:
    """A directory under ``$FAKE_MODAL_ROOT/volumes``."""

    def __init__(self, name: str, *, read_only: bool = False) -> None:
        self.name = name
        self.is_read_only = read_only
        self.path = _root() / "volumes" / name

    @classmethod
    def from_name(
        cls, name: str, *, environment_name: str | None = None, create_if_missing: bool = False
    ) -> Volume:
        del environment_name
        volume = cls(name)
        if create_if_missing:
            volume.path.mkdir(parents=True, exist_ok=True)
        return volume

    def read_only(self) -> Volume:
        return Volume(self.name, read_only=True)

    def commit(self) -> None:
        """Nothing to do: the directory is the volume."""

    def iterdir(self, path: str, *, recursive: bool = True) -> Iterator[FileEntry]:
        base = self.path / path
        found = base.rglob("*") if recursive else base.iterdir()
        for item in sorted(found):
            kind = FileEntryType.FILE if item.is_file() else FileEntryType.DIRECTORY
            yield FileEntry(path=item.relative_to(self.path).as_posix(), type=kind)

    def read_file(self, path: str) -> Iterator[bytes]:
        target = self.path / path
        if not target.is_file():
            raise exception.NotFoundError(path)
        yield target.read_bytes()


class FunctionCall:
    """A call whose child interpreter's pid and result live under ``$FAKE_MODAL_ROOT/calls``."""

    def __init__(self, object_id: str) -> None:
        self.object_id = object_id
        self.record = _root() / "calls" / f"{object_id}.json"

    @classmethod
    def from_id(cls, function_call_id: str) -> FunctionCall:
        return cls(function_call_id)

    def get(self, timeout: float | None = None) -> object:
        del timeout
        result = self.record.with_suffix(".result")
        if not result.exists():
            raise TimeoutError
        outcome = json.loads(result.read_text())
        if not outcome["ok"]:
            raise exception.ExecutionError(outcome["error"])
        return outcome["value"]

    def cancel(self, terminate_containers: bool = False) -> None:
        del terminate_containers
        pid = json.loads(self.record.read_text())["pid"]
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            return


class Function:
    def __init__(self, function: Callable[..., object], options: dict[str, object]) -> None:
        self.function = function
        self.options = options

    def spawn(self, *args: str) -> FunctionCall:
        image = self.options["image"]
        assert isinstance(image, Image)
        volumes = self.options.get("volumes") or {}
        assert isinstance(volumes, dict)
        local = {**image.local_dirs, **{path: str(v.path) for path, v in volumes.items()}}
        arguments = [local.get(argument, argument) for argument in args]
        LAST["spawn"] = args
        call = FunctionCall(f"fc-{secrets.token_hex(4)}")
        call.record.parent.mkdir(parents=True, exist_ok=True)
        app = self.options["app"]
        assert isinstance(app, App)
        log = _root() / "apps" / f"{app.app_id}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        program = (
            "import json, sys\n"
            f"from {self.function.__module__} import {self.function.__qualname__} as run\n"
            "arguments, result = json.loads(sys.argv[1]), sys.argv[2]\n"
            "try:\n"
            "    outcome = {'ok': True, 'value': run(*arguments)}\n"
            "except Exception as error:\n"
            "    outcome = {'ok': False, 'error': repr(error)}\n"
            "open(result, 'w').write(json.dumps(outcome))\n"
        )
        argv = [
            sys.executable,
            "-c",
            program,
            json.dumps(arguments),
            str(call.record.with_suffix(".result")),
        ]
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        pid = os.posix_spawn(
            sys.executable,
            argv,
            dict(os.environ),
            file_actions=[
                (os.POSIX_SPAWN_OPEN, 1, str(log), flags, 0o644),
                (os.POSIX_SPAWN_DUP2, 1, 2),
            ],
            setpgroup=0,
        )
        call.record.write_text(json.dumps({"pid": pid}))
        return call


class App:
    def __init__(self, name: str | None = None) -> None:
        self.name = name
        self.app_id: str | None = None

    def function(self, **options: object) -> Callable[[Callable[..., object]], Function]:
        def register(function: Callable[..., object]) -> Function:
            created = Function(function, {**options, "app": self})
            LAST["function"] = created
            return created

        return register

    @contextmanager
    def run(
        self, *, detach: bool = False, environment_name: str | None = None
    ) -> Generator[App, None, None]:
        del detach, environment_name
        self.app_id = f"ap-{secrets.token_hex(4)}"
        LAST["app"] = self
        yield self


__all__ = ["App", "FunctionCall", "Image", "Volume", "exception", "types", "volume"]
