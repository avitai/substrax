"""``substrax-compute``: run a project's configured jobs on a compute backend and follow them.

```text
substrax-compute run JOB [--backend NAME] [--accelerator KIND[:COUNT]] [--project DIR]
                         [--detach] [--into DIR]
substrax-compute status [RUN_ID]
substrax-compute logs RUN_ID [--follow]
substrax-compute fetch RUN_ID [--into DIR]
substrax-compute cancel RUN_ID
```

``run`` submits the job, follows its log, waits for it and fetches its outputs; with ``--detach``
it returns once the job is submitted, and the other commands find the run again by the id it
prints. Fetched outputs go to ``--into``, or under the state directory, never into the project.
Exit codes: 0 for success, 1 for a run that did not succeed, 2 for a refused command.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from substrax.compute.backend import (
    ComputeBackend,
    default_state_dir,
    RunHandle,
    RunState,
    RunStore,
)
from substrax.compute.config import (
    ComputeConfig,
    parse_accelerator,
    read_compute_config,
    require_locked_substrax,
    resolve_job,
)
from substrax.compute.registry import load_backend


_SUCCESS, _UNSUCCESSFUL, _REFUSED = 0, 1, 2


def main(argv: Sequence[str] | None = None) -> int:
    """Run one ``substrax-compute`` command.

    Args:
        argv: The arguments after the program name; the process's when ``None``.

    Returns:
        The exit code.
    """
    arguments = _parser().parse_args(argv)
    store = RunStore()
    try:
        return arguments.command(arguments, store)
    except (LookupError, FileNotFoundError, ValueError) as error:
        message = error.args[0] if isinstance(error, KeyError) and error.args else str(error)
        sys.stderr.write(f"substrax-compute: {message}\n")
        return _REFUSED


def _run(arguments: argparse.Namespace, store: RunStore) -> int:
    project = arguments.project.resolve()
    config = read_compute_config(project)
    require_locked_substrax(project)
    accelerator = parse_accelerator(arguments.accelerator) if arguments.accelerator else None
    spec = resolve_job(config, arguments.job, project=project, accelerator=accelerator)
    name = arguments.backend or config.backend
    if name is None:
        msg = "no backend: pass --backend or set backend in [tool.substrax.compute]"
        raise LookupError(msg)
    backend = load_backend(name, config.backends.get(name, {}))
    handle = backend.submit(spec, project=project)
    store.save(handle)
    sys.stdout.write(f"Submitted {handle.run_id} to {name}.\n")
    if arguments.detach:
        return _SUCCESS
    for line in backend.logs(handle, follow=True):
        sys.stdout.write(line + "\n")
    return _finish(backend, handle, arguments.into)


def _status(arguments: argparse.Namespace, store: RunStore) -> int:
    handles = [store.load(arguments.run_id)] if arguments.run_id else store.handles()
    for handle in handles:
        state = _backend_for(handle).status(handle)
        sys.stdout.write(f"{handle.run_id}  {handle.backend}  {handle.job}  {state}\n")
    return _SUCCESS


def _logs(arguments: argparse.Namespace, store: RunStore) -> int:
    handle = store.load(arguments.run_id)
    for line in _backend_for(handle).logs(handle, follow=arguments.follow):
        sys.stdout.write(line + "\n")
    return _SUCCESS


def _fetch(arguments: argparse.Namespace, store: RunStore) -> int:
    handle = store.load(arguments.run_id)
    target = _backend_for(handle).fetch(handle, _destination(arguments.into))
    sys.stdout.write(f"Fetched {handle.run_id} to {target}.\n")
    return _SUCCESS


def _cancel(arguments: argparse.Namespace, store: RunStore) -> int:
    handle = store.load(arguments.run_id)
    _backend_for(handle).cancel(handle)
    sys.stdout.write(f"Cancelled {handle.run_id}.\n")
    return _SUCCESS


def _finish(backend: ComputeBackend, handle: RunHandle, into: Path | None) -> int:
    state = backend.status(handle)
    target = backend.fetch(handle, _destination(into))
    sys.stdout.write(f"{handle.run_id} {state}; outputs in {target}.\n")
    return _SUCCESS if state is RunState.SUCCEEDED else _UNSUCCESSFUL


def _backend_for(handle: RunHandle) -> ComputeBackend:
    """Rebuild the backend of a stored run with the settings its project configures now."""
    project = Path(handle.project)
    config = (
        read_compute_config(project) if (project / "pyproject.toml").is_file() else ComputeConfig()
    )
    return load_backend(handle.backend, config.backends.get(handle.backend, {}))


def _destination(into: Path | None) -> Path:
    return into if into is not None else default_state_dir() / "compute" / "fetched"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="substrax-compute", description="Run a project's jobs on a compute backend."
    )
    commands = parser.add_subparsers(required=True)

    run = commands.add_parser("run", help="submit a configured job")
    run.add_argument("job", help="the job's name in [tool.substrax.compute.jobs]")
    run.add_argument("--backend", help="the backend; the configured default when omitted")
    run.add_argument("--accelerator", help="KIND or KIND:COUNT, replacing the configured one")
    run.add_argument("--project", type=Path, default=Path(), help="the project root")
    run.add_argument("--detach", action="store_true", help="return once submitted")
    run.add_argument("--into", type=Path, help="where to fetch the outputs")
    run.set_defaults(command=_run)

    status = commands.add_parser("status", help="show one run, or every stored run")
    status.add_argument("run_id", nargs="?")
    status.set_defaults(command=_status)

    logs = commands.add_parser("logs", help="print a run's log")
    logs.add_argument("run_id")
    logs.add_argument("--follow", action="store_true", help="keep printing until the run ends")
    logs.set_defaults(command=_logs)

    fetch = commands.add_parser("fetch", help="copy a run's outputs")
    fetch.add_argument("run_id")
    fetch.add_argument("--into", type=Path, help="where to copy them")
    fetch.set_defaults(command=_fetch)

    cancel = commands.add_parser("cancel", help="stop a run")
    cancel.add_argument("run_id")
    cancel.set_defaults(command=_cancel)
    return parser


if __name__ == "__main__":
    sys.exit(main())
