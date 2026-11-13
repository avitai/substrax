"""Run a clean copy of an example as a notebook, from its repository's root, and describe the run.

:func:`execute_example` reads the example script as a notebook and executes it cell by cell with
nbclient, in a kernel on this environment's interpreter (:class:`NativeKernelManager`), from the
repository root, with ``AVITAI_OUTPUT_DIR`` set to a temporary directory for its artifacts. A run
that writes a file the repository tracks, or an untracked file it does not ignore, fails; ignored
paths, such as ``__pycache__``, are not checked. The committed notebook is never touched. The run
is described by its
:class:`Provenance`: the script's sha256, the measured duration and device, and the kinds of output
each code cell produced, which do not depend on the machine.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import sys
import tempfile
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jupytext
import nbformat
from jupyter_client.kernelspec import KernelSpecManager, NATIVE_KERNEL_NAME
from jupyter_client.manager import AsyncKernelManager
from jupyter_core.paths import jupyter_runtime_dir
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError, CellTimeoutError, DeadKernelError
from traitlets import default

from substrax.notebooks import kernel_setup, repository
from substrax.notebooks.config import refuse_unusable_budget
from substrax.notebooks.terminal import strip_terminal_controls
from substrax.typing import JsonValue


_SETUP_SOURCE = inspect.getsource(kernel_setup)
"""The kernel setup's source, which a hidden cell runs in a namespace of its own: the kernel never
imports substrax, so the example sees a script run's modules."""


_SOCKET_PATH_LIMIT = 107
"""The longest path a Unix socket binds: ``sockaddr_un.sun_path`` holds 108 bytes, NUL included."""
_SOCKET_SUFFIX = "-99"
"""Room for the ``-<N>`` jupyter_client adds to the socket path for each of a kernel's channels."""

_PYTEST_WRITES = frozenset({"PYTEST_CURRENT_TEST", "PYTEST_VERSION"})
"""What pytest writes into its own process's environment (``_pytest/runner.py`` and
``_pytest/config/__init__.py``), so a run made inside a test session would otherwise inherit it. A
script run has neither, and ipykernel stops capturing what is written to file descriptors 1 and 2
while ``PYTEST_CURRENT_TEST`` is set (``ipykernel/iostream.py``)."""


class RunError(RuntimeError):
    """An example's run failed: a cell raised, the run passed its budget, or the kernel died.

    The error carries ``raised``, the exception the example raised as Python's last traceback line
    names it (``ValueError: dataset is not set``), and ``raised_class``, that exception's class by
    the qualified name it carries, as the kernel recorded it (``builtins.ValueError``). ``raised``
    is ``None`` when the example raised nothing (its budget passed, its kernel died);
    ``raised_class`` is ``None`` then too, and when the kernel left no readable record. Neither the
    message nor ``raised`` holds terminal control sequences.
    """

    def __init__(
        self, message: str, *, raised: str | None = None, raised_class: str | None = None
    ) -> None:
        """Make the error.

        Args:
            message: What failed, naming the example.
            raised: The exception the example raised, as ``<type>: <message>``.
            raised_class: That exception's class, as ``<module>.<qualified name>``.
        """
        super().__init__(message)
        self.raised = raised
        self.raised_class = raised_class


@dataclass(frozen=True, slots=True, kw_only=True)
class Provenance:
    """Where a page came from and what its run produced.

    Attributes:
        source: The example script, relative to the repository root.
        sha256: The script's sha256 when the page was made.
        seconds: The run's measured wall time.
        device: The devices the run saw, or ``cpu`` when it imported no jax.
        versions: Python's and, where installed, jax's, substrax's and the repository's
            distribution's versions.
        outputs: Per code cell, the kinds of output it produced (``text``, ``figure``, ``rich``,
            ``error``), standard error left out.
    """

    source: str
    sha256: str
    seconds: float
    device: str
    versions: Mapping[str, str] = field(default_factory=dict)
    outputs: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class ExampleRun:
    """An executed clean copy of an example.

    Attributes:
        notebook: The notebook with every cell's outputs, exactly the cells the script holds.
        provenance: Where the run came from and what it produced.
        output_dir: The run's ``AVITAI_OUTPUT_DIR``, a temporary directory removed when the run
            ends; a page names it by a placeholder, so it reads the same on every run.
        returned: What ``main()`` returned, decoded from JSON (a 0-d array as its scalar);
            ``None`` when it returned nothing or no ``__main__`` guard ran it.
    """

    notebook: nbformat.NotebookNode
    provenance: Provenance
    output_dir: Path
    returned: JsonValue = None


class NativeKernelManager(AsyncKernelManager):
    """A kernel manager that only knows the native kernel on this environment's interpreter.

    It is asynchronous, as nbclient's own default is: nbclient waits for a cell's output and its
    reply at once, and a blocking client's wait for output holds the event loop, so neither the
    cell's time limit nor the check that the kernel is alive could ever run.

    A ``python3`` kernelspec installed elsewhere (a user's ``~/.local/share/jupyter``) names a bare
    ``python`` resolved on PATH; with no kernelspec directories, ipykernel's native spec is the only
    one, and it starts ``sys.executable``. nbclient creates this manager, so it owns the kernel
    and shuts it down and closes its channels when the run ends.
    """

    @default("kernel_spec_manager")
    def _native_kernel_only(self) -> KernelSpecManager:
        return KernelSpecManager(kernel_dirs=[], ensure_native_kernel=True)

    @default("transport")
    def _ipc_where_supported(self) -> str:
        # Unix sockets, not plain-text TCP on localhost; Windows has only TCP.
        return "tcp" if sys.platform == "win32" else "ipc"

    @default("ip")
    def _absolute_socket_path(self) -> str:
        # An ipc address is a socket path. The default is relative, so the kernel binds it in its
        # working directory (the repository) while the client looks in its own: absolute, in
        # Jupyter's runtime directory, or the temporary directory when that path would pass the
        # length a Unix socket path may have or cannot be created (a read-only home). The manager
        # removes the sockets on shutdown.
        if self.transport != "ipc":
            return "127.0.0.1"
        name = f"substrax-{uuid.uuid4().hex[:12]}"
        for directory in (Path(jupyter_runtime_dir()), Path(tempfile.gettempdir())):
            if len(f"{directory / name}{_SOCKET_SUFFIX}") > _SOCKET_PATH_LIMIT:
                continue
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except OSError:
                continue
            return str(directory / name)
        msg = (
            f"no kernel socket path of at most {_SOCKET_PATH_LIMIT} characters can be made under "
            f"{jupyter_runtime_dir()} or {tempfile.gettempdir()}; set TMPDIR to a shorter, "
            "writable directory"
        )
        raise RuntimeError(msg)


def execute_example(  # noqa: DOC503  # the ValueError is raised by refuse_unusable_budget
    script: Path, *, repo_root: Path, distribution: str, budget_seconds: float
) -> ExampleRun:
    """Execute a clean copy of an example's notebook from the repository root and describe the run.

    Args:
        script: The example script, under ``repo_root``.
        repo_root: The repository root; the run's working directory.
        distribution: The distribution the repository builds, whose version the run
            records.
        budget_seconds: How long the run may take, from the kernel's start to the example's last
            cell; each cell may use what is left of it.

    Returns:
        The executed notebook and its provenance.

    Raises:
        ValueError: If ``budget_seconds`` is not a finite number above zero.
        RunError: If the kernel does not start, a cell raises, the run passes its budget, the
            kernel dies, the run writes a file the repository tracks or an untracked one it does
            not ignore, or the repository is not under git.
    """
    refuse_unusable_budget(budget_seconds)
    before = _repository_state(repo_root)
    notebook = jupytext.read(script, fmt="py:percent")
    # The exporter fences code with the notebook's language; jupytext's reading names none.
    notebook.metadata["language_info"] = {"name": "python"}
    # A script stops at its first exception: no cell tag lets one fail and the run go on.
    client = NotebookClient(
        notebook,
        force_raise_errors=True,
        kernel_manager_class=NativeKernelManager,
        kernel_name=NATIVE_KERNEL_NAME,
        startup_timeout=_whole_seconds(budget_seconds),
        resources={"metadata": {"path": str(repo_root)}},
    )
    with tempfile.TemporaryDirectory(prefix="substrax-notebooks-") as directory:
        artifacts = Path(directory)
        (artifacts / "outputs").mkdir()
        report = artifacts / "report.json"
        raised_record = artifacts / "raised.json"
        setup = nbformat.v4.new_code_cell(
            _setup_source(
                script,
                repo_root=repo_root,
                report_path=report,
                raised_path=raised_record,
                distribution=distribution,
            )
        )
        started = time.perf_counter()
        _start_kernel(client, environment=_kernel_environment(artifacts), script=script)
        try:
            # The kernel and its client are started; setup_kernel reuses them and shuts them
            # down when the run ends.
            with client.setup_kernel():
                _execute_hidden(
                    client, setup, budget_seconds=budget_seconds, started=started, script=script
                )
                seconds = _execute_within(client, budget_seconds, started=started, script=script)
                _report_the_run(
                    client, budget_seconds=budget_seconds, started=started, script=script
                )
        except CellExecutionError as error:
            # nbclient's message holds the kernel's coloured traceback and whatever controls the
            # example printed; a terminal or a test report shows the escape codes as text.
            msg = f"{script}: {strip_terminal_controls(str(error))}"
            raised = strip_terminal_controls(f"{error.ename}: {error.evalue}")
            raise RunError(
                msg, raised=raised, raised_class=_read_raised_class(raised_record)
            ) from error
        measured = _read_report(report, script=script)
    after = _repository_state(repo_root)
    changed = sorted(
        part
        for part in before.keys() | after.keys()
        if before.get(part, _CLEAN) != after.get(part, _CLEAN)
    )
    if changed:
        msg = f"{script}: the run wrote into the repository: {', '.join(changed)}"
        raise RunError(msg)
    # One stream output where the kernel flushed several in a row, so the page and the recorded
    # output kinds do not depend on timing.
    for cell in notebook.cells:
        if cell.cell_type == "code":
            coalesce_streams(cell)
    if "returned_error" in measured:
        msg = f"{script}: {measured['returned_error']}"
        raise RunError(msg)
    provenance = Provenance(
        source=script.resolve().relative_to(repo_root.resolve()).as_posix(),
        sha256=script_sha256(script),
        seconds=round(seconds, 1),
        device=measured["device"],
        versions=measured["versions"],
        outputs=tuple(
            tuple(_output_kinds(cell)) for cell in notebook.cells if cell.cell_type == "code"
        ),
    )
    return ExampleRun(
        notebook=notebook,
        provenance=provenance,
        output_dir=artifacts / "outputs",
        returned=measured.get("returned"),
    )


def _kernel_environment(artifacts: Path) -> dict[str, str]:
    """The environment an example's kernel starts with: the caller's, as a script run gets it.

    The inline backend turns ``plt.show()`` into a captured figure; a caller's ``MPLBACKEND`` (Agg
    in a test run) would display nothing. A fresh IPython directory keeps the user's profile,
    startup files and config out of the kernel. The platform is the caller's, unchanged. What
    pytest wrote into this process (:data:`_PYTEST_WRITES`) is not the caller's.

    Args:
        artifacts: The run's temporary directory, holding its outputs and IPython directory.

    Returns:
        The kernel's environment.
    """
    return {
        **{name: value for name, value in os.environ.items() if name not in _PYTEST_WRITES},
        "AVITAI_OUTPUT_DIR": str(artifacts / "outputs"),
        "IPYTHONDIR": str(artifacts / "ipython"),
        "MPLBACKEND": "module://matplotlib_inline.backend_inline",
    }


def _start_kernel(client: NotebookClient, *, environment: dict[str, str], script: Path) -> None:
    """Start the run's kernel and its client, within the client's ``startup_timeout``.

    jupyter_client reports a kernel that does not reply in time, or dies before it replies, with a
    plain ``RuntimeError``; nbclient shuts down a kernel whose client fails to start.

    Args:
        client: The run's notebook client, its kernel not yet started.
        environment: The kernel's environment.
        script: The example, for the error message.

    Raises:
        RunError: If the kernel does not start.
    """
    client.create_kernel_manager()
    try:
        client.start_new_kernel(env=environment)
        client.start_new_kernel_client()
    except RuntimeError as error:
        msg = f"{script}: the kernel did not start within {client.startup_timeout} s: {error}"
        raise RunError(msg) from error


def _setup_source(
    script: Path, *, repo_root: Path, report_path: Path, raised_path: Path, distribution: str
) -> str:
    """The setup cell for ``script``: the kernel setup's source, run in a namespace of its own.

    Args:
        script: The example script.
        repo_root: The repository root, the run's working directory.
        report_path: Where the run's report is written, in the run's temporary directory.
        raised_path: Where the qualified class of a cell's exception is written, beside it.
        distribution: The distribution the repository builds.

    Returns:
        The cell's source.
    """
    arguments = {
        "directory": str(script.resolve().parent),
        "argv0": script.resolve().relative_to(repo_root.resolve()).as_posix(),
        "file": str(script.absolute()),
        "report_path": str(report_path),
        "raised_path": str(raised_path),
        "distribution": distribution,
    }
    keywords = ", ".join(f"{name}={value!r}" for name, value in arguments.items())
    setup = kernel_setup.setup_kernel.__name__
    return (
        "_substrax_setup = {}\n"
        f"exec(compile({_SETUP_SOURCE!r}, {kernel_setup.__name__!r}, 'exec'), _substrax_setup)\n"
        f"_substrax_setup[{setup!r}](globals(), get_ipython(), {keywords})\n"
        "del _substrax_setup\n"
    )


def _read_raised_class(path: Path) -> str | None:
    """The qualified class the setup cell's recorder wrote for a failing cell, if it wrote one.

    The run's directory is the example's to reach, so a record it removed, replaced or spoiled
    names no class.
    """
    try:
        recorded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return recorded if isinstance(recorded, str) else None


def _read_report(path: Path, *, script: Path) -> dict[str, Any]:
    """The run's report from the file the setup cell's reporter wrote.

    Args:
        path: The report's file, in the run's temporary directory.
        script: The example, for the error message.

    Returns:
        The decoded report: the devices, the versions and ``main()``'s value.

    Raises:
        RunError: If no report was written, as when the example deleted the reporter.
    """
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        msg = (
            f"{script}: the run's report of its devices, versions and main()'s value "
            "was not written"
        )
        raise RunError(msg) from error
    if not isinstance(report, dict):
        msg = f"{script}: the run's report is not a JSON object: {report!r}"
        raise RunError(msg)
    return report


def _report_the_run(
    client: NotebookClient, *, budget_seconds: float, started: float, script: Path
) -> None:
    """Call the setup cell's reporter in the kernel; a failure there is substrax's alone.

    Args:
        client: The client, its kernel started and the example run.
        budget_seconds: The run's budget, which the report counts against.
        started: When the run started, on :func:`time.perf_counter`'s clock.
        script: The example, for the error message.

    Raises:
        RunError: If the reporter raises, with ``raised`` left empty.
    """
    try:
        _execute_hidden(
            client,
            nbformat.v4.new_code_cell(f"{kernel_setup.REPORTER}()"),
            budget_seconds=budget_seconds,
            started=started,
            script=script,
        )
    except CellExecutionError as error:
        msg = f"{script}: substrax could not report the run: {error.ename}: {error.evalue}"
        raise RunError(msg) from error


def _execute_hidden(  # noqa: DOC502  # RunError is raised by _execute_cell
    client: NotebookClient,
    cell: nbformat.NotebookNode,
    *,
    budget_seconds: float,
    started: float,
    script: Path,
) -> None:
    """Execute a cell of substrax's own in the example's kernel; it never joins the notebook.

    The cell counts against the run's one budget, as the example's cells do: it may take what is
    left of it, in nbclient's whole seconds. nbclient writes an executed cell back into the
    notebook at its index, so the cell joins the notebook for its run and leaves at once.

    Args:
        client: The client, its kernel started.
        cell: The cell to execute.
        budget_seconds: The run's budget.
        started: When the run started, on :func:`time.perf_counter`'s clock.
        script: The example, for the error message.

    Raises:
        RunError: If the cell outlasts what is left of the budget, or the kernel dies.
    """
    client.nb.cells.append(cell)
    try:
        _execute_cell(
            client,
            len(client.nb.cells) - 1,
            seconds=budget_seconds - (time.perf_counter() - started),
            budget_seconds=budget_seconds,
            script=script,
            where="substrax's own cell",
        )
    finally:
        client.nb.cells.pop()


def _execute_within(
    client: NotebookClient, budget_seconds: float, *, started: float, script: Path
) -> float:
    """Execute every cell, each limited to what is left of the budget; return the seconds taken.

    nbclient waits for a cell in whole seconds, so a cell may run up to a second past what was
    left; the measured total is then held to the budget itself.

    Args:
        client: The client, its kernel started, holding the notebook to run.
        budget_seconds: How long the run may take, from the kernel's start to the last cell.
        started: When the run started, on :func:`time.perf_counter`'s clock.
        script: The example, for the error message.

    Returns:
        The seconds from the kernel's start to the end of the last cell.

    Raises:
        RunError: If the run passes its budget or the kernel dies.
    """
    for index in range(len(client.nb.cells)):
        remaining = budget_seconds - (time.perf_counter() - started)
        if remaining <= 0:
            break
        _execute_cell(
            client,
            index,
            seconds=remaining,
            budget_seconds=budget_seconds,
            script=script,
            where=f"cell {index + 1}",
        )
    seconds = time.perf_counter() - started
    if seconds > budget_seconds:
        msg = f"{script}: exceeded its budget of {budget_seconds} s ({seconds:.1f} s)"
        raise RunError(msg)
    return seconds


def _execute_cell(
    client: NotebookClient,
    index: int,
    *,
    seconds: float,
    budget_seconds: float,
    script: Path,
    where: str,
) -> None:
    """Execute the notebook's cell at ``index``, stopping it after ``seconds``.

    Args:
        client: The client, its kernel started.
        index: The cell's index in the client's notebook.
        seconds: How long the cell may take: what is left of the budget.
        budget_seconds: The run's budget, for the error message.
        script: The example, for the error message.
        where: The cell, as the error message names it.

    Raises:
        RunError: If the cell takes longer than ``seconds`` or the kernel dies.
    """
    client.timeout = _whole_seconds(seconds)
    try:
        client.execute_cell(client.nb.cells[index], index)
    except CellTimeoutError as error:
        msg = f"{script}: exceeded its budget of {budget_seconds} s at {where}"
        raise RunError(msg) from error
    except DeadKernelError as error:
        msg = f"{script}: the kernel died at {where}"
        raise RunError(msg) from error


_CLEAN = "clean as committed"
"""The state of a path the repository's listing leaves out: tracked and unchanged, or absent."""


def _repository_state(repo_root: Path) -> dict[str, object]:
    """What a run could change in the repository, each part comparable before and after it.

    The files are the repository's modified, removed and untracked ones, each named on its own,
    by size and modification time, so a write into a file already changed, or a new file in a
    directory already untracked, changes the state; a path absent from the listing is clean as
    committed, so one the run removes or changes joins it and differs. Ignored paths, such as
    ``__pycache__``, are left out, as git leaves them out. A clean file rewritten with its own
    bytes is not seen: git, too, reports it unchanged. Git's own state is compared as well: the
    commit HEAD names, the index's listing, and each file in the hooks directory.

    Args:
        repo_root: The repository root.

    Returns:
        Each part of the state by the name a failing run reports it under.

    Raises:
        RunError: If git is not installed or ``repo_root`` is not in a git repository.
    """
    try:
        paths = repository.unclean_files(repo_root)
        state: dict[str, object] = {
            "HEAD": repository.head(repo_root),
            "the index": hashlib.sha256(repository.index_listing(repo_root).encode()).hexdigest(),
            "the hooks directory": _files_by_stat(repository.hooks_directory(repo_root)),
        }
    except repository.RepositoryError as error:
        msg = f"{error}, so a run's writes into the repository cannot be checked"
        raise RunError(msg) from error
    for path in paths:
        try:
            found = (repo_root / path).lstat()
        except FileNotFoundError:
            state[path] = None
        else:
            state[path] = (found.st_size, found.st_mtime_ns)
    return state


def _files_by_stat(directory: Path) -> tuple[tuple[str, int, int], ...]:
    """Each file under ``directory`` by name, size and modification time; empty when absent."""
    if not directory.is_dir():
        return ()
    files = []
    for path in directory.rglob("*"):
        if path.is_file():
            found = path.stat()
            files.append((path.relative_to(directory).as_posix(), found.st_size, found.st_mtime_ns))
    return tuple(sorted(files))


def _whole_seconds(seconds: float) -> int:
    """The whole seconds nbclient's ``timeout`` takes: ``seconds`` rounded up, at least 1."""
    return max(1, math.ceil(seconds))


def _output_kinds(cell: nbformat.NotebookNode) -> list[str]:
    """The kinds of output a code cell produced, standard error left out.

    Standard error goes before standard output is joined, so the text printed on either side of
    a warning one machine writes and another does not is one ``text``, as it is with no warning.
    """
    kinds = []
    follows_stdout = False
    for output in cell.get("outputs", []):
        if output.output_type == "stream":
            if output.name == "stdout" and not follows_stdout:
                kinds.append("text")
            follows_stdout = follows_stdout or output.name == "stdout"
            continue
        follows_stdout = False
        if output.output_type == "error":
            kinds.append("error")
        elif "image/png" in output.get("data", {}):
            kinds.append("figure")
        elif set(output.get("data", {})) == {"text/plain"}:
            kinds.append("text")
        else:
            kinds.append("rich")
    return kinds


def coalesce_streams(cell: nbformat.NotebookNode) -> None:
    """Merge consecutive outputs of one stream in a code cell and resolve carriage returns.

    The rules are nbconvert's ``CoalesceStreamsPreprocessor`` (nbconvert 7.17,
    ``nbconvert/preprocessors/coalescestreams.py``): only consecutive outputs of the same stream
    merge, so text printed after a figure stays after it, and on each line the text up to the
    last carriage return that more of the line follows is dropped, as a terminal overwrites it.
    The preprocessor itself is not called: its carriage-return pattern takes quadratic time on a
    long line after a carriage return (80,000 characters, 1.6 s), and this pass is linear.
    nbclient's own ``coalesce_streams`` is not used either: it merges every output of a stream
    into its first one, across figures.

    Args:
        cell: A code cell with outputs, changed in place.
    """
    cell.outputs = _merged_streams(cell.get("outputs", []))
    for output in cell.outputs:
        if output.output_type == "stream" and "\r" in output.text:
            output.text = "\n".join(_after_overwrites(line) for line in output.text.split("\n"))


def _merged_streams(outputs: list[nbformat.NotebookNode]) -> list[nbformat.NotebookNode]:
    """The outputs with each run of consecutive outputs of one stream joined into the first."""
    merged: list[nbformat.NotebookNode] = []
    for output in outputs:
        last = merged[-1] if merged else None
        if (
            last is not None
            and output.output_type == "stream"
            and last.output_type == "stream"
            and last.name == output.name
        ):
            last.text += output.text
        else:
            merged.append(output)
    return merged


def _after_overwrites(line: str) -> str:
    """A line as a terminal shows it: the text after its last carriage return that text follows."""
    cut = line.rfind("\r", 0, len(line) - 1)
    return line[cut + 1 :] if cut >= 0 else line


def script_sha256(script: Path) -> str:
    """Return the sha256 of an example script's bytes, the identity a page records.

    Windows line endings count as Unix ones, as a page's own digest counts them: git's
    ``core.autocrlf`` checks a script out with them, and the script is the same version.

    Args:
        script: The example script.

    Returns:
        The hex digest.
    """
    return hashlib.sha256(script.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
