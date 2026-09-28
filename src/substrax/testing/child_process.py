"""Fresh-interpreter runs whose JAX configuration the caller chooses."""

from __future__ import annotations

import dataclasses
import json
import subprocess  # nosec B404
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from pydantic import TypeAdapter

from substrax.runtime import child_environment, JaxRuntime
from substrax.typing import JsonValue


_STDERR_TAIL_LINES = 40


@dataclass(frozen=True, slots=True, kw_only=True)
class ChildResult:
    """What a child interpreter did.

    Attributes:
        argv: The command line the child ran.
        returncode: Its exit code.
        stdout: Everything it wrote to standard output.
        stderr: Everything it wrote to standard error.
        seconds: Wall-clock seconds from start to exit.
    """

    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    seconds: float

    def check(self) -> ChildResult:
        """Return this result when the child exited with code 0.

        Returns:
            This result.

        Raises:
            ChildFailedError: If the exit code is not 0; the message ends with the last lines of
                standard error.
        """
        if self.returncode != 0:
            raise ChildFailedError(self)
        return self

    def last_json(self) -> JsonValue:
        """Parse the last non-empty line of standard output as JSON.

        Returns:
            The decoded value; a caller narrows it to the shape it expects.

        Raises:
            ValueError: If the child wrote no output, or its last line is not JSON.
        """
        line = self._last_line()
        try:
            return json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"the child's last line of output is not JSON: {line!r}") from error

    def last_json_as[T](self, shape: type[T]) -> T:  # noqa: DOC502  # raised by _last_line and pydantic
        """Parse the last non-empty line of standard output as JSON of ``shape``.

        The line is validated by pydantic in strict JSON mode, so a value of another type is
        refused rather than converted (an integer is accepted where ``shape`` has a float, as
        JSON does not tell them apart).

        Args:
            shape: The type the value must have, such as a ``TypedDict`` or ``list[int]``.

        Returns:
            The decoded value, of ``shape``.

        Raises:
            ValueError: If the child wrote no output.
            pydantic.ValidationError: If the line is not JSON, or not a value of ``shape``; the
                message names the offending field.
        """
        return TypeAdapter(shape).validate_json(self._last_line(), strict=True)

    def _last_line(self) -> str:
        """The last non-empty line of standard output.

        Returns:
            The line.

        Raises:
            ValueError: If the child wrote no output.
        """
        lines = [line for line in self.stdout.splitlines() if line.strip()]
        if not lines:
            raise ValueError(f"the child wrote no output to parse as JSON: {self.argv[:2]}")
        return lines[-1]


class ChildFailedError(AssertionError):
    """A child interpreter exited with a non-zero code.

    Attributes:
        result: The failed run.
    """

    result: ChildResult

    def __init__(self, result: ChildResult) -> None:
        """Describe the failed run by its exit code and the end of its standard error.

        Args:
            result: The failed run.
        """
        tail = tail_lines(result.stderr)
        super().__init__(
            f"child {result.argv[1:2]} failed with exit code {result.returncode}; "
            f"last lines of stderr:\n{tail}"
        )
        self.result = result


def run_python(  # noqa: DOC502
    program: str | Path,
    *args: str,
    timeout: float,
    runtime: JaxRuntime | None = None,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
) -> ChildResult:
    """Run Python code or a script in a fresh interpreter and return what it did.

    jax fixes its backends and device set when they start, so tests of device topology or process
    configuration need their own interpreter. The child runs this interpreter
    (``sys.executable``). Its environment is this process's without any ``JAX_*`` or ``XLA_*``
    variable, so settings exported in a developer's shell cannot change what a test measures, then
    ``env``, then ``runtime`` as :func:`~substrax.runtime.runtime_environment` renders it. Unless
    ``runtime`` or ``env`` chooses them, the child runs on the CPU backend without preallocation,
    so a test never starts an accelerator it did not ask for.

    Args:
        program: Code to run with ``-c``, or the path of a script.
        *args: Arguments passed to the program.
        timeout: Seconds before the child is killed; every run states its budget.
        runtime: JAX settings of the child.
        env: Variables set in the child before ``runtime`` is applied.
        cwd: The child's working directory; this process's when ``None``.

    Returns:
        The child's exit code, output and duration. A non-zero exit is returned, not raised.

    Raises:
        XlaFlagConflictError: If a flag in ``runtime`` conflicts with ``XLA_FLAGS`` in ``env``;
            the child is not started.
        subprocess.TimeoutExpired: If the child runs longer than ``timeout``.
    """
    child_env = child_environment(_with_safe_defaults(runtime, env or {}), env)
    command = (
        (sys.executable, "-c", program)
        if isinstance(program, str)
        else (sys.executable, str(program))
    )
    argv = (*command, *args)
    started = time.perf_counter()
    completed = subprocess.run(  # noqa: S603  # nosec B603  # this interpreter and the caller's program; no shell
        argv,
        env=child_env,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return ChildResult(
        argv=argv,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        seconds=time.perf_counter() - started,
    )


def cuda_is_visible(*, timeout: float = 120.0) -> bool:
    """Whether a fresh interpreter can start jax's CUDA backend and sees at least one GPU.

    The probe runs in a child without preallocation, so it claims no accelerator memory.

    Args:
        timeout: Seconds before the probe is abandoned.

    Returns:
        ``True`` when the child found a GPU.
    """
    probe = "import sys, jax; sys.exit(0 if jax.devices('gpu') else 1)"
    result = run_python(probe, runtime=JaxRuntime(platforms=("cuda",)), timeout=timeout)
    return result.returncode == 0


def tail_lines(stderr: str) -> str:
    """Return the last lines of a child's standard error, where the error that ended it is.

    Args:
        stderr: Everything the child wrote to standard error.

    Returns:
        Its last 40 lines.
    """
    return "\n".join(stderr.splitlines()[-_STDERR_TAIL_LINES:])


def _with_safe_defaults(runtime: JaxRuntime | None, env: Mapping[str, str]) -> JaxRuntime:
    """Default the child to the CPU backend without preallocation, unless the caller chose."""
    chosen = runtime or JaxRuntime()
    platforms_chosen = chosen.platforms is not None or "JAX_PLATFORMS" in env
    preallocate_chosen = chosen.preallocate is not None or "XLA_PYTHON_CLIENT_PREALLOCATE" in env
    return dataclasses.replace(
        chosen,
        platforms=chosen.platforms if platforms_chosen else ("cpu",),
        preallocate=chosen.preallocate if preallocate_chosen else False,
    )
