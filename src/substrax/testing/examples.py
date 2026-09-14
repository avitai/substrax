"""List a repository's examples and run each one in a fresh interpreter."""

from __future__ import annotations

import subprocess  # nosec B404
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from substrax.artifacts import OUTPUT_DIR_ENV
from substrax.runtime import JaxRuntime
from substrax.testing.child_process import ChildResult, run_python, tail_lines


_ENTRY_POINT = Path(__file__).with_name("_example_main.py")


@dataclass(frozen=True, slots=True, kw_only=True)
class ExampleRun:
    """One run of an example.

    Attributes:
        path: The example that ran.
        result: The child interpreter's exit code, output and duration.
        summary: What ``main()`` returned, decoded from JSON, for a run that called it and exited 0;
            ``None`` otherwise.
    """

    path: Path
    result: ChildResult
    summary: Any


class ExampleTimeoutError(AssertionError):
    """An example ran past its budget and was killed.

    Attributes:
        path: The example.
        timeout: Its budget in seconds.
        stderr_tail: The last lines it wrote to standard error before it was killed.
    """

    path: Path
    timeout: float
    stderr_tail: str

    def __init__(self, path: Path, timeout: float, stderr: str) -> None:
        """Name the example, its budget and the end of its standard error.

        Args:
            path: The example.
            timeout: Its budget in seconds.
            stderr: Everything it wrote to standard error before it was killed.
        """
        tail = tail_lines(stderr)
        super().__init__(f"{path} exceeded its {timeout:g} s budget; last lines of stderr:\n{tail}")
        self.path = path
        self.timeout = timeout
        self.stderr_tail = tail


def discover_examples(root: Path, *, include: Callable[[Path], bool] | None = None) -> list[Path]:
    """List the example scripts under ``root``, sorted.

    A file or directory whose name starts with ``_`` is private, as ``_common`` helpers,
    ``_templates`` and ``__init__.py`` are, and so is everything under it. Only the parts below
    ``root`` count, so ``root`` itself may have such a name.

    Args:
        root: The examples directory.
        include: Keeps a path when it returns ``True``, such as a repository's numbering rule.

    Returns:
        The ``*.py`` files under ``root`` that are not private and that ``include`` keeps.
    """
    return [
        path
        for path in sorted(root.rglob("*.py"))
        if not any(part.startswith("_") for part in path.relative_to(root).parts)
        and (include is None or include(path))
    ]


def run_example(  # noqa: DOC503
    path: Path,
    *,
    repo_root: Path,
    output_dir: Path,
    timeout: float,
    call_main: bool,
    runtime: JaxRuntime | None = None,
    env: Mapping[str, str] | None = None,
) -> ExampleRun:
    """Run one example in a fresh interpreter from the repository root, with outputs redirected.

    The child starts through :func:`~substrax.testing.run_python`, so the JAX settings are the
    ``runtime`` and ``env`` given here, not the parent shell's. ``AVITAI_OUTPUT_DIR`` is set to
    ``output_dir``, so an example resolving its outputs with
    :func:`~substrax.artifacts.resolve_output_dir` never writes into the repository. Whatever the
    example configures at import, such as logging or 64-bit types, ends with the child.

    Args:
        path: The example.
        repo_root: The child's working directory, which relative paths in examples assume.
        output_dir: Where the example's outputs go.
        timeout: Seconds before the child is killed; every run states its budget.
        call_main: ``True`` loads the example under a name other than ``__main__``, calls its
            ``main()`` without arguments and decodes the return value; ``False`` runs the file
            as a script.
        runtime: JAX settings of the child.
        env: Variables set in the child before ``runtime`` is applied.

    Returns:
        The run. A non-zero exit is returned, not raised; ``run.result.check()`` raises it.

    Raises:
        ValueError: If ``env`` sets ``AVITAI_OUTPUT_DIR`` to a directory other than ``output_dir``.
        ExampleTimeoutError: If the example runs longer than ``timeout``.
        XlaFlagConflictError: If a flag in ``runtime`` conflicts with ``XLA_FLAGS`` in ``env``; the
            child is not started.
    """
    example = path.resolve()
    target = output_dir.resolve()
    variables = dict(env or {})
    chosen = variables.get(OUTPUT_DIR_ENV)
    if chosen is not None and Path(chosen).resolve() != target:
        raise ValueError(f"env sets {OUTPUT_DIR_ENV} to {chosen!r}, which is not {target}")
    variables[OUTPUT_DIR_ENV] = str(target)
    program = _ENTRY_POINT if call_main else example
    arguments = (str(example),) if call_main else ()
    try:
        result = run_python(
            program, *arguments, timeout=timeout, runtime=runtime, env=variables, cwd=repo_root
        )
    except subprocess.TimeoutExpired as error:
        raise ExampleTimeoutError(example, timeout, _decoded(error.stderr)) from error
    summary = result.last_json() if call_main and result.returncode == 0 else None
    return ExampleRun(path=example, result=result, summary=summary)


def unavailable_reason(run: ExampleRun, signatures: Sequence[str]) -> str | None:
    """Return the first signature found in a failed run's standard error, ignoring case.

    A repository lists the messages that mean an example cannot run in this environment, such as a
    dataset that is not configured, and skips the example when one matches. A run that exited 0
    matches nothing, and standard output is not searched.

    Args:
        run: The finished run.
        signatures: Messages that mean the example cannot run here.

    Returns:
        The first signature found, as given, or ``None``.
    """
    if run.result.returncode == 0:
        return None
    stderr = run.result.stderr.lower()
    return next((signature for signature in signatures if signature.lower() in stderr), None)


def _decoded(output: str | bytes | None) -> str:
    """Return what a killed child wrote as text; ``subprocess`` collects it as bytes on timeout."""
    if isinstance(output, bytes):
        return output.decode(errors="replace")
    return output or ""
