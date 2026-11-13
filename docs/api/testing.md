# Testing

`substrax.testing` holds the test infrastructure JAX packages share: fresh-interpreter runs with a
chosen JAX configuration, trace, compile and work counts, the shared example tests, and an opt-in
pytest plugin. Install it with `substrax[testing]`; the example tests also need the `notebooks`
extra, `substrax[notebooks,testing]`.

```python
from substrax.runtime import JaxRuntime
from substrax.testing import run_python

result = run_python(
    "import jax; from substrax.testing import report; report(jax.device_count())",
    runtime=JaxRuntime(cpu_devices=8),
    timeout=120,
)
assert result.check().report_as(int) == 8
```

`run_python` runs this interpreter on code or a script. The child's environment drops every
`JAX_*` and `XLA_*` variable the parent inherited, then applies `env` and the `runtime`. Unless
either of them chooses otherwise, the child runs on the CPU backend without preallocation, so a
test never starts an accelerator it did not ask for. `check()` raises `ChildFailedError` with
the end of the child's standard error.

The child sends its result with `report(value)`. `run_python` names a file in the child's
`SUBSTRAX_CHILD_REPORT` variable, a file only that child knows, so nothing the child or a library
prints can corrupt the result, even a C++ library writing to standard output after it. A child
reports once, and a second `report` raises. `report_as(shape)` reads the result as a value of
`shape`, a `TypedDict` or a type such as `dict[str, list[str]]`, validated by pydantic in strict
JSON mode, so a test reads a structured report with its fields typed and a report of another
shape fails naming the field. `report_json()` returns the decoded JSON for a caller that narrows
it itself. Both raise `ValueError` for a child that sent no report. A child that checks which
modules an import loaded measures before it imports `report`.
`cuda_is_visible()` asks a child on the CUDA backend whether jax sees a GPU.

## Counting traces

`TraceCounter` counts how often a function's Python body runs. Under `jax.jit` and `nnx.jit`
the body runs once per trace, so the count is the number of traces a test caused:

```python
import jax
import jax.numpy as jnp

from substrax.testing import TraceCounter


def double(x):
    return x * 2.0


counter = TraceCounter()
step = jax.jit(counter.wrap(double))

with counter.expect(new_traces=1):
    step(jnp.ones(3))
with counter.expect(new_traces=0):  # same shape and dtype: no new trace
    step(jnp.zeros(3))
```

`expect` raises `RetraceError`, an `AssertionError` that names the expected and observed
counts, and an error raised inside the block propagates unchanged. Each counter keeps its own
count, so there is no global registry to clear. Wrap the Python function, then jit the wrapper.
The counter does not listen to jax's `jax.monitoring` trace event: jax records that event only
for top-level traces, and its name is not documented.

## Counting compiles

A trace and a compile are different events: a trace builds a jaxpr, a compile builds an XLA
executable, and a call that hits the executable cache does neither. `substrax.testing.compiles`
records the executables jax builds inside a block, by name:

```python
import jax
import jax.numpy as jnp

from substrax.testing.compiles import compiled_programs, expect_compiles

step, x = jax.jit(lambda v: v * 2.0), jnp.ones(3)  # jnp.ones compiles too: make inputs first

with compiled_programs() as names:
    step(x)
assert names == ["jit(<lambda>)"]
with expect_compiles(0):  # same shape and dtype: the cached executable
    step(x)
```

It listens to `/jax/core/compile/backend_compile_duration`, which jax records around
`compile_or_get_cached` when its in-process executable cache misses, so an executable the
persistent compilation cache supplies counts as well. The event name is jax's own constant
(`jax._src.dispatch.BACKEND_COMPILE_EVENT`), not a documented name; the package's tests fail if
jax stops recording it. The listener lives for the block only, and it sees every compile in the
process, including another thread's. `expect_compiles` raises `CompileCountError`, an
`AssertionError` with the expected count and the compiled names. The module imports jax, so
`substrax.testing` does not import it.

## Counting work

A test that bounds how much work a code path does by timing it measures the machine as much as the
code. The functions the path starts and the calls it makes are the same on every machine.
`counted_calls` counts them with `sys.monitoring` on the block's own threads: the thread that
opened it and every thread started after it opened, so work a block hands to a thread it starts is
counted too:

```python
import jax

from substrax.testing import by_package, counted_calls, per_iteration
from substrax.testing.jax_calls import JAX_CALL_WEIGHTS

keys = by_package("mypackage", "grain", "jax", named=["mypackage"])


def count(batches: int) -> dict[str, int]:
    run_pass(batches)  # warm: nothing traces or compiles inside the counted pass
    with counted_calls(keys, weights=JAX_CALL_WEIGHTS) as counts:
        run_pass(batches)
    return counts


per_batch = per_iteration(count, short=100, long=200)
assert per_batch["device_put"] == 1  # one placement per batch
assert per_batch["jit_dispatch"] == 1  # one compiled step per batch
assert per_batch["mypackage:Loader.__next__"] == 1
```

- `by_package` adds one to a package's name for each function started in its files, and one to
  `"<package>:<qualified name>"` for a package in `named`; anything else counts as `other`.
  Packages are matched by the directories `importlib` finds for them, and a name it cannot find
  is refused, so a misspelt package fails instead of counting nothing.
- `threading` and `queue` are left out by default (`excluded`): how often a thread waits on a
  condition depends on timing, which a count must not.
- Automatic garbage collection is off inside the block and one collection runs at its end, still
  counted: the collector runs by allocation count, so finalizers would otherwise land in or out of
  the block depending on what ran before it.
- `per_iteration(count, short=, long=)` returns `(count(long) - count(short)) / (long - short)` per
  key, so work done once per pass (setup, a thread's start and close) cancels. A count that does
  not grow by a whole number per iteration raises `UnevenCountError` naming it, instead of being
  averaged.
- `weights` count calls made from Python code: each weight sees the callable and the first argument
  the call passes, and adds what it returns. `substrax.testing.jax_calls.JAX_CALL_WEIGHTS` counts
  `jit_dispatch` (calls of a `jax.jit`-compiled callable, `nnx.jit` included), `device_put` (by
  identity, so an alias counts) and `device_put_leaves` (the leaves placed). It imports jax, so
  `substrax.testing` does not import it.

Code whose calls depend on object addresses varies between passes, and the counter cannot make it
deterministic: an `__eq__` of a key hashed by `id()` runs only when a new object reuses a dead one's
address, which the allocator decides. `per_iteration` names such a key (count its package by name
with `named` to see the function), and a keys function that wraps `by_package` can leave that
code object out, as `by_package` leaves out `threading`.

A start is a Python function or generator starting; a generator resuming and a C function are not
counted, and a count sees nothing a C function does without calling back into Python. A thread
already running Python when the block opens is left out: a library's monitor or heartbeat (tqdm's
progress-bar monitor wakes every 10 seconds) would otherwise add its calls to whichever pass it
woke in, and `per_iteration` would find the count uneven. Work a block hands to such a thread, a
thread pool created before the block for example, is left out with it; create the pool inside the
block to count it. A thread is told apart by the frame its Python execution started in, not by its
id: CPython gives the id of a thread that ended to the next one it starts.

`counted_calls` takes a `sys.monitoring` tool id for the block and frees it at the end, also when
the block raises. It tries the ids CPython leaves unassigned first (3 and 4), so coverage.py's
`sysmon` core (id 1) and `cProfile` (id 2) keep theirs, and raises `ToolIdsInUseError` naming the
holders when none is free. Python 3.12 and 3.13 run it; on a free-threaded build the callbacks run
on several threads at once and a lock keeps the counts exact. `cProfile` has used `sys.monitoring`
since 3.12 and records the calls of every thread, with no way to leave a thread out.

## Checking gradients

A test that only checks that a gradient exists, is finite or is non-zero passes on a wrong one.
`substrax.testing.gradients` compares a module's gradient with finite differences, through
`jax.test_util.check_grads` (forward and reverse mode along one random direction), and returns it:

```python
from substrax.testing.gradients import check_input_gradients, check_parameter_gradients

gradient = check_parameter_gradients(module, lambda m: loss(m(x), y))  # in its nnx.Param leaves
input_gradient = check_input_gradients(module, lambda m, v: loss(m(v), y), x)
```

The check runs on a float64 copy of the module, split in tree mode: a derivative summed over many
output elements is below a float32 finite difference's resolution (`check_grads`'s default
tolerance is 1e-5 in float64 against 2e-3 in float32). The caller's module keeps its dtype and
values. A gradient that is zero everywhere is refused unless `allow_zero=True`, since a loss that
ignores what is differentiated passes a finite-difference check. A function with a
`jax.custom_vjp` has no forward mode; check it with `modes=("rev",)`.

## Example tests

A repository's example tests are a pytest plugin, configured in its pyproject and written nowhere
else. `uv run substrax-notebooks test` loads it by name and runs pytest on `examples` and, where it
exists, `tests/examples`, without the repository's pytest configuration or conftests, any other
plugin, or the caller's `PYTEST_ADDOPTS` and `PYTEST_PLUGINS` (`examples_pytest_options` in
`substrax.notebooks.cli`), so each kernel gets the environment a script run gets;
`--changed-since REF` passes `--examples-changed-since=REF`, and PATHs, from the working directory
and under the repository, replace the two directories. pytest reads a configuration file of
substrax's that sets nothing, so none of the repository's ini keys (`addopts`, `filterwarnings`,
`faulthandler_timeout`, `required_plugins`, `strict_config`) reaches the session, and takes the
repository as its rootdir, whatever directory holds it; unusable settings are refused first.

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]                      # examples run apart:
addopts = "--ignore=tests/examples"         # uv run substrax-notebooks test

[tool.substrax.examples]
source_url = "https://github.com/avitai/datarax/blob/main"
budget_seconds = 600
unavailable = ["wod_motion_tfrecord_path is not set"]
page_size_warn_bytes = 102400   # the defaults: a larger page is written with a warning
page_size_bytes = 204800        # and a larger one refused
page_size_exempt = []
budgets = { "examples/training/long_run.py" = 1800 }   # an example's own budget, else the default
```

| Collected | Checks |
| --- | --- |
| `examples::corpus` | There is an example; no page under `docs/examples` lacks one; a hand-written `uv run python examples/...` command names a file that exists; `mkdocs.yml` sets `validation.nav.omitted_files: warn` |
| `examples/<path>.py::follows-the-guide` | Every rule of `substrax.notebooks.conformance` |
| `examples/<path>.py::runs-as-its-page-shows` | The example runs as its clean notebook within its budget (its `budgets` entry, else `budget_seconds`), changes nothing git tracks or would track (paths it ignores aside), and its page shows that run; marked `example_run` |

- `discover_examples` decides what is an example: every `.py` under `examples/` whose path has no
  part starting with `_`, so `_common` helpers are not examples. An example reached through a
  symlink, a linked `examples/` directory included, is a usage error: its page and its changes
  name it by its path, which a link would make a second name or put outside the repository.
- A run whose raised exception, as `<type>: <message>`, holds one of the `unavailable` messages is
  skipped with it; the failing cell's source and traceback are not read, and a run that raised
  nothing (its budget passed, its kernel dead) is never skipped. A `GPU-required` example whose run
  raised `GPUUnavailableError`, as `require_gpu()` does without a GPU, is skipped with that message
  before the `unavailable` messages are read, and so is every test of its results; any other example
  raising it is held to those messages like any failure. The class is known by the qualified name it
  carries, which the kernel records in `RunError.raised_class` whatever raised it. The record is
  made in the example's own process, so an example that rewrites a class's `__module__`, clears
  IPython's callbacks or writes the run's files alters its own run, which the check does not guard
  against. The suite probes no device, never imports jax in the pytest process, and loads none of
  substrax's other testing plugins: `substrax.testing.pytest_plugin`'s markers and configuration
  isolation apply to a repository's tests only where the repository enables it. A GPU lane runs
  `substrax-devices --require-gpu` before the examples and sets no platform
  ([Checks](../guides/writing-examples.md#checks)), so a lane that has lost its GPU fails there
  rather than skipping its GPU-required runs.
- `--examples-changed-since=REF` keeps the runs of the examples whose script, notebook or page
  changed since `REF`, and every run when `src/`, `pyproject.toml` or `uv.lock` changed, or a file
  under `examples/` that is no example's own script, notebook or page (a private helper, data). `REF`
  must name a commit, and is read as one even when it looks like a git option; anything else is a
  usage error. A change counts whether committed, staged or untracked (what git ignores does not),
  and only below the repository root, so a project inside a larger git repository sees its own.
- A repository's own test of an example's results takes the `example_run` fixture, which returns
  the session's one run of the example, `main()`'s value included; no conftest or other plugin is
  loaded, so it is the one fixture beyond pytest's own, and it carries no device marker, its
  example's Devices line deciding. In a directory pytest walks, it is collected by pytest's default
  naming (`test_*.py` or `*_test.py`, `test`-prefixed functions, `Test`-prefixed classes): the
  repository's naming keys do not reach the session. A file named on the command line is collected
  whatever its name. The suite's checks run before any other test:

```python
def test_the_fit_reaches_the_true_weights(example_run) -> None:
    summary = example_run("training/linear_fit.py").returned
    assert summary["relative_error"] < 1e-3
```

## The pytest plugin

Enable it in the top-level `conftest.py`:

```python
pytest_plugins = ["substrax.testing.pytest_plugin"]
```

| Provides | Behaviour |
| --- | --- |
| `@pytest.mark.x64` | Runs the test inside `jax.enable_x64(True)`, a thread-local setting |
| Configuration isolation | Snapshots `jax.config.values` around every test and fails a test that changed a global value, after setting it back |
| Collection guard | Fails the session when a module left x64 on at import although `JAX_ENABLE_X64` did not ask for it |
| `@pytest.mark.devices(count, kind=None)` | Skips unless at least `count` devices, of `kind` when given, are visible |
| `@pytest.mark.accelerator(kind=None)` | Skips unless jax's default backend is an accelerator, of `kind` when given |
| `output_dir` fixture | A fresh `outputs` directory under `tmp_path`, with `AVITAI_OUTPUT_DIR` pointing at it for the test |

The isolation follows jax's own test harness, whose `JaxTestCase` fails a test that changes a
global configuration value. A test whose subject changes the global configuration sets it back
with `restored_jax_config()`, which reports the names it restored:

```python
import jax

from substrax.runtime import JaxRuntime, apply_runtime
from substrax.testing import restored_jax_config


def test_apply_runtime_enables_x64() -> None:
    with restored_jax_config() as changed:
        apply_runtime(JaxRuntime(enable_x64=True))
        assert jax.config.jax_enable_x64

    assert changed == ["jax_enable_x64"]
```

A malformed `devices` or `accelerator` marker is a usage error at collection. No entry point
loads the plugin, so installing substrax changes no other project's tests, and importing it
imports no jax.

## Source scans

`substrax.testing.source_scans` holds the checks a repository's contract tests run over its own
source without importing it. Two import-time side effects change the process for everything
imported after the module, test collection included: configuring the root logger and writing the
environment.

```python
from pathlib import Path

from substrax.testing.source_scans import (
    configures_logging,
    import_time_lines,
    python_files,
    writes_environment,
)

ROOT = Path(__file__).resolve().parents[1]


def test_no_module_changes_the_process_at_import() -> None:
    modules = python_files(ROOT, ("src", "scripts", "tests"), excluded_parts=("example_data",))
    found = [
        f"{path.relative_to(ROOT)}:{line}"
        for path in modules
        for predicate in (configures_logging, writes_environment)
        for line in import_time_lines(path, predicate)
    ]
    assert found == []
```

`import_time_lines` skips function, class and lambda bodies and the `if __name__ == "__main__":`
block, whose code runs only when called or run as a script; `!=` and chained comparisons are not
guards. `writes_environment` matches assignment, deletion and `|=` on `os.environ`, its mutating
methods, and `os.putenv`/`os.unsetenv`.

The mesh scan finds documented `jax.make_mesh` calls that do not name `axis_types`. From jax 0.11
`jax.make_mesh` builds Explicit axes by default, and the backward pass of a batch-sharded step
raises under them, so every mesh a reader copies names its axis types:

```python
from substrax.testing.source_scans import documented_texts, unguarded_make_mesh_calls


def test_documented_meshes_name_their_axis_types() -> None:
    texts = documented_texts(ROOT, package="datarax", script_directories=("examples",))
    assert {name: calls for name, text in texts.items() if (calls := unguarded_make_mesh_calls(text))} == {}
```

`documented_texts` reads every docstring under `src/<package>`, the README (or the `pages` given),
every Markdown page under `docs` and every Python file under the script directories. A location
that does not exist raises `FileNotFoundError`, so a mistyped directory fails the check instead of
shrinking what it reads.

::: substrax.testing

::: substrax.testing.compiles

::: substrax.testing.jax_calls

::: substrax.testing.gradients

::: substrax.testing.source_scans

::: substrax.examples

::: substrax.testing.example_suite

::: substrax.testing.compute
