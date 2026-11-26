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
`jax.test_util.check_grads` (forward and reverse mode along one random direction of unit norm),
and returns it:

```python
from substrax.testing.gradients import check_input_gradients, check_parameter_gradients

gradient = check_parameter_gradients(module, lambda m: loss(m(x), y))  # in its nnx.Param leaves
input_gradient = check_input_gradients(module, lambda m, v: loss(m(v), y), x)
```

The direction has unit norm, as JAX's automatic-differentiation docs and PyTorch's fast
`gradcheck` take it: `check_grads` differentiates `t -> loss(m + t * u)` at `t = 0`, so its
finite-difference step is a length in parameter space whatever the number of parameters.
`check_grads` applied to the parameters directly draws a direction of norm about `sqrt(n)`, and on
a module with thousands of parameters its truncation error fails a correct gradient.

The step is 1e-6, PyTorch `gradcheck`'s default, not `check_grads`' own 1e-4. A central
difference's truncation error grows as the step squared: at 1e-4 it failed correct gradients of
modules with strong curvature along the direction, and of modules with a parameter within a step of
a `leaky_relu` kink, where the difference straddles the kink. In float64 the rounding a smaller step
brings stays near 1e-10 relative.

The check runs on a float64 copy of the module, split in tree mode: a derivative summed over many
output elements is below a float32 finite difference's resolution (`check_grads`'s default
tolerance is 1e-5 in float64 against 2e-3 in float32). `as_float64` makes the copy, casting
floating arrays and leaving integer arrays, such as labels that index a table, as they are. The
copy raises the parameters, not a dtype a module fixed when it was built: `nnx.Embed` keeps
`self.dtype = dtype or self.embedding.dtype` from its float32 table, so its float64 copy still
computes in float32. The caller's module keeps its dtype and values. A gradient that is zero everywhere is refused unless `allow_zero=True`, since a loss that
ignores what is differentiated passes a finite-difference check. A function with a
`jax.custom_vjp` has no forward mode; check it with `modes=("rev",)`.

## Example tests

A repository's example tests are a pytest plugin, configured in its pyproject and written nowhere
else. `uv run substrax-notebooks test` loads it by name and runs pytest on `examples` and, where they
exist, `recipes` and `tests/examples`, without the repository's pytest configuration or conftests, any other
plugin, or the caller's `PYTEST_ADDOPTS` and `PYTEST_PLUGINS` (`examples_pytest_options` in
`substrax.notebooks.cli`), so each kernel gets the environment a script run gets;
`--changed-since REF` passes `--examples-changed-since=REF`, `--budget-seconds SECONDS` passes
`--examples-budget-seconds=SECONDS`, and PATHs, from the working directory
and under the repository, replace those directories. pytest reads a configuration file of
substrax's that sets nothing, so none of the repository's ini keys (`addopts`, `filterwarnings`,
`faulthandler_timeout`, `required_plugins`, `strict_config`) reaches the session, and takes the
repository as its rootdir, whatever directory holds it; unusable settings are refused first.

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]                      # examples run apart:
addopts = "--ignore=tests/examples"         # uv run substrax-notebooks test

[tool.substrax.examples]
source_url = "https://github.com/avitai/<repository>/blob/main"
budget_seconds = 600
unavailable = ["<the message an example fails with when its dataset is missing>"]
page_size_warn_bytes = 102400   # the defaults: a larger page is written with a warning
page_size_bytes = 204800        # and a larger one refused
page_size_exempt = []
budgets = { "examples/training/long_run.py" = 1800 }   # an example's own budget, else the default
pending = { "examples/training/gan.py" = "waits for the critic fix (issue 12)" }   # held back, with why
```

| Collected | Checks |
| --- | --- |
| `examples::corpus` | There is an example; every file under `docs/examples` is an example's page or one of its figures, or a recipe's page (an overview lives at `docs/examples.md`); every notebook under `examples` has its script; every recipe under `recipes` has its example; a hand-written `python examples/...` or `uv run python examples/...` command names a file that exists, a placeholder such as `<name>` aside; `mkdocs.yml` sets `validation.nav.omitted_files: warn` |
| `examples/<path>.py::follows-the-guide` | Every rule of `substrax.notebooks.conformance`; skipped with its reason for a `pending` example that breaks one, failed for a pending one that breaks none |
| `recipes/<path>.py::follows-the-guide` | The guide's rules about code (`RECIPE_RULES`), over a recipe; a recipe runs on a GPU lane, never here |
| `examples/<path>.py::runs-as-its-page-shows` | The example runs as its clean notebook within its budget (its `budgets` entry, else `budget_seconds`), changes nothing git tracks or would track (paths it ignores aside), and its page shows that run; skipped with its reason, never started, for a `pending` example; marked `example_run` |

- The examples are collected directory by directory, as pytest's own directory walk collects, so a
  nested example is selected by its path (`examples/<topic>/<name>.py`), its directory or the node
  id a failure prints, and `--ignore` and `--ignore-glob` leave an example directory out. No other
  plugin collects a file under `examples/`, so an example named `test_*.py` is not imported as a
  test module.
- `discover_examples` decides what is an example: every `.py` under `examples/` whose path has no
  part starting with `_`, so a private helper beside an example is not one. `example_source`, beside it, gives
  an example's path from the root, `examples/<path>.py`, to every tool that names one: the
  commands, this session, the run's provenance, the pages, the settings and compute's tasks. An
  example reached through a symlink below the root, a linked `examples/` directory included, is
  refused by each of them: its page and its changes name it by its path, which a link would make a
  second name or put outside the repository. A link above the root (one filesystem mounted at two
  paths) changes nothing.
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
- `--examples-budget-seconds=SECONDS` is a lane's budget for every example without a `budgets`
  entry, in place of `budget_seconds`; an example's own entry still applies, as pytest-timeout's
  command line yields to an item's own marker. A value that is not a finite number above zero is a
  usage error.
- `--examples-changed-since=REF` keeps the runs of the examples whose script, notebook or page
  changed since `REF`'s merge base with HEAD, as a pull request's diff counts changes (what `REF`
  gained after the branch point does not count), and every run when `src/`, `pyproject.toml` or
  `uv.lock` changed, or a file under `examples/` that is no example's own script, notebook or
  page (a private helper, data). `REF` must name a commit that shares history with HEAD, and is
  read as one even when it looks like a git option; anything else is a usage error. A change
  counts whether committed, staged or untracked (what git ignores does not), and only below the
  repository root, so a project inside a larger git repository sees its own.
- A repository's own test of an example's results takes the `example_run` fixture, which returns
  the session's one run of the example, `main()`'s value included; no conftest or other plugin is
  loaded, so it is the one fixture beyond pytest's own, and it carries no device marker, its
  example's Devices line deciding. In a directory pytest walks, it is collected by pytest's default
  naming (`test_*.py` or `*_test.py`, `test`-prefixed functions, `Test`-prefixed classes): the
  repository's naming keys do not reach the session. A file named on the command line is collected
  whatever its name. With `--examples-changed-since`, the fixture runs an example whose run the
  change left out when the requesting test's own file changed, or a file under `tests/examples`
  that is no test module (a helper any result test may import), as pytest-picked runs a changed
  test file; otherwise it skips the test. The suite's checks run before any other test. The
  fixture's type is `substrax.testing.example_suite.ExampleRunner`, and
  `ExampleRun.returned_as(shape)` reads `main()`'s value as a `TypedDict` or other type, validated
  strictly as `ChildResult.report_as` reads a child's report, so the test type-checks:

```python
from typing import TypedDict

from substrax.testing.example_suite import ExampleRunner


class Fit(TypedDict):
    relative_error: float


def test_the_fit_reaches_the_true_weights(example_run: ExampleRunner) -> None:
    fit = example_run("training/linear_fit.py").returned_as(Fit)
    assert fit["relative_error"] < 1e-3
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
    texts = documented_texts(ROOT, package="<package>", script_directories=("examples",))
    assert {name: calls for name, text in texts.items() if (calls := unguarded_make_mesh_calls(text))} == {}
```

`documented_texts` reads every docstring under `src/<package>`, the README (or the `pages` given),
every Markdown page under `docs` and every Python file under the script directories. A location
that does not exist raises `FileNotFoundError`, so a mistyped directory fails the check instead of
shrinking what it reads.

## Flax NNX rules and conformance

Three modules hold the checks a Flax NNX library runs over its own code. Each rule and check
names a defect class found in the stack's libraries, and each has a snippet or module that must
pass it and one that must fail it.

**Source rules.** `substrax.testing.nnx_rules` reads a repository's files without importing
them and finds nine defect classes:

- `seed-literal`: `nnx.Rngs(0)`, `jax.random.key(0)`, `np.random.default_rng(7)`, `seed = 0` and
  `seed=0` defaults.
- `rngs-probe`: `"sample" in rngs`, `hasattr(rngs, ...)`.
- `rngs-fallback`: `rngs or nnx.Rngs(...)`.
- `trainable-metadata`: `nnx.Param(..., trainable=False)`, which still trains.
- `salted-hash`.
- `mode-override`: `deterministic: bool = False` on a call path, which decides the mode on every
  call that omits it, whatever `.train()`, `.eval()` or `nnx.view` set. A constructor's default
  is the initial mode, as in `nnx.Dropout(deterministic=False)`, and is allowed.
- `unread-mode-flag`: a mode flag the function never reads, so passing it changes nothing.
- `unreachable-mode`: a mode `.train()` and `.eval()` cannot switch, because they set only
  `deterministic` and `use_running_average`: `self.training = ...`, or a constructor flag that
  only decides at build time.
- `noop-super-init`.

Together the three mode rules describe flax's mode contract: a stochastic leaf holds its mode in
`self.deterministic` or `self.use_running_average` and defines `set_view`, as `nnx.Dropout` and
`nnx.BatchNorm` do, and a composite takes no mode flag and lets `.train()`, `.eval()` and
`nnx.view` reach its leaves.

A repository enables a rule once its sites are fixed:

```python
from pathlib import Path

from substrax.testing.nnx_rules import findings

ROOT = Path(__file__).resolve().parents[1]


def test_library_code_seeds_no_randomness_of_its_own() -> None:
    assert findings(ROOT, ("src/<package>",), rules=("seed-literal",)) == []
```

A rule cannot tell some sites from legitimate uses, such as an in-process cache key built with
`hash` or a fixed key filling a shape-only input. Such a site is exempted on its own line, with
the rule and a reason. An exemption missing either raises `MalformedExemptionError`.

```python
key = jax.random.key(0)  # substrax-allow: seed-literal: shape-only dummy input
```

**Premises.** The rules and checks rest on behaviours of flax, jax and Python. For example,
`"sample" in nnx.Rngs(0)` is `False` although `rngs.sample()` draws, and tree mode refuses a
Variable that two modules share. `substrax.testing.nnx_premises.check_premises()` measures each
premise on what is installed. If one no longer holds, it raises `PremiseChangedError`, naming the
premise and the rules resting on it. Run it where the newest jax and flax a repository admits are
installed, so a release that changes a premise is found before code relying on it misbehaves.

**Conformance.** `substrax.testing.nnx_conformance` checks the properties every module owes under
NNX's transforms:

- state fixed at construction;
- Variables holding arrays;
- tree-mode splitting;
- a deterministic `.eval()`, or for a call that draws, an output that follows the caller's key;
- dropout under `.train()`;
- batch statistics that move in training only;
- `nnx.view` computing what `.eval()` and `.train()` do, which a leaf owning its mode without
  `set_view`, or a module storing an `nnx.Rngs`, fails;
- `nnx.jit` matching eager with one trace;
- independent rows under `nnx.vmap`;
- gradients that are finite and match float64 finite differences.

A repository describes each module class as a `Case` and runs the checks each case owes.
`module_classes(package)` is the denominator, so a class without a case is found:

```python
import pytest

from substrax.testing.nnx_conformance import CHECKS, Case

CASES: dict[str, Case] = {...}  # one per class module_classes(<package>) finds


@pytest.mark.parametrize("check", sorted(CHECKS))
@pytest.mark.parametrize("name", sorted(CASES))
def test_module_conforms(name: str, check: str) -> None:
    run, applies = CHECKS[check]
    if applies(CASES[name]):
        run(CASES[name])
```

A check comparing two computations of one function allows `DEPTH * eps(dtype) * scale`. Both
computations start from cloned state, so a module that draws randomness in the wrong mode fails
`eval_deterministic` alone rather than every check.

A module that takes its randomness at call time (`nnx.Dropout(rate)` with `rngs=` passed on each
call, the style flax recommends for new code) names its streams in the case:

```python
Case(
    build=lambda rngs: Block(rngs=rngs),
    inputs=inputs,
    call=lambda module, x, *, rngs: module(x, rngs=rngs),
    has_dropout=True,
    rng_streams=("dropout",),
)
```

Every check then calls the case with an `nnx.Rngs` of those streams, built from a key the check
chooses and the same for two computations it compares. `train_dropout_varies` asks that different
keys give different outputs and the same key the same output. Its same-key run starts with every
PRNG key the module holds changed, so randomness the key does not control fails it: a dropout leaf
the key was not passed to, noise from a stream or raw key the module holds, a host random
generator. A legacy key held as `uint32` data (`substrax.rng.is_legacy_key`) is changed too.
`eval_deterministic` gives its two calls different keys, so an eval output that depends on the
key fails it.

A call whose job is to draw (a distribution's `sample`, a model's `generate`) depends on the key
in every mode, so its case says so:

```python
Case(
    build=lambda rngs: Prior(rngs=rngs),
    inputs=inputs,
    call=lambda module, x, *, rngs: module.sample(x, rngs=rngs),
    is_rowwise=False,
    rng_streams=("sample",),
    draws=True,
)
```

Such a case owes `output_follows_key` in place of `eval_deterministic`: in `.eval()` and in
`.train()`, different keys give different outputs and the same key the same output, with every
PRNG key the module holds changed for the same-key run. A sampler that ignores its key, draws
from a stream or a key it holds, or returns no sample in one mode fails it. A drawing case names
its `rng_streams` and does not claim `is_rowwise` (one key drawn for a batch and the same key
drawn per row give different samples); a case built otherwise is refused.

::: substrax.testing

::: substrax.testing.compiles

::: substrax.testing.jax_calls

::: substrax.testing.gradients

::: substrax.testing.source_scans

::: substrax.testing.nnx_rules

::: substrax.testing.nnx_premises

::: substrax.testing.nnx_conformance

::: substrax.examples

::: substrax.recipes

::: substrax.testing.example_suite

::: substrax.testing.compute
