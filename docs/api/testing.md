# Testing

`substrax.testing` holds the test infrastructure JAX packages share: fresh-interpreter runs with a
chosen JAX configuration, trace counts, example runs, and an opt-in pytest plugin. Install it with
`substrax[testing]`.

```python
from substrax.runtime import JaxRuntime
from substrax.testing import run_python

result = run_python(
    "import json, jax; print(json.dumps(jax.device_count()))",
    runtime=JaxRuntime(cpu_devices=8),
    timeout=120,
)
assert result.check().last_json() == 8
```

`run_python` runs this interpreter on code or a script. The child's environment drops every
`JAX_*` and `XLA_*` variable the parent inherited, then applies `env` and the `runtime`. Unless
either of them chooses otherwise, the child runs on the CPU backend without preallocation, so a
test never starts an accelerator it did not ask for. `check()` raises `ChildFailedError` with
the end of the child's standard error, and `last_json()` parses the last line the child printed.
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

## Running examples

`discover_examples` lists a repository's example scripts, and `run_example` runs one in a fresh
interpreter from the repository root:

```python
from pathlib import Path

import pytest

from substrax.testing import discover_examples, run_example

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = discover_examples(ROOT / "examples")


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda path: path.relative_to(ROOT).as_posix())
def test_example_runs(path: Path, output_dir: Path) -> None:
    run = run_example(
        path, repo_root=ROOT, output_dir=output_dir, timeout=600, call_main=True
    )
    assert run.result.check().returncode == 0
    assert run.summary["relative_l2_error"] < 1e-2
```

- `discover_examples` skips every file or directory whose name starts with `_`, such as
  `__init__.py`, `_common` helpers and `_templates`. `include` narrows the list further, for
  example to a numbering rule.
- `call_main=True` loads the example under a name other than `__main__`, calls `main()` and decodes
  its return value from JSON into `run.summary`. 0-d numpy and jax values become Python numbers,
  and a value JSON cannot hold fails the run, naming its key. `call_main=False` runs the file as
  a script. In both modes the example's own directory comes first on `sys.path`, and it sees no
  command-line arguments, as in a script run.
- The child runs through `run_python`, with `AVITAI_OUTPUT_DIR` set to `output_dir`. An example that
  resolves its outputs with `resolve_output_dir` never writes into the repository, and logging or
  64-bit settings made at import end with the child.
- A timeout raises `ExampleTimeoutError`, an `AssertionError` naming the budget and the end of the
  example's standard error. A timeout is a failure, not a skip.
- `unavailable_reason(run, signatures)` returns the first of a repository's messages found in a
  failed run's standard error, for skipping an example whose dataset is not configured here.

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

::: substrax.testing
