# Testing

`substrax.testing` holds the test infrastructure JAX packages share: fresh-interpreter runs with a
chosen JAX configuration, and an opt-in pytest plugin. Install it with `substrax[testing]`.

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
