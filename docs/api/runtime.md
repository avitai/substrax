# Runtime

`substrax.runtime` declares the settings a JAX process starts with and applies them at the
point where they still take effect.

```python
import os
import subprocess
import sys

from substrax.runtime import JaxRuntime, apply_runtime, runtime_environment

runtime = JaxRuntime(platforms=("cpu",), cpu_devices=8, enable_x64=True)

# A child process on eight emulated CPU devices.
env = {**os.environ, **runtime_environment(runtime, os.environ)}
subprocess.run([sys.executable, "train.py"], env=env, check=True)

# This process, before it imports jax.
apply_runtime(runtime)

import jax

assert jax.device_count() == 8
```

`runtime_environment` renders each field as the variable jax or XLA reads when it starts:

| Field | Variable |
| --- | --- |
| `platforms` | `JAX_PLATFORMS` |
| `cpu_devices` | `JAX_NUM_CPU_DEVICES`, which jax prefers over `--xla_force_host_platform_device_count` |
| `enable_x64` | `JAX_ENABLE_X64` |
| `matmul_precision` | `JAX_DEFAULT_MATMUL_PRECISION` |
| `compilation_cache_dir` | `JAX_COMPILATION_CACHE_DIR` |
| `xla_flags` | `XLA_FLAGS`, merged by flag name |
| `preallocate` | `XLA_PYTHON_CLIENT_PREALLOCATE` |
| `memory_fraction` | `XLA_CLIENT_MEM_FRACTION` |

A requested XLA flag that is already set to a different value raises `XlaFlagConflictError`,
so a device count or GPU flag exported by the caller is never silently replaced. A memory
fraction raises while the deprecated `XLA_PYTHON_CLIENT_MEM_FRACTION` is still set, because
jax refuses both at once. `JaxRuntime` applies no per-backend flag presets: `xla_flags` is empty
unless a run names the flags it needs.

Once jax is imported, `apply_runtime` applies the CPU device count, 64-bit types, matmul
precision and compilation cache directory through `jax.config`; jax refuses a new device count
after its first operation, and that refusal is raised as `RuntimeConfigurationError`.
Platforms, XLA flags and accelerator memory settings are read only when jax starts its
backends, and jax gives no public signal of whether that has happened, so after the import
they raise instead of being written where they may change nothing.

A test suite chooses its backend and emulated CPU devices before anything imports jax:

```python
# conftest.py
import importlib.util
import os

from substrax.runtime import resolve_test_runtime, runtime_environment

runtime = resolve_test_runtime(
    os.environ,
    prefix="DATARAX_TEST_",
    cuda_plugin_available=importlib.util.find_spec("jax_cuda12_plugin") is not None,
)
os.environ.update(runtime_environment(runtime, os.environ))
```

Tests run on eight emulated CPU devices unless `DATARAX_TEST_JAX_PLATFORMS` asks for an
accelerator or `DATARAX_TEST_DEVICE_COUNT` names another count; an inherited
`JAX_PLATFORMS` does not move them onto a GPU.

Entry points configure logging inside `main()`, never at import:

```python
from substrax.runtime import configure_entry_point_logging


def main() -> None:
    configure_entry_point_logging()
```

A second call replaces only the handler the first one installed, so it needs no `force=True`
and leaves pytest's log capture in place. Importing `substrax.runtime` imports no jax and
changes no process state.

::: substrax.runtime
