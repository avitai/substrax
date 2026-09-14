# Substrax

Shared training and hardware infrastructure for the Avitai JAX stack. Substrax sits at
the bottom of the dependency chain — `substrax → calibrax → datarax → artifex → opifex` —
and holds the code those packages used to carry separately.

| Subpackage | What it owns |
| --- | --- |
| `substrax.runtime` | JAX process settings applied before or after jax is imported, XLA flags merged by name, test-run device emulation, entry-point logging |
| `substrax.artifacts` | Output directories from an argument, `AVITAI_OUTPUT_DIR` or a per-run temporary directory, never the working tree |
| `substrax.testing` | Opt-in test infrastructure: fresh-interpreter runs with a chosen JAX configuration, and a pytest plugin with device markers and jax configuration isolation |
| `substrax.devices` | Device information and device placement |
| `substrax.mesh` | Device meshes, mesh rules, partition-spec helpers, sharding strategies |
| `substrax.spmd` | Data-parallel placement, `spmd_train_step`, gradient reduction, collectives |
| `substrax.checkpoint` | The `CheckpointStore` protocol and its Orbax implementation |
| `substrax.callbacks` | Training callbacks, `BestMetricTracker`, `EarlyStopping` |
| `substrax.tracking` | Step-wise experiment tracking backends |

Each subpackage has an [API page](api/devices.md).

## Installation

```bash
uv add substrax
```

Python 3.12 or later, `jax>=0.11.1`, `flax>=0.12.9`, `orbax-checkpoint>=0.11.33`.
