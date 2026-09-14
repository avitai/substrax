# Migrating from the sibling packages

Every module in Substrax was moved from calibrax, datarax, artifex or opifex. This page
maps the old import paths to the new ones; the [changelog](https://github.com/avitai/substrax/blob/main/CHANGELOG.md)
lists what changed in each module and what was not carried over.

| Old import | New import |
| --- | --- |
| `datarax.distributed.device_placement` | `substrax.devices` |
| `datarax.distributed.device_mesh.DeviceMeshManager` | `substrax.mesh.DeviceMeshManager` |
| `datarax.distributed.sharding` | `substrax.mesh` |
| `artifex.generative_models.scaling.sharding` | `substrax.mesh` |
| `datarax.distributed.data_parallel` | `substrax.spmd` |
| `datarax.distributed.metrics` | `substrax.spmd` |
| `opifex.core.training.components.checkpoint_store` | `substrax.checkpoint` |
| `artifex.generative_models.training.callbacks.base` | `substrax.callbacks` |
| `opifex.core.training.callbacks.EarlyStopping` | `substrax.callbacks.EarlyStopping` |
| `artifex.generative_models.training.callbacks.early_stopping.EarlyStopping` | `substrax.callbacks.EarlyStoppingCallback` |
| `artifex.generative_models.utils.logging` | `substrax.tracking` |
| `opifex.setup_jax_optimization` | `substrax.runtime.apply_runtime` with a `JaxRuntime` |
| `opifex.core.device_utils.configure_jax_precision` | `substrax.runtime.apply_runtime(JaxRuntime(enable_x64=...))` |
| `artifex.generative_models.core.jax_config.configure_jax` | `substrax.runtime.apply_runtime` with a `JaxRuntime` |
| `datarax.performance.xla_optimization.apply_xla_flags` | `substrax.runtime.merge_xla_flags`, or `JaxRuntime(xla_flags=...)` |
| `datarax.performance.xla_optimization.XLAOptimizer` | `substrax.runtime.apply_runtime` with a `JaxRuntime` |

`substrax.runtime` carries no per-backend flag presets: the fast-math CPU flags, GPU Triton
flags and forced single-device count that `setup_jax_optimization`, `get_xla_flags` and
`XLAOptimizer` added are not applied by default. Pass the flags a run needs in
`JaxRuntime(xla_flags=...)`. A flag already set to a different value raises instead of being
appended after it.

Two classes were named `EarlyStopping` with incompatible interfaces; the trainer-hook
one is now `EarlyStoppingCallback`. `ReduceLROnPlateau` has no replacement here: use
`optax.contrib.reduce_on_plateau`.
