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

Two classes were named `EarlyStopping` with incompatible interfaces; the trainer-hook
one is now `EarlyStoppingCallback`. `ReduceLROnPlateau` has no replacement here: use
`optax.contrib.reduce_on_plateau`.
