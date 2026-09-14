# Changelog

All notable changes to Substrax are tracked here.

This project follows the spirit of [Keep a Changelog](https://keepachangelog.com/)
and uses semantic versioning while the public API stabilizes.

## [Unreleased]

### Added

- `substrax.runtime` declares the settings a JAX process starts with and applies them where
  they still take effect. `JaxRuntime` holds the backends, CPU device count, 64-bit types,
  matmul precision, compilation cache directory, XLA flags and accelerator memory settings,
  and refuses invalid values. `runtime_environment` renders them as the variables jax and XLA
  read at start-up, using `XLA_CLIENT_MEM_FRACTION`, the name jaxlib reads, for the memory
  fraction. `apply_runtime` writes that environment before jax is imported. After the import
  it applies the device count, 64-bit types, matmul precision and cache directory through
  `jax.config`, and raises for platforms, XLA flags and memory settings, which jax reads only
  when its backends start. `merge_xla_flags` merges by flag name and raises on a conflicting
  value instead of letting the last one win. `resolve_test_runtime` chooses a test run's
  backend and emulated CPU devices, and `configure_entry_point_logging` sets up the root
  logger from an entry point without `force=True`.
- `substrax.artifacts.resolve_output_dir` chooses where a run writes its outputs: an explicit
  directory, then `$AVITAI_OUTPUT_DIR/<name>`, then a directory created once per process under
  the system temporary directory with `tempfile.mkdtemp`. It never defaults into the working
  tree, refuses a relative `AVITAI_OUTPUT_DIR` and a name that leaves the output directory, and
  creates the directory it returns.
- `substrax.testing`, installed with the new `testing` extra, holds shared test infrastructure.
  `run_python` runs code or a script in a fresh interpreter whose JAX settings the test
  chooses. It drops inherited `JAX_*` and `XLA_*` variables, and defaults to the CPU backend
  without preallocation. It returns a `ChildResult` with `check()` and `last_json()`.
  `cuda_is_visible` probes the CUDA backend in such a child. `restored_jax_config` sets back the
  global jax configuration a block changed. The opt-in plugin `substrax.testing.pytest_plugin`
  adds the `x64`, `devices` and `accelerator` markers, fails a test that changes jax's global
  configuration as jax's own `JaxTestCase` does, and fails a session left in x64 by a module at
  import.
- `substrax.testing.TraceCounter` counts how often a function's Python body runs, which under
  `jax.jit` and `nnx.jit` is once per trace. `expect(new_traces=n)` raises `RetraceError` unless
  exactly `n` traces happen inside its block. Each counter is independent, with no global
  registry to clear.

### Fixed

- The module docstring of `substrax.mesh.rules` and the `spmd_train_step` example built their
  meshes with `jax.make_mesh` and no `axis_types`. From jax 0.11 that gives `Explicit` axes, under
  which the backward pass of a batch-sharded step raises. Both examples now use
  `DeviceMeshManager.create_device_mesh`, which builds `Auto` axes. A contract test fails on any
  `make_mesh(` call without `axis_types` in a docstring, the README or a docs page.

## [0.1.5] - 2026-09-09

### Fixed

- `OrbaxCheckpointStore.restore(target)` restores every array onto its target leaf's device
  placement and dtype. The template alone gave Orbax only the tree structure, so it fell back
  to the sharding file written at save time; a checkpoint saved on the second of two CPU
  devices then failed to restore in a process exposing one ("Device cpu:1 was not found in
  jax.local_devices()") even though the target lived on the available device. The per-leaf
  restore arguments are now built from the target. A target-free restore still comes back as
  stored. The two-device save / one-device restore is pinned in a fresh interpreter, with
  the template-free failure as its control; accelerator-to-CPU restoration has its own
  `gpu`-marked test.

## [0.1.4] - 2026-09-09

### Fixed

- `DeviceMeshManager.create_device_mesh` builds every axis as `AxisType.Auto` unless the new
  `axis_types` argument says otherwise, on both the `jax.make_mesh` path and the explicit-devices
  path. jax 0.11 made `jax.make_mesh` default to `Explicit` axes, under which the backward pass
  of any layer over a batch sharded along `data` raised `ShardingTypeError` ("Contracting
  dimensions are sharded"); the data-parallel helpers here leave that inference to XLA, which
  needs `Auto`. A one-device mesh now differentiates through a batch-sharded linear layer, and
  the explicit-axes failure is pinned as the control.

## [0.1.3] - 2026-09-09

### Added

- `WandbLogger` forwards keyword arguments it does not name to `wandb.init`, so a training
  callback can pass `mode="offline"` or `resume="allow"` without reaching for the SDK
  itself.
- `OrbaxCheckpointStore(max_to_keep=None)` keeps every checkpoint, the policy a callback
  that saves only on improvement needs when asked to keep them all.

## [0.1.2] - 2026-09-09

### Changed

- `OrbaxCheckpointStore.restore` without a target returns the payload as it was
  stored, arrays and plain leaves alike, instead of only the metadata: a checkpoint
  describes its own tree, so a consumer that validates structure itself (datarax's
  `set_state`) restores without a template whose list lengths would have to match.
- A checkpoint that cannot be read into the given target raises (`ValueError` for a tree
  mismatch) instead of being reported as missing.

## [0.1.1] - 2026-09-09

### Changed

- `OrbaxCheckpointStore` writes its payload with Orbax's `PyTreeSave` and reads it with
  `PyTreeRestore` instead of `StandardSave`/`StandardRestore`. `StandardSave` rejects
  every leaf that is not an array, so a dictionary carrying an iterator position, a
  sampler's repr or a typed PRNG key could not be saved; `PyTreeSave` round-trips
  arrays, typed keys and plain-Python leaves alike, still without pickle.
- `save` takes `loss` as an optional argument. A payload with no training loss, such as
  data-iterator state, records none, and `best_step` passes over checkpoints that do
  not carry the requested metric.

## [0.1.0] - 2026-09-08

The first release. Every module is the implementation one of the sibling packages
(calibrax, datarax, artifex, opifex) carried before, moved here with its tests and
refactored against the shared engineering standards; the siblings switch to these
modules in their next releases.

### Added

- `substrax.devices`: `DeviceInfo`, `DeviceKind` and `detect_devices()`;
  `DevicePlacement`, `HardwareType`, `BatchSizeRecommendation`, `place_on_device`,
  `distribute_batch` and `get_batch_size_recommendation` (from
  `datarax.distributed.device_placement`).
- `substrax.mesh`: `DeviceMeshManager` (from `datarax.distributed.device_mesh`);
  `MeshRules`, `data_parallel_rules`, `fsdp_rules`, `create_named_sharding` and
  `partition_spec_for_names` (from `datarax.distributed.sharding`); `ShardingConfig`,
  `ParallelismConfig`, `ShardingStrategy` and the data-parallel, FSDP, tensor-parallel,
  pipeline-parallel and multi-dimensional strategies (from `artifex.generative_models.scaling.sharding`).
- `substrax.spmd`: `create_data_parallel_sharding`, `place_batch_on_shards`,
  `place_nnx_state_on_shards`, `spmd_train_step` and `reduce_gradient_tree` (from
  `datarax.distributed.data_parallel`); the metric collectives `reduce_mean`,
  `reduce_sum`, `reduce_max`, `reduce_min`, `reduce_custom`, `reduce_mean_collective`,
  `reduce_sum_collective`, `all_gather` and `collect_from_devices` (from
  `datarax.distributed.metrics`).
- `substrax.checkpoint`: the `CheckpointStore` protocol and `OrbaxCheckpointStore`
  over `orbax.checkpoint.CheckpointManager` (from
  `opifex.core.training.components.checkpoint_store`); the on-disk layout is the one
  opifex 0.2 writes.
- `substrax.callbacks`: `TrainingCallback`, `BaseCallback`, `CallbackList` and
  `TrainerLike` (from `artifex.generative_models.training.callbacks.base`); `BestMetricTracker`,
  `PlateauMode` and `EarlyStopping` (from `opifex.core.training.callbacks`);
  `EarlyStoppingCallback` with `EarlyStoppingConfig` (from
  `artifex.generative_models.training.callbacks.early_stopping`), now composing `BestMetricTracker`.
- `substrax.tracking`: `Logger`, `ConsoleLogger`, `FileLogger`, `create_logger`,
  `WandbLogger` and `MLFlowLogger` (from `artifex.generative_models.utils.logging`).
  Weights & Biases, MLflow and matplotlib ship in the `wandb`, `mlflow` and `plots`
  extras and load when a logger is constructed; importing `substrax.tracking` imports
  none of them.
- Package scaffold shared with the sibling repositories: `setup.sh`, `activate.sh`,
  `scripts/setup_env.py`, the six CI workflows, the pre-commit gates (including the
  test suite as a commit hook), `scripts/derive_status.py --check` against the README,
  trusted publishing.

### Changed, relative to the sources

- `FileLogger` writes its metrics CSV in long form (`timestamp,step,name,value`), so a
  changing scalar set no longer produces rows that disagree with the header.
- `WandbLogger.log_text` HTML-escapes the text; logging on a finished W&B run raises
  `RuntimeError`.
- The blanket `except Exception` blocks around the W&B and MLflow SDK calls are gone;
  a missing `plots` extra is the one condition that is logged as a warning.
- `EarlyStoppingCallback` keeps its counters in a `BestMetricTracker` instead of its own
  improvement bookkeeping; `wait_count`, `best_score` and `should_stop` are unchanged.
- `MultiDimensionalStrategy` takes a `Mapping[str, ShardingStrategy]`.
- Every configuration dataclass is frozen, keyword-only and slotted.

### Not carried over

- `datarax.distributed.data_parallel.data_parallel_train_step` (`jax.pmap`) and the
  `lax` gradient collective: no consumer outside their own tests.
- `datarax.distributed.device_placement.prefetch_to_device`: built on datarax's
  prefetcher and stays in datarax.
- `artifex.generative_models.scaling.mesh_utils`: no consumer outside its own tests.
- `artifex`'s `ReduceLROnPlateau`: use `optax.contrib.reduce_on_plateau`.
- The logger methods' `**kwargs`, `create_logger(log_to_console=...)` (a no-op), the
  `console_log` parameters, `WandbLogger(resume=..., anonymous=...)` and the run-id
  file, and `MLFlowLogger.log_model` (which pickled arbitrary objects through
  `mlflow.pyfunc`): no caller in artifex, opifex or their examples.

[Unreleased]: https://github.com/avitai/substrax/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/avitai/substrax/releases/tag/v0.1.0
