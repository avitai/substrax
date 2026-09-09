# Changelog

All notable changes to Substrax are tracked here.

This project follows the spirit of [Keep a Changelog](https://keepachangelog.com/)
and uses semantic versioning while the public API stabilizes.

## [Unreleased]

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
