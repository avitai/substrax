# Changelog

All notable changes to Substrax are tracked here.

This project follows the spirit of [Keep a Changelog](https://keepachangelog.com/)
and uses semantic versioning while the public API stabilizes.

## [Unreleased]

### Changed

- Requires `jax>=0.11.1,<0.11.2` and the same for jaxlib and `jax[cuda12]`: jax 0.11.2 renamed
  `jax.experimental.hijax.HiPrimitive`, which flax 0.12.9 imports at module load, so a fresh
  environment resolving both fails on `import flax.nnx`. The cap lifts with the flax release
  that imports jax 0.11.2, which a test detects; the lock holds 0.11.1 and does not move.
- The format-2 checkpoint fixtures are generated, not committed: `scripts/make_format2_fixtures.py`
  writes them with substrax 0.1.5 in an isolated environment whose every package is the
  committed lock `scripts/format2_fixture_requirements.txt`, CI runs it before the tests, and
  the migration tests fail naming that command when the roots are missing.

## [0.1.10] - 2026-09-17

### Added

- Checkpoint format 3. A checkpoint is a step holding named items, the things a training
  loop owns (`model`, `optimizer`, `rng`, `data_iterator`, `extensions`), each an Orbax
  pytree item, beside one `CheckpointMetadata` record (format, format version, step, epoch,
  item names, library versions, producer, metrics, extra, creation time).
  On disk each item holds its pytree under a single `tree` node, so an item that is a bare
  array (the `rng` key) is accepted by the Orbax floor.
  `OrbaxCheckpointStore(directory, *, max_to_keep=5)` opens Orbax on first use, so nothing is
  created before the first save; `save(step, items, *, epoch, metrics, producer, extra,
  overwrite)` refuses an existing step unless `overwrite=True` and a step below the latest,
  raising `CheckpointNotWrittenError` with the step, the latest step and the reason;
  `restore(step, *, templates, legacy_layout)` returns a `Checkpoint` with every templated
  item placed on its template's device, raising `CheckpointNotFoundError` for a missing step
  and `UnsupportedCheckpointError` for a newer format; `read_metadata`, `best_step(metric,
  mode="min" | "max")` and `delete` (which raises for a missing step) complete the protocol.
- A migration registry. A format-2 checkpoint (substrax 0.1.5 to 0.1.9) is upgraded in
  memory on restore, its payload split into items by a `LegacyLayout`: the module-only
  layout (`MODULE_ONLY_FORMAT2`) by default, or the producer's own. `upgrade_checkpoints`
  and `python -m substrax.checkpoint upgrade SOURCE DESTINATION` rewrite a root in the
  current format into a new root, never in place. `tests/checkpoint/fixtures/format2` holds
  one checkpoint per producer layout, written by substrax 0.1.5 through
  `scripts/make_format2_fixtures.py`.
- `resolve_checkpoint_dir(checkpoint_dir, run_dir)`: an explicit directory, else the run
  directory's `checkpoints` subdirectory, created by nothing but the first save.
- CI runs `tests/checkpoint` against the Orbax floor, `orbax-checkpoint==0.11.33`, beside the
  locked version.

### Removed

- `ModelLike` and the Flax `TrainState` support (`create_train_state`, `save_train_state`,
  `restore_train_state`); the store is pytree-only, and a trainer writes its optimizer state
  as the `optimizer` item.
- `save(model, step, loss, physics_metadata=..., additional_metadata=...)`; the loss is
  `metrics={"loss": ...}` and the rest goes in `extra`, under keys that are not fields of
  the record.
- `restore(target_model, step, return_original_on_missing=..., restrict_to_nnx_module=...)`;
  a missing step raises `CheckpointNotFoundError`, and templates replace the target.
- `delete` no longer returns a boolean; a missing step raises.

## [0.1.9] - 2026-09-17

### Changed

- `substrax.optim.OptimizerConfig` refuses an `optimizer_type` it does not build and a
  `gradient_clip_norm` or `gradient_clip_value` that is not positive when it is constructed;
  the unknown type used to pass construction and fail only in `create_transformation`, and a
  zero or negative clip was accepted.

## [0.1.8] - 2026-09-17

### Added

- `substrax.rng`: `key_from(rng, *, streams, context)` takes a key from the first stream an
  `nnx.Rngs` holds, in the order given, or returns a key given directly, and raises
  `MissingRngStreamError` or `TypeError` instead of falling back to a default seed;
  `rngs_from_seed(seed, streams)` derives each stream from the seed and the stream's name, so
  adding or reordering streams leaves the others' keys unchanged; `split_key(key, num)` refuses
  `num` below one; `fold_in_name(key, name)` folds in the name's BLAKE2b digest, the same in
  every interpreter. The pytest plugin gains `rng_key` (a typed key from seed 42) and `rngs`
  (an `nnx.Rngs` from the same seed with `default`, `params`, `dropout` and `sample` streams).
  This is the home of artifex's `extract_rng_key`, DiffBio's `ensure_rngs` and `get_rng_key`
  and datarax's `create_rngs`, each removed in its own release.
- `substrax.optim`: `OptimizerConfig`, a frozen specification in optax's terms (`optimizer_type`
  among `adam`, `adamw`, `sgd`, `rmsprop`, `adagrad`, `lamb`, `radam` and `nadam`,
  `learning_rate` as a constant or an `optax.Schedule`, `b1`, `b2`, `eps`, `momentum`,
  `weight_decay`, `weight_decay_filter`, one clip field and `wrt`), which refuses a decay
  for an optimizer without decoupled decay, momentum for one without it, both clip fields,
  a string filter and a non-positive constant rate; `create_transformation` and
  `create_optimizer`, which pass a schedule as the base optimizer's learning rate and the
  filter as a static mask of Python booleans, so the optimizer state has one structure
  whatever the filter, and refuse a filter that selects no parameter; `EXCLUDE_BIAS_AND_NORM_SCALE`;
  `current_learning_rate`, the rate the last update applied, read on device through
  `optax.inject_hyperparams`. optax is a declared dependency (`optax>=0.2.8`) rather than a
  constraint. artifex's `OptimizerConfig` and `create_optimizer`, opifex's `OptimizationConfig`
  and DiffBio's `MiniBatchConfig` move onto this spec in their own releases.

## [0.1.7] - 2026-09-14

### Changed

- `substrax.spmd.create_data_parallel_sharding` is annotated to return the `NamedSharding` it
  builds, so callers read `.mesh` and `.spec` without a cast, and it builds that sharding
  through `substrax.mesh.create_named_sharding`.
- The numpy requirement allows numpy 2.5 (`numpy>=1.24,<2.6.0`), and the lock resolves 2.5.3.
- `substrax.spmd.place_batch_on_shards` documents that it places host batches outside `jax.jit`;
  inside a traced function its result is not placed on the sharding.

### Fixed

- `substrax.spmd.place_batch_on_shards` returned NumPy array leaves unplaced, so a host batch
  from a data loader stayed on the host. It could not assemble a global batch across
  processes either, because it put each leaf with `jax.device_put`. It now passes every
  array leaf, NumPy or `jax.Array`, to one `jax.make_array_from_process_local_data` call.
  On one process that call is a single batched `jax.device_put` instead of one transfer per
  leaf. On several processes each process passes the slice it loaded, and jax builds the
  global array.

### Removed

- The documentation's migration page. A name that moved into Substrax is recorded in the
  CHANGELOG of the package that removed it, which names its Substrax replacement. The runtime
  page now states that `JaxRuntime` applies no per-backend XLA flag presets.
- `substrax.devices.distribute_batch` and `DevicePlacement.distribute_batch`, which duplicated
  batch placement. Use `substrax.spmd.place_batch_on_shards(batch, sharding)`, which places the
  batch's array leaves and returns other leaves as they are.

## [0.1.6] - 2026-09-14

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
- `substrax.testing.run_example` runs one example in a fresh interpreter from the repository root,
  with `AVITAI_OUTPUT_DIR` set to a given directory. It runs the file as a script, or calls its
  `main()` and decodes the returned summary from JSON. A timeout raises `ExampleTimeoutError`.
  `discover_examples` lists example scripts, skipping names that start with `_`.
  `unavailable_reason` matches a failed run's standard error against messages that mean the
  example cannot run here. The pytest plugin gains an `output_dir` fixture.

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
