# Substrax

**JAX/Flax NNX training infrastructure: device detection and placement, device meshes and
SPMD sharding (data, FSDP, tensor and pipeline strategies), an Orbax checkpoint store that
restores onto the current devices, early stopping and callbacks, and W&B/MLflow logging.**
It is the shared layer of the Avitai JAX stack.

[![CI](https://github.com/avitai/substrax/actions/workflows/ci.yml/badge.svg)](https://github.com/avitai/substrax/actions/workflows/ci.yml)
[![Build](https://github.com/avitai/substrax/actions/workflows/build-verification.yml/badge.svg)](https://github.com/avitai/substrax/actions/workflows/build-verification.yml)
[![Docs](https://github.com/avitai/substrax/actions/workflows/docs.yml/badge.svg)](https://github.com/avitai/substrax/actions/workflows/docs.yml)
[![PyPI](https://img.shields.io/pypi/v/substrax.svg)](https://pypi.org/project/substrax/)

[Documentation](https://substrax.readthedocs.io) · [Changelog](CHANGELOG.md) · [Issues](https://github.com/avitai/substrax/issues)

> **Research preview.** Substrax is under rapid iteration and the API will change while
> the sibling packages migrate onto it. Pin a version.

## Where it sits

Substrax is the bottom of the Avitai dependency chain and depends on none of the siblings:

```text
substrax → calibrax → datarax → artifex → opifex
```

It holds the code those packages used to carry separately, so that each concern has one
home and one test suite:

| Subpackage | What it owns |
| --- | --- |
| `substrax.devices` | `detect_devices()` (platform, device kind, count), device placement, the batch-size recommendation table |
| `substrax.mesh` | Device meshes with `Auto` axes by default, mesh rules and partition-spec helpers, sharding strategies (data, FSDP, tensor, pipeline, multi-dimensional) on `flax.nnx.spmd` |
| `substrax.spmd` | Data-parallel sharding and batch placement, `spmd_train_step`, gradient reduction and collectives |
| `substrax.checkpoint` | One `CheckpointStore` protocol and one Orbax implementation, `OrbaxCheckpointStore`, over `CheckpointManager` |
| `substrax.callbacks` | The training-callback protocol, `CallbackList`, `BestMetricTracker`, `EarlyStopping` and `EarlyStoppingCallback` |
| `substrax.tracking` | Step-wise experiment logging with console, file, Weights & Biases and MLflow backends |

Not in Substrax: optimizers and schedules (optax), loss scaling and gradient accumulation
(`flax.training.dynamic_scale.DynamicScale`, `optax.MultiSteps`), profiling and hardware
spec tables (calibrax), data pipelines (datarax), models and trainers (artifex, opifex).

## Installation

```bash
uv add substrax          # or: pip install substrax
uv add "substrax[wandb]"  # Weights & Biases backend
uv add "substrax[mlflow]" # MLflow backend
```

Substrax requires Python 3.12 or 3.13, `jax>=0.11.1`, `flax>=0.12.9` and
`orbax-checkpoint>=0.11.33`. The `cuda12` and `metal` extras select the JAX backend.

## Quick start

One data-parallel step over every visible device, a checkpoint, and an early-stopping
decision. The same code runs on one CPU; on several devices the batch is sharded on its
leading axis and XLA inserts the gradient all-reduce.

```python
import jax
import jax.numpy as jnp
import optax
from flax import nnx

from substrax.callbacks import EarlyStopping
from substrax.checkpoint import OrbaxCheckpointStore
from substrax.devices import detect_devices
from substrax.mesh import DeviceMeshManager
from substrax.spmd import create_data_parallel_sharding, place_batch_on_shards, spmd_train_step

info = detect_devices()  # DeviceInfo(platform='cpu', kind=<DeviceKind.CPU>, count=1, ...)

model = nnx.Linear(8, 1, rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

mesh = DeviceMeshManager.create_device_mesh({"data": info.count})
sharding = create_data_parallel_sharding(mesh)
batch = place_batch_on_shards(
    {"x": jnp.ones((32, 8)), "y": jnp.zeros((32, 1))},
    sharding,
)


def loss_fn(model: nnx.Module, batch: dict[str, jax.Array]) -> jax.Array:
    return jnp.mean((model(batch["x"]) - batch["y"]) ** 2)


stopper = EarlyStopping(patience=3, min_delta=1e-4)
with OrbaxCheckpointStore("checkpoints/quick-start", max_to_keep=2) as store:
    for step in range(5):
        with jax.set_mesh(mesh):
            loss = spmd_train_step(model, optimizer, loss_fn, batch)
        store.save(model, step, float(loss))
        stopper.update(float(loss))  # True when the loss improved on the best so far
        if stopper.should_stop:
            break

    restored, metadata = store.restore(model, store.latest_step())
    assert metadata["loss"] == float(loss)
```

## The subpackages

### Devices

`detect_devices()` is the one reading of the hardware the whole stack shares.

```python
from substrax.devices import DeviceKind, detect_devices

info = detect_devices()
assert info.platform in {"cpu", "gpu", "tpu", "metal"}
if info.kind is DeviceKind.GPU:
    print(f"{info.count} GPU(s): {info.device_kinds}")
```

`place_on_device(pytree, device)` moves a pytree, and `get_batch_size_recommendation()`
reads the per-hardware batch-size table.

### Mesh and SPMD

`DeviceMeshManager.create_device_mesh` takes the mesh shape as a mapping from axis name
to size and builds every axis as `Auto` unless `axis_types` says otherwise: with jax 0.11
that is what lets XLA infer the sharding of the backward pass over a batch sharded on the
`data` axis.

```python
import jax
from substrax.mesh import DeviceMeshManager, data_parallel_rules, create_named_sharding
from substrax.spmd import create_data_parallel_sharding

mesh = DeviceMeshManager.create_device_mesh({"data": jax.device_count()})
print(DeviceMeshManager.get_mesh_info(mesh))  # {'total_devices': 1, 'axes': {'data': 1}} on one device

batch_sharding = create_data_parallel_sharding(mesh)      # leading axis over "data"
replicated = create_named_sharding(mesh, None)             # every device holds a copy
rules = data_parallel_rules()                              # MeshRules for nnx.spmd
```

The strategies in `substrax.mesh.strategies` (`DataParallelStrategy`, `FSDPStrategy`,
`TensorParallelStrategy`, `PipelineParallelStrategy`, `MultiDimensionalStrategy`) build
partition specs for a `ParallelismConfig`; `substrax.spmd` adds `reduce_gradient_tree`,
`all_gather` and the `reduce_*` collectives.

### Checkpoint

`OrbaxCheckpointStore` writes a step-addressed store: the payload (an `nnx.Module`, a
`TrainState` or a pytree of arrays) with `PyTreeSave`, and a JSON sidecar with the step,
the loss and any extra metadata. Restoring onto a target places every array on the
target's device, so a checkpoint written on `cuda:0` restores in a CPU-only process.

```python
from flax import nnx
from substrax.checkpoint import OrbaxCheckpointStore

model = nnx.Linear(4, 4, rngs=nnx.Rngs(0))
with OrbaxCheckpointStore("checkpoints/demo", max_to_keep=3) as store:
    store.save(model, step=100, loss=0.25, additional_metadata={"epoch": 2})
    store.save(model, step=200, loss=0.20)

    assert store.list_steps() == [100, 200]
    assert store.best_step("loss") == 200
    fresh = nnx.Linear(4, 4, rngs=nnx.Rngs(1))
    restored, metadata = store.restore(fresh, step=100)  # fresh is updated in place
    assert metadata["epoch"] == 2

    payload, _ = store.restore(step=200)  # no target: the payload as it was stored
```

`restore` with a target raises `ValueError` when the checkpoint's arrays do not fit it,
which is how a store written for another architecture is refused rather than loaded.

### Callbacks

`EarlyStopping` is the plain tracker: `update(value)` records a metric and returns
whether it improved on the best by more than `min_delta`, and `should_stop` is true once
`patience` updates in a row have not. `EarlyStoppingCallback` is the same rule as a
training callback that a trainer drives
through `on_epoch_end`, with `check_finite`, `stopping_threshold` and
`divergence_threshold` from its `EarlyStoppingConfig`.

```python
from substrax.callbacks import EarlyStopping

stopper = EarlyStopping(patience=2, min_delta=0.01, mode="min")
improved = []
for step, loss in enumerate([1.0, 0.5, 0.49, 0.495, 0.5]):
    improved.append(stopper.update(loss))
    if stopper.should_stop:
        break
assert improved == [True, True, False, False]  # 0.49 is not 0.01 better than 0.5
assert step == 3  # two updates in a row without improvement
```

### Tracking

Every logger has the same surface (`log_scalar`, `log_scalars`, `log_hyperparams`,
`log_text`, `log_image`, `log_histogram`, `close`).
`create_logger` gives console output plus a file under `log_dir`; `WandbLogger` and
`MLFlowLogger` need the `wandb` and `mlflow` extras and import their SDK at construction,
so a missing extra fails at the call site, not at import time.

```python
from substrax.tracking import create_logger

logger = create_logger("demo", log_dir="logs")
logger.log_hyperparams({"lr": 1e-3, "batch_size": 32})
for step in range(3):
    logger.log_scalar("loss", 1.0 / (step + 1), step=step)
logger.close()
```

```python
from substrax.tracking import MLFlowLogger, WandbLogger

wandb_logger = WandbLogger("demo", project="my-project", config={"lr": 1e-3})
mlflow_logger = MLFlowLogger("demo", experiment_name="my-experiment")
```

## Development setup

```bash
git clone https://github.com/avitai/substrax.git
cd substrax
./setup.sh
source ./activate.sh
```

`setup.sh` creates the environment with `uv`, syncs the `dev` and `test` extras plus the
backend extra for this machine, and writes the managed environment file `.substrax.env`
that `activate.sh` loads. A user-owned `.env` is never modified.

| Flag | Effect |
| --- | --- |
| `--backend <auto\|cpu\|cuda12\|metal>` | Choose the backend policy; `auto` resolves to `cuda12` on Linux with a visible NVIDIA GPU, `metal` on Apple Silicon, otherwise `cpu` |
| `--python <version>` | Create the environment with a specific Python version |
| `--extra <name>` | Sync an additional extra (repeatable), e.g. `--extra docs` |
| `--recreate` | Remove the existing `.venv` before syncing |
| `--force-clean` | Remove `.venv`, `.substrax.env` and repo-local test artifacts |
| `--dry-run` | Print the resolved backend and the `uv` commands without changing files |

Run the checks the way CI does:

```bash
uv run --locked pytest
uv run --locked pre-commit run --all-files
uv run --locked mkdocs build --strict --clean
```

The test suite also runs as a pre-commit hook, so a commit takes a few seconds longer
than a lint pass.

## Documentation

<https://substrax.readthedocs.io> — one page per subpackage under *API Reference*, plus
the [migration page](https://substrax.readthedocs.io/en/latest/migration/) that maps the
names the sibling packages used to carry to their Substrax homes.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Security reports go to the address in
[SECURITY.md](SECURITY.md), not to a public issue.

## License

MIT — see [LICENSE](LICENSE).
