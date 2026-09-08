# Substrax

**Shared training and hardware infrastructure for the Avitai JAX stack.**

> **Research preview.** Substrax is under rapid iteration and the API will change while
> the sibling packages migrate onto it. Pin a version.

Substrax is the bottom of the Avitai dependency chain:

```text
substrax → calibrax → datarax → artifex → opifex
```

It holds the code those packages used to carry separately, so that each concern has one
home and one test suite:

| Subpackage | What it owns |
| --- | --- |
| `substrax.devices` | Device information and device placement, including the batch-size recommendation table |
| `substrax.mesh` | Device meshes on `jax.make_mesh`, mesh rules and partition-spec helpers, sharding strategies on `flax.nnx.spmd` |
| `substrax.spmd` | Data-parallel placement, `spmd_train_step`, gradient reduction and collectives on the flat-state path |
| `substrax.checkpoint` | One `CheckpointStore` protocol and one Orbax implementation over `CheckpointManager` |
| `substrax.callbacks` | The training-callback protocol, `BestMetricTracker`, `EarlyStopping` |
| `substrax.tracking` | Step-wise experiment tracking with console, CSV, Weights & Biases and MLflow backends |

Not in Substrax: optimizers and schedules (optax), loss scaling and gradient accumulation
(`flax.training.dynamic_scale.DynamicScale`, `optax.MultiSteps`), profiling and hardware
spec tables (calibrax), data pipelines (datarax), models and trainers (artifex, opifex).

## Installation

```bash
uv add substrax   # or: pip install substrax
```

Substrax requires Python 3.12 or later, `jax>=0.11.1`, `flax>=0.12.9` and
`orbax-checkpoint>=0.11.33`.

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

## Documentation

<https://substrax.readthedocs.io>

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Security reports go to the address in
[SECURITY.md](SECURITY.md), not to a public issue.

## License

MIT — see [LICENSE](LICENSE).
