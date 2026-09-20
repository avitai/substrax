# Compute

`substrax.compute` runs a project's jobs on a compute backend: this machine, Modal, or any cloud
SkyPilot reaches. A job is declared once, in the project's `pyproject.toml`, and runs the same
way on every backend, because every backend runs the same program on the remote machine: the
worker.

## Declaring jobs

```toml
[tool.substrax.compute]
backend = "modal"                        # used when the command line names none

[tool.substrax.compute.backends.modal]   # passed to the backend as its settings
outputs_volume = "demo-outputs"

[tool.substrax.compute.jobs.examples]
examples = ["examples/metrics"]          # one task per script
commands = { gpu-tests = ["python", "-m", "pytest", "tests/gpu"] }
extras = ["cuda12"]
accelerator = { kind = "L4", count = 1 }
timeout_seconds = 1800
task_timeout_seconds = 300
runtime = { platforms = ["cuda"], xla_flags = ["--xla_gpu_deterministic_ops=true"] }
env = { TF_CPP_MIN_LOG_LEVEL = "1" }
mounts = [{ name = "datasets", path = "/data", access = "read-only" }]
```

A misspelled key is refused, not ignored. `examples` lists each directory's scripts as
`substrax.examples.discover_examples` does, skipping every name that starts with `_`; a task's
leading `"python"` is the project environment's interpreter.

## Running them

```bash
uv run substrax-compute run examples                  # submit, follow, wait, fetch
uv run substrax-compute run examples --accelerator H100:2 --detach
uv run substrax-compute status                        # every stored run
uv run substrax-compute logs <run-id> --follow
uv run substrax-compute fetch <run-id> --into ~/runs
uv run substrax-compute cancel <run-id>
```

The command runs from the project's own environment, whose lock must include substrax: the
worker is the project's locked substrax. A run's handle is kept under the state directory
(`$SUBSTRAX_STATE_DIR`, else `$XDG_STATE_HOME/substrax`), so a detached run is found again from
any shell. Fetched outputs go to `--into` or under the state directory, never into the
project.

## What a run produces

```text
<run-id>/
  manifest.json            # every task: argv, status, exit code, seconds, files written
  <task>/stdout.log
  <task>/stderr.log
  <task>/<name>/...        # what the task saved through resolve_output_dir(name)
```

Each task runs from the project root with `AVITAI_OUTPUT_DIR` set to its own directory and the
job's `JaxRuntime` applied, for at most its own budget or what is left of the job's. A failed
task does not stop the ones after it. The worker rewrites the manifest after every task, and
marks it `finished` after the last.

## Backends

| Backend | Runs on | Needs |
| --- | --- | --- |
| `local` | this machine, in the project's current environment | nothing |
| `modal` | Modal, one detached function per run, outputs on a Volume | `substrax[modal]`, a Modal account, and a workspace image builder of 2025.06 or later |
| `skypilot` | a cluster SkyPilot provisions for the run on any cloud it supports, torn down after it; outputs on a `gs://` or `s3://` bucket | SkyPilot installed as its documentation says (`uv tool install --with pip "skypilot[gcp]"`), and `gcloud` or `aws` for the bucket |

The Modal image is built in layers keyed on what they depend on: the locked dependencies
(`Image.uv_sync`), then the project's source, then the project installed without
dependencies, so an edit to the source rebuilds only the last two. `Image.uv_sync` needs image
builder 2025.06 or later, set for the workspace in Modal's dashboard under Settings → Image
config; an older builder fails the build with `Image builder version <= 2024.10 requires modal
to be specified in your pyproject.toml file`. The image runs the Python this interpreter runs,
which the serialized worker requires, and which `uv sync` is told to use so a project's own
`python-preference` cannot choose another. SkyPilot is never imported:
the backend writes a task file and runs `sky launch`, `sky queue -o json`, `sky logs`,
`sky cancel` and `sky down`.

## Adding a provider

A backend is any object that satisfies `ComputeBackend` and is built from its settings table.
Register the factory in the `substrax.compute.backends` entry-point group:

```toml
[project.entry-points."substrax.compute.backends"]
myprovider = "mypackage.backend:MyBackend"
```

and inherit the contract every backend must pass:

```python
from pathlib import Path

from substrax.compute.backend import ComputeBackend
from substrax.testing.compute import BackendContract

from mypackage.backend import MyBackend


class TestMyBackend(BackendContract):
    def make_backend(self, state_dir: Path) -> ComputeBackend:
        return MyBackend({"state_dir": str(state_dir)})
```

::: substrax.compute

::: substrax.compute.backend

::: substrax.compute.config

::: substrax.compute.worker

::: substrax.compute.registry

::: substrax.compute.local

::: substrax.compute.modal_backend

::: substrax.compute.skypilot_backend
