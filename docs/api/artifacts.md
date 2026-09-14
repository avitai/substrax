# Artifacts

`substrax.artifacts` decides where a run writes its figures, tables and other outputs, and
never defaults into the working tree.

```python
import argparse
from pathlib import Path

from substrax.artifacts import resolve_output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    location = resolve_output_dir("fno_darcy", explicit=args.output_dir)
    figure_path = location.path / "prediction.png"
```

`resolve_output_dir` takes the first of these that applies and creates the directory:

| Order | Directory | `source` |
| --- | --- | --- |
| 1 | `explicit`, resolved against the working directory when it is relative | `"argument"` |
| 2 | `$AVITAI_OUTPUT_DIR/<name>`; the variable must hold an absolute path | `"environment"` |
| 3 | `<name>` inside a directory created once per process under the system temporary directory | `"run_default"` |

Writing figures into the documentation is an explicit choice:
`python examples/neural-operators/fno_darcy.py --output-dir docs/assets/examples/fno_darcy`.
A test run or a CI job sets `AVITAI_OUTPUT_DIR` instead, so running an example never rewrites
tracked files.

The per-run directory is created with `tempfile.mkdtemp`, so its name cannot be claimed by
another user of a shared temporary directory and only its owner can read it. Every default
output in one process shares it. `name` must be a relative path that stays inside the output
directory. Importing the package creates nothing.

::: substrax.artifacts
