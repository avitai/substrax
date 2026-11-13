# Artifacts

`substrax.artifacts` decides where a run writes its figures, tables and other outputs, and
never defaults into the working tree.

```python
from substrax.artifacts import resolve_output_dir

location = resolve_output_dir("darcy_training")
figure_path = location.path / "prediction.png"
```

`resolve_output_dir` takes the first of these that applies and creates the directory:

| Order | Directory | `source` |
| --- | --- | --- |
| 1 | `explicit`, resolved against the working directory when it is relative | `"argument"` |
| 2 | `$AVITAI_OUTPUT_DIR/<name>`; the variable must hold an absolute path | `"environment"` |
| 3 | `<name>` inside a directory created once per process under the system temporary directory | `"run_default"` |

A caller that chooses the directory passes it as `explicit`. A test run, a CI job and the example
page generator set `AVITAI_OUTPUT_DIR` to a temporary directory, so running an example never
rewrites tracked files. An example's figures reach its docs page as the output its run captures
([Writing examples](../guides/writing-examples.md)), never as files it writes into `docs/`.

The per-run directory is created with `tempfile.mkdtemp`, so its name cannot be claimed by
another user of a shared temporary directory and only its owner can read it. Every default
output in one process shares it. `name` must be a relative path that stays inside the output
directory. Importing the package creates nothing.

::: substrax.artifacts
