# Writing examples

Every repository built on substrax writes its examples the same way, and this guide is the one
place that says how. An example shows a feature, API or capability of its repository and runs as
written; an example that shows none of its repository is removed rather than kept under another
label.

A rule the shared checks enforce ends with `Checked:` and the id the checks report it by. Every
other rule is a judgement a reviewer makes.

## The process

1. Plan: who reads the example, its tier, its learning goals, its prerequisites, and the
   repository's existing APIs and tests it builds on.
2. Copy the [template](example_template.py) to `examples/<topic>/<name>.py`.
3. Write the script. It is the only file you author.
4. Write its notebook: `uv run substrax-notebooks convert examples/<topic>/<name>.py`.
5. Run it and write its page: `uv run substrax-notebooks page examples/<topic>/<name>.py`.
6. Review it against the [checklist](#checklist).
7. Run the gate: `uv run substrax-notebooks test` and `pre-commit run --all-files`.

Every command runs from the repository root after `source ./activate.sh`.

## Files

- An example is a `.py` file under `examples/`. Every `.py` there whose path has no part starting
  with `_` is one, so a helper an example imports has a private name. Its run, like
  `python examples/<topic>/<name>.py`, puts only the example's own directory on `sys.path`, so
  the helper sits beside the example (`examples/<topic>/_plotting.py`), or in the repository's
  package.
- Every part of an example's path below `examples/`, its directories and its file name, is
  `snake_case` with no number prefix. Checked: `names`. The docs navigation sets the reading order.
- The `.ipynb` beside the script is written by `substrax-notebooks convert` and committed exactly as
  it writes it: no outputs, no execution counts, no cell ids, no other metadata. Never edit it, run
  it in place, or let a tool rewrite it. Checked: `pair`
- The page `docs/examples/<topic>/<name>.md` is written by `substrax-notebooks page` from a run of
  the notebook and committed with its figures. It shows the example's cells, every output of the
  run, and a header with the run's duration, device and versions (Python, jax, substrax and the
  distribution `[project]` names), links to the script and notebook, and the
  command that runs it. `docs/examples` holds generated pages only: rerun `page` whenever the script
  changes, and never edit a page or its figures. The page records the digest of its text, of its
  provenance record and of each figure as written, so a change by hand, or by a formatter, is
  found. Checked: `page`
- A page whose example was renamed, moved or removed goes with it; so does a hand-written
  `uv run python examples/...` command naming a file that no longer exists. Checked: `corpus`

## The script

The script is jupytext's percent format: `# %%` opens a code cell, and `# %% [markdown]` opens a
markdown cell written as a triple-quoted string, never as comments. A marker carries a title at
most, never an option such as a `tags=` or `active=` value or `[raw]`: the notebook run obeys
one and `python <script>` does not. A docstring inside a markdown cell would close that string
early, so the whole script must parse as Python. The file is UTF-8 without a byte order mark.
Checked: `parses`

The first markdown cell opens with the title, `# <subject>`, the only level-1 heading in the
example. The title names the subject, not the tier. Checked: `title`

The metadata table directly follows the title and opens with its `| Metadata | Value |` header:

```markdown
| Metadata | Value |
| --- | --- |
| **Tier** | Tutorial |
| **Devices** | CPU-compatible |
| **Prerequisites** | Basic JAX arrays; the MNIST dataset (11 MB) |
```

The table has exactly these three fields, in this order. Checked: `metadata`

- Tier is one of `Quick Reference`, `Tutorial` or `Advanced Guide`:
  - A Quick Reference shows one capability, briefly, to a reader who already knows the basics.
  - A Tutorial walks a reader new to a capability through it, step by step.
  - An Advanced Guide combines capabilities on a realistic workload: scale, performance or a full
    pipeline.

  Every tier uses the same sections. Tiers differ in depth, not in headings.
- Devices is one of `CPU-compatible`, `GPU-optional` or `GPU-required`. An example is
  CPU-compatible unless its work is inherently GPU-bound; then it is GPU-required.
- Prerequisites lists what the reader needs: knowledge, each dataset with its size, and, for a
  GPU-required example, the GPU.

No other field is written. The page records the run's real duration and device, and nothing else
numeric is estimated: no runtime, memory, line count or goal count.

## Sections

The example's second-level sections come in this order; Core Concepts and Troubleshooting may be
left out. Checked: `skeleton`

1. `## Overview`: one short paragraph on what the example demonstrates and why it matters; the
   problem and its benchmark, when there is one.
2. `## Learning Goals`: a numbered list. Checked: `goals`. Each goal starts with an action verb
   such as Create, Configure, Train, Evaluate, Compare, Debug or Optimize.
3. `## Setup`: the imports, in one code cell; no other cell imports at its top level (an import
   inside a function or a block is not read). Checked: `setup`. A
   GPU-required example then calls `substrax.devices.require_gpu()` with no arguments right after
   the imports, as a statement of its own (`require_gpu()`) or the value of an assignment
   (`gpus = require_gpu()`), with nothing but a docstring between and the name bound by the
   imports before it, which raises unless jax's default backend is a GPU, so the example stops
   before any work rather than run it on the CPU. Checked: `gpu-check`
4. `## Core Concepts` (optional): the idea a reader needs before the code. A diagram or a table
   where it helps.
5. `## Implementation`: one `### Step N: <action>` per step, numbered from 1. A parameter sweep the
   example runs is a step. Checked: `steps`
6. `## Results`: what the run shows, under concrete `###` titles. Checked: `results`. A table
   where it helps.
7. `## Troubleshooting` (optional): one `### <symptom or error>` per entry. Checked:
   `troubleshooting`. Each entry gives the symptom, its cause and the solution, and how to prevent
   it where that helps.
8. `## Next Steps`: links to related examples and to the API reference, one line each. Its only
   subsection is the optional `### Experiments to Try`. Checked: `next-steps`. Experiments to Try
   only suggests.

## Code

- The script defines `main()` at top level, not `async`, and ends with
  `if __name__ == "__main__": main()`, with `main()` alone in that block and called with no
  arguments. Checked: `main`.
  `main()` prints the results a reader checks and may return them as a value JSON holds (a
  number, a string, a list or a dict of them, a zero-dimensional array as its item), which the
  run records; one JSON cannot hold, such as a set, fails the run.
- The example reads no command-line arguments, from `sys.argv` or a parser such as `argparse`:
  its run passes none, so a value a reader may change is a named constant in the script. Checked:
  `no-arguments`. The run gives the example what a script run does: its own
  `sys.argv[0]` and `__file__`, and a `main()` that exits with status 0 ends the run cleanly.
- The example reads the environment and never writes it: no `os.environ[...] =`, `setdefault`,
  `update`, `pop`, `del`, `putenv` or `unsetenv`. Checked: `no-env-writes`
- The example never chooses or forces a backend or a device: no `JAX_PLATFORMS`,
  `CUDA_VISIBLE_DEVICES`, `jax_platform_name` or `jax_default_device`, in code or in prose telling
  the reader to set them, and no device chosen in code (`jax.devices("cpu")`,
  `jax.local_devices(None, "cpu")`, a `backend=` argument, `jax.default_device`; a `None` platform
  asks for jax's default and is no choice). A name means what the module's imports bind it to, those
  in a module-level `try`, `if` or `with` included, the last import of a name deciding; a function's
  or class's own parameters, assignments and imports are not followed, nor an assignment such as
  `listing = jax.devices`, so a parameter that shadows an imported jax function is read as that
  function. JAX picks the device; only a lane may set the platform. Checked:
  `no-backend-forcing`
- Artifacts go through `substrax.artifacts.resolve_output_dir`, never under `docs/`, and a figure
  saved with `savefig` is saved there. Checked: `artifacts`. A figure is shown with `plt.show()`, so
  the run captures it for the page. A run that changes or removes anything git tracks, adds an
  untracked path git does not ignore, or changes git's own state (a commit, a staged file, a hook)
  fails; a path the repository ignores, such as `__pycache__`, is a byproduct of running and is
  not checked. Checked: `runs-as-its-page-shows`
- No `from x import *`. Checked: `no-star-imports`
- Keys come from `jax.random.key`, never `jax.random.PRNGKey`, whether called through
  `jax.random` or imported by name. Checked: `typed-keys`. Each use derives its own key with
  `jax.random.split` or `jax.random.fold_in`; a key is never reused.
- The code passes the repository's ruff and pyright configuration like any other code: imports as
  ruff sorts them, builtin generics (`dict[str, float]`), Google docstrings with Args and Returns.
  The repository exempts examples from only the four rules the format itself breaks: the first
  markdown cell is the module docstring and opens with its title, imports follow markdown cells,
  and `main()` prints.

  ```toml
  [tool.ruff.lint.per-file-ignores]
  "examples/**/*.py" = ["D212", "D415", "E402", "T201"]
  ```

- Data is synthetic, generated in code, unless the example's goal needs a real dataset; then the
  dataset and its size are in Prerequisites, and the message the example fails with when the
  dataset is missing is listed in the repository's settings, so the run is skipped with it rather
  than failed.

## Outputs

- What the page shows is what the run printed and drew: no comment or line of prose opens by
  stating an output (`# Expected output:`, `## Expected Output`, `Expected:`). Checked:
  `no-expected-output`. There are no hand-written output blocks; the captured run is the output.
  A comment naming what code returns, a shape or a figure (`# Output: logits of shape (batch,
  classes)`) states no run's output and is allowed.
- Print what lets a reader check a step: shapes, dtypes, key values. Show the figures the learning
  goals need, and no more.

## Checks

A commit runs the `example-conformance` hook on each changed example: every rule above except
`page`, `corpus` and `runs-as-its-page-shows`, without running anything. Each repository enables it in `.pre-commit-config.yaml`:

```yaml
- repo: https://github.com/avitai/substrax
  rev: <the release the repository locks>
  hooks:
    - id: example-conformance
```

Generated pages are left out of every hook that rewrites files or scans their content, as a page
records the digests of its text and figures and any change reads as an edit. Three common hooks
change a page: markdownlint-fix reformats the indented blocks nbconvert writes for outputs,
trailing-whitespace strips the trailing spaces an output keeps, and detect-secrets flags the
digests in the page's provenance record. Each such hook excludes the directory, joined into any
`exclude` pattern it already has (`^(docs/examples/|uv\.lock$)`):

```yaml
- id: trailing-whitespace
  exclude: ^docs/examples/
- id: markdownlint-fix
  exclude: ^docs/examples/
- id: detect-secrets
  exclude: ^docs/examples/
```

The shared example tests run in pytest, from the repository's settings alone; no conftest names
them. A repository's development dependencies hold `substrax[notebooks,testing]`:

```toml
# pyproject.toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--ignore=tests/examples"   # a plain pytest leaves the tests of results out

[tool.substrax.examples]
source_url = "https://github.com/avitai/<repository>/blob/main"
budget_seconds = 600
unavailable = ["<the message an example fails with when its dataset is missing>"]
# page_size_warn_bytes = 102400, page_size_bytes = 204800 and page_size_exempt = [] by default
```

```yaml
# mkdocs.yml
validation:
  nav:
    omitted_files: warn
```

- The examples run in their own invocation, `uv run substrax-notebooks test`: pytest on
  `examples` and, where it exists, `tests/examples`, with the shared example tests the one plugin
  loaded: the repository's pytest configuration and conftests, every other installed plugin and
  the caller's `PYTEST_ADDOPTS` and `PYTEST_PLUGINS` are left out (a configuration file of
  substrax's that sets nothing, with the repository as rootdir; `--noconftest`;
  `--disable-plugin-autoload`). So no unit-test timeout or coverage floor reaches a run, each
  kernel gets the environment `uv run python examples/...` gets, and no other test has started
  jax in the process, which on a GPU would leave the example's kernel short of its memory. PATHs
  given to it, from the working directory, replace the two directories, and one outside the
  repository is refused. A plain `uv run pytest` runs `tests` without `tests/examples`.
- `examples::corpus` checks the examples as a whole: the rules of [Files](#files) that span
  several examples.
- `examples/<path>.py::follows-the-guide` checks every rule of this guide.
- `examples/<path>.py::runs-as-its-page-shows`, marked `example_run`, runs the example as its clean
  notebook within `budget_seconds` and compares the run with the committed page.
- A pull request runs the examples it touches,
  `uv run substrax-notebooks test --changed-since origin/main`, and every example
  when `src/`, `pyproject.toml` or `uv.lock` changed, or a file under `examples/` that is no
  example's own (a private helper, data); a scheduled job runs every example.
- Every lane runs every example. Off a GPU, a GPU-required example's `require_gpu()` raises
  `GPUUnavailableError`, and its run and the tests of its results are skipped with that message;
  any other example raising it is held to the `unavailable` messages like any failure. A GPU lane
  checks for its GPU first, and sets no platform:

  ```bash
  uv run substrax-devices --require-gpu
  uv run substrax-notebooks test
  ```

  `--require-gpu` exits 1, naming the backend, unless jax's default backend is a GPU, so a GPU lane
  that has lost its GPU fails instead of skipping. With no `JAX_PLATFORMS`, jax puts the GPU first
  and keeps the CPU, where host callbacks (`jax.debug.print`, `jax.pure_callback`, `io_callback`)
  place their operands, and the examples run as a reader's do. A platform list cannot do the
  check's job: jax skips a listed `cuda` on a machine with no NVIDIA device node. The pytest
  process probes no device.
- `budget_seconds` comes from the examples' measured runs, which each page records; write the
  measurement beside the number.
- A page is held to a size as Documenter.jl holds its pages: `substrax-notebooks page` warns above
  `page_size_warn_bytes` (100 KiB) and refuses a page above `page_size_bytes` (200 KiB), its
  figures not counted, and the `page` check reports a committed page over the limit. Print what
  lets a reader check a step, not bulk data; list an example in `page_size_exempt` only when its
  page must be large, which is preferred over raising the limit for every page. An entry is the
  example's path as its page records it, `examples/<path>.py`; one in another form, or naming no
  example, is refused when the settings are read.
- A repository's own test of an example's results lives in `tests/examples/` and reads the same
  run through the `example_run` fixture, the one fixture it has beyond pytest's own (no conftest or
  other plugin is loaded), and carries no device marker: its example's Devices line decides whether
  it runs. It is named as pytest's defaults name a test (a `test_*.py` or `*_test.py` file,
  `test`-prefixed functions, `Test`-prefixed classes), since the repository's `python_files`,
  `python_classes` and `python_functions` do not reach the session. The naming sorts the files
  pytest finds in a directory it walks, so one named otherwise there is not collected; a file named
  as a PATH on the command line is collected whatever its name. In a session that did not collect
  the example's run, the fixture fails the test, naming the invocation:

```python
# tests/examples/test_normalising.py
def test_the_whole_batch_is_normalised(example_run) -> None:
    returned = example_run("statistics/normalising.py").returned
    assert returned["count"] == 128
```

## Checklist

A reviewer reads the page and the script for what no check can see:

- The example shows a feature, API or capability of this repository.
- The Overview says what is shown and why it matters, in one short paragraph.
- Each learning goal starts with an action verb, is met by a step and is visible in the results.
- No environment variable or flag switches the example to a smaller or quicker run: what a
  reader runs is the run its page shows.
- Each use of a key derives its own with `jax.random.split` or `jax.random.fold_in`; no key is
  used twice.
- Each step does one thing, and the prose before it says what and why.
- The printed values and figures let a reader check every step; `main()` prints the results the
  page shows, and each figure is shown with `plt.show()`.
- Tables, diagrams and callouts appear where they help and nowhere else.
- Next Steps links to examples and API pages that exist.
