"""A conforming example for the notebooks tests, and a writer for its test repository."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from substrax.notebooks.config import ExamplesConfig
from substrax.notebooks.pairs import convert


CONFIG = ExamplesConfig(
    distribution="demo",
    source_url="https://github.com/avitai/demo/blob/main",
    budget_seconds=120.0,
)
"""The settings the notebooks tests run examples with; :func:`write_example` writes them too."""

PYPROJECT = (
    f'[project]\nname = "{CONFIG.distribution}"\nversion = "0"\n\n'
    f'[tool.substrax.examples]\nsource_url = "{CONFIG.source_url}"\n'
    f"budget_seconds = {CONFIG.budget_seconds}\n"
)

GOOD = '''# %% [markdown]
"""
# Sorting a Batch of Records

| Metadata | Value |
| --- | --- |
| **Tier** | Quick Reference |
| **Devices** | CPU-compatible |
| **Prerequisites** | Basic Python |

## Overview

Sorts records by key.

## Learning Goals

1. Sort a batch of records by key
2. Check the order of the result
"""

# %% [markdown]
"""
## Setup
"""

# %%
import jax

# %% [markdown]
"""
## Implementation

### Step 1: Make the records
"""


# %%
def make_records() -> list[int]:
    key = jax.random.key(0)
    return [int(value) for value in jax.random.permutation(key, 5)]


# %% [markdown]
"""
### Step 2: Sort them
"""


# %%
def sort_records(records: list[int]) -> list[int]:
    return sorted(records)


# %% [markdown]
"""
## Results

### Sorted order

## Next Steps

- [Another example](another.md)
"""


# %%
def main() -> None:
    print(sort_records(make_records()))


# %%
if __name__ == "__main__":
    main()
'''

GPU_REQUIRED = GOOD.replace("| **Devices** | CPU-compatible |", "| **Devices** | GPU-required |")
"""``GOOD`` declaring that it needs a GPU; its Setup does not yet check for one."""

_MAIN = "def main() -> None:\n    print(sort_records(make_records()))"


def returning_environment(*names: str) -> str:
    """``GOOD`` whose ``main()`` returns each of ``names`` as its kernel's environment holds it."""
    assert _MAIN in GOOD
    returned = ", ".join(f"{name!r}: os.environ.get({name!r})" for name in names)
    return GOOD.replace(
        _MAIN,
        "def main() -> dict[str, str | None]:\n    print(sort_records(make_records()))\n"
        f"    return {{{returned}}}",
    ).replace("import jax\n", "import jax\nimport os\n", 1)


def write_example(
    tmp_path: Path, text: str = GOOD, *, relative: str = "sorting.py"
) -> tuple[Path, Path]:
    """Write an example, its clean notebook and settings as a committed repository in ``tmp_path``.

    Args:
        tmp_path: The test's temporary directory.
        text: The example script.
        relative: The script's path below ``examples``.

    Returns:
        The repository root and the script.
    """
    repo = tmp_path / "repo"
    script = repo / "examples" / relative
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(text, encoding="utf-8")
    (repo / "pyproject.toml").write_text(PYPROJECT, encoding="utf-8")
    (repo / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")
    convert(script)
    _commit(repo)
    return repo, script


def _commit(repo: Path) -> None:
    """Make the repository a git repository holding one commit of everything in it."""
    for command in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "fixture"]):
        git(repo, *command)


def git(repo: Path, *command: str) -> None:
    """Run git in a test repository, as a fixed test identity.

    Git runs without the ``GIT_*`` variables a commit hook running pytest sets, which would point
    it at the index of the commit being made.

    Args:
        repo: The repository.
        *command: The git command and its arguments.
    """
    executable = shutil.which("git") or "git"
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    identity = ["-c", "user.name=Example Tests", "-c", "user.email=tests@example.invalid"]
    subprocess.run(  # noqa: S603  # nosec B603 - git with fixed arguments in a temporary repo
        [executable, *identity, *command],
        cwd=repo,
        env=environment,
        check=True,
        capture_output=True,
    )
