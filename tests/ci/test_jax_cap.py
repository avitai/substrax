"""The jax cap below 0.11.2 stands exactly as long as flax imports the name 0.11.2 renamed.

jax 0.11.2 renamed ``jax.experimental.hijax.HiPrimitive``; flax 0.12.9 reads it at module
load, so the pair fails on ``import flax.nnx``. This test fails the day the installed flax no
longer names ``HiPrimitive``: lift the cap then, and delete the test.
"""

from __future__ import annotations

import inspect
import tomllib
from pathlib import Path

from flax.nnx import variablelib


REPO_ROOT = Path(__file__).resolve().parents[2]


def _dependencies() -> list[str]:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]
    return [*project["dependencies"], *project["optional-dependencies"]["cuda12"]]


def test_jax_is_capped_below_0_11_2_while_flax_reads_the_renamed_name() -> None:
    flax_reads_it = "HiPrimitive" in inspect.getsource(variablelib)
    capped = [
        requirement
        for requirement in _dependencies()
        if requirement.split("[")[0].split(">")[0].split("=")[0] in {"jax", "jaxlib"}
    ]
    assert capped, "no jax requirement"
    for requirement in capped:
        assert ("<0.11.2" in requirement) == flax_reads_it, (
            f"{requirement!r}: the cap stands exactly while flax names HiPrimitive "
            f"(flax reads it: {flax_reads_it})"
        )
