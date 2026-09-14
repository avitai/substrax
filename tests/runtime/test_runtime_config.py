"""``JaxRuntime`` holds declarative JAX process settings and refuses invalid ones."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest

from substrax.runtime import JaxRuntime, XlaFlagConflictError


def test_every_field_defaults_to_leaving_the_process_alone() -> None:
    runtime = JaxRuntime()

    assert (
        runtime.platforms,
        runtime.cpu_devices,
        runtime.enable_x64,
        runtime.matmul_precision,
        runtime.compilation_cache_dir,
        runtime.xla_flags,
        runtime.preallocate,
        runtime.memory_fraction,
    ) == (None, None, None, None, None, (), None, None)


def test_a_fully_specified_runtime_keeps_its_values(tmp_path: Path) -> None:
    runtime = JaxRuntime(
        platforms=("cuda", "cpu"),
        cpu_devices=8,
        enable_x64=True,
        matmul_precision="high",
        compilation_cache_dir=tmp_path,
        xla_flags=("--xla_gpu_deterministic_ops=true",),
        preallocate=False,
        memory_fraction=0.5,
    )

    assert runtime.platforms == ("cuda", "cpu")
    assert runtime.cpu_devices == 8
    assert runtime.compilation_cache_dir == tmp_path
    assert runtime.memory_fraction == 0.5


def test_a_runtime_is_immutable() -> None:
    runtime = JaxRuntime(cpu_devices=2)

    with pytest.raises(dataclasses.FrozenInstanceError):
        runtime.cpu_devices = 4  # type: ignore[misc]


@pytest.mark.parametrize(
    "precision", ["default", "high", "highest", "bfloat16", "tensorfloat32", "float32"]
)
def test_every_matmul_precision_jax_accepts_is_accepted(precision: Any) -> None:
    assert JaxRuntime(matmul_precision=precision).matmul_precision == precision


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"cpu_devices": 0}, "cpu_devices"),
        ({"cpu_devices": -2}, "cpu_devices"),
        ({"memory_fraction": 0.0}, "memory_fraction"),
        ({"memory_fraction": 1.5}, "memory_fraction"),
        ({"memory_fraction": float("nan")}, "memory_fraction"),
        ({"compilation_cache_dir": Path("relative/cache")}, "absolute"),
        ({"platforms": ()}, "platforms"),
        ({"platforms": ("cpu", "")}, "platforms"),
        ({"platforms": ("cuda,cpu",)}, "platforms"),
        ({"matmul_precision": "medium"}, "matmul_precision"),
        ({"matmul_precision": "float16"}, "matmul_precision"),
        ({"xla_flags": ("xla_gpu_autotune_level=2",)}, "xla_gpu_autotune_level=2"),
    ],
)
def test_an_invalid_value_raises_and_names_the_field(fields: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        JaxRuntime(**fields)


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"cpu_devices": True}, "cpu_devices"),
        ({"memory_fraction": True}, "memory_fraction"),
        ({"enable_x64": 1}, "enable_x64"),
        ({"preallocate": "false"}, "preallocate"),
    ],
)
def test_a_value_of_the_wrong_type_raises_and_names_the_field(
    fields: dict[str, Any], message: str
) -> None:
    with pytest.raises(TypeError, match=message):
        JaxRuntime(**fields)


def test_conflicting_values_for_one_requested_flag_raise() -> None:
    with pytest.raises(XlaFlagConflictError, match="--xla_gpu_autotune_level"):
        JaxRuntime(xla_flags=("--xla_gpu_autotune_level=2", "--xla_gpu_autotune_level=4"))
