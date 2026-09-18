"""The managed environment file a repository's ``setup.sh`` writes and ``activate.sh`` sources."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from substrax.runtime import managed_env
from substrax.testing import run_python


def _render(tmp_path: Path, backend: str, extra: dict[str, str] | None = None) -> str:
    values = managed_env.managed_environment(
        prefix="DEMO", backend=backend, project_root=tmp_path, extra=extra or {}
    )
    return managed_env.render(prefix="DEMO", values=values)


def _sourced(env_file: Path, stale: dict[str, str]) -> dict[str, str]:
    """The environment a shell holds after sourcing ``env_file`` over ``stale``."""
    result = subprocess.run(  # noqa: S603 - fixed interpreter and a temporary file
        [
            "/bin/bash",
            "-eu",
            "-c",
            'source "$1"; "$2" -c "import json, os; print(json.dumps(dict(os.environ)))"',
            "managed-env-test",
            str(env_file),
            sys.executable,
        ],
        env={**os.environ, **stale},
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return json.loads(result.stdout)


STALE = {
    "JAX_PLATFORMS": "stale",
    "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.9",
    "XLA_CLIENT_MEM_FRACTION": "0.8",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
    "JAX_ENABLE_X64": "true",
}


def test_cpu_pins_the_cpu_platform_and_clears_accelerator_memory_settings(tmp_path: Path) -> None:
    env_file = tmp_path / ".demo.env"
    env_file.write_text(_render(tmp_path, "cpu"))

    env = _sourced(env_file, STALE)

    assert env["DEMO_BACKEND"] == "cpu"
    assert env["DEMO_ENV_ROOT"] == str(tmp_path)
    assert env["JAX_PLATFORMS"] == "cpu"
    assert env["JAX_ENABLE_X64"] == "false"
    for name in (
        "XLA_CLIENT_MEM_FRACTION",
        "XLA_PYTHON_CLIENT_MEM_FRACTION",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
    ):
        assert name not in env


@pytest.mark.parametrize("backend", ["cuda12", "metal"])
def test_an_accelerator_leaves_the_platform_to_jax_and_uses_the_current_memory_name(
    tmp_path: Path, backend: str
) -> None:
    env_file = tmp_path / ".demo.env"
    env_file.write_text(_render(tmp_path, backend))

    env = _sourced(env_file, STALE)

    assert env["DEMO_BACKEND"] == backend
    assert "JAX_PLATFORMS" not in env
    assert env["XLA_CLIENT_MEM_FRACTION"] == "0.75"
    assert env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    # jaxlib raises when both memory-fraction names are set, so the deprecated one is always cleared.
    assert "XLA_PYTHON_CLIENT_MEM_FRACTION" not in env


def test_the_manifest_names_every_variable_the_file_manages(tmp_path: Path) -> None:
    text = _render(tmp_path, "cuda12", extra={"TF_CPP_MIN_LOG_LEVEL": "1"})

    manifest_line = next(
        line for line in text.splitlines() if line.startswith("export DEMO_MANAGED_ENV_VARS=")
    )
    managed = set(manifest_line.split("=", 1)[1].strip("'\"").split())
    written = {
        line.split()[1].split("=", 1)[0]
        for line in text.splitlines()
        if line.startswith(("export ", "unset ")) and "DEMO_MANAGED_ENV_VARS" not in line
    }
    assert managed == written
    assert "TF_CPP_MIN_LOG_LEVEL" in managed
    assert "XLA_PYTHON_CLIENT_MEM_FRACTION" in managed


@pytest.mark.parametrize("backend", ["cpu", "cuda12", "metal"])
def test_the_file_never_injects_library_paths_or_compiler_flags(
    tmp_path: Path, backend: str
) -> None:
    text = _render(tmp_path, backend)

    for name in ("LD_LIBRARY_PATH", "CUDA_HOME", "XLA_FLAGS"):
        assert name not in text


def test_repository_variables_are_written_verbatim_and_shell_safe(tmp_path: Path) -> None:
    root = tmp_path / "a project's root"
    root.mkdir()
    env_file = root / ".demo.env"
    values = managed_env.managed_environment(
        prefix="DEMO",
        backend="cpu",
        project_root=root,
        extra={"DEMO_NOTE": "value with 'quotes' and $dollar", "TF_CPP_MIN_LOG_LEVEL": "1"},
    )
    env_file.write_text(managed_env.render(prefix="DEMO", values=values))

    env = _sourced(env_file, {})

    assert env["DEMO_ENV_ROOT"] == str(root)
    assert env["DEMO_NOTE"] == "value with 'quotes' and $dollar"
    assert env["TF_CPP_MIN_LOG_LEVEL"] == "1"


def test_an_extra_variable_cannot_override_a_managed_jax_setting(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="JAX_PLATFORMS"):
        managed_env.managed_environment(
            prefix="DEMO", backend="cpu", project_root=tmp_path, extra={"JAX_PLATFORMS": "cuda"}
        )


@pytest.mark.parametrize("prefix", ["", "demo", "1DEMO", "DE-MO"])
def test_a_prefix_must_be_an_upper_case_shell_name(tmp_path: Path, prefix: str) -> None:
    with pytest.raises(ValueError, match="prefix"):
        managed_env.managed_environment(
            prefix=prefix, backend="cpu", project_root=tmp_path, extra={}
        )


def test_an_unknown_backend_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="backend"):
        managed_env.managed_environment(
            prefix="DEMO", backend="tpu", project_root=tmp_path, extra={}
        )


@pytest.mark.parametrize(
    ("system", "machine", "gpu", "expected"),
    [
        ("Linux", "x86_64", True, "cuda12"),
        ("Linux", "x86_64", False, "cpu"),
        ("Darwin", "arm64", False, "metal"),
        ("Darwin", "x86_64", False, "cpu"),
        ("Windows", "AMD64", True, "cpu"),
    ],
)
def test_auto_resolves_by_platform(
    monkeypatch: pytest.MonkeyPatch, system: str, machine: str, gpu: bool, expected: str
) -> None:
    monkeypatch.setattr(managed_env.platform, "system", lambda: system)
    monkeypatch.setattr(managed_env.platform, "machine", lambda: machine)
    monkeypatch.setattr(managed_env, "nvidia_gpu_visible", lambda: gpu)

    assert managed_env.resolve_backend("auto") == expected
    assert managed_env.resolve_backend("cpu") == "cpu"


def test_no_nvidia_smi_means_no_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(managed_env.shutil, "which", lambda _name: None)

    assert managed_env.nvidia_gpu_visible() is False


@pytest.mark.parametrize(
    ("returncode", "stdout", "expected"),
    [(0, "GPU 0: NVIDIA RTX\n", True), (0, "", False), (9, "GPU 0: NVIDIA RTX\n", False)],
)
def test_a_gpu_is_visible_only_when_nvidia_smi_lists_one(
    monkeypatch: pytest.MonkeyPatch, returncode: int, stdout: str, expected: bool
) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, returncode, stdout=stdout, stderr="")

    monkeypatch.setattr(managed_env.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(managed_env.subprocess, "run", fake_run)

    assert managed_env.nvidia_gpu_visible() is expected
    assert calls == [["/usr/bin/nvidia-smi", "-L"]]


@pytest.mark.parametrize(
    "error", [OSError("not executable"), subprocess.TimeoutExpired("nvidia-smi", 5)]
)
def test_an_nvidia_smi_that_cannot_run_means_no_gpu(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise error

    monkeypatch.setattr(managed_env.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(managed_env.subprocess, "run", fake_run)

    assert managed_env.nvidia_gpu_visible() is False


def test_show_reports_an_unknown_backend_for_a_file_without_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    env_file = tmp_path / ".demo.env"
    env_file.write_text("export OTHER=1\n")

    managed_env.main(
        ["show", "--prefix", "DEMO", "--env-file", str(env_file), "--project-root", str(tmp_path)]
    )

    assert "Configured backend: unknown" in capsys.readouterr().out


def test_the_write_command_writes_the_file_and_reports_the_backend(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / ".demo.env"

    code = managed_env.main(
        [
            "write",
            "--prefix",
            "DEMO",
            "--backend",
            "cpu",
            "--output",
            str(output),
            "--project-root",
            str(tmp_path),
            "--set",
            "TF_CPP_MIN_LOG_LEVEL=1",
        ]
    )

    assert code == 0
    assert "export DEMO_BACKEND=cpu" in output.read_text()
    assert "export TF_CPP_MIN_LOG_LEVEL=1" in output.read_text()
    assert f"Wrote {output} for backend 'cpu'" in capsys.readouterr().out


def test_a_set_without_a_value_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        managed_env.main(
            [
                "write",
                "--prefix",
                "DEMO",
                "--backend",
                "cpu",
                "--output",
                str(tmp_path / "e"),
                "--set",
                "NO_EQUALS_SIGN",
            ]
        )


def test_the_show_command_reports_each_layer(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    env_file = tmp_path / ".demo.env"
    env_file.write_text(_render(tmp_path, "cuda12"))
    (tmp_path / ".env.local").write_text("export EXAMPLE=1\n")

    code = managed_env.main(
        [
            "show",
            "--prefix",
            "DEMO",
            "--env-file",
            str(env_file),
            "--project-root",
            str(tmp_path),
            "--user-env",
            ".env",
            "--user-env",
            ".env.local",
        ]
    )

    assert code == 0
    out = capsys.readouterr().out
    assert ".demo.env present: True" in out
    assert ".env present: False" in out
    assert ".env.local present: True" in out
    assert "Configured backend: cuda12" in out


def test_show_without_a_managed_file_reports_no_backend(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = managed_env.main(
        [
            "show",
            "--prefix",
            "DEMO",
            "--env-file",
            str(tmp_path / ".demo.env"),
            "--project-root",
            str(tmp_path),
        ]
    )

    assert code == 0
    assert "Configured backend: none" in capsys.readouterr().out


def test_running_the_module_imports_no_jax() -> None:
    result = run_python(
        "import sys, substrax.runtime.managed_env; print('jax' in sys.modules)",
        timeout=60,
    )

    assert result.stdout.strip() == "False"
