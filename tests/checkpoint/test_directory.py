"""The checkpoint directory resolver chooses a directory and never creates it."""

from __future__ import annotations

from pathlib import Path

import pytest

from substrax.checkpoint import resolve_checkpoint_dir


def test_an_explicit_directory_wins(tmp_path: Path) -> None:
    explicit = tmp_path / "explicit"
    resolved = resolve_checkpoint_dir(explicit, tmp_path / "run")
    assert resolved == explicit.resolve()
    assert resolved.is_absolute()


def test_a_relative_explicit_directory_resolves_against_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert resolve_checkpoint_dir(Path("ckpt"), None) == (tmp_path / "ckpt").resolve()


def test_a_run_directory_holds_a_checkpoints_subdirectory(tmp_path: Path) -> None:
    assert (
        resolve_checkpoint_dir(None, tmp_path / "run")
        == (tmp_path / "run" / "checkpoints").resolve()
    )


def test_neither_is_an_error() -> None:
    with pytest.raises(ValueError, match="checkpoint_dir or run_dir"):
        resolve_checkpoint_dir(None, None)


def test_the_resolver_writes_nothing(tmp_path: Path) -> None:
    resolved = resolve_checkpoint_dir(None, tmp_path / "run")
    assert not resolved.exists()
    assert not (tmp_path / "run").exists()
