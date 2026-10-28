"""Copied artifacts must match a complete remote manifest before teardown."""

from pathlib import Path

import pytest

from substrax.compute.sky_transfer import snapshot, verify


def test_snapshot_and_verified_copy(tmp_path: Path) -> None:
    """Files and links are distinct identities; links are never followed."""
    root = tmp_path / "output"
    root.mkdir()
    (root / "value").write_bytes(b"result")
    (root / "link").symlink_to("/unavailable/input")
    manifest = snapshot(root)
    verify(root, manifest)
    assert manifest["link"] == {"kind": "symlink", "target": "/unavailable/input"}
    (root / "value").write_bytes(b"changed")
    with pytest.raises(ValueError, match="does not match"):
        verify(root, manifest)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "x/../escape"])
def test_unsafe_remote_manifest_is_rejected(tmp_path: Path, name: str) -> None:
    """Remote metadata cannot select arbitrary local verification targets."""
    with pytest.raises(ValueError, match="unsafe path"):
        verify(tmp_path, {name: {"kind": "symlink", "target": "anything"}})


def test_extra_or_missing_file_is_rejected(tmp_path: Path) -> None:
    """A subset comparison is insufficient for a complete recovery receipt."""
    manifest = snapshot(tmp_path)
    (tmp_path / "unexpected").write_text("extra")
    with pytest.raises(ValueError, match="does not match"):
        verify(tmp_path, manifest)
