"""Manifests of a run's output directory, for SkyPilot's copy transport.

The cluster prints the manifest of its outputs (``python -m substrax.compute.sky_transfer DIR``)
before and after they are copied, and the copy is verified against it. A link is recorded by its
target and never followed.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path, PurePosixPath


type Manifest = dict[str, dict[str, str | int]]
"""Each relative path's identity: a file's size and digest, a directory, or a link's target."""


def snapshot(root: Path) -> Manifest:
    """The manifest of every entry under ``root``.

    Args:
        root: The output directory.

    Returns:
        Each relative path's identity.

    Raises:
        ValueError: If ``root`` is not a directory, is a link, or holds an entry that is neither a
            file, a directory nor a link.
    """
    if not root.is_dir() or root.is_symlink():
        msg = f"{root} is not an output directory"
        raise ValueError(msg)
    return {path.relative_to(root).as_posix(): _identity(path) for path in sorted(root.rglob("*"))}


def verify(root: Path, expected: Manifest) -> None:
    """Check that ``root`` holds exactly what ``expected`` describes.

    Args:
        root: The copied directory.
        expected: The manifest of the directory it was copied from.

    Raises:
        ValueError: If a path in ``expected`` leaves ``root``, or the copy differs from it.
    """
    for name in expected:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or path.as_posix() != name or name == ".":
            msg = f"unsafe path {name!r} in the manifest"
            raise ValueError(msg)
    if snapshot(root) != expected:
        msg = f"the copy in {root} does not match the manifest it was copied from"
        raise ValueError(msg)


def _identity(path: Path) -> dict[str, str | int]:
    if path.is_symlink():
        return {"kind": "symlink", "target": str(path.readlink())}
    if path.is_dir():
        return {"kind": "directory"}
    if path.is_file():
        return _file_identity(path)
    msg = f"{path} is neither a file, a directory nor a link"
    raise ValueError(msg)


def _file_identity(path: Path) -> dict[str, str | int]:
    """A file's size and SHA-256; a file that changes while it is read is refused."""
    before = path.stat()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        msg = f"{path} changed while it was read"
        raise ValueError(msg)
    return {"kind": "file", "bytes": after.st_size, "sha256": digest}


def main() -> None:
    """Print the manifest of the directory named on the command line."""
    sys.stdout.write(json.dumps(snapshot(Path(sys.argv[1])), sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
