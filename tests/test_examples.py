"""``discover_examples`` lists a repository's example scripts."""

from __future__ import annotations

from pathlib import Path

from substrax.examples import discover_examples


def test_discovery_skips_private_parts_and_package_markers(tmp_path: Path) -> None:
    root = tmp_path / "_site" / "examples"
    for relative in (
        "a/01_first.py",
        "a/__init__.py",
        "_templates/template.py",
        "_private.py",
        "b/helpers/_shared.py",
        "b/02_second.py",
        "b/notes.txt",
    ):
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text("", encoding="utf-8")

    found = discover_examples(root)
    narrowed = discover_examples(root, include=lambda path: path.name.startswith("02_"))

    assert [path.relative_to(root).as_posix() for path in found] == [
        "a/01_first.py",
        "b/02_second.py",
    ]
    assert [path.relative_to(root).as_posix() for path in narrowed] == ["b/02_second.py"]
