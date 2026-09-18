"""Mesh construction that the package documents uses Auto axes.

``jax.make_mesh`` builds ``AxisType.Explicit`` axes unless it is given ``axis_types`` (jax 0.11), and
under Explicit axes the backward pass of a batch-sharded step raises. Every example a reader copies,
whether in a docstring, the README or a docs page, therefore either names ``axis_types`` or builds
its mesh with ``DeviceMeshManager.create_device_mesh``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from substrax.testing.source_scans import documented_texts, unguarded_make_mesh_calls


ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.contract


def test_corpus_reaches_docstrings_readme_and_docs_pages() -> None:
    texts = documented_texts(ROOT, package="substrax")

    mesh_docstrings = [
        text for name, text in texts.items() if name.startswith("src/substrax/mesh/device_mesh.py:")
    ]
    assert any("``jax.make_mesh`` defaults to ``Explicit``" in text for text in mesh_docstrings)
    assert "README.md" in texts
    assert "docs/api/mesh.md" in texts


def test_documented_meshes_name_their_axis_types() -> None:
    violations = {
        name: calls
        for name, text in documented_texts(ROOT, package="substrax").items()
        if (calls := unguarded_make_mesh_calls(text))
    }

    assert violations == {}
