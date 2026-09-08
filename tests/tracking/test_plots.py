"""Tests for the figure helpers behind the file-backed loggers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from substrax.tracking._plots import save_histogram, save_image_grid


@pytest.mark.parametrize(
    "images",
    [
        [np.zeros((8, 8))],
        [np.zeros((8, 8, 3))],
        [np.zeros((8, 8, 1)), np.ones((8, 8, 4)), np.zeros((8, 8))],
    ],
    ids=["gray", "rgb", "mixed"],
)
def test_save_image_grid_writes_a_png(tmp_path: Path, images: list[np.ndarray]) -> None:
    path = tmp_path / "grid.png"
    save_image_grid(path, images)
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_save_histogram_writes_a_png(tmp_path: Path) -> None:
    path = tmp_path / "hist.png"
    save_histogram(path, np.random.default_rng(0).normal(size=(4, 25)), title="Histogram")
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
