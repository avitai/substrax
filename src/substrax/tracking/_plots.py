"""Matplotlib figures for the file-backed loggers.

Matplotlib ships in the ``plots`` extra; every function here raises ``ImportError``
when it is missing, and the loggers turn that into a warning.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from substrax.tracking._optional import import_optional


if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from numpy.typing import NDArray

_COLOR_CHANNEL_COUNTS = (1, 3, 4)
_IMAGE_NDIM = 3
_HISTOGRAM_BINS = 50
_INCHES_PER_IMAGE = 3
_HISTOGRAM_SIZE = (8, 6)


def _is_color_image(image: NDArray[Any]) -> bool:
    """Return whether ``image`` has a trailing channel axis of 1, 3 or 4."""
    return image.ndim == _IMAGE_NDIM and image.shape[2] in _COLOR_CHANNEL_COUNTS


def save_image_grid(path: Path, images: Sequence[NDArray[Any]]) -> None:
    """Save ``images`` side by side as one PNG.

    Colour images (a trailing axis of 1, 3 or 4 channels) are drawn as given; anything
    else is drawn in greyscale.

    Args:
        path: Destination file; its directory must exist.
        images: One or more arrays of shape ``(H, W)`` or ``(H, W, C)``.

    Propagates the ``ImportError`` of :func:`import_optional` when matplotlib is missing.
    """
    plt = import_optional("matplotlib.pyplot", extra="plots")
    figure, axes = plt.subplots(
        1, len(images), figsize=(_INCHES_PER_IMAGE * len(images), _INCHES_PER_IMAGE), squeeze=False
    )
    for axis, image in zip(axes[0], images, strict=True):
        axis.imshow(image, cmap=None if _is_color_image(image) else "gray")
        axis.set_axis_off()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def save_histogram(path: Path, values: NDArray[Any], *, title: str) -> None:
    """Save a histogram of ``values`` as a PNG.

    Args:
        path: Destination file; its directory must exist.
        values: Any array; it is flattened.
        title: Figure title.

    Propagates the ``ImportError`` of :func:`import_optional` when matplotlib is missing.
    """
    plt = import_optional("matplotlib.pyplot", extra="plots")
    figure, axis = plt.subplots(figsize=_HISTOGRAM_SIZE)
    axis.hist(values.ravel(), bins=_HISTOGRAM_BINS)
    axis.set_title(title)
    axis.set_xlabel("Value")
    axis.set_ylabel("Frequency")
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
