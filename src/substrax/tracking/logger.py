"""Logger interface with console and file backends.

Every backend writes a human-readable line through :mod:`logging`. The file backend
also records scalars in a CSV file and saves images, histograms, texts and
hyperparameters as files under its log directory.
"""

from __future__ import annotations

import csv
import logging
import sys
from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

from substrax.tracking._plots import save_histogram, save_image_grid


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from jax.typing import ArrayLike
    from numpy.typing import NDArray

_LINE_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
_SCALAR_FIELDS = ("timestamp", "step", "name", "value")


def step_prefix(step: int | None) -> str:
    """Return the ``[Step N] `` prefix, or an empty string when there is no step."""
    return f"[Step {step}] " if step is not None else ""


def artifact_filename(name: str, step: int | None, suffix: str) -> str:
    """Return ``<name>_<timestamp>[_step<N>]<suffix>``, unique per second and step."""
    step_suffix = f"_step{step}" if step is not None else ""
    return f"{name}_{_timestamp()}{step_suffix}{suffix}"


def summarize(values: ArrayLike | Sequence[float]) -> dict[str, float]:
    """Return the min, max, mean, std and median of ``values`` as Python floats."""
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "median": float(np.median(array)),
    }


def as_images(image: ArrayLike | Sequence[ArrayLike]) -> list[NDArray[Any]]:
    """Return ``image`` as a list of numpy arrays, whether it is one image or several."""
    items = image if isinstance(image, Sequence) else [image]
    return [np.asarray(item) for item in items]


def format_scalars(scalars: Mapping[str, float]) -> str:
    """Return ``name: value`` pairs joined by commas, six significant digits each."""
    return ", ".join(f"{name}: {value:.6g}" for name, value in scalars.items())


def describe_images(name: str, images: Sequence[NDArray[Any]]) -> str:
    """Return the log line for one image (with its shape) or several."""
    if len(images) == 1:
        return f"Logged image {name} with shape {images[0].shape}"
    return f"Logged {len(images)} images for {name}"


def describe_histogram(name: str, summary: Mapping[str, float]) -> str:
    """Return the log line for a histogram from its :func:`summarize` output."""
    statistics = ", ".join(f"{key}={summary[key]:.6g}" for key in ("min", "max", "mean", "std"))
    return f"Logged histogram {name}: {statistics}"


def _timestamp() -> str:
    """Return the current local time as a filename-safe string."""
    return datetime.now().strftime(_TIMESTAMP_FORMAT)


def _configured(handler: logging.Handler, level: int) -> logging.Handler:
    """Return ``handler`` with the shared level and line format applied."""
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_LINE_FORMAT))
    return handler


class Logger(ABC):
    """Base class for all loggers.

    Concrete loggers record scalars, images, histograms, text and hyperparameters
    somewhere; every logger also owns a :mod:`logging` logger named after it whose
    handlers it replaces, so lines reach stdout exactly once.
    """

    def __init__(
        self, name: str, log_dir: str | Path | None = None, level: int = logging.INFO
    ) -> None:
        """Initialise the logger.

        Args:
            name: Name of the logger and of its :mod:`logging` logger.
            log_dir: Directory for files, created if missing. ``None`` writes no files.
            level: Logging level from the :mod:`logging` module.
        """
        self.name = name
        self.log_dir = None if log_dir is None else Path(log_dir)
        self.level = level
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)

        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        for handler in self._logger.handlers[:]:
            self._logger.removeHandler(handler)
        self._logger.addHandler(_configured(logging.StreamHandler(sys.stdout), level))

    def log(self, msg: str, level: int = logging.INFO) -> None:
        """Log a message at ``level``."""
        self._logger.log(level, msg)

    def debug(self, msg: str) -> None:
        """Log a debug message."""
        self._logger.debug(msg)

    def info(self, msg: str) -> None:
        """Log an info message."""
        self._logger.info(msg)

    def warning(self, msg: str) -> None:
        """Log a warning message."""
        self._logger.warning(msg)

    def error(self, msg: str) -> None:
        """Log an error message."""
        self._logger.error(msg)

    def critical(self, msg: str) -> None:
        """Log a critical message."""
        self._logger.critical(msg)

    def log_scalar(self, name: str, value: float, step: int | None = None) -> None:
        """Log one scalar; the default records it through :meth:`log_scalars`.

        Args:
            name: Name of the scalar.
            value: Scalar value to log.
            step: Global step value to record, for example the training iteration.
        """
        self.log_scalars({name: value}, step)

    @abstractmethod
    def log_scalars(self, scalars: Mapping[str, float], step: int | None = None) -> None:
        """Log several scalars.

        Args:
            scalars: Scalar names to values.
            step: Global step value to record.
        """

    @abstractmethod
    def log_image(
        self, name: str, image: ArrayLike | Sequence[ArrayLike], step: int | None = None
    ) -> None:
        """Log an image or a list of images.

        Args:
            name: Name of the image.
            image: One array of shape ``(H, W)`` or ``(H, W, C)``, or a list of them.
            step: Global step value to record.
        """

    @abstractmethod
    def log_histogram(
        self, name: str, values: ArrayLike | Sequence[float], step: int | None = None
    ) -> None:
        """Log a histogram of values.

        Args:
            name: Name of the histogram.
            values: Values to build the histogram from.
            step: Global step value to record.
        """

    @abstractmethod
    def log_text(self, name: str, text: str, step: int | None = None) -> None:
        """Log text.

        Args:
            name: Name of the text entry.
            text: Text to log.
            step: Global step value to record.
        """

    @abstractmethod
    def log_hyperparams(self, params: Mapping[str, Any]) -> None:
        """Log hyperparameters.

        Args:
            params: Hyperparameter names to values.
        """

    def close(self) -> None:
        """Close the logger's handlers and release their resources."""
        for handler in self._logger.handlers[:]:
            handler.close()
            self._logger.removeHandler(handler)


class ConsoleLogger(Logger):
    """Logger that writes lines to the console only."""

    def log_scalars(self, scalars: Mapping[str, float], step: int | None = None) -> None:
        """Log several scalars as one line."""
        self.info(f"{step_prefix(step)}{format_scalars(scalars)}")

    def log_image(
        self, name: str, image: ArrayLike | Sequence[ArrayLike], step: int | None = None
    ) -> None:
        """Log the image count or shape, not the pixels."""
        self.info(f"{step_prefix(step)}{describe_images(name, as_images(image))}")

    def log_histogram(
        self, name: str, values: ArrayLike | Sequence[float], step: int | None = None
    ) -> None:
        """Log the summary statistics of ``values``, not the histogram."""
        self.info(f"{step_prefix(step)}{describe_histogram(name, summarize(values))}")

    def log_text(self, name: str, text: str, step: int | None = None) -> None:
        """Log ``text`` under ``name``."""
        self.info(f"{step_prefix(step)}{name}:\n{text}")

    def log_hyperparams(self, params: Mapping[str, Any]) -> None:
        """Log one line per hyperparameter."""
        self.info("Hyperparameters:")
        for name, value in params.items():
            self.info(f"  {name}: {value}")


class FileLogger(ConsoleLogger):
    """Logger that writes the console lines to a file as well.

    Scalars also go to ``<name>_<timestamp>_metrics.csv`` (one ``timestamp, step, name,
    value`` row per scalar), images to ``images/``, histograms to ``histograms/``, texts
    to ``texts/`` and hyperparameters to ``hyperparams_<timestamp>.txt``. Images and
    histograms need the ``plots`` extra; without it they are logged as lines only.
    """

    def __init__(
        self,
        name: str,
        log_dir: str | Path,
        filename: str | None = None,
        level: int = logging.INFO,
    ) -> None:
        """Initialise the file logger.

        Args:
            name: Name of the logger.
            log_dir: Directory to save logs to.
            filename: Log filename. ``None`` derives one from ``name`` and the time.
            level: Logging level.
        """
        super().__init__(name, log_dir, level)
        self._root = Path(log_dir)
        stamp = _timestamp()
        self.log_file = self._root / (filename or f"{name}_{stamp}.log")
        self._logger.addHandler(_configured(logging.FileHandler(self.log_file), level))
        self.metrics_file = self._root / f"{name}_{stamp}_metrics.csv"
        self.info(f"Log file created at {self.log_file}")
        self.info(f"Metrics file created at {self.metrics_file}")

    def log_scalars(self, scalars: Mapping[str, float], step: int | None = None) -> None:
        """Log several scalars as one line and append them to the metrics CSV."""
        super().log_scalars(scalars, step)
        self._append_scalars(scalars, step)

    def log_image(
        self, name: str, image: ArrayLike | Sequence[ArrayLike], step: int | None = None
    ) -> None:
        """Log the image and save it under ``images/``."""
        images = as_images(image)
        super().log_image(name, images, step)
        self._save_plot("images", name, step, lambda path: save_image_grid(path, images))

    def log_histogram(
        self, name: str, values: ArrayLike | Sequence[float], step: int | None = None
    ) -> None:
        """Log the summary statistics and save the histogram under ``histograms/``."""
        super().log_histogram(name, values, step)
        array = np.asarray(values)
        self._save_plot(
            "histograms",
            name,
            step,
            lambda path: save_histogram(path, array, title=f"Histogram of {name}"),
        )

    def log_text(self, name: str, text: str, step: int | None = None) -> None:
        """Log ``text`` and save it under ``texts/``."""
        super().log_text(name, text, step)
        directory = self._subdirectory("texts")
        (directory / artifact_filename(name, step, ".txt")).write_text(text)

    def log_hyperparams(self, params: Mapping[str, Any]) -> None:
        """Log the hyperparameters and save them to ``hyperparams_<timestamp>.txt``."""
        super().log_hyperparams(params)
        lines = "".join(f"{name}: {value}\n" for name, value in params.items())
        (self._root / f"hyperparams_{_timestamp()}.txt").write_text(lines)

    def _append_scalars(self, scalars: Mapping[str, float], step: int | None) -> None:
        """Append one row per scalar to the metrics CSV, writing the header first."""
        is_new = not self.metrics_file.exists()
        stamp = datetime.now().isoformat()
        with self.metrics_file.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=_SCALAR_FIELDS)
            if is_new:
                writer.writeheader()
            for name, value in scalars.items():
                writer.writerow(
                    {
                        "timestamp": stamp,
                        "step": "" if step is None else step,
                        "name": name,
                        "value": value,
                    }
                )

    def _subdirectory(self, kind: str) -> Path:
        """Return ``<log_dir>/<kind>``, creating it if needed."""
        directory = self._root / kind
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _save_plot(
        self, kind: str, name: str, step: int | None, draw: Callable[[Path], None]
    ) -> None:
        """Draw a figure into ``<log_dir>/<kind>``, or warn when matplotlib is missing."""
        path = self._subdirectory(kind) / artifact_filename(name, step, ".png")
        try:
            draw(path)
        except ImportError as err:
            self.warning(f"{step_prefix(step)}Not saving {kind} for {name}: {err}")


def create_logger(
    name: str,
    log_dir: str | Path | None = None,
    *,
    log_to_file: bool = True,
    level: int = logging.INFO,
) -> Logger:
    """Create a console logger, or a file logger that also logs to the console.

    Args:
        name: Name of the logger.
        log_dir: Directory to save logs to. ``None`` with ``log_to_file`` means
            ``./logs/<name>``.
        log_to_file: Whether to write files as well as console lines.
        level: Logging level.

    Returns:
        A :class:`FileLogger` when ``log_to_file`` is set, else a :class:`ConsoleLogger`.
    """
    if not log_to_file:
        return ConsoleLogger(name, log_dir, level)
    return FileLogger(name, Path("logs") / name if log_dir is None else log_dir, level=level)
