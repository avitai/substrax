"""MLflow logger.

The ``mlflow`` SDK ships in the ``mlflow`` extra and is imported when the logger is
constructed, so importing this module never imports it.
"""

from __future__ import annotations

import logging
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

from substrax.tracking._optional import import_optional
from substrax.tracking._plots import save_histogram, save_image_grid
from substrax.tracking.logger import (
    artifact_filename,
    as_images,
    describe_histogram,
    describe_images,
    format_scalars,
    Logger,
    step_prefix,
    summarize,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Mapping, Sequence

    from jax.typing import ArrayLike


class MLFlowLogger(Logger):
    """Logger that records metrics, parameters and artifacts in an MLflow run.

    Images, histograms and texts are written to ``<log_dir>/artifacts/<kind>/`` when a
    log directory is given, else to a temporary file, and logged as run artifacts under
    ``<kind>``. Images and histograms need the ``plots`` extra.
    """

    def __init__(
        self,
        name: str,
        *,
        log_dir: str | Path | None = None,
        experiment_name: str | None = None,
        run_name: str | None = None,
        run_id: str | None = None,
        tracking_uri: str | None = None,
        registry_uri: str | None = None,
        level: int = logging.INFO,
    ) -> None:
        """Start or resume the MLflow run.

        Args:
            name: Name of the logger; also the experiment name when none is given.
            log_dir: Local directory for artifacts, in addition to the run.
            experiment_name: MLflow experiment, created if missing.
            run_name: Name for a new run. MLflow generates one when omitted.
            run_id: An existing run to continue; ``experiment_name`` and ``run_name``
                are then ignored.
            tracking_uri: MLflow tracking server. ``None`` keeps the environment's.
            registry_uri: MLflow model registry. ``None`` keeps the tracking URI.
            level: Logging level.

        Constructing the logger raises ``ImportError`` when the ``mlflow`` extra is not
        installed.
        """
        super().__init__(name, log_dir, level)
        self._mlflow = import_optional("mlflow", extra="mlflow")
        if tracking_uri is not None:
            self._mlflow.set_tracking_uri(tracking_uri)
        if registry_uri is not None:
            self._mlflow.set_registry_uri(registry_uri)

        self.experiment_name = experiment_name or name
        run = self._resume_run(run_id) if run_id is not None else self._start_run(run_name)
        self.active_run: Any | None = run
        self.run_id: str = run.info.run_id
        self.artifact_dir = None if self.log_dir is None else self.log_dir / "artifacts"
        if self.artifact_dir is not None:
            self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def _resume_run(self, run_id: str) -> Any:
        """Continue an existing run."""
        run = self._mlflow.start_run(run_id=run_id)
        self.info(f"Resumed MLflow run: {run_id}")
        return run

    def _start_run(self, run_name: str | None) -> Any:
        """Start a new run in the experiment, creating the experiment if needed."""
        experiment = self._mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            experiment_id = self._mlflow.create_experiment(self.experiment_name)
            self.info(f"Created new MLflow experiment: {self.experiment_name}")
        else:
            experiment_id = experiment.experiment_id
            self.info(f"Using existing MLflow experiment: {self.experiment_name}")
        run = self._mlflow.start_run(experiment_id=experiment_id, run_name=run_name)
        self.info(f"Started new MLflow run: {run.info.run_id}")
        return run

    def log_scalar(self, name: str, value: float, step: int | None = None) -> None:
        """Log one metric."""
        self._mlflow.log_metric(name, value, step=step)
        self.info(f"{step_prefix(step)}{format_scalars({name: value})}")

    def log_scalars(self, scalars: Mapping[str, float], step: int | None = None) -> None:
        """Log several metrics."""
        self._mlflow.log_metrics({name: float(value) for name, value in scalars.items()}, step=step)
        self.info(f"{step_prefix(step)}{format_scalars(scalars)}")

    def log_image(
        self, name: str, image: ArrayLike | Sequence[ArrayLike], step: int | None = None
    ) -> None:
        """Log an image or a list of images as an artifact under ``images``."""
        images = as_images(image)
        self.info(f"{step_prefix(step)}{describe_images(name, images)}")
        self._log_plot("images", name, step, lambda path: save_image_grid(path, images))

    def log_histogram(
        self, name: str, values: ArrayLike | Sequence[float], step: int | None = None
    ) -> None:
        """Log the summary statistics as metrics and the histogram as an artifact."""
        summary = summarize(values)
        self.log_scalars({f"{name}/{key}": value for key, value in summary.items()}, step)
        self.info(f"{step_prefix(step)}{describe_histogram(name, summary)}")
        array = np.asarray(values)
        self._log_plot(
            "histograms",
            name,
            step,
            lambda path: save_histogram(path, array, title=f"Histogram of {name}"),
        )

    def log_text(self, name: str, text: str, step: int | None = None) -> None:
        """Log ``text`` as an artifact under ``texts``."""
        with self._artifact_directory("texts") as directory:
            path = directory / artifact_filename(name, step, ".txt")
            path.write_text(text)
            self._mlflow.log_artifact(str(path), "texts")
        self.info(f"{step_prefix(step)}Logged text for {name}")

    def log_hyperparams(self, params: Mapping[str, Any]) -> None:
        """Log the hyperparameters as run parameters.

        MLflow accepts strings, numbers and booleans; anything else is logged as its
        ``str`` representation.
        """
        accepted = {
            name: value if isinstance(value, str | int | float | bool) else str(value)
            for name, value in params.items()
        }
        self._mlflow.log_params(accepted)
        self.info(f"Logged {len(accepted)} hyperparameters to MLflow")

    def log_artifact(self, local_path: str | Path, artifact_path: str | None = None) -> None:
        """Log one file as a run artifact.

        Args:
            local_path: File to log.
            artifact_path: Directory within the run's artifacts to place it in.
        """
        self._mlflow.log_artifact(str(local_path), artifact_path)
        self.info(f"Logged artifact from {local_path} to MLflow")

    def log_artifacts(self, local_dir: str | Path, artifact_path: str | None = None) -> None:
        """Log every file in a directory as run artifacts.

        Args:
            local_dir: Directory whose files to log.
            artifact_path: Directory within the run's artifacts to place them in.
        """
        self._mlflow.log_artifacts(str(local_dir), artifact_path)
        self.info(f"Logged artifacts from {local_dir} to MLflow")

    def end_run(self) -> None:
        """End the run; a second call is a no-op."""
        if self.active_run is not None:
            self._mlflow.end_run()
            self.info(f"Ended MLflow run: {self.run_id}")
            self.active_run = None

    def close(self) -> None:
        """End the run and close the logger."""
        self.end_run()
        super().close()

    @contextmanager
    def _artifact_directory(self, kind: str) -> Generator[Path, None, None]:
        """Yield ``<artifact_dir>/<kind>``, or a temporary directory without a log dir."""
        if self.artifact_dir is not None:
            directory = self.artifact_dir / kind
            directory.mkdir(parents=True, exist_ok=True)
            yield directory
            return
        with tempfile.TemporaryDirectory() as temporary:
            yield Path(temporary)

    def _log_plot(
        self, kind: str, name: str, step: int | None, draw: Callable[[Path], None]
    ) -> None:
        """Draw a figure and log it under ``kind``, or warn when matplotlib is missing."""
        with self._artifact_directory(kind) as directory:
            path = directory / artifact_filename(name, step, ".png")
            try:
                draw(path)
            except ImportError as err:
                self.warning(f"{step_prefix(step)}Not logging {kind} for {name}: {err}")
                return
            self._mlflow.log_artifact(str(path), kind)
