"""Weights & Biases logger.

The ``wandb`` SDK ships in the ``wandb`` extra and is imported when the logger is
constructed, so importing this module never imports it.
"""

from __future__ import annotations

import html
import logging
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

from substrax.tracking._optional import import_optional
from substrax.tracking.logger import (
    as_images,
    describe_histogram,
    describe_images,
    format_scalars,
    Logger,
    step_prefix,
    summarize,
)


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from jax.typing import ArrayLike


class WandbLogger(Logger):
    """Logger that records metrics, media and artifacts in a Weights & Biases run."""

    def __init__(
        self,
        name: str,
        project: str,
        *,
        entity: str | None = None,
        log_dir: str | Path | None = None,
        config: Mapping[str, Any] | None = None,
        tags: Sequence[str] | None = None,
        notes: str | None = None,
        level: int = logging.INFO,
        **init_options: Any,
    ) -> None:
        """Start the W&B run.

        Args:
            name: Name of the run and of the logger.
            project: Name of the W&B project.
            entity: User or team that owns the project.
            log_dir: Local directory for W&B's own files, in addition to the run.
            config: Hyperparameters to record on the run.
            tags: Tags for the run.
            notes: Notes about the run.
            level: Logging level.
            **init_options: Further keyword arguments for ``wandb.init``, such as
                ``mode="offline"`` or ``resume="allow"``; the SDK owns their meaning.

        Constructing the logger raises ``ImportError`` when the ``wandb`` extra is not
        installed.
        """
        super().__init__(name, log_dir, level)
        self._wandb = import_optional("wandb", extra="wandb")
        run = self._wandb.init(
            project=project,
            entity=entity,
            name=name,
            config=None if config is None else dict(config),
            tags=None if tags is None else list(tags),
            notes=notes,
            dir=None if self.log_dir is None else str(self.log_dir),
            **init_options,
        )
        self.run: Any | None = run
        self.info(f"Initialized W&B run: {run.name} (ID: {run.id})")

    def _active_run(self) -> Any:
        """Return the run, or fail when it has already been finished.

        Returns:
            The W&B run object.

        Raises:
            RuntimeError: If :meth:`finish` or :meth:`close` was called.
        """
        if self.run is None:
            raise RuntimeError(f"W&B run {self.name!r} has been finished.")
        return self.run

    def log_scalars(self, scalars: Mapping[str, float], step: int | None = None) -> None:
        """Log several scalars to the run."""
        self._wandb.log(dict(scalars), step=step)
        self.info(f"{step_prefix(step)}{format_scalars(scalars)}")

    def log_image(
        self, name: str, image: ArrayLike | Sequence[ArrayLike], step: int | None = None
    ) -> None:
        """Log an image or a list of images to the run."""
        images = as_images(image)
        self._wandb.log({name: [self._wandb.Image(item) for item in images]}, step=step)
        self.info(f"{step_prefix(step)}{describe_images(name, images)}")

    def log_histogram(
        self, name: str, values: ArrayLike | Sequence[float], step: int | None = None
    ) -> None:
        """Log the histogram and its summary statistics (``<name>/min`` and so on)."""
        summary = summarize(values)
        statistics = {f"{name}/{key}": value for key, value in summary.items()}
        histogram = self._wandb.Histogram(np.asarray(values).tolist())
        self._wandb.log({**statistics, name: histogram}, step=step)
        self.info(f"{step_prefix(step)}{describe_histogram(name, summary)}")

    def log_text(self, name: str, text: str, step: int | None = None) -> None:
        """Log ``text`` as a preformatted HTML block."""
        self._wandb.log({name: self._wandb.Html(f"<pre>{html.escape(text)}</pre>")}, step=step)
        self.info(f"{step_prefix(step)}Logged text for {name}")

    def log_hyperparams(self, params: Mapping[str, Any]) -> None:
        """Record the hyperparameters on the run's config."""
        self._active_run().config.update(dict(params))
        self.info(f"Logged {len(params)} hyperparameters to W&B")

    def log_code(self, root: str | Path | None = None) -> None:
        """Save the code under ``root`` (default: the current directory) to the run."""
        self._active_run().log_code(root=None if root is None else str(root))
        self.info("Logged code to W&B for reproducibility")

    def log_model(
        self,
        model_path: str | Path,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Upload a model file or directory as a W&B artifact of type ``model``.

        Args:
            model_path: File or directory to upload.
            name: Artifact name; defaults to ``<run name>_model``.
            metadata: Metadata to record on the artifact.
        """
        run = self._active_run()
        path = Path(model_path)
        artifact = self._wandb.Artifact(
            name=name or f"{run.name}_model",
            type="model",
            metadata=None if metadata is None else dict(metadata),
        )
        if path.is_dir():
            artifact.add_dir(str(path))
        else:
            artifact.add_file(str(path))
        run.log_artifact(artifact)
        self.info(f"Logged model from {path} to W&B Artifacts")

    def finish(self, exit_code: int = 0) -> None:
        """Finish the run; a second call is a no-op.

        Args:
            exit_code: Exit code to report to W&B.
        """
        if self.run is not None:
            self._wandb.finish(exit_code=exit_code)
            self.info(f"Finished W&B run: {self.run.name} (ID: {self.run.id})")
            self.run = None

    def close(self) -> None:
        """Finish the run and close the logger."""
        self.finish()
        super().close()
