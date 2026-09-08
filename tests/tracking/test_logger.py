"""Tests for the console and file loggers."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

import substrax.tracking._plots
from substrax.tracking import ConsoleLogger, create_logger, FileLogger, Logger


@pytest.fixture
def no_matplotlib(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every figure helper behave as if the plots extra were missing."""

    def missing(module: str, *, extra: str) -> None:
        raise ImportError(f"{module} is not installed ({extra})")

    monkeypatch.setattr(substrax.tracking._plots, "import_optional", missing)


def read_metrics(logger: FileLogger) -> list[dict[str, str]]:
    with logger.metrics_file.open(newline="") as handle:
        return list(csv.DictReader(handle))


class TestConstruction:
    def test_console_logger_creation(self) -> None:
        logger = ConsoleLogger(name="test_console_logger")
        assert logger.name == "test_console_logger"
        assert logger.log_dir is None
        logger.close()

    def test_file_logger_creation(self, tmp_path: Path) -> None:
        logger = FileLogger(name="test_file_logger", log_dir=tmp_path)
        assert logger.name == "test_file_logger"
        assert logger.log_dir == tmp_path
        assert logger.log_file.exists()
        # The metrics file is created by the first scalar, not by construction.
        assert not logger.metrics_file.exists()
        logger.close()

    def test_file_logger_accepts_a_string_directory_and_creates_it(self, tmp_path: Path) -> None:
        logger = FileLogger(name="nested", log_dir=str(tmp_path / "a" / "b"))
        assert logger.log_dir == tmp_path / "a" / "b"
        assert logger.log_file.parent.is_dir()
        logger.close()

    def test_logger_factory(self, tmp_path: Path) -> None:
        console = create_logger(name="test_factory_console", log_to_file=False)
        assert isinstance(console, ConsoleLogger)
        console.close()

        file_logger = create_logger(name="test_factory_file", log_dir=tmp_path)
        assert isinstance(file_logger, FileLogger)
        assert file_logger.log_dir == tmp_path
        file_logger.close()

    def test_logger_factory_defaults_to_logs_under_the_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        logger = create_logger(name="run")
        assert isinstance(logger, FileLogger)
        assert logger.log_dir == Path("logs") / "run"
        logger.close()


class TestHandlers:
    def test_logger_replaces_every_existing_handler(self) -> None:
        underlying = logging.getLogger("substrax-test-handlers")
        underlying.addHandler(logging.NullHandler())
        underlying.addHandler(logging.NullHandler())

        logger = ConsoleLogger("substrax-test-handlers")

        assert len(underlying.handlers) == 1
        assert isinstance(underlying.handlers[0], logging.StreamHandler)
        logger.close()

    def test_close_removes_the_handlers(self, tmp_path: Path) -> None:
        logger = FileLogger("substrax-test-close", log_dir=tmp_path)
        assert len(logging.getLogger("substrax-test-close").handlers) == 2
        logger.close()
        assert logging.getLogger("substrax-test-close").handlers == []

    def test_level_is_applied(self, caplog: pytest.LogCaptureFixture) -> None:
        logger = ConsoleLogger("substrax-test-level", level=logging.WARNING)
        # No at_level here: it would reset the level this test checks.
        logger.debug("hidden")
        logger.warning("shown")
        assert "hidden" not in caplog.text
        assert "shown" in caplog.text
        logger.close()

    def test_facade_levels(self, caplog: pytest.LogCaptureFixture) -> None:
        logger = ConsoleLogger("substrax-test-facade", level=logging.DEBUG)
        with caplog.at_level(logging.DEBUG, logger="substrax-test-facade"):
            logger.debug("d")
            logger.info("i")
            logger.warning("w")
            logger.error("e")
            logger.critical("c")
            logger.log("l", logging.ERROR)
        levels = [record.levelname for record in caplog.records]
        assert levels == ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "ERROR"]
        logger.close()


class TestConsoleLogger:
    def test_log_scalar(self, caplog: pytest.LogCaptureFixture) -> None:
        logger = ConsoleLogger(name="test_scalar")
        with caplog.at_level(logging.INFO, logger="test_scalar"):
            logger.log_scalar("test_metric", 0.5, step=10)
        assert "[Step 10] test_metric: 0.5" in caplog.text
        logger.close()

    def test_log_scalars_without_a_step(self, caplog: pytest.LogCaptureFixture) -> None:
        logger = ConsoleLogger(name="test_scalars")
        with caplog.at_level(logging.INFO, logger="test_scalars"):
            logger.log_scalars({"metric1": 0.5, "metric2": 0.7})
        assert "metric1: 0.5, metric2: 0.7" in caplog.text
        assert "[Step" not in caplog.text
        logger.close()

    def test_log_image(self, caplog: pytest.LogCaptureFixture) -> None:
        logger = ConsoleLogger(name="test_image")
        with caplog.at_level(logging.INFO, logger="test_image"):
            logger.log_image("test_image", np.zeros((32, 32, 3)), step=10)
            logger.log_image("test_images", [np.zeros((32, 32, 3)) for _ in range(3)], step=10)
        assert "Logged image test_image with shape (32, 32, 3)" in caplog.text
        assert "Logged 3 images for test_images" in caplog.text
        logger.close()

    def test_log_histogram_reports_summary_statistics(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        logger = ConsoleLogger(name="test_histogram")
        with caplog.at_level(logging.INFO, logger="test_histogram"):
            logger.log_histogram("values", jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]), step=0)
        assert "Logged histogram values: min=1, max=5, mean=3" in caplog.text
        logger.close()

    def test_log_text_and_hyperparams(self, caplog: pytest.LogCaptureFixture) -> None:
        logger = ConsoleLogger(name="test_text")
        with caplog.at_level(logging.INFO, logger="test_text"):
            logger.log_text("note", "This is a test message.", step=10)
            logger.log_hyperparams({"learning_rate": 0.001, "model_type": "VAE"})
        assert "[Step 10] note:\nThis is a test message." in caplog.text
        assert "  learning_rate: 0.001" in caplog.text
        assert "  model_type: VAE" in caplog.text
        logger.close()


class TestFileLoggerScalars:
    def test_log_scalar(self, tmp_path: Path) -> None:
        logger = FileLogger(name="test_scalar", log_dir=tmp_path)
        logger.log_scalar("test_metric", 0.5, step=10)
        rows = read_metrics(logger)
        assert rows == [
            {"timestamp": rows[0]["timestamp"], "step": "10", "name": "test_metric", "value": "0.5"}
        ]
        logger.close()

    def test_log_scalars(self, tmp_path: Path) -> None:
        logger = FileLogger(name="test_scalars", log_dir=tmp_path)
        logger.log_scalars({"metric1": 0.5, "metric2": 0.7}, step=10)
        rows = read_metrics(logger)
        assert [(row["name"], row["value"]) for row in rows] == [
            ("metric1", "0.5"),
            ("metric2", "0.7"),
        ]
        logger.close()

    def test_csv_keeps_one_header_when_the_scalar_set_changes(self, tmp_path: Path) -> None:
        logger = FileLogger(name="changing", log_dir=tmp_path)
        logger.log_scalars({"a": 1.0}, step=1)
        logger.log_scalars({"b": 2.0, "c": 3.0}, step=2)
        logger.log_scalar("a", 4.0)
        rows = read_metrics(logger)
        assert [(row["step"], row["name"], row["value"]) for row in rows] == [
            ("1", "a", "1.0"),
            ("2", "b", "2.0"),
            ("2", "c", "3.0"),
            ("", "a", "4.0"),
        ]
        assert logger.metrics_file.read_text().count("timestamp,step,name,value") == 1
        logger.close()

    def test_file_logger_with_jax_arrays(self, tmp_path: Path) -> None:
        logger = FileLogger("test_logger", log_dir=tmp_path)
        logger.log_scalar("loss", float(jnp.array(0.123)), step=0)
        logger.log_scalar("accuracy", float(jnp.array(0.987)), step=1)
        assert len(list(tmp_path.glob("test_logger_*_metrics.csv"))) == 1
        assert logger.log_file.read_text().count("Step") == 2
        logger.close()


class TestFileLoggerArtifacts:
    def test_log_image_saves_one_file_per_call(self, tmp_path: Path) -> None:
        logger = FileLogger(name="test_image", log_dir=tmp_path)
        logger.log_image("test_image", np.zeros((32, 32, 3)), step=10)
        logger.log_image("test_image", [np.zeros((32, 32)) for _ in range(3)], step=11)
        files = sorted((tmp_path / "images").glob("test_image_*.png"))
        assert [path.name.endswith(("_step10.png", "_step11.png")) for path in files] == [
            True,
            True,
        ]
        logger.close()

    def test_log_histogram_saves_a_figure(self, tmp_path: Path) -> None:
        logger = FileLogger("test_logger", log_dir=tmp_path)
        logger.log_histogram("values", jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]), step=0)
        assert len(list((tmp_path / "histograms").glob("values_*_step0.png"))) == 1
        logger.close()

    @pytest.mark.usefixtures("no_matplotlib")
    def test_missing_plots_extra_warns_and_keeps_logging(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        logger = FileLogger("no-plots", log_dir=tmp_path)
        with caplog.at_level(logging.INFO, logger="no-plots"):
            logger.log_image("img", np.zeros((4, 4)), step=1)
            logger.log_histogram("hist", np.arange(4.0), step=1)
        assert "Not saving images for img" in caplog.text
        assert "Not saving histograms for hist" in caplog.text
        assert "Logged image img with shape (4, 4)" in caplog.text
        assert list((tmp_path / "images").iterdir()) == []
        assert list((tmp_path / "histograms").iterdir()) == []
        logger.close()

    def test_log_hyperparams(self, tmp_path: Path) -> None:
        logger = FileLogger(name="test_hyperparams", log_dir=tmp_path)
        logger.log_hyperparams({"learning_rate": 0.001, "batch_size": 32, "model_type": "VAE"})
        files = list(tmp_path.glob("hyperparams_*.txt"))
        assert len(files) == 1
        assert files[0].read_text() == "learning_rate: 0.001\nbatch_size: 32\nmodel_type: VAE\n"
        logger.close()

    def test_log_text(self, tmp_path: Path) -> None:
        logger = FileLogger(name="test_text", log_dir=tmp_path)
        logger.log_text("test_text", "This is a test message.", step=10)
        files = list((tmp_path / "texts").glob("test_text_*_step10.txt"))
        assert len(files) == 1
        assert files[0].read_text() == "This is a test message."
        logger.close()


class TestLoggerUsageInNNXModule:
    def test_nnx_module_with_logging(self, caplog: pytest.LogCaptureFixture) -> None:
        class TrainingModule(nnx.Module):
            def __init__(self, *, rngs: nnx.Rngs) -> None:
                super().__init__()
                self.dense = nnx.Linear(10, 1, rngs=rngs)

            def train_step(
                self, x: jax.Array, y: jax.Array, logger: Logger, step: int
            ) -> jax.Array:
                loss = jnp.mean((self.dense(x) - y) ** 2)
                # The conversion to float happens outside the computation.
                logger.log_scalar("train/loss", float(loss), step=step)
                return loss

        module = TrainingModule(rngs=nnx.Rngs(42))
        logger = ConsoleLogger("training")
        with caplog.at_level(logging.INFO, logger="training"):
            loss = module.train_step(jnp.ones((4, 10)), jnp.ones((4, 1)), logger, step=0)
        assert isinstance(loss, jax.Array)
        assert "[Step 0] train/loss: " in caplog.text
        logger.close()
