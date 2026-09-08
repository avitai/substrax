"""Tests for the MLflow logger against a mocked SDK."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

import substrax.tracking._plots
from substrax.tracking import MLFlowLogger


@pytest.fixture
def mlflow(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    sdk = MagicMock()
    run = MagicMock()
    run.info.run_id = "test_run_id"
    sdk.start_run.return_value = run
    sdk.get_experiment_by_name.return_value = None
    sdk.create_experiment.return_value = "exp-1"
    monkeypatch.setitem(sys.modules, "mlflow", sdk)
    return sdk


@pytest.fixture
def logger(request: pytest.FixtureRequest, tmp_path: Path) -> MLFlowLogger:
    request.getfixturevalue("mlflow")
    return MLFlowLogger(
        name="test_mlflow",
        log_dir=tmp_path,
        experiment_name="test_experiment",
        run_name="test_run",
    )


def test_init_creates_a_missing_experiment_and_starts_a_run(
    mlflow: MagicMock, tmp_path: Path
) -> None:
    logger = MLFlowLogger(
        name="test_mlflow",
        log_dir=tmp_path,
        experiment_name="test_experiment",
        run_name="test_run",
        tracking_uri="http://tracking",
        registry_uri="http://registry",
    )
    mlflow.set_tracking_uri.assert_called_once_with("http://tracking")
    mlflow.set_registry_uri.assert_called_once_with("http://registry")
    mlflow.create_experiment.assert_called_once_with("test_experiment")
    mlflow.start_run.assert_called_once_with(experiment_id="exp-1", run_name="test_run")
    assert logger.run_id == "test_run_id"
    assert logger.artifact_dir == tmp_path / "artifacts"
    assert (tmp_path / "artifacts").is_dir()
    logger.close()


def test_init_reuses_an_existing_experiment_named_after_the_logger(mlflow: MagicMock) -> None:
    mlflow.get_experiment_by_name.return_value = MagicMock(experiment_id="exp-existing")
    logger = MLFlowLogger(name="by-name")
    mlflow.get_experiment_by_name.assert_called_once_with("by-name")
    mlflow.create_experiment.assert_not_called()
    mlflow.start_run.assert_called_once_with(experiment_id="exp-existing", run_name=None)
    assert logger.artifact_dir is None
    logger.close()


def test_init_resumes_a_run_by_id(mlflow: MagicMock) -> None:
    logger = MLFlowLogger(name="resume", run_id="old-run", experiment_name="ignored")
    mlflow.start_run.assert_called_once_with(run_id="old-run")
    mlflow.get_experiment_by_name.assert_not_called()
    logger.close()


def test_import_error_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "mlflow", None)
    with pytest.raises(ImportError, match=r"substrax\[mlflow\]"):
        MLFlowLogger(name="test_mlflow_error")


def test_log_scalar(logger: MLFlowLogger, mlflow: MagicMock) -> None:
    logger.log_scalar("test_metric", 0.5, step=10)
    mlflow.log_metric.assert_called_with("test_metric", 0.5, step=10)


def test_log_scalars(logger: MLFlowLogger, mlflow: MagicMock) -> None:
    logger.log_scalars({"metric1": 0.5, "metric2": 0.7}, step=10)
    mlflow.log_metrics.assert_called_with({"metric1": 0.5, "metric2": 0.7}, step=10)


def test_log_hyperparams_stringifies_complex_values(
    logger: MLFlowLogger, mlflow: MagicMock
) -> None:
    logger.log_hyperparams({"learning_rate": 0.001, "batch_size": 32, "shape": (2, 3)})
    mlflow.log_params.assert_called_with(
        {"learning_rate": 0.001, "batch_size": 32, "shape": "(2, 3)"}
    )


def test_log_text_writes_under_the_artifact_dir(logger: MLFlowLogger, mlflow: MagicMock) -> None:
    logger.log_text("note", "hello", step=2)
    files = list((logger.artifact_dir or Path()).glob("texts/note_*_step2.txt"))
    assert len(files) == 1
    assert files[0].read_text() == "hello"
    mlflow.log_artifact.assert_called_once_with(str(files[0]), "texts")


def test_log_text_without_a_log_dir_uses_a_temporary_file(mlflow: MagicMock) -> None:
    logger = MLFlowLogger(name="no-dir")
    logger.log_text("note", "hello")
    (path, kind), _ = mlflow.log_artifact.call_args
    assert kind == "texts"
    assert Path(path).name.startswith("note_")
    assert not Path(path).exists()
    logger.close()


def test_log_image_logs_a_png_artifact(logger: MLFlowLogger, mlflow: MagicMock) -> None:
    logger.log_image("img", [np.zeros((4, 4)), np.zeros((4, 4, 3))], step=1)
    (path, kind), _ = mlflow.log_artifact.call_args
    assert kind == "images"
    assert Path(path).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert Path(path).parent == (logger.artifact_dir or Path()) / "images"


def test_log_histogram_logs_statistics_and_a_png_artifact(
    logger: MLFlowLogger, mlflow: MagicMock
) -> None:
    logger.log_histogram("values", [1.0, 2.0, 3.0, 4.0, 5.0], step=3)
    metrics = mlflow.log_metrics.call_args.args[0]
    assert metrics == {
        "values/min": 1.0,
        "values/max": 5.0,
        "values/mean": 3.0,
        "values/std": pytest.approx(np.std([1.0, 2.0, 3.0, 4.0, 5.0])),
        "values/median": 3.0,
    }
    assert mlflow.log_metrics.call_args.kwargs == {"step": 3}
    (path, kind), _ = mlflow.log_artifact.call_args
    assert kind == "histograms"
    assert Path(path).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_missing_plots_extra_warns_and_logs_no_artifact(
    logger: MLFlowLogger,
    mlflow: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def missing(module: str, *, extra: str) -> None:
        raise ImportError(f"{module} ({extra})")

    monkeypatch.setattr(substrax.tracking._plots, "import_optional", missing)
    with caplog.at_level(logging.WARNING, logger="test_mlflow"):
        logger.log_image("img", np.zeros((4, 4)), step=1)
    assert "Not logging images for img" in caplog.text
    mlflow.log_artifact.assert_not_called()


def test_log_artifact_and_log_artifacts(
    logger: MLFlowLogger, mlflow: MagicMock, tmp_path: Path
) -> None:
    logger.log_artifact(tmp_path / "file.txt", "outputs")
    mlflow.log_artifact.assert_called_with(str(tmp_path / "file.txt"), "outputs")
    logger.log_artifacts(tmp_path)
    mlflow.log_artifacts.assert_called_with(str(tmp_path), None)


def test_close_ends_the_run_once(logger: MLFlowLogger, mlflow: MagicMock) -> None:
    logger.close()
    logger.end_run()
    mlflow.end_run.assert_called_once()
    assert logger.active_run is None
