"""Tests for the Weights & Biases logger against a mocked SDK."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from substrax.tracking import WandbLogger


@pytest.fixture
def wandb(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    sdk = MagicMock()
    run = MagicMock()
    run.name = "test_run"
    run.id = "test_run_id"
    sdk.init.return_value = run
    monkeypatch.setitem(sys.modules, "wandb", sdk)
    return sdk


@pytest.fixture
def logger(request: pytest.FixtureRequest, tmp_path: Path) -> WandbLogger:
    request.getfixturevalue("wandb")
    return WandbLogger(name="test_wandb", project="test_project", log_dir=tmp_path)


def test_init_starts_the_run(wandb: MagicMock, tmp_path: Path) -> None:
    logger = WandbLogger(
        name="test_wandb",
        project="test_project",
        entity="team",
        log_dir=tmp_path,
        config={"lr": 0.1},
        tags=("a", "b"),
        notes="note",
    )
    wandb.init.assert_called_once_with(
        project="test_project",
        entity="team",
        name="test_wandb",
        config={"lr": 0.1},
        tags=["a", "b"],
        notes="note",
        dir=str(tmp_path),
    )
    assert logger.run is wandb.init.return_value
    logger.close()


def test_init_without_a_log_dir(wandb: MagicMock) -> None:
    logger = WandbLogger(name="test_wandb", project="test_project")
    assert wandb.init.call_args.kwargs["dir"] is None
    logger.close()


def test_import_error_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "wandb", None)
    with pytest.raises(ImportError, match=r"substrax\[wandb\]"):
        WandbLogger(name="test_wandb", project="test_project")


def test_log_scalar(logger: WandbLogger, wandb: MagicMock) -> None:
    logger.log_scalar("test_metric", 0.5, step=10)
    wandb.log.assert_called_with({"test_metric": 0.5}, step=10)


def test_log_scalars(logger: WandbLogger, wandb: MagicMock) -> None:
    logger.log_scalars({"metric1": 0.5, "metric2": 0.7}, step=10)
    wandb.log.assert_called_with({"metric1": 0.5, "metric2": 0.7}, step=10)


def test_log_image_wraps_every_image(logger: WandbLogger, wandb: MagicMock) -> None:
    logger.log_image("single", np.zeros((4, 4, 3)), step=1)
    assert wandb.Image.call_count == 1
    logger.log_image("many", [np.zeros((4, 4)) for _ in range(3)], step=2)
    assert wandb.Image.call_count == 4
    payload, kwargs = wandb.log.call_args
    assert list(payload[0]) == ["many"]
    assert len(payload[0]["many"]) == 3
    assert kwargs == {"step": 2}


def test_log_histogram_logs_statistics_and_histogram(logger: WandbLogger, wandb: MagicMock) -> None:
    logger.log_histogram("values", np.array([1.0, 2.0, 3.0, 4.0, 5.0]), step=3)
    wandb.Histogram.assert_called_once_with([1.0, 2.0, 3.0, 4.0, 5.0])
    payload = wandb.log.call_args.args[0]
    assert payload["values/min"] == 1.0
    assert payload["values/max"] == 5.0
    assert payload["values/mean"] == 3.0
    assert payload["values/median"] == 3.0
    assert payload["values"] is wandb.Histogram.return_value
    assert wandb.log.call_args.kwargs == {"step": 3}


def test_log_text_escapes_html(logger: WandbLogger, wandb: MagicMock) -> None:
    logger.log_text("note", "<b>bold</b>", step=4)
    wandb.Html.assert_called_once_with("<pre>&lt;b&gt;bold&lt;/b&gt;</pre>")
    wandb.log.assert_called_with({"note": wandb.Html.return_value}, step=4)


def test_log_hyperparams_updates_the_run_config(logger: WandbLogger, wandb: MagicMock) -> None:
    logger.log_hyperparams({"learning_rate": 0.001, "batch_size": 32})
    wandb.init.return_value.config.update.assert_called_once_with(
        {"learning_rate": 0.001, "batch_size": 32}
    )


def test_log_code(logger: WandbLogger, wandb: MagicMock, tmp_path: Path) -> None:
    logger.log_code()
    wandb.init.return_value.log_code.assert_called_with(root=None)
    logger.log_code(tmp_path)
    wandb.init.return_value.log_code.assert_called_with(root=str(tmp_path))


def test_log_model_adds_a_file_or_a_directory(
    logger: WandbLogger, wandb: MagicMock, tmp_path: Path
) -> None:
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"\x00")
    logger.log_model(weights, metadata={"epoch": 3})
    wandb.Artifact.assert_called_with(name="test_run_model", type="model", metadata={"epoch": 3})
    wandb.Artifact.return_value.add_file.assert_called_once_with(str(weights))

    logger.log_model(tmp_path, name="dir-model")
    wandb.Artifact.assert_called_with(name="dir-model", type="model", metadata=None)
    wandb.Artifact.return_value.add_dir.assert_called_once_with(str(tmp_path))
    assert wandb.init.return_value.log_artifact.call_count == 2


def test_close_finishes_the_run_once(logger: WandbLogger, wandb: MagicMock) -> None:
    logger.close()
    logger.finish()
    wandb.finish.assert_called_once_with(exit_code=0)
    assert logger.run is None


def test_finish_reports_the_exit_code(logger: WandbLogger, wandb: MagicMock) -> None:
    logger.finish(exit_code=1)
    wandb.finish.assert_called_once_with(exit_code=1)
    logger.close()


def test_logging_after_finish_fails_fast(logger: WandbLogger) -> None:
    logger.close()
    with pytest.raises(RuntimeError, match="has been finished"):
        logger.log_hyperparams({"lr": 0.1})
