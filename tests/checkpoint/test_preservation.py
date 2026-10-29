"""Native preservation composes newest and best without competing deletion loops."""

from pathlib import Path
from typing import Any, Literal

import jax.numpy as jnp
import pytest

from substrax.checkpoint import OrbaxCheckpointStore


@pytest.mark.parametrize("mode", ["min", "max"])
def test_newest_and_old_best_survive_reopen_and_a_better_checkpoint(
    tmp_path: Path, mode: Literal["min", "max"]
) -> None:
    for step in range(1, 8):
        score = float(step) if mode == "min" else -float(step)
        with OrbaxCheckpointStore(tmp_path, max_to_keep=5, best=("score", mode)) as store:
            store.save(
                step, {"model": {"weight": jnp.asarray(float(step))}}, metrics={"score": score}
            )
    with OrbaxCheckpointStore(tmp_path, max_to_keep=5, best=("score", mode)) as store:
        assert store.list_steps() == [1, 3, 4, 5, 6, 7]
        assert store.best_step("score", mode=mode) == 1
        better = -1.0 if mode == "min" else 1.0
        store.save(8, {"model": {"weight": jnp.asarray(8.0)}}, metrics={"score": better})
        assert store.list_steps() == [4, 5, 6, 7, 8]
        assert store.best_step("score", mode=mode) == 8


def test_unscored_checkpoints_do_not_replace_the_best(tmp_path: Path) -> None:
    with OrbaxCheckpointStore(tmp_path, max_to_keep=1, best=("score", "min")) as store:
        store.save(1, {"data_iterator": {"position": 1}}, metrics={"score": 1.0})
        store.save(2, {"data_iterator": {"position": 2}})
        store.save(3, {"data_iterator": {"position": 3}}, metrics={"other": 0.0})
        assert store.list_steps() == [1, 3]
    with OrbaxCheckpointStore(tmp_path, max_to_keep=1, best=("score", "min")) as store:
        assert store.best_step("score") == 1
        store.save(4, {"data_iterator": {"position": 4}}, metrics={"score": 2.0})
        assert store.list_steps() == [1, 4]


def test_keep_all_with_best_policy_keeps_every_checkpoint(tmp_path: Path) -> None:
    with OrbaxCheckpointStore(tmp_path, max_to_keep=None, best=("score", "min")) as store:
        for step in range(7):
            store.save(step, {"data_iterator": {"position": step}}, metrics={"score": float(step)})
        assert store.list_steps() == list(range(7))


def test_equal_best_scores_preserve_the_newest_tie(tmp_path: Path) -> None:
    with OrbaxCheckpointStore(tmp_path, max_to_keep=1, best=("score", "min")) as store:
        for step in (1, 2):
            store.save(step, {"data_iterator": {"position": step}}, metrics={"score": 1.0})
        assert store.list_steps() == [2]
        assert store.best_step("score") == 2


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_ranking_cannot_replace_a_valid_checkpoint(tmp_path: Path, score: float) -> None:
    with OrbaxCheckpointStore(tmp_path, max_to_keep=1, best=("score", "min")) as store:
        store.save(1, {"data_iterator": {"position": 1}}, metrics={"score": 1.0})
        with pytest.raises(ValueError, match="finite"):
            store.save(2, {"data_iterator": {"position": 2}}, metrics={"score": score})
        assert store.list_steps() == [1]
        assert store.best_step("score") == 1


@pytest.mark.parametrize("best", [None, ("score", "min")])
def test_negative_retention_is_refused_before_publication(tmp_path: Path, best: Any) -> None:
    with (
        pytest.raises(ValueError, match=r"non-negative|max_to_keep"),
        OrbaxCheckpointStore(tmp_path, max_to_keep=-1, best=best) as store,
    ):
        store.save(0, {"data_iterator": {"position": 0}}, metrics={"score": 1.0})
    assert not (tmp_path / "0").exists()


@pytest.mark.parametrize("best", [("", "min"), ("score", "sideways")])
def test_best_configuration_is_validated_before_storage(tmp_path: Path, best: Any) -> None:
    with pytest.raises(ValueError, match="best"):
        OrbaxCheckpointStore(tmp_path, best=best)
