"""Native asynchronous capture, completion and resource ownership."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Event
from typing import Any

import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import pytest
from flax import nnx

from substrax.checkpoint import OrbaxCheckpointStore
from substrax.typing import JsonValue


class _Weights(nnx.Module):
    """A typed native graph for the donated update."""

    def __init__(self) -> None:
        self.weight = nnx.Param(jnp.arange(4, dtype=jnp.float32))


class _DeferredMetadata(ocp.AsyncCheckpointHandler):
    """Gate real JSON persistence through Orbax's public handler/future interface."""

    def __init__(self, reached: Event, release: Event) -> None:
        self.reached = reached
        self.release = release
        self.writer = ocp.JsonCheckpointHandler()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.fail = False

    async def async_save(self, directory: Any, args: Any) -> list[ocp.Future]:
        def write() -> None:
            self.reached.set()
            if not self.release.wait(15):
                raise TimeoutError("test did not release metadata persistence")
            if self.fail:
                raise OSError("injected metadata persistence failure")
            self.writer.save(directory, args=args)

        return [self.executor.submit(write)]

    def save(self, directory: Any, args: Any) -> None:
        self.writer.save(directory, args=args)

    def restore(self, directory: Any, args: Any) -> Any:
        return self.writer.restore(directory, args=args)

    def metadata(self, directory: Any) -> Any:
        return self.writer.metadata(directory)

    def close(self) -> None:
        self.executor.shutdown(wait=True)
        self.writer.close()


def _defer_metadata(
    monkeypatch: pytest.MonkeyPatch, reached: Event, release: Event
) -> _DeferredMetadata:
    native_manager = ocp.CheckpointManager
    handler = _DeferredMetadata(reached, release)

    def manager(directory: Any, *, options: ocp.CheckpointManagerOptions) -> ocp.CheckpointManager:
        registry = ocp.DefaultCheckpointHandlerRegistry()
        for argument in (
            ocp.args.PyTreeSave,
            ocp.args.PyTreeRestore,
            ocp.args.JsonSave,
            ocp.args.JsonRestore,
        ):
            registry.add(None, argument)
        registry.add("metadata", ocp.args.JsonSave, handler)
        registry.add("metadata", ocp.args.JsonRestore, handler)
        return native_manager(directory, options=options, handler_registry=registry)

    monkeypatch.setattr(ocp, "CheckpointManager", manager)
    return handler


def test_async_capture_allows_a_donated_update_before_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    finishing, release = Event(), Event()
    native_manager = ocp.CheckpointManager

    def hold_completion() -> None:
        finishing.set()
        if not release.wait(15):
            raise TimeoutError("test did not release checkpoint completion")

    def manager(directory: Any, *, options: ocp.CheckpointManagerOptions) -> ocp.CheckpointManager:
        return native_manager(
            directory,
            options=replace(
                options,
                async_options=ocp.options.AsyncOptions(post_finalization_callback=hold_completion),
            ),
        )

    # Configure the real backend through its public completion hook. No storage
    # operation is mocked: Orbax performs array capture, persistence and restore.
    monkeypatch.setattr(ocp, "CheckpointManager", manager)
    model = _Weights()

    @nnx.jit(graph=True, graph_updates=False, donate_argnums=0)
    def update(model: _Weights) -> None:
        model.weight[...] += 1

    with OrbaxCheckpointStore(tmp_path) as store:
        try:
            path = store.save_async(1, {"model": nnx.state(model)})
            assert path == tmp_path / "1"
            assert finishing.wait(5), "the real backend did not reach its completion hook"
            assert not release.is_set()
            update(model)
            np.testing.assert_array_equal(model.weight[...], [1, 2, 3, 4])
        finally:
            release.set()
        store.wait_until_finished()
        restored = store.restore(1, templates={"model": nnx.state(model)})
        np.testing.assert_array_equal(restored.items["model"]["weight"][...], [0, 1, 2, 3])


def test_async_metadata_is_isolated_from_caller_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reached, release = Event(), Event()
    _defer_metadata(monkeypatch, reached, release)
    labels: list[JsonValue] = ["original"]
    extra: dict[str, JsonValue] = {"run": {"labels": labels}}
    with OrbaxCheckpointStore(tmp_path) as store:
        try:
            store.save_async(1, {"data_iterator": {"position": 1}}, extra=extra)
            assert reached.wait(5), "the real JSON handler did not start"
            labels[0] = "changed"
        finally:
            release.set()
        store.wait_until_finished()
        assert store.read_metadata(1).extra == {"run": {"labels": ["original"]}}


def test_invalid_replacement_keeps_the_completed_checkpoint(tmp_path: Path) -> None:
    with OrbaxCheckpointStore(tmp_path, best=("loss", "min")) as store:
        store.save(1, {"data_iterator": {"position": 3}}, metrics={"loss": 2.0})
        with pytest.raises(ValueError, match="must be finite"):
            store.save_async(
                1,
                {"data_iterator": {"position": 9}},
                metrics={"loss": float("nan")},
                overwrite=True,
            )
        assert store.list_steps() == [1]
        assert store.restore(1).items["data_iterator"] == {"position": 3}


@pytest.mark.parametrize("observe", ["wait", "close"])
def test_async_completion_errors_propagate_and_keep_the_previous_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, observe: str
) -> None:
    reached, release = Event(), Event()
    release.set()
    handler = _defer_metadata(monkeypatch, reached, release)
    store = OrbaxCheckpointStore(tmp_path)
    try:
        store.save(1, {"data_iterator": {"position": 3}})
        handler.fail = True
        store.save_async(2, {"data_iterator": {"position": 9}})
        complete = store.wait_until_finished if observe == "wait" else store.close
        with pytest.raises(OSError, match="injected metadata persistence failure"):
            complete()
    finally:
        store.close()
    with OrbaxCheckpointStore(tmp_path) as reopened:
        assert reopened.list_steps() == [1]
        assert reopened.restore(1).items["data_iterator"] == {"position": 3}


def test_unused_wait_and_close_have_no_storage_side_effect(tmp_path: Path) -> None:
    path = tmp_path / "unused"
    store = OrbaxCheckpointStore(path)
    store.wait_until_finished()
    store.close()
    store.close()
    assert not path.exists()


@pytest.mark.parametrize("observe", ["query", "close", "save"])
def test_completion_boundaries_wait_for_native_persistence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, observe: str
) -> None:
    reached, release = Event(), Event()
    _defer_metadata(monkeypatch, reached, release)
    store = OrbaxCheckpointStore(tmp_path)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            try:
                if observe == "save":
                    pending = executor.submit(store.save, 1, {"data_iterator": {"position": 3}})
                else:
                    store.save_async(1, {"data_iterator": {"position": 3}})
                    pending = executor.submit(
                        store.list_steps if observe == "query" else store.close
                    )
                assert reached.wait(5), "the real JSON handler did not start"
                with pytest.raises(TimeoutError):
                    pending.result(timeout=0.1)
            finally:
                release.set()
            result = pending.result(timeout=10)
            if observe == "query":
                assert result == [1]
            elif observe == "save":
                assert result == tmp_path / "1"
    finally:
        release.set()
        store.close()
    with OrbaxCheckpointStore(tmp_path) as reopened:
        assert reopened.restore(1).items["data_iterator"] == {"position": 3}


def test_async_saves_share_native_preservation_across_reopen(tmp_path: Path) -> None:
    with OrbaxCheckpointStore(tmp_path, max_to_keep=1, best=("loss", "min")) as store:
        for step, loss in ((1, 1.0), (2, 3.0), (3, 2.0)):
            store.save_async(step, {"data_iterator": {"position": step}}, metrics={"loss": loss})
        store.wait_until_finished()
    with OrbaxCheckpointStore(tmp_path, max_to_keep=1, best=("loss", "min")) as reopened:
        assert reopened.list_steps() == [1, 3]
        assert reopened.best_step("loss") == 1
        assert reopened.restore(3).items["data_iterator"] == {"position": 3}
