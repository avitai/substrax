"""Format 2 checkpoints restore through the migration registry and upgrade to new roots.

The fixtures under ``fixtures/format2`` were written by substrax 0.1.5 through
``scripts/make_format2_fixtures.py``, one per producer layout: the module-only payload
substrax, opifex and cellifex wrote, artifex's trainer tree, pertrax's phase tree,
DiffAV's model and model-plus-optimizer payloads, and a datarax iterator state. Each
producer's own layout is a ``LegacyLayout`` it passes to ``restore``; substrax ships the
module-only one.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import optax
import pytest
from _helpers import FIXTURE_STEP, FORMAT2_FIXTURES, raw_metadata
from flax import nnx

from substrax.checkpoint import (
    CURRENT_FORMAT_VERSION,
    LegacyLayout,
    Migration,
    MigrationRegistry,
    MODULE_ONLY_FORMAT2,
    OrbaxCheckpointStore,
    UnsupportedCheckpointError,
    upgrade_checkpoints,
)
from substrax.checkpoint.migration import DEFAULT_REGISTRY, FORMAT2_TO_3


LAYOUTS = (
    "substrax_module",
    "opifex_module",
    "artifex_trainer",
    "pertrax_phase",
    "diffav_model",
    "diffav_model_optimizer",
    "datarax_iterator",
)

# The producers' own layouts, as their adoption changes define them.
ARTIFEX_TRAINER_FORMAT2 = LegacyLayout(
    name="artifex-trainer",
    items_of=lambda payload: {
        "model": payload["model"],
        "optimizer": payload["opt_state"],
        "rng": payload["rng"],
        "extensions": payload["extensions"],
    },
    template_of=lambda templates: {
        "model": templates["model"],
        "opt_state": templates["optimizer"],
        "rng": templates["rng"],
        "extensions": templates["extensions"],
    },
)
PERTRAX_PHASE_FORMAT2 = LegacyLayout(
    name="pertrax-phase",
    items_of=lambda payload: {
        **ARTIFEX_TRAINER_FORMAT2.items_of(payload["trainer"]),
        "data_iterator": payload["iterator"],
    },
    template_of=lambda templates: {
        "trainer": ARTIFEX_TRAINER_FORMAT2.template_of(templates),
        "step": 0,
        "iterator": templates["data_iterator"],
    },
)
DIFFAV_FORMAT2 = LegacyLayout(name="diffav", items_of=dict, template_of=dict)
DATARAX_ITERATOR_FORMAT2 = LegacyLayout(
    name="datarax-iterator",
    items_of=lambda payload: {"data_iterator": payload},
    template_of=lambda templates: templates["data_iterator"],
)


def _model() -> nnx.Module:
    return nnx.Linear(in_features=4, out_features=2, rngs=nnx.Rngs(0))


def _trainer_templates() -> dict[str, Any]:
    model = _model()
    return {
        "model": nnx.state(model),
        "optimizer": optax.adam(1e-3).init(nnx.state(model, nnx.Param)),
        "rng": jax.random.PRNGKey(0),
        "extensions": {"regulariser": nnx.state(_model())},
    }


def _iterator_template() -> dict[str, Any]:
    """A fresh pipeline's state: plain Python leaves, which Orbax accepts as a template."""
    return {"position": 0, "epoch": 0, "rng_counts": [0, 0]}


class TestRegistry:
    def test_the_default_registry_covers_every_version_below_the_current_one(self) -> None:
        assert DEFAULT_REGISTRY.source_versions == tuple(range(2, CURRENT_FORMAT_VERSION))

    def test_a_gap_in_the_registry_is_refused(self) -> None:
        one = Migration(source_version=1, applies=lambda _raw: False, upgrade=FORMAT2_TO_3.upgrade)
        with pytest.raises(ValueError, match="contiguous"):
            MigrationRegistry([one])

    def test_two_migrations_for_one_version_are_refused(self) -> None:
        with pytest.raises(ValueError, match="two migrations read format 2"):
            MigrationRegistry([FORMAT2_TO_3, FORMAT2_TO_3])

    def test_upgrading_a_current_record_is_refused(self, tmp_path: Path) -> None:
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        store.save(1, {"model": nnx.state(_model())})
        raw = raw_metadata(store.directory, 1)
        with pytest.raises(UnsupportedCheckpointError, match="no migration"):
            DEFAULT_REGISTRY.upgrade({}, raw, MODULE_ONLY_FORMAT2)

    def test_a_migration_for_the_current_format_is_refused(self) -> None:
        current = Migration(
            source_version=CURRENT_FORMAT_VERSION,
            applies=lambda _raw: False,
            upgrade=FORMAT2_TO_3.upgrade,
        )
        with pytest.raises(ValueError, match="current"):
            MigrationRegistry([FORMAT2_TO_3, current])

    @pytest.mark.parametrize("layout", LAYOUTS)
    def test_the_format_2_predicate_recognises_every_fixture(self, layout: str) -> None:
        raw = raw_metadata(FORMAT2_FIXTURES / layout, FIXTURE_STEP)
        assert raw["checkpoint_version"] == "2.0"
        assert FORMAT2_TO_3.applies(raw)
        assert DEFAULT_REGISTRY.for_metadata(raw) is FORMAT2_TO_3

    def test_the_format_2_predicate_rejects_a_format_3_record(self, tmp_path: Path) -> None:
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        store.save(1, {"model": nnx.state(_model())})
        raw = raw_metadata(store.directory, 1)
        assert not FORMAT2_TO_3.applies(raw)
        assert DEFAULT_REGISTRY.for_metadata(raw) is None


class TestFixturesRestore:
    def test_the_module_only_layout_is_the_default(self) -> None:
        store = OrbaxCheckpointStore(FORMAT2_FIXTURES / "substrax_module")
        model = _model()

        checkpoint = store.restore(FIXTURE_STEP, templates={"model": nnx.state(model)})

        assert set(checkpoint.items) == {"model"}
        assert checkpoint.step == FIXTURE_STEP
        assert checkpoint.metadata.format_version == CURRENT_FORMAT_VERSION
        assert checkpoint.metadata.metrics == {"loss": 0.25}
        assert checkpoint.metadata.epoch is None
        assert checkpoint.metadata.extra["model_type"] == "nnx_module"
        assert checkpoint.metadata.extra["upgraded_from"] == {"format_version": 2}
        assert checkpoint.metadata.created_at.endswith("+00:00")
        nnx.update(model, checkpoint.items["model"])
        assert checkpoint.items["model"]["kernel"][...].shape == (4, 2)

    def test_the_module_only_fixture_restores_without_templates(self) -> None:
        checkpoint = OrbaxCheckpointStore(FORMAT2_FIXTURES / "substrax_module").restore(
            FIXTURE_STEP
        )
        assert checkpoint.items["model"]["kernel"]["value"].shape == (4, 2)

    def test_opifex_metadata_maps_onto_the_record(self) -> None:
        store = OrbaxCheckpointStore(FORMAT2_FIXTURES / "opifex_module")
        metadata = store.read_metadata(FIXTURE_STEP)

        assert metadata.epoch == 2
        assert metadata.metrics == {"loss": 0.25}
        assert metadata.extra["physics_metadata"] == {"pde": "burgers"}
        assert "step" not in metadata.extra

    def test_the_artifex_trainer_tree_becomes_four_items(self) -> None:
        store = OrbaxCheckpointStore(FORMAT2_FIXTURES / "artifex_trainer")
        templates = _trainer_templates()

        checkpoint = store.restore(
            FIXTURE_STEP, templates=templates, legacy_layout=ARTIFEX_TRAINER_FORMAT2
        )

        assert set(checkpoint.items) == {"model", "optimizer", "rng", "extensions"}
        assert checkpoint.metadata.items == ("model", "optimizer", "rng", "extensions")
        assert checkpoint.items["rng"].shape == (2,)
        assert set(checkpoint.items["extensions"]) == {"regulariser"}
        assert jax.tree.structure(checkpoint.items["optimizer"]) == jax.tree.structure(
            templates["optimizer"]
        )

    def test_the_pertrax_phase_tree_adds_the_iterator(self) -> None:
        store = OrbaxCheckpointStore(FORMAT2_FIXTURES / "pertrax_phase")
        templates = {**_trainer_templates(), "data_iterator": _iterator_template()}

        checkpoint = store.restore(
            FIXTURE_STEP, templates=templates, legacy_layout=PERTRAX_PHASE_FORMAT2
        )

        assert set(checkpoint.items) == {"model", "optimizer", "rng", "extensions", "data_iterator"}
        assert checkpoint.items["data_iterator"]["rng_counts"] == [3, 0]
        assert int(checkpoint.items["data_iterator"]["position"]) == 96

    @pytest.mark.parametrize(
        ("layout", "items"),
        [("diffav_model", {"model"}), ("diffav_model_optimizer", {"model", "optimizer"})],
    )
    def test_the_diffav_payloads_keep_their_sidecar(self, layout: str, items: set[str]) -> None:
        store = OrbaxCheckpointStore(FORMAT2_FIXTURES / layout)

        checkpoint = store.restore(FIXTURE_STEP, legacy_layout=DIFFAV_FORMAT2)

        assert set(checkpoint.items) == items
        assert checkpoint.metadata.epoch == 2
        assert checkpoint.metadata.metrics == {"loss": 0.25, "val_loss": 0.5}
        assert checkpoint.metadata.extra["architecture_version"] == 3

    def test_the_datarax_iterator_state_becomes_the_data_iterator_item(self) -> None:
        store = OrbaxCheckpointStore(FORMAT2_FIXTURES / "datarax_iterator")

        checkpoint = store.restore(
            FIXTURE_STEP,
            templates={"data_iterator": _iterator_template()},
            legacy_layout=DATARAX_ITERATOR_FORMAT2,
        )

        assert set(checkpoint.items) == {"data_iterator"}
        assert int(checkpoint.items["data_iterator"]["position"]) == 96
        assert checkpoint.items["data_iterator"]["epoch"] == 1
        assert checkpoint.metadata.extra["pipeline"] == "train"

    def test_the_wrong_layout_names_the_missing_key(self) -> None:
        store = OrbaxCheckpointStore(FORMAT2_FIXTURES / "substrax_module")
        with pytest.raises(KeyError, match="model"):
            store.restore(FIXTURE_STEP, legacy_layout=ARTIFEX_TRAINER_FORMAT2)

    def test_the_fixtures_stay_format_2_on_disk(self) -> None:
        for layout in LAYOUTS:
            OrbaxCheckpointStore(FORMAT2_FIXTURES / layout).read_metadata(FIXTURE_STEP)
            assert (
                raw_metadata(FORMAT2_FIXTURES / layout, FIXTURE_STEP)["checkpoint_version"] == "2.0"
            )


class TestUpgrade:
    def test_upgrade_writes_a_format_3_root_and_leaves_the_source_alone(
        self, tmp_path: Path
    ) -> None:
        source = FORMAT2_FIXTURES / "opifex_module"
        destination = tmp_path / "upgraded"

        steps = upgrade_checkpoints(source, destination)

        assert steps == [FIXTURE_STEP]
        assert raw_metadata(source, FIXTURE_STEP)["checkpoint_version"] == "2.0"
        upgraded = raw_metadata(destination, FIXTURE_STEP)
        assert upgraded["format_version"] == CURRENT_FORMAT_VERSION
        assert upgraded["items"] == ["model"]
        assert upgraded["metrics"] == {"loss": 0.25}
        assert upgraded["epoch"] == 2
        checkpoint = OrbaxCheckpointStore(destination).restore(
            FIXTURE_STEP, templates={"model": nnx.state(_model())}
        )
        assert checkpoint.items["model"]["kernel"][...].shape == (4, 2)
        assert checkpoint.metadata.extra["upgraded_from"] == {"format_version": 2}

    def test_upgrade_takes_a_producer_layout(self, tmp_path: Path) -> None:
        destination = tmp_path / "upgraded"

        upgrade_checkpoints(
            FORMAT2_FIXTURES / "artifex_trainer", destination, legacy_layout=ARTIFEX_TRAINER_FORMAT2
        )

        assert raw_metadata(destination, FIXTURE_STEP)["items"] == [
            "model",
            "optimizer",
            "rng",
            "extensions",
        ]

    def test_upgrade_carries_every_step(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        with OrbaxCheckpointStore(source, max_to_keep=None) as store:
            for step in (1, 2, 3):
                store.save(step, {"model": nnx.state(_model())}, metrics={"loss": 1.0 / step})
        destination = tmp_path / "upgraded"

        assert upgrade_checkpoints(source, destination) == [1, 2, 3]
        assert OrbaxCheckpointStore(destination).list_steps() == [1, 2, 3]
        assert OrbaxCheckpointStore(destination).best_step("loss") == 3

    def test_upgrade_never_rewrites_in_place(self, tmp_path: Path) -> None:
        source = FORMAT2_FIXTURES / "substrax_module"
        with pytest.raises(ValueError, match="same directory"):
            upgrade_checkpoints(source, source)
        occupied = tmp_path / "occupied"
        occupied.mkdir()
        (occupied / "note.txt").write_text("keep", encoding="utf-8")
        with pytest.raises(ValueError, match="not empty"):
            upgrade_checkpoints(source, occupied)
        assert (occupied / "note.txt").read_text(encoding="utf-8") == "keep"

    def test_the_command_line_upgrades_a_root(self, tmp_path: Path) -> None:
        destination = tmp_path / "upgraded"

        completed = subprocess.run(  # noqa: S603  # the interpreter running this test, fixed arguments
            [
                sys.executable,
                "-m",
                "substrax.checkpoint",
                "upgrade",
                str(FORMAT2_FIXTURES / "substrax_module"),
                str(destination),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )

        assert completed.returncode == 0, completed.stderr
        assert str(FIXTURE_STEP) in completed.stdout
        assert raw_metadata(destination, FIXTURE_STEP)["format_version"] == CURRENT_FORMAT_VERSION

    def test_the_command_line_refuses_an_unknown_layout(self, tmp_path: Path) -> None:
        completed = subprocess.run(  # noqa: S603  # the interpreter running this test, fixed arguments
            [
                sys.executable,
                "-m",
                "substrax.checkpoint",
                "upgrade",
                str(FORMAT2_FIXTURES / "substrax_module"),
                str(tmp_path / "upgraded"),
                "--layout",
                "trainer",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )
        assert completed.returncode != 0
        assert "module-only" in completed.stderr


def test_a_format_3_checkpoint_needs_no_layout(tmp_path: Path) -> None:
    """The layout argument is read only for a format-2 checkpoint."""
    store = OrbaxCheckpointStore(tmp_path / "ckpt")
    store.save(1, {"model": nnx.state(_model())})

    checkpoint = store.restore(1, legacy_layout=ARTIFEX_TRAINER_FORMAT2)

    assert set(checkpoint.items) == {"model"}
    assert MODULE_ONLY_FORMAT2.name == "module-only"
    assert jnp.asarray(checkpoint.items["model"]["kernel"]["value"]).shape == (4, 2)
