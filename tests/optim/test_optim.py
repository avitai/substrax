"""``substrax.optim``: one optimizer spec, one builder over optax, the learning rate read on device."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from substrax.optim import (
    create_optimizer,
    create_transformation,
    current_learning_rate,
    EXCLUDE_BIAS_AND_NORM_SCALE,
    OptimizerConfig,
    OptimizerType,
    weight_decay_mask,
)


class KnobModel(nnx.Module):
    """Preprocessing knobs, a norm and a probe head, the shape of a learned-preprocessing model."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        super().__init__()
        self.preprocessing = nnx.Linear(3, 3, rngs=rngs)
        self.norm = nnx.LayerNorm(3, rngs=rngs)
        self.probe = nnx.Linear(3, 2, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.probe(self.norm(self.preprocessing(x)))


@pytest.fixture
def model() -> KnobModel:
    return KnobModel(rngs=nnx.Rngs(0))


def _params(model: nnx.Module) -> dict[str, Any]:
    # Independent host snapshots remain valid after native buffer donation.
    return jax.tree.map(np.array, nnx.to_pure_dict(nnx.state(model, nnx.Param)))


def _grads_like(model: nnx.Module, fill: Callable[[jax.Array], jax.Array]) -> Any:
    return jax.tree.map(fill, nnx.state(model, nnx.Param))


def _zero_grads(model: nnx.Module) -> Any:
    return _grads_like(model, jnp.zeros_like)


def _unit_grads(model: nnx.Module) -> Any:
    return _grads_like(model, jnp.ones_like)


def _assert_params_close(model: nnx.Module, reference: nnx.Module) -> None:
    """Every parameter agrees to a few float32 ULPs; the trees have the same structure."""
    mine = jax.tree_util.tree_leaves_with_path(_params(model))
    theirs = dict(jax.tree_util.tree_leaves_with_path(_params(reference)))
    assert [path for path, _ in mine] == list(theirs)
    for path, value in mine:
        np.testing.assert_allclose(value, theirs[path], rtol=1e-5, err_msg=str(path))


def _at(schedule: optax.Schedule, step: int) -> float:
    return float(jnp.asarray(schedule(step)))


def _flat_mask(mask: object) -> dict[str, bool]:
    return {
        "/".join(str(getattr(k, "key", k)) for k in path if not hasattr(k, "name")): bool(leaf)
        for path, leaf in jax.tree_util.tree_leaves_with_path(mask)
    }


class TestConfig:
    def test_defaults_follow_optax(self) -> None:
        config = OptimizerConfig(learning_rate=1e-3)

        assert (config.optimizer_type, config.b1, config.b2, config.eps) == (
            "adam",
            0.9,
            0.999,
            1e-8,
        )
        assert (config.weight_decay, config.weight_decay_filter, config.momentum) == (
            0.0,
            None,
            None,
        )
        assert (config.gradient_clip_norm, config.gradient_clip_value) == (None, None)
        assert config.wrt is nnx.Param

    def test_both_clip_fields_raise(self) -> None:
        with pytest.raises(ValueError, match=r"gradient_clip_norm.*gradient_clip_value"):
            OptimizerConfig(learning_rate=1e-3, gradient_clip_norm=1.0, gradient_clip_value=0.5)

    @pytest.mark.parametrize(
        "optimizer_type", ["adam", "sgd", "rmsprop", "adagrad", "radam", "nadam"]
    )
    def test_weight_decay_needs_a_decoupled_optimizer(self, optimizer_type: str) -> None:
        with pytest.raises(ValueError, match=f"{optimizer_type}.*weight_decay"):
            OptimizerConfig(optimizer_type=optimizer_type, learning_rate=1e-3, weight_decay=0.1)  # pyright: ignore[reportArgumentType]

    def test_negative_weight_decay_raises(self) -> None:
        with pytest.raises(ValueError, match="weight_decay"):
            OptimizerConfig(optimizer_type="adamw", learning_rate=1e-3, weight_decay=-0.1)

    def test_momentum_is_for_sgd_and_rmsprop_only(self) -> None:
        with pytest.raises(ValueError, match="momentum"):
            OptimizerConfig(optimizer_type="adam", learning_rate=1e-3, momentum=0.9)
        assert (
            OptimizerConfig(optimizer_type="sgd", learning_rate=1e-3, momentum=0.9).momentum == 0.9
        )

    def test_a_string_filter_is_a_tag_not_a_path(self) -> None:
        with pytest.raises(TypeError, match="tag"):
            OptimizerConfig(optimizer_type="adamw", learning_rate=1e-3, weight_decay_filter="probe")

    def test_a_constant_learning_rate_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="learning_rate"):
            OptimizerConfig(learning_rate=0.0)

    def test_an_unknown_optimizer_type_is_refused_at_construction(self) -> None:
        with pytest.raises(ValueError, match=r"lion.*adam"):
            OptimizerConfig(optimizer_type="lion", learning_rate=1e-3)  # pyright: ignore[reportArgumentType]

    @pytest.mark.parametrize("value", [0.0, -1.0])
    def test_a_clip_norm_must_be_positive(self, value: float) -> None:
        with pytest.raises(ValueError, match="gradient_clip_norm"):
            OptimizerConfig(learning_rate=1e-3, gradient_clip_norm=value)

    def test_a_clip_value_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="gradient_clip_value"):
            OptimizerConfig(learning_rate=1e-3, gradient_clip_value=0.0)


class TestWeightDecayMask:
    def test_the_default_decays_every_leaf_in_wrt(self, model: KnobModel) -> None:
        mask = weight_decay_mask(model, OptimizerConfig(optimizer_type="adamw", learning_rate=1e-3))

        assert set(_flat_mask(mask).values()) == {True}
        assert len(_flat_mask(mask)) == 6

    def test_a_path_filter_selects_its_leaves(self, model: KnobModel) -> None:
        config = OptimizerConfig(
            optimizer_type="adamw",
            learning_rate=1e-3,
            weight_decay_filter=nnx.PathContains("probe"),
        )

        flat = _flat_mask(weight_decay_mask(model, config))

        assert {k for k, v in flat.items() if v} == {"probe/bias", "probe/kernel"}

    def test_the_named_filter_excludes_biases_and_norm_scales(self, model: KnobModel) -> None:
        config = OptimizerConfig(
            optimizer_type="adamw",
            learning_rate=1e-3,
            weight_decay_filter=EXCLUDE_BIAS_AND_NORM_SCALE,
        )

        flat = _flat_mask(weight_decay_mask(model, config))

        assert {k for k, v in flat.items() if v} == {"preprocessing/kernel", "probe/kernel"}

    def test_a_filter_selecting_nothing_raises_and_names_the_filter(self, model: KnobModel) -> None:
        config = OptimizerConfig(
            optimizer_type="adamw",
            learning_rate=1e-3,
            weight_decay_filter=nnx.PathContains("absent"),
        )

        with pytest.raises(ValueError, match="absent"):
            weight_decay_mask(model, config)
        with pytest.raises(ValueError, match="absent"):
            create_transformation(model, config)

    def test_the_mask_is_static_python_booleans(self, model: KnobModel) -> None:
        mask = weight_decay_mask(model, OptimizerConfig(optimizer_type="adamw", learning_rate=1e-3))

        assert all(type(leaf) is bool for leaf in jax.tree.leaves(mask))


class TestTransformation:
    def test_matches_optax_chain_of_clip_and_adam_over_five_steps(self, model: KnobModel) -> None:
        config = OptimizerConfig(optimizer_type="adam", learning_rate=1e-3, gradient_clip_norm=1.0)
        reference = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))
        reference_model = KnobModel(rngs=nnx.Rngs(0))
        ours = nnx.Optimizer(model, create_transformation(model, config), wrt=nnx.Param)
        theirs = nnx.Optimizer(reference_model, reference, wrt=nnx.Param)

        for sub in jax.random.split(jax.random.key(1), 5):
            grads = jax.tree.map(
                lambda p, k=sub: jax.random.normal(k, p.shape) * 3.0, nnx.state(model, nnx.Param)
            )
            ours.update(model, grads)
            theirs.update(reference_model, grads)

        # inject_hyperparams keeps b1, b2 and eps as float32 arrays where optax.adam keeps
        # Python floats; measured over the five steps the parameters differ by at most
        # 2.3e-6 relative (a few float32 ULPs at |p| ~ 0.1), never in structure.
        _assert_params_close(model, reference_model)

    def test_adams_step_follows_the_schedule_and_a_prescaled_chain_does_not(
        self, model: KnobModel
    ) -> None:
        schedule = optax.piecewise_constant_schedule(1e-3, {1: 2.0, 2: 2.0})  # 1e-3, 2e-3, 4e-3
        ours = create_optimizer(
            model, OptimizerConfig(optimizer_type="adam", learning_rate=schedule)
        )
        control_model = KnobModel(rngs=nnx.Rngs(0))
        control = nnx.Optimizer(
            control_model,
            optax.chain(optax.scale_by_schedule(schedule), optax.adam(1.0)),
            wrt=nnx.Param,
        )

        ours_steps, control_steps = [], []
        for _step in range(3):
            before = _params(model)["probe"]["kernel"]
            ours.update(model, _unit_grads(model))
            ours_steps.append(float(np.abs(_params(model)["probe"]["kernel"] - before).mean()))
            before = _params(control_model)["probe"]["kernel"]
            control.update(control_model, _unit_grads(control_model))
            control_steps.append(
                float(np.abs(_params(control_model)["probe"]["kernel"] - before).mean())
            )

        np.testing.assert_allclose(ours_steps, [1e-3, 2e-3, 4e-3], rtol=1e-3)
        assert not np.allclose(control_steps, [1e-3, 2e-3, 4e-3], rtol=1e-1)

    def test_nadam_is_adam_with_nesterov(self, model: KnobModel) -> None:
        ours = nnx.Optimizer(
            model,
            create_transformation(
                model, OptimizerConfig(optimizer_type="nadam", learning_rate=1e-2)
            ),
            wrt=nnx.Param,
        )
        reference_model = KnobModel(rngs=nnx.Rngs(0))
        reference = nnx.Optimizer(reference_model, optax.adam(1e-2, nesterov=True), wrt=nnx.Param)

        for _ in range(3):
            ours.update(model, _unit_grads(model))
            reference.update(reference_model, _unit_grads(reference_model))

        _assert_params_close(model, reference_model)


class TestWeightDecay:
    LR = 0.01
    DECAY = 0.1

    def _decayed(
        self, model: KnobModel, weight_decay_filter: nnx.filterlib.Filter | None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        before = _params(model)
        config = OptimizerConfig(
            optimizer_type="adamw",
            learning_rate=self.LR,
            weight_decay=self.DECAY,
            weight_decay_filter=weight_decay_filter,
        )
        create_optimizer(model, config).update(model, _zero_grads(model))
        return before, _params(model)

    def test_selected_leaves_decay_by_exactly_lr_times_decay_and_the_rest_stay(
        self, model: KnobModel
    ) -> None:
        before, after = self._decayed(model, nnx.PathContains("probe"))

        for name in ("kernel", "bias"):
            np.testing.assert_allclose(
                after["probe"][name],
                before["probe"][name] * (1.0 - self.LR * self.DECAY),
                rtol=1e-6,
            )
            np.testing.assert_array_equal(
                after["preprocessing"][name], before["preprocessing"][name]
            )
        np.testing.assert_array_equal(after["norm"]["scale"], before["norm"]["scale"])

    def test_the_inverted_filter_decays_the_others_instead(self, model: KnobModel) -> None:
        before, after = self._decayed(model, nnx.Not(nnx.PathContains("probe")))

        np.testing.assert_array_equal(after["probe"]["kernel"], before["probe"]["kernel"])
        np.testing.assert_allclose(
            after["preprocessing"]["kernel"],
            before["preprocessing"]["kernel"] * (1.0 - self.LR * self.DECAY),
            rtol=1e-6,
        )

    def test_a_probe_only_filter_leaves_every_preprocessing_parameter_unchanged(
        self, model: KnobModel
    ) -> None:
        before, after = self._decayed(model, nnx.PathContains("probe"))

        np.testing.assert_array_equal(
            after["preprocessing"]["kernel"], before["preprocessing"]["kernel"]
        )
        np.testing.assert_array_equal(
            after["preprocessing"]["bias"], before["preprocessing"]["bias"]
        )

    def test_an_ndim_mask_is_the_failing_control(self, model: KnobModel) -> None:
        """A mask decaying every 2-D leaf, the timm convention, decays the preprocessing kernel."""
        before, after = self._decayed(model, EXCLUDE_BIAS_AND_NORM_SCALE)

        assert not np.array_equal(
            after["preprocessing"]["kernel"], before["preprocessing"]["kernel"]
        )

    def test_the_optimizer_state_has_the_same_structure_for_every_filter(self) -> None:
        structures = set()
        for weight_decay_filter in (None, nnx.PathContains("probe"), EXCLUDE_BIAS_AND_NORM_SCALE):
            config = OptimizerConfig(
                optimizer_type="adamw",
                learning_rate=self.LR,
                weight_decay=self.DECAY,
                weight_decay_filter=weight_decay_filter,
            )
            optimizer = create_optimizer(KnobModel(rngs=nnx.Rngs(0)), config)
            structures.add(str(jax.tree_util.tree_structure(optimizer.opt_state)))

        assert len(structures) == 1

    def test_decay_under_clipping_and_a_schedule_follows_the_schedule(
        self, model: KnobModel
    ) -> None:
        schedule = optax.piecewise_constant_schedule(self.LR, {1: 2.0})  # 0.01 then 0.02
        config = OptimizerConfig(
            optimizer_type="adamw",
            learning_rate=schedule,
            weight_decay=self.DECAY,
            gradient_clip_norm=1.0,
        )
        optimizer = create_optimizer(model, config)

        for step in range(2):
            before = _params(model)["probe"]["kernel"]
            optimizer.update(model, _zero_grads(model))
            expected = before * (1.0 - _at(schedule, step) * self.DECAY)
            np.testing.assert_allclose(_params(model)["probe"]["kernel"], expected, rtol=1e-6)


class TestCurrentLearningRate:
    def test_reads_the_rate_the_last_update_applied_on_device(self, model: KnobModel) -> None:
        schedule = optax.warmup_cosine_decay_schedule(0.0, 1e-3, warmup_steps=4, decay_steps=20)
        optimizer = create_optimizer(model, OptimizerConfig(learning_rate=schedule))

        assert isinstance(current_learning_rate(optimizer), jax.Array)
        assert float(current_learning_rate(optimizer)) == pytest.approx(_at(schedule, 0))
        for _ in range(3):
            optimizer.update(model, _unit_grads(model))
        assert float(current_learning_rate(optimizer)) == pytest.approx(_at(schedule, 2))

    def test_does_not_advance_on_a_step_apply_if_finite_skips(self, model: KnobModel) -> None:
        schedule = optax.warmup_cosine_decay_schedule(0.0, 1e-3, warmup_steps=4, decay_steps=20)
        transformation = optax.apply_if_finite(
            create_transformation(model, OptimizerConfig(learning_rate=schedule)),
            max_consecutive_errors=3,
        )
        optimizer = nnx.Optimizer(model, transformation, wrt=nnx.Param)
        optimizer.update(model, _unit_grads(model))
        optimizer.update(model, _unit_grads(model))
        assert float(current_learning_rate(optimizer)) == pytest.approx(_at(schedule, 1))

        optimizer.update(model, _grads_like(model, lambda p: jnp.full_like(p, jnp.nan)))

        assert float(current_learning_rate(optimizer)) == pytest.approx(_at(schedule, 1))

    @pytest.mark.parametrize("optimizer_type", ["adamw", "sgd"])
    def test_reads_the_same_rate_under_nnx_jit(
        self, model: KnobModel, optimizer_type: OptimizerType
    ) -> None:
        schedule = optax.linear_schedule(1e-3, 2e-3, transition_steps=4)
        optimizer = create_optimizer(
            model, OptimizerConfig(optimizer_type=optimizer_type, learning_rate=schedule)
        )
        optimizer.update(model, _unit_grads(model))

        @nnx.jit
        def read(optimizer: nnx.Optimizer[KnobModel]) -> jax.Array:
            return current_learning_rate(optimizer)

        assert float(read(optimizer)) == pytest.approx(_at(schedule, 0))
        assert float(read(optimizer)) == float(current_learning_rate(optimizer))

    def test_an_optimizer_without_an_injected_rate_is_refused(self, model: KnobModel) -> None:
        optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

        with pytest.raises(ValueError, match="no injected learning rate"):
            current_learning_rate(optimizer)

    def test_create_optimizer_uses_the_configured_wrt(self, model: KnobModel) -> None:
        optimizer = create_optimizer(model, OptimizerConfig(learning_rate=1e-3))

        assert optimizer.wrt is nnx.Param


class TestFiniteSkipping:
    @pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
    def test_rejected_gradients_hold_parameters_and_allow_recovery(
        self, model: KnobModel, invalid: float
    ) -> None:
        optimizer = create_optimizer(model, OptimizerConfig(learning_rate=1e-3))
        reference_model = KnobModel(rngs=nnx.Rngs(0))
        reference = nnx.Optimizer(reference_model, optax.adam(1e-3), wrt=nnx.Param)
        optimizer.update(model, _unit_grads(model))
        reference.update(reference_model, _unit_grads(reference_model))
        before = _params(model)

        for _ in range(3):
            optimizer.update(model, _grads_like(model, lambda p: jnp.full_like(p, invalid)))
            for actual, expected in zip(
                jax.tree.leaves(_params(model)), jax.tree.leaves(before), strict=True
            ):
                np.testing.assert_array_equal(actual, expected)

        optimizer.update(model, _unit_grads(model))
        reference.update(reference_model, _unit_grads(reference_model))
        _assert_params_close(model, reference_model)
        assert int(optimizer.step[...]) == 5

    def test_rejection_counter_saturates_without_enabling_give_up(self, model: KnobModel) -> None:
        optimizer = create_optimizer(model, OptimizerConfig(learning_rate=1e-3))
        maximum = jnp.iinfo(jnp.int32).max
        # Seed the native public state near the boundary instead of taking billions
        # of steps. Optax's finite-limit control below must actually give up.
        optimizer.opt_state.notfinite_count[...] = jnp.asarray(maximum - 1, jnp.int32)
        before = _params(model)
        invalid = _grads_like(model, lambda p: jnp.full_like(p, jnp.nan))
        for _ in range(3):
            optimizer.update(model, invalid)
            assert int(optimizer.opt_state.notfinite_count[...]) == maximum
            assert not bool(optimizer.opt_state.last_finite[...])
            for actual, expected in zip(
                jax.tree.leaves(_params(model)), jax.tree.leaves(before), strict=True
            ):
                np.testing.assert_array_equal(actual, expected)
        optimizer.update(model, _unit_grads(model))
        assert int(optimizer.opt_state.notfinite_count[...]) == 0
        assert bool(optimizer.opt_state.last_finite[...])

        control = optax.apply_if_finite(optax.sgd(0.1), max_consecutive_errors=1)
        state = control.init(jnp.asarray(1.0))
        first, state = control.update(jnp.asarray(jnp.nan), state)
        second, _ = control.update(jnp.asarray(jnp.nan), state)
        assert isinstance(first, jax.Array)
        assert isinstance(second, jax.Array)
        assert float(first) == 0.0
        assert bool(jnp.isnan(second))

    def test_skips_hold_the_inner_schedule_and_state_under_configured_clipping(
        self, model: KnobModel
    ) -> None:
        schedule = optax.piecewise_constant_schedule(0.01, {1: 2.0})
        optimizer = create_optimizer(
            model, OptimizerConfig(learning_rate=schedule, gradient_clip_norm=1.0)
        )
        optimizer.update(model, _unit_grads(model))
        before = jax.tree.map(np.asarray, nnx.state(optimizer.opt_state.inner_state))
        for _ in range(3):
            optimizer.update(model, _grads_like(model, lambda p: jnp.full_like(p, jnp.nan)))
            assert float(current_learning_rate(optimizer)) == pytest.approx(0.01)
            for actual, expected in zip(
                jax.tree.leaves(nnx.state(optimizer.opt_state.inner_state)),
                jax.tree.leaves(before),
                strict=True,
            ):
                np.testing.assert_array_equal(actual, expected)
        optimizer.update(model, _unit_grads(model))
        assert float(current_learning_rate(optimizer)) == pytest.approx(0.02)
        assert int(optimizer.step[...]) == 5

    @pytest.mark.parametrize("clip", [None, 1.0])
    def test_large_finite_gradients_need_explicit_clipping_to_protect_moments(
        self, model: KnobModel, clip: float | None
    ) -> None:
        optimizer = create_optimizer(
            model, OptimizerConfig(learning_rate=1e-3, gradient_clip_norm=clip)
        )
        optimizer.update(model, _grads_like(model, lambda p: jnp.full_like(p, 1e30)))
        assert bool(optimizer.opt_state.last_finite[...])
        inner_finite = all(
            bool(jnp.all(jnp.isfinite(leaf)))
            for leaf in jax.tree.leaves(nnx.state(optimizer.opt_state.inner_state))
        )
        assert inner_finite == (clip is not None)

    @pytest.mark.parametrize("graph", [False, True])
    def test_donated_jit_skips_and_recovers_without_retracing(
        self, model: KnobModel, graph: bool
    ) -> None:
        optimizer = create_optimizer(
            model, OptimizerConfig(optimizer_type="sgd", learning_rate=0.1)
        )
        before = _params(model)
        traces: list[int] = []

        @nnx.jit(graph=graph, graph_updates=False, donate_argnums=0)
        def run(bundle: tuple[KnobModel, nnx.Optimizer[KnobModel]], value: jax.Array) -> None:
            traces.append(1)
            current, owned = bundle
            owned.update(current, _grads_like(current, lambda p: jnp.full_like(p, value)))

        for value in (1.0, float("nan"), float("inf"), 2.0):
            run((model, optimizer), jnp.asarray(value, jnp.float32))
        assert traces == [1]
        assert int(optimizer.step[...]) == 4
        for actual, initial in zip(
            jax.tree.leaves(_params(model)), jax.tree.leaves(before), strict=True
        ):
            np.testing.assert_allclose(actual, initial - 0.3, rtol=1e-6, atol=1e-7)

    def test_scan_accumulates_attempts_but_not_rejected_updates(self, model: KnobModel) -> None:
        optimizer = create_optimizer(
            model, OptimizerConfig(optimizer_type="sgd", learning_rate=0.1)
        )
        before = _params(model)

        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
        def run(
            bundle: tuple[KnobModel, nnx.Optimizer[KnobModel]], value: jax.Array
        ) -> tuple[tuple[KnobModel, nnx.Optimizer[KnobModel]], jax.Array]:
            current, owned = bundle
            owned.update(current, _grads_like(current, lambda p: jnp.full_like(p, value)))
            return bundle, owned.opt_state.last_finite[...]

        _, accepted = run((model, optimizer), jnp.asarray([1.0, jnp.nan, 2.0]))
        np.testing.assert_array_equal(accepted, [True, False, True])
        assert int(optimizer.step[...]) == 3
        for actual, initial in zip(
            jax.tree.leaves(_params(model)), jax.tree.leaves(before), strict=True
        ):
            np.testing.assert_allclose(actual, initial - 0.3, rtol=1e-6, atol=1e-7)

    def test_vmap_keeps_each_members_rejection_and_recovery_independent(self) -> None:
        @nnx.vmap(in_axes=0, out_axes=0)
        def create(seed: jax.Array) -> tuple[KnobModel, nnx.Optimizer[KnobModel]]:
            member = KnobModel(rngs=nnx.Rngs(seed))
            owned = create_optimizer(
                member, OptimizerConfig(optimizer_type="sgd", learning_rate=0.1)
            )
            return member, owned

        models, optimizers = create(jax.random.split(jax.random.key(0), 3))
        before = _params(models)

        @nnx.vmap(in_axes=(0, 0, 0), out_axes=0)
        def update(
            member: KnobModel, owned: nnx.Optimizer[KnobModel], value: jax.Array
        ) -> jax.Array:
            owned.update(member, _grads_like(member, lambda p: jnp.full_like(p, value)))
            return owned.opt_state.last_finite[...]

        first = update(models, optimizers, jnp.asarray([1.0, jnp.nan, jnp.inf]))
        second = update(models, optimizers, jnp.asarray([2.0, 3.0, 4.0]))
        np.testing.assert_array_equal(first, [True, False, False])
        np.testing.assert_array_equal(second, [True, True, True])
        for actual, initial in zip(
            jax.tree.leaves(_params(models)), jax.tree.leaves(before), strict=True
        ):
            offsets = np.asarray([0.3, 0.3, 0.4]).reshape((3,) + (1,) * (actual.ndim - 1))
            np.testing.assert_allclose(actual, initial - offsets, rtol=1e-6, atol=1e-7)

    @pytest.mark.parametrize(("gradient", "derivative"), [(1.0, 5.88), (float("nan"), 6.0)])
    def test_differentiating_a_guarded_update_preserves_the_parameter_derivative(
        self, gradient: float, derivative: float
    ) -> None:
        def after(initial: jax.Array) -> jax.Array:
            model = KnobModel(rngs=nnx.Rngs(0))
            model.probe.kernel[...] = jnp.full_like(model.probe.kernel[...], initial)
            optimizer = create_optimizer(
                model,
                OptimizerConfig(optimizer_type="adamw", learning_rate=0.1, weight_decay=0.2),
            )
            optimizer.update(model, _grads_like(model, lambda p: jnp.full_like(p, gradient)))
            return jnp.sum(model.probe.kernel[...])

        # Six kernel entries: accepted decoupled decay differentiates to
        # 6 * (1 - 0.1 * 0.2); rejection leaves the identity derivative, 6.
        actual = jax.jit(jax.grad(after))(jnp.asarray(1.0))
        assert float(actual) == pytest.approx(derivative, rel=1e-6)
