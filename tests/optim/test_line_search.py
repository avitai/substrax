"""Line-search optimizers on an NNX model, and switching from one optax optimizer to another."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from substrax.optim import switch_at, update_with_line_search, with_strong_state_types
from substrax.testing import TraceCounter


TARGET = jnp.asarray([1.0, -2.0, 3.0, 0.5])


class Quartic(nnx.Module):
    """A convex quartic whose loss also reads a non-parameter variable."""

    def __init__(self, *, rngs: nnx.Rngs, param_dtype: jnp.dtype = jnp.float32) -> None:
        super().__init__()
        self.w = nnx.Param(jax.random.normal(rngs.params(), (4,), param_dtype))
        self.scale = nnx.BatchStat(jnp.asarray(2.0, param_dtype))

    def loss(self) -> jax.Array:
        w = self.w[...]
        return self.scale[...] * jnp.sum((w - TARGET) ** 2) + 0.1 * jnp.sum(w**4)


def _loss(model: Quartic) -> jax.Array:
    return model.loss()


def _shifted_loss(model: Quartic) -> jax.Array:
    return model.loss() + 7.0


def _lbfgs() -> optax.GradientTransformationExtraArgs:
    return with_strong_state_types(optax.lbfgs())


def _reference_lbfgs(model: Quartic, steps: int) -> jax.Array:
    """optax's own functional L-BFGS loop on the parameter array, the loss written out."""
    scale = model.scale[...]

    def loss(w: jax.Array) -> jax.Array:
        return scale * jnp.sum((w - TARGET) ** 2) + 0.1 * jnp.sum(w**4)

    solver = optax.lbfgs()
    w = model.w[...]
    state = solver.init(w)
    value_and_grad = optax.value_and_grad_from_state(loss)
    for _ in range(steps):
        value, grad = value_and_grad(w, state=state)
        updates, state = solver.update(grad, state, w, value=value, grad=grad, value_fn=loss)
        updated = optax.apply_updates(w, updates)
        assert isinstance(updated, jax.Array)
        w = updated
    return w


@nnx.jit
def _step(
    model: Quartic, optimizer: nnx.Optimizer[Quartic], objective_changed: jax.Array
) -> jax.Array:
    return update_with_line_search(model, optimizer, _loss, objective_changed=objective_changed)


FIXED = jnp.asarray(False)
CHANGED = jnp.asarray(True)


def _evaluations_over(steps: int, *, objective_changed: bool, ensemble: int = 0) -> int:
    """How many times the loss runs over ``steps`` steps, for one model or a vmapped ensemble."""
    evaluations = {"count": 0}

    def counted_loss(model: Quartic) -> jax.Array:
        jax.debug.callback(lambda: evaluations.__setitem__("count", evaluations["count"] + 1))
        return model.loss()

    def update(m: Quartic, o: nnx.Optimizer[Quartic], changed: jax.Array) -> jax.Array:
        return update_with_line_search(m, o, counted_loss, objective_changed=changed)

    if ensemble:

        @nnx.vmap
        def create(seed: jax.Array) -> tuple[Quartic, nnx.Optimizer[Quartic]]:
            model = Quartic(rngs=nnx.Rngs(jax.random.key(seed)))
            return model, nnx.Optimizer(model, _lbfgs(), wrt=nnx.Param)

        model, optimizer = create(jnp.arange(ensemble))
        step = nnx.jit(nnx.vmap(update, in_axes=(0, 0, None)))
    else:
        model = Quartic(rngs=nnx.Rngs(0))
        optimizer = nnx.Optimizer(model, _lbfgs(), wrt=nnx.Param)
        step = nnx.jit(update)
    for _ in range(steps):
        step(model, optimizer, jnp.asarray(objective_changed))
    jax.effects_barrier()
    return evaluations["count"]


class TestUpdateWithLineSearch:
    """One step of an optimizer that searches along its direction with the loss of the parameters."""

    def test_lbfgs_steps_match_optax_on_the_parameter_array(self) -> None:
        model = Quartic(rngs=nnx.Rngs(0))
        reference = _reference_lbfgs(Quartic(rngs=nnx.Rngs(0)), steps=6)
        optimizer = nnx.Optimizer(model, _lbfgs(), wrt=nnx.Param)

        for _ in range(6):
            _step(model, optimizer, FIXED)

        np.testing.assert_allclose(model.w[...], reference, rtol=1e-6)

    def test_recomputing_every_step_follows_the_same_path(self) -> None:
        reused, recomputed = Quartic(rngs=nnx.Rngs(0)), Quartic(rngs=nnx.Rngs(0))
        reused_opt = nnx.Optimizer(reused, _lbfgs(), wrt=nnx.Param)
        recomputed_opt = nnx.Optimizer(recomputed, _lbfgs(), wrt=nnx.Param)

        for _ in range(6):
            _step(reused, reused_opt, FIXED)
            _step(recomputed, recomputed_opt, CHANGED)

        np.testing.assert_allclose(reused.w[...], recomputed.w[...], rtol=1e-6)

    def test_the_line_search_reads_the_model_state_outside_wrt(self) -> None:
        model = Quartic(rngs=nnx.Rngs(0))
        model.scale[...] = jnp.asarray(5.0)
        reference = _reference_lbfgs(model, steps=4)
        optimizer = nnx.Optimizer(model, _lbfgs(), wrt=nnx.Param)

        for _ in range(4):
            _step(model, optimizer, FIXED)

        np.testing.assert_allclose(model.w[...], reference, rtol=1e-6)
        assert float(model.scale[...]) == 5.0

    def test_it_returns_the_loss_before_the_update(self) -> None:
        model = Quartic(rngs=nnx.Rngs(0))
        before = float(_loss(model))
        optimizer = nnx.Optimizer(model, _lbfgs(), wrt=nnx.Param)

        returned = _step(model, optimizer, FIXED)

        assert float(returned) == pytest.approx(before)
        assert float(_loss(model)) < before

    def test_a_changed_objective_is_evaluated_afresh(self) -> None:
        model = Quartic(rngs=nnx.Rngs(0))
        optimizer = nnx.Optimizer(model, _lbfgs(), wrt=nnx.Param)
        for _ in range(3):
            _step(model, optimizer, FIXED)
        expected = float(_shifted_loss(model))

        returned = nnx.jit(
            lambda m, o: update_with_line_search(m, o, _shifted_loss, objective_changed=True)
        )(model, optimizer)

        assert float(returned) == pytest.approx(expected)

    def test_an_unchanged_objective_evaluates_the_loss_once_fewer_per_step(self) -> None:
        steps = 5
        # The first step has no stored values, so only the later ones are saved.
        assert (
            _evaluations_over(steps, objective_changed=True)
            - _evaluations_over(steps, objective_changed=False)
            == steps - 1
        )

    def test_under_vmap_reuse_saves_no_evaluation(self) -> None:
        # Each member's stored value makes the reuse predicate batched, and a batched lax.cond
        # runs both branches (jax/_src/lax/control_flow/conditionals.py, cond's docstring), so an
        # ensemble evaluates as it would with objective_changed true. Documented in
        # update_with_line_search; if this starts failing, the saving reached vmap: update both.
        assert _evaluations_over(5, objective_changed=True, ensemble=3) == _evaluations_over(
            5, objective_changed=False, ensemble=3
        )

    def test_it_traces_once_under_nnx_jit_whatever_the_objective_flag(self) -> None:
        counter = TraceCounter()
        step = nnx.jit(
            counter.wrap(
                lambda m, o, changed: update_with_line_search(
                    m, o, _loss, objective_changed=changed
                )
            )
        )
        model = Quartic(rngs=nnx.Rngs(0))
        optimizer = nnx.Optimizer(model, _lbfgs(), wrt=nnx.Param)

        with counter.expect(new_traces=1):
            step(model, optimizer, FIXED)
        with counter.expect(new_traces=0):
            for changed in (CHANGED, FIXED, CHANGED):
                step(model, optimizer, changed)

    def test_steps_run_inside_one_compiled_loop(self) -> None:
        looped, stepped = Quartic(rngs=nnx.Rngs(0)), Quartic(rngs=nnx.Rngs(0))
        looped_opt = nnx.Optimizer(looped, _lbfgs(), wrt=nnx.Param)
        stepped_opt = nnx.Optimizer(stepped, _lbfgs(), wrt=nnx.Param)

        def body(
            _: int, carry: tuple[Quartic, nnx.Optimizer[Quartic]]
        ) -> tuple[Quartic, nnx.Optimizer[Quartic]]:
            model, optimizer = carry
            update_with_line_search(model, optimizer, _loss, objective_changed=False)
            return model, optimizer

        nnx.jit(lambda m, o: nnx.fori_loop(0, 5, body, (m, o)))(looped, looped_opt)
        for _ in range(5):
            _step(stepped, stepped_opt, FIXED)

        np.testing.assert_allclose(looped.w[...], stepped.w[...], rtol=1e-6)

    def test_an_ensemble_steps_under_nnx_vmap(self) -> None:
        @nnx.vmap
        def create(seed: jax.Array) -> tuple[Quartic, nnx.Optimizer[Quartic]]:
            model = Quartic(rngs=nnx.Rngs(jax.random.key(seed)))
            return model, nnx.Optimizer(model, _lbfgs(), wrt=nnx.Param)

        seeds = jnp.arange(3)
        models, optimizers = create(seeds)
        step = nnx.jit(
            nnx.vmap(lambda m, o: update_with_line_search(m, o, _loss, objective_changed=False))
        )
        for _ in range(4):
            step(models, optimizers)

        for index, seed in enumerate(seeds):
            single = Quartic(rngs=nnx.Rngs(int(seed)))
            single_opt = nnx.Optimizer(single, _lbfgs(), wrt=nnx.Param)
            for _ in range(4):
                _step(single, single_opt, FIXED)
            np.testing.assert_allclose(models.w[...][index], single.w[...], rtol=1e-5)

    def test_a_weakly_typed_optimizer_state_is_refused(self) -> None:
        # optax.lbfgs() initialises three line-search counters weakly typed, so a jitted step
        # would compile again once they turn strong; the adapter is the remedy the error names.
        model = Quartic(rngs=nnx.Rngs(0))
        optimizer = nnx.Optimizer(model, optax.lbfgs(), wrt=nnx.Param)

        with pytest.raises(ValueError, match="with_strong_state_types"):
            update_with_line_search(model, optimizer, _loss, objective_changed=False)

    def test_a_wider_loss_is_searched_in_the_parameter_dtype(self) -> None:
        # Under x64 a float32 model can compute a float64 loss; optax's line search keeps its
        # values in the parameters' dtype and refuses a cond whose branches disagree.
        with jax.enable_x64(True):
            model = Quartic(rngs=nnx.Rngs(0))

            def wide_loss(m: Quartic) -> jax.Array:
                return m.loss().astype(jnp.float64)

            optimizer = nnx.Optimizer(model, _lbfgs(), wrt=nnx.Param)
            returned = nnx.jit(
                lambda m, o: update_with_line_search(m, o, wide_loss, objective_changed=False)
            )(model, optimizer)

            assert returned.dtype == jnp.float32
            assert model.w[...].dtype == jnp.float32

    def test_parameters_of_several_floating_dtypes_are_refused(self) -> None:
        class Mixed(nnx.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = nnx.Param(jnp.ones((2,), jnp.float32))
                self.b = nnx.Param(jnp.ones((2,), jnp.bfloat16))

        model = Mixed()
        optimizer = nnx.Optimizer(model, _lbfgs(), wrt=nnx.Param)

        with pytest.raises(ValueError, match=r"bfloat16.*float32|float32.*bfloat16"):
            update_with_line_search(
                model,
                optimizer,
                lambda m: jnp.sum(m.a[...]) + jnp.sum(m.b[...].astype(jnp.float32)),
                objective_changed=True,
            )


class TestSwitchAt:
    """The first transformation for a fixed number of updates, the second after them."""

    def test_the_first_runs_before_the_switch_and_the_second_after(self) -> None:
        params = jnp.asarray([1.0, 2.0])
        grads = jnp.asarray([0.5, -0.25])
        first, second = optax.sgd(0.1), optax.sgd(1.0)
        hybrid = switch_at(first, second, step=2)
        state = hybrid.init(params)

        updates = []
        for _ in range(4):
            update, state = hybrid.update(grads, state, params)
            updates.append(update)

        np.testing.assert_allclose(updates[0], -0.1 * grads)
        np.testing.assert_allclose(updates[1], -0.1 * grads)
        np.testing.assert_allclose(updates[2], -1.0 * grads)
        np.testing.assert_allclose(updates[3], -1.0 * grads)

    def test_adam_then_lbfgs_matches_lbfgs_from_the_switch(self) -> None:
        switch = 3
        model = Quartic(rngs=nnx.Rngs(0))
        optimizer = nnx.Optimizer(
            model, switch_at(optax.adam(1e-2), optax.lbfgs(), step=switch), wrt=nnx.Param
        )
        for _ in range(switch):
            _step(model, optimizer, FIXED)
        at_switch = Quartic(rngs=nnx.Rngs(0))
        nnx.update(at_switch, nnx.state(model))
        reference = _reference_lbfgs(at_switch, steps=4)

        for _ in range(4):
            _step(model, optimizer, FIXED)

        np.testing.assert_allclose(model.w[...], reference, rtol=1e-6)

    def test_the_switch_is_traced_not_retraced(self) -> None:
        counter = TraceCounter()
        step = nnx.jit(
            counter.wrap(lambda m, o: update_with_line_search(m, o, _loss, objective_changed=False))
        )
        model = Quartic(rngs=nnx.Rngs(0))
        optimizer = nnx.Optimizer(
            model, switch_at(optax.adam(1e-2), optax.lbfgs(), step=2), wrt=nnx.Param
        )

        with counter.expect(new_traces=1):
            for _ in range(5):
                step(model, optimizer)

    @pytest.mark.parametrize("step", [0, -1])
    def test_the_switch_step_must_be_positive(self, step: int) -> None:
        with pytest.raises(ValueError, match="positive"):
            switch_at(optax.adam(1e-3), optax.lbfgs(), step=step)
