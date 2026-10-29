# Optim

`substrax.optim` turns one specification into an optax transformation or an NNX optimizer,
and reads back the learning rate the last update applied.

```python
import optax
from flax import nnx

from substrax.optim import (
    EXCLUDE_BIAS_AND_NORM_SCALE,
    OptimizerConfig,
    create_optimizer,
    current_learning_rate,
)

config = OptimizerConfig(
    optimizer_type="adamw",
    learning_rate=optax.warmup_cosine_decay_schedule(0.0, 3e-4, warmup_steps=500, decay_steps=10_000),
    weight_decay=0.05,
    weight_decay_filter=EXCLUDE_BIAS_AND_NORM_SCALE,
    gradient_clip_norm=1.0,
)
optimizer = create_optimizer(model, config)

grads = nnx.grad(loss)(model, batch)
optimizer.update(model, grads)
print(float(current_learning_rate(optimizer)))
```

| Name | Behaviour |
| --- | --- |
| `OptimizerConfig` | Frozen, keyword-only; `optimizer_type` is one of `adam`, `adamw`, `sgd`, `rmsprop`, `adagrad`, `lamb`, `radam`, `nadam` (`adam` with Nesterov momentum), `learning_rate` a positive constant or an `optax.Schedule`, `b1`, `b2`, `eps`, `momentum` (`sgd` and `rmsprop`), `weight_decay` (`adamw` and `lamb`), `weight_decay_filter`, one of `gradient_clip_norm` and `gradient_clip_value` (positive), and `wrt`; anything else is refused when the configuration is constructed |
| `weight_decay_filter` | A Flax NNX filter naming the parameters that are decayed, the polarity of optax's `mask`; `None` decays every leaf in `wrt`, as `torch.optim.AdamW(model.parameters())` does; `EXCLUDE_BIAS_AND_NORM_SCALE` skips biases and normalisation scales, the Hugging Face and timm convention; a string raises, because NNX reads it as a tag, not a path |
| `weight_decay_mask(model, config)` | The filter as a static tree of Python booleans over `nnx.state(model, config.wrt)`; a filter that selects nothing raises instead of becoming optax's silent no-op |
| `create_transformation(model, config)` | Clipping first, then the base alias wrapped in `optax.inject_hyperparams`; a schedule is the base optimizer's learning rate, never a `scale_by_schedule` placed before an adaptive optimizer, whose normalisation would cancel it |
| `create_optimizer(model, config)` | NNX optimizer over the configured transformation, guarded by native `optax.apply_if_finite`; nonfinite gradients are skipped without giving up, using `config.wrt` |
| `current_learning_rate(optimizer)` | The rate the last update applied, a scalar array on device; the schedule at step zero before the first update, and unchanged by a step `optax.apply_if_finite` skips |

The optimizer-state tree has the same structure whatever the filter, so a checkpoint written
with one filter restores into an optimizer built with another.

## Finite-gradient skipping

`create_optimizer` wraps the configured clip-plus-optimizer transformation with
native [Optax `apply_if_finite`](https://optax.readthedocs.io/en/latest/api/optimizer_wrappers.html#optax.apply_if_finite).
The inner optimizer state and its schedule count hold on a rejected gradient;
NNX's attempt count still advances. A later finite gradient resumes the inner
optimizer normally. Optax's public state records `last_finite` for deferred
failure observation by a training loop.

The wrapper uses the maximum of its native int32 counter as the rejection
limit. Optax advances that counter with saturating `safe_increment` and gives
up only when the count exceeds the limit, so invalid gradients can never
activate its give-up branch. This adds native state leaves to checkpoints;
build fresh matching optimizer templates rather than reusing old artifacts.

No clipping value is selected by default. Explicit global-norm clipping can
prevent large finite gradients from overflowing adaptive moments. Without
it, all gradient elements may be finite while Adam's moments become infinite;
`last_finite=True` reports the gradient check, not validity of the resulting
state. Loss, resulting parameters/optimizer state and mutable model state
still require the trainer's observation and checkpoint-admission checks.

`create_transformation` remains an unguarded transformation builder. An NNX
optimizer constructed directly from it has exactly the policy the caller
supplies; downstream code must not assume it acquired this guard. No custom
rollback, state-copy loop or clipping implementation is used.

::: substrax.optim
