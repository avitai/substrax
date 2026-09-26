# Callbacks

`substrax.callbacks` holds the training-callback protocol and metric-driven early
stopping, in two shapes:

- `EarlyStopping` for a loop that feeds it one validation value per epoch and asks
  `should_stop`;
- `EarlyStoppingCallback` for a trainer that drives `TrainingCallback` hooks and reads
  the metric from the epoch's logs.

`EarlyStopping` keeps its counters in a `BestMetricTracker`; `EarlyStoppingCallback` applies
an `EarlyStopping` to the logged metric and adds the goal and divergence thresholds and the
non-finite check. `patience` is at least 1: patience counts epochs without improvement, so an
improving epoch never stops training.

```python
from substrax.callbacks import EarlyStopping, EarlyStoppingCallback, EarlyStoppingConfig

stopper = EarlyStopping(patience=3, min_delta=1e-3)
for epoch in range(epochs):
    stopper.update(validate())
    if stopper.should_stop:
        break

callback = EarlyStoppingCallback(EarlyStoppingConfig(monitor="val_loss", patience=10))
```

Learning-rate decay on a plateau is not here: use `optax.contrib.reduce_on_plateau`.

## Resuming a run

`BestMetricTracker`, `EarlyStopping`, `EarlyStoppingCallback` and `CallbackList` satisfy
`substrax.typing.Checkpointable`: `get_state()` returns their bookkeeping as a dictionary a
checkpoint store writes, and `set_state()` on objects built the same way continues where the
saved ones stood, so a resumed run stops at the epoch the uninterrupted run would have.
`CallbackList` keeps each stateful callback's state under its position, skips callbacks without
state, and refuses a state saved from a list whose stateful callbacks stood elsewhere.

```python
from substrax.callbacks import CallbackList
from substrax.checkpoint import OrbaxCheckpointStore

callbacks = CallbackList([EarlyStoppingCallback(EarlyStoppingConfig(patience=10))])
with OrbaxCheckpointStore(directory) as store:
    store.save(step, {"extensions": {"callbacks": callbacks.get_state()}})
    callbacks.set_state(store.restore(step).items["extensions"]["callbacks"])
```

::: substrax.callbacks
