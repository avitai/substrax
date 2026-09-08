# Callbacks

`substrax.callbacks` holds the training-callback protocol and metric-driven early
stopping, in two shapes:

- `EarlyStopping` for a loop that feeds it one validation value per epoch and asks
  `should_stop`;
- `EarlyStoppingCallback` for a trainer that drives `TrainingCallback` hooks and reads
  the metric from the epoch's logs.

Both keep their counters in a `BestMetricTracker`.

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

::: substrax.callbacks
