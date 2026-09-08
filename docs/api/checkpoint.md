# Checkpoint

`substrax.checkpoint` is one checkpoint store over
[Orbax](https://orbax.readthedocs.io): step-addressed, pickle-free, with retention and
best-step selection delegated to `orbax.checkpoint.CheckpointManager`.

```python
from substrax.checkpoint import OrbaxCheckpointStore

with OrbaxCheckpointStore("checkpoints", max_to_keep=5) as store:
    store.save(model, step=100, loss=0.42)
    model, metadata = store.restore(model)          # the latest step
    best = store.best_step(metric="loss")
```

Array state is written with `StandardSave` and metadata with `JsonSave`, so restoring a
checkpoint never executes code. Consumers depend on the `CheckpointStore` protocol; the
Orbax class is the one implementation.

::: substrax.checkpoint
