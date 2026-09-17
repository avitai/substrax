# Checkpoint

`substrax.checkpoint` is one checkpoint store over
[Orbax](https://orbax.readthedocs.io): step-addressed, pickle-free, with retention delegated
to `orbax.checkpoint.CheckpointManager`. A checkpoint is a step holding named items, the
things a training loop owns (`model`, `optimizer`, `rng`, `data_iterator`, `extensions`),
beside one `CheckpointMetadata` record.

```python
from flax import nnx
from substrax.checkpoint import OrbaxCheckpointStore

with OrbaxCheckpointStore("checkpoints", max_to_keep=5) as store:
    store.save(100, {"model": nnx.state(model), "rng": key}, epoch=2, metrics={"loss": 0.42})
    checkpoint = store.restore(100, templates={"model": nnx.state(model)})
    nnx.update(model, checkpoint.items["model"])
    best = store.best_step("loss")
```

Every item is written with `PyTreeSave`, which carries arrays, typed PRNG keys and
plain-Python leaves (ints, strings, booleans, lists) alike, and the metadata with `JsonSave`,
so restoring a checkpoint never executes code. On disk each item holds its pytree under a
single `tree` node, so an item that is a bare array (the `rng` key) is accepted by every
supported Orbax, the 0.11.33 floor included. Restoring onto templates places every array on
its template leaf's device and dtype, whatever topology the checkpoint was written on; an item
without a template comes back as stored.

Writes are strict: an existing step is refused unless `overwrite=True`, a step below the
latest is refused, and a step Orbax declines raises. Every refusal is a
`CheckpointNotWrittenError` naming the step, the latest step and the reason. Nothing is
created before the first save, and `resolve_checkpoint_dir` picks a run's directory without
creating it.

A format-2 checkpoint (substrax 0.1.5 to 0.1.9, one `model` payload with a
`checkpoint_version` sidecar) restores through the migration registry, split into items by a
`LegacyLayout`: the module-only layout by default, or the producer's own. `upgrade_checkpoints`
and `python -m substrax.checkpoint upgrade SOURCE DESTINATION` rewrite a root in the current
format into a new root, never in place.

::: substrax.checkpoint
