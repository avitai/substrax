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
its template leaf's device, whatever topology the checkpoint was written on; an item without a
template comes back as stored.

A template never changes a saved array's dtype silently. Orbax casts every array to its
template leaf's dtype, so a float64 array restored onto a float32 template, or a float32 model
onto a bfloat16 one, would lose precision or change without a word. `save` writes every
array's dtype, by item and leaf path, as a `dtypes` JSON item beside the metadata record (which
`best_step` reads for every step, so it stays small), and `restore` compares each template leaf
against it before reading any array, raising
`CheckpointDtypeMismatchError` with a `DtypeMismatch` (item, leaf, saved and restored dtype)
for every differing array. An item restored without a template is compared after it is read:
while jax's x64 mode is off, jax creates a 64-bit array at 32 bits, and the error says how to
enable x64. `restore(..., cast_dtypes=True)` accepts the cast in either case. Template leaves
may be arrays or `jax.ShapeDtypeStruct`; a typed PRNG key compares by its `uint32` data, as
Orbax stores it, and strings and Python numbers are not compared. A checkpoint written without
the `dtypes` item is compared against Orbax's per-array metadata instead, which opens every
array's store on each restore.

Writes are strict: an existing step is refused unless `overwrite=True`, a step below the
latest is refused, and a step Orbax declines raises. Every refusal is a
`CheckpointNotWrittenError` naming the step, the latest step and the reason. Nothing is
created before the first save, and `resolve_checkpoint_dir` picks a run's directory without
creating it.

A checkpoint whose metadata names another format, another producer or another version number is
refused with `UnsupportedCheckpointError` naming what it holds, rather than read as this one.

::: substrax.checkpoint
