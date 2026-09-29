# SPMD

`substrax.spmd` is the data-parallel training path on top of `substrax.mesh`: place a
batch on the data axis, run one step, reduce metrics across devices.

```python
import jax
from flax import nnx

from substrax.mesh import DeviceMeshManager
from substrax.spmd import create_data_parallel_sharding, place_batch_on_shards, spmd_train_step

mesh = DeviceMeshManager.create_device_mesh({"data": jax.device_count()})
sharding = create_data_parallel_sharding(mesh)


@nnx.jit
def train_step(model, optimizer, batch):
    return spmd_train_step(model, optimizer, loss_fn, batch)


with jax.set_mesh(mesh):
    loss = train_step(model, optimizer, place_batch_on_shards(batch, sharding))
```

`place_batch_on_shards` also takes a pytree prefix of the batch whose leaves are shardings, one
per subtree, for a batch whose fields are not all split along the batch axis: rows on the data
axis and a batch-level value, which has no batch axis, replicated.

```python
from jax.sharding import NamedSharding, PartitionSpec

replicated = NamedSharding(mesh, PartitionSpec())
batch = {"rows": {"x": x, "y": y}, "batch_state": {"lam": lam}}
placed = place_batch_on_shards(batch, {"rows": sharding, "batch_state": replicated})
```

On several processes each passes the rows it loaded and jax assembles the global batch; a
replicated leaf must be equal on every process. A leaf with fewer dimensions than its sharding
splits, such as a scalar under the data-axis sharding, is refused with its path.

`spmd_train_step` differentiates with `nnx.value_and_grad` and updates the
`nnx.Optimizer`; XLA inserts the gradient all-reduce when the parameters are sharded
under `jax.set_mesh`. The reductions (`reduce_mean`, `reduce_sum`, `reduce_max`,
`reduce_min`) and the collectives (`reduce_mean_collective`, `reduce_sum_collective`,
`all_gather`) reduce every array leaf of the tree they are given, at any depth, so a gradient tree
beside scalar metrics is reduced leaf by leaf, and return a tree of the same structure; any other
leaf passes through. `reduce_custom` and `collect_from_devices` take a flat
mapping from metric name to value.

::: substrax.spmd
