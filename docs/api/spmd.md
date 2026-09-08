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

`spmd_train_step` differentiates with `nnx.value_and_grad` and updates the
`nnx.Optimizer`; XLA inserts the gradient all-reduce when the parameters are sharded
under `jax.set_mesh`. The collectives (`reduce_mean`, `reduce_sum`, `all_gather`,
`collect_from_devices` and friends) work on metric dictionaries.

::: substrax.spmd
