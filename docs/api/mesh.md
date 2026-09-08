# Mesh

`substrax.mesh` builds device meshes and describes how arrays are laid out on them.

```python
from substrax.mesh import DeviceMeshManager, FSDPStrategy, ShardingConfig, fsdp_rules

mesh = DeviceMeshManager.create_device_mesh({"data": 2, "model": 4})
rules = fsdp_rules(data_axis="data", model_axis="model")
config = ShardingConfig(data_parallel_size=2, tensor_parallel_size=4, fsdp_enabled=True)
strategy = FSDPStrategy("model", mesh_axis=1, min_weight_size=config.fsdp_min_weight_size)
```

Three layers, from low to high:

| Layer | Names | Role |
| --- | --- | --- |
| Meshes | `DeviceMeshManager` | `jax.make_mesh` over the visible devices, plus mesh introspection |
| Rules | `MeshRules`, `data_parallel_rules`, `fsdp_rules`, `create_named_sharding`, `partition_spec_for_names` | Map logical axis names to mesh axes and build `NamedSharding`s |
| Strategies | `ShardingConfig`, `ParallelismConfig`, `ShardingStrategy` and its data-parallel, FSDP, tensor-parallel, pipeline-parallel and multi-dimensional implementations | Decide the partition spec of every parameter in a model |

::: substrax.mesh
