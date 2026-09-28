"""Run a project's commands on remote accelerators through provider plugins.

A ``JobSpec`` states what runs: tasks, the machine's resources (accelerators, CPU cores, memory),
extras, JAX settings and mounts. The worker
(``python -m substrax.compute.worker``) runs it the same way on every provider. Importing this
package imports no jax and no provider library.
"""

from substrax.compute.spec import (
    Accelerator,
    example_tasks,
    JOB_SPEC_VERSION,
    JobSpec,
    Mount,
    MountAccess,
    read_job_spec,
    Resources,
    Task,
)


__all__ = [
    "JOB_SPEC_VERSION",
    "Accelerator",
    "JobSpec",
    "Mount",
    "MountAccess",
    "Resources",
    "Task",
    "example_tasks",
    "read_job_spec",
]
