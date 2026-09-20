"""JAX process configuration, declared once and applied before or after jax is imported.

``JaxRuntime`` describes the settings. ``runtime_environment`` renders them as the environment of a
process that has not imported jax yet (a child process, or a conftest before its first
``import jax``), ``child_environment`` builds a child's whole environment from them, ``apply_runtime`` applies them to the current process, and
``resolve_test_runtime`` chooses the backend and emulated CPU devices of a test run.
``configure_entry_point_logging`` sets up the root logger from an entry point. Importing this
package imports no jax and changes no process state.
"""

from substrax.runtime.apply import apply_runtime
from substrax.runtime.config import JaxRuntime, MatmulPrecision
from substrax.runtime.entry_point_logging import configure_entry_point_logging
from substrax.runtime.environment import (
    child_environment,
    resolve_test_runtime,
    runtime_environment,
)
from substrax.runtime.errors import RuntimeConfigurationError, XlaFlagConflictError
from substrax.runtime.xla_flags import merge_xla_flags, parse_xla_flags


__all__ = [
    "JaxRuntime",
    "MatmulPrecision",
    "RuntimeConfigurationError",
    "XlaFlagConflictError",
    "apply_runtime",
    "child_environment",
    "configure_entry_point_logging",
    "merge_xla_flags",
    "parse_xla_flags",
    "resolve_test_runtime",
    "runtime_environment",
]
