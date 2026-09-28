"""Importing ``substrax.tracking`` loads neither jax nor an optional backend.

The loggers turn values into NumPy arrays and load matplotlib, W&B and MLflow only when a
logger needs them, so a process that only logs does not pay for them at import.
"""

from __future__ import annotations

from substrax.testing import run_python


_PROBE = (
    "import json, sys; import substrax.tracking; "
    "print(json.dumps({name: name in sys.modules "
    "for name in ('jax', 'matplotlib', 'wandb', 'mlflow')}))"
)


def test_importing_tracking_loads_no_jax_and_no_optional_backend() -> None:
    loaded = run_python(_PROBE, timeout=180.0).check().last_json_as(dict[str, bool])

    assert loaded == {"jax": False, "matplotlib": False, "wandb": False, "mlflow": False}
