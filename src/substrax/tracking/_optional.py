"""Loading of the optional tracking backends and plotting support."""

from __future__ import annotations

import importlib
from types import ModuleType


def import_optional(module: str, *, extra: str) -> ModuleType:
    """Import a module that ships in one of substrax's optional extras.

    Args:
        module: Dotted module name, for example ``"wandb"`` or ``"matplotlib.pyplot"``.
        extra: The extra that provides it, for example ``"wandb"``.

    Returns:
        The imported module.

    Raises:
        ImportError: If the module is not installed. The message names the extra.
    """
    try:
        return importlib.import_module(module)
    except ImportError as err:
        raise ImportError(
            f"{module} is not installed. Install it with `uv add 'substrax[{extra}]'`."
        ) from err
