"""How a format-2 payload maps onto format-3 items.

Format 2 wrote one payload pytree under the Orbax item ``model``, and each producer laid
its payload out its own way: substrax, opifex and cellifex wrote a module's state alone,
artifex a trainer tree, DiffAV a ``{"model", "optimizer"}`` dictionary. A
:class:`LegacyLayout` says how such a payload splits into named items and, for a restore
onto templates, how the templates assemble into the payload's template. substrax ships
the module-only layout; a producer passes its own to ``restore`` and ``upgrade_checkpoints``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True, kw_only=True)
class LegacyLayout:
    """A format-2 payload layout.

    Attributes:
        name: The layout's name, for messages and the command line.
        items_of: Splits a restored payload into format-3 items by name.
        template_of: Assembles format-3 templates (by item name) into the payload's template,
            so a restore onto templates places every array as the templates say.
    """

    name: str
    items_of: Callable[[Any], Mapping[str, Any]]
    template_of: Callable[[Mapping[str, Any]], Any]


MODULE_ONLY_FORMAT2 = LegacyLayout(
    name="module-only",
    items_of=lambda payload: {"model": payload},
    template_of=lambda templates: templates["model"],
)
"""The payload is a module's state and nothing else: the ``model`` item."""
