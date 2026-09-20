"""Backends by name, from the ``substrax.compute.backends`` entry-point group.

substrax registers its own backends (``local``, ``modal``, ``skypilot``) in that group, and any
other distribution adds a provider by registering a :data:`~substrax.compute.backend.BackendFactory`
there, so a built-in backend and a third-party one load the same way. A backend's module is
imported only when its name is chosen; one whose provider library is missing raises the
``ImportError`` its module raises, naming the extra to install.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from importlib.metadata import entry_points, EntryPoint, EntryPoints

from substrax.compute.backend import ComputeBackend
from substrax.typing import JsonValue


ENTRY_POINT_GROUP = "substrax.compute.backends"


class UnknownBackendError(LookupError):
    """No registered backend has the requested name."""


def backend_names(*, discover: Callable[..., EntryPoints] = entry_points) -> tuple[str, ...]:
    """The names of the registered backends, sorted.

    Args:
        discover: Lists entry points by group; :func:`importlib.metadata.entry_points` by default.

    Returns:
        The names.
    """
    return tuple(sorted(point.name for point in discover(group=ENTRY_POINT_GROUP)))


def load_backend(
    name: str,
    settings: Mapping[str, JsonValue] | None = None,
    *,
    discover: Callable[..., EntryPoints] = entry_points,
) -> ComputeBackend:
    """Build the backend registered under ``name`` from its settings.

    Args:
        name: The backend's name.
        settings: Its settings table; empty when ``None``.
        discover: Lists entry points by group; :func:`importlib.metadata.entry_points` by default.

    Returns:
        The backend.

    Raises:
        UnknownBackendError: If no backend has that name; the message lists the registered ones.
        TypeError: If the entry point builds something that is not a backend.
    """
    points: dict[str, EntryPoint] = {
        point.name: point for point in discover(group=ENTRY_POINT_GROUP)
    }
    if name not in points:
        known = ", ".join(sorted(points)) or "none"
        msg = f"no compute backend named {name!r}; registered: {known}"
        raise UnknownBackendError(msg)
    factory = points[name].load()
    backend = factory(settings or {})
    if not isinstance(backend, ComputeBackend):
        msg = f"entry point {points[name].value!r} built {type(backend).__name__}, not a backend"
        raise TypeError(msg)
    return backend
