"""Substrax: shared training and hardware infrastructure for the Avitai JAX stack.

The package sits below calibrax, datarax, artifex and opifex and holds the code those
packages used to carry separately: device information and placement (``devices``), device
meshes and sharding strategies (``mesh``), data-parallel training helpers (``spmd``), one
Orbax checkpoint store (``checkpoint``), training callbacks (``callbacks``) and step-wise
experiment tracking (``tracking``). Import from the subpackages; this module exposes only
the version.
"""

__version__ = "0.1.1"

__all__ = ["__version__"]
