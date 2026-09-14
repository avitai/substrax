"""Root conftest for substrax test suite.

``pytester`` runs the plugin's own tests in separate pytest processes, and the substrax plugin
restores jax's numeric settings after every test and provides the ``x64``, ``devices`` and
``accelerator`` markers.
"""

pytest_plugins = ["pytester", "substrax.testing.pytest_plugin"]
