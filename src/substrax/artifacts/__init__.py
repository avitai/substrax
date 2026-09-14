"""Output directories that never default into the working tree.

``resolve_output_dir`` picks the directory a run writes to: an explicit argument, then
``AVITAI_OUTPUT_DIR``, then a directory created once per process under the system temporary
directory. Importing this package creates nothing.
"""

from substrax.artifacts.output_dir import OUTPUT_DIR_ENV, OutputLocation, resolve_output_dir


__all__ = ["OUTPUT_DIR_ENV", "OutputLocation", "resolve_output_dir"]
