"""``python -m substrax.notebooks``: the ``substrax-notebooks`` command on the running interpreter.

A compute task runs it this way, so the command is the project environment's whatever its
``PATH`` holds.
"""

from substrax.notebooks.cli import main


raise SystemExit(main())
