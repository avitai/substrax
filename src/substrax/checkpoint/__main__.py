"""Command line: ``python -m substrax.checkpoint upgrade SOURCE DESTINATION [--layout NAME]``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from substrax.checkpoint.legacy import LegacyLayout, MODULE_ONLY_FORMAT2
from substrax.checkpoint.upgrade import upgrade_checkpoints


LAYOUTS: dict[str, LegacyLayout] = {MODULE_ONLY_FORMAT2.name: MODULE_ONLY_FORMAT2}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m substrax.checkpoint", description="Checkpoint root maintenance."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    upgrade = commands.add_parser(
        "upgrade", help="Write every step of SOURCE to DESTINATION in the current format."
    )
    upgrade.add_argument("source", type=Path, help="The root to read; it is not modified.")
    upgrade.add_argument("destination", type=Path, help="The new root; must not exist or be empty.")
    upgrade.add_argument(
        "--layout",
        choices=sorted(LAYOUTS),
        default=MODULE_ONLY_FORMAT2.name,
        help="How a format-2 payload splits into items (default: %(default)s).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command line and return the exit status."""
    arguments = _parser().parse_args(argv)
    steps = upgrade_checkpoints(
        arguments.source, arguments.destination, legacy_layout=LAYOUTS[arguments.layout]
    )
    sys.stdout.write(
        f"upgraded {len(steps)} step(s) from {arguments.source} to {arguments.destination}: "
        f"{', '.join(str(step) for step in steps)}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
