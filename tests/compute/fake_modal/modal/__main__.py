"""``python -m modal app logs APP_ID [--env E] [--since T] [--follow]`` against the fake's app logs."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(prog="modal")
    parser.add_argument("group", choices=["app"])
    parser.add_argument("command", choices=["logs"])
    parser.add_argument("app_id")
    parser.add_argument("--env")
    parser.add_argument("--since")
    parser.add_argument("--follow", action="store_true")
    arguments = parser.parse_args()
    log = Path(os.environ["FAKE_MODAL_ROOT"]) / "apps" / f"{arguments.app_id}.log"
    with log.open(encoding="utf-8") as lines:
        while True:
            line = lines.readline()
            if line:
                sys.stdout.write(line)
                sys.stdout.flush()
            elif arguments.follow:
                time.sleep(0.1)
            else:
                return 0


if __name__ == "__main__":
    sys.exit(main())
