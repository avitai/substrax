"""The volume listing types of the fake Modal."""

from __future__ import annotations

import enum
from dataclasses import dataclass


class FileEntryType(enum.IntEnum):
    UNSPECIFIED = 0
    FILE = 1
    DIRECTORY = 2


@dataclass(frozen=True)
class FileEntry:
    path: str
    type: FileEntryType
