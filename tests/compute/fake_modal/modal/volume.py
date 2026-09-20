"""The fake Modal's ``modal.volume``, which re-exports the listing types as Modal's does."""

from .types import FileEntry, FileEntryType


__all__ = ["FileEntry", "FileEntryType"]
