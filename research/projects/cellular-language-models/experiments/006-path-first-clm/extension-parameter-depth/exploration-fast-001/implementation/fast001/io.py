"""Reuse repository atomic JSON writer, creating the destination parent first."""

from pathlib import Path
from exp006.reuse import write_json as atomic_write


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, value)
