"""Placement compatibility module.

The implementation lives with structural inspection; this module keeps the
public import path explicit for downstream users.
"""

from .inspector import CellPlacement, normalize_placements

__all__ = ["CellPlacement", "normalize_placements"]

