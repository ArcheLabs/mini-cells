"""Compatibility alias for app and research scripts."""
import sys
from minicells_native import backend as _module

sys.modules[__name__] = _module
