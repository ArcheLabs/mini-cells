"""MINI Cells Echo research package."""

from .config import load_config
from .vocab import CharVocab

__all__ = ["CharVocab", "EchoModel", "load_config"]


def __getattr__(name):
    if name == "EchoModel":
        from .model import EchoModel
        return EchoModel
    raise AttributeError(name)
