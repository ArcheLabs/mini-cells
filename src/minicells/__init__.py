"""Public MiniCells model and inference API."""

from .config import load_config
from .vocab import CharVocab

__all__ = ["CLM", "GenerationResult", "CharVocab", "EchoModel", "load_config"]


def __getattr__(name):
    if name == "EchoModel":
        from .model import EchoModel
        return EchoModel
    if name in ("CLM", "GenerationResult"):
        from .clm_release import CLM, GenerationResult
        return {"CLM": CLM, "GenerationResult": GenerationResult}[name]
    raise AttributeError(name)
