from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    return config


def resolved_config(config: dict[str, Any], vocab_size: int) -> dict[str, Any]:
    result = deepcopy(config)
    configured = result["model"].get("vocab_size", "auto")
    if configured not in ("auto", vocab_size):
        raise ValueError(f"vocab_size is {configured}, expected 'auto' or {vocab_size}")
    result["model"]["vocab_size"] = vocab_size
    if result["data"]["max_length"] > result["model"]["max_seq_len"]:
        raise ValueError("data.max_length exceeds model.max_seq_len")
    if result["model"]["max_seq_len"] > result["model"]["num_cells"]:
        raise ValueError("model.max_seq_len exceeds model.num_cells")
    return result
