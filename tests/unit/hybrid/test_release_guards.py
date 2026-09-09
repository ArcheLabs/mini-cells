from __future__ import annotations

import hashlib
import json
from pathlib import Path

FORMAL_SEED_REGISTRY_SHA256 = "d86da9656ef18c75bab8fa41a9053a68cf4dbb64fbba1a93c80850c29b42e06a"


def test_formal_seed_registry_is_unchanged() -> None:
    path = Path(__file__).resolve().parents[2] / ".." / "research" / "formal_seed_registry.json"
    digest = hashlib.sha256(path.resolve().read_bytes()).hexdigest()
    assert digest == FORMAL_SEED_REGISTRY_SHA256
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    assert payload
