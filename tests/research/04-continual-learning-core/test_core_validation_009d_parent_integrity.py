from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
VALIDATIONS = ROOT / "research" / "validations"
ARTIFACTS = ROOT / "artifacts" / "experiments"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_core009d_parent_protocol_hashes_match_published_evidence() -> None:
    protocol009d = _load(
        VALIDATIONS / "core-009d-compositional-operator-geometry" / "protocol.json"
    )
    parent = protocol009d["parent_evidence"]

    protocol009a_path = (
        VALIDATIONS / "core-009a-factorized-functional-coordinates" / "protocol.json"
    )
    decision009a = _load(
        ARTIFACTS
        / "core-validation-009a-factorized-functional-coordinates"
        / "confirmation"
        / "decision.json"
    )
    actual009a_sha = _sha256(protocol009a_path)
    assert decision009a["protocol_sha256"] == actual009a_sha
    assert parent["core009a_protocol_sha256"] == actual009a_sha
    assert decision009a["status"] == parent["core009a_status"]
    assert decision009a["supported"] is True
    assert decision009a["locked_split"] == {
        "left_dim": parent["core009a_locked_left_dim"],
        "right_dim": parent["core009a_locked_right_dim"],
    }

    protocol009c_path = (
        VALIDATIONS / "core-009c-sparse-local-effect-geometry" / "protocol.json"
    )
    decision009c = _load(
        ARTIFACTS
        / "core-validation-009c-sparse-local-effect-geometry"
        / "discovery"
        / "decision.json"
    )
    actual009c_sha = _sha256(protocol009c_path)
    assert decision009c["protocol_sha256"] == actual009c_sha
    assert parent["core009c_protocol_sha256"] == actual009c_sha
    assert decision009c["status"] == parent["core009c_status"]
    assert decision009c["confirmation_allowed"] is False
