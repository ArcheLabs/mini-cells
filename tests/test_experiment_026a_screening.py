from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "research" / "stages" / "03-routing-and-growth" / "sources" / "experiment-026a-protocol.json"


def test_026a_protocol_is_low_cost_screening() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["experiment_id"] == "026a"
    assert protocol["format"] == "minicells.cell-granularity-screening-30m.v1"
    assert protocol["arms"]["granularities"] == [1, 4, 8]
    assert protocol["arms"]["persistent_growth"] is False
    assert protocol["continuation"]["tokens_per_arm"] == 5_000_000
    assert protocol["continuation"]["total_training_tokens"] == 15_000_000
    assert protocol["continuation"]["eval_tokens"] == [0, 1_000_000, 2_000_000, 5_000_000]


def test_026a_gate_is_explicitly_nonconfirmatory() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    gate = protocol["screening_gate"]
    assert gate["minimum_specialization_gain_delta"] == 0.02
    assert gate["maximum_balanced_nll_ratio"] == 1.03
    assert gate["positive_status"] == "PROCEED_TO_026_CONFIRMATION"
    assert gate["negative_status"] == "DO_NOT_PROCEED_TO_026_CONFIRMATION"
    limits = " ".join(protocol["interpretation_limits"]).lower()
    assert "screening" in limits
    assert "not a confirmatory" in limits
