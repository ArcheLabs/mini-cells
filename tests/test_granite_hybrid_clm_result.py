from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "research" / "granite_hybrid_clm_v01" / "validate_result.py"
SPEC = importlib.util.spec_from_file_location("granite_hybrid_validate_result", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
validate_payload = MODULE.validate_payload


def _protocol() -> dict:
    return {
        "experiment": "GRANITE_HYBRID_CLM_V0_1",
        "milestone": {
            "controlled_facts": 2,
            "minimum_committed_facts": 2,
            "minimum_final_semantic_choice_accuracy": 0.98,
            "contextual_child_required": True,
            "manifest_fork_merge_required": True,
        },
    }


def _result() -> dict:
    return {
        "experiment": "GRANITE_HYBRID_CLM_V0_1",
        "foundation": {"trainable": False},
        "compatibility_max_abs_logit_delta": 0.0,
        "committed_facts": 2,
        "retention_choice_accuracy": 1.0,
        "cells": [
            {"cell_id": "fact-001", "status": "COMMITTED", "production_choice_accuracy": 1.0},
            {"cell_id": "fact-002", "status": "COMMITTED", "production_choice_accuracy": 1.0},
        ],
        "contextual_child": {
            "cell_id": "fact-001-v2",
            "status": "COMMITTED",
            "old_choice_accuracy_with_child": 1.0,
            "new_choice_accuracy": 1.0,
            "rollback_old_choice_accuracy": 1.0,
        },
        "manifest": {
            "foundation_model_id": "foundation",
            "foundation_revision": "revision",
            "cells": [
                {"cell_id": "fact-001", "digest": "a"},
                {"cell_id": "fact-002", "digest": "b"},
                {"cell_id": "fact-001-v2", "digest": "c"},
            ],
        },
    }


def test_acceptance_contract_rejects_child_that_damages_parent() -> None:
    result = _result()
    assert validate_payload(result, _protocol()) == []
    result["contextual_child"]["old_choice_accuracy_with_child"] = 0.5
    errors = validate_payload(result, _protocol())
    assert "contextual child damages parent v1 semantics" in errors
