from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT
    / "research"
    / "validations"
    / "hybrid-clm-prompt-address-001"
    / "protocol.json"
)
MILESTONE_PROTOCOL = (
    ROOT / "research" / "validations" / "granite-hybrid-clm-v0.1" / "protocol.json"
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def test_prompt_address_protocol_is_frozen_and_preserves_write_gates() -> None:
    protocol = _load(PROTOCOL)
    milestone = _load(MILESTONE_PROTOCOL)
    assert protocol["experiment"] == "HYBRID_CLM_PROMPT_ADDRESS_001"
    assert protocol["protocol_version"] == 1.1
    assert protocol["status"] == "DIAGNOSTIC_PROTOCOL_FROZEN_GPU_PENDING"
    assert protocol["diagnostic_scope"]["formal_milestone_decision"] is False
    assert protocol["diagnostic_scope"]["thresholds_must_not_be_relaxed_after_gpu_observation"]

    routing = protocol["routing"]
    assert routing["address_scope"] == "prompt_anchor"
    assert routing["write_scope"] == "anchor_and_later"
    assert routing["candidate_answer_affects_routing"] is False
    assert routing["competitive_normalization"] is False
    assert routing["gate_threshold"] == milestone["substrate"]["gate_threshold"] == 0.8

    address = protocol["address_gates"]
    assert address["minimum_train_positive_recall"] == 1.0
    assert address["maximum_train_negative_false_positive_rate"] == 0.02
    assert address["minimum_heldout_positive_recall"] == 1.0
    assert address["maximum_history_anchor_false_positive_rate"] == 0.0
    assert address["heldout_positive_used_for_address_training"] is False
    assert address["history_anchor_used_for_address_training"] is False

    write = protocol["write_gates"]
    milestone_gate = milestone["milestone"]
    assert write["maximum_history_kl"] == milestone_gate["maximum_history_kl_per_commit"] == 0.02
    assert write["minimum_target_nll_gain"] == milestone_gate["minimum_target_nll_gain_per_commit"] == 0.5
    assert write["minimum_semantic_choice_accuracy"] == 1.0


def test_prompt_address_protocol_registers_exact_implementation_blobs() -> None:
    protocol = _load(PROTOCOL)
    blobs = protocol["implementation_git_blobs"]
    assert blobs
    for relative_path, expected in blobs.items():
        path = ROOT / relative_path
        assert path.is_file(), relative_path
        assert _git_blob_sha(path) == expected, relative_path


def test_hosted_policy_requires_tokens_and_publishes_failures() -> None:
    protocol = _load(PROTOCOL)
    hosted = protocol["hosted_environment"]
    assert hosted["require_cuda"] is True
    assert hosted["require_hf_token"] is True
    assert hosted["require_github_token_for_publish"] is True
    assert protocol["publishing"]["terminal_pass_and_fail_are_published"] is True
    assert protocol["smoke"]["fresh_reload_required_for_pass"] is True
    assert protocol["smoke"]["clean_seed_workspace_before_run"] is True
    assert protocol["gpu_policy"]["second_gpu_for_smoke"] == "intentionally_unused"
