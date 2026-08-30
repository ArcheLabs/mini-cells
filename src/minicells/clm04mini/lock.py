"""Build and validate the pre-formal CLM-0.4-mini protocol lock."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

from .protocol import CandidateOptimizerConfig, candidate_grid, file_sha256, validate_protocol


def _registered(config: CandidateOptimizerConfig, options: list[CandidateOptimizerConfig]) -> bool:
    return config.to_dict() in [item.to_dict() for item in options]


def build_protocol_lock(
    *,
    protocol: Mapping[str, Any],
    template: Mapping[str, Any],
    protocol_path: str | Path,
    direct_optimizer: CandidateOptimizerConfig,
    growth_optimizer: CandidateOptimizerConfig,
    tokenizer_manifest: Mapping[str, Any],
    base_corpus_manifest: Mapping[str, Any],
    curriculum_manifest: Mapping[str, Any],
    dataset_revision: str,
    routing_salt: str,
    minimum_base_cell_activation: int,
    code_commit: str,
    code_tree: str,
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    validate_protocol(protocol)
    if not _registered(direct_optimizer, candidate_grid(protocol, "direct")):
        raise ValueError("direct optimizer is outside the registered calibration grid")
    if not _registered(growth_optimizer, candidate_grid(protocol, "growth")):
        raise ValueError("growth optimizer is outside the registered calibration grid")
    if not dataset_revision:
        raise ValueError("dataset revision must be pinned before formal execution")
    if not code_commit or not code_tree:
        raise ValueError("code commit/tree are required for the protocol lock")
    lock = copy.deepcopy(dict(template))
    lock["protocol_sha256"] = file_sha256(protocol_path)
    lock["lock_status"] = "LOCKED"
    lock["code"] = {
        "commit": str(code_commit),
        "tree": str(code_tree),
        "tracked_tree_dirty": False,
    }
    lock["routing"] = {
        "stable_hash_algorithm": "sha256",
        "protocol_salt": str(routing_salt),
        "structural_logit_tolerance": float(protocol["metrics"]["structural_logit_tolerance"]),
    }
    lock["tokenizer"] = {
        "type": "byte-level-BPE",
        "vocab_size": int(protocol["model"]["vocab_size"]),
        "hash": str(tokenizer_manifest["tokenizer_sha256"]),
        "training_manifest_hash": str(tokenizer_manifest["manifest_sha256"]),
    }
    lock["base_corpus"] = {
        "dataset_revision": str(dataset_revision),
        "sample_manifest_hash": str(base_corpus_manifest["manifest_sha256"]),
        "preprocessing_version": str(base_corpus_manifest["generator_version"]),
        "target_tokens": int(protocol["base_training"]["target_tokens"]),
    }
    lock["curriculum"] = {
        "generator_version": str(curriculum_manifest["generator_version"]),
        "transaction_manifest_hash": str(curriculum_manifest["manifest_sha256"]),
        "transactions": int(protocol["continual_curriculum"]["total_transactions"]),
    }
    lock["base_cell_minimum_activation"] = int(minimum_base_cell_activation)
    lock["selected_direct_candidate"] = direct_optimizer.to_dict()
    lock["selected_growth_private_candidate"] = growth_optimizer.to_dict()
    lock["environment"] = dict(environment)
    lock["development_seed_used"] = int(protocol["replication"]["development_seed"])
    lock["formal_seeds"] = [int(value) for value in protocol["replication"]["formal_model_seeds"]]
    lock["formal_results_observed_when_locked"] = False
    lock["notes"] = (
        "Generated after development-seed calibration. Formal seeds must remain unopened "
        "until this LOCKED file is committed with a clean tracked tree."
    )
    validate_protocol_lock(lock, protocol=protocol)
    return lock


def validate_protocol_lock(lock: Mapping[str, Any], *, protocol: Mapping[str, Any]) -> None:
    validate_protocol(protocol)
    if lock.get("lock_status") != "LOCKED":
        raise ValueError("protocol lock is not LOCKED")
    if bool(lock.get("formal_results_observed_when_locked")):
        raise ValueError("lock must be created before any formal result is observed")
    required = [
        lock.get("protocol_sha256"),
        lock.get("code", {}).get("commit"),
        lock.get("code", {}).get("tree"),
        lock.get("routing", {}).get("protocol_salt"),
        lock.get("tokenizer", {}).get("hash"),
        lock.get("tokenizer", {}).get("training_manifest_hash"),
        lock.get("base_corpus", {}).get("dataset_revision"),
        lock.get("base_corpus", {}).get("sample_manifest_hash"),
        lock.get("curriculum", {}).get("transaction_manifest_hash"),
        lock.get("base_cell_minimum_activation"),
    ]
    if any(value is None or value == "" for value in required):
        raise ValueError("protocol lock contains unresolved required fields")
    direct = CandidateOptimizerConfig(**lock["selected_direct_candidate"])
    growth = CandidateOptimizerConfig(**lock["selected_growth_private_candidate"])
    if not _registered(direct, candidate_grid(protocol, "direct")):
        raise ValueError("locked direct optimizer is not registered")
    if not _registered(growth, candidate_grid(protocol, "growth")):
        raise ValueError("locked growth optimizer is not registered")


def write_protocol_lock(path: str | Path, lock: Mapping[str, Any]) -> None:
    Path(path).write_text(json.dumps(dict(lock), indent=2, sort_keys=True) + "\n", encoding="utf-8")
