#!/usr/bin/env python3
"""Freeze PCU-KILL-001 inputs before any formal seed is allowed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from minicells.pcu_kill_001.governance import (  # noqa: E402
    FORMAL_SEEDS,
    ProtocolMismatch,
    assert_seed_registry,
    git_provenance,
    sha256_file,
    verify_protocol_hash,
    write_json,
    write_protocol_hash,
)


BRANCH = "codex/pcu-composability-kill-001"
TEMPLATE = ROOT / "research/protocols/pcu-kill-001/PROTOCOL_TEMPLATE.json"
FROZEN_DIR = ROOT / "artifacts/research/pcu-kill-001/frozen"
DEFAULT_PROTOCOL = FROZEN_DIR / "PROTOCOL.json"
DEFAULT_HASH = FROZEN_DIR / "PROTOCOL.sha256"
SEED_REGISTRY = ROOT / "research/formal_seed_registry.json"


REQUIRED_DECISION_GATES = (
    "g0",
    "cache",
    "dataset_audit",
    "gradient_allocation",
    "branch_a",
    "branch_b",
    "functional_composition",
    "functional_rollback",
    "lora_exact_merge",
    "lora_parameter_match",
    "foundation_immutable",
    "formal_seed_untouched",
)
REQUIRED_SELECTED = (
    "k",
    "optimizer",
    "learning_rate",
    "max_optimizer_steps",
    "lora_rank",
    "max_training_tokens",
    "cells_a",
    "cells_b",
)
REQUIRED_THRESHOLDS = (
    "g0_top1_token_agreement",
    "cache_top1_token_agreement",
    "context_oracle_accuracy",
    "direct_accuracy",
    "merge_retention",
    "composition_accuracy",
    "composition_synergy",
    "anchor_regression",
    "matched_lora_parameter_tolerance",
)


def _required_mapping(payload: dict, key: str) -> dict:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ProtocolMismatch(f"engineering decision is missing mapping {key}")
    return value


def _validate_engineering_decision(decision: dict, model_manifest: dict, provenance: dict) -> tuple[dict, dict, dict]:
    if decision.get("experiment") != "PCU-KILL-001" or decision.get("phase") != "engineering":
        raise ProtocolMismatch("engineering decision is for a different experiment or phase")
    if decision.get("scientific_evidence") is not False:
        raise ProtocolMismatch("engineering decision must explicitly declare scientific_evidence=false")
    if decision.get("formal_ready") is not True:
        raise ProtocolMismatch("engineering decision is not formal-ready")
    gates = _required_mapping(decision, "gates")
    missing = [key for key in REQUIRED_DECISION_GATES if gates.get(key) is not True]
    if missing:
        raise ProtocolMismatch(f"engineering decision gates failed or are missing: {missing}")
    selected = _required_mapping(decision, "selected")
    for key in REQUIRED_SELECTED:
        if key not in selected or selected[key] in (None, "", 0):
            raise ProtocolMismatch(f"engineering decision has no selected {key}")
    if int(selected["k"]) not in (1, 2, 4, 8):
        raise ProtocolMismatch("selected k is outside the registered capacity ladder")
    for key in ("cells_a", "cells_b"):
        if not isinstance(selected[key], list) or len(selected[key]) != int(selected["k"]):
            raise ProtocolMismatch(f"engineering decision {key} does not match selected k")
        if len(set(selected[key])) != len(selected[key]):
            raise ProtocolMismatch(f"engineering decision {key} contains duplicate Cells")
    if float(selected["learning_rate"]) <= 0 or int(selected["max_optimizer_steps"]) <= 0:
        raise ProtocolMismatch("engineering decision contains invalid optimizer settings")
    thresholds = _required_mapping(decision, "thresholds")
    for key in REQUIRED_THRESHOLDS:
        if key not in thresholds or thresholds[key] is None:
            raise ProtocolMismatch(f"engineering decision has no selected threshold {key}")
    decision_source = _required_mapping(decision, "source")
    source_ref = decision_source.get("source_ref")
    if source_ref is not None and source_ref != provenance.get("source_ref"):
        raise ProtocolMismatch("engineering decision was produced on a different source ref")
    source_commit = decision_source.get("source_commit", decision_source.get("commit"))
    source_tree = decision_source.get("source_tree", decision_source.get("tree"))
    if source_commit != provenance.get("source_commit"):
        raise ProtocolMismatch("engineering decision was produced from a different source commit")
    if source_tree != provenance.get("source_tree"):
        raise ProtocolMismatch("engineering decision was produced from a different source tree")
    foundation = _required_mapping(decision, "foundation")
    for key in ("model_repo", "model_revision", "config_sha256", "weight_file_sha256", "tokenizer_sha256"):
        if key not in foundation or foundation[key] in (None, "", []):
            raise ProtocolMismatch(f"engineering decision has no immutable foundation field {key}")
        if foundation[key] != model_manifest.get(key):
            raise ProtocolMismatch(f"engineering decision foundation does not match model manifest: {key}")
    if decision.get("architecture") != model_manifest.get("architecture"):
        raise ProtocolMismatch("engineering decision architecture does not match model manifest")
    budget = _required_mapping(decision, "parameter_budget")
    for key in ("pcu_trainable_parameters", "lora_trainable_parameters", "relative_difference"):
        if key not in budget or budget[key] is None:
            raise ProtocolMismatch(f"engineering decision has no parameter budget field {key}")
    return gates, selected, thresholds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--engineering-decision", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--hash-path", type=Path, default=DEFAULT_HASH)
    args = parser.parse_args()
    provenance = git_provenance(ROOT)
    if args.branch != BRANCH:
        raise ProtocolMismatch(f"freeze requires branch {BRANCH}")
    if provenance["source_ref"] != BRANCH:
        raise ProtocolMismatch(f"current branch is {provenance['source_ref']}, expected {BRANCH}")
    if provenance["source_dirty"]:
        raise ProtocolMismatch("freeze requires a clean source tree")
    if not provenance.get("source_commit") or not provenance.get("source_tree"):
        raise ProtocolMismatch("cannot record immutable git provenance")
    assert_seed_registry(SEED_REGISTRY)
    model_manifest = json.loads(args.model_manifest.read_text(encoding="utf-8"))
    if model_manifest.get("model_repo") != "ibm-granite/granite-3.1-1b-a400m-base":
        raise ProtocolMismatch("freeze requires the registered Granite foundation")
    if not model_manifest.get("model_revision") or not model_manifest.get("config_sha256"):
        raise ProtocolMismatch("model revision and config hash must be resolved before freeze")
    if not model_manifest.get("weight_file_sha256") or not model_manifest.get("tokenizer_sha256"):
        raise ProtocolMismatch("weight and tokenizer file hashes must be resolved before freeze")
    architecture = model_manifest.get("architecture", {})
    for key in ("target_layer", "target_path", "hidden_size", "intermediate_size", "local_experts", "experts_per_token"):
        if key not in architecture or architecture[key] in (None, ""):
            raise ProtocolMismatch(f"model manifest is missing frozen architecture field {key}")
    if (
        int(architecture["hidden_size"]) != 1024
        or int(architecture["intermediate_size"]) != 512
        or int(architecture["local_experts"]) != 32
        or int(architecture.get("cells", 4)) != 4
        or architecture.get("fused_order") != "gate_up"
    ):
        raise ProtocolMismatch("model manifest does not satisfy the registered Granite invariants")
    decision = json.loads(args.engineering_decision.read_text(encoding="utf-8"))
    gates, selected, thresholds = _validate_engineering_decision(decision, model_manifest, provenance)
    protocol = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    protocol.update({
        "status": "FROZEN_BEFORE_FORMAL",
        "source_ref": provenance["source_ref"],
        "source_commit": provenance["source_commit"],
        "source_tree": provenance["source_tree"],
        "formal_seeds": list(FORMAL_SEEDS),
    })
    protocol["model"].update({
        "model_repo": model_manifest["model_repo"],
        "model_revision": model_manifest["model_revision"],
        "config_sha256": model_manifest["config_sha256"],
        "weight_file_sha256": model_manifest.get("weight_file_sha256", []),
        "tokenizer_sha256": model_manifest.get("tokenizer_sha256", []),
        "target_layer": architecture["target_layer"],
        "target_path": architecture["target_path"],
        "hidden_size": architecture["hidden_size"],
        "intermediate_size": architecture["intermediate_size"],
        "local_experts": architecture["local_experts"],
        "experts_per_token": architecture["experts_per_token"],
        "cells_per_expert": architecture.get("cells", 4),
        "cell_width": architecture["intermediate_size"] // architecture.get("cells", 4),
        "fused_projection_order": architecture.get("fused_order", "gate_up"),
    })
    protocol["architecture"] = dict(architecture)
    protocol["training"].update({
        "optimizer": selected["optimizer"],
        "learning_rate": selected["learning_rate"],
        "max_optimizer_steps": selected["max_optimizer_steps"],
        "max_training_tokens": selected["max_training_tokens"],
        "selected_k": selected["k"],
        "lora_rank": selected["lora_rank"],
    })
    protocol["baseline"] = {"lora_rank": selected["lora_rank"]}
    protocol["thresholds"] = dict(thresholds)
    protocol["engineering_decision"] = {
        "path": str(args.engineering_decision),
        "sha256": sha256_file(args.engineering_decision),
        "gates": dict(gates),
        "selected": dict(selected),
    }
    protocol["formal_execution"] = {"formal_seeds_executed": [], "scientific_evidence": False, "formal_execution_not_started": True}
    write_json(args.protocol, protocol)
    digest = write_protocol_hash(args.protocol, args.hash_path)
    frozen_dir = args.protocol.parent
    write_json(frozen_dir / "MODEL_MANIFEST.json", model_manifest)
    write_json(frozen_dir / "ENGINEERING_DECISION.json", decision)
    write_json(frozen_dir / "FREEZE_PROVENANCE.json", {
        "schema": "minicells.pcu-kill-001.freeze-provenance.v1",
        "experiment": "PCU-KILL-001",
        "source": provenance,
        "model_manifest_sha256": sha256_file(args.model_manifest),
        "engineering_decision_sha256": sha256_file(args.engineering_decision),
        "protocol_sha256": digest,
        "formal_execution_not_started": True,
    })
    print(json.dumps({"status": "FROZEN_BEFORE_FORMAL", "protocol_sha256": digest, "formal_seeds": list(FORMAL_SEEDS)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
