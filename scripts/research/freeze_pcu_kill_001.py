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
DEFAULT_PROTOCOL = ROOT / "research/protocols/pcu-kill-001/FROZEN_PROTOCOL.json"
DEFAULT_HASH = ROOT / "research/protocols/pcu-kill-001/FROZEN_PROTOCOL.sha256"
SEED_REGISTRY = ROOT / "research/formal_seed_registry.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--engineering-summary", type=Path)
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
    if args.engineering_summary:
        summary = json.loads(args.engineering_summary.read_text(encoding="utf-8"))
        if summary.get("scientific_evidence") is not False:
            raise ProtocolMismatch("engineering summary must explicitly declare scientific_evidence=false")
        for key in ("g0", "cache", "dataset_audit"):
            if summary.get(key) is not True:
                raise ProtocolMismatch(f"engineering summary gate failed: {key}")
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
    protocol["training"].update({"learning_rate": 1e-3, "max_optimizer_steps": 128, "selected_k": 1, "lora_rank": 114})
    protocol["formal_execution"] = {"formal_seeds_executed": [], "scientific_evidence": False, "formal_execution_not_started": True}
    write_json(args.protocol, protocol)
    digest = write_protocol_hash(args.protocol, args.hash_path)
    print(json.dumps({"status": "FROZEN_BEFORE_FORMAL", "protocol_sha256": digest, "formal_seeds": list(FORMAL_SEEDS)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
