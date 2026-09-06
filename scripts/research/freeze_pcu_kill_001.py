#!/usr/bin/env python3
"""Freeze PCU-KILL-001 inputs before any formal seed is allowed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
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
    write_json,
    write_protocol_hash,
)
from minicells.pcu_kill_001.synthetic import POSITIVE_CONTROL_VERSION  # noqa: E402


BRANCH = "codex/pcu-composability-kill-001"
TEMPLATE = ROOT / "research/protocols/pcu-kill-001/PROTOCOL_TEMPLATE.json"
FROZEN_DIR = ROOT / "artifacts/research/pcu-kill-001/frozen"
DEFAULT_PROTOCOL = FROZEN_DIR / "PROTOCOL.json"
DEFAULT_HASH = FROZEN_DIR / "PROTOCOL.sha256"
SEED_REGISTRY = ROOT / "research/formal_seed_registry.json"
ENGINEERING_EVIDENCE_PREFIX = "artifacts/research/pcu-kill-001/engineering/"


REQUIRED_DECISION_GATES = (
    "g0",
    "cache",
    "dataset_audit",
    "context_oracle",
    "gradient_allocation",
    "capacity_ladder",
    "branch_a_capability",
    "branch_b_capability",
    "functional_composition_runtime",
    "functional_rollback",
    "merge_retention",
    "anchor_regression",
    "composition",
    "lora_training",
    "lora_exact_merge",
    "lora_parameter_match",
    "foundation_immutable",
    "formal_seed_untouched",
    "artifact_roundtrip",
)
REQUIRED_SELECTED = (
    "k",
    "optimizer",
    "learning_rate",
    "max_optimizer_steps",
    "lora_rank",
    "max_training_tokens",
)
REQUIRED_THRESHOLDS = (
    "g0_top1_token_agreement",
    "cache_top1_token_agreement",
    "context_oracle_accuracy",
    "context_oracle_retrieval_accuracy",
    "context_oracle_composition_accuracy",
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


def _git(*args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=check,
    )
    return result.stdout.strip()


def _validate_engineering_source_snapshot(decision_source: dict, provenance: dict) -> None:
    """Accept the run source commit or a descendant containing evidence only.

    An E0 decision is produced before its artifacts are committed. Publishing
    that evidence necessarily advances HEAD by one or more commits. Freeze may
    therefore run on a descendant, but every path changed since the recorded
    run source must remain under the engineering-evidence prefix. Any source,
    protocol, notebook, script, or test change after the E0 run invalidates the
    freeze candidate and requires a new engineering run.
    """
    source_ref = decision_source.get("source_ref")
    if source_ref is not None and source_ref != provenance.get("source_ref"):
        raise ProtocolMismatch("engineering decision was produced on a different source ref")
    source_commit = decision_source.get("source_commit", decision_source.get("commit"))
    source_tree = decision_source.get("source_tree", decision_source.get("tree"))
    current_commit = provenance.get("source_commit")
    if not source_commit or not source_tree or not current_commit:
        raise ProtocolMismatch("engineering decision has incomplete source provenance")

    try:
        recorded_tree = _git("rev-parse", f"{source_commit}^{{tree}}")
    except subprocess.CalledProcessError as exc:
        raise ProtocolMismatch("engineering source commit is not present in the repository") from exc
    if recorded_tree != source_tree:
        raise ProtocolMismatch("engineering decision source tree does not match its source commit")
    if source_commit == current_commit:
        return

    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", str(source_commit), str(current_commit)],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if ancestor.returncode != 0:
        raise ProtocolMismatch("current HEAD is not descended from the engineering source commit")
    changed = [
        line.strip()
        for line in _git("diff", "--name-only", f"{source_commit}..{current_commit}").splitlines()
        if line.strip()
    ]
    unexpected = [path for path in changed if not path.startswith(ENGINEERING_EVIDENCE_PREFIX)]
    if unexpected:
        raise ProtocolMismatch(
            "source/protocol changed after engineering run; new E0 required: "
            + ", ".join(unexpected[:12])
        )


def _validate_engineering_decision(
    decision: dict,
    model_manifest: dict,
    provenance: dict,
) -> tuple[dict, dict, dict]:
    if decision.get("experiment") != "PCU-KILL-001" or decision.get("phase") != "engineering":
        raise ProtocolMismatch("engineering decision is for a different experiment or phase")
    if decision.get("scientific_evidence") is not False:
        raise ProtocolMismatch("engineering decision must explicitly declare scientific_evidence=false")
    if decision.get("formal_protocol_ready", decision.get("formal_ready")) is not True:
        raise ProtocolMismatch("engineering decision is not formal-ready")
    gates = _required_mapping(decision, "gates")
    missing = [key for key in REQUIRED_DECISION_GATES if gates.get(key) is not True]
    if missing:
        raise ProtocolMismatch(f"engineering decision gates failed or are missing: {missing}")
    positive_control = _required_mapping(decision, "positive_control")
    if positive_control.get("version") != POSITIVE_CONTROL_VERSION:
        raise ProtocolMismatch("engineering decision does not use the registered context-oracle v2")
    if positive_control.get("passed") is not True:
        raise ProtocolMismatch("engineering positive control did not pass")
    if positive_control.get("free_generation_gate") is not False:
        raise ProtocolMismatch("free generation must remain diagnostic-only for the base model")

    selected = _required_mapping(decision, "selected")
    for key in REQUIRED_SELECTED:
        if key not in selected or selected[key] in (None, "", 0):
            raise ProtocolMismatch(f"engineering decision has no selected {key}")
    if int(selected["k"]) not in (1, 2, 4, 8):
        raise ProtocolMismatch("selected k is outside the registered capacity ladder")
    if float(selected["learning_rate"]) <= 0 or int(selected["max_optimizer_steps"]) <= 0:
        raise ProtocolMismatch("engineering decision contains invalid optimizer settings")

    thresholds = _required_mapping(decision, "thresholds")
    for key in REQUIRED_THRESHOLDS:
        if key not in thresholds or thresholds[key] is None:
            raise ProtocolMismatch(f"engineering decision has no selected threshold {key}")
    decision_source = _required_mapping(decision, "source")
    _validate_engineering_source_snapshot(decision_source, provenance)

    foundation = _required_mapping(decision, "foundation")
    for key in (
        "model_repo",
        "model_revision",
        "config_sha256",
        "foundation_tensor_sha256",
        "weight_file_sha256",
        "tokenizer_sha256",
    ):
        if key not in foundation or foundation[key] in (None, "", []):
            raise ProtocolMismatch(f"engineering decision has no immutable foundation field {key}")
        if foundation[key] != model_manifest.get(key):
            raise ProtocolMismatch(f"engineering decision foundation does not match model manifest: {key}")
    if decision.get("architecture") != model_manifest.get("architecture"):
        raise ProtocolMismatch("engineering decision architecture does not match model manifest")

    allocation = _required_mapping(decision, "allocation")
    for key in ("method", "calibration_split", "calibration_sample_rule", "tie_break", "selected_k"):
        if allocation.get(key) in (None, "", []):
            raise ProtocolMismatch(f"engineering decision has no frozen allocation field {key}")
    if int(allocation["selected_k"]) != int(selected["k"]):
        raise ProtocolMismatch("allocation K does not match selected K")
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
        "foundation_tensor_sha256": model_manifest.get("foundation_tensor_sha256"),
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
    protocol["allocation"] = dict(decision["allocation"])
    positive_control = dict(protocol.get("evaluation", {}).get("positive_control", {}))
    protocol["evaluation"] = {
        "generation": {
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "max_new_tokens": 16,
        },
        "positive_control": positive_control,
        "composition_primary": "both_exact",
    }
    protocol["baseline"] = {"lora_rank": selected["lora_rank"]}
    protocol["thresholds"] = dict(thresholds)
    protocol["engineering_decision"] = {
        "path": str(args.engineering_decision),
        "sha256": sha256_file(args.engineering_decision),
        "gates": dict(gates),
        "selected": dict(selected),
        "positive_control": dict(decision["positive_control"]),
    }
    protocol["formal_execution"] = {
        "formal_seeds_executed": [],
        "scientific_evidence": False,
        "formal_execution_not_started": True,
    }
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
        "positive_control_version": POSITIVE_CONTROL_VERSION,
        "formal_execution_not_started": True,
    })
    print(json.dumps({
        "status": "FROZEN_BEFORE_FORMAL",
        "protocol_sha256": digest,
        "formal_seeds": list(FORMAL_SEEDS),
        "positive_control_version": POSITIVE_CONTROL_VERSION,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
