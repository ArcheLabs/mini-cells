"""Read-only historical evidence audit. Missing identity is UNKNOWN, never inferred PASS."""

import copy
import subprocess
from pathlib import Path

import torch
from minicells_r3e.common import fingerprint

from .common import BASE, LINEAGE, ROOT, atomic, digest, git, read, sha
from .native import build, parameter_report


def reserve_seeds():
    policy = read(BASE / "config/SEED_SELECTION_POLICY.json")
    selected = []
    collisions = []
    candidate = policy["start"]
    exclude = ":(exclude)" + str(BASE.relative_to(ROOT)) + "/**"
    while len(selected) < 13:
        result = subprocess.run(
            [
                "git",
                "grep",
                "-I",
                "-l",
                "-E",
                "(^|[^0-9])" + str(candidate) + "([^0-9]|$)",
                "--",
                ".",
                exclude,
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode not in (0, 1):
            raise RuntimeError("CLM002_SEED_SCAN_FAILED")
        if result.returncode == 0:
            collisions.append({"seed": candidate, "source_files": result.stdout.splitlines()})
        else:
            selected.append(candidate)
        candidate += 1
    tracked = git("ls-files").splitlines()
    missing = [p for p in tracked if not (ROOT / p).exists()]
    return {
        "status": "RESERVED_UNEXECUTED_METADATA_SEEDS_NOT_CONFIRMATION_REGISTRATION"
        if not missing
        else "UNKNOWN_SPARSE_COVERAGE",
        "source_sha": git("rev-parse", "HEAD"),
        "coverage": "ALL_PRESENT_GIT_TRACKED_TEXT; BINARY_CHECKPOINT_INTERNALS_NOT_SCANNED",
        "missing_tracked_files": len(missing),
        "tracked_file_count": len(tracked),
        "collisions": collisions,
        "development": selected[:3],
        "stage_a_confirmation": selected[3:8],
        "stage_b_confirmation": selected[8:],
        "task_or_model_quality_used": False,
        "all_groups_disjoint": len(set(selected)) == 13,
        "scientific_training_started": False,
    }


def historical_audit(c):
    model, r = build(c, c["engineering"]["seed"], unit=False)
    report = parameter_report(model)
    ledger = read(LINEAGE / "r3m/registration/GLOBAL_DATASET_USAGE_LEDGER.json")
    tokenizer = ROOT / c["substrate"]["tokenizer_path"]
    decisions = {}
    for name, binding in read(BASE / "config/HISTORICAL_REFERENCE_BINDINGS.json").items():
        path = ROOT / binding["path"]
        if digest(path) != binding["sha256"]:
            raise RuntimeError("CLM002_HISTORICAL_DECISION_BYTES_CHANGED:" + name)
        decisions[name] = dict(
            binding,
            decision=read(path),
            substrate="HISTORICAL_TEXTNCA_NOT_CURRENT_TOKEN_PREDICTIVE_CLM",
        )
    coverage = read(
        LINEAGE / "r3m/registration/content-isolation-001/HISTORICAL_SOURCE_COVERAGE.json"
    )
    source_checks = [
        {
            "path": row["path"],
            "expected_sha256": row["sha256"],
            "exists": (ROOT / row["path"]).is_file(),
            "sha256_matches": (ROOT / row["path"]).is_file()
            and digest(ROOT / row["path"]) == row["sha256"],
        }
        for row in ledger["parsed_metadata_files"]
    ]
    architecture = {
        "substrate_class": "minicells_r3h.routing.Model",
        "historical_config_sha256": sha(r),
        "historical_source_sha": c["substrate"]["historical_source"],
        "parent": c["substrate"],
        "parent_bytes_verified": False,
        "parameters": report,
        "cells": len(model.cells),
        "calls_per_token": model.steps,
        "context_reads_per_forward": 1,
        "self_connections": True,
        "public_components_frozen": True,
        "readout": "TIED_FROZEN_EMBEDDING_LM_HEAD",
        "cell_update": "R3E_EXECUTED_PATH_UNIFORM_CE_ADAMW; R3H_KEYED_HARD_PLUS_EPSILON_SAMPLER",
        "connection_update": "R3H_TRAINER_FREEZES_CONNECTION; R3E_Q_OPTIMIZER_AVAILABLE_BUT_CADENCE_AND_GOVERNANCE_NOT_REGISTERED",
        "actual_inherited_budget": {
            "updates": r["training"]["steps"],
            "tokens_per_step": r["training"]["tokens_per_step"],
        },
        "not_equivalent_to": ["JCT_AFFINE_126_PARAMETER_FIXTURE", "TEXTNCA_PHENOTYPE"],
        "history": decisions,
    }
    if architecture["actual_inherited_budget"] != {
        "updates": c["budget"]["updates"],
        "tokens_per_step": c["budget"]["target_tokens_per_update"],
    }:
        raise RuntimeError("CLM002_BUDGET_REQUIRES_PROTOCOL_AMENDMENT_BEFORE_TRAINING")
    boundary = {
        "status": "BLOCKED",
        "new_ranges_selected": False,
        "confirmation_materialized": False,
        "tokenizer": {
            "status": "PASS_LOCAL_ORIGINAL_BYTES"
            if digest(tokenizer) == c["substrate"]["tokenizer_sha256"]
            else "MISMATCH",
            "sha256": digest(tokenizer),
        },
        "dataset": {
            "name": r["engine"]["input"]["data_config"]["dataset"],
            "revision": r["engine"]["input"]["data_config"]["revision"],
            "split": "train",
        },
        "historical_ledger": {
            "path": str(
                (LINEAGE / "r3m/registration/GLOBAL_DATASET_USAGE_LEDGER.json").relative_to(ROOT)
            ),
            "sha256": digest(LINEAGE / "r3m/registration/GLOBAL_DATASET_USAGE_LEDGER.json"),
            "declared_intervals": len(ledger["intervals"]),
            "reservation_markers": len(ledger["reservation_markers"]),
            "parsed_metadata_files": len(ledger["parsed_metadata_files"]),
            "notes": ledger["notes"],
        },
        "previous_fragment_audit": {
            "path": str(
                (LINEAGE / "r3m/registration/content-isolation-001/REVIEW.md").relative_to(ROOT)
            ),
            "sha256": digest(LINEAGE / "r3m/registration/content-isolation-001/REVIEW.md"),
            "accepted_as_new_confirmation": False,
            "global_coverage": "INCOMPLETE_WITH_RECORDED_FRAGMENT_HITS",
        },
        "story_allocations": c["data"],
        "arithmetic_allocations": "UNREGISTERED; TEMPLATE_OPERATOR_RANGE_AND_TOKEN_WINDOW_PROOF_REQUIRED",
        "historical_coverage_counts": coverage["counts"],
        "inherited_unclosed_sources": [
            row
            for row in coverage["claims"]
            if row["coverage_class"] == "FROZEN_LATER_ROLE_RANGE_CONTAINED_ACTUAL_ASSET_UNKNOWN"
        ],
        "inherited_global_gaps": coverage["global_gaps"],
        "current_metadata_integrity": {
            "files": len(source_checks),
            "failures": [r for r in source_checks if not r["sha256_matches"]],
            "coverage_scope": "PUBLISHED_METADATA_BYTES_ONLY_NOT_FULL_CONTENT_OR_WINDOW_ISOLATION",
        },
        "raw_data_loaded": False,
        "model_quality_evaluated": False,
    }
    return architecture, boundary


def inspect_parent(c, path, output):
    # Verify known original bytes before deserializing the trusted project checkpoint.
    if digest(path) != c["substrate"]["selected_parent_checkpoint_sha256"]:
        raise RuntimeError("CLM002_PARENT_SHA256_MISMATCH_BEFORE_DESERIALIZATION")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected = c["substrate"]["historical_identity"]
    expected_ident = {
        k: expected[k] for k in ("experiment_id", "source_sha", "config_sha256", "run_id")
    }
    expected_ident.update(
        seed=c["substrate"]["selected_parent_seed"], architecture="B", training_router="H"
    )
    state = payload["state"]
    if (
        payload["identity"] != expected_ident
        or payload["state_fingerprint"] != fingerprint(state)
        or state["step"] != 300
        or state["data_cursor"] != 300 * 512
    ):
        raise RuntimeError("CLM002_PARENT_IDENTITY_OR_STATE_CHANGED")
    from minicells_r3m.anchor import runtime
    from minicells_r3me.parent import _restore_for_cpu_inspection

    old = read(LINEAGE / "r3m-e/config/exploratory.released.json")
    original_sha = fingerprint(state)
    trainer = _restore_for_cpu_inspection(
        runtime(old), copy.deepcopy(state), c["substrate"]["selected_parent_seed"], "B-H", "cpu"
    )
    proof = {
        "status": "EXACT_ORIGINAL_PARENT_CPU_INSPECTION_PASS",
        "parent_sha256": digest(path),
        "identity": payload["identity"],
        "model_values_exact": fingerprint(trainer.model.state_dict())
        == fingerprint(state["model"]),
        "cell_optimizer_values_exact": fingerprint(trainer.cell_opt.state_dict())
        == fingerprint(state["cell_optimizer"]),
        "original_state_unchanged": fingerprint(state) == original_sha,
        "cpu_inspection": trainer.cpu_inspection_evidence,
        "scientific_updates": 0,
        "gpu_used": False,
        "clm002_scientific_registration": "BLOCKED",
        "cross_session_recertified": False,
    }
    if not all(
        proof[k]
        for k in ("model_values_exact", "cell_optimizer_values_exact", "original_state_unchanged")
    ):
        raise RuntimeError("CLM002_PARENT_RESTORED_VALUES_DIFFER")
    atomic(Path(output) / "PARENT_INSPECTION.json", proof)
    return proof
