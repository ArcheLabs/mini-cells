"""Read-only orchestration. No training, proposal generation or optimizer instantiation."""

from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import os
import time
import uuid
import torch
from minicells_r3e.runner import secrets
from minicells_artifacts.hf_storage import error_code
from minicells_r3f.stable_connection import Model
from minicells_r3f.data import ensure
from .common import (
    BASE,
    ID,
    atomic,
    configuration,
    parent_config,
    canonical,
    source_sha,
    fingerprint,
)
from .parent_restore import Inputs
from . import (
    branch_restore,
    parameter_diff,
    crossover,
    routing_distribution,
    trajectory_audit,
    evaluation,
    reports,
)


def models(c, arm, states, device):
    out = {}
    for key in ("official", "old", "new"):
        m = Model(c["parent_runtime"], c["seed"], arm).to(device)
        m.load_state_dict(states[key]["model"], strict=True)
        m.eval()
        out[key] = m
    return out


@torch.no_grad()
def arm_run(c, arm, inputs, backend, assets, device):
    started = time.perf_counter()
    print("[R3g]", arm, "restoring real paired branches; optimizer updates = 0", flush=True)
    restored = branch_restore.restore(inputs, c, arm)
    model = models(c, arm, restored, device)
    # Only Calibration arrays are materialized. Train/Admission/Development/Confirmation are not loaded.
    data = ensure(c["parent_runtime"], backend, assets, ("calibration",))
    from .common import digest

    asset_files = []
    for name in ("manifest.json", "tokenizer.json", "windows.json", "calibration.npy"):
        path = Path(assets) / name
        if path.exists():
            asset_files.append({"path": name, "bytes": path.stat().st_size, "sha256": digest(path)})
        elif not c.get("synthetic"):
            raise RuntimeError("CALIBRATION_ASSET_PROVENANCE_MISSING")
    hashes = {k: fingerprint(m.state_dict()) for k, m in model.items()}
    params = parameter_diff.cells(restored["old"]["model"], restored["new"]["model"])
    evolution = []
    for sequence in c["intermediate_sequences"]:
        package, receipt = inputs.checkpoint(restored["transaction_root"], sequence)
        if (
            package.get("phase") != "ADAPTING"
            or package.get("adapt_done") != sequence
            or package["old"]["step"] != c["start_step"] + sequence
            or package["new"]["step"] != c["start_step"] + sequence
        ):
            raise RuntimeError("INTERMEDIATE_PAIR_STATE_INVALID")
        if any(
            fingerprint(package["old"][key]) != fingerprint(package["new"][key])
            for key in ("data_rng", "route_rng", "data_cursor")
        ):
            raise RuntimeError("INTERMEDIATE_PAIRED_STREAMS_DIFFER")
        difference = parameter_diff.cells(package["old"]["model"], package["new"]["model"])
        evolution.append(
            {
                "step": c["start_step"] + sequence,
                "receipt": receipt,
                "all_cells_bitwise_equal": difference["all_cells_bitwise_equal"],
                "cells": [
                    {k: v for k, v in row.items() if k != "tensors"} for row in difference["cells"]
                ],
            }
        )
    params["saved_step_evolution"] = evolution
    trajectory = trajectory_audit.review(c, arm, restored)
    print(
        "[R3g]", arm, "six Calibration configurations with bounded repeat measurements", flush=True
    )
    cross = crossover.evaluate(c, arm, model, data)
    row = {
        "input": {
            "reference": inputs.reference,
            "source_identity": c["parent"],
            "asset_files": asset_files,
            "optimizer_updates": 0,
        },
        "branches": restored["verification"],
        "parameters": params,
        "crossover": cross,
        "trajectory": trajectory,
    }
    # A mismatch retains the measured numbers, but cannot produce quality-attribution claims.
    if not cross["historical_nll_reproduced"]:
        row.update(
            valid=False,
            failure={"code": "HISTORICAL_CALIBRATION_MISMATCH"},
            attribution={"attribution_status": "INVALID_OR_INCOMPLETE"},
        )
        return row
    original = restored["parent_result"]["adaptation"]
    if any(
        abs(cross["L_" + label] - original[field]) > c["historical_nll_atol"]
        for label, field in (("OO", "old_calibration_nll"), ("NN", "new_calibration_nll"))
    ):
        row.update(
            valid=False,
            failure={"code": "PRIVATE_PARENT_NLL_MISMATCH"},
            attribution={"attribution_status": "INVALID_OR_INCOMPLETE"},
        )
        return row
    row["routing"] = routing_distribution.analyze(c, model, data)
    row["attribution"] = crossover.attribution(cross, params, trajectory)
    descriptions = {
        k: v["train_inference_description"] for k, v in row["routing"]["connections"].items()
    }
    row["attribution"]["train_inference_distribution_assessment"] = descriptions
    gap = all(v["high_entropy_near_uniform"] for v in descriptions.values())
    if row["attribution"]["route_only_benefit"] or gap:
        recommendation = "PRIORITIZE_TRAIN_INFERENCE_ROUTING_ALIGNMENT_STUDY"
    elif (
        row["attribution"]["cell_adaptation_benefit"] or row["attribution"]["favorable_interaction"]
    ):
        recommendation = "PLAN_MULTISEED_COADAPTATION_REPLICATION"
    else:
        recommendation = "REEXAMINE_CONNECTION_VALUE_OBJECTIVE_AND_CELL_COMPLEMENTARITY"
    row["attribution"]["next_research_recommendation"] = recommendation
    row["attribution"]["automatic_next_training"] = False
    if any(fingerprint(m.state_dict()) != hashes[k] for k, m in model.items()):
        raise RuntimeError("AUDIT_MUTATED_MODEL_PARAMETERS")
    row["valid"] = True
    row["cost"] = {
        "wall_seconds": time.perf_counter() - started,
        "optimizer_updates": 0,
        "primary_configurations": 6,
        "bounded_quality_repeats": c["quality_repeats"],
        "unique_evaluated_model_hashes": len(
            {r["model_hash"] for r in cross["evaluations"].values()}
        ),
        "parameter_snapshots_compared": len(evolution),
        "sampling_monte_carlo": False,
    }
    return row


def failure(error):
    text = str(error)
    return {
        "code": text if text.isupper() and " " not in text else type(error).__name__,
        **error_code(error),
    }


def run(c, output, assets, device, parent_output=None):
    started = datetime.now(timezone.utc).isoformat()
    execution = uuid.uuid4().hex
    output = Path(output) / execution
    output.mkdir(parents=True, exist_ok=False)
    identity = {
        "experiment_id": ID,
        "source_sha": source_sha(),
        "config_sha256": canonical(c),
        "execution_id": execution,
        "seed": c["seed"],
        "parent_run_id": c["parent"]["run_id"],
        "parent_execution_id": c["parent"]["execution_id"],
    }
    identity["run_id"] = canonical(identity)
    atomic(output / "identity.json", identity)
    runtime = parent_config(c)
    c = {**c, "parent_runtime": runtime}
    inputs = None
    arms = {}
    backend = None
    env = evaluation.environment(device)
    try:
        secrets()  # Existing labels, no Kaggle API token and no write preflight.
        from minicells_artifacts.hf_storage import HubBackend

        backend = HubBackend(c["parent"]["hf_repo"])
        local = (
            Path(parent_output)
            if parent_output
            else Path("/kaggle/working/r3fv2-output") / c["parent"]["execution_id"]
        )
        inputs = Inputs(c, output, local=local, backend=backend)
        for arm in c["arms"]:
            try:
                arms[arm] = arm_run(c, arm, inputs, backend, Path(assets), device)
            except Exception as error:
                arms[arm] = {
                    "valid": False,
                    "failure": failure(error),
                    "attribution": {"attribution_status": "INVALID_OR_INCOMPLETE"},
                }
                print("[R3g]", arm, arms[arm]["failure"], flush=True)
        # Independently verify that the original published report contains these archived observations.
        try:
            from .public_input import read_trajectory

            public, public_ref = read_trajectory(c, output, local=local)
            for arm, row in arms.items():
                if "trajectory" not in row:
                    continue
                if fingerprint(public[arm]) != fingerprint(
                    [
                        {
                            k: v
                            for k, v in r.items()
                            if k
                            not in (
                                "cell_call_count_difference",
                                "cell_gradient_norm_difference",
                                "tokens_per_branch",
                            )
                        }
                        for r in row["trajectory"]["rows"]
                    ]
                ):
                    raise RuntimeError("ORIGINAL_PUBLIC_TRAINING_REPORT_MISMATCH")
                row["trajectory"]["public_report_comparison"] = "EXACT_MATCH"
                row["trajectory"]["public_source"] = public_ref
        except Exception as error:
            for row in arms.values():
                if "trajectory" in row:
                    row["trajectory"]["public_report_comparison"] = {
                        "status": "NOT_VERIFIED",
                        **failure(error),
                    }
                    row["valid"] = False
                    row["attribution"] = {
                        "attribution_status": "INVALID_OR_INCOMPLETE",
                        "reason": "ORIGINAL_PUBLIC_TRAINING_REPORT_NOT_VERIFIED",
                    }
                    row["failure"] = {
                        "code": "ORIGINAL_PUBLIC_TRAINING_REPORT_NOT_VERIFIED",
                        **failure(error),
                    }
        inputs.unchanged()
    except Exception as error:
        for arm in c["arms"]:
            row = arms.setdefault(arm, {})
            row.update(
                valid=False,
                failure=failure(error),
                attribution={"attribution_status": "INVALID_OR_INCOMPLETE"},
            )
    publication = {"status": "NOT_ATTEMPTED"}
    result = reports.write(output, identity, arms, env, publication, started, inputs)
    try:
        from .publication import publish

        publication = (
            publish(output)
            if os.environ.get("GITHUB_TOKEN")
            else {"status": "GITHUB_TOKEN_UNAVAILABLE_LOCAL_RETAINED"}
        )
    except Exception as error:
        publication = {"status": "LOCAL_RETAINED_REPUBLISH_ONLY", **failure(error)}
    atomic(output / "PUBLICATION.json", publication)
    # Refresh only the local manifest; do not rewrite the public pre-push reports.
    from .common import read

    manifest = read(output / "reports/RUN_MANIFEST.json")
    manifest["publication_status"] = publication["status"]
    if publication.get("remote_commit_sha"):
        manifest["remote_commit_sha"] = publication["remote_commit_sha"]
    atomic(output / "reports/RUN_MANIFEST.json", manifest)
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(output),
                "publication": publication,
                "optimizer_updates": 0,
            }
        ),
        flush=True,
    )
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(BASE / "config/frozen.json"))
    p.add_argument("--run-all", action="store_true")
    p.add_argument("--output", default="/kaggle/working/r3g-output")
    p.add_argument("--assets", default="/kaggle/working/r3g-calibration-assets")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--parent-output")
    p.add_argument("--publish-existing")
    a = p.parse_args()
    if a.publish_existing:
        secrets()
        from .publication import publish

        print(json.dumps(publish(a.publish_existing)))
    elif a.run_all:
        result = run(configuration(a.config), a.output, a.assets, a.device, a.parent_output)
        if result["status"] != "READONLY_ATTRIBUTION_AUDIT_COMPLETE":
            raise SystemExit(1)
    else:
        p.error("--run-all required")
