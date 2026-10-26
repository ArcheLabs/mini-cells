"""One CPU Gate 0 transaction, reusable reports, no GPU scientific executor."""

import copy
import csv
import importlib.metadata
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from minicells_r3e.common import fingerprint

from . import publication
from .audit import historical_audit
from .common import BASE, ID, atomic, digest, identity, read, sha
from .data import PairedStream, unit_pools
from .engine import Engine
from .gates import status
from .measurement import (
    contributions,
    gradient_exposure,
    js_divergence,
    profile_once,
    specialization,
)
from .native import build, public_fingerprint, treatment
from .seeds import load_registry


def _write_json(path, ident, value):
    atomic(path, dict(identity=ident, **value))


def run(c, output, publish_github=False, parent_checkpoint=None):
    torch.set_num_threads(1)
    ident = identity(c)
    if parent_checkpoint is not None:
        ident = copy.deepcopy(ident)
        ident.pop("run_id")
        ident["parent_inspection_requested"] = True
        ident["run_id"] = sha(ident)[:32]
    root = Path(output) / ident["run_id"]
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "RUN_MANIFEST.json"
    if manifest_path.exists():
        existing = read(manifest_path)
        if existing["identity"] != ident:
            raise RuntimeError("CLM002_RUN_IDENTITY_CHANGED")
        if existing["run_status"] == "CPU_ENGINEERING_COMPLETE_GATE0_BLOCKED":
            for name, row in existing["artifact_files"].items():
                if (
                    digest(root / name) != row["sha256"]
                    or (root / name).stat().st_size != row["bytes"]
                ):
                    raise RuntimeError("CLM002_RETAINED_REPORT_CHANGED")
            return {
                "status": existing["run_status"],
                "output": str(root),
                "scientific_optimizer_updates": 0,
                "reused_completed": True,
                "publication": publication.publish(root) if publish_github else None,
            }
    manifest = {
        "identity": ident,
        "experiment_id": ID,
        "run_id": ident["run_id"],
        "source_sha": ident["source_sha"],
        "config_sha256": ident["config_sha256"],
        "scope": ident["scope"],
        "run_status": "CPU_ENGINEERING_STARTED",
        "scientific_optimizer_updates": 0,
        "gpu_used": False,
        "confirmation_materialized": False,
        "publication_status": "NOT_REQUESTED",
        "artifact_files": {},
        "started_at_unix": time.time(),
    }
    atomic(manifest_path, manifest)
    if publish_github:
        publication.preflight(root)
        manifest = read(manifest_path)
    try:
        architecture, boundary = historical_audit(c)
        if parent_checkpoint is not None:
            from .audit import inspect_parent

            parent_proof = inspect_parent(c, parent_checkpoint, root / "private/parent-inspection")
            architecture["parent_bytes_verified"] = True
            architecture["parent_inspection"] = parent_proof
        seeds = load_registry()
        seed = c["engineering"]["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model, r = build(c, seed, unit=True)
        base = Engine(model, r, "BASE", ident, root / "private/base-ledger.jsonl", unit=True)
        baseline = base.state()
        pools = unit_pools(c)
        data_sha = sha(
            {
                name: [{"x": r["x"].tolist(), "y": r["y"].tolist()} for r in rows]
                for name, rows in pools.items()
            }
        )
        results = {}
        profiles = {}
        matrices = {}
        restore = {}
        cost = {}
        initial_public = public_fingerprint(base.model)
        for arm in ("B00-H", "B10-H", "B01-H", "B11-H"):
            m, rc = build(c, seed, unit=True)
            tr = Engine(
                m, rc, arm, ident, root / "private" / (arm + "-ledger.jsonl"), baseline, unit=True
            )
            stream = PairedStream(pools, seed, "H", r["training"]["tokens_per_step"])
            checkpoint = root / "private" / (arm + ".pt")
            if checkpoint.exists():
                tr.restore(checkpoint, stream)
            refs = [pools["Story"][0]["x"]]
            while tr.step < rc["training"]["steps"]:
                tr.update(stream, checkpoint, refs)
            tr.save(checkpoint, stream)
            clone_model, clone_rc = build(c, seed, unit=True)
            clone = Engine(
                clone_model,
                clone_rc,
                arm,
                ident,
                root / "private" / (arm + "-ledger.jsonl"),
                baseline,
                unit=True,
            )
            restored_stream = PairedStream(pools, seed, "H", r["training"]["tokens_per_step"])
            clone.restore(checkpoint, restored_stream)
            exact = fingerprint((tr.state(), stream.state())) == fingerprint(
                (clone.state(), restored_stream.state())
            )
            if not exact or public_fingerprint(tr.model) != initial_public:
                raise RuntimeError("CLM002_UNIT_RESTORE_OR_PUBLIC_BOUNDARY_FAILED")
            flags = treatment(arm)
            moved = {
                g: any(row["parameter_delta"][g] > 0 for row in tr.curve)
                for g in ("cells", "connection")
            }
            if (moved["cells"], moved["connection"]) != flags:
                raise RuntimeError("CLM002_REQUIRED_UNIT_PARAMETER_UPDATE_MISSING")
            rows = [dict(pools[task][0], task=task) for task in ("Story", "Arithmetic")]
            with tr.journal.phase("offline_fixed_path_measurement"):
                matrix = contributions(tr.model, rows)
            with tr.journal.phase("isolated_profile"):
                profile = profile_once(tr.model, pools["Story"][0]["x"])
            with tr.journal.phase("offline_task_gradient_measurement"):
                gradients = gradient_exposure(tr.model, rows)
            matrix_summary = specialization(matrix)
            matrices[arm] = {
                "tasks": matrix,
                "specialization": matrix_summary,
                "task_gradient_exposure": gradients,
                "routing_js": js_divergence(
                    matrix["Story"]["call_counts"], matrix["Arithmetic"]["call_counts"]
                ),
            }
            profiles[arm] = profile
            restore[arm] = {"local_cpu_full_state_exact": exact, "cross_session_verified": False}
            results[arm] = {
                "scope": "SYNTHETIC_CPU_ENGINEERING_TOKEN_MODEL_NOT_SCIENTIFIC_RESULTS",
                "curve": tr.curve,
                "mask": {n: p.requires_grad for n, p in tr.model.named_parameters()},
                "shared_initial_checkpoint_sha256": tr.parent_sha,
                "public_parameters_unchanged": True,
                "task_labels_enter_model": False,
                "parameter_groups_moved": moved,
            }
            cost[arm] = tr.journal.summary()
            cost[arm]["effective_targets_attempted"] = sum(
                e["data"].get("target_tokens", 0)
                for e in tr.journal.events
                if e["kind"] == "ATTEMPT_BEGIN"
            )
            cost[arm]["effective_targets_committed"] = tr.tokens
            destination = (
                root / "artifacts" / ident["source_sha"] / data_sha / str(seed) / "H" / arm
            )
            child = dict(ident, seed=seed, task_condition="H", arm=arm)
            files = {
                "RUN_MANIFEST.json": {
                    "parent_checkpoint": "SYNTHETIC_CPU_BASELINE_NOT_HISTORICAL_PARENT",
                    "run_status": "CPU_UNIT_COMPLETE",
                    "effective_targets": tr.tokens,
                    "logical_updates": tr.step,
                    "model_sha256": fingerprint(tr.model.state_dict()),
                    "model_config_sha256": sha(rc),
                    "data_manifest_sha256": data_sha,
                    "tokenizer_identity": "SYNTHETIC_RANDOM_TOKEN_IDS_NO_TOKENIZER",
                    "historical_tokenizer_expected_sha256": c["substrate"]["tokenizer_sha256"],
                    "parent_checkpoint_sha256": tr.parent_sha,
                    "parameter_update_list": [
                        n for n, p in tr.model.named_parameters() if p.requires_grad
                    ],
                    "runtime": {
                        "python": platform.python_version(),
                        "torch": importlib.metadata.version("torch"),
                        "numpy": importlib.metadata.version("numpy"),
                        "hardware": "CPU",
                    },
                    "rng_sha256": fingerprint(
                        (tr.route_rng.get_state(), tr.cf_rng.get_state(), stream.state())
                    ),
                    "scientific_optimizer_updates": 0,
                },
                "CONFIG.json": {"configuration": c},
                "METRICS.json": {"unit_curve": tr.curve, "scientific_results": False},
                "CELL_SPECIALIZATION.json": matrices[arm],
                "ROUTING_SUMMARY.json": {"tasks": matrix, "scientific_claim": False},
                "TRAINING_COST.json": {"attempts": cost[arm], "offline_profile": profile},
                "CHECKPOINT_MANIFEST.json": dict(
                    receipt=tr.save(checkpoint, stream),
                    private_checkpoint_retained=True,
                    **restore[arm],
                ),
            }
            for name, value in files.items():
                if name != "RUN_MANIFEST.json":
                    _write_json(destination / name, child, value)
            files["RUN_MANIFEST.json"]["artifact_files"] = {
                name: {
                    "bytes": (destination / name).stat().st_size,
                    "sha256": digest(destination / name),
                }
                for name in files
                if name != "RUN_MANIFEST.json"
            }
            _write_json(destination / "RUN_MANIFEST.json", child, files["RUN_MANIFEST.json"])
            print(
                "[CLM002 CPU] "
                + arm
                + " parameter boundary, fixed-path measurement and restore passed",
                flush=True,
            )
        hashes = {
            tuple(row["input_sha256"] for row in result["curve"]) for result in results.values()
        }
        if len(hashes) != 1:
            raise RuntimeError("CLM002_FACTORIAL_INPUTS_NOT_PAIRED")
        gates = status(c)
        if architecture["parent_bytes_verified"]:
            gates["blockers"] = [
                b
                for b in gates["blockers"]
                if b != "REAL_PARENT_BYTES_AND_RESTORED_VALUES_NOT_VERIFIED_IN_THIS_EXPERIMENT"
            ]
        reports = {
            "ARCHITECTURE_AUDIT.json": architecture,
            "DATA_BOUNDARY_PROOF.json": boundary,
            "CONFIG.json": {"configuration": c},
            "SEED_REGISTRY.json": seeds,
            "CPU_ENGINEERING.json": {
                "status": "CPU_ENGINEERING_PASS",
                "arms": results,
                "restoration": restore,
                "actual_cpu_unit_optimizer_calls": sum(
                    row["completed_optimizer_calls"] for row in cost.values()
                ),
                "scientific_updates": 0,
                "gpu_used": False,
                "same_paired_data_order": True,
                "historical_parent_used_for_unit_training": False,
                "parent_inspected": architecture["parent_bytes_verified"],
            },
            "MEASUREMENT_UNIT_PROOF.json": {
                "arms": matrices,
                "scope": "RANDOM_TOKEN_CPU_FIXTURE_NO_SCIENTIFIC_SPECIALIZATION_CLAIM",
            },
            "COST_UNIT.json": {
                "arms": cost,
                "profiles": profiles,
                "gpu_time_seconds": None,
                "performance_extrapolation": "UNKNOWN_NOT_MEASURED_ON_GPU",
            },
            "GATE_STATUS.json": gates,
            "DECISION.json": {
                "status": "INCOMPLETE_OR_PROTOCOL_VIOLATION",
                "category": "PREREQUISITES_INCOMPLETE_NOT_SCIENTIFIC_NEGATIVE_RESULT",
                "engineering_status": "CPU_ENGINEERING_PASS",
                "scientific_training_started": False,
                "stage_b_authorized": False,
                "blockers": gates["blockers"],
            },
        }
        for name, value in reports.items():
            _write_json(root / name, ident, value)
        (root / "PROTOCOL.zh-CN.md").write_bytes((BASE / "PROTOCOL.zh-CN.md").read_bytes())
        for name in (
            "STAGE_A_DEVELOPMENT_REPORT.md",
            "STAGE_A_CONFIRMATION_REPORT.md",
            "STAGE_B_FACTORIAL_REPORT.md",
        ):
            (root / name).write_text(
                "# "
                + name
                + "\n\nNOT RUN. Gate 0 remains BLOCKED. Only synthetic CPU engineering fixtures were evaluated. No development/confirmation Seeds were trained and no Stage A/B scientific conclusion exists.\n"
            )
        for name, header in [
            (
                "SEED_LEVEL_RESULTS.csv",
                ["seed", "stage", "condition", "arm", "status", "I_H", "I_S", "delta_I"],
            ),
            (
                "CELL_TASK_CONTRIBUTION.csv",
                ["seed", "condition", "arm", "cell", "task", "D", "call_share"],
            ),
            ("ROUTE_STABILITY.csv", ["seed", "condition", "arm", "checkpoint", "switch_rate"]),
            (
                "COST_COMPARISON.csv",
                ["seed", "condition", "arm", "effective_targets", "optimizer_calls", "gpu_seconds"],
            ),
        ]:
            with (root / name).open("w", newline="") as handle:
                csv.writer(handle).writerow(header)
        manifest.update(
            run_status="CPU_ENGINEERING_COMPLETE_GATE0_BLOCKED",
            exit_code=0,
            finished_at_unix=time.time(),
        )
    except Exception as error:
        manifest.update(
            run_status="CPU_ENGINEERING_FAILED_GATE0_BLOCKED",
            exit_code=1,
            finished_at_unix=time.time(),
            failure_type=type(error).__name__,
        )
        for path in root.rglob("*"):
            if path.is_file() and publication.permitted(str(path.relative_to(root))):
                manifest["artifact_files"][str(path.relative_to(root))] = {
                    "bytes": path.stat().st_size,
                    "sha256": digest(path),
                }
        atomic(manifest_path, manifest)
        if publish_github:
            try:
                publication.publish(root)
            except publication.PublicationError as publication_error:
                print(
                    "[CLM002 CPU] Partial report return failed; Output retained: "
                    + type(publication_error).__name__,
                    flush=True,
                )
        raise
    for path in root.rglob("*"):
        if path.is_file() and publication.permitted(str(path.relative_to(root))):
            manifest["artifact_files"][str(path.relative_to(root))] = {
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
    atomic(manifest_path, manifest)
    receipt = publication.publish(root) if publish_github else None
    return {
        "status": manifest["run_status"],
        "output": str(root),
        "run_id": ident["run_id"],
        "scientific_optimizer_updates": 0,
        "gpu_used": False,
        "publication": receipt,
    }
