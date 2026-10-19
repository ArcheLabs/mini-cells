"""Runs new I/O orchestration with the immutable original measurement package."""

from pathlib import Path
import argparse
import os
import runpy
import subprocess
import sys
import time
from types import SimpleNamespace

ENGINE_SHA = "ed2accbc90f541325ca2f6c53229e9c81475768d"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-root", type=Path, required=True)
    parser.add_argument("--orchestrator-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cache", type=Path, default=Path("/kaggle/temp/minicells-r3b-historical-input")
    )
    parser.add_argument("--training-output-version", type=int)
    parser.add_argument("--prior-analysis-output-version", type=int)
    parser.add_argument("--candidate-version", type=int, default=65)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--session-seconds", type=float, default=6900)
    parser.add_argument("--publish-github", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from kaggle_posthoc.source_guard import verify_engine

    engine_attestation = verify_engine(args.engine_root)
    base_relative = Path(
        "research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3b"
    )
    runpy.run_path(
        str(args.engine_root / base_relative / "analyze.py"), run_name="frozen_engine_bootstrap"
    )
    # clm001_r3b is already bound to the frozen checkout. New I/O has a distinct package name.
    sys.path.insert(0, str(args.orchestrator_root / base_relative / "implementation"))
    from clm001_r3b import analysis_runner as ar, runner, full_analysis as frozen
    from clm001_r3b.diagnostic_schedule import load_plan, full_matrix
    from clm001_r3b.analysis_aggregate import aggregate_tasks
    from clm001_r3b.posthoc_performance import build, markdown
    from clm001_r3b.plots import render_analysis, render_coverage
    from minicells_runtime.store import atomic, canonical, digest, read, ContinuationError
    from kaggle_posthoc.kaggle_source import (
        KaggleOutputAPI,
        discover,
        locate_training,
        fetch,
        KERNEL,
        relative,
    )
    from kaggle_posthoc.local_store import LocalResultStore
    from kaggle_posthoc.recover import scan_raw_results, ReadOnlyHFRecovery
    from kaggle_posthoc.execution import run_tasks
    from kaggle_posthoc.reports import (
        scientific_findings,
        public_projection,
        finalize_snapshot,
        publish,
        check_publication,
    )

    for module in (ar, runner, frozen):
        if args.engine_root.resolve() not in Path(module.__file__).resolve().parents:
            raise ContinuationError("SCIENTIFIC_ENGINE_IMPORT_ORIGIN_MISMATCH")
    execution_source = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.orchestrator_root, text=True
    ).strip()
    output = args.output.resolve()
    cache = args.cache.resolve()
    if (
        output == cache
        or output in cache.parents
        or cache in output.parents
        or Path("/kaggle/working") in cache.parents
    ):
        raise ContinuationError("INPUT_OUTPUT_DIRECTORY_ISOLATION_REQUIRED")
    output.mkdir(parents=True, exist_ok=True)
    session_start = time.perf_counter()
    try:
        atomic(output / "measurement-engine-attestation.json", engine_attestation)
        api = KaggleOutputAPI()
        print(
            {
                "stage": "DISCOVER_FIXED_KAGGLE_OUTPUTS",
                "candidate_version": args.candidate_version,
                "HF_INPUT_REQUIRED": False,
            },
            flush=True,
        )
        discovery_started = time.perf_counter()
        inventories, discovery = discover(api, args.candidate_version)
        discovery["discovery_wall_seconds"] = time.perf_counter() - discovery_started
        atomic(
            output / "kaggle-version-inventory.json",
            {"versions": inventories, "discovery": discovery},
        )
        chosen = (
            [i for i in inventories if i["version_number"] == args.training_output_version]
            if args.training_output_version
            else inventories
        )
        engine = SimpleNamespace(
            REPORTS=frozen.REPORTS,
            SOURCE=frozen.SOURCE,
            INTERNAL_RUN=frozen.INTERNAL_RUN,
            ENGINE_SHA=ENGINE_SHA,
            verify=runner.verify,
            validate_input=frozen.validate_input,
            full_matrix=full_matrix,
            resolve=frozen.resolve,
            execute=ar.execute,
            unavailable=frozen.unavailable,
        )
        inv, root, meta, transfer = locate_training(api, chosen, cache, engine)
        plan = load_plan()
        if canonical(plan) != "8f9bd833ab371ce8c624693222825269889da1fde1860f8be7a73f2bb2eb90d0":
            raise ContinuationError("FROZEN_ANALYSIS_PROTOCOL_MISMATCH")
        index = read(frozen.REPORTS / "checkpoint-index.json")
        ref = index["checkpoint_ref"]  # identity compatibility only; not a download route
        provenance = {
            "transport": "KAGGLE_OUTPUT_VERIFIED",
            "kernel": KERNEL,
            "version_number": inv["version_number"],
            "inventory_sha256": inv["inventory_sha256"],
            "measurement_engine_source_sha": ENGINE_SHA,
            "execution_orchestrator_source_sha": execution_source,
        }
        input_ref = {
            **frozen.input_identity(index, ENGINE_SHA, plan),
            "storage_provenance": provenance,
            "actual_input_transport": "KAGGLE_OUTPUT_VERIFIED",
            "verified_checkpoint_records": 57,
            "asset_and_probe_verification": "VERIFIED",
            "checkpoint_index_sha256": digest(frozen.REPORTS / "checkpoint-index.json"),
            "kaggle_script_version_id": inv["metadata"].get("script_version_id"),
            "training_updates": 0,
        }
        atomic(output / "kaggle-output-input-reference.json", input_ref)
        atomic(
            output / "checkpoint-validation.json",
            {
                "checkpoint_records": index["checkpoints"],
                "verified": 57,
                "runner_verify": "PASS",
                "assets_and_probe": "PASS",
                "training_updates": 0,
            },
        )
        atomic(
            output / "kaggle-version-inventory.json",
            {"versions": inventories, "discovery": discovery},
        )
        print(
            {
                "R3B_TRAINING_IDENTITY": "VERIFIED",
                "KAGGLE_TRAINING_OUTPUT_VERSION": inv["version_number"],
                "CHECKPOINT_RECORDS_VERIFIED": "57/57",
                "PROBE_AND_ASSET_IDENTITY": "VERIFIED",
            },
            flush=True,
        )
        if args.publish_github:
            check_publication(root, execution_source, canonical(plan))
        store = LocalResultStore(output / "private-results", context=provenance)
        tasks, resolution_errors = [], []
        for slot in full_matrix(plan):
            try:
                tasks.append(frozen.resolve(root, ref, plan, ENGINE_SHA, slot, meta))
            except Exception as error:
                resolution_errors.append(
                    {
                        "slot_key": slot["slot_key"],
                        "reason": getattr(error, "code", type(error).__name__),
                    }
                )
        recovered_keys, conflicts, prior_versions, recovery_transfers = set(), [], [], []
        for old_inv in sorted(inventories, key=lambda i: i["version_number"], reverse=True):
            if (
                args.prior_analysis_output_version
                and old_inv["version_number"] != args.prior_analysis_output_version
            ):
                continue
            names = [
                r["name"]
                for r in old_inv["files"]
                if (
                    r["name"].endswith("/snapshot-manifest.json")
                    or r["name"] == "snapshot-manifest.json"
                    or r["name"].endswith("/result.json")
                    or "/tasks/" in r["name"]
                    or "/private-results/" in r["name"]
                )
                and r["name"].endswith(".json")
                and "immutable-training-input/" not in r["name"]
            ]
            if not names:
                continue
            old_root = cache / ("prior-" + str(old_inv["version_number"]))
            transfer_old = fetch(api, old_inv, names, old_root)
            recovery_transfers.append(transfer_old)
            origin = {
                "transport": "KAGGLE_SAVED_OUTPUT",
                "kernel": KERNEL,
                "version_number": old_inv["version_number"],
                "inventory_sha256": old_inv["inventory_sha256"],
            }
            # A cumulative snapshot validates its declared hashes before importing any result.
            invalid_snapshot = False
            for manifest_path in old_root.rglob("snapshot-manifest.json"):
                snapshot = read(manifest_path)
                if snapshot.get("measurement_engine_source_sha") != ENGINE_SHA:
                    continue
                try:
                    for name, sha in snapshot["files"].items():
                        if digest(manifest_path.parent / relative(name)) != sha:
                            raise ContinuationError("SAVED_ANALYSIS_SNAPSHOT_HASH_MISMATCH")
                except Exception:
                    invalid_snapshot = True
                    conflicts.append(
                        {
                            "reason": "SAVED_ANALYSIS_SNAPSHOT_HASH_MISMATCH",
                            "file_sha256": digest(manifest_path),
                        }
                    )
            if invalid_snapshot:
                continue
            reused, bad = scan_raw_results(old_root, tasks, store, origin)
            recovered_keys.update(reused)
            conflicts.extend(bad)
            prior_versions.append(old_inv["version_number"])
        hf, hf_status = None, "NOT_CONFIGURED"
        if os.environ.get("HF_TOKEN"):
            try:
                from minicells_artifacts.hf_storage import HubBackend

                scope = {
                    "analysis_source_sha": ENGINE_SHA,
                    "analysis_protocol_sha256": canonical(plan),
                    "training_run_id": meta["run_id"],
                    "training_source_sha": meta["source_sha"],
                }
                hf = ReadOnlyHFRecovery(HubBackend(ref["hf_repo"]), scope)
                hf_status = "READ_ONLY_PINNED"
            except Exception as error:
                hf_status = "OPTIONAL_UNAVAILABLE_" + getattr(error, "code", type(error).__name__)
        print(
            {
                "PRIOR_RESULTS_REUSED": len(recovered_keys),
                "KAGGLE_PRIOR_ANALYSIS_OUTPUT_VERSIONS": prior_versions,
                "OPTIONAL_HF": hf_status,
                "HF_COMMITS": 0,
            },
            flush=True,
        )
        tasks, results, rows, timing = run_tasks(
            engine,
            root,
            ref,
            plan,
            meta,
            store,
            provenance,
            device=args.device,
            session_seconds=max(0, args.session_seconds - (time.perf_counter() - session_start)),
            hf=hf,
        )
        packet = finalize_snapshot(output, provenance)
        counts = frozen.counts(rows)
        public = aggregate_tasks(read(root / "runtime-config.json"), meta, tasks, results, rows)
        durable = sum(r.get("cross_session_persistence_verified", False) for r in rows)
        insufficient = sum(r.get("coverage_status") == "INSUFFICIENT_COVERAGE" for r in rows)
        status = (
            "INSUFFICIENT_COVERAGE"
            if counts["complete"] == 102 and insufficient
            else "COMPLETE"
            if counts["complete"] == durable == 102
            else "COMPUTED_AWAITING_KAGGLE_SAVE"
            if counts["complete"] == 102
            else "PARTIAL"
        )
        public["experiment-manifest.json"].update(
            status="TRAINING_COMPLETE_ANALYSIS_" + status,
            planned_tasks=102,
            coverage_counts=counts,
            measurement_engine_source_sha=ENGINE_SHA,
            execution_orchestrator_source_sha=execution_source,
            cross_session_verified_tasks=durable,
            insufficient_coverage_tasks=insufficient,
            formal_confirmation=False,
            scientific_decision=False,
        )
        public["experiment-manifest.json"]["missing"] = [
            r for r in rows if r["status"] != "COMPLETE"
        ]
        public["checkpoint-validation.json"] = {
            "verified": 57,
            "checkpoint_sha256": [r["sha256"] for r in index["checkpoints"]],
            "runner_verify": "PASS",
            "assets_and_probe": "PASS",
            "training_updates": 0,
        }
        public["kaggle-output-input-reference.json"] = input_ref
        public["coverage-ledger.json"] = {"planned": 102, "counts": counts, "tasks": rows}
        recovery = {
            "tasks": rows,
            "kaggle_recovered": len(recovered_keys),
            "hf_recovered": timing["hf_old_results_recovered"],
            "conflicts": conflicts,
            "resolution_errors": resolution_errors,
            "source_versions": prior_versions,
            "hf_commits": 0,
        }
        atomic(output / "recovery-ledger.json", recovery)
        public["recovery-ledger.json"] = recovery
        public["kaggle-version-inventory.json"] = {"versions": inventories, "discovery": discovery}
        report = build(read(frozen.REPORTS / "resource-ledger.json"), rows, [timing])
        report["kaggle_input_transfer"] = {
            "training": transfer,
            "prior_analysis": recovery_transfers,
            "classification": "MEASURED",
        }
        report["hf_commits"] = 0
        report["private_cumulative_package_bytes"] = {
            "value": packet["package_bytes"],
            "classification": "DERIVED",
            "scope": "private result payload/receipts/index; excludes public reports",
        }
        report["kaggle_saved_output_bytes"] = {
            "value": None,
            "classification": "NOT_MEASURED",
            "reason": "Only known after Kaggle retains the version",
        }
        report["hf_read_only_seconds"] = {
            "value": timing.get("hf_read_only_seconds"),
            "classification": "MEASURED",
            "source": "current task loop",
        }
        report["model_and_probe_restore_scope"] = (
            "Frozen engine combines model/Probe restore; separate Probe-only cost NOT_MEASURED"
        )
        report["local_persistence_seconds"] = {
            "value": sum(r.get("persist_seconds", 0) for r in rows),
            "classification": "DERIVED",
            "source": "current task atomic commits",
        }
        report["output_saved_version"] = "OUTPUT_VERSION_PENDING_SAVE"
        report["kaggle_save_duration_seconds"] = {"value": None, "classification": "NOT_MEASURED"}
        public["performance-summary.json"] = report
        public["session-resource-ledger.json"] = timing
        atomic(output / "session-resource-ledger.json", timing)
        atomic(output / "analysis-resource-ledger.json", public["diagnostic-resource-ledger.json"])
        for name, value in public_projection(public).items():
            atomic(output / "public" / name, value)
        (output / "public/SCIENTIFIC-FINDINGS.md").write_text(
            scientific_findings(public, counts), encoding="utf-8"
        )
        (output / "public/CLM-001-R3b-POSTHOC-PERFORMANCE.md").write_text(
            markdown(report, counts), encoding="utf-8"
        )
        render_analysis(public, output / "figures", plan)
        render_coverage(rows, output / "figures")
        if args.publish_github:
            publish(root, public_projection(public), output, execution_source, canonical(plan))
        timing["orchestrator_wall_seconds_before_save"] = time.perf_counter() - session_start
        timing["scope"] = (
            "this session including discovery/download/analysis/publication, excluding server-side Save"
        )
        atomic(output / "session-resource-ledger.json", timing)
        finalize_snapshot(output, provenance)
        result = {
            "status": status,
            "TASK_COVERAGE": counts,
            "CURRENT_ANALYSIS_OUTPUT": "OUTPUT_VERSION_PENDING_SAVE",
            "NEXT_RUN_REQUIRED": counts["pending"] > 0
            or counts["blocked"] > 0
            or durable < counts["complete"],
            "PRIOR_RESULTS_REUSED": len(recovered_keys),
            "NEW_TASKS_COMPUTED": timing["new_tasks_computed"],
            "HF_OLD_RESULTS_READ_ONLY_RECOVERED": timing["hf_old_results_recovered"],
            "HF_COMMITS": 0,
            "TRAINING_UPDATES": 0,
            "CONFIRMATION": "SEALED",
            "SCIENTIFIC_DECISION": False,
        }
        atomic(output / "result.json", result)
        print(result, flush=True)
    except Exception as error:
        from clm001_r3b.common import safe_error

        if "inventories" in locals():
            atomic(
                output / "kaggle-version-inventory.json",
                {"versions": inventories, "discovery": discovery},
            )
        # Always retain the 102-slot denominator, including failures before input resolution.
        if not (output / "coverage-ledger.json").exists():
            blocked = [
                {
                    **slot,
                    "status": "BLOCKED",
                    "reason": safe_error(error)["failure_code"],
                    "result_data_verified": False,
                    "cross_session_persistence_verified": False,
                }
                for slot in full_matrix(load_plan())
            ]
            atomic(
                output / "coverage-ledger.json",
                {"planned": 102, "tasks": blocked, "counts": frozen.counts(blocked)},
            )
        if not (output / "checkpoint-validation.json").exists():
            atomic(
                output / "checkpoint-validation.json",
                {
                    "verified": 0,
                    "status": "NOT_VERIFIED",
                    "reason": safe_error(error)["failure_code"],
                    "training_updates": 0,
                },
            )
        atomic(
            output / "failure.json",
            {
                "status": "BLOCKED",
                **safe_error(error),
                "HF_INPUT_REQUIRED": False,
                "training_updates": 0,
            },
        )
        print({"status": "BLOCKED", **safe_error(error)}, flush=True)
        # HTTP exceptions may contain signed download URLs; never rethrow their text.
        raise ContinuationError(safe_error(error)["failure_code"]) from None


if __name__ == "__main__":
    main()
