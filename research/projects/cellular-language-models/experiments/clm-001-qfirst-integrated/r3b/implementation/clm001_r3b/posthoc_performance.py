"""Evidence-labelled statistics; CUDA resource sums never masquerade as wall time."""

from collections import defaultdict
import numpy as np


def metric(value, source, classification="MEASURED", reason=None):
    return {
        "value": value,
        "classification": classification if value is not None else "NOT_MEASURED",
        "source": source,
        **({"reason": reason} if reason else {}),
    }


def percentiles(values):
    if not values:
        return None
    return {
        "count": len(values),
        **{
            k: float(np.percentile(values, q))
            for k, q in (
                ("median", 50),
                ("p90", 90),
                ("p95", 95),
                ("p99", 99),
                ("min", 0),
                ("max", 100),
            )
        },
    }


def build(ledger, coverage, sessions, source="resource-ledger.json"):
    training = ledger.get("training", {})
    uploads = ledger.get("storage", {}).get("uploads", [])
    arms = {
        arm: {
            key: metric(row.get(key), source + "/training/" + arm + "/" + key)
            for key in (
                "step",
                "gpu_training_seconds",
                "training_worker_host_seconds",
                "update_host_wall_percentiles_seconds",
                "checkpoint_serialize_seconds",
                "oracle_seconds",
                "probe_bank_capture_seconds",
            )
        }
        for arm, row in training.items()
    }
    storage = {}
    for key in (
        "seconds",
        "commit_seconds",
        "readback_seconds",
        "new_content_object_bytes",
        "payload_bytes_submitted",
    ):
        values = [
            u.get(key) if key == "seconds" else u.get("transport", {}).get(key) for u in uploads
        ]
        storage[key] = metric(
            sum(values) if values and all(v is not None for v in values) else None,
            source + "/storage/uploads",
            "DERIVED",
        )
    storage["final_logical_bundle_bytes"] = metric(
        uploads[-1].get("logical_bundle_bytes") if uploads else None, source + "/storage/uploads/-1"
    )
    storage["final_full_readback_seconds"] = metric(
        uploads[-1].get("transport", {}).get("readback_seconds")
        if uploads and uploads[-1].get("full_readback")
        else None,
        source + "/storage/uploads/-1",
    )
    grouped = defaultdict(list)
    for row in coverage:
        if row["status"] != "COMPLETE" or row.get("elapsed_seconds") is None:
            continue
        for key in ("stage", "device", "arm", "step", "role"):
            grouped[key + ":" + str(row.get(key))].append(row["elapsed_seconds"])
    timings = {
        k: metric(percentiles(v), "coverage-ledger.json/tasks", "DERIVED")
        for k, v in grouped.items()
    }
    gpu = [r.get("gpu_stream_seconds") for r in coverage if r["status"] == "COMPLETE"]
    train_gpu = [r.get("gpu_training_seconds") for r in training.values()]
    train_sum = sum(train_gpu) if train_gpu and all(v is not None for v in train_gpu) else None
    analysis_gpu = sum(gpu) if gpu and all(v is not None for v in gpu) else None
    wall = sum(s["elapsed_seconds"] for s in sessions) if sessions else None
    todo = [r for r in coverage if r["status"] in ("PENDING", "BLOCKED")]
    estimates = []
    for row in todo:
        sample = [
            r["elapsed_seconds"]
            for r in coverage
            if r["status"] == "COMPLETE"
            and r.get("elapsed_seconds") is not None
            and (r["stage"], r["role"]) == (row["stage"], row["role"])
        ]
        if len(sample) < 3:
            estimates = None
            break
        estimates.append(float(np.median(sample)))
    remaining = metric(
        sum(estimates) if estimates is not None else None,
        "coverage-ledger.json; stage/role observed medians with at least 3 tasks",
        "ESTIMATED" if todo else "DERIVED",
        reason="Retryable PENDING/BLOCKED compute only; excludes unavailable assets, I/O and hardware variation",
    )
    # Old ledger has no trusted start/end pair; never manufacture it from workers.
    return {
        "schema": "minicells.r3b.posthoc-performance.v1",
        "arms": arms,
        "stage_coverage": {
            stage: {
                state.lower(): sum(r["stage"] == stage and r["status"] == state for r in coverage)
                for state in ("COMPLETE", "UNAVAILABLE", "BLOCKED", "PENDING")
            }
            for stage in ("quality", "functional", "action")
        },
        "gpu_measurement_scope": "CUDA stream elapsed during measured updates/analysis only; excludes unmeasured semantic-event GPU work",
        "remote_ack_only_seconds": metric(
            None, source, reason="NOT_RECORDED separately from publisher roundtrip"
        ),
        "storage": storage,
        "mandatory_event_cost": metric(
            None, source, reason="NOT_RECORDED; oracle_seconds is not the complete event cost"
        ),
        "serialization_scope": "latest checkpoint only; not a sum over all serializations",
        "byte_definitions": {
            "logical_bundle": "all referenced file lengths at final revision",
            "payload": "submitted metadata plus new objects, not wire bytes",
            "new_objects": "new content-addressed objects only",
        },
        "training_critical_path_seconds": metric(
            None, source, reason="NOT_RECORDED: trustworthy end-to-end timestamps absent"
        ),
        "two_hour_verdict": "NOT_VERIFIED",
        "training_cuda_resource_seconds": metric(
            train_sum, source + "/training/*/gpu_training_seconds", "DERIVED"
        ),
        "posthoc_gpu_scope": "verified COMPLETE task compute only; failed-task GPU time is NOT_RECORDED",
        "posthoc_gpu_seconds": metric(
            analysis_gpu, "coverage-ledger.json/tasks/gpu_stream_seconds", "DERIVED"
        ),
        "posthoc_session_wall_seconds": metric(wall, "private HF coverage/sessions", "DERIVED"),
        "posthoc_session_wall_scope": "task-loop wall only; excludes initial download, aggregation and GitHub publication",
        "posthoc_end_to_end_wall_seconds": metric(
            None,
            "private HF coverage/sessions",
            reason="End-to-end publication-inclusive wall not recorded",
        ),
        "task_timing_distributions_seconds": timings,
        "task_transport": [
            {
                "slot_key": r["slot_key"],
                "timings": r.get("transport_timings", {}),
                "compute_timings": r.get("compute_timings", {}),
            }
            for r in coverage
            if r["status"] == "COMPLETE"
        ],
        "new_tasks_this_session": sum(r.get("executed_this_session", False) for r in coverage),
        "reused_tasks_this_session": sum(r.get("reused", False) for r in coverage),
        "observed_update_plus_analysis_cuda_seconds": metric(
            train_sum + analysis_gpu
            if train_sum is not None and analysis_gpu is not None
            else None,
            "training update + posthoc CUDA stream measurements (partial resource scope)",
            "DERIVED",
        ),
        "total_study_cuda_resource_seconds": metric(
            None,
            source,
            reason="Complete resource total unavailable: semantic-event GPU time not separately recorded",
        ),
        "total_elapsed_wall_seconds": metric(
            None, source, reason="No common trusted study start/end"
        ),
        "remaining_cost_seconds": remaining,
        "scientific_decision": False,
        "confirmation": "SEALED",
    }


def markdown(report, counts):
    lines = [
        "# CLM-001-R3b — Posthoc performance",
        "",
        "Training run directory: `33734fdd1558`.",
        "",
        "Training end-to-end wall time: **NOT_MEASURED**. Two-hour verdict: **NOT_VERIFIED**.",
        "",
        "CUDA resource time and serial HF costs are not end-to-end wall time.",
        "",
        "| Arm | Steps (measured) | CUDA seconds (measured) | Update P50 / P99 seconds (measured) |",
        "| --- | ---: | ---: | --- |",
    ]
    for arm, row in report["arms"].items():
        p = row["update_host_wall_percentiles_seconds"]["value"] or {}
        lines.append(
            f"| {arm} | {row['step']['value']} | {row['gpu_training_seconds']['value']} | {p.get('p50')} / {p.get('p99')} |"
        )
    lines += [
        "",
        "## Recorded storage and resource costs",
        "",
        "| Quantity | Value | Classification | Source |",
        "| --- | ---: | --- | --- |",
    ]
    entries = {
        "Training CUDA resource seconds": report["training_cuda_resource_seconds"],
        "Posthoc CUDA resource seconds": report["posthoc_gpu_seconds"],
        "Posthoc session wall seconds": report["posthoc_session_wall_seconds"],
        **report["storage"],
    }
    for name, entry in entries.items():
        lines.append(
            f"| {name} | {entry['value']} | {entry['classification']} | {entry['source']} |"
        )
    lines += [
        "",
        "Coverage: `" + str(counts) + "`.",
        "",
        "Mandatory event totals and ACK-only time: NOT_RECORDED. Oracle time is only a component.",
        "Checkpoint serialization values describe the final checkpoint, not all saves.",
        "All values, classifications, timing groups and source paths: `performance-summary.json`.",
        "Missing analysis observations are not scientific failures. Confirmation remains sealed.",
        "No training update is performed by this pipeline.",
    ]
    return "\n".join(lines) + "\n"
