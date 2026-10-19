"""A local-first task loop around the unchanged, byte-verified measurement engine."""

import signal
import time
from datetime import datetime, timezone
from minicells_runtime.store import canonical


def clock():
    return datetime.now(timezone.utc).isoformat()


def run_tasks(
    engine, root, ref, plan, meta, store, provenance, *, device="cpu", session_seconds=6900, hf=None
):
    start = time.perf_counter()
    hf_read_seconds = 0.0
    rows, results, tasks, new, recovered = [], [], [], 0, 0
    stop = {"signal": None}

    def request_stop(signum, frame):
        # Complete/persist the current task, then stop at the next boundary.
        stop["signal"] = signal.Signals(signum).name

    previous = signal.signal(signal.SIGTERM, request_stop)
    session = {
        "started_at": clock(),
        "training_updates": 0,
        "hf_commits": 0,
        "actual_input_transport": "KAGGLE_OUTPUT_VERIFIED",
    }
    try:
        for slot in engine.full_matrix(plan):
            row = {
                **slot,
                "status": "PENDING",
                "reason": "NOT_YET_EXECUTED",
                "device": device,
                "result_data_verified": False,
                "cross_session_persistence_verified": False,
                "storage_provenance": provenance,
            }
            rows.append(row)
        for row in rows:
            slot = {
                k: row[k] for k in ("slot_key", "arm", "step", "phase", "stage", "policies", "role")
            }
            began = time.perf_counter()
            try:
                identity = engine.resolve(root, ref, plan, engine.ENGINE_SHA, slot, meta)
                tasks.append(identity)
                row.update(identity=identity, task_key=canonical(identity))
                result = store.completed(identity)
                if result is None and hf is not None:
                    hf_started = time.perf_counter()
                    try:
                        found = hf.get(identity)
                        if found:
                            result, origin = found
                            store.commit(result, origin, persistence="HF_VERIFIED")
                            recovered += 1
                    except Exception as error:
                        # No HF availability, write or repair dependency remains.
                        row["optional_hf_recovery"] = {
                            "status": "UNAVAILABLE",
                            "reason": getattr(error, "code", type(error).__name__),
                        }
                        if getattr(getattr(error, "response", None), "status_code", None) == 429:
                            hf = None  # do not repeat a throttled optional request for every task
                    finally:
                        hf_read_seconds += time.perf_counter() - hf_started
                if result is None:
                    if stop["signal"] or time.perf_counter() - start >= session_seconds:
                        row["reason"] = stop["signal"] or "SESSION_BOUNDARY_RESUMABLE"
                        continue
                    print(
                        {
                            "arm": row["arm"],
                            "step": row["step"],
                            "stage": row["stage"],
                            "role": row["role"],
                            "new_computed": new,
                        },
                        flush=True,
                    )
                    row["started_at"] = clock()
                    result = engine.execute(root, identity, plan, device=device)
                    persist_start = time.perf_counter()
                    # The full raw observation is committed before any network operation.
                    store.commit(result, provenance, persistence="COMPUTED_LOCAL")
                    row["persist_seconds"] = time.perf_counter() - persist_start
                    row["computed_this_session"] = True
                    new += 1
                else:
                    row["reused"] = True
                entry = store.index["tasks"][canonical(identity)]
                row.update(
                    status="COMPLETE",
                    reason="LOCAL_RESULT_HASH_AND_READ_ONLY_PROOF_VERIFIED",
                    result_data_verified=True,
                    cross_session_persistence_verified=entry["cross_session_persistence_verified"],
                    persistence=entry["persistence"],
                    result_sha256=entry["sha256"],
                    result_storage_provenance=entry["storage_provenance"],
                    recovery_fact="HF_RESULT_VERIFIED"
                    if entry["persistence"] == "HF_VERIFIED"
                    else "KAGGLE_RESULT_VERIFIED"
                    if entry["persistence"] == "KAGGLE_VERSION_PERSISTED"
                    else "COMPUTED_LOCAL",
                    elapsed_seconds=result["elapsed_seconds"],
                    gpu_stream_seconds=result.get("gpu_stream_seconds"),
                    compute_timings=result.get("timings", {}),
                    sample_count=result["sample_count"],
                    coverage_status=result["coverage_status"],
                    device=result.get("device", device),
                    ended_at=clock(),
                )
                results.append(result)
            except Exception as error:
                code, status = engine.unavailable(error)
                if isinstance(error, MemoryError) or "out of memory" in str(error).lower():
                    code, status = "GPU_OOM", "BLOCKED"
                row.update(status=status, reason=code, ended_at=clock())
            row["task_loop_wall_seconds"] = time.perf_counter() - began
            session.update(
                ended_at=clock(),
                elapsed_seconds=time.perf_counter() - start,
                new_tasks_computed=new,
                hf_old_results_recovered=recovered,
            )
            store.save_coverage(rows, session)
        session.update(
            ended_at=clock(),
            elapsed_seconds=time.perf_counter() - start,
            new_tasks_computed=new,
            hf_old_results_recovered=recovered,
            hf_read_only_seconds=hf_read_seconds,
            stop_reason=stop["signal"] or "BUDGET_BOUNDARY"
            if any(r["status"] == "PENDING" for r in rows)
            else "TASK_PLAN_TERMINAL",
        )
        store.save_coverage(rows, session)
        return tasks, results, rows, session
    finally:
        signal.signal(signal.SIGTERM, previous)
