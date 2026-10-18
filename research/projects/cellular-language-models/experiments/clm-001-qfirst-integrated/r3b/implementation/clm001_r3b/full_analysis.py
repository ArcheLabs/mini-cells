"""Complete frozen posthoc matrix, resumable from private HF, with no training entry."""

from pathlib import Path
from datetime import datetime, timezone
import time
import uuid
from minicells_runtime.store import atomic, canonical, digest, read, ContinuationError
from .common import BASE, safe_error
from .diagnostic_schedule import full_matrix, load_plan
from . import analysis_runner as ar
from .analysis_aggregate import aggregate_tasks
from .posthoc_performance import build, markdown

PUBLIC_RUN = "33734fdd1558"
SOURCE = "a7792ef30582dfcaf324cca7e5245c8430df48b2"
INTERNAL_RUN = "r3b-a7792ef30582-ea3ea51c"
STATES = ("COMPLETE", "UNAVAILABLE", "BLOCKED", "PENDING")
REPORTS = BASE / "reports/runs" / PUBLIC_RUN


def now():
    return datetime.now(timezone.utc).isoformat()


def counts(rows):
    if (
        len(rows) != 102
        or len({r["slot_key"] for r in rows}) != 102
        or any(r["status"] not in STATES for r in rows)
    ):
        raise ContinuationError("FROZEN_MATRIX_COUNT_MISMATCH")
    return {s.lower(): sum(r["status"] == s for r in rows) for s in STATES}


def validate_input(index, meta, ref):
    expected = read(REPORTS / "checkpoint-index.json")
    if index != expected or ref != expected["checkpoint_ref"]:
        raise ContinuationError("PINNED_INPUT_INDEX_MISMATCH")
    mapping = {
        "source_sha": SOURCE,
        "run_id": INTERNAL_RUN,
        "runtime_config_sha256": index["runtime_config_sha256"],
        "assets_manifest_sha256": index["assets_manifest_sha256"],
        "tokenizer_sha256": index["tokenizer_sha256"],
        "protocol_sha256": index["protocol_sha256"],
    }
    if any(meta.get(k) != v for k, v in mapping.items()):
        raise ContinuationError("TARGET_TRAINING_IDENTITY_MISMATCH")
    actual = {
        (a, r["step"], r["revision"], r["checkpoint_sha256"])
        for a, progress in meta.get("arms", {}).items()
        for r in progress["checkpoints"]
    }
    expected_records = {
        (r["arm"], r["step"], r["event_revision"], r["sha256"]) for r in index["checkpoints"]
    }
    if actual != expected_records:
        raise ContinuationError("TARGET_CHECKPOINT_INVENTORY_MISMATCH")
    if not meta.get("complete"):
        raise ContinuationError("TARGET_TRAINING_NOT_COMPLETE")


def input_identity(index, analysis_source, plan):
    return {
        "public_run_directory": PUBLIC_RUN,
        "training_run_id": index["run_id"],
        "training_source_sha": index["source_sha"],
        "result_branch": "results/clm001-r3-a7792ef30582-33734fdd1558",
        "training_result_commit_sha": "0df519b6352412bf483945ee7e3c34e02d9ebf97",
        **{
            k: index[k]
            for k in ("runtime_config_sha256", "assets_manifest_sha256", "tokenizer_sha256")
        },
        "training_protocol_sha256": index["protocol_sha256"],
        "checkpoint_index_sha256": digest(REPORTS / "checkpoint-index.json"),
        "input_checkpoint_ref": index["checkpoint_ref"],
        "analysis_source_sha": analysis_source,
        "analysis_protocol_sha256": canonical(plan),
        "scientific_decision": False,
        "confirmation": "SEALED",
    }


def resolve(root, ref, plan, analysis_source, slot, meta):
    records = [
        r
        for r in meta.get("arms", {}).get(slot["arm"], {}).get("checkpoints", [])
        if r["step"] == slot["step"]
    ]
    if not records:
        raise FileNotFoundError("EXACT_CHECKPOINT_UNAVAILABLE")
    revision = 0 if slot["phase"] == "pre_event" else max(r["revision"] for r in records)
    return ar.identity(
        root,
        ref,
        plan,
        analysis_source,
        slot["arm"],
        slot["step"],
        revision,
        slot["phase"],
        slot["stage"],
        slot["role"],
        slot["policies"],
    )


def unavailable(error):
    code = getattr(error, "code", type(error).__name__)
    if isinstance(error, FileNotFoundError):
        code = "REQUIRED_IMMUTABLE_ASSET_MISSING"
    return code, "UNAVAILABLE" if code in (
        "REQUIRED_IMMUTABLE_ASSET_MISSING",
        "PILOT_BANK_IS_NOT_DEVELOPMENT",
        "EXACT_EVENT_PHASE_REQUIRED",
    ) else "BLOCKED"


def run_tasks(
    root,
    ref,
    plan,
    analysis_source,
    storage,
    output,
    *,
    device="cpu",
    session_seconds=10 * 3600,
    execute=ar.execute,
    meta=None,
):
    """Budget checked only between tasks. No process termination or sample changes."""
    from .runner import verify

    start = time.perf_counter()
    meta = verify(root) if meta is None else meta
    scope = {
        "training_source_sha": meta["source_sha"],
        "training_run_id": meta["run_id"],
        "analysis_source_sha": analysis_source,
        "analysis_protocol_sha256": canonical(plan),
    }
    previous = storage.coverage(scope) or {"tasks": []}
    prior = {r["slot_key"]: r for r in previous["tasks"]}
    slots = full_matrix(plan)
    rows = [
        {
            **s,
            "identity": {**scope, **s, "input_checkpoint_ref": ref},
            "identity_resolution": "LOGICAL_SLOT_ONLY",
            "status": "PENDING",
            "reason": "NOT_YET_EXECUTED",
            "device": device,
            "started_at": None,
            "ended_at": None,
            "elapsed_seconds": None,
            "sample_count": None,
            "result_sha256": None,
            "hf_revision": None,
            "required_probe_bank": "arms/"
            + s["arm"]
            + "/"
            + ("development-probe-bank" if s["role"] == "development" else "frozen-probe-bank"),
        }
        for s in slots
    ]
    results, tasks = [], []
    session = {
        "session_id": uuid.uuid4().hex,
        "started_at": now(),
        "device": device,
        "elapsed_seconds": 0.0,
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for row in rows:
        slot = {
            k: row[k] for k in ("slot_key", "arm", "step", "phase", "stage", "policies", "role")
        }
        task = None
        try:
            task = resolve(root, ref, plan, analysis_source, slot, meta)
            row["identity"] = task
            row["identity_resolution"] = "EXACT_CHECKPOINT_AND_ASSETS"
            row["task_key"] = canonical(task)
            tasks.append(task)
            lookup_start = time.perf_counter()
            cached = storage.completed(task)
            lookup_seconds = time.perf_counter() - lookup_start
            if cached is None and time.perf_counter() - start >= session_seconds:
                row["reason"] = "SESSION_BOUNDARY_RESUMABLE"
                continue
            if cached is None:
                print(
                    {
                        "analysis": row["stage"],
                        "arm": row["arm"],
                        "step": row["step"],
                        "phase": row["phase"],
                        "role": row["role"],
                        "counts": counts(rows),
                    },
                    flush=True,
                )
                row["started_at"] = now()
                result = execute(root, task, plan, device=device)
                upload_start = time.perf_counter()
                entry = storage.commit(result)
                commit_seconds = time.perf_counter() - upload_start
                verify_start = time.perf_counter()
                cached = storage.completed(task)
                if cached != result:
                    raise ContinuationError("ANALYSIS_RESULT_READBACK_FAILED")
                row["transport_timings"] = {
                    "lookup_seconds": lookup_seconds,
                    "commit_and_index_verify_seconds": commit_seconds,
                    "final_readback_seconds": time.perf_counter() - verify_start,
                }
                row["ended_at"] = now()
                row["executed_this_session"] = True
            else:
                _, index = storage.index(task)
                entry = index["tasks"][canonical(task)]
                old = prior.get(row["slot_key"], {})
                for key in ("started_at", "ended_at", "transport_timings"):
                    if key in old:
                        row[key] = old[key]
                row["started_at"] = row.get("started_at") or cached.get("started_at")
                row["ended_at"] = row.get("ended_at") or cached.get("ended_at")
                row["reused"] = True
            # COMPLETE only after authoritative pinned result + index hash verification.
            row.update(
                status="COMPLETE",
                reason="REMOTE_HASH_AND_READ_ONLY_STATE_VERIFIED",
                result_sha256=entry["sha256"],
                hf_revision=entry["hf_revision"],
                hf_path=entry["hf_path"],
                elapsed_seconds=cached["elapsed_seconds"],
                sample_count=cached["sample_count"],
                coverage_status=cached["coverage_status"],
                device=cached.get("device", device),
                gpu_stream_seconds=cached.get("gpu_stream_seconds"),
            )
            row["compute_timings"] = cached.get("timings", {})
            results.append(cached)
        except Exception as error:
            code, status = unavailable(error)
            row.update(status=status, reason=code, failure=safe_error(error), ended_at=now())
        session["elapsed_seconds"] = time.perf_counter() - start
        session["ended_at"] = now()
        revision, _ = storage.save_coverage(scope, rows, session)
        atomic(
            output / "coverage-ledger.json",
            {"tasks": rows, "counts": counts(rows), "hf_revision": revision},
        )
    session["elapsed_seconds"] = time.perf_counter() - start
    session["ended_at"] = now()
    revision, saved = storage.save_coverage(scope, rows, session)
    return tasks, results, rows, saved["sessions"], revision


def _run(output, *, device="cpu", session_seconds=10 * 3600, publish=True):
    from clm001_r3.common import source_sha
    from minicells_artifacts.hf_storage import HubBackend, HFCheckpointStore
    from .analysis_store import AnalysisStore
    from .runner import verify

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    index = read(REPORTS / "checkpoint-index.json")
    ref = index["checkpoint_ref"]
    backend = HubBackend(ref["hf_repo"])
    root = output / "immutable-training-input"
    begin = time.perf_counter()
    if not root.exists():
        # Incomplete local downloads are never accepted as a finished mount.
        import shutil

        pending = output / "download-in-progress"
        if pending.exists():
            shutil.rmtree(pending)
        HFCheckpointStore(backend).fetch_verified_checkpoint(ref, pending, expected_source=SOURCE)
        pending.rename(root)
    meta = verify(root)
    validate_input(index, meta, ref)
    download_seconds = time.perf_counter() - begin
    plan = load_plan()
    c = read(root / "runtime-config.json")
    if canonical(c["r3b"].get("diagnostic_execution")) != canonical(plan):
        raise ContinuationError("FROZEN_TRAINING_ANALYSIS_PLAN_MISMATCH")
    analysis_source = source_sha()
    reference = input_identity(index, analysis_source, plan)
    atomic(output / "analysis-input-reference.json", reference)
    tasks, results, rows, sessions, revision = run_tasks(
        root,
        ref,
        plan,
        analysis_source,
        AnalysisStore(backend),
        output,
        device=device,
        session_seconds=session_seconds,
    )
    summary = counts(rows)
    c = read(root / "runtime-config.json")
    values = aggregate_tasks(c, meta, tasks, results, rows)
    status = (
        "COMPLETE"
        if summary["complete"] == 102
        else "PARTIAL"
        if summary["complete"]
        else "BLOCKED"
    )
    values["experiment-manifest.json"].update(
        analysis_status=status,
        planned_tasks=102,
        coverage_counts=summary,
        scientific_decision=False,
        confirmation="SEALED",
    )
    values["experiment-manifest.json"]["status"] = "TRAINING_COMPLETE_ANALYSIS_" + status
    values["experiment-manifest.json"]["missing"] = [r for r in rows if r["status"] != "COMPLETE"]
    for name in (
        "metrics-summary.json",
        "gradient-audit-summary.json",
        "routing-longitudinal-summary.json",
        "specialization-summary.json",
        "statistical-contrast.json",
    ):
        values[name]["coverage_status"] = status
        values[name]["planned_tasks"] = 102
    values["analysis-input-reference.json"] = reference
    values["coverage-ledger.json"] = {
        "tasks": rows,
        "counts": summary,
        "planned": 102,
        "hf_revision": revision,
    }
    report = build(read(REPORTS / "resource-ledger.json"), rows, sessions)
    report["input_download_and_verify_seconds"] = {
        "value": download_seconds,
        "classification": "MEASURED",
        "source": "this analysis session; verified HF mount",
    }
    values["performance-summary.json"] = report
    for name, value in values.items():
        atomic(output / "public" / name, value)
    performance_md = output / "CLM-001-R3b-POSTHOC-PERFORMANCE.md"
    performance_md.write_text(markdown(report, summary), encoding="utf-8")
    from .plots import render_analysis, render_coverage

    render_analysis(values, output / "figures", plan)
    render_coverage(rows, output / "figures")
    if publish:
        from .publication import publish_documents

        publish_documents(
            root,
            values,
            scope={
                "analysis_source_sha": analysis_source,
                "analysis_protocol_sha256": canonical(plan),
            },
            attachments={
                "CLM-001-R3b-POSTHOC-PERFORMANCE.md": performance_md,
                **{p.name: p for p in (output / "figures").glob("*.png")},
            },
        )
    result = {
        "status": status,
        "counts": summary,
        "planned": 102,
        "hf_revision": revision,
        "training_updates": 0,
        "scientific_decision": False,
        "confirmation": "SEALED",
    }
    atomic(output / "result.json", result)
    print(result, flush=True)
    return result


def run(output, *, device="cpu", session_seconds=10 * 3600, publish=True):
    """Fail closed while retaining a 102-cell failure ledger, never a smaller matrix."""
    output = Path(output)
    try:
        return _run(output, device=device, session_seconds=session_seconds, publish=publish)
    except Exception as error:
        failure_started = time.perf_counter()
        from clm001_r3.common import source_sha

        output.mkdir(parents=True, exist_ok=True)
        failure = safe_error(error)
        index = read(REPORTS / "checkpoint-index.json")
        plan = load_plan()
        reference = input_identity(index, source_sha(), plan)
        old = (
            read(output / "coverage-ledger.json")
            if (output / "coverage-ledger.json").exists()
            else None
        )
        rows = (
            old["tasks"]
            if old
            else [
                {**s, "status": "PENDING", "identity": {**reference, **s}, "device": device}
                for s in full_matrix(plan)
            ]
        )
        for row in rows:
            if row["status"] != "COMPLETE":
                row.update(status="BLOCKED", reason=failure["failure_code"], ended_at=now())
        coverage = {
            "tasks": rows,
            "counts": counts(rows),
            "planned": 102,
            "status": "BLOCKED",
            "failure": failure,
            "private_hf_coverage_verified": False,
        }
        # If HF itself is unavailable, retain local/GitHub failure evidence and
        # do not claim that an authoritative private index was committed.
        try:
            from minicells_artifacts.hf_storage import HubBackend
            from .analysis_store import AnalysisStore

            scope = {
                k: reference[k]
                for k in (
                    "training_source_sha",
                    "training_run_id",
                    "analysis_source_sha",
                    "analysis_protocol_sha256",
                )
            }
            revision, _ = AnalysisStore(
                HubBackend(index["checkpoint_ref"]["hf_repo"])
            ).save_coverage(
                scope,
                rows,
                {
                    "session_id": uuid.uuid4().hex,
                    "started_at": now(),
                    "ended_at": now(),
                    "elapsed_seconds": time.perf_counter() - failure_started,
                    "status": "BLOCKED",
                },
            )
            coverage.update(hf_revision=revision, private_hf_coverage_verified=True)
        except Exception:
            pass
        atomic(output / "coverage-ledger.json", coverage)
        atomic(
            output / "analysis-failure.json",
            {"status": "BLOCKED", **failure, "training_status_modified": False},
        )
        report_path = output / "public/performance-summary.json"
        report = (
            read(report_path)
            if report_path.exists()
            else build(read(REPORTS / "resource-ledger.json"), rows, [])
        )
        report["analysis_failure"] = failure
        values = {
            "coverage-ledger.json": coverage,
            "analysis-input-reference.json": reference,
            "performance-summary.json": report,
            "experiment-manifest.json": {
                "status": "TRAINING_COMPLETE_ANALYSIS_BLOCKED",
                "coverage_counts": counts(rows),
                "scientific_decision": False,
                "confirmation": "SEALED",
                "input_verified": (output / "analysis-input-reference.json").exists(),
                "failure": failure,
            },
        }
        for name, value in values.items():
            atomic(output / "public" / name, value)
        # Metadata-only identity is separate from the failed input mount.
        publication_root = output / "failure-publication-identity"
        atomic(
            publication_root / "experiment.lock.json",
            {"source_sha": SOURCE, "run_id": INTERNAL_RUN},
        )
        if publish:
            from .publication import publish_documents

            publish_documents(
                publication_root,
                values,
                scope={
                    "analysis_source_sha": source_sha(),
                    "analysis_protocol_sha256": canonical(plan),
                },
            )
        result = {
            "status": "BLOCKED",
            "counts": counts(rows),
            "failure": failure,
            "training_updates": 0,
            "scientific_decision": False,
        }
        print(result, flush=True)
        return result
