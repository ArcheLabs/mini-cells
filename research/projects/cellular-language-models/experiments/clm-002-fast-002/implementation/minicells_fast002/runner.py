"""Train-only calibration first; conditional 20-job repair, never repeat historical baseline training."""

import json
import time

import torch
from minicells_fast001 import analysis, data
from minicells_fast001.evaluation import evaluate
from minicells_jct001.connection import TorchAdapter
from minicells_jct001.cost import Journal
from minicells_r3e.common import fingerprint
from minicells_r3k.reuse import bind

from . import accounting, assets, audit, checkpoints, policy, publication
from .common import ID, atomic, config, digest, historical_reports, read, require_ci, sha, source


def close(engine):
    for hook in engine._work_handles:
        hook.remove()
    del engine
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def q_jobs(c):
    return [
        (seed, f"B{factor}-{condition}")
        for seed in c["seeds"]
        for condition in ("S", "H")
        for factor in ("01", "11")
    ]


def identity(c):
    return {
        "experiment_id": ID,
        "source_sha": source(),
        "config_sha256": sha(c),
        "historical_run_id": c["historical"]["run_id"],
        "historical_commit": c["historical"]["results_commit"],
    }


def calibration(c, old_c, root, folder, reports, paths, device, ident):
    output = root / "CALIBRATION.json"
    if output.exists():
        result = read(output)
        if result["identity"] != ident:
            raise RuntimeError("FAST002_CALIBRATION_IDENTITY_CHANGED")
        return result
    rows = []
    for seed, arm in q_jobs(c):
        key = f"{seed}-{arm}"
        receipt = root / "private/calibration" / (key + ".json")
        if receipt.exists():
            row = read(receipt)
            if row["identity"] != ident:
                raise RuntimeError("FAST002_CALIBRATION_RECEIPT_CONFLICT")
            rows.append(row)
            continue
        parent_key = f"{seed}-A-{arm[-1]}"
        parent_result = read(reports / "jobs" / (parent_key + ".json"))
        state = checkpoints.load(paths[parent_key], parent_result)
        ledger = root / "private/calibration" / (key + ".jsonl")
        if ledger.exists():
            raise RuntimeError("FAST002_INCOMPLETE_CALIBRATION_RETAINED_NO_AUTOMATIC_RETRY:" + key)
        engine = policy.make_engine(
            c,
            state,
            seed,
            arm,
            dict(ident, stage="CALIBRATION", seed=seed),
            ledger,
            device,
            parent_sha=parent_result["checkpoint_sha256"],
        )
        stream = data.Stream(old_c, folder, seed, arm[-1], "B", device)
        for _ in range(c["repair"]["calibration_stream_step"]):
            batch = stream.next()
        refs = [
            r["x"]
            for r in data.evaluation_rows(old_c, folder, device, role="train", condition=arm[-1])
        ]
        original = fingerprint(engine.model.state_dict())
        rng = fingerprint((engine.route_rng.get_state(), stream.state()))
        collector = torch.no_grad()(
            bind(
                policy.collect.__wrapped__,
                train_batches=lambda *args, batch=batch: [(r["x"], r["y"]) for r in batch],
            )
        )
        optimize = bind(
            policy.optimize,
            collect=collector,
            reference=lambda *args, refs=refs: refs,
            torch=TorchAdapter(engine.journal),
        )
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        with engine.journal.phase("train_only_scale_calibration"):
            candidate, report = optimize(engine, {})
        report.pop("proposal_optimizer_final_state")
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        if original != fingerprint(engine.model.state_dict()) or rng != fingerprint(
            (engine.route_rng.get_state(), stream.state())
        ):
            raise RuntimeError("FAST002_CALIBRATION_MUTATED_PRODUCTION_STATE")
        counts = engine.journal.summary()
        row = {
            "identity": ident,
            "seed": seed,
            "arm": arm,
            "parent_sha256": parent_result["checkpoint_sha256"],
            "proposal": report,
            "input_sha256": fingerprint([(r["x"], r["y"]) for r in batch]),
            "candidate_optimizer_calls": counts["optimizer_calls_by_group"].get("connection", 0),
            "main_updates": 0,
            "production_connection_commits": 0,
            "gpu_used": str(device).startswith("cuda"),
            "synchronized_wall_seconds": time.perf_counter() - start,
            "gate_passed": candidate is not None and report["connection_parameters_changed"],
            "scope": c["repair"]["calibration_parent"],
            "confirm_accessed": False,
        }
        if row["candidate_optimizer_calls"] != c["connection"]["policy_steps"]:
            raise RuntimeError("FAST002_CALIBRATION_OPTIMIZER_BUDGET_CHANGED")
        atomic(receipt, row)
        rows.append(row)
        close(engine)
        del candidate, state
        print("[FAST002] calibration", key, "scale=", report["chosen_scale"], flush=True)
    passed = len(rows) == 20 and all(r["gate_passed"] for r in rows)
    result = {
        "identity": ident,
        "status": "CALIBRATION_PASS" if passed else "CALIBRATION_FAILED_STOP",
        "passed": passed,
        "rows": rows,
        "main_updates": 0,
        "production_connection_commits": 0,
        "candidate_optimizer_calls": sum(r["candidate_optimizer_calls"] for r in rows),
        "confirm_accessed": False,
        "thresholds_unchanged": True,
        "next_action": "CONDITIONAL_TARGETED_REPAIR"
        if passed
        else "STOP_FIX_UPDATE_MECHANISM_OR_TRUST_METRIC_NO_THRESHOLD_RELAXATION",
    }
    atomic(output, result)
    return result


def fresh_baselines(c, root, folder, reports, paths, device, ident):
    for seed in c["seeds"]:
        for arm in ("A-S", "A-H", "B00-S", "B10-S", "B00-H", "B10-H"):
            key = f"{seed}-{arm}"
            output = root / "jobs" / (key + ".json")
            if output.exists():
                if read(output)["identity"] != ident:
                    raise RuntimeError("FAST002_BASELINE_RESULT_IDENTITY_CHANGED")
                continue
            old = read(reports / "jobs" / (key + ".json"))
            state = checkpoints.load(paths[key], old)
            engine = policy.make_engine(
                c,
                state,
                seed,
                arm,
                dict(ident, seed=seed, stage="READONLY_REEVALUATION"),
                root / "private/reevaluation" / (key + ".jsonl"),
                device,
                parent_sha=old["checkpoint_sha256"],
            )
            before = fingerprint(engine.model.state_dict())
            with engine.journal.phase("confirm_readonly"):
                evaluation = evaluate(engine.model, c, folder, mechanism=True)
            if (
                before != fingerprint(engine.model.state_dict())
                or engine.journal.summary()["actual_optimizer_attempts"]
            ):
                raise RuntimeError("FAST002_BASELINE_REEVALUATION_CHANGED_STATE")
            atomic(
                output,
                {
                    "identity": ident,
                    "seed": seed,
                    "arm": arm,
                    "historical_checkpoint_sha256": old["checkpoint_sha256"],
                    "reused_weights": True,
                    "new_main_updates": 0,
                    "optimizer_calls": 0,
                    "evaluation": evaluation,
                    "budget": accounting.summary(engine, device=device),
                },
            )
            close(engine)
            print("[FAST002] readonly fresh Confirm", key, flush=True)


def repaired_jobs(c, root, folder, reports, paths, device, ident):
    refs = {
        condition: [
            r["x"]
            for r in data.evaluation_rows(c, folder, device, role="train", condition=condition)
        ]
        for condition in ("S", "H")
    }
    for seed, arm in q_jobs(c):
        key = f"{seed}-{arm}"
        output = root / "jobs" / (key + ".json")
        directory = root / "private/repair" / key
        latest = directory / "latest.pt"
        if output.exists():
            row = read(output)
            if (
                row["identity"] != ident
                or row["step"] != 500
                or digest(latest) != row["checkpoint_sha256"]
            ):
                raise RuntimeError("FAST002_COMPLETED_REPAIR_CHANGED")
            continue
        parent_key = f"{seed}-A-{arm[-1]}"
        old = read(reports / "jobs" / (parent_key + ".json"))
        state = checkpoints.load(paths[parent_key], old)
        engine = policy.make_engine(
            c,
            state,
            seed,
            arm,
            dict(ident, seed=seed, arm=arm),
            directory / "cost.jsonl",
            device,
            parent_sha=old["checkpoint_sha256"],
        )
        stream = data.Stream(c, folder, seed, arm[-1], "B", device)
        if latest.exists():
            engine.restore(latest, stream)
        elif len(engine.journal.events) != 1:
            raise RuntimeError("FAST002_UNCHECKPOINTED_WORK_NO_AUTOMATIC_RETRAIN")
        else:
            engine.save(latest, stream)
        baseline_arm = ("B00-" if arm.startswith("B01") else "B10-") + arm[-1]
        old_curve = read(reports / "jobs" / f"{seed}-{baseline_arm}.json")["curve"]
        for row in engine.curve:
            if row["input_sha256"] != old_curve[row["step"] - 1]["input_sha256"]:
                raise RuntimeError("FAST002_HISTORICAL_PAIRED_STREAM_MISMATCH")
        while engine.step < 500:
            engine.update(stream, latest, refs[arm[-1]])
            if engine.curve[-1]["input_sha256"] != old_curve[engine.step - 1]["input_sha256"]:
                raise RuntimeError("FAST002_HISTORICAL_PAIRED_STREAM_MISMATCH")
            if engine.step % 25 == 0:
                print("[FAST002]", key, "STEP", engine.step, flush=True)
        with engine.journal.phase("confirm_final"):
            evaluation = evaluate(engine.model, c, folder, mechanism=True)
        engine.save(latest, stream)
        row = {
            "identity": ident,
            "seed": seed,
            "arm": arm,
            "step": engine.step,
            "new_main_updates": engine.step,
            "main_target_tokens": engine.tokens,
            "parent_sha256": old["checkpoint_sha256"],
            "checkpoint_sha256": digest(latest),
            "curve": engine.curve,
            "evaluation": evaluation,
            "budget": accounting.summary(engine, device=device),
        }
        atomic(output, row, compact=True)
        close(engine)


def final_analysis(c, root):
    rows, commits = [], []
    for seed in c["seeds"]:
        values = {}
        for arm in c["arms"]:
            if arm.startswith("B"):
                job = read(root / "jobs" / f"{seed}-{arm}.json")
                values[arm] = job["evaluation"]
                if arm.startswith(("B01", "B11")):
                    commits.append(
                        {
                            "seed": seed,
                            "arm": arm,
                            "committed_connection_updates": job["budget"][
                                "committed_connection_updates"
                            ],
                        }
                    )
        rows.append({"seed": seed, "arms": values})
    result = analysis.factorial(c, rows)
    delivered = all(r["committed_connection_updates"] > 0 for r in commits)
    result.update(
        connection_delivery=commits,
        scope=c["repair"]["scope"],
        historical_A_gate_reused=True,
        fresh_confirm_used=True,
        mechanism_treatment_delivered_all_jobs=delivered,
    )
    if not delivered:
        result["nominal_statistics_without_valid_mechanism_delivery"] = result["status"]
        result.update(
            status="INSUFFICIENT_EVIDENCE_CONNECTION_TREATMENT_NOT_DELIVERED",
            scientific_support=False,
        )
    atomic(root / "FINAL_ANALYSIS.json", result)
    return result


def costs(root):
    rows = []
    for path in sorted((root / "private").glob("**/*.jsonl")):
        header = json.loads(path.read_text().splitlines()[0])["data"]
        summary = Journal(path, header).summary()
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "completed_optimizer_calls": summary["completed_optimizer_calls"],
                "optimizer_calls_by_group": summary["optimizer_calls_by_group"],
                "logical_updates": summary["logical_updates"],
                "uncertain_optimizer_calls": summary["uncertain_optimizer_calls"],
                "pending_attempts": summary["pending_attempts"],
                "phase_wall_seconds": summary["cpu_seconds_by_phase"],
            }
        )
    completed = [read(p)["budget"] for p in (root / "jobs").glob("*.json")]
    return {
        "scope": "NEW_FAST002_WORK_ONLY_HISTORICAL_COSTS_SEPARATE",
        "completed_job_connection_commits": sum(
            r["committed_connection_updates"] for r in completed
        ),
        "completed_job_rejected_candidate_optimizer_calls": sum(
            r["rejected_candidate_optimizer_calls"] for r in completed
        ),
        "connection_commit_count_scope": "COMPLETED_JOBS_ONLY_LOWER_BOUND_IF_PARTIAL_JOBS_EXIST",
        "observed_phase_wall_seconds": sum(sum(r["phase_wall_seconds"].values()) for r in rows),
        "actual_optimizer_attempts": sum(
            r["completed_optimizer_calls"] + r["uncertain_optimizer_calls"] for r in rows
        ),
        "uncertain_optimizer_calls": sum(r["uncertain_optimizer_calls"] for r in rows),
        "calibration_candidate_optimizer_calls": sum(
            r["optimizer_calls_by_group"].get("connection", 0)
            for r in rows
            if r["path"].startswith("private/calibration/")
        ),
        "journals": rows,
        "actual_optimizer_calls": sum(r["completed_optimizer_calls"] for r in rows),
        "main_logical_updates": sum(r["logical_updates"] for r in rows),
        "main_target_tokens": 512 * sum(r["logical_updates"] for r in rows),
        "main_logical_update_cap": 10000,
        "calibration_candidate_optimizer_cap": 160,
        "gpu_kernel_seconds": None,
        "gpu_time_status": "NOT_MEASURED; SYNCHRONIZED_WALL_PER_JOB_SEPARATE",
    }


def main(args):
    c = config()
    if args.publish_existing:
        return publication.existing(args.output)
    ident = identity(c)
    ident["run_id"] = sha(ident)[:32]
    root = args.output / ident["run_id"]
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment_id": ID,
        "scope": "CONTROLLED_SYNTHETIC_MECHANISM_STUDY",
        "identity": ident,
        "run_id": ident["run_id"],
        "source_sha": ident["source_sha"],
        "config_sha256": sha(c),
        "run_status": "PREPARING",
        "artifact_files": {},
    }
    atomic(root / "RUN_MANIFEST.json", manifest)
    atomic(root / "REPAIR_IDENTITY.json", {"identity": ident, "configuration": c})
    report = None
    try:
        reports = historical_reports(c, args.output / "historical-input")
        report = audit.reconcile(c, reports, root)
        if not args.run_repair:
            manifest["run_status"] = "READONLY_HISTORICAL_AUDIT_COMPLETE"
        else:
            if not torch.cuda.is_available():
                raise RuntimeError("FAST002_ENABLE_GPU_AND_ATTACH_FAST001_OUTPUT")
            torch.set_num_threads(1)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            atomic(root / "CI_ACCEPTANCE.json", require_ci(ident["source_sha"]))
            if args.publish_github:
                bind(publication.old.transport.preflight, _checkout=publication.checkout)(root)
            device = "cuda:0"
            paths = checkpoints.locate(c, reports, args.checkpoint_roots)
            atomic(
                root / "CHECKPOINT_REUSE_PROOF.json",
                {
                    "status": "ALL_30_HISTORICAL_CHECKPOINT_HASHES_MATCH",
                    "files": {
                        k: {"sha256": digest(p), "bytes": p.stat().st_size}
                        for k, p in paths.items()
                    },
                    "baseline_retraining_allowed": False,
                },
            )
            old_c = assets.training(args.assets / "original")
            calibrated = calibration(
                c, old_c, root, args.assets / "original", reports, paths, device, ident
            )
            if not calibrated["passed"]:
                manifest["run_status"] = "CALIBRATION_FAILED_STOP_NO_SCIENTIFIC_REPAIR"
            else:
                dataset, proof = assets.fresh(c, args.assets / "fresh", args.assets / "original")
                atomic(root / "DATASET_MANIFEST.json", dataset)
                atomic(root / "FRESH_CONFIRM_PROOF.json", proof)
                fresh_baselines(c, root, args.assets / "fresh", reports, paths, device, ident)
                repaired_jobs(c, root, args.assets / "fresh", reports, paths, device, ident)
                manifest["run_status"] = final_analysis(c, root)["status"]
    except Exception as error:
        manifest["run_status"] = "BLOCKED_OR_PARTIAL"
        atomic(
            root / "FAILURE.json",
            {
                "type": type(error).__name__,
                "message": str(error)[:1000],
                "no_automatic_baseline_retraining": True,
            },
        )
        raise
    finally:
        budget = costs(root)
        budget["historical_audit_totals"] = report["totals"] if report is not None else None
        atomic(root / "BUDGET.json", budget)
        manifest["scientific_optimizer_updates"] = budget["actual_optimizer_calls"]
        manifest["gpu_used"] = args.run_repair and bool(budget["journals"])
        receipt = publication.finalize(root, manifest, args.publish_github)
        print(
            json.dumps(
                {"status": manifest["run_status"], "output": str(root), "publication": receipt}
            ),
            flush=True,
        )
