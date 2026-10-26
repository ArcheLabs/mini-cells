"""One launch: deterministic assets, one parent acceptance, A gate, paired B, report return."""

import importlib.metadata
import json
import os
import pickle
import platform
import selectors
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch
from minicells_clm002.measurement import profile_once

from . import analysis, data, publication
from .common import (
    BASE,
    ID,
    atomic,
    configuration,
    digest,
    identity,
    read,
    release,
    require_ci,
    sha,
    source_sha,
)
from .evaluation import evaluate
from .parent import acceptance, make_engine, original_state


def job_path(root, phase, seed, arm):
    return Path(root) / "private" / phase / (str(seed) + "-" + arm)


def completed(root, phase, seed, arm):
    return job_path(root, phase, seed, arm) / "COMPLETE.json"


def confirm(engine, c, assets):
    with engine.journal.phase("confirm_" + str(engine.step)):
        return evaluate(engine.model, c, assets, mechanism=True)


def worker(c, root, assets, checkpoint, phase, index, workers, *, device="cuda:0"):
    manifest = data.read(Path(assets) / "DATASET_MANIFEST.json")
    release(c, sha(manifest))
    ci = read(root / "CI_ACCEPTANCE.json")
    if ci["source_sha"] != source_sha() or ci["conclusion"] != "success":
        raise RuntimeError("FAST_WORKER_EXACT_SOURCE_CI_REQUIRED")
    if phase == "B" and not read(root / "STAGE_A_DECISION.json")["stage_b_allowed"]:
        raise RuntimeError("FAST_STAGE_A_NOT_PASSED_STAGE_B_FORBIDDEN")
    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        raise RuntimeError("FAST_SCIENTIFIC_WORKER_REQUIRES_GPU")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ident = read(root / "RUN_MANIFEST.json")["identity"]
    arms = [a for a in c["arms"] if a.startswith(phase)]
    jobs = [(seed, arm) for seed in c["seeds"] for arm in arms]
    queues, loads = [[] for _ in range(workers)], [0] * workers
    for seed, arm in jobs:
        target = min(range(workers), key=lambda n: (loads[n], n))
        queues[target].append((seed, arm))
        loads[target] += {"B00": 1, "B10": 2, "B01": 2, "B11": 3}.get(arm.split("-")[0], 2)
    for seed, arm in queues[index]:
        job_started = time.perf_counter()
        directory = job_path(root, phase, seed, arm)
        directory.mkdir(parents=True, exist_ok=True)
        finished = completed(root, phase, seed, arm)
        if finished.exists():
            result = read(finished)
            if result["seed"] != seed or result["arm"] != arm or result["step"] != c["updates"]:
                raise RuntimeError("FAST_COMPLETED_JOB_IDENTITY_CHANGED")
            if digest(directory / "latest.pt") != result["checkpoint_sha256"]:
                raise RuntimeError("FAST_COMPLETED_CHECKPOINT_CHANGED")
            atomic(root / "jobs" / (str(seed) + "-" + arm + ".json"), result, compact=True)
            continue
        child_ident = dict(ident, seed=seed, arm=arm, phase=phase)
        if phase == "A":
            source = original_state(c, checkpoint)
            parent_sha = c["parent"]["sha256"]
        else:
            condition = arm[-1]
            parent_directory = job_path(root, "A", seed, "A-" + condition)
            parent_result = read(parent_directory / "COMPLETE.json")
            parent_sha = digest(parent_directory / "latest.pt")
            if parent_sha != parent_result["checkpoint_sha256"]:
                raise RuntimeError("FAST_STAGE_A_PARENT_HASH_CHANGED")
            saved = torch.load(
                parent_directory / "latest.pt", map_location="cpu", weights_only=False
            )
            expected = dict(ident, seed=seed, arm="A-" + condition, phase="A")
            # ParentEngine adds the same arm field; no unrelated seed/condition state is accepted.
            if (
                saved["identity"] != expected
                or saved["state"]["step"] != c["updates"]
                or saved["state"]["fast"]["phase"] != "A"
            ):
                raise RuntimeError("FAST_STAGE_B_MUST_FORK_ITS_OWN_STAGE_A_FINAL")
            source = saved["state"]
        engine = make_engine(
            c,
            source,
            seed,
            arm,
            child_ident,
            directory / "cost.jsonl",
            device,
            parent_sha=parent_sha,
        )
        stream = data.Stream(c, assets, seed, arm[-1], phase, device)
        latest = directory / "latest.pt"
        if latest.exists():
            engine.restore(latest, stream)
        else:
            engine.save(latest, stream)
        refs = [
            r["x"] for r in data.evaluation_rows(c, assets, device, role="train", condition=arm[-1])
        ]
        evaluations = (
            read(directory / "EVALUATIONS.json")
            if (directory / "EVALUATIONS.json").exists()
            else {}
        )
        torch.cuda.reset_peak_memory_stats(torch.device(device))
        print(f"[FAST] {phase} {seed} {arm} RESTORED_STEP {engine.step}", flush=True)
        if phase == "A":
            for required in c["stage_a"]["persistence_steps"]:
                if engine.step >= required and str(required) not in evaluations:
                    if engine.step != required:
                        raise RuntimeError(
                            "FAST_REQUIRED_A_CHECKPOINT_EVALUATION_MISSING_NO_RETRAIN"
                        )
                    evaluations[str(required)] = confirm(engine, c, assets)
                    atomic(directory / "EVALUATIONS.json", evaluations)
                    engine.save(latest, stream)
        while engine.step < c["updates"]:
            engine.update(stream, latest, refs)
            if engine.step % 25 == 0:
                print(
                    f"[FAST] {phase} {seed} {arm} STEP {engine.step} CE {engine.curve[-1]['ce']:.6f}",
                    flush=True,
                )
            if phase == "A" and engine.step in c["stage_a"]["persistence_steps"]:
                evaluations[str(engine.step)] = confirm(engine, c, assets)
                atomic(directory / "EVALUATIONS.json", evaluations)
                engine.save(latest, stream)
        if phase == "B":
            evaluations[str(c["updates"])] = confirm(engine, c, assets)
            atomic(directory / "EVALUATIONS.json", evaluations)
            engine.save(latest, stream)
        peak_bytes = torch.cuda.max_memory_allocated(torch.device(device))
        one_probe = next(data.evaluation_rows(c, assets, device))["x"]
        with engine.journal.phase("inference_profile"):
            profile = profile_once(engine.model, one_probe, repeats=3)
        engine.save(latest, stream)
        result = {
            "identity": child_ident,
            "seed": seed,
            "arm": arm,
            "phase": phase,
            "step": engine.step,
            "main_target_tokens": engine.tokens,
            "parent_sha256": parent_sha,
            "parent_origin": engine.parent_origin,
            "checkpoint_sha256": digest(latest),
            "evaluations": evaluations,
            "curve": engine.curve,
            "budget": engine.journal.summary(),
            "training_seconds": engine.training_seconds,
            "peak_gpu_bytes": peak_bytes,
            "inference_profile": profile,
            "connection_proposals": engine.proposal_calls,
            "gpu_used": True,
            "phase_wall_seconds_this_session": time.perf_counter() - job_started,
            "timing_scope": "SYNCHRONIZED_MAIN_AND_Q_WALL_SECONDS_PLUS_SEPARATE_DIAGNOSTICS_IO_SESSION_WALL",
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "numpy": importlib.metadata.version("numpy"),
                "tokenizers": importlib.metadata.version("tokenizers"),
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(torch.device(device)),
            },
        }
        atomic(finished, result)
        atomic(root / "jobs" / (str(seed) + "-" + arm + ".json"), result, compact=True)
        print(f"[FAST] {phase} {seed} {arm} COMPLETE", flush=True)
        for hook in engine._work_handles:
            hook.remove()
        del engine, stream, source
        torch.cuda.empty_cache()


def workers(c, root, assets, checkpoint, phase):
    count = min(c["max_workers"], torch.cuda.device_count())
    if not count:
        raise RuntimeError("FAST_ENABLE_GPU_FOR_AUTHORIZED_SCIENCE")
    processes = []
    selector = selectors.DefaultSelector()
    for index in range(count):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(index), PYTHONUNBUFFERED="1")
        command = [
            sys.executable,
            str(BASE / "run.py"),
            "--output",
            str(root),
            "--assets",
            str(assets),
            "--checkpoint",
            str(checkpoint),
            "--worker",
            str(index),
            "--workers",
            str(count),
            "--phase",
            phase,
        ]
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, bufsize=1
        )
        selector.register(process.stdout, selectors.EVENT_READ, index)
        processes.append(process)
    while selector.get_map():
        for key, _ in selector.select(timeout=2):
            line = key.fileobj.readline()
            if line:
                print(f"[GPU {key.data}] {line.rstrip()}", flush=True)
            else:
                selector.unregister(key.fileobj)
                key.fileobj.close()
    codes = [p.wait() for p in processes]
    selector.close()
    if any(codes):
        raise RuntimeError("FAST_WORKER_FAILED_RETAIN_CHECKPOINTS:" + str(codes))


def collect_a(c, root):
    rows = []
    for seed in c["seeds"]:
        rows.append(
            {
                "seed": seed,
                **{
                    condition: read(completed(root, "A", seed, "A-" + condition))["evaluations"]
                    for condition in ("S", "H")
                },
            }
        )
    return analysis.stage_a(c, rows)


def collect_b(c, root):
    rows = []
    for seed in c["seeds"]:
        values = {
            arm: read(completed(root, "B", seed, arm))["evaluations"][str(c["updates"])]
            for arm in c["arms"]
            if arm.startswith("B")
        }
        rows.append({"seed": seed, "arms": values})
    return analysis.factorial(c, rows)


def budget(c, root):
    from minicells_jct001.cost import Journal

    results = {p.stem: read(p) for p in sorted((root / "jobs").glob("*.json"))}
    partial = []
    for ledger in sorted((root / "private").glob("*/*/cost.jsonl")):
        key = ledger.parent.name
        if key in results:
            continue
        complete = ledger.parent / "COMPLETE.json"
        if complete.exists():
            results[key] = read(complete)
            continue
        first = json.loads(ledger.read_text().splitlines()[0])["data"]
        summary = Journal(ledger, first).summary()
        checkpoint = ledger.parent / "latest.pt"
        saved = (
            torch.load(checkpoint, map_location="cpu", weights_only=False)
            if checkpoint.exists()
            else None
        )
        state = saved["state"] if saved else {}
        partial.append(
            {
                "job": key,
                "checkpointed_step": state.get("step", 0),
                "logged_logical_updates": summary["logical_updates"],
                "budget": summary,
                "training_seconds_checkpointed_lower_bound": state.get("fast", {}).get(
                    "training_seconds", 0
                ),
                "gpu_used": bool(state.get("fast", {}).get("cuda_rng")) if saved else None,
                "restoration_uncertain": saved is None
                or len(Journal(ledger, first).events) != saved["watermark"]["events"],
            }
        )
    jobs = list(results.values())
    summaries = [j["budget"] for j in jobs] + [p["budget"] for p in partial]
    logical = sum(j["step"] for j in jobs) + sum(p["logged_logical_updates"] for p in partial)
    return {
        "maximum_logical_updates": 25000,
        "maximum_main_target_tokens": 12800000,
        "completed_logical_updates": logical,
        "main_target_tokens": logical * c["targets_per_update"],
        "actual_cell_optimizer_calls": sum(
            s["optimizer_calls_by_group"].get("cell", 0) for s in summaries
        ),
        "actual_connection_optimizer_calls": sum(
            s["optimizer_calls_by_group"].get("connection", 0) for s in summaries
        ),
        "actual_optimizer_attempts": sum(s["actual_optimizer_attempts"] for s in summaries),
        "uncertain_optimizer_calls": sum(s["uncertain_optimizer_calls"] for s in summaries),
        "connection_proposals": sum(j["connection_proposals"] for j in jobs),
        "counterfactual_target_evaluations": sum(
            s["counterfactual_target_evaluations"] for s in summaries
        ),
        "training_seconds_sum_over_jobs": sum(j["training_seconds"] for j in jobs),
        "completed_jobs": len(jobs),
        "partial_jobs": partial,
        "gpu_used": True
        if any(j.get("gpu_used") for j in jobs + partial)
        else None
        if any(j.get("gpu_used") is None for j in partial)
        else False,
        "diagnostic_seconds": sum(
            e.get("diagnostic_seconds", 0) for j in jobs for e in j["evaluations"].values()
        ),
        "diagnostic_forward_calls": sum(
            e.get("diagnostic_forward_calls", 0) for j in jobs for e in j["evaluations"].values()
        ),
        "diagnostic_target_evaluations": sum(
            e.get("diagnostic_target_evaluations", 0)
            for j in jobs
            for e in j["evaluations"].values()
        ),
        "main_and_search_costs_separately_recorded": True,
        "diagnostic_wall_seconds_from_all_cost_journals": sum(
            v
            for summary in summaries
            for phase, v in summary["cpu_seconds_by_phase"].items()
            if phase.startswith("confirm_") or phase == "inference_profile"
        ),
        "logical_updates_are_not_optimizer_call_or_gpu_cost_estimates": True,
    }


def main(args):
    c = configuration()
    if args.publish_existing:
        return publication.publish_existing(args.output)
    if args.worker is not None:
        worker(c, args.output, args.assets, args.checkpoint, args.phase, args.worker, args.workers)
        return None
    print("[FAST] Generate/verify new controlled tasks; no historical dataset reads", flush=True)
    dataset = data.generate(c, args.assets)
    dataset_sha = sha(dataset)
    registered = read(BASE / "registration/DATASET_MANIFEST.json")
    if sha(registered) != dataset_sha or dataset["engineering_small_fixture"]:
        raise RuntimeError("FAST_DATASET_NOT_THE_PREREGISTERED_FULL_DATA")
    ident = identity(c, dataset_sha)
    root = args.output / ident["run_id"]
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment_id": ID,
        "scope": "CONTROLLED_SYNTHETIC_MECHANISM_STUDY",
        "identity": ident,
        "run_id": ident["run_id"],
        "source_sha": ident["source_sha"],
        "config_sha256": ident["config_sha256"],
        "dataset_sha256": dataset_sha,
        "run_status": "PREPARING",
        "scientific_optimizer_updates": 0,
        "artifact_files": {},
        "gpu_used": False,
    }
    if (root / "RUN_MANIFEST.json").exists():
        prior = read(root / "RUN_MANIFEST.json")
        if prior["identity"] != ident:
            raise RuntimeError("FAST_OUTPUT_IDENTITY_CONFLICT")
    atomic(root / "DATASET_MANIFEST.json", dataset)
    atomic(
        root / "REGISTRATION.json",
        {
            "identity": ident,
            "configuration": c,
            "scientific_review": read(BASE / "release/SCIENTIFIC_REVIEW.json"),
            "old_clm002_and_fourteen_boundaries_unchanged": True,
        },
    )
    atomic(root / "RUN_MANIFEST.json", manifest)
    try:
        if args.checkpoint is None:
            raise RuntimeError("FAST_FIXED_PARENT_CHECKPOINT_REQUIRED_USE_THE_NOTEBOOK")
        parent_proof = acceptance(c, args.checkpoint, root, ident)
        atomic(root / "PARENT_FORK_ACCEPTANCE.json", parent_proof)
        review = read(BASE / "release/SCIENTIFIC_REVIEW.json")
        if review.get("status") != "PROTOCOL_REVIEW_COMPLETE" or not args.run_science:
            for name in dataset["files"]:
                if name.endswith(".json"):
                    destination = root / "data-manifests" / name
                    destination.parent.mkdir(exist_ok=True)
                    shutil.copyfile(args.assets / name, destination)
            manifest["run_status"] = "DATA_AND_ZERO_UPDATE_PARENT_READY_FOR_SINGLE_PROTOCOL_REVIEW"
            receipt = publication.finalize(root, manifest, publish_github=args.publish_github)
            return {
                "status": manifest["run_status"],
                "publication": receipt,
                "output": str(root),
                "scientific_optimizer_updates": 0,
                "gpu_used": False,
            }
        release(c, dataset_sha)
        ci = require_ci(ident["source_sha"])
        atomic(root / "CI_ACCEPTANCE.json", ci)
        if not args.run_science:
            raise RuntimeError("FAST_SCIENCE_NOT_REQUESTED")
        if (
            not review.get("runtime_zero_update_acceptance_required")
            or parent_proof["status"] != "ZERO_UPDATE_50_FORK_ACCEPTANCE_PASS"
            or parent_proof["implementation_sha256"] != review["implementation_sha256"]
        ):
            raise RuntimeError("FAST_PARENT_ACCEPTANCE_CRITERIA_NOT_SATISFIED")
        workers(c, root, args.assets, args.checkpoint, "A")
        decision = collect_a(c, root)
        atomic(root / "STAGE_A_DECISION.json", decision)
        if decision["stage_b_allowed"]:
            workers(c, root, args.assets, args.checkpoint, "B")
            final = collect_b(c, root)
        else:
            final = {
                "status": "STAGE_A_DIFFERENTIATION_NOT_SUPPORTED",
                "stage_b_run": False,
                "scientific_support": False,
                "joint_effect_conclusion": "INSUFFICIENT_EVIDENCE_STAGE_B_NOT_RUN",
                "stage_a_conclusion": "DIFFERENTIATION_PREREQUISITE_NOT_SUPPORTED",
                "reason": "Predeclared A gate failed; no budget extension or seed replacement",
            }
        atomic(root / "FINAL_ANALYSIS.json", final)
        costs = budget(c, root)
        atomic(root / "BUDGET.json", costs)
        if costs["completed_logical_updates"] != (25000 if decision["stage_b_allowed"] else 5000):
            raise RuntimeError("FAST_COMPLETED_MATRIX_BUDGET_NOT_EXACT")
        manifest.update(
            run_status=final["status"],
            gpu_used=costs["gpu_used"],
            scientific_optimizer_updates=costs["actual_cell_optimizer_calls"]
            + costs["actual_connection_optimizer_calls"],
        )
        receipt = publication.finalize(root, manifest, publish_github=args.publish_github)
        return {
            "status": final["status"],
            "publication": receipt,
            "output": str(root),
            "budget": costs,
        }
    except publication.transport.PublicationError:
        print(
            "[FAST] Scientific/CPU results retained; publication failed. Retry --publish-existing without training.",
            flush=True,
        )
        raise
    except Exception as error:
        atomic(
            root / "FAILURE.json",
            {
                "type": type(error).__name__,
                "message": str(error)[:300],
                "status": "INCOMPLETE_OR_PROTOCOL_VIOLATION",
                "retrain_on_publication_failure": False,
            },
        )
        try:
            costs = budget(c, root)
        except (
            ValueError,
            RuntimeError,
            OSError,
            KeyError,
            TypeError,
            EOFError,
            pickle.PickleError,
        ) as accounting_error:
            costs = {
                "gpu_used": None,
                "actual_cell_optimizer_calls": None,
                "actual_connection_optimizer_calls": None,
                "status": "BUDGET_EVIDENCE_INCOMPLETE",
                "error_type": type(accounting_error).__name__,
            }
        atomic(root / "BUDGET.json", costs)
        known_updates = (
            None
            if costs["actual_cell_optimizer_calls"] is None
            else costs["actual_cell_optimizer_calls"] + costs["actual_connection_optimizer_calls"]
        )
        manifest.update(
            run_status="INCOMPLETE_OR_PROTOCOL_VIOLATION",
            gpu_used=costs["gpu_used"],
            scientific_optimizer_updates=known_updates,
        )
        try:
            publication.finalize(root, manifest, publish_github=args.publish_github)
        except publication.transport.PublicationError:
            print(
                "[FAST] Results retained locally; retry --publish-existing, never retrain",
                flush=True,
            )
        raise
