"""Fixed Stage B matrix: original A500 parents, final-only Confirm, existing report transport."""

import copy
import csv
import importlib.metadata
import json
import os
import platform
import selectors
import subprocess
import sys
from collections import Counter
from pathlib import Path

import torch
from minicells_fast001 import analysis, data
from minicells_fast001.evaluation import evaluate
from minicells_fast002 import checkpoints, recovery
from minicells_fast002.common import BASE as FAST002_BASE
from minicells_fast002.common import config as fast002_config
from minicells_fast002.common import historical_reports
from minicells_jct001.cost import Journal
from minicells_r3k.reuse import bind

from . import assets, publication
from .common import BASE, ID, atomic, config, digest, read, require_ci, sha, source
from .runner import safe_failure
from .stageb_engine import make_engine, matrix_arms

SCOPE = "FORMAL_STAGE_B_CONTROLLED_JOINT_VALUE"


def configuration():
    c = read(BASE / "config/stage-b.json")
    review = read(BASE / "review/gate1-decision-001/GATE_DECISION.json")
    if (
        sha(c) != (BASE / "config/stage-b.sha256").read_text().strip()
        or c["stage_b"]["base_config_sha256"] != sha(config())
        or digest(BASE / "STAGE_B_PROTOCOL.zh-CN.md") != c["stage_b"]["protocol_sha256"]
        or not review["gate1_approved"]
        or review["review_basis"]["result_commit"] != c["stage_b"]["gate1_result_commit"]
    ):
        raise RuntimeError("R1_STAGEB_CONFIG_PROTOCOL_OR_GATE1_REVIEW_CHANGED")
    return c


def queue(c, index, workers):
    queues, loads = [[] for _ in range(workers)], [0] * workers
    for seed in c["seeds"]:
        for arm in matrix_arms():
            target = min(range(workers), key=lambda n: (loads[n], n))
            queues[target].append((seed, arm))
            loads[target] += {"B00": 1, "B10": 2, "B01": 3, "B11": 4}[arm[:3]]
    return queues[index]


def worker(c, root, assets_root, index, workers):
    if not torch.cuda.is_available():
        raise RuntimeError("R1_STAGEB_GPU_REQUIRED")
    ci = read(root / "CI_ACCEPTANCE.json")
    if ci["source_sha"] != source() or ci["conclusion"] != "success":
        raise RuntimeError("R1_STAGEB_WORKER_CURRENT_SOURCE_CI_REQUIRED")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    parents = read(root / "PARENT_STATE_AUDIT.json")["parents"]
    ident = read(root / "RUN_MANIFEST.json")["identity"]
    bundle = assets.refs(c, assets_root / "admission", "cuda:0")
    for seed, arm in queue(c, index, workers):
        directory = root / "private/B" / (str(seed) + "-" + arm)
        directory.mkdir(parents=True, exist_ok=True)
        public = root / "jobs" / (directory.name + ".json")
        complete = directory / "COMPLETE.json"
        child = dict(ident, seed=seed, arm=arm, phase="B")
        if complete.exists():
            result = read(complete)
            if (
                result["identity"] != child
                or digest(directory / "latest.pt") != result["checkpoint_sha256"]
            ):
                raise RuntimeError("R1_STAGEB_COMPLETED_JOB_IDENTITY_OR_CHECKPOINT_CHANGED")
            atomic(public, result, compact=True)
            continue
        parent = parents[str(seed) + "-A-" + arm[-1]]
        state = checkpoints.load(Path(parent["path"]), parent["record"])
        engine = make_engine(
            c,
            state,
            seed,
            arm,
            child,
            directory,
            root / "proposals",
            bundle,
            parent["record"],
            device="cuda:0",
        )
        stream = data.Stream(c, assets_root / "train", seed, arm[-1], "B", "cuda:0")
        latest = directory / "latest.pt"
        if latest.exists():
            engine.restore(latest, stream)
        elif len(engine.journal.events) != 1:
            raise RuntimeError("R1_STAGEB_UNCERTAIN_WORK_WITHOUT_CHECKPOINT")
        else:
            engine.save(latest, stream)
        torch.cuda.reset_peak_memory_stats()
        print("[R1 StageB]", seed, arm, "RESTORED", engine.step, flush=True)
        while engine.step < c["updates"]:
            engine.update(stream, latest, [])
            if engine.step % 25 == 0:
                print(
                    "[R1 StageB]",
                    seed,
                    arm,
                    "STEP",
                    engine.step,
                    "CE",
                    engine.curve[-1]["ce"],
                    flush=True,
                )
        eval_file = directory / "FINAL_EVALUATION.json"
        if eval_file.exists():
            stored = read(eval_file)
            if stored["identity"] != child or stored["checkpoint_sha256"] != digest(latest):
                raise RuntimeError("R1_STAGEB_FINAL_EVALUATION_IDENTITY_CHANGED")
            evaluation = stored["evaluation"]
        else:
            with engine.journal.phase("confirm_final500"):
                evaluation = evaluate(engine.model, c, assets_root / "confirm", mechanism=True)
            # Save the read-only phase watermark so resume never repeats optimizer work.
            engine.save(latest, stream)
            atomic(
                eval_file,
                {"identity": child, "checkpoint_sha256": digest(latest), "evaluation": evaluation},
            )
        budget = engine.journal.summary()
        budget.update(
            gpu_used=True,
            accounting_status="RECONCILED_MAIN_ENGINE_ACTUAL_WORK",
            claim_scope="ACTUAL_MAIN_Q_SCAN_AND_FINAL_DIAGNOSTIC_WORK",
            input_unit="LANGUAGE_EXAMPLES_WITH_EXPLICIT_TARGET_TOKEN_COUNTS",
            training_tokens=engine.tokens,
        )
        proposals = [r["proposal"] for r in engine.curve if r["proposal"] is not None]
        expected_proposals = (
            len(c["connection"]["proposal_steps"]) if arm[:3] in ("B01", "B11") else 0
        )
        candidate_calls = sum(
            Journal(path, json.loads(path.read_text().splitlines()[0])["data"])
            .summary()["optimizer_calls_by_group"]
            .get("connection", 0)
            for path in (directory / "proposals").glob("*/cost.jsonl")
        )
        if (
            engine.step != c["updates"]
            or len(proposals) != expected_proposals
            or candidate_calls != expected_proposals * c["connection"]["policy_steps"]
            or budget["uncertain_optimizer_calls"]
        ):
            raise RuntimeError("R1_STAGEB_JOB_BUDGET_MISMATCH")
        result = {
            "identity": child,
            "seed": seed,
            "arm": arm,
            "step": engine.step,
            "main_target_tokens": engine.tokens,
            "checkpoint_sha256": digest(latest),
            "parent_sha256": parent["record"]["checkpoint_sha256"],
            "parent_origin": engine.parent_origin,
            "evaluation": evaluation,
            "curve": engine.curve,
            "budget": budget,
            "candidate_optimizer_calls": candidate_calls,
            "formal_connection_commits": sum(p["accepted"] for p in proposals),
            "proposal_count": len(proposals),
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
            "gpu_used": True,
            "training_seconds": engine.training_seconds,
        }
        atomic(complete, result, compact=True)
        atomic(public, result, compact=True)
        print(
            "[R1 StageB]",
            seed,
            arm,
            "COMPLETE commits",
            result["formal_connection_commits"],
            flush=True,
        )
        for handle in engine._work_handles:
            handle.remove()
        del engine, state, stream
        torch.cuda.empty_cache()


def workers(c, root, assets_root):
    count = min(c["stage_b"]["max_workers"], torch.cuda.device_count())
    selector = selectors.DefaultSelector()
    processes = []
    try:
        for index in range(count):
            command = [
                sys.executable,
                str(BASE / "stage_b.py"),
                "--output",
                str(root),
                "--assets",
                str(assets_root),
                "--worker",
                str(index),
                "--workers",
                str(count),
            ]
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(index), PYTHONUNBUFFERED="1"),
            )
            processes.append(process)
            selector.register(process.stdout, selectors.EVENT_READ, index)
        while selector.get_map():
            for key, _ in selector.select(timeout=2):
                line = key.fileobj.readline()
                if line:
                    print("[GPU " + str(key.data) + "] " + line.rstrip(), flush=True)
                else:
                    selector.unregister(key.fileobj)
                    key.fileobj.close()
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError("R1_STAGEB_WORKER_FAILED_RETAIN_CHECKPOINTS")
        if any(p.wait() for p in processes):
            raise RuntimeError("R1_STAGEB_WORKER_FAILED_RETAIN_CHECKPOINTS")
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        selector.close()


def costs(root):
    result = {
        "main_update_attempts": 0,
        "committed_main_updates": 0,
        "cell_optimizer_calls": 0,
        "candidate_optimizer_calls": 0,
        "uncertain_optimizer_calls": 0,
        "formal_connection_commits": 0,
        "completed_jobs": len(list((root / "jobs").glob("*.json"))),
    }
    operator_work, phase_wall = Counter(), Counter()
    for path in (root / "private/B").glob("*/cost.jsonl"):
        journal = Journal(path, json.loads(path.read_text().splitlines()[0])["data"])
        summary = journal.summary()
        operator_work.update(summary["executed_operator_work"])
        phase_wall.update(summary["cpu_seconds_by_phase"])
        result["main_update_attempts"] += sum(r["kind"] == "ATTEMPT_BEGIN" for r in journal.events)
        result["committed_main_updates"] += summary["logical_updates"]
        result["cell_optimizer_calls"] += summary["optimizer_calls_by_group"].get("cell", 0)
        result["uncertain_optimizer_calls"] += summary["uncertain_optimizer_calls"]
    for path in (root / "private/B").glob("*/proposals/*/cost.jsonl"):
        summary = Journal(path, json.loads(path.read_text().splitlines()[0])["data"]).summary()
        result["candidate_optimizer_calls"] += summary["optimizer_calls_by_group"].get(
            "connection", 0
        )
        result["uncertain_optimizer_calls"] += summary["uncertain_optimizer_calls"]
    for path in (root / "jobs").glob("*.json"):
        result["formal_connection_commits"] += read(path)["formal_connection_commits"]
    result["committed_main_target_tokens"] = result["committed_main_updates"] * 512
    result["connection_commit_count_scope"] = (
        "COMPLETED_JOBS_ONLY; PARTIAL_JOB_CURVES_RETAINED_IN_CHECKPOINTS"
    )
    result["gpu_kernel_seconds"] = None
    result["executed_operator_work"] = dict(operator_work)
    result["observed_wall_seconds_by_phase"] = dict(phase_wall)
    result["wall_time_scope"] = (
        "MAIN_AND_CANDIDATE_SYNCHRONIZED_BY_ORIGINAL_ENGINE; NOT_GPU_KERNEL_TIME"
    )
    decisions = [read(path) for path in (root / "proposals").glob("*/ADMISSION_DECISION.json")]
    result["recorded_candidates"] = len(decisions)
    result["scale_checks"] = sum(len(row["attempts"]) for row in decisions)
    result["proposal_forward_and_scan_operator_work_scope"] = (
        "INCLUDED_IN_MAIN_ENGINE_CONNECTION_CANDIDATE_SEARCH_PHASES; CHILD_LEDGERS_NOT_DOUBLE_ADDED"
    )
    return result


def main(args):
    if args.publish_existing:
        return publication.existing(args.output)
    c = configuration()
    if args.worker is not None:
        return worker(c, args.output, args.assets, args.worker, args.workers)
    torch.set_num_threads(1)
    ident = {"experiment_id": ID, "scope": SCOPE, "source_sha": source(), "config_sha256": sha(c)}
    ident["run_id"] = sha(ident)[:32]
    root = args.output / ident["run_id"]
    root.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        ident,
        identity=ident,
        run_status="PREPARING",
        gate3_authorized=True,
        gpu_used=False,
        scientific_optimizer_updates=0,
        command=sys.argv,
        environment={
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "numpy": importlib.metadata.version("numpy"),
        },
    )
    atomic(root / "RUN_MANIFEST.json", manifest)
    failure = None
    try:
        if args.publish_github:
            bind(publication.previous.old.transport.preflight, _checkout=publication.checkout)(root)
        atomic(root / "CI_ACCEPTANCE.json", require_ci())
        if not torch.cuda.is_available():
            raise RuntimeError("R1_STAGEB_GPU_REQUIRED_NO_CPU_TRAINING_FALLBACK")
        manifest["gpu_used"] = True
        manifest["environment"]["gpu"] = torch.cuda.get_device_name(0)
        atomic(
            root / "REGISTRATION.json",
            {
                "identity": ident,
                "configuration": c,
                "gate1_review": read(BASE / "review/gate1-decision-001/GATE_DECISION.json"),
            },
        )
        assets.training(args.assets / "train")
        assets.admission(c, args.assets / "admission")
        # Generate exactly the old FAST002 frozen Confirm identity, not the new experiment ID.
        dataset = data.generate(fast002_config(), args.assets / "confirm")
        if sha(dataset) != sha(read(FAST002_BASE / "registration/FRESH_DATASET_MANIFEST.json")):
            raise RuntimeError("R1_STAGEB_REGISTERED_CONFIRM_IDENTITY_MISMATCH")
        atomic(
            root / "DATASET_MANIFEST.json",
            {
                "train": read(args.assets / "train/DATASET_MANIFEST.json"),
                "confirm": dataset,
                "admission": read(args.assets / "admission/ADMISSION_MANIFEST.json"),
            },
        )
        reports = historical_reports(c, args.assets / "historical-reports")
        selection = copy.deepcopy(c)
        selection["arms"] = ["A-S", "A-H"]
        paths, proof = recovery.resolve(
            selection, reports, [], args.assets / "kaggle-output", version=116
        )
        parent_rows = {}
        for key, path in sorted(paths.items()):
            record = read(reports / "jobs" / (key + ".json"))
            checkpoints.load(path, record)
            parent_rows[key] = {
                "path": str(path),
                "record": {
                    name: record[name] for name in ("identity", "seed", "arm", "checkpoint_sha256")
                },
            }
        atomic(root / "PARENT_STATE_AUDIT.json", {"recovery": proof, "parents": parent_rows})
        atomic(root / "RUN_MANIFEST.json", manifest)
        workers(c, root, args.assets)
        budget = costs(root)
        if (
            budget["completed_jobs"] != 40
            or budget["committed_main_updates"] != 20000
            or budget["main_update_attempts"] != 20000
            or budget["cell_optimizer_calls"] != 10000
            or budget["candidate_optimizer_calls"] != 1600
            or budget["uncertain_optimizer_calls"]
            or budget["recorded_candidates"] != 200
            or budget["scale_checks"] != 4400
        ):
            raise RuntimeError("R1_STAGEB_FINAL_MATRIX_BUDGET_MISMATCH")
        rows = [
            {
                "seed": seed,
                "arms": {
                    arm: read(root / "jobs" / (str(seed) + "-" + arm + ".json"))["evaluation"]
                    for arm in matrix_arms()
                },
            }
            for seed in c["seeds"]
        ]
        final = analysis.factorial(c, rows)
        final.update(
            gate1_candidates_reused=False,
            baselines="ALL_40_PATHS_NEW_MATCHED_EXECUTION_FROM_A500",
            registered_scope=c["stage_b"]["scope"],
        )
        atomic(root / "FINAL_ANALYSIS.json", final)
        manifest["run_status"] = "FORMAL_STAGE_B_COMPLETE_" + final["status"]
    except Exception as error:  # noqa: BLE001 -- publish safely retained partial evidence
        failure = error
        manifest["run_status"] = "BLOCKED_OR_PARTIAL"
        atomic(root / "FAILURE.json", safe_failure(error))
    finally:
        try:
            budget = costs(root)
        except (OSError, RuntimeError, ValueError, KeyError) as error:
            budget = {
                "status": "UNRECONCILED_PARTIAL_LEDGER_BLOCKED",
                "failure": safe_failure(error),
                "cell_optimizer_calls": None,
                "candidate_optimizer_calls": None,
            }
            failure = failure or error
            manifest["run_status"] = "BLOCKED_OR_PARTIAL"
        atomic(root / "COST_LEDGER.json", budget)
        manifest["scientific_optimizer_updates"] = (
            None
            if budget["cell_optimizer_calls"] is None
            else budget["cell_optimizer_calls"] + budget["candidate_optimizer_calls"]
        )
        atomic(
            root / "GATE_DECISION.json",
            {
                "gate1_approved": True,
                "gate3_authorized": True,
                "status": manifest["run_status"],
                "scientific_result_requires_review": True,
            },
        )
        fields = [
            "context",
            "scale",
            "kl",
            "action_change",
            "path_change",
            "parameter_delta_norm",
            "passed",
            "r3fv2_nonempty_path_pass",
        ]
        with (root / "CONNECTION_SCALE_SCAN.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for path in sorted((root / "proposals").glob("*/ADMISSION_DECISION.json")):
                for row in read(path)["attempts"]:
                    writer.writerow(dict(row, context=path.parent.name))
        (root / "FINAL_REPORT.md").write_text(
            "# R1 formal Stage B\n\nStatus: "
            + manifest["run_status"]
            + "\n\n40 fresh B paths from original A500; fixed 500 updates. Admission and final-only Confirm remain separate. "
            "No Gate1 candidate initialization. Consult FINAL_ANALYSIS.json before interpreting scientific outcomes. "
            "Controlled-generator evidence, reused Seeds/Confirm; no independent replication claim.\n"
        )
        receipt = publication.finalize(root, manifest, args.publish_github)
    print(
        json.dumps(
            {"status": manifest["run_status"], "output": str(root), "publication": receipt},
            sort_keys=True,
        ),
        flush=True,
    )
    if failure:
        raise RuntimeError("R1_STAGEB_STOPPED_SEE_RETAINED_REPORTS") from None
