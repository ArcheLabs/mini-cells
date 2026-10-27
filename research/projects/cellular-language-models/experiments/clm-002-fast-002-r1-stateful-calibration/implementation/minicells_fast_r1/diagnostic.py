"""Authorized, bounded Gate 1 only; never commit a Connection or enter Gate 3."""

import copy
import csv
import importlib.metadata
import json
import platform
import sys
import time

import torch
from minicells_fast001.data import Stream
from minicells_fast002 import checkpoints, recovery
from minicells_fast002.common import historical_reports
from minicells_jct001.cost import Journal
from minicells_r3e.common import fingerprint
from minicells_r3k.reuse import bind

from . import accounting, assets, candidate, publication, replay, scan
from .common import BASE, ID, atomic, config, digest, read, require_ci, sha, source
from .runner import safe_failure

SCOPE = "BOUNDED_STATEFUL_GATE1_DIAGNOSTIC"


class BudgetBlocked(RuntimeError):
    def __init__(self, evidence):
        super().__init__("R1_DIAGNOSTIC_BUDGET_OR_UNCERTAIN_OPTIMIZER_BLOCKED")
        self.evidence = evidence


def authorized_config():
    c = config()
    approval = read(BASE / "config/gate1-authorization.json")
    expected = {
        "contexts": 20,
        "main_updates": 1000,
        "cell_optimizer_calls": 500,
        "candidate_optimizer_calls": 160,
        "scale_checks": 440,
        "formal_connection_commits": 0,
        "confirm_model_reads": 0,
    }
    if approval["base_config_sha256"] != sha(c) or approval["limits"] != expected:
        raise RuntimeError("R1_GATE1_AUTHORIZATION_IDENTITY_CHANGED")
    c["diagnostic"]["gate1_execution_authorized"] = True
    c["diagnostic"]["gate3_execution_authorized"] = False
    return c, approval


def totals(root):
    result = {
        "main_replay_updates": 0,
        "cell_optimizer_calls": 0,
        "candidate_optimizer_calls": 0,
        "uncertain_optimizer_calls": 0,
        "formal_connection_commits": 0,
        "completed_contexts": 0,
    }
    # Attempt ledgers remain authoritative even when a context fails before its receipt.
    for path in sorted((root / "private").glob("*/replay.jsonl")):
        identity = json.loads(path.read_text().splitlines()[0])["data"]
        j = Journal(path, identity)
        summary = j.summary()
        result["main_replay_updates"] += sum(e["kind"] == "ATTEMPT_BEGIN" for e in j.events)
        result["cell_optimizer_calls"] += summary["optimizer_calls_by_group"].get("cell", 0)
        result["uncertain_optimizer_calls"] += summary["uncertain_optimizer_calls"]
    for path in sorted((root / "private").glob("*/candidate.jsonl")):
        identity = json.loads(path.read_text().splitlines()[0])["data"]
        summary = Journal(path, identity).summary()
        result["candidate_optimizer_calls"] += summary["optimizer_calls_by_group"].get(
            "connection", 0
        )
        result["uncertain_optimizer_calls"] += summary["uncertain_optimizer_calls"]
    result["completed_contexts"] = len(list((root / "contexts").glob("*/COMPLETE.json")))
    if (
        result["main_replay_updates"] > 1000
        or result["cell_optimizer_calls"] > 500
        or result["candidate_optimizer_calls"] > 160
        or result["uncertain_optimizer_calls"]
    ):
        raise BudgetBlocked(result)
    return result


def execute_context(c, state, parent, train, bundle, root, ident, historical, *, unit=False):
    key = str(ident["seed"]) + "-" + ident["arm"]
    private, public = root / "private" / key, root / "contexts" / key
    private.mkdir(parents=True, exist_ok=True)
    public.mkdir(parents=True, exist_ok=True)
    if (public / "COMPLETE.json").exists():
        complete = read(public / "COMPLETE.json")
        if complete["identity"] != ident:
            raise RuntimeError("R1_COMPLETED_CONTEXT_IDENTITY_CHANGED")
        for name, expected in complete["files"].items():
            if digest(public / name) != expected:
                raise RuntimeError("R1_COMPLETED_CONTEXT_REPORT_CHANGED")
        return read(public / "ADMISSION_DECISION.json")
    device = "cpu" if unit else "cuda:0"
    engine = replay.make_engine(
        c,
        state,
        ident["seed"],
        ident["arm"],
        ident,
        private / "replay.jsonl",
        device=device,
        unit=unit,
        parent_sha=parent["checkpoint_sha256"],
        parent_identity=parent["identity"],
    )
    stream = Stream(c, train, ident["seed"], ident["arm"][-1], "B", device=device)
    if not unit:
        torch.cuda.reset_peak_memory_stats()
    try:
        print("[R1 Gate1] " + key + " replay to pre-proposal main50", flush=True)
        proof = replay.run(engine, stream, private, c, ident, historical_curve=historical)
        # Validate all restored rows too: a cached PAUSE cannot bypass historical matching.
        rows = engine.curve + [proof["pending_main_curve_row"]]
        if len(rows) != 50 or len(historical) < 50:
            raise RuntimeError("R1_HISTORICAL_FIFTY_ROWS_REQUIRED")
        for actual, expected in zip(rows, historical[:50]):
            replay.check_curve(actual, expected, c)
        before = replay.scientific_fingerprint(engine.state())
        pending = fingerprint(replay.pending_values(engine))
        replay.restore(engine, stream, private, c, ident)
        if (
            before != proof["scientific_state_sha256"]
            or replay.scientific_fingerprint(engine.state()) != before
            or fingerprint(replay.pending_values(engine)) != pending
        ):
            raise RuntimeError("R1_EXACT_PAUSE_RESTORE_FAILED")
        atomic(
            public / "REPLAY_AUDIT.json",
            dict(
                proof,
                exact_same_environment_restore=True,
                historical_batch_and_ce_rows=50,
                full_replay_executions=1,
                historical_gpu_bitwise_certification="NOT_CLAIMED",
            ),
        )
        cj = Journal(private / "candidate.jsonl", ident)
        cj.meter = engine.journal.meter
        package = candidate.generate(engine, cj, private, ident)
        engine.cf_rng.set_state(package["cf_after"].cpu())
        atomic(
            public / "CANDIDATE.json",
            dict(
                read(private / "CANDIDATE.json"),
                losses=package["losses"],
                original_connection_sha256=fingerprint(package["original_connection"]),
                candidate_connection_sha256=fingerprint(package["candidate_connection"]),
                optimizer_initial_sha256=fingerprint(package["optimizer_initial"]),
                optimizer_final_sha256=fingerprint(package["optimizer_final"]),
                cf_before_sha256=fingerprint(package["cf_before"]),
                cf_after_sha256=fingerprint(package["cf_after"]),
                train_input_sha256=package["train_input_sha256"],
            ),
        )
        sj = Journal(private / "scan.jsonl", ident)
        sj.meter = engine.journal.meter
        print("[R1 Gate1] " + key + " candidate ready; scan all22 scales", flush=True)
        decision = scan.run(engine, package, bundle, sj, public, c, ident)
        decision.update(seed=ident["seed"], arm=ident["arm"])
        atomic(public / "ADMISSION_DECISION.json", decision)
        cost = accounting.context(engine, cj, sj, proof)
        if (
            cost["status"] != "REPLAY_PAUSE_AND_CANDIDATE_EVENTS_RECONCILED"
            or cost["candidate_optimizer_calls"] != c["connection"]["policy_steps"]
        ):
            raise RuntimeError("R1_CONTEXT_COST_RECONCILIATION_FAILED")
        cost["peak_gpu_bytes"] = None if unit else torch.cuda.max_memory_allocated()
        cost["peak_scope"] = "THIS_CONTEXT_MODEL_REPLAY_CANDIDATE_AND_SCAN"
        atomic(public / "COST_LEDGER.json", cost)
        names = [
            "REPLAY_AUDIT.json",
            "CANDIDATE.json",
            "ADMISSION_DECISION.json",
            "COST_LEDGER.json",
            "CONNECTION_SCALE_SCAN.csv",
        ]
        atomic(
            public / "COMPLETE.json",
            {"identity": ident, "files": {name: digest(public / name) for name in names}},
        )
        print("[R1 Gate1] " + key + " COMPLETE scale=" + str(decision["chosen_scale"]), flush=True)
        return decision
    finally:
        for handle in engine._work_handles:
            handle.remove()
        if not unit:
            torch.cuda.empty_cache()


def diagnostic(args):
    if args.publish_existing:
        return publication.existing(args.output)
    c, authorization = authorized_config()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ident = {
        "experiment_id": ID,
        "source_sha": source(),
        "config_sha256": sha(c),
        "authorization_sha256": sha(authorization),
        "scope": SCOPE,
    }
    ident["run_id"] = sha(ident)[:32]
    root = args.output / ident["run_id"]
    root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    manifest = dict(
        ident,
        identity=ident,
        command=sys.argv,
        run_status="PREPARING",
        gpu_used=False,
        scientific_optimizer_updates=0,
        gate3_authorized=False,
        confirmation_accessed=False,
        environment={
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "numpy": importlib.metadata.version("numpy"),
        },
    )
    atomic(root / "RUN_MANIFEST.json", manifest)
    failure = None
    decisions = []
    try:
        if args.publish_github:
            bind(publication.previous.old.transport.preflight, _checkout=publication.checkout)(root)
        atomic(root / "CI_ACCEPTANCE.json", require_ci())
        if not torch.cuda.is_available():
            raise RuntimeError("R1_GATE1_REQUIRES_GPU_NO_CPU_TRAINING_FALLBACK")
        manifest["gpu_used"] = True
        manifest["environment"]["gpu"] = torch.cuda.get_device_name(0)
        atomic(
            root / "REGISTRATION.json",
            {"identity": ident, "configuration": c, "authorization": authorization},
        )
        atomic(root / "DATASET_MANIFEST.json", read(BASE / "registration/DATASET_MANIFEST.json"))
        reports = historical_reports(c, args.assets / "historical-reports")
        assets.training(args.assets / "train")
        bundle = assets.refs(c, args.assets / "admission", "cuda:0")
        selection = copy.deepcopy(c)
        selection["arms"] = ["A-S", "A-H"]
        paths, proof = recovery.resolve(
            selection, reports, [], args.assets / "kaggle-output", version=116
        )
        atomic(
            root / "PARENT_STATE_AUDIT.json",
            {
                "recovery": proof,
                "parents": [
                    {"key": key, "sha256": digest(path)} for key, path in sorted(paths.items())
                ],
            },
        )
        for seed in c["seeds"]:
            for condition in ("S", "H"):
                parent = read(reports / "jobs" / (str(seed) + "-A-" + condition + ".json"))
                state = checkpoints.load(paths[str(seed) + "-A-" + condition], parent)
                for arm in ("B01-" + condition, "B11-" + condition):
                    totals(root)
                    historical = read(reports / "jobs" / (str(seed) + "-" + arm + ".json"))
                    context_identity = dict(
                        ident, seed=seed, arm=arm, parent_sha256=parent["checkpoint_sha256"]
                    )
                    manifest["current_context"] = context_identity
                    atomic(root / "RUN_MANIFEST.json", manifest)
                    decisions.append(
                        execute_context(
                            c,
                            state,
                            parent,
                            args.assets / "train",
                            bundle,
                            root,
                            context_identity,
                            historical["curve"],
                        )
                    )
                    atomic(root / "COST_LEDGER.json", totals(root))
                del state
        aggregate = scan.aggregate(c, decisions)
        atomic(root / "ADMISSION_DECISION.json", aggregate)
        if not aggregate["complete_contexts"]:
            raise RuntimeError("R1_INCOMPLETE_CONTEXT_MATRIX")
        manifest["run_status"] = "GATE1_DIAGNOSTIC_COMPLETE_INDEPENDENT_REVIEW_REQUIRED"
    except Exception as error:  # noqa: BLE001 -- retain bounded run evidence before return
        failure = error
        manifest["run_status"] = "BLOCKED_OR_PARTIAL"
        atomic(root / "FAILURE.json", safe_failure(error))
    finally:
        try:
            cost = totals(root)
        except Exception as error:  # noqa: BLE001 -- retain bounded run evidence before return
            cost = dict(
                getattr(error, "evidence", {}),
                status="UNRECONCILED_BLOCKED",
                failure=safe_failure(error),
            )
            manifest["run_status"] = "BLOCKED_OR_PARTIAL"
            failure = failure or error
        cost.update(
            formal_connection_commits=0,
            confirm_model_reads=0,
            wall_seconds_this_session=time.perf_counter() - started,
        )
        atomic(root / "COST_LEDGER.json", cost)
        manifest["scientific_optimizer_updates"] = cost.get("cell_optimizer_calls", 0) + cost.get(
            "candidate_optimizer_calls", 0
        )
        atomic(
            root / "GATE_DECISION.json",
            {
                "gate3_authorized": False,
                "status": manifest["run_status"],
                "scientific_joint_benefit_tested": False,
            },
        )
        fields = [
            "seed",
            "arm",
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
            for path in sorted((root / "contexts").glob("*/SCAN_PROGRESS.json")):
                data = read(path)
                for row in data["attempts"]:
                    writer.writerow(
                        dict(row, seed=data["identity"]["seed"], arm=data["identity"]["arm"])
                    )
        (root / "FINAL_REPORT.md").write_text(
            "# R1 bounded Gate 1 diagnostic\n\nStatus: "
            + manifest["run_status"]
            + "\n\n20 fixed contexts; main50 pre-proposal pause, one candidate and all22 scales each. "
            "Parameter-nonzero + original trust caps is primary; historical nonempty-path pass is reported separately. "
            "No official Connection updates, Confirm reads or Gate3 training. "
            "Historical batch/CE matching and exact same-environment snapshot restore do not certify historical GPU bitwise replay.\n"
        )
        receipt = publication.finalize(root, manifest, args.publish_github)
    result = {
        "status": manifest["run_status"],
        "output": str(root),
        "publication": receipt,
        "gate3_authorized": False,
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    if failure:
        raise RuntimeError("R1_DIAGNOSTIC_STOPPED_SEE_RETAINED_REPORTS") from None
    return result
