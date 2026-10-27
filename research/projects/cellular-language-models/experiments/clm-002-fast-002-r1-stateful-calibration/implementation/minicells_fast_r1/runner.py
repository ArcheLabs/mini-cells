"""Read-only historical closure / optional real-parent CPU checks; all scientific launches blocked."""

import copy
import importlib.metadata
import platform
import shutil
import sys
import time

import torch
from minicells_fast001.parent import make_engine
from minicells_fast002 import checkpoints, recovery
from minicells_fast002.common import historical_reports
from minicells_r3e.common import fingerprint

from . import assets, history, publication
from .common import BASE, ID, atomic, config, digest, read, require_ci, sha, source


def inspect_parents(c, reports, paths, root, identity):
    checked = []
    for key, path in sorted(paths.items()):
        row = read(reports / "jobs" / (key + ".json"))
        state = checkpoints.load(path, row)
        if state["fast"]["phase"] != "A" or state["step"] != 500:
            raise RuntimeError("R1_MUST_START_FROM_MATCHING_A_STEP500")
        engine = make_engine(
            c,
            state,
            row["seed"],
            "B01-" + row["arm"][-1],
            dict(
                identity,
                seed=row["seed"],
                source_parent_arm=row["arm"],
                operation="ZERO_UPDATE_PARENT_INSPECTION",
            ),
            root / "private/parent-checks" / (key + ".jsonl"),
            device="cpu",
            parent_sha=row["checkpoint_sha256"],
        )
        if (
            fingerprint(engine.model.state_dict()) != fingerprint(state["model"])
            or fingerprint(engine.cell_optimizer.state_dict())
            != fingerprint(state["cell_optimizer"])
            or engine.journal.summary()["actual_optimizer_attempts"]
        ):
            raise RuntimeError("R1_CPU_PARENT_VALUES_OR_OPTIMIZER_MISMATCH")
        checked.append(
            {
                "seed": row["seed"],
                "arm": row["arm"],
                "sha256": digest(path),
                "model_values_exact": True,
                "cell_optimizer_values_exact": True,
                "local_parent_step": 500,
                "absolute_parent_step": engine.parent_step,
                "parameters": sum(p.numel() for p in engine.model.parameters()),
            }
        )
        for hook in engine._work_handles:
            hook.remove()
        del engine, state
    return {
        "status": "REAL_A_PARENT_CPU_IDENTITY_PASS",
        "checkpoints": checked,
        "gpu_used": False,
        "forward_calls": 0,
        "optimizer_calls": 0,
        "exact_step50_state": "NOT_RECOVERED; STATEFUL_REPLAY_REQUIRES_SEPARATE_APPROVAL",
    }


def engineering(args):
    if args.publish_existing:
        return publication.existing(args.output)
    torch.set_num_threads(1)
    c = config()
    started = time.perf_counter()
    ident = {"experiment_id": ID, "source_sha": source(), "config_sha256": sha(c)}
    ident["run_id"] = sha(ident)[:32]
    root = args.output / ident["run_id"]
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment_id": ID,
        "scope": "STATEFUL_CALIBRATION_ENGINEERING",
        "identity": ident,
        "run_id": ident["run_id"],
        "source_sha": ident["source_sha"],
        "config_sha256": sha(c),
        "command": sys.argv,
        "run_status": "PREPARING",
        "scientific_optimizer_updates": 0,
        "gpu_used": False,
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "numpy": importlib.metadata.version("numpy"),
            "tokenizers": importlib.metadata.version("tokenizers"),
        },
    }
    atomic(root / "RUN_MANIFEST.json", manifest)
    try:
        atomic(
            root / "REGISTRATION.json",
            {"identity": ident, "configuration": c, "execution_authorized": False},
        )
        shutil.copyfile(BASE / "PROTOCOL_AMENDMENT.md", root / "PROTOCOL_AMENDMENT.md")
        atomic(root / "DATASET_MANIFEST.json", read(BASE / "registration/DATASET_MANIFEST.json"))
        reports = historical_reports(c, args.assets / "historical-reports")
        history.audit(c, reports, root)
        assets.admission(c, args.assets / "admission")
        if args.inspect_cloud_parents:
            atomic(root / "CI_ACCEPTANCE.json", require_ci())
            selection = copy.deepcopy(c)
            selection["arms"] = ["A-S", "A-H"]
            paths, recovery_proof = recovery.resolve(
                selection, reports, [], args.assets / "kaggle-output"
            )
            old = assets.training(args.assets / "train")
            parent = inspect_parents(c, reports, paths, root, ident)
            parent["runtime_recovery"] = recovery_proof
            parent["historical_train_config_sha256"] = sha(old)
        else:
            parent = {
                "status": "PENDING_REAL_PARENT_BYTES_CPU_ENGINEERING_ONLY",
                "gpu_used": False,
                "optimizer_calls": 0,
            }
        atomic(root / "PARENT_STATE_AUDIT.json", parent)
        atomic(
            root / "REPLAY_AUDIT.json",
            {
                "status": "NOT_RUN_REQUIRES_EXPLICIT_GATE1_APPROVAL",
                "cpu_fixture_verification": read(BASE / "verification/CPU_VALIDATION.json"),
                "historical_runtime_bitwise_replay": "NOT_CERTIFIED",
            },
        )
        atomic(
            root / "ADMISSION_DECISION.json",
            {
                "status": "NOT_RUN",
                "all_22_scales_required": True,
                "historical_anchor": 1 / 4096,
                "formal_connection_commits": 0,
                "effectiveness_definition_review_required": True,
            },
        )
        (root / "CONNECTION_SCALE_SCAN.csv").write_text(
            "seed,arm,scale,kl,action_change,path_change,parameter_delta_norm,passed\n"
        )
        gates = {
            "gate0": parent["status"],
            "gate1": "BLOCKED_REVIEW_AND_EXPLICIT_DIAGNOSTIC_APPROVAL_REQUIRED",
            "gate2": "INDEPENDENT_REVIEW_REQUIRED",
            "gate3_authorized": False,
            "review_items": [
                "NONZERO_PARAMETER_VS_HISTORICAL_NONEMPTY_HARD_PATH_RULE",
                "DUPLICATE_FULL_GPU_REPLAY_WITHIN_1000_MAIN_UPDATE_CAP",
            ],
            "scientific_joint_benefit_tested": False,
        }
        atomic(root / "GATE_DECISION.json", gates)
        (root / "FINAL_REPORT.md").write_text(
            "# R1 engineering / read-only Gate 0\n\nHistorical results preserved. Admission identity and isolation are frozen.\n\n"
            + "Parent status: "
            + parent["status"]
            + ". Real Step50 replay and candidates have NOT been executed.\n\n"
            + "CPU fixtures prove pre-proposal suspension, B01/B11 masks, RNG/optimizer/batch/gradient restore, full22 scales and original Q optimizer equivalence. They do not prove historical GPU bitwise reproduction.\n\n"
            + "Independent review must resolve parameter-only versus historical nonempty-path effectiveness and repeat-replay budget accounting. Gate1 needs explicit approval; Gate3 remains forbidden.\n"
        )
        manifest["run_status"] = "CPU_ENGINEERING_COMPLETE_DIAGNOSTIC_EXECUTION_BLOCKED"
    except Exception as error:
        manifest["run_status"] = "BLOCKED_OR_PARTIAL"
        atomic(root / "FAILURE.json", {"type": type(error).__name__, "message": str(error)[:1000]})
        raise
    finally:
        atomic(
            root / "COST_LEDGER.json",
            {
                "scope": "READONLY_ENGINEERING_THIS_RUN",
                "main_replay_updates": 0,
                "cell_optimizer_calls": 0,
                "candidate_optimizer_calls": 0,
                "formal_connection_commits": 0,
                "gpu_used": False,
                "gpu_kernel_seconds": None,
                "engineering_wall_seconds": time.perf_counter() - started,
            },
        )
        receipt = publication.finalize(root, manifest, args.publish_github)
    return {
        "status": manifest["run_status"],
        "output": str(root),
        "publication": receipt,
        "gate1_execution_authorized": False,
        "gate3_execution_authorized": False,
    }
