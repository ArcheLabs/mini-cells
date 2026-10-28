"""Native CLM initialization and event-matched random controls; no training launcher."""

import copy
import hashlib
import json
import math
import platform
from pathlib import Path

import torch
from minicells_clm002.native import build, parameter_report
from minicells_fast001.parent import substrate
from minicells_fast_r1 import publication as transport
from minicells_fast_r1.common import ROOT, atomic, digest, read, sha, source
from minicells_r3e.common import fingerprint
from minicells_r3k.reuse import bind

BASE = Path(__file__).resolve().parents[1]
ID = "CLM-003-ROUTING-VALUE-BENCHMARK-001"
REPORTS = BASE.relative_to(ROOT) / "reports/published"
NAMES = {
    "CPU_GATE.json",
    "REGISTRATION.json",
    "MODEL_MATCHING.json",
    "DATA_ROLES.json",
    "FAILURE.json",
    "FINAL_REPORT.md",
}


def random_connection(model, seed):
    """Reset all native Connection Linear parameters; retain the exact legal graph."""
    connection = copy.deepcopy(model.connection).cpu()
    covered = set()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for module in connection.modules():
            if isinstance(module, torch.nn.Linear):
                module.reset_parameters()
                covered.update(id(p) for p in module.parameters(recurse=False))
    if covered != {id(p) for p in connection.parameters()}:
        raise RuntimeError("CLM003_UNSPECIFIED_RANDOM_CONNECTION_PARAMETER_DISTRIBUTION")
    if not all(torch.isfinite(p).all() for p in connection.parameters()):
        raise RuntimeError("CLM003_NONFINITE_RANDOM_CONNECTION")
    return copy.deepcopy(connection.state_dict())


def paired_initialization(parent, seed):
    state = random_connection(parent, seed)
    models = {arm: copy.deepcopy(parent) for arm in "ABC"}
    for arm, model in models.items():
        model.connection.load_state_dict(state, strict=True)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(
                name.startswith("cells.") or (arm == "A" and name.startswith("connection."))
            )
        model.train_connection = arm == "A"
    return models


class RewiringSchedule:
    """Use actual A commit receipts, never nominal proposal dates or quality results."""

    def __init__(self):
        self.last_step = 0
        self.events = []

    def observe(self, step, committed, a_commit_sha256, random_seed):
        if step <= self.last_step or type(committed) is not bool:
            raise RuntimeError("CLM003_EVENT_ORDER_OR_COMMIT_STATUS_INVALID")
        if committed and (
            not isinstance(a_commit_sha256, str)
            or len(a_commit_sha256) != 64
            or any(c not in "0123456789abcdef" for c in a_commit_sha256)
        ):
            raise RuntimeError("CLM003_COMMITTED_A_RECEIPT_REQUIRED")
        self.last_step = step
        row = {
            "step": step,
            "A_committed": committed,
            "C_rewire": committed,
            "A_commit_sha256": a_commit_sha256 if committed else None,
            "random_seed": random_seed if committed else None,
            "perturbation_magnitude_matched": False,
        }
        self.events.append(row)
        return row


def paired_effects(a, b, c):
    if set(a) != set(b) or set(a) != set(c) or len(a) < 2:
        raise RuntimeError("CLM003_COMPLETE_PAIRED_SEEDS_REQUIRED")
    if not all(math.isfinite(x) for d in (a, b, c) for x in d.values()):
        raise RuntimeError("CLM003_NONFINITE_QUALITY")
    return [{"seed": s, "A_minus_B": a[s] - b[s], "A_minus_C": a[s] - c[s]} for s in sorted(a)]


def allowed(name):
    return name in NAMES


def payload(root):
    manifest = read(root / "RUN_MANIFEST.json")
    if manifest.get("scope") != "PHASE0_SYNTHETIC_CPU_ONLY":
        raise RuntimeError("CLM003_ENGINEERING_REPORT_SCOPE_REQUIRED")
    if manifest.get("experiment_id") != ID:
        raise RuntimeError("CLM003_REPORT_IDENTITY_CHANGED")
    files = {}
    for name, expected in manifest["artifact_files"].items():
        path = root / name
        if not allowed(name) or any(p.is_symlink() for p in (path, *path.parents)):
            raise RuntimeError("CLM003_UNSAFE_REPORT")
        raw = path.read_bytes()
        if len(raw) > 4_000_000 or expected != {"bytes": len(raw), "sha256": digest(path)}:
            raise RuntimeError("CLM003_REPORT_HASH_OR_SIZE_CHANGED")
        if name.endswith(".json"):
            json.loads(raw)
        files[name] = raw
    if sum(map(len, files.values())) > 12_000_000:
        raise RuntimeError("CLM003_PAYLOAD_TOO_LARGE")
    snapshot = {
        k: v
        for k, v in manifest.items()
        if not k.startswith("publication") and k != "remote_commit_sha"
    }
    snapshot["publication_status"] = "GIT_PAYLOAD_SNAPSHOT"
    files["RUN_MANIFEST.json"] = (json.dumps(snapshot, sort_keys=True, indent=2) + "\n").encode()
    index = {
        n: {"bytes": len(v), "sha256": hashlib.sha256(v).hexdigest()} for n, v in files.items()
    }
    return manifest, files, index, sha(index)


def checkout(root, env):
    return bind(transport.previous.old.transport._checkout, REPORTS=REPORTS)(root, env)


def publish(root):
    return bind(
        transport.previous.old.transport.publish,
        payload=payload,
        _checkout=checkout,
        REPORTS=REPORTS,
        EXPERIMENT_ID=ID,
        BRANCH_PREFIX="results/clm003-phase0-",
    )(root)


def engineering(args):
    if args.publish_existing:
        return bind(transport.previous.old.transport.publish_existing, publish=publish)(args.output)
    torch.set_num_threads(1)
    config = read(BASE / "config.json")
    identity = {
        "experiment_id": ID,
        "source_sha": source(),
        "config_sha256": sha(config),
        "protocol_sha256": digest(BASE / "PROTOCOL.v1.zh-CN.md"),
    }
    identity["run_id"] = sha(identity)[:32]
    root = args.output / identity["run_id"]
    root.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        identity,
        identity=identity,
        scope="PHASE0_SYNTHETIC_CPU_ONLY",
        scientific_optimizer_updates=0,
        gpu_used=False,
        run_status="PREPARING",
        environment={"python": platform.python_version(), "torch": str(torch.__version__)},
    )
    failure = None
    try:
        if args.publish_github:
            bind(transport.previous.old.transport.preflight, _checkout=checkout)(root)
        model, _ = build(substrate(), 7001, unit=True)
        before = fingerprint(model.state_dict())
        rng = torch.get_rng_state().clone()
        models = paired_initialization(model, 7002)
        assert torch.equal(rng, torch.get_rng_state())
        assert before == fingerprint(model.state_dict())
        assert len({fingerprint(m.state_dict()) for m in models.values()}) == 1
        for m in models.values():
            assert fingerprint(m.cells.state_dict()) == fingerprint(model.cells.state_dict())
        schedule = RewiringSchedule()
        assert not schedule.observe(50, False, None, 7003)["C_rewire"]
        assert schedule.observe(100, True, "a" * 64, 7004)["C_rewire"]
        blockers = sorted(
            k for k, v in config["pending_scientific_registration"].items() if v is None
        )
        atomic(
            root / "REGISTRATION.json",
            {"identity": identity, "configuration": config, "blockers": blockers},
        )
        atomic(
            root / "CPU_GATE.json",
            {
                "status": "CPU_MECHANISM_PASS_SCIENCE_BLOCKED",
                "scope": "SYNTHETIC_NATIVE_CLM_FIXTURE_NO_REAL_PARENT_RECOVERY",
                "common_ABC_full_state": True,
                "original_parent_unchanged": True,
                "dedicated_random_stream_preserved": True,
                "actual_commit_timing_contract": schedule.events,
                "real_parent_optimizer_and_cursor_recovery": "PENDING_PARENT_SELECTION",
                "new_optimizer_updates": 0,
                "confirmation_model_reads": 0,
            },
        )
        atomic(
            root / "MODEL_MATCHING.json",
            {
                "native_synthetic_fixture": parameter_report(models["A"]),
                "D_MoE": "NOT_IMPLEMENTED_PENDING_ARCHITECTURE_FREEZE",
                "E_Dense": "NOT_IMPLEMENTED_PENDING_ARCHITECTURE_FREEZE",
                "comparison_tracks": ["TOTAL_PARAMETER_MATCHED", "ACTIVATED_COMPUTE_MATCHED"],
                "activation_and_FLOPs": "NOT_MEASURED_DO_NOT_EQUATE_PARAMETER_COUNT_WITH_FLOPs",
                "latency_memory_throughput": "PENDING_REAL_HARDWARE_PROTOCOL",
            },
        )
        atomic(
            root / "DATA_ROLES.json",
            {
                "required_roles": ["Train", "Development", "Calibration", "Confirmation"],
                "independent_confirmation": "UNPROVEN_SCIENCE_BLOCKED",
                "historical_confirm_can_be_reused_as_independent": False,
                "model_data_reads": 0,
            },
        )
        manifest["run_status"] = "PHASE0_CPU_PASS_SCIENTIFIC_SPECIFICATION_PENDING"
    except Exception as error:  # noqa: BLE001 -- retain partial engineering evidence
        from minicells_fast_r1.runner import safe_failure

        failure = error
        atomic(root / "FAILURE.json", safe_failure(error))
        manifest["run_status"] = "PHASE0_FAILED"
    finally:
        (root / "FINAL_REPORT.md").write_text(
            "# CLM003 phase0\n\n"
            + manifest["run_status"]
            + "\n\nSynthetic native CLM initialization and event contracts only. No GPU, training or model quality evaluation. "
            "MoE/Dense, real parent/data identities, budgets and statistical thresholds remain unregistered.\n"
        )
        manifest["artifact_files"] = {
            p.name: {"bytes": p.stat().st_size, "sha256": digest(p)}
            for p in root.iterdir()
            if p.is_file() and allowed(p.name)
        }
        atomic(root / "RUN_MANIFEST.json", manifest)
        receipt = publish(root) if args.publish_github else {"status": "LOCAL_RETAINED"}
    print(
        json.dumps(
            {"status": manifest["run_status"], "output": str(root), "publication": receipt},
            sort_keys=True,
        ),
        flush=True,
    )
    if failure:
        raise RuntimeError("CLM003_PHASE0_FAILED_SEE_REPORTS") from None
    return root
