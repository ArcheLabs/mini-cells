"""Inspect retained evidence and, only if preserved, scan the actual candidate without updates."""

from pathlib import Path
import copy
import json
import torch
from .common import atomic, canonical, configuration, digest, identity, read, source_sha
from .evidence import execution_directory
from .storage import Store


def inspect(retained, destination, assets=None, device="cpu"):
    retained = Path(retained).resolve()
    destination = Path(destination).resolve()
    if destination == retained or retained in destination.parents:
        raise RuntimeError("READONLY_OUTPUT_MUST_BE_SEPARATE")
    c = configuration()
    ident = read(retained / "identity.json")
    manifest = read(retained / "reports/RUN_MANIFEST.json")
    keys = ("experiment_id", "source_sha", "config_sha256", "run_id")
    if any(ident[k] != manifest[k] for k in keys):
        raise RuntimeError("RETAINED_IDENTITY_CONFLICT")
    if ident["config_sha256"] != canonical(c) or ident["run_id"] != canonical(
        {"source": ident["source_sha"], "config": canonical(c)}
    ):
        raise RuntimeError("RETAINED_CONFIG_OR_RUN_ID_CONFLICT")
    for item in manifest.get("artifact_files", []):
        path = (retained / item["path"]).resolve()
        if retained not in path.parents or path.is_symlink():
            raise RuntimeError("RETAINED_ARTIFACT_PATH_INVALID")
        if path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
            raise RuntimeError("RETAINED_ARTIFACT_HASH_MISMATCH")
    out = execution_directory(destination)
    result = {
        "analysis_source_sha": source_sha(),
        "retained_identity": ident,
        "training_updates": 0,
        "policy_optimizer_updates": 0,
        "thresholds_modified": False,
        "scientific_decision": False,
        "arms": {},
        "input_files": [],
        "provenance_status": "EXECUTION_HASHES_VERIFIED"
        if manifest.get("artifact_files")
        else "RUN_PROVENANCE_INCOMPLETE",
    }
    for name in (
        "identity.json",
        "reports/RUN_MANIFEST.json",
        "reports/ROUTING_DIAGNOSTICS.json",
        "reports/SINGLE_SEED_TRIAL.json",
        "reports/TRUST_REGION_TRACE.jsonl",
    ):
        path = retained / name
        if path.exists():
            if path.suffix == ".json":
                read(path)
            else:
                for line in path.read_text().splitlines():
                    if line.strip():
                        json.loads(line)
            result["input_files"].append(
                {"path": name, "bytes": path.stat().st_size, "sha256": digest(path)}
            )
    for arm in ("A-C", "B-C"):
        expected = identity(c, ident["source_sha"], 151001301, arm)
        job = retained / "private-checkpoints/jobs" / ("seed-151001301-arm-" + arm)
        latest = job / "latest.json"
        evidence = {"identity": expected, "step": 100}
        if latest.exists():
            state = Store(job, expected).verified(read(latest))
            proposals = [r for r in state["proposals"] if r["at_step"] == 100]
            evidence["verified_proposals"] = proposals
            evidence["receipt_sha256"] = digest(latest)
            evidence["checkpoint_sha256"] = read(latest)["sha256"]
        else:
            evidence["checkpoint_status"] = "NOT_AVAILABLE"
        archive = job / "readonly-proposals/transactions/proposal-0100"
        receipt = archive / "latest.json"
        if not receipt.exists():
            evidence["scan_status"] = "CANDIDATE_NOT_RETAINED_NO_REOPTIMIZATION"
        elif assets is None:
            evidence["scan_status"] = "CANDIDATE_RETAINED_CALIBRATION_REQUIRED"
        else:
            package = Store(archive, {**expected, "proposal_at_step": 100}).verified(read(receipt))
            evidence.update(scan(c, expected, package, assets, device))
        result["arms"][arm] = evidence
    result["status"] = (
        "READONLY_SCAN_COMPLETE"
        if all(v.get("scan_status") == "SCANNED" for v in result["arms"].values())
        else "READONLY_EVIDENCE_EXPORTED_SCAN_BLOCKED"
    )
    atomic(out / "READONLY_AUDIT.json", result)
    print(
        json.dumps({"status": result["status"], "output": str(out), "training_updates": 0}),
        flush=True,
    )
    return result


@torch.no_grad()
def scan(c, ident, package, assets, device):
    from .data import load
    from .stable_connection import Model
    from .evaluation import reference, states, trust, admission
    from .common import fingerprint

    if package["official"]["step"] != package["at_step"]:
        raise RuntimeError("READONLY_PROPOSAL_STEP_CONFLICT")
    data = load(c, assets, ("calibration",))
    old = Model(c, ident["seed"], ident["arm"]).to(device)
    old.load_state_dict(package["official"]["model"], strict=True)
    if (
        float(old.route_switch_margin) != package["route_switch_margin"]
        or float(old.route_numeric_tolerance) != package["numeric_tolerance"]
    ):
        raise RuntimeError("READONLY_MARGIN_CONFLICT")
    before = fingerprint(old.state_dict())
    candidate = copy.deepcopy(old)
    candidate.connection.load_state_dict(package["candidate_connection"], strict=True)
    refs = reference(old, c, data)
    fixed = states(old, refs)
    expected = [candidate.scores(r["state"], r["stage"], r["previous"])[1].clone() for r in fixed]
    original = {n: p.clone() for n, p in old.connection.named_parameters()}
    proposed = {n: p.clone() for n, p in candidate.connection.named_parameters()}
    rows = []
    scales = [0.0] + list(c["proposal"]["backtrack"]) + [2.0**-i for i in range(11, 21)]
    for scale in scales:
        for name, p in candidate.connection.named_parameters():
            p.copy_(original[name] + scale * (proposed[name] - original[name]))
        metrics = trust(
            old,
            candidate,
            refs,
            fixed=fixed,
            expected_delta=expected,
            scale=scale,
            rtol=c["proposal"]["interpolation_rtol"],
            atol=c["proposal"]["interpolation_atol"],
        )
        if scale == 0 and (
            abs(metrics["kl"]) > 1e-12
            or metrics["action_changed_count"]
            or metrics["path_changed_count"]
        ):
            raise RuntimeError("READONLY_ZERO_SCALE_MISMATCH")
        rows.append(
            {"scale": scale, **metrics, **admission(metrics, c, package["numeric_tolerance"])}
        )
    if fingerprint(old.state_dict()) != before:
        raise RuntimeError("READONLY_MODEL_MUTATED")
    return {
        "scan_status": "SCANNED",
        "route_switch_margin": package["route_switch_margin"],
        "numeric_tolerance": package["numeric_tolerance"],
        "attempts": rows,
        "matrix_release": False,
        "scan_scope": "DIAGNOSTIC_ONLY_NOT_REGISTERED_ADMISSION",
    }


def recover_trial(destination):
    """Read only the two final trial states from the user-supplied immutable HF revision."""
    from minicells_artifacts.hf_storage import HubBackend
    from minicells_r3c_moe_bridge.development import verified_manifest
    from minicells_r3c_moe_bridge.checkpoint import restore_segment
    from minicells_r3e.runner import secrets
    import uuid

    c = configuration()
    source = "98cf8f6c214f33af72d1badf28d80a0c202dc934"
    ident = {
        "experiment_id": c["experiment_id"],
        "source_sha": source,
        "config_sha256": canonical(c),
        "run_id": canonical({"source": source, "config": canonical(c)}),
    }
    ref = {
        "hf_repo": "libingjiang/minicells-research-checkpoints",
        "hf_revision": "af7f6fa03569840a024172b69c6133276b6ba04b",
        "manifest_path": "r3c-moe-bridge/segments/be6d179c1ed37c4bbaf54d1626a4d6433e3469704830c5b42416287fe1546128.json",
        "manifest_sha256": "c841ebfb471505f9ad8a4d1bcf3f76052056a67c0b91e2de234a8a7504f98cb3",
    }
    destination = Path(destination)
    if destination.exists():
        raise RuntimeError("RECOVERY_DESTINATION_ALREADY_EXISTS")
    secrets()
    backend = HubBackend(ref["hf_repo"])
    manifest = verified_manifest(backend, ref, ident)
    selected = set()
    for arm in ("A-C", "B-C"):
        prefix = "jobs/seed-151001301-arm-" + arm + "/"
        name = prefix + "latest.json"
        rowfile = manifest["files"][name]
        receipt = backend.get(rowfile["object"], ref["hf_revision"])
        if digest(receipt) != rowfile["sha256"]:
            raise RuntimeError("RECOVERY_RECEIPT_HASH_MISMATCH")
        row = read(receipt)
        expected = identity(c, source, 151001301, arm)
        if row["identity"] != expected or row["step"] < 100:
            raise RuntimeError("RECOVERY_TRIAL_IDENTITY_CONFLICT")
        selected.update(
            [
                name,
                prefix + row["file"],
                Store(Path(prefix), expected).receipt_file(row["step"]).as_posix(),
            ]
        )
    # Separate cache, no remote writes and no shadow/optimizer execution.
    destination.mkdir(parents=True, exist_ok=False)
    restore_segment(
        backend,
        ref,
        destination / "private-checkpoints",
        ident,
        uuid.uuid4().hex,
        include_paths=selected,
    )
    atomic(destination / "identity.json", ident)
    atomic(
        destination / "reports/RUN_MANIFEST.json",
        {
            **ident,
            "recovery_reference": ref,
            "recovered_subset_only": True,
            "original_execution_manifest_not_recovered": True,
        },
    )
    return destination
