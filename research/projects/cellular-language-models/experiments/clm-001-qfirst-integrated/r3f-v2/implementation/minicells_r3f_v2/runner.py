"""One bounded local experiment, never a 500-step matrix launcher."""

from pathlib import Path
from datetime import datetime, timezone
import argparse
import copy
import json
import os
import time
import uuid
import torch
from minicells_artifacts.hf_storage import HubBackend, error_code
from minicells_r3e.runner import secrets
from minicells_r3f.evaluation import trust
from .common import BASE, ID, configuration, canonical, atomic, source_sha, fingerprint
from . import storage, parent_import, proposal, scale_scan, adaptation, evaluation


def hypotheses(arms):
    valid = [r for r in arms.values() if r.get("valid", False)]
    admitted = any(r["trust"]["chosen_scale"] is not None for r in valid)
    adapted = [r for r in valid if r["adaptation"]["cell_adaptation_executed"]]
    invalid = any(not r.get("valid", False) for r in arms.values())
    return {
        "H1": "SUPPORTED_EXPLORATORY"
        if admitted
        else "INVALID_RUN"
        if invalid
        else "NOT_SUPPORTED",
        "H2": "SUPPORTED_EXPLORATORY" if adapted else "INVALID_RUN" if invalid else "NOT_SUPPORTED",
        "H3": (
            "SUPPORTED_EXPLORATORY"
            if any(r["adaptation"]["final_quality_accepted"] for r in adapted)
            else "NOT_SUPPORTED"
            if adapted
            else "NOT_TESTED"
        ),
    }


def arm_run(c, backend, output, private, ident, arm, device, durable, assets):
    arm_ident = {**ident, "arm": arm, "seed": 151001301}
    root = private / "jobs" / arm
    finished = root / "result.json"
    if finished.exists():
        from .common import read

        result = read(finished)
        if any(result["identity"].get(k) != v for k, v in arm_ident.items()):
            raise RuntimeError("RESULT_IDENTITY_CONFLICT")
        return result
    wall = time.perf_counter()
    parent_state, parent = parent_import.restore(c, backend, output / "parent-input", arm)
    arm_ident["parent_checkpoint_sha256"] = parent["parent_checkpoint_sha256"]
    # Completed result identity includes the imported checkpoint, unlike the coordinator identity.
    if not (root / "PARENT_IMPORT.json").exists():
        root.mkdir(parents=True, exist_ok=True)
    tr, verification = parent_import.import_state(
        c, parent_state, parent, arm, device, root, arm_ident
    )
    data = evaluation.data(c, backend, assets)
    refs = list(evaluation.admission_windows(tr.model, c, data))
    zero = trust(tr.model, copy.deepcopy(tr.model), refs)
    if zero["kl"] != 0 or zero["action_changed_count"] or zero["path_changed_count"]:
        raise RuntimeError("ZERO_SCALE_FORWARD_INCONSISTENT")
    verification["zero_self_comparison"] = zero
    before = fingerprint(tr.model.state_dict())
    for x in refs:
        if not torch.equal(tr.model(x).actions, tr.model(x).actions):
            raise RuntimeError("DETERMINISTIC_PATH_NOT_REPRODUCIBLE")
    if fingerprint(tr.model.state_dict()) != before:
        raise RuntimeError("PARENT_MODEL_MUTATED")
    # Baseline check overhead is explicitly separate from proposal/adaptation work.
    baseline_work = {k: v - parent_state["work"].get(k, 0) for k, v in tr.model.work.items()}
    tr.model.work = copy.deepcopy(parent_state["work"])
    proposal_root = private / "proposals" / arm
    package = proposal.generate(tr, data, proposal_root, arm_ident, durable)
    scan = scale_scan.scan(tr, data, package, private / "scans" / arm, arm_ident)
    event, routes = adaptation.execute(tr, data, package, scan, root, arm_ident)
    selected, receipt = storage.save(root / "selected", arm_ident, tr.state(), tr.step)
    if scan["chosen_scale"] is not None and selected["step"] != 110:
        raise RuntimeError("ADAPTATION_FINAL_STEP_INVALID")
    # A receipt with no admission remains the unchanged imported official state.
    result = {
        "identity": arm_ident,
        "valid": True,
        "parent": verification,
        "proposal": {
            "candidate_fingerprint": package["candidate_fingerprint"],
            "route_switch_margin": package["route_switch_margin"],
            "numeric_tolerance": package["numeric_tolerance"],
            "original_fingerprint": package["original_fingerprint"],
            "state_sample_fingerprint": package["state_sample_fingerprint"],
            "losses": package["losses"],
            "optimizer_configuration": package["optimizer_configuration"],
            "receipt": json.loads((proposal_root / "latest.json").read_text()),
            "new_candidate_not_historical_reconstruction": True,
        },
        "trust": scan,
        "adaptation": event,
        "training_routes": routes,
        "selected_receipt": receipt,
        "cost": {
            "proposal": package["cost"],
            "scan_seconds": scan["seconds"],
            "scan_work": scan["work"],
            "adaptation_and_selection": event.get("work_by_phase", {}),
            "baseline_check_incremental_work": baseline_work,
            "selected_branch_incremental": {
                k: v - package["official"]["cost"].get(k, 0)
                for k, v in tr.cost.items()
                if isinstance(v, (int, float))
            },
            "total_arm_wall_seconds_this_invocation": time.perf_counter() - wall,
        },
    }
    atomic(finished, result)
    durable()
    return result


def reports(output, ident, arms, publication, started_at):
    root = output / "reports"
    root.mkdir(parents=True, exist_ok=True)
    h = hypotheses(arms)
    invalid = any(not r.get("valid", False) for r in arms.values()) or len(arms) != 2
    status = (
        "INVALID_OR_PARTIAL_RUN"
        if invalid
        else "LOCAL_COADAPTATION_MECHANISM_VERIFIED"
        if h["H2"] == "SUPPORTED_EXPLORATORY"
        else "LOCAL_CONNECTION_ADMISSION_NOT_ESTABLISHED"
    )
    summary = {
        "identity": ident,
        "status": status,
        "hypotheses": h,
        "scientific_decision": False,
        "full_matrix_authorized": False,
        "arms": {
            a: {
                "valid": r.get("valid", False),
                "failure": r.get("failure"),
                "chosen_scale": r.get("trust", {}).get("chosen_scale"),
                "adapt_steps": r.get("adaptation", {}).get("adapt_steps", 0),
                "final_quality_accepted": r.get("adaptation", {}).get(
                    "final_quality_accepted", False
                ),
            }
            for a, r in arms.items()
        },
    }
    mappings = {
        "PARENT_RESTORE_VERIFICATION.json": "parent",
        "PROPOSAL_MANIFEST.json": "proposal",
        "LOCAL_SCALE_SCAN.json": "trust",
        "TRUST_REGION_RESULTS.json": "trust",
        "ADAPTATION_RESULTS.json": "adaptation",
        "TRAINING_ROUTE_DIFFERENCE.json": "training_routes",
        "COMPUTE_COST.json": "cost",
    }
    for name, key in mappings.items():
        atomic(
            root / name,
            {
                a: r.get(key, {"status": "NOT_EXECUTED", "failure": r.get("failure")})
                for a, r in arms.items()
            },
        )
    atomic(root / "METRICS.json", summary)
    (root / "SCIENTIFIC-FINDINGS.md").write_text(
        "# R3f-v2 local mechanism experiment\n\n"
        + status
        + "\n\n"
        + "\n".join(k + ": " + v for k, v in h.items())
        + "\n\nAdmission selects stability; Calibration selects quality. Development and Confirmation unread.\n"
        "No full matrix is authorized. These are exploratory mechanism findings, not statistical confirmation.\n"
    )
    from minicells_r3f.evidence import seal

    seal(
        output,
        ident,
        status=status,
        exit_code=1 if invalid else 0,
        publication=publication,
        started_at=started_at,
    )
    return summary


def automatic(c, parent_output, assets=None):
    started = datetime.now(timezone.utc).isoformat()
    runner_begin = time.perf_counter()
    secrets()
    backend = HubBackend(c["storage"]["private_repo"])
    if not backend.info().private:
        raise RuntimeError("PRIVATE_REPO_REQUIRED")
    backend.check_write()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU_REQUIRED_FOR_REAL_LOCAL_EXPERIMENT")
    source = source_sha()
    run = canonical(
        {
            "source_sha": source,
            "config_sha256": canonical(c),
            "parent_reference": c["local_experiment"]["parent_hf_reference"],
        }
    )
    pointer = backend.optional("r3fv2/" + run + "/latest.json", backend.info().sha)
    prior = None
    execution = uuid.uuid4().hex if pointer is None else pointer["identity"]["execution_id"]
    ident = {
        "experiment_id": ID,
        "source_sha": source,
        "config_sha256": canonical(c),
        "run_id": run,
        "execution_id": execution,
        "parent_source_sha": c["local_experiment"]["parent_source_sha"],
        "parent_run_id": c["local_experiment"]["parent_run_id"],
    }
    output = Path(parent_output) / execution
    output.mkdir(parents=True, exist_ok=True)
    from .common import read

    if (output / "identity.json").exists() and read(output / "identity.json") != ident:
        raise RuntimeError("OUTPUT_IDENTITY_CONFLICT")
    atomic(output / "identity.json", ident)
    private = output / "private-checkpoints"
    session = uuid.uuid4().hex
    if pointer:
        prior = {**pointer, "hf_repo": backend.repo, "hf_revision": backend.info().sha}
        storage.restore(backend, prior, private, ident, session)
    elif (output / "HF_REFERENCE.json").exists():
        prior = read(output / "HF_REFERENCE.json")

    def durable():
        nonlocal prior
        prior = storage.upload(backend, private, output, ident, session, prior)

    arms = {}
    for i, arm in enumerate(c["arms"]):
        device = "cuda:" + str(i % torch.cuda.device_count())
        print("[R3f-v2]", arm, "one proposal from official step 100 on", device, flush=True)
        try:
            arms[arm] = arm_run(
                c,
                backend,
                output,
                private,
                ident,
                arm,
                device,
                durable,
                Path(assets or "/kaggle/working/r3fv2-assets"),
            )
        except Exception as error:
            arms[arm] = {
                "valid": False,
                "failure": {
                    "code": str(error)
                    if str(error).isupper() and " " not in str(error)
                    else type(error).__name__,
                    **error_code(error),
                },
            }
            print("[R3f-v2]", arm, arms[arm]["failure"], flush=True)
        # Each arm remains independent. Never release a full matrix.
    publication = {"status": "NOT_ATTEMPTED"}
    result = reports(output, ident, arms, publication, started)
    cost_path = output / "reports/COMPUTE_COST.json"
    costs = read(cost_path)
    elapsed = time.perf_counter() - runner_begin
    atomic(
        cost_path,
        {
            "arms": costs,
            "runner_wall_seconds_before_publication": elapsed,
            "visible_gpu_count": torch.cuda.device_count(),
            "allocated_gpu_seconds_estimate": elapsed * torch.cuda.device_count(),
            "scope": "Runner wall time including storage; GPU allocation estimate is not summed kernel time; Notebook setup and post-publication excluded",
        },
    )
    from minicells_r3f.evidence import seal

    seal(
        output,
        ident,
        status=result["status"],
        exit_code=1 if result["status"] == "INVALID_OR_PARTIAL_RUN" else 0,
        publication=publication,
        started_at=started,
    )
    try:
        from .publication import publish

        publication = (
            publish(output)
            if os.environ.get("GITHUB_TOKEN")
            else {"status": "GITHUB_TOKEN_UNAVAILABLE"}
        )
    except Exception as error:
        publication = {"status": "LOCAL_RETAINED_REPUBLISH_ONLY", **error_code(error)}
    atomic(output / "PUBLICATION.json", publication)
    # Keep published aggregates immutable; only refresh the local manifest with verified return status.
    from minicells_r3f.evidence import seal

    seal(
        output,
        ident,
        status=result["status"],
        exit_code=1 if result["status"] == "INVALID_OR_PARTIAL_RUN" else 0,
        publication=publication,
        started_at=started,
    )
    print(json.dumps({**result, "publication": publication, "output": str(output)}), flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(BASE / "config/frozen.json"))
    p.add_argument("--run-all", action="store_true")
    p.add_argument("--publish-existing")
    p.add_argument("--output", default="/kaggle/working/r3fv2-output")
    p.add_argument("--assets")
    a = p.parse_args()
    if a.publish_existing:
        secrets()
        from .publication import publish

        print(json.dumps(publish(a.publish_existing)))
    elif a.run_all:
        result = automatic(configuration(a.config), a.output, a.assets)
        if result["status"] == "INVALID_OR_PARTIAL_RUN":
            raise SystemExit(1)
    else:
        p.error("--run-all required")
