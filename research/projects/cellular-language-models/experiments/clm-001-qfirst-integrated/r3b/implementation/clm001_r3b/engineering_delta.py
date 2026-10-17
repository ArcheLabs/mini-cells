"""Exactly two independent one-update engineering traces; never a scientific run."""

from copy import deepcopy
from pathlib import Path
import tempfile
import time
import torch
from minicells_runtime.store import (
    read,
    atomic,
    canonical,
    digest,
    MountedInputStore,
    ContinuationError,
)
from clm001_r3.data import load_assets, batches
from clm001_r3 import checkpoint as cp
from . import runner
from .analysis_runner import fingerprint
from .diagnostic_schedule import load_plan, telemetry
from .instrumentation import isolated
from .pilot import equivalent


def compare(root, output, *, arm="QC-R3b", step=3, revision=0, device="cuda:0"):
    """Restore same parent twice; old update vs Journal-only observation + old update."""
    root, output = Path(root), Path(output)
    meta = runner.verify(root)
    selected = MountedInputStore(root).select(arm, step, revision)
    if output.exists():
        report = read(output / "delta-report.json")
        if (
            report["parent_checkpoint_sha256"] != selected["checkpoint_sha256"]
            or report["verification_source_sha"] != runner.reused.source_sha()
        ):
            raise ContinuationError("ENGINEERING_OUTPUT_IDENTITY_MISMATCH")
        return report
    original_files = {
        p.relative_to(root).as_posix(): digest(p) for p in root.rglob("*") if p.is_file()
    }
    c = read(root / "runtime-config.json")
    data = load_assets(c, root / "assets")
    engineering_id = {
        "experiment_id": "R3B-ONE-UPDATE-ENGINEERING",
        "run_id": "delta-"
        + canonical(
            {"parent": selected["checkpoint_sha256"], "source": runner.reused.source_sha()}
        )[:16],
        "source_sha": runner.reused.source_sha(),
        "parent_identity": read(root / "arms" / arm / "identity.json"),
        "parent_checkpoint_sha256": selected["checkpoint_sha256"],
        "scientific_decision": False,
    }
    traces, payloads = [], []
    with tempfile.TemporaryDirectory() as tmp, isolated(torch.nn.Identity()):
        for observe in (False, True):
            trainer, nursery, rng, _ = runner.load_state(
                root / "arms" / arm,
                deepcopy(c),
                arm,
                data,
                meta["source_sha"],
                device,
                selector=selected,
            )
            folder = Path(tmp) / ("observed" if observe else "control")
            atomic(folder / "identity.json", engineering_id)
            cp.save(folder, trainer, nursery, rng, engineering_id)
            observed_config = deepcopy(c)
            observed_config["r3b"]["diagnostic_execution"] = load_plan()
            if observe:
                telemetry(folder, trainer, observed_config, "engineering-before-update")
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            started = time.perf_counter()
            trainer.update(batches(data, c, rng, device))
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            cp.save(folder, trainer, nursery, rng, engineering_id)
            if observe:
                telemetry(folder, trainer, observed_config, "engineering-after-update")
            state = folder / read(folder / "latest.json")["file"]
            payloads.append(torch.load(state, map_location="cpu", weights_only=False))
            traces.append(
                {
                    "observation_enabled": observe,
                    "updates": 1,
                    "step": trainer.step,
                    "state_fingerprint": runner.reused.state_fingerprint(state),
                    "update_seconds": seconds,
                    "private_candidate_count": len(nursery.candidates),
                    "registry_sha256": fingerprint(vars(trainer.model.registry)),
                }
            )
            del trainer, nursery
    if original_files != {
        p.relative_to(root).as_posix(): digest(p) for p in root.rglob("*") if p.is_file()
    }:
        raise ContinuationError("ENGINEERING_MUTATED_PARENT")

    def invariants(payload):
        extra = {k: v for k, v in payload["extra"].items() if k != "journal"}
        return {
            **{k: payload[k] for k in ("step", "python_rng", "numpy_rng", "torch_rng", "cuda_rng")},
            "extra": extra,
        }

    exact_invariants = fingerprint(invariants(payloads[0])) == fingerprint(invariants(payloads[1]))
    model_optimizer_equal = all(
        equivalent(payloads[0][k], payloads[1][k]) for k in ("model", "optimizer")
    )
    passed = (
        traces[0]["state_fingerprint"] == traces[1]["state_fingerprint"]
        if device == "cpu"
        else exact_invariants and model_optimizer_equal
    )
    report = {
        "status": "TRAINING_DELTA_PASS" if passed else "TRAINING_DELTA_FAILED",
        "experiment_id": engineering_id["experiment_id"],
        "run_id": engineering_id["run_id"],
        "verification_source_sha": runner.reused.source_sha(),
        "parent_training_source_sha": meta["source_sha"],
        "parent_checkpoint_sha256": selected["checkpoint_sha256"],
        "arm": arm,
        "parent_step": step,
        "parent_event_revision": revision,
        "traces": traces,
        "total_optimizer_updates": 2,
        "production_steps_added": 0,
        "formal_500_step_authorized": False,
        "scientific_decision": False,
        "comparison": "CPU exact full fingerprint; GPU model/optimizer rtol=1e-5 atol=1e-7 with exact RNG/cursor/registry/Nursery/events",
        "exact_invariants_pass": exact_invariants,
        "model_optimizer_pass": model_optimizer_equal,
        "mandatory_production_event_cost": "NOT_MEASURED",
        "historical_parent_unchanged": True,
        "device": torch.cuda.get_device_name() if device.startswith("cuda") else "CPU_SYNTHETIC",
    }
    output.mkdir(parents=True)
    atomic(output / "delta-report.json", report)
    if not passed:
        raise ContinuationError("TRAINING_DELTA_MISMATCH")
    return report


def main():
    import argparse
    from minicells_artifacts.hf_storage import HubBackend, HFCheckpointStore
    from .common import BASE, safe_error
    from .publication import publish_documents

    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    try:
        backend = HubBackend("libingjiang/minicells-research-checkpoints")
        store = HFCheckpointStore(backend)
        baseline = read(BASE / "reports/diagnostic-optimization/baseline-evidence.json")
        ref = baseline["evidence"]["checkpoint-index.json"]["checkpoint_ref"]
        root = a.output / "verified-parent"
        if not root.exists():
            store.fetch_verified_checkpoint(ref, root)
            atomic(a.output / "parent-reference.json", ref)
        elif read(a.output / "parent-reference.json") != ref:
            raise ContinuationError("ENGINEERING_INPUT_REFERENCE_CHANGED")
        backend.check_write()
        result = compare(root, a.output / "engineering-result")
        path = (
            "engineering/r3b-delta/"
            + canonical(
                {
                    "parent": result["parent_checkpoint_sha256"],
                    "source": result["verification_source_sha"],
                }
            )
            + ".json"
        )
        report_file = a.output / "engineering-result/delta-report.json"
        head = backend.info()
        if not head.private:
            raise ContinuationError("PRIVATE_REPOSITORY_REQUIRED")
        revision = backend.commit({path: report_file}, head.sha, "bounded R3b one-update delta")
        if digest(backend.get(path, revision)) != digest(report_file):
            raise ContinuationError("ENGINEERING_REPORT_READBACK_FAILED")
        result["engineering_hf_evidence"] = {
            "hf_repo": backend.repo,
            "hf_revision": revision,
            "hf_path": path,
            "sha256": digest(report_file),
        }
        publish_documents(
            root,
            {
                "experiment-manifest.json": {
                    "experiment_id": result["experiment_id"],
                    "status": result["status"],
                    "formal_training_started": False,
                    "scientific_decision": False,
                },
                "resource-ledger.json": result,
            },
            scope={"engineering_delta_source": result["verification_source_sha"]},
        )
        print(result, flush=True)
    except Exception as error:
        atomic(
            a.output / "engineering-failure.json",
            {"status": "ENGINEERING_FAILED", **safe_error(error)},
        )
        print({"status": "ENGINEERING_FAILED", **safe_error(error)}, flush=True)
        raise
