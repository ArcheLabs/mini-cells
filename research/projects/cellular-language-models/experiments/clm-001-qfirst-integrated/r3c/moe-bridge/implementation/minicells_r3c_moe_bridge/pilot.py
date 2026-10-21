"""One-launch bounded pilot: preflight, pinned assets, joint updates, checkpoint, return.

Only this short pilot is released. No formal/Confirmation execution is reachable here.
"""

from pathlib import Path
import os
import random
import shutil
import time
import uuid
from copy import deepcopy
import numpy as np
import torch
from torch.nn import functional as F
from clm001_r3b.instrumentation import isolated
from minicells_artifacts.hf_storage import HubBackend
from .common import (
    BASE,
    ARMS,
    ID,
    atomic,
    canonical,
    digest,
    read,
    source_sha,
    validate,
    ContinuationError,
)
from .checkpoint import Store, publish_segment, restore_segment
from .data import train_batches, fixed_windows, private_asset_subset
from .model import BridgeModel
from .training import Trainer
from .instrumentation import MatmulAudit, parameters
from .release import require_gpu_smoke, choose_balance
from .runner import identity


def pilot_configuration():
    c = read(BASE / "config/pilot.json")
    validate(c)
    approval = c["pilot"]["authorization"]
    if (
        approval["scope"] != "GPU_SMOKE_ONLY"
        or not c["release"]["gpu_smoke_authorized"]
        or not c["pilot"]["gpu_authorized"]
        or approval["max_optimizer_updates"] != 40
        or approval["max_train_tokens"] != 20480
        or c["pilot"]["total_optimizer_updates"] > 40
        or c["pilot"]["training_tokens"] > 20480
    ):
        raise ContinuationError("SHORT_PILOT_RELEASE_INVALID")
    require_gpu_smoke(
        c, {"scope": "GPU_SMOKE_ONLY", "approved": True, "config_sha256": canonical(c)}
    )
    return c


def load_secrets():
    for name in ("HF_TOKEN", "GITHUB_TOKEN"):
        token = os.environ.get(name, "").strip()
        if not token:
            try:
                from kaggle_secrets import UserSecretsClient

                token = UserSecretsClient().get_secret(name).strip()
            except Exception:
                raise ContinuationError("ENABLE_EXISTING_KAGGLE_SECRET_" + name) from None
        if not token:
            raise ContinuationError("ENABLE_EXISTING_KAGGLE_SECRET_" + name)
        os.environ[name] = token


def timed_update(trainer, batches, device):
    cuda = str(device).startswith("cuda")
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
    before = time.perf_counter()
    row = trainer.update(batches)
    if cuda:
        end.record()
        end.synchronize()
    seconds = time.perf_counter() - before
    row.update(
        wall_seconds=seconds,
        cuda_time_ms=start.elapsed_time(end) if cuda else "NOT_MEASURED",
        tokens_per_second=sum(x.numel() for x, y in batches) / seconds,
    )
    return row


def _receipts(store):
    rows = [read(path) for path in (store.root / "receipts").glob("*.json")]
    for row in rows:
        if row["identity"] != store.identity:
            raise ContinuationError("R3C_CHECKPOINT_IDENTITY_MISMATCH")
    return sorted(rows, key=lambda row: row["step"])


def execute_job(c, data, root, source, job, arm, coefficient, steps, device):
    """Testable with synthetic CPU data; the public automatic entry requires CUDA."""
    seed = c["candidate_training"]["seeds"][0]
    ident = {
        **identity(c, arm, seed, source),
        "pilot_job": job,
        "effective_balance_coefficient": coefficient,
    }
    store = Store(Path(root) / "jobs" / job, ident)
    rows = _receipts(store)
    saved = rows[-1] if rows else None
    if saved and saved["step"] > steps:
        raise ContinuationError("PILOT_BUDGET_MISMATCH")
    result_path = store.root / "result.json"
    if result_path.exists():
        result = read(result_path)
        if (
            not saved
            or result["identity"] != ident
            or result["checkpoint"]["sha256"] != saved["sha256"]
            or saved["step"] != steps
        ):
            raise ContinuationError("PILOT_RESULT_IDENTITY_MISMATCH")
        store.verified(saved)
        print("[Pilot] reuse completed " + job + "; optimizer updates added: 0", flush=True)
        return result
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    model_c = deepcopy(c)
    model_c["balance"]["coefficient"] = coefficient
    trainer = Trainer(BridgeModel(model_c, arm, seed).to(device), coefficient)
    if saved:
        store.restore(trainer, saved)
    print(
        "[Pilot] " + job + " restored_step=" + str(trainer.step) + " target=" + str(steps),
        flush=True,
    )
    while trainer.step < steps:
        batches = train_batches(data, c, trainer.data_rng, device)
        row = {**ident, **timed_update(trainer, batches, device)}
        # Every completed update has a full state receipt before the next begins.
        before = time.perf_counter()
        receipt = store.save(trainer)
        row["checkpoint_save_seconds"] = time.perf_counter() - before
        row["checkpoint_sha256"] = receipt["sha256"]
        atomic(store.root / "logs" / ("step-%04d.json" % trainer.step), row)
        print(
            "[Pilot] "
            + job
            + " "
            + str(trainer.step)
            + "/"
            + str(steps)
            + " CE="
            + str(round(row["ce"], 5)),
            flush=True,
        )
    logs = []
    for step in range(1, steps + 1):
        path = store.root / "logs" / ("step-%04d.json" % step)
        logs.append(
            read(path)
            if path.exists()
            else {
                "step": step,
                "status": "CHECKPOINT_VERIFIED_METRICS_NOT_SAVED",
                "wall_seconds": "NOT_MEASURED",
            }
        )
    shares = []
    losses = []
    with torch.no_grad(), isolated(trainer.model):
        trainer.model.eval()
        rng = torch.Generator().manual_seed(seed + 333)
        for story, x, y in fixed_windows(data, c, "calibration", device):
            out = trainer.model(x, route_rng=rng)
            shares.append(
                (out.operator_rows / out.operator_rows.sum(-1, keepdim=True)).cpu().tolist()
            )
            losses.append(float(F.cross_entropy(out.logits.flatten(0, 1), y.flatten())))
            if len(shares) == c["balance"]["measurement_windows"]:
                break
        # Independent read-only input batch, without advancing the training cursor.
        sample = train_batches(data, c, np.random.default_rng(seed + 4999), device)[0][0]
        with MatmulAudit(trainer.model) as audit:
            out = trainer.model(sample, route_rng=torch.Generator().manual_seed(seed + 4999))
        cost = audit.finish(out)
    seconds = [
        row["wall_seconds"] for row in logs if isinstance(row.get("wall_seconds"), (float, int))
    ]
    result = {
        "identity": ident,
        "arm": arm,
        "job": job,
        "coefficient": coefficient,
        "steps": steps,
        "logs": logs,
        "role": "calibration",
        "cell_shares": shares,
        "calibration_mean_nll": float(np.mean(losses)),
        "inference_matmul_cost": cost,
        "parameters": parameters(trainer.model),
        "checkpoint": _receipts(store)[-1],
        "projected_500_update_seconds": float(np.median(seconds)) * 500
        if seconds
        else "NOT_MEASURED",
        "projection_scope": "short Train compute updates only, excluding Checkpoint/data/evaluation/upload; noisy early-step estimate",
    }
    atomic(result_path, result)
    del trainer
    return result


def execute_pilot(c, data, output, source, device):
    output = Path(output)
    private = output / "private-checkpoints"
    measurements = {}
    candidates = []
    for i, coefficient in enumerate(c["balance"]["candidates"]):
        row = execute_job(
            c,
            data,
            private,
            source,
            "balance-" + str(i),
            "C",
            coefficient,
            c["balance"]["pilot_steps_per_candidate"],
            device,
        )
        candidates.append(row)
        measurements[coefficient] = {"role": "calibration", "cell_shares": row["cell_shares"]}
    selected = choose_balance(c, measurements)
    arms = []
    if selected["status"] != "NO_GO":
        for arm in ARMS:
            arms.append(
                execute_job(
                    c,
                    data,
                    private,
                    source,
                    "arm-" + arm,
                    arm,
                    selected["coefficient"],
                    c["pilot"]["steps_per_arm"],
                    device,
                )
            )
    results = candidates + arms
    count = sum(row["steps"] for row in results)
    report = {
        "schema": "minicells.r3c.pilot.v1",
        "experiment_id": ID,
        "source_sha": source,
        "config_sha256": canonical(c),
        "status": "PILOT_COMPLETED_NOT_FORMAL" if arms else "NO_GO_BALANCE_EXPOSURE",
        "selection": selected,
        "candidates": candidates,
        "arms": arms,
        "logical_optimizer_updates": count,
        "logical_train_tokens": count
        * c["training"]["sequence_length"]
        * c["training"]["effective_batch"],
        "maximum_optimizer_updates": c["pilot"]["total_optimizer_updates"],
        "gpu_pilot_completed": str(device).startswith("cuda"),
        "model_data_roles": ["train", "calibration"],
        "asset_integrity_checked_roles": ["train", "calibration", "admission", "development"],
        "development_evaluated": False,
        "confirmation": "SEALED",
        "formal_500_step_authorized": False,
        "actual_kaggle_persistence_verified": False,
        "scientific_decision": False,
    }
    atomic(output / "reports/pilot.json", report)
    return report


def export_latest(private, destination):
    """Upload latest full state per job only; intermediate receipts stay in Kaggle Output."""
    private, destination = Path(private), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for directory in sorted((private / "jobs").glob("*")):
        rows = [(read(path), path) for path in (directory / "receipts").glob("*.json")]
        if not rows:
            continue
        row, receipt = max(rows, key=lambda pair: pair[0]["step"])
        Store(directory, row["identity"]).verified(row)
        from minicells_runtime.store import safe

        files = [receipt, safe(directory, row["file"])]
        if (directory / "result.json").exists():
            files.append(directory / "result.json")
        else:
            files += list((directory / "logs").glob("*.json"))
        for path in files:
            target = destination / path.relative_to(private)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                try:
                    os.link(path, target)
                except OSError:
                    shutil.copyfile(path, target)
            elif digest(target) != digest(path):
                raise ContinuationError("PILOT_EXPORT_COLLISION")
    return destination


def automatic(output, assets):
    c = pilot_configuration()
    source = source_sha()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    run = canonical({"source": source, "config": canonical(c)})
    ident = {
        "experiment_id": ID,
        "run_id": "r3c-pilot-" + run[:16],
        "source_sha": source,
        "config_sha256": canonical(c),
    }
    atomic(output / "pilot-identity.json", ident)
    assets = Path(assets) if assets else output.parent / ("r3c-pilot-assets-" + source[:12])
    if not torch.cuda.is_available():
        raise ContinuationError("SELECT_KAGGLE_GPU_ACCELERATOR")
    load_secrets()
    from .publication import prepare_pilot_publication, publish_pilot_result

    print("[Pilot] checking existing HF/GitHub return paths before GPU work", flush=True)
    backend = HubBackend(c["input"]["asset_bundle_ref"]["hf_repo"])
    if not backend.info().private:
        raise ContinuationError("HF_PRIVATE_REPO_REQUIRED")
    backend.check_write()
    publication = prepare_pilot_publication(output / ("metadata-" + uuid.uuid4().hex[:8]), run)
    private = output / "private-checkpoints"
    session = uuid.uuid4().hex
    pointer = "r3c-moe-bridge/pilots/" + run + "/latest.json"
    restored = None
    known_ref = (
        read(output / "hf-reference.json") if (output / "hf-reference.json").exists() else None
    )
    if not list(private.glob("jobs/*/receipts/*.json")):
        revision = backend.info().sha
        latest = backend.optional(pointer, revision)
        if latest:
            restored = {**latest, "hf_repo": backend.repo, "hf_revision": revision}
            restore_segment(backend, restored, private, ident, session)
            known_ref = restored
            print(
                "[Pilot] verified pinned HF checkpoint bundle; completed updates will be reused",
                flush=True,
            )
    print("[Pilot] loading pinned existing assets only (no historical model weights)", flush=True)
    if (assets / "manifest.json").exists():
        from .data import load_assets

        data = load_assets(c, assets)
        print("[Pilot] reuse locally hash-verified assets; no asset download", flush=True)
    else:
        data = private_asset_subset(backend, c["input"]["asset_bundle_ref"], assets, c)
    failure = None
    try:
        report = execute_pilot(c, data, output, source, "cuda:0")
    except Exception as error:
        failure = type(error).__name__
        report = partial_report(c, output, source, error)
    export = export_latest(private, output / ("hf-export-" + session))
    try:
        if not any(export.rglob("*.pt")):
            report["checkpoint_storage_status"] = "NO_CHECKPOINTS_CREATED"
        elif known_ref and export_matches(backend, known_ref, export):
            # A fully completed remote run is immutable and needs no second upload.
            report["hf_checkpoint_reference"] = known_ref
            report["checkpoint_storage_status"] = (
                "HF_OBJECTS_AND_FULL_STATE_RESTORED"
                if restored
                else "HF_UPLOAD_REUSED_MANIFEST_VERIFIED"
            )
        else:
            report["hf_checkpoint_reference"] = publish_segment(
                backend, export, ident, session, pointer_path=pointer
            )
            report["checkpoint_storage_status"] = (
                "HF_UPLOADED_MANIFEST_VERIFIED_OBJECT_READBACK_PENDING"
            )
        if "hf_checkpoint_reference" in report:
            atomic(output / "hf-reference.json", report["hf_checkpoint_reference"])
    except Exception as error:
        from minicells_artifacts.hf_storage import error_code

        report["checkpoint_storage_status"] = "HF_UPLOAD_FAILED_LOCAL_CHECKPOINTS_RETAINED"
        report["storage_error"] = error_code(error)
        atomic(output / "reports/pilot.json", report)
        # Publish aggregate failure evidence; never re-run completed updates just to retry upload.
        publish_pilot_result(report, publication)
        raise ContinuationError(
            "HF_UPLOAD_FAILED_SAVE_KAGGLE_OUTPUT_RELAUNCH_REUSES_LOCAL_CHECKPOINTS"
        ) from None
    atomic(output / "reports/pilot.json", report)
    result = publish_pilot_result(report, publication)
    atomic(output / "reports/publication.json", result)
    if failure:
        raise ContinuationError("PILOT_FAILED_SAVED_CHECKPOINTS_" + failure) from None
    print("[Pilot] done. Retain Kaggle Output: " + str(output), flush=True)
    return {
        "status": report["status"],
        "logical_optimizer_updates": report["logical_optimizer_updates"],
        "logical_train_tokens": report["logical_train_tokens"],
        "checkpoint_storage_status": report["checkpoint_storage_status"],
        "results_branch": result["branch"],
        "scientific_decision": False,
        "formal_500_step_authorized": False,
    }


def export_matches(backend, ref, export):
    manifest = read(backend.get(ref["manifest_path"], ref["hf_revision"]))
    return {name: row["sha256"] for name, row in manifest["files"].items()} == {
        path.relative_to(export).as_posix(): digest(path)
        for path in export.rglob("*")
        if path.is_file()
    }


def partial_report(c, output, source, error):
    from minicells_artifacts.hf_storage import error_code
    import traceback

    private = Path(output) / "private-checkpoints"
    completed = []
    count = 0
    for directory in sorted((private / "jobs").glob("*")):
        receipts = [read(path) for path in (directory / "receipts").glob("*.json")]
        if receipts:
            count += max(row["step"] for row in receipts)
        if (directory / "result.json").exists():
            completed.append(read(directory / "result.json"))
    report = {
        "schema": "minicells.r3c.pilot.v1",
        "experiment_id": ID,
        "source_sha": source,
        "config_sha256": canonical(c),
        "status": "FAILED_PARTIAL_CHECKPOINTS_RETAINED",
        "selection": {"status": "NOT_MEASURED_ON_FAILURE"},
        "candidates": [row for row in completed if row["job"].startswith("balance-")],
        "arms": [row for row in completed if row["job"].startswith("arm-")],
        "logical_optimizer_updates": count,
        "logical_train_tokens": count
        * c["training"]["effective_batch"]
        * c["training"]["sequence_length"],
        "gpu_pilot_completed": False,
        "confirmation": "SEALED",
        "scientific_decision": False,
        "failure": {
            **error_code(error),
            "frames": [
                {"file": Path(frame.filename).name, "function": frame.name, "line": frame.lineno}
                for frame in traceback.extract_tb(error.__traceback__)
            ],
        },
        "physical_uncommitted_update_work": "NOT_MEASURED",
    }
    atomic(Path(output) / "reports/pilot.json", report)
    return report
