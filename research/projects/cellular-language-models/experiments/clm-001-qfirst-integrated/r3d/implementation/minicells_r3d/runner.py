"""Single automatic runner. Independent seeds run on at most two GPU processes."""

import argparse
import queue
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid
import torch
from minicells_artifacts.hf_storage import HubBackend
from minicells_r3c_moe_bridge.data import load_assets, private_asset_subset, train_batches
from minicells_r3c_moe_bridge.model import BridgeModel
from minicells_r3c_moe_bridge.pilot import load_secrets
from minicells_r3c_moe_bridge.publication import prepare_pilot_publication
from .common import (
    BASE,
    ROOT,
    ID,
    ARMS,
    atomic,
    read,
    canonical,
    source_sha,
    configuration,
    identity,
    ContinuationError,
)
from .connection import PathModel
from .training import Trainer
from .evaluation import final_evaluation, exposure, train_cost, sync
from .reports import conclusions, markdown
from .storage import (
    Store,
    parent_state,
    restore_parents,
    save,
    export,
    restore_latest,
    verified_manifest,
    publish_segment,
    public_publish,
)


def build_model(c, parent, seed, device, learned=None):
    model = PathModel(c, seed)
    expected = {k for k in model.state_dict() if not k.startswith("connection.")}
    if set(parent["model"]) != expected:
        raise ContinuationError("PARENT_MODEL_KEY_MISMATCH")
    full = model.state_dict()
    full.update(parent["model"])
    model.load_state_dict(full, strict=True)
    if learned is not None:
        model.connection.load_state_dict(learned, strict=True)
    return model.to(device)


@torch.no_grad()
def initial_parity(model, c, x, parent):
    old = BridgeModel(c["engine"], "D", model.seed).to(x.device)
    old.load_state_dict(parent["model"], strict=True)
    original = old(x)
    current = model(x)
    if not torch.equal(original.logits, current.logits) or any(
        not torch.equal(a.indices, b.indices) or not torch.equal(a.weights, b.weights)
        for a, b in zip(original.stages, current.stages)
    ):
        raise ContinuationError("ZERO_CORRECTION_D_EQUIVALENCE_FAILED")
    return {
        "exact_logits": True,
        "exact_actions": True,
        "exact_weights": True,
        "cell_calls_per_token": 8,
        "context_calls_per_token": 1,
    }


def job(c, data, output, source, seed, arm, parent, device, learned=None):
    output = Path(output)
    directory = output / "private-checkpoints/jobs" / ("seed-%s-arm-%s" % (seed, arm))
    ident = identity(c, source, seed, arm)
    store = Store(directory, ident)
    rows = [read(p) for p in (directory / "receipts").glob("*.json")]
    latest = max(rows, key=lambda r: r["step"]) if rows else None
    target = (
        0
        if arm == "F0"
        else c["training"]["a_steps"]
        if arm in ("L", "M")
        else c["training"]["b_steps"]
    )
    if (directory / "result.json").exists():
        result = read(directory / "result.json")
        if (
            result["identity"] != ident
            or latest is None
            or latest["step"] != target
            or latest["sha256"] != result["checkpoint"]["sha256"]
        ):
            raise ContinuationError("RESULT_IDENTITY_MISMATCH")
        store.verified(latest)
        print("[R3d] reuse completed", seed, arm, flush=True)
        return result
    model = build_model(c, parent, seed, device, learned)
    tr = Trainer(model, c, arm)
    curve = []
    lineage = {
        "parent_step": 500,
        "parent_sha256": ident["parent_checkpoint_sha256"],
        "cell_optimizer": "FRESH_IDENTICAL_FOR_F_P_J",
        "connection_source": "L" if arm in ("P", "J") else "ZERO",
    }
    if latest:
        tr.restore(store.verified(latest))
        curve = latest["observations"]["curve"]
        lineage = latest["observations"]["lineage"]
    else:
        latest = save(tr, store, curve, lineage)
    if tr.step > target or tr.data_cursor != tr.step * c["training"]["train_tokens_per_step"]:
        raise ContinuationError("STEP_TOKEN_BUDGET_MISMATCH")
    print("[R3d]", seed, arm, "restored", tr.step, "/", target, flush=True)
    while tr.step < target:
        sync(device)
        begin = time.perf_counter()
        batches = train_batches(data, c["engine"], tr.data_rng, device)
        row = tr.update(batches)
        sync(device)
        row["update_wall_seconds"] = time.perf_counter() - begin
        curve.append(row)
        if tr.step % c["training"]["checkpoint_every"] == 0 or tr.step == target:
            latest = save(tr, store, curve, lineage)
        if tr.step % c["training"]["print_every"] == 0 or tr.step == target:
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "arm": arm,
                        "step": tr.step,
                        "ce": row["ce"],
                        "connection_loss": row["connection_loss"],
                    }
                ),
                flush=True,
            )
    latest = save(tr, store, curve, lineage)
    quality, cost = final_evaluation(model, c, data, device)
    result = {
        "status": "COMPLETED",
        "identity": ident,
        "checkpoint": latest,
        "quality": quality,
        "inference_cost": cost,
        "training_cost": train_cost(curve, c, cost["per_token_forward_matmul_flops"]),
        "exposure": exposure(curve, c),
        "curve": curve,
        "lineage": lineage,
        "actual_kaggle_persistence_verified": False,
        "confirmation": "SEALED",
    }
    atomic(directory / "result.json", result)
    return result


def seed_run(c, output, assets, source, seed, device):
    data = load_assets(c["engine"], assets)
    parent = parent_state(c, Path(output) / "parent-r3c", seed)
    fresh = build_model(c, parent, seed, device)
    from minicells_r3c_moe_bridge.data import fixed_windows

    _, x, _ = next(iter(fixed_windows(data, c["engine"], "calibration", device)))
    parity = initial_parity(fresh, c, x, parent)
    atomic(Path(output) / "reports" / ("parity-%s.json" % seed), parity)
    del fresh
    for arm in ARMS:
        learned = None
        if arm in ("P", "J"):
            root = Path(output) / "private-checkpoints/jobs" / ("seed-%s-arm-L" % seed)
            r = read(root / "result.json")
            state = Store(root, identity(c, source, seed, "L")).verified(r["checkpoint"])
            learned = {
                k[len("connection.") :]: v
                for k, v in state["model"].items()
                if k.startswith("connection.")
            }
        job(c, data, output, source, seed, arm, parent, device, learned)


def gather(c, output, source):
    results = {}
    statuses = {}
    for seed in c["seeds"]:
        per = {}
        for arm in ARMS:
            p = (
                Path(output)
                / "private-checkpoints/jobs"
                / ("seed-%s-arm-%s" % (seed, arm))
                / "result.json"
            )
            if p.exists():
                per[arm] = read(p)
            statuses[str(seed) + "/" + arm] = "COMPLETED" if p.exists() else "NOT_COMPLETED"
        results[str(seed)] = per
    report = conclusions(c, results)
    failures = [read(p) for p in (Path(output) / "reports").glob("failure-*.json")]
    if failures:
        report["status"] = "PARTIAL_INVALID_RUN"
        report["hypotheses"] = {h: "INVALID_RUN" for h in ("H1", "H2", "H3", "H4")}
    report["failures"] = failures
    report.update(
        experiment_id=ID,
        source_sha=source,
        config_sha256=canonical(c),
        arm_status=statuses,
        actual_job_steps=sum(
            r["checkpoint"]["step"] for per in results.values() for r in per.values()
        ),
        actual_cell_updates=sum(
            r["training_cost"]["cell_optimizer_updates"]
            for per in results.values()
            for r in per.values()
        ),
        actual_main_training_tokens=sum(
            r["training_cost"]["main_training_tokens"]
            for per in results.values()
            for r in per.values()
        ),
        actual_policy_updates=sum(
            r["training_cost"]["policy_optimizer_updates"]
            for per in results.values()
            for r in per.values()
        ),
        confirmation="SEALED",
        actual_kaggle_persistence_verified=False,
    )
    folder = Path(output) / "reports"
    atomic(folder / "METRICS.json", report)
    atomic(
        folder / "ROUTING_SUMMARY.json",
        {"exposure": report["exposure"], "candidate_learning": report["candidate_learning"]},
    )
    atomic(folder / "TRAINING_COST.json", report["cost"])
    (folder / "SCIENTIFIC-FINDINGS.md").write_text(markdown(report))
    return report


def run_workers(c, config_path, output, assets, source, checkpoint_callback=None):
    count = min(c["training"]["max_workers"], torch.cuda.device_count())
    processes = []
    readers = []
    events = queue.Queue()

    def forward(process, index):
        for line in process.stdout:
            print("[GPU %s] %s" % (index, line.rstrip()), flush=True)
            if line.startswith("[R3D_SEED_COMPLETE] "):
                events.put(int(line.split()[-1]))

    for index in range(count):
        args = [
            sys.executable,
            "-u",
            str(ROOT / "scripts/run_r3d.py"),
            "--config",
            str(config_path),
            "--output",
            str(output),
            "--assets",
            str(assets),
            "--worker",
            str(index),
            "--workers",
            str(count),
        ]
        proc = subprocess.Popen(
            args,
            cwd=ROOT,
            env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        thread = threading.Thread(target=forward, args=(proc, index))
        thread.start()
        processes.append(proc)
        readers.append(thread)
    while any(p.poll() is None for p in processes):
        try:
            seed = events.get(timeout=0.25)
        except queue.Empty:
            continue
        if checkpoint_callback:
            checkpoint_callback(seed)
    codes = [p.wait() for p in processes]
    for thread in readers:
        thread.join()
    while not events.empty():
        seed = events.get_nowait()
        if checkpoint_callback:
            checkpoint_callback(seed)
    return codes


def automatic(c, config_path, output, assets):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source = source_sha()
    if not torch.cuda.is_available():
        raise ContinuationError("GPU_UNAVAILABLE_NO_TRAINING_STARTED")
    try:
        load_secrets()
    except (ImportError, ModuleNotFoundError):
        raise ContinuationError(
            "EXISTING_HF_TOKEN_AND_GITHUB_TOKEN_REQUIRED_USE_KAGGLE_SECRETS"
        ) from None
    backend = HubBackend(c["parent"]["hf_reference"]["hf_repo"])
    if not backend.info().private:
        raise ContinuationError("HF_PRIVATE_REPO_REQUIRED")
    backend.check_write()
    run = canonical({"source": source, "config": canonical(c)})
    ident = {
        "experiment_id": ID,
        "source_sha": source,
        "config_sha256": canonical(c),
        "run_id": run,
    }
    if (output / "identity.json").exists() and read(output / "identity.json") != ident:
        raise ContinuationError("OUTPUT_IDENTITY_CONFLICT_PRESERVE_EXISTING_RUN")
    atomic(output / "identity.json", ident)
    prepared = prepare_pilot_publication(output / ("metadata-" + uuid.uuid4().hex[:8]), run)
    private = output / "private-checkpoints"
    pointer = "r3d/" + run + "/latest.json"
    session = uuid.uuid4().hex
    prior = None
    if (output / "hf-reference.json").exists():
        prior = read(output / "hf-reference.json")
    if not list(private.glob("jobs/*/receipts/*.json")):
        revision = backend.info().sha
        ref = backend.optional(pointer, revision)
        if ref:
            prior = {**ref, "hf_repo": backend.repo, "hf_revision": revision}
            restore_latest(backend, prior, private, ident, session)
            atomic(output / "hf-reference.json", prior)
    restore_parents(c, backend, output / "parent-r3c")
    assets = (
        Path(assets)
        if assets
        else output.parent / ("r3c-assets-" + c["engine"]["input"]["assets_manifest_sha256"][:12])
    )
    if (assets / "manifest.json").exists():
        load_assets(c["engine"], assets)
    else:
        private_asset_subset(backend, c["engine"]["input"]["asset_bundle_ref"], assets, c["engine"])
    print(
        "[R3d] checks passed; automatically run all 3 Seeds / F0,L,M,F,P,J. No Pilot or intermediate release.",
        flush=True,
    )
    started = time.time()
    storage_failure = []

    def persist(completed_only):
        nonlocal prior
        exported = export(
            c,
            private,
            output / ("hf-export-" + uuid.uuid4().hex[:8]),
            completed_only=completed_only,
        )
        previous = verified_manifest(backend, prior, ident) if prior else None
        old = (
            {}
            if previous is None
            else {name: row["sha256"] for name, row in previous["files"].items()}
        )
        from .common import digest

        changed = any(
            old.get(p.relative_to(exported).as_posix()) != digest(p)
            for p in exported.rglob("*")
            if p.is_file()
        )
        if changed:
            prior = publish_segment(
                backend, exported, ident, session, pointer_path=pointer, prior_manifest=previous
            )
            atomic(output / "hf-reference.json", prior)

    def completed(seed):
        if storage_failure:
            return
        try:
            persist(True)
            print(
                "[R3d] Seed",
                seed,
                "private state committed; remaining work continues automatically",
                flush=True,
            )
        except Exception as error:
            from minicells_artifacts.hf_storage import error_code

            storage_failure.append(error_code(error))
            atomic(
                output / "STOP_STORAGE.json",
                {"status": "LOCAL_STATES_RETAINED", "error": storage_failure[-1]},
            )
            print(
                "[R3d] remote storage failed; retain Output. Active seed finishes; no new seed starts.",
                flush=True,
            )

    stop = output / "STOP_STORAGE.json"
    if stop.exists():
        stop.unlink()
    codes = run_workers(c, config_path, output, assets, source, completed)
    report = gather(c, output, source)
    manifest = {
        "identity": ident,
        "scope": "EXPLORATORY_DEVELOPMENT",
        "started_at_unix": started,
        "wall_seconds": time.time() - started,
        "worker_exit_codes": codes,
        "training_config": c["training"],
        "arm_status": report["arm_status"],
        "confirmation": "SEALED",
        "parent_reference": c["parent"],
        "checkpoint_storage": "LOCAL_FULL_STATE_SAVED",
        "actual_kaggle_persistence_verified": False,
        "hypotheses": report["hypotheses"],
    }
    atomic(output / "reports/RUN_MANIFEST.json", manifest)
    try:
        if storage_failure:
            raise ContinuationError("HF_UPLOAD_FAILED_LOCAL_STATE_RETAINED")
        persist(False)
        report["hf_reference"] = prior
        report["checkpoint_storage"] = "HF_COMMIT_AND_MANIFEST_VERIFIED_ONLY"
    except Exception as error:
        from minicells_artifacts.hf_storage import error_code

        report["storage_error"] = storage_failure[-1] if storage_failure else error_code(error)
        report["checkpoint_storage"] = "UPLOAD_FAILED_LOCAL_CHECKPOINTS_RETAINED"
    atomic(output / "reports/METRICS.json", report)
    publication = public_publish(report, prepared)
    atomic(output / "reports/publication.json", publication)
    manifest.update(
        checkpoint_storage=report["checkpoint_storage"], hf_reference=prior, publication=publication
    )
    atomic(output / "reports/RUN_MANIFEST.json", manifest)
    if any(codes):
        raise ContinuationError("PARTIAL_OUTPUT_SAVED_RESUME_REMAINING_STEPS")
    if "storage_error" in report:
        raise ContinuationError("HF_UPLOAD_FAILED_SAVE_KAGGLE_OUTPUT")
    return {
        "status": report["status"],
        "hypotheses": report["hypotheses"],
        "results_branch": publication["branch"],
        "confirmation": "SEALED",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(BASE / "config/frozen.json"))
    p.add_argument("--run-all", action="store_true")
    p.add_argument("--output", default="/kaggle/working/minicells-r3d-output")
    p.add_argument("--assets")
    p.add_argument("--worker", type=int)
    p.add_argument("--workers", type=int, default=1)
    a = p.parse_args()
    c = configuration(a.config)
    if a.worker is not None:
        torch.set_num_threads(1)
        torch.cuda.set_device(a.worker)
        failures = []
        for seed in c["seeds"][a.worker :: a.workers]:
            if (Path(a.output) / "STOP_STORAGE.json").exists():
                failures.append(seed)
                break
            try:
                seed_run(c, a.output, a.assets, source_sha(), seed, "cuda:" + str(a.worker))
                failed = Path(a.output) / "reports" / ("failure-%s.json" % seed)
                if failed.exists():
                    failed.unlink()
                print("[R3D_SEED_COMPLETE]", seed, flush=True)
            except Exception as error:
                atomic(
                    Path(a.output) / "reports" / ("failure-%s.json" % seed),
                    {
                        "seed": seed,
                        "type": type(error).__name__,
                        "error": str(error)[:500],
                        "status": "INVALID_RUN",
                    },
                )
                failures.append(seed)
                print("[R3d] seed failed", seed, type(error).__name__, str(error)[:500], flush=True)
        if failures:
            raise ContinuationError("SEED_FAILURES", str(failures))
    elif a.run_all:
        print(json.dumps(automatic(c, Path(a.config).resolve(), a.output, a.assets)), flush=True)
    else:
        p.error("use --run-all; checks and execution are automatic")
