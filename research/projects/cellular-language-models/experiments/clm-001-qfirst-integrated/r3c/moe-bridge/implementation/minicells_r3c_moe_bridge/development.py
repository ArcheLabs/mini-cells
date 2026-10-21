"""Bounded exploratory development; unchanged joint Trainer, no scientific release.

Seed blocks are serialized for publication. Independent models within a block use
at most two subprocesses/GPUs. E is explicitly derived from C when lambda is zero.
"""

from pathlib import Path
import os
import random
import shutil
import subprocess
import sys
import threading
import uuid
from copy import deepcopy
import numpy as np
import torch
from minicells_artifacts.hf_storage import HubBackend
from minicells_runtime.store import safe
from .common import (
    BASE,
    ROOT,
    ID,
    atomic,
    canonical,
    digest,
    read,
    source_sha,
    validate,
    ContinuationError,
    fingerprint,
)
from .checkpoint import Store, publish_segment, restore_segment
from .data import load_assets, private_asset_subset, train_batches, fixed_windows
from .evaluation import evaluate
from .instrumentation import MatmulAudit, parameters
from .model import BridgeModel
from .training import Trainer
from .pilot import load_secrets, timed_update, _receipts
from .runner import identity
from .statistics import paired_seed_comparison, seed_variation


def configuration():
    c = read(BASE / "config/development.json")
    validate(c)
    d = c["development"]
    if (
        not c["release"].get("exploratory_development_authorized")
        or d["authorization"]["scope"] != "EXPLORATORY_DEVELOPMENT_ONLY"
    ):
        raise ContinuationError("DEVELOPMENT_NOT_AUTHORIZED")
    if (
        d["steps"] != 500
        or d["seeds"] != [151001301, 151001302, 151001303]
        or d["physical_arms"] != ["A", "B", "C", "D"]
        or d["aliases"] != {"E": "C"}
        or c["balance"]["coefficient"] != 0
    ):
        raise ContinuationError("DEVELOPMENT_MATRIX_CHANGED")
    if (
        d["checkpoint_steps"] != [0, 50, 100, 200, 350, 500]
        or d["recovery_interval"] != 50
        or d["new_update_upper_bound"] != 6000
    ):
        raise ContinuationError("DEVELOPMENT_BUDGET_CHANGED")
    if (
        c["release"]["formal_development_authorized"]
        or c["statistics"]["formal_confirmation"]
        or c["scientific_decision"]
    ):
        raise ContinuationError("EXPLORATORY_ONLY_CONFIRMATION_SEALED")
    if d["evaluation"] != {
        "when": "TERMINAL_CHECKPOINT_ONLY",
        "roles": ["calibration", "development"],
        "early_stopping": False,
        "confirmation": "SEALED",
    }:
        raise ContinuationError("DEVELOPMENT_EVALUATION_CHANGED")
    pilot = read(BASE / "config/pilot.json")
    parent = d["parent_pilot"]
    if (
        canonical(pilot) != parent["config_sha256"]
        or any(c[key] != pilot[key] for key in ("model", "routing", "training"))
        or c["input"] != pilot["input"]
    ):
        raise ContinuationError("PILOT_TRAINING_COMPATIBILITY_CHANGED")
    for name, sha in parent["core_file_sha256"].items():
        if digest(ROOT / name) != sha:
            raise ContinuationError("SCIENTIFIC_CORE_CHANGED_SINCE_PILOT", name)
    return c


def job_name(seed, arm):
    return "seed-" + str(seed) + "-arm-" + arm


def job_identity(c, source, seed, arm):
    return {
        **identity(c, arm, seed, source),
        "phase": "EXPLORATORY_DEVELOPMENT",
        "effective_balance_coefficient": c["balance"]["coefficient"],
        "target_steps": c["development"]["steps"],
    }


def parent_state(c, parent_root, seed, arm, device):
    d = c["development"]
    proof = d["parent_pilot"]
    if not d["reuse_pilot_seed"] or seed != proof["seed"]:
        return None, None
    directory = Path(parent_root) / "jobs" / ("arm-" + arm)
    result = read(directory / "result.json")
    row = result["checkpoint"]
    pilot = read(BASE / "config/pilot.json")
    expected = {
        **identity(pilot, arm, seed, proof["source_sha"]),
        "pilot_job": "arm-" + arm,
        "effective_balance_coefficient": proof["selected_coefficient"],
    }
    if (
        row["identity"] != expected
        or row["sha256"] != proof["checkpoint_sha256"][arm]
        or row["step"] != 5
    ):
        raise ContinuationError("PILOT_PARENT_IDENTITY_MISMATCH")
    state = Store(directory, expected).verified(row)
    if (
        state["step"] != row["step"]
        or state["data_cursor"]
        != state["step"] * c["training"]["effective_batch"] * c["training"]["sequence_length"]
    ):
        raise ContinuationError("PILOT_PARENT_CURSOR_MISMATCH")
    if (
        str(device).startswith("cuda")
        and len(state["global_cuda_rng"]) != torch.cuda.device_count()
    ):
        print(
            "[Development] pilot GPU layout differs; preserve parent and start seeded model at step 0",
            flush=True,
        )
        return None, {"imported_steps": 0, "reason": "PILOT_GPU_LAYOUT_DIFFERS"}
    return state, {
        "imported_steps": state["step"],
        "parent_source_sha": proof["source_sha"],
        "checkpoint_sha256": row["sha256"],
        "state_fingerprint": row["state_fingerprint"],
        "training_history": "pilot steps 1..5 count toward the same 500-step budget",
        "curve": [compact(log, "PILOT") for log in result["logs"]],
    }


def compact(row, origin="DEVELOPMENT"):
    keys = (
        "step",
        "ce",
        "balance_loss",
        "gate_task_grad_norm",
        "gate_balance_grad_norm",
        "cell_grad_norms",
        "train_tokens",
        "wall_seconds",
        "cuda_time_ms",
        "tokens_per_second",
        "peak_gpu_bytes",
    )
    return {**{k: row.get(k, "NOT_MEASURED") for k in keys}, "origin": origin}


def save(trainer, store, curve, lineage):
    row = store.save(trainer, observations={"curve": curve, "lineage": lineage})
    atomic(store.root / "curve.json", {"identity": store.identity, "rows": curve})
    atomic(
        store.root / "progress.json",
        {"identity": store.identity, "step": trainer.step, "lineage": lineage},
    )
    atomic(store.root / "latest.json", row)
    return row


def train_job(c, data, output, source, seed, arm, device, parent_root=None):
    d = c["development"]
    name = job_name(seed, arm)
    ident = job_identity(c, source, seed, arm)
    store = Store(Path(output) / "private-checkpoints/jobs" / name, ident)
    rows = _receipts(store)
    latest = rows[-1] if rows else None
    result_path = store.root / "result.json"
    if latest and latest["step"] > d["steps"]:
        raise ContinuationError("DEVELOPMENT_STEP_OVER_BUDGET")
    if result_path.exists():
        result = read(result_path)
        if (
            not latest
            or result["identity"] != ident
            or latest["step"] != d["steps"]
            or result["checkpoint"]["sha256"] != latest["sha256"]
        ):
            raise ContinuationError("DEVELOPMENT_RESULT_IDENTITY_MISMATCH")
        store.verified(latest)
        print("[Development] reuse completed " + name, flush=True)
        return result
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model_c = deepcopy(c)
    tr = Trainer(BridgeModel(model_c, arm, seed).to(device))
    curve = []
    lineage = {"imported_steps": 0}
    if latest:
        state = store.verified(latest)
        if (
            str(device).startswith("cuda")
            and len(state["global_cuda_rng"]) != torch.cuda.device_count()
        ):
            raise ContinuationError("RESTORE_SAME_GPU_LAYOUT_AS_CHECKPOINT")
        tr.restore(state)
        if (store.root / "curve.json").exists():
            curve = [
                row for row in read(store.root / "curve.json")["rows"] if row["step"] <= tr.step
            ]
        if (store.root / "progress.json").exists():
            lineage = read(store.root / "progress.json")["lineage"]
        if "observations" in latest:
            curve = latest["observations"]["curve"]
            lineage = latest["observations"]["lineage"]
    else:
        latest = save(tr, store, curve, lineage)
    if tr.step == 0 and parent_root is not None:
        state, info = parent_state(c, parent_root, seed, arm, device)
        if state is not None:
            tr.restore(state)
            if fingerprint(tr.state()) != fingerprint(state):
                raise ContinuationError("PILOT_RESTORE_STATE_DRIFT")
            lineage = {k: v for k, v in info.items() if k != "curve"}
            curve = info["curve"]
            latest = save(tr, store, curve, lineage)
        elif info:
            lineage = info
    print(
        "[Development] " + name + " restored_step=" + str(tr.step) + " / " + str(d["steps"]),
        flush=True,
    )
    while tr.step < d["steps"]:
        if stop_requested(output) and tr.step == latest["step"]:
            return {"status": "PAUSED_AT_COMPLETE_CHECKPOINT", "identity": ident, "step": tr.step}
        batches = train_batches(data, c, tr.data_rng, device)
        row = timed_update(tr, batches, device)
        curve.append(compact(row))
        if tr.step % d["recovery_interval"] == 0 or tr.step in d["checkpoint_steps"]:
            latest = save(tr, store, curve, lineage)
            atomic(store.root / "routing-gradient-snapshot" / ("step-%06d.json" % tr.step), row)
        if tr.step % d["print_interval"] == 0 or tr.step == d["steps"]:
            print(
                "[Development] "
                + name
                + " "
                + str(tr.step)
                + "/"
                + str(d["steps"])
                + " CE="
                + str(round(row["ce"], 5)),
                flush=True,
            )
    latest = save(tr, store, curve, lineage)
    # Terminal checkpoint only: no Development peeking, stopping or model selection mid-training.
    quality = {}
    for role in d["evaluation"]["roles"]:
        quality[role] = evaluate(tr.model, fixed_windows(data, c, role, device), seed + 9001)
    sample = train_batches(data, c, np.random.default_rng(seed + 4999), device)[0][0]
    with torch.no_grad(), MatmulAudit(tr.model) as audit:
        out = tr.model(sample, route_rng=torch.Generator().manual_seed(seed + 4999))
    result = {
        "status": "COMPLETED",
        "identity": ident,
        "seed": seed,
        "arm": arm,
        "step": tr.step,
        "train_tokens": tr.data_cursor,
        "pilot_steps_imported": lineage.get("imported_steps", 0),
        "lineage": lineage,
        "parameters": parameters(tr.model),
        "cost": audit.finish(out),
        "checkpoint": latest,
        "quality": quality,
        "curve": curve,
        "confirmation": "SEALED",
        "scientific_decision": False,
    }
    atomic(result_path, result)
    return result


def stop_requested(output):
    output = Path(output)
    return (
        (output / "STOP_AFTER_CHECKPOINT.json").exists()
        or any(output.glob("STOP-worker-*.json"))
        or any(output.glob("STOP-parent-*.json"))
    )


def worker(output, assets, seed, index, count):
    c = configuration()
    torch.set_num_threads(1)
    torch.cuda.set_device(index)
    data = load_assets(c, assets)
    source = source_sha()
    for arm in c["development"]["physical_arms"][index::count]:
        if stop_requested(output):
            break
        try:
            train_job(
                c,
                data,
                output,
                source,
                seed,
                arm,
                "cuda:" + str(index),
                Path(output) / "parent-pilot",
            )
        except Exception:
            atomic(
                Path(output) / ("STOP-worker-" + str(index) + ".json"),
                {"reason": "WORKER_FAILED", "worker": index, "seed": seed, "arm": arm},
            )
            raise


def seed_workers(c, output, assets, seed, count):
    processes = []
    readers = []

    def forward(process, index):
        for line in process.stdout:
            print("[GPU " + str(index) + "] " + line.rstrip(), flush=True)

    for index in range(count):
        args = [
            sys.executable,
            "-u",
            str(BASE / "run.py"),
            "--mode",
            "development-worker",
            "--output",
            str(output),
            "--assets",
            str(assets),
            "--seed",
            str(seed),
            "--worker-index",
            str(index),
            "--worker-count",
            str(count),
        ]
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        process = subprocess.Popen(
            args, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        thread = threading.Thread(target=forward, args=(process, index))
        thread.start()
        processes.append(process)
        readers.append(thread)
    codes = [None] * len(processes)

    def wait_child(process, index):
        codes[index] = process.wait()
        if codes[index]:
            atomic(
                Path(output) / ("STOP-parent-" + str(index) + ".json"),
                {"reason": "WORKER_EXITED", "code": codes[index]},
            )

    waiters = [
        threading.Thread(target=wait_child, args=(process, index))
        for index, process in enumerate(processes)
    ]
    for thread in waiters:
        thread.start()
    for thread in waiters:
        thread.join()
    for thread in readers:
        thread.join()
    return codes


def verified_manifest(backend, ref, identity=None):
    if backend.repo != ref["hf_repo"]:
        raise ContinuationError("R3C_REPOSITORY_IDENTITY_MISMATCH")
    path = backend.get(ref["manifest_path"], ref["hf_revision"])
    if digest(path) != ref["manifest_sha256"]:
        raise ContinuationError("HF_READBACK_MISMATCH")
    manifest = read(path)
    if identity is not None and manifest["identity"] != identity:
        raise ContinuationError("R3C_CHECKPOINT_IDENTITY_MISMATCH")
    return manifest


def restore_latest(backend, ref, root, ident, session):
    manifest = verified_manifest(backend, ref, ident)
    selected = set()
    for name, row in manifest["files"].items():
        if not name.endswith("/latest.json"):
            continue
        path = backend.get(row["object"], ref["hf_revision"])
        if digest(path) != row["sha256"]:
            raise ContinuationError("HF_READBACK_MISMATCH")
        latest = read(path)
        directory = Path(name).parent
        selected.update((name, (directory / latest["file"]).as_posix()))
        # The Store's deterministic receipt key avoids fetching all historical receipts.
        receipt = Store(directory, latest["identity"]).receipt_file(latest["step"]).as_posix()
        if receipt not in manifest["files"]:
            raise ContinuationError("LATEST_RECEIPT_MISSING")
        selected.add(receipt)
        for leaf in ("curve.json", "progress.json", "result.json"):
            candidate = (directory / leaf).as_posix()
            if candidate in manifest["files"]:
                selected.add(candidate)
    restore_segment(backend, ref, root, ident, session, include_paths=selected)
    return manifest


def link_file(path, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if digest(target) != digest(path):
            raise ContinuationError("DEVELOPMENT_EXPORT_CONFLICT")
        return
    try:
        os.link(path, target)
    except OSError:
        shutil.copyfile(path, target)


def export_checkpoints(c, private, destination):
    private, destination = Path(private), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    keep = set(c["development"]["checkpoint_steps"])
    for directory in sorted((private / "jobs").glob("*")):
        rows = [(read(path), path) for path in (directory / "receipts").glob("*.json")]
        if not rows:
            continue
        last = max(row["step"] for row, path in rows)
        latest = max(rows, key=lambda pair: pair[0]["step"])[0]
        Store(directory, latest["identity"]).verified(latest)
        atomic(directory / "latest.json", latest)
        for row, path in rows:
            if row["step"] not in keep and row["step"] != last:
                continue
            Store(directory, row["identity"]).verified(row)
            for source in (path, safe(directory, row["file"])):
                link_file(source, destination / source.relative_to(private))
        for leaf in ("latest.json", "curve.json", "progress.json", "result.json"):
            source = directory / leaf
            if source.exists():
                link_file(source, destination / source.relative_to(private))
    return destination


def collect(c, output, source):
    d = c["development"]
    results = {}
    counts = {}
    comparisons = []
    for seed in d["seeds"]:
        per = {}
        for arm in d["physical_arms"]:
            directory = Path(output) / "private-checkpoints/jobs" / job_name(seed, arm)
            rows = [read(path) for path in (directory / "receipts").glob("*.json")]
            counts[job_name(seed, arm)] = max((row["step"] for row in rows), default=0)
            path = directory / "result.json"
            if path.exists():
                per[arm] = read(path)
        if set(per) == set(d["physical_arms"]):
            paired = {a: row["quality"]["development"] for a, row in per.items()}
            paired["E"] = paired["C"]
            comparison = paired_seed_comparison(
                paired, seed, c["statistics"]["bootstrap_replicates"]
            )
            comparison["comparisons"]["C-E"] = {
                "mean": 0.0,
                "status": "STRUCTURALLY_IDENTICAL_NOT_AN_ABLATION",
                "independent_model_runs": False,
            }
            comparisons.append(comparison)
        results[str(seed)] = per
    complete = all(len(per) == len(d["physical_arms"]) for per in results.values())
    return {
        "schema": "minicells.r3c.development.v1",
        "experiment_id": ID,
        "source_sha": source,
        "config_sha256": canonical(c),
        "status": "EXPLORATORY_DEVELOPMENT_COMPLETE" if complete else "DEVELOPMENT_PARTIAL_SAVED",
        "results": results,
        "completed_steps": counts,
        "committed_physical_updates": sum(counts.values()),
        "physical_arms": d["physical_arms"],
        "aliases": {
            "E": {
                "source": "C",
                "reason": "lambda_balance=0; identical training rule; no extra model or independent repetition",
            }
        },
        "paired_analysis": seed_variation(comparisons)
        if comparisons
        else {"status": "NOT_MEASURED"},
        "mechanism_diagnostics": "DEFERRED_NOT_EVIDENCE_OF_SPECIALIZATION",
        "scientific_thresholds_frozen": False,
        "formal_confirmation": False,
        "confirmation": "SEALED",
        "scientific_decision": False,
        "actual_kaggle_persistence_verified": False,
    }


def public_summary(report):
    if report.get("schema") != "minicells.r3c.development.v1":
        raise ValueError("known development schema required")

    def result(row):
        import math

        means = {
            role: float(
                np.average(
                    [v["mean_nll"] for v in values.values()],
                    weights=[v["tokens"] for v in values.values()],
                )
            )
            for role, values in row["quality"].items()
        }
        return {
            "perplexity": {
                role: math.exp(value) if value < 700 else "OVERFLOW"
                for role, value in means.items()
            },
            "arm": row["arm"],
            "seed": row["seed"],
            "step": row["step"],
            "train_tokens": row["train_tokens"],
            "pilot_steps_imported": row["pilot_steps_imported"],
            "parameters": row["parameters"],
            "cost": row["cost"],
            "checkpoint_sha256": row["checkpoint"]["sha256"],
            "mean_nll": {
                role: float(
                    np.average(
                        [v["mean_nll"] for v in values.values()],
                        weights=[v["tokens"] for v in values.values()],
                    )
                )
                for role, values in row["quality"].items()
            },
            "curve": row["curve"],
        }

    values = {
        seed: {arm: result(row) for arm, row in per.items()}
        for seed, per in report["results"].items()
    }
    for seed, per in values.items():
        if "C" in per:
            per["E"] = {
                "arm": "E",
                "seed": int(seed),
                "status": "DERIVED_EQUIVALENT_NO_EXTRA_TRAINING",
                "source_arm": "C",
                "source_checkpoint_sha256": per["C"]["checkpoint_sha256"],
                "mean_nll": per["C"]["mean_nll"],
                "perplexity": per["C"]["perplexity"],
                "independent_model_run": False,
            }
    return {
        **{k: v for k, v in report.items() if k != "results"},
        "schema": "minicells.r3c.public-development.v1",
        "results": values,
    }


def automatic(output, assets):
    c = configuration()
    d = c["development"]
    source = source_sha()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    assets = (
        Path(assets)
        if assets
        else output.parent / ("r3c-assets-" + c["input"]["assets_manifest_sha256"][:12])
    )
    run = canonical({"source_sha": source, "config": canonical(c)})
    ident = {
        "experiment_id": ID,
        "phase": "EXPLORATORY_DEVELOPMENT",
        "source_sha": source,
        "config_sha256": canonical(c),
        "run_id": run,
    }
    atomic(output / "identity.json", ident)
    if not torch.cuda.is_available():
        raise ContinuationError("SELECT_GPU_ACCELERATOR")
    count = min(d["max_workers"], torch.cuda.device_count())
    load_secrets()
    from .publication import prepare_pilot_publication, publish_aggregate

    backend = HubBackend(c["input"]["asset_bundle_ref"]["hf_repo"])
    if not backend.info().private:
        raise ContinuationError("HF_PRIVATE_REPO_REQUIRED")
    backend.check_write()
    prepared = prepare_pilot_publication(output / ("metadata-" + uuid.uuid4().hex[:8]), run)
    private = output / "private-checkpoints"
    session = uuid.uuid4().hex
    pointer = "r3c-moe-bridge/development/" + run + "/latest.json"
    prior = None
    if (output / "hf-reference.json").exists():
        prior = read(output / "hf-reference.json")
    if not list(private.glob("jobs/*/receipts/*.json")):
        revision = backend.info().sha
        latest = backend.optional(pointer, revision)
        if latest:
            prior = {**latest, "hf_repo": backend.repo, "hf_revision": revision}
            restore_latest(backend, prior, private, ident, session)
            atomic(output / "hf-reference.json", prior)
    # Full parent byte/state checks happen once before launching any training worker.
    parent_ref = d["parent_pilot"]["hf_reference"]
    parent_manifest = verified_manifest(backend, parent_ref)
    selected = {
        name
        for name in parent_manifest["files"]
        if any(name.startswith("jobs/arm-" + arm + "/") for arm in d["physical_arms"])
    }
    restore_segment(
        backend,
        parent_ref,
        output / "parent-pilot",
        parent_manifest["identity"],
        session,
        include_paths=selected,
    )
    if (assets / "manifest.json").exists():
        load_assets(c, assets)
    else:
        private_asset_subset(backend, c["input"]["asset_bundle_ref"], assets, c)
    for path in (
        list(output.glob("STOP-worker-*.json"))
        + list(output.glob("STOP-parent-*.json"))
        + [output / "STOP_AFTER_CHECKPOINT.json"]
    ):
        if path.exists():
            path.unlink()
    print(
        "[Development] 500 steps x 3 Seeds; physical A/B/C/D, E aliases C; workers=" + str(count),
        flush=True,
    )
    for seed in d["seeds"]:
        codes = seed_workers(c, output, assets, seed, count)
        report = collect(c, output, source)
        export = export_checkpoints(c, private, output / ("hf-export-" + uuid.uuid4().hex[:8]))
        try:
            previous = verified_manifest(backend, prior, ident) if prior else None
            local = {
                path.relative_to(export).as_posix(): digest(path)
                for path in export.rglob("*")
                if path.is_file()
            }
            old = (
                {}
                if previous is None
                else {name: row["sha256"] for name, row in previous["files"].items()}
            )
            if not prior or any(old.get(name) != sha for name, sha in local.items()):
                prior = publish_segment(
                    backend, export, ident, session, pointer_path=pointer, prior_manifest=previous
                )
                atomic(output / "hf-reference.json", prior)
            report["hf_checkpoint_reference"] = prior
            report["checkpoint_storage_status"] = (
                "HF_COMMIT_AND_MANIFEST_VERIFIED_OBJECT_SESSION_PROOF_PENDING"
            )
        except Exception as error:
            from minicells_artifacts.hf_storage import error_code

            report["storage_error"] = error_code(error)
            report["checkpoint_storage_status"] = "HF_UPLOAD_FAILED_LOCAL_STATE_RETAINED"
            atomic(output / "reports/development.json", report)
            publish_aggregate(public_summary(report), prepared, "r3c-development")
            raise ContinuationError(
                "HF_UPLOAD_FAILED_SAVE_OUTPUT_NO_COMPLETED_STEPS_RETRAINED"
            ) from None
        atomic(output / "reports/development.json", report)
        publication = publish_aggregate(public_summary(report), prepared, "r3c-development")
        atomic(output / "reports/publication.json", publication)
        if any(codes) or stop_requested(output):
            raise ContinuationError("DEVELOPMENT_PARTIAL_SAVED_RESUME_FROM_CHECKPOINT")
    return {
        "status": report["status"],
        "committed_physical_updates": report["committed_physical_updates"],
        "logical_arms": ["A", "B", "C", "D", "E"],
        "aliases": report["aliases"],
        "results_branch": publication["branch"],
        "confirmation": "SEALED",
        "scientific_decision": False,
    }
