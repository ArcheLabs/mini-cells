"""EPCLM-001 v0.3: one explicitly authorized seed, three historical workers."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

from .common import BASE, ROOT, digest, native, sha, source, write


def configuration():
    c = json.loads((BASE / "retrain_config.json").read_text())
    if (
        list(c["models"]) != ["T1", "X3", "X4"]
        or c["seed"] != 91002
        or c["training_tokens"] != 10000000
        or c["steps"] != 10000
        or c["gpu_budget_seconds"] != 7200
    ):
        raise RuntimeError("CONTROLLED_RETRAIN_AUTHORIZATION_CHANGED")
    for name, value in c["legacy_source_files"].items():
        if digest(ROOT / name) != value:
            raise RuntimeError("FROZEN_NATIVE_SOURCE_CHANGED")
    if digest(BASE / "PROTOCOL.v0.3.md") != c["amendment_sha256"]:
        raise RuntimeError("RETRAIN_AMENDMENT_CHANGED")
    return c


def manifest(root, c):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "TRAINING_MANIFEST.json"
    if path.exists():
        value = json.loads(path.read_text())
        if value["config_sha256"] != sha(c):
            raise RuntimeError("EXISTING_RUN_CONFIG_MISMATCH_NO_RESTART")
        return value
    value = {
        "experiment_id": "EPCLM-001",
        "asset_generation": "CONTROLLED_RETRAIN_001",
        "scope": "CONTROLLED_RETRAIN_001",
        "run_id": uuid.uuid4().hex,
        "source_sha": source(),
        "config_sha256": sha(c),
        "seed": 91002,
        "models": {k: "NOT_STARTED" for k in c["models"]},
        "gpu_budget_seconds": 7200,
        "gpu_seconds_charged": 0.0,
        "status": "PENDING_R0",
        "control_source_files": {
            str(p.relative_to(ROOT)): digest(p)
            for p in sorted((BASE / "implementation/epclm001").glob("retrain*.py"))
        },
        "optimizer_updates": 0,
        "config": c,
    }
    write(path, value)
    return value


def preflight(root, cache):
    c = configuration()
    from .retrain_assets import restore_controller

    restore_error = None
    try:
        restore_controller(root, c)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        restore_error = type(error).__name__
    m = manifest(root, c)
    checks = {"controller_restore": "PASS" if restore_error is None else "BLOCKED_" + restore_error}
    checks["source"] = "PASS"
    try:
        import torch

        base, _, phase_c = native()
        for key, name in c["models"].items():
            model = base.build_model(name) if key == "T1" else phase_c.build_phase_c(name)
            if sum(p.numel() for p in model.parameters()) != c["parameter_counts"][key]:
                raise RuntimeError("MODEL_PARAMETER_MISMATCH")
        checks["configuration"] = "PASS"
        m["environment"] = base.environment_record()
        import importlib.metadata

        m["library_versions"] = {}
        for library in ("torch", "numpy", "datasets", "tokenizers", "huggingface_hub"):
            try:
                m["library_versions"][library] = importlib.metadata.version(library)
            except importlib.metadata.PackageNotFoundError:
                m["library_versions"][library] = None
        checks["cuda"] = "PASS" if torch.cuda.is_available() else "CUDA_UNAVAILABLE"
        from .retrain_data import prepare

        prepare(cache, c)
        checks["data"] = "HISTORICAL_DATA_IDENTITY_MATCHED"
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        checks["data"] = "BLOCKED_" + type(error).__name__
        checks["data_error"] = str(error).splitlines()[0][:300]
    try:
        from .retrain_assets import probe

        checks["persistence"] = probe(root, c, m["run_id"])
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        checks["persistence"] = {
            "status": "BLOCKED",
            "code": type(error).__name__,
            "message": str(error).splitlines()[0][:200],
        }
    ready = (
        checks.get("controller_restore") == "PASS"
        and checks.get("configuration") == "PASS"
        and checks.get("cuda") == "PASS"
        and checks.get("data") == "HISTORICAL_DATA_IDENTITY_MATCHED"
        and checks["persistence"].get("status") == "PERSISTENCE_PREFLIGHT_PASS"
    )
    m.update(
        status="R0_PASS" if ready else "R0_BLOCKED_NO_TRAINING",
        preflight=checks,
        cache=str(Path(cache).resolve()),
    )
    write(Path(root) / "TRAINING_MANIFEST.json", m)
    if not (Path(root) / "ASSET_MANIFEST.json").exists():
        write(
            Path(root) / "ASSET_MANIFEST.json",
            {
                "asset_generation": "CONTROLLED_RETRAIN_001",
                "assets": [
                    {
                        "model": k,
                        "status": "PENDING",
                        "readback_verified": False,
                        "seed": 91002,
                        "training_tokens": 0,
                        "target_training_tokens": 10000000,
                        "checkpoint_sha256": None,
                        "inference_checkpoint_sha256": None,
                        "training_source_sha": None,
                        "tokenizer_sha256": c["tokenizer_sha256"],
                        "dataset_manifest_sha256": None,
                        "persistent_artifact_uri": None,
                    }
                    for k in c["models"]
                ],
            },
        )
    if not (Path(root) / "QUALITY_RESULTS.json").exists():
        write(
            Path(root) / "QUALITY_RESULTS.json",
            {
                "status": "NOT_RUN_R0_PENDING",
                "models": {k: {"nll": None, "ppl": None, "status": "NOT_RUN"} for k in c["models"]},
            },
        )
    return m


def validate_resume(path, c, m, name):
    import torch

    x = torch.load(path, map_location="cpu", weights_only=True)
    expected = {
        "model": name,
        "seed": 91002,
        "profile": "baseline",
        "asset_generation": "CONTROLLED_RETRAIN_001",
        "controlled_config_sha256": sha(c),
        "retrain_run_id": m["run_id"],
        "training_source_sha": m["source_sha"],
    }
    expected["corpus_manifest_sha256"] = hashlib.sha256(
        json.dumps(c["historical_corpus_manifest"], sort_keys=True).encode()
    ).hexdigest()
    if (
        any(x.get(k) != v for k, v in expected.items())
        or not 0 <= x["step"] <= 10000
        or x["consumed_tokens"] != 1000 * x["step"]
    ):
        raise RuntimeError("RESUME_IDENTITY_MISMATCH")
    return x


def worker(root, model_id, seconds):
    import torch

    from .retrain_data import load_verified
    from .retrain_guard import ResourceStop, TrainingGuard

    c = configuration()
    m = manifest(root, c)
    if m["status"] != "R1_RUNNING" or model_id not in c["models"]:
        raise RuntimeError("WORKER_GATE_OR_MODEL_INVALID")
    for name, value in m["control_source_files"].items():
        if digest(ROOT / name) != value:
            raise RuntimeError("CONTROL_SOURCE_CHANGED_SINCE_R0")
    if (
        m["models"].get(model_id) != "RUNNING"
        or m.get("worker_budget_seconds") != seconds
        or m["gpu_seconds_charged"] != 7200
    ):
        raise RuntimeError("WORKER_RESERVATION_REQUIRED")
    base, _, _ = native()
    from .common import NATIVE

    if str(NATIVE) not in sys.path:
        sys.path.insert(0, str(NATIVE))
    import native_clm_replication_runlib as historical

    from .publication import parent

    _, bind = parent()
    corpus = load_verified(m["cache"], c)
    name = c["models"][model_id]
    path = Path(root) / "models" / name / "resume.pt"
    guard = TrainingGuard(
        seconds, Path(root) / "WORKER_RESOURCE.json", m["source_sha"], sha(c), m["run_id"]
    )
    if path.exists():
        guard.actual_updates = validate_resume(path, c, m, name)["actual_optimizer_updates"]
    old_train, old_save, old_eval = base.train_one, base._atomic_save, base.evaluate

    def save(payload, path):
        return guard.checkpoint(payload, path, old_save)

    def evaluate(*args, **kwargs):
        begin = time.monotonic()
        try:
            return old_eval(*args, **kwargs)
        finally:
            guard.validation_seconds += time.monotonic() - begin

    def train(*args, **kwargs):
        return bind(old_train, torch=guard.proxy(torch), _atomic_save=save, evaluate=evaluate)(
            *args, **kwargs
        )

    base.train_one = train
    # Reporting-only NumPy compatibility, not a training-loop change.
    import numpy as np

    if not hasattr(np, "trapezoid"):
        historical.log_token_auc = lambda rows: {
            "auc": float(
                np.trapz(
                    [r["validation_nll"] for r in rows],
                    np.log([r["consumed_tokens"] for r in rows]),
                )
            ),
            "mean_nll": float(
                np.trapz(
                    [r["validation_nll"] for r in rows],
                    np.log([r["consumed_tokens"] for r in rows]),
                )
                / (np.log(rows[-1]["consumed_tokens"]) - np.log(rows[0]["consumed_tokens"]))
            ),
        }
    try:
        historical.train_one_replication(
            name,
            corpus=corpus,
            profile=base.PROFILES["baseline"],
            output_dir=Path(root) / "models",
            seed=91002,
            device=torch.device("cuda:0"),
            resume=True,
        )
    except ResourceStop:
        return 75
    finally:
        guard.record()
    return 0


def run(root, cache):
    c = configuration()
    m = manifest(root, c)
    if m["status"] not in ("R0_PASS", "RESOURCE_STOP", "FAILED_RETAINED", "R2_PENDING"):
        m = preflight(root, cache)
    if m["status"] == "R0_BLOCKED_NO_TRAINING":
        return m
    from .retrain_data import load_verified

    load_verified(m["cache"], c)
    for model_id, name in c["models"].items():
        if m["models"][model_id] == "ASSET_AVAILABLE":
            continue
        path = Path(root) / "models" / name / "resume.pt"
        complete = path.exists() and validate_resume(path, c, m, name)["step"] == 10000
        if not complete:
            remaining = 7200 - m["gpu_seconds_charged"]
            if remaining <= 0:
                m["status"] = "RESOURCE_STOP"
                break
            # Reserve remaining budget durably before worker start. Parent loss cannot reset it.
            old_charge = m["gpu_seconds_charged"]
            m.update(
                status="R1_RUNNING", gpu_seconds_charged=7200.0, worker_budget_seconds=remaining
            )
            m["models"][model_id] = "RUNNING"
            write(Path(root) / "TRAINING_MANIFEST.json", m)
            from .retrain_assets import persist_controller

            try:
                persist_controller(root, c, m)
            except (OSError, RuntimeError, ValueError):
                m.update(gpu_seconds_charged=old_charge, status="PERSISTENCE_BLOCKED_NO_GPU_START")
                write(Path(root) / "TRAINING_MANIFEST.json", m)
                raise
            begin = time.monotonic()
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = "0"
            code = None
            try:
                process = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "epclm001.retrain",
                        "_worker",
                        "--root",
                        str(root),
                        "--model",
                        model_id,
                        "--seconds",
                        str(remaining),
                    ],
                    env=env,
                    check=False,
                    timeout=remaining + 120,
                )
                code = process.returncode
            finally:
                elapsed = time.monotonic() - begin
                m["gpu_worker_observed_seconds"] = (
                    m.get("gpu_worker_observed_seconds", 0.0) + elapsed
                )
                m.setdefault("worker_runs", []).append(
                    {"model": model_id, "exit_code": code, "observed_wall_seconds": elapsed}
                )
                m["gpu_seconds_charged"] = min(7200.0, old_charge + elapsed)
                m["models"][model_id] = (
                    "TRAINED" if code == 0 else "RESOURCE_STOP" if code == 75 else "FAILED_RETAINED"
                )
                m["status"] = (
                    "R2_PENDING"
                    if code == 0
                    else "RESOURCE_STOP"
                    if code == 75
                    else "FAILED_RETAINED"
                )
                saved = Path(root) / "models" / name / "resume.pt"
                if saved.exists():
                    state = validate_resume(saved, c, m, name)
                    m.setdefault("saved_optimizer_updates", {})[model_id] = state[
                        "actual_optimizer_updates"
                    ]
                m["optimizer_updates_known_from_saved_checkpoints"] = sum(
                    m.get("saved_optimizer_updates", {}).values()
                )
                m["optimizer_updates"] = (
                    m["optimizer_updates_known_from_saved_checkpoints"] if code in (0, 75) else None
                )
                write(Path(root) / "TRAINING_MANIFEST.json", m)
            from .retrain_assets import persist_controller

            # Back up completed/partial resume bytes before quality or export can fail.
            persist_controller(root, c, m)
            if code != 0:
                break
        # Persist each model before starting the next. Failed publication never retrains it.
        verify_and_publish(root, model_id)
        m = json.loads((Path(root) / "TRAINING_MANIFEST.json").read_text())
    if all(value == "ASSET_AVAILABLE" for value in m["models"].values()):
        m["status"] = "CONTROLLED_RETRAIN_COMPLETE_E1_BENCHMARK_NOT_AUTHORIZED"
    write(Path(root) / "TRAINING_MANIFEST.json", m)
    return m


def verify_and_publish(root, model_id, publish=True):
    c = configuration()
    m = manifest(root, c)
    from .retrain_assets import publish_model
    from .retrain_data import load_verified
    from .retrain_validation import verify

    corpus = load_verified(m["cache"], c)
    directory = Path(root) / "models" / c["models"][model_id]
    identity_path = directory / "IDENTITY.json"
    if publish and identity_path.exists():
        identity = json.loads(identity_path.read_text())
        if (
            identity["resume_sha256"] != digest(directory / "resume.pt")
            or identity["inference_sha256"] != digest(directory / "inference.pt")
            or identity.get("quality_sha256") != digest(directory / "QUALITY.json")
        ):
            raise RuntimeError("VERIFIED_ASSET_CHANGED_BEFORE_PUBLICATION")
        result = json.loads((directory / "QUALITY.json").read_text())
    else:
        result = verify(root, c, model_id, corpus)
    quality_path = Path(root) / "QUALITY_RESULTS.json"
    all_quality = json.loads(quality_path.read_text()) if quality_path.exists() else {"models": {}}
    all_quality["models"][model_id] = result
    all_quality["status"] = (
        "QUALITY_COMPLETE"
        if all(v.get("status") == "QUALITY_VALID" for v in all_quality["models"].values())
        and len(all_quality["models"]) == 3
        else "PARTIAL_VERIFIED"
    )
    write(quality_path, all_quality)
    if publish:
        available = publish_model(root, c, m["run_id"], model_id)
        index_path = Path(root) / "ASSET_MANIFEST.json"
        index = json.loads(index_path.read_text())
        index["assets"] = [available if x["model"] == model_id else x for x in index["assets"]]
        write(index_path, index)
        m["models"][model_id] = "ASSET_AVAILABLE"
        write(Path(root) / "TRAINING_MANIFEST.json", m)
        from .retrain_assets import persist_controller

        persist_controller(root, c, m)
    return result


def lock(path):
    import fcntl

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError("ANOTHER_CONTROLLED_WORKER_IS_ACTIVE") from None
    return handle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "retrain-preflight",
            "retrain-run",
            "retrain-verify",
            "retrain-publish",
            "_worker",
            "_readback",
        ),
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("/kaggle/working/epclm001-retrain-data"))
    parser.add_argument("--model", choices=("T1", "X3", "X4"))
    parser.add_argument("--publish-github", action="store_true")
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--identity", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--inference", type=Path)
    args = parser.parse_args()
    os.environ["PYTHONPATH"] = (
        str(BASE / "implementation") + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
    if args.command == "_worker":
        if args.seconds is None or not 0 < args.seconds <= 7200:
            parser.error("INVALID_GPU_BUDGET")
        with lock("/tmp/epclm001-controlled-retrain-gpu.lock"):
            raise SystemExit(worker(args.root, args.model, args.seconds))
    if args.command == "_readback":
        from .retrain_assets import readback

        readback(args.identity, args.resume, args.inference)
        return
    controller_lock = lock(args.root / ".controller.lock")
    from .retrain_publication import preflight_reports, publish_reports

    if args.publish_github:
        preflight_reports(args.root, configuration())
    try:
        if args.command == "retrain-preflight":
            result = preflight(args.root, args.cache)
        elif args.command == "retrain-run":
            result = run(args.root, args.cache)
        else:
            if args.model is None:
                parser.error("--model required")
            result = verify_and_publish(
                args.root, args.model, publish=args.command == "retrain-publish"
            )
    finally:
        if args.publish_github:
            publish_reports(args.root, configuration())
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    controller_lock.close()


if __name__ == "__main__":
    main()
