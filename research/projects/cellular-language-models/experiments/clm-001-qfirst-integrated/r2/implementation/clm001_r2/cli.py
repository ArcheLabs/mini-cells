"""One launch: synthetic Gate -> return check -> assets -> dual GPU Pilot -> authorized development."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import torch
from .common import (
    BASE,
    ROOT,
    load,
    validate,
    source_sha,
    canonical_sha256,
    write_json,
    authorize,
    file_sha256,
)
from .model import Model
from .accounting import parameters
from .report import seal, aggregate, verified


def cpu_gate(c, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "GIT_ALLOW_PROTOCOL": "file",
    }
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(ROOT / "tests/research/clm001_r2")],
        env=env,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(proc.stdout, flush=True)
    report = {
        "experiment_id": c["experiment_id"],
        "status": "CPU_GATE_PASS" if proc.returncode == 0 else "FAILED",
        "scope": "SYNTHETIC_CPU_ONLY",
        "source_sha": source_sha(),
        "config_hash": canonical_sha256(c),
        "tests": proc.stdout,
        "gpu_pilot_completed": False,
        "scientific_decision": False,
    }
    seal(output / "cpu-gate.json", report)
    inventory = {}
    for arm in c["arms"]:
        model = Model(c, arm)
        inventory[arm] = {
            **parameters(model),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        }
    if len({v["total_parameters"] for v in inventory.values()}) != 1:
        raise RuntimeError("unmatched resident inventory")
    seal(output / "parameter-audit.json", {"arms": inventory, "config_hash": canonical_sha256(c)})
    if proc.returncode:
        raise RuntimeError("CPU Gate failed; next phase blocked")
    return report


def workers(c, config, output, assets, mode):
    # Same log-tee pattern as pd001.orchestration; distinct command/partial-failure semantics.
    from concurrent.futures import ThreadPoolExecutor

    def worker(index):
        arm = c["arms"][index]
        folder = output / ("preflight" if mode == "gpu-pilot" else "arms") / arm
        folder.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(BASE / "run.py"),
            "--mode",
            mode,
            "--config",
            str(config),
            "--output",
            str(folder),
            "--assets",
            str(assets),
            "--arm",
            arm,
            "--device",
            "cuda:" + str(index),
        ]
        print("[R2] " + mode + " " + arm, flush=True)
        with (folder / "worker.log").open("a") as log:
            child = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            for line in child.stdout:
                log.write(line)
                log.flush()
                print("[" + arm + "] " + line.rstrip(), flush=True)
            return child.wait()

    with ThreadPoolExecutor(max_workers=2) as pool:
        return list(pool.map(worker, range(len(c["arms"]))))


def collect_pilot(c, output, codes):
    """Always retain successful siblings; a failed worker need not have a success file."""
    arms = {}
    failures = []
    for arm, code in zip(c["arms"], codes):
        folder = Path(output) / "preflight" / arm
        path = folder / "gpu-pilot.json"
        failure = folder / "gpu-pilot-failure.json"
        record = verified(path) if path.exists() else None
        if record is not None:
            arms[arm] = record
        if code or record is None or record.get("status") != "GPU_PILOT_PASS":
            detail = (
                verified(failure)
                if failure.exists()
                else {
                    "message": "worker report unavailable; inspect preflight/" + arm + "/worker.log"
                }
            )
            failures.append({"arm": arm, "exit_code": code, **detail})
    result = {
        "passed": len(codes) == len(c["arms"]) and not failures,
        "arms": arms,
        "failures": failures,
        "scientific_decision": False,
        "training_status": "NOT_TRAINED",
    }
    seal(Path(output) / "preflight-summary.json", result)
    return result


def automatic(c, output, assets):
    from .data import prepare, load_assets
    from .publication import publish

    source = source_sha()
    namespace = {"config": canonical_sha256(c), "seed": c["seed"]}
    checkout = output.parent / "clm001-r2-result-checkout"
    return_ready = False
    cpu_gate(c, output)
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            raise RuntimeError("Select Kaggle T4 x2; actual GPU Pilot is required")
        publish(output, checkout, source, namespace, preflight_only=True)
        return_ready = True
        if not assets.exists():
            prepare(c, assets)
        data = load_assets(c, assets)
        if data["tokenizer_sha256"] != c["data"]["required_r1_tokenizer_sha256"]:
            raise RuntimeError(
                "Tokenizer differs from R1; retain original tokenizer, do not silently change it"
            )
        lock = {
            "source_sha": source,
            "protocol_sha256": file_sha256(BASE / "SPEC.md"),
            "seed": c["seed"],
            "run_id": canonical_sha256(
                {"source": source, "config": canonical_sha256(c), "assets": data["asset_sha256"]}
            )[:24],
            "config_hash": canonical_sha256(c),
            "asset_sha256": data["asset_sha256"],
            "tokenizer_sha256": data["tokenizer_sha256"],
            "scientific_decision": False,
        }
        lock_path = output / "r2.lock.json"
        if lock_path.exists() and json.loads(lock_path.read_text()) != lock:
            raise RuntimeError("immutable run lock changed")
        write_json(lock_path, lock)
        config = output / "runtime-config.json"
        write_json(config, c)
        codes = workers(c, config, output, assets, "gpu-pilot")
        pilot = collect_pilot(c, output, codes)
        if not pilot["passed"]:
            raise RuntimeError(
                "GPU Pilot failed; training not started: " + json.dumps(pilot["failures"])
            )
        if not c["authorization"]["development_training"]:
            print(
                "[R2] Pilot complete; full development is not authorized by this implementation request.",
                flush=True,
            )
            return aggregate(c, output, source)
        authorize(c, "train-dev", "cuda:0")
        codes = workers(c, config, output, assets, "train-dev")
        from .profiling import completed

        completed(c, output, assets)
        report = aggregate(c, output, source)
        if any(codes):
            report["worker_exit_codes"] = codes
        return report
    finally:
        aggregate(c, output, source)
        if return_ready:
            try:
                state = publish(output, checkout, source, namespace)
                write_json(output / "publication-state.json", state)
            except Exception as error:
                write_json(
                    output / "publication-state.json",
                    {
                        "publication_status": "FAILED",
                        "error": str(error),
                        "retry_mode": "publish-only",
                        "retrain": False,
                    },
                )
                raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["auto", "cpu-preflight", "gpu-pilot", "train-dev", "analyze", "publish-only"],
        default="auto",
    )
    parser.add_argument("--config", default=str(BASE / "config/proposed.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--arm", choices=["QF-R2", "J0-R2"])
    parser.add_argument("--device", default="cpu")
    a = parser.parse_args()
    c = load(a.config)
    validate(c)
    a.output.mkdir(parents=True, exist_ok=True)
    if a.mode == "cpu-preflight":
        result = cpu_gate(c, a.output)
    elif a.mode == "auto":
        result = automatic(c, a.output, a.assets or a.output.parent / "clm001-r2-assets")
    elif a.mode == "gpu-pilot":
        from .pilot import run

        result = run(c, a.assets, a.output, a.arm, a.device)
    elif a.mode == "train-dev":
        from .runner import development

        if not a.arm:
            raise ValueError("arm required")
        result = development(c, a.arm, a.output, a.assets, a.device)
    elif a.mode == "analyze":
        result = aggregate(c, a.output, source_sha())
    else:
        from .publication import publish

        aggregate(c, a.output, source_sha())
        result = publish(
            a.output,
            a.output.parent / "clm001-r2-result-checkout",
            source_sha(),
            {"config": canonical_sha256(c), "seed": c["seed"]},
        )
    print(json.dumps(result, indent=2), flush=True)
