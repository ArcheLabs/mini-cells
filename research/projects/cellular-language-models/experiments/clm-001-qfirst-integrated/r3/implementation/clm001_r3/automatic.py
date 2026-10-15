"""One launch: functional checks, explicit gradient/alignment gates, automatic frozen development."""

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import subprocess
import sys
import os
import time
import numpy as np
import torch
from .common import BASE, source_sha, canonical_sha256, file_sha256, write_json
from .report import seal, verified, aggregate


def select_beta(candidate, pilot):
    rows = pilot["gradient_rows"]
    ratios = [
        row["lcg_grad_norm"] / row["q_grad_norm"]
        for row in rows
        if row.get("lcg_grad_norm") is not None
        and row["lcg_grad_norm"] > 0
        and row.get("q_grad_norm", 0) > 0
    ]
    if not ratios:
        raise RuntimeError("LCG_BETA_UNIDENTIFIABLE: no finite nonzero gradient ratio")
    ratio = float(np.median(ratios))
    beta = candidate["competition"]["beta_target_ratio"] / ratio
    if not np.isfinite(beta) or not 0 < beta <= candidate["competition"]["beta_max"]:
        raise RuntimeError("LCG beta outside predeclared numerical safety range")
    bounds = candidate["competition"]["beta_ratio_bounds"]
    observed = beta * ratio
    if not bounds[0] <= observed <= bounds[1]:
        raise RuntimeError("LCG gradient ratio outside predeclared protection interval")
    return beta, {
        "beta": beta,
        "raw_ratio_median": ratio,
        "scaled_ratio_median": observed,
        "selection_scope": "private_Train_gradient_Pilot_only_no_Development_tuning",
    }


def runtime(candidate, assets, beta, pilot_hash):
    c = deepcopy(candidate)
    c["competition"]["beta"] = beta
    c["authorization"]["development_training"] = True
    c["frozen_assets_sha256"] = assets["asset_sha256"]
    c["frozen_tokenizer_sha256"] = assets["tokenizer_sha256"]
    c["execution"] = {
        "source_sha": source_sha(),
        "cpu_pass": True,
        "gradient_pilot_pass": True,
        "local_alignment_pass": True,
        "pilot_hash": pilot_hash,
        "candidate_configuration_sha256": canonical_sha256(candidate),
        "configuration_sha256": canonical_sha256(c),
    }
    return c


def device_queues(c):
    # Rotate assignments on subsequent registered seeds; seed1 only is presently enabled.
    queues = c["resources"]["device_queues"]
    if c["seed"] in c["seeds"] and c["seeds"].index(c["seed"]) % 2:
        queues = list(reversed(queues))
    flat = [a for queue in queues for a in queue]
    if sorted(flat) != sorted(c["arms"]) or len(queues) != 2:
        raise ValueError("exact three arms, two independent device queues")
    return queues


def workers(c, config, output, assets):
    def queue(index):
        results = []
        for arm in device_queues(c)[index]:
            folder = Path(output) / "arms" / arm
            folder.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                str(BASE / "run.py"),
                "--mode",
                "train-dev",
                "--config",
                str(config),
                "--output",
                str(folder),
                "--assets",
                str(assets),
                "--device",
                "cuda:" + str(index),
                "--arm",
                arm,
            ]
            print("[R3] " + arm + " starts on cuda:" + str(index), flush=True)
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
                code = child.wait()
            results.append({"arm": arm, "exit_code": code, "device": "cuda:" + str(index)})
            # A failed arm does not discard or prevent independent sibling-arm evidence.
        return results

    with ThreadPoolExecutor(max_workers=2) as pool:
        return [r for batch in pool.map(queue, range(2)) for r in batch]


def run(c, output, assets):
    from .cli import cpu_gate
    from .data import prepare, load_assets
    from .publication import publish
    from .pilot import run as pilot_run
    from .model import Model
    from .training import Trainer
    from .checkpoint import restore, identity
    from .audit import capture, measure
    from .specialization import snapshot
    from .profiling import completed

    begin = time.perf_counter()
    output = Path(output)
    assets = Path(assets)
    output.mkdir(parents=True, exist_ok=True)
    source = source_sha()
    namespace = {"candidate": canonical_sha256(c), "seed": c["seed"]}
    checkout = output.parent / "clm001-r3-result-checkout"
    cpu_gate(c, output)
    ready = False
    active = c
    phase = "hardware"
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            raise RuntimeError("select Kaggle T4 x2")
        publish(output, checkout, source, namespace, preflight_only=True)
        ready = True
        phase = "assets"
        print("[R3] preparing/loading role-isolated assets", flush=True)
        if not assets.exists():
            prepare(c, assets)
        data = load_assets(c, assets)
        if data["tokenizer_sha256"] != c["data"]["required_r1_tokenizer_sha256"]:
            raise RuntimeError("Tokenizer changed from R2 identity")
        existing = output / "runtime-config.json"
        if existing.exists():
            from .common import load, authorize

            active = load(existing)
            authorize(active, "train-dev", "cuda:0")
            summary = verified(output / "gpu-pilot-summary.json")
            if active["execution"].get("candidate_configuration_sha256") != canonical_sha256(
                c
            ) or active["execution"]["pilot_hash"] != canonical_sha256(summary):
                raise RuntimeError("cached preflight/config identity mismatch")
            if (
                active.get("frozen_assets_sha256") != data["asset_sha256"]
                or active.get("frozen_tokenizer_sha256") != data["tokenizer_sha256"]
            ):
                raise RuntimeError("cached asset identity mismatch")
            if not all(summary["gates"].values()):
                raise RuntimeError("cached preflight did not pass")
            namespace = {**namespace, "runtime": canonical_sha256(active)}
            publish(output, checkout, source, namespace, preflight_only=True)
            codes = workers(active, existing, output, assets)
            completed(active, output, assets)
            result = aggregate(active, output, source)
            if any(row["exit_code"] for row in codes):
                raise RuntimeError("R3 development partial failure: " + json.dumps(codes))
            return result
        trial = deepcopy(c)
        trial["pilot_rehearsal"] = True
        phase = "gradient-pilot"
        pilots = {}
        pilot_failures = []
        for arm in c["arms"]:
            print("[R3] real GPU gradient Pilot " + arm, flush=True)
            try:
                pilots[arm] = pilot_run(trial, assets, output / "preflight" / arm, arm, "cuda:0")
            except Exception as error:
                pilot_failures.append(
                    {"arm": arm, "type": type(error).__name__, "message": str(error)}
                )
            seal(
                output / "gpu-pilot-summary.json",
                {
                    "status": "GRADIENT_PILOT_FAILED"
                    if pilot_failures
                    else "DIRECTION_AUDIT_PENDING",
                    "arms": pilots,
                    "failures": pilot_failures,
                    "training_status": "NOT_TRAINED",
                    "scientific_decision": False,
                    "source_sha": source,
                    "config_hash": canonical_sha256(c),
                },
            )
        if pilot_failures:
            raise RuntimeError("R3 GPU preflight failed: " + json.dumps(pilot_failures))
        phase = "beta-freeze"
        beta, selection = select_beta(c, pilots["QC-R3"])
        # Reload only the private rehearsal for the real held-out direction audit; never reuse as development initialization.
        trainer = Trainer(Model(trial, "QC-R3").to("cuda:0"))
        rng = np.random.default_rng(1)
        restore(
            output / "preflight/QC-R3/restart", trainer, rng, identity(trial, "QC-R3", source, data)
        )
        phase = "direction-audit"
        print("[R3] held-out fixed-suffix direction audit", flush=True)
        alignment_begin = time.perf_counter()
        bank, missing = capture(trainer.model, data, c)
        alignment = measure(trainer.model, bank, c, resampled=True)
        write_json(output / "preflight/paired-local-vs-suffix.json", alignment)
        seal(
            output / "local-vs-suffix-agreement.json",
            {
                "summary": alignment["summary"],
                "coverage_missing": len(missing),
                "source_sha": source,
                "config_hash": canonical_sha256(c),
                "scientific_decision": False,
            },
        )
        alignment_seconds = time.perf_counter() - alignment_begin
        diagnostic_begin = time.perf_counter()
        if c["evaluation"].get("periodic_diagnostics_enabled", True):
            snapshot(trainer.model, data, c, bank, 0, validation_role="calibration")
        torch.cuda.synchronize()
        diagnostic_seconds = time.perf_counter() - diagnostic_begin
        from .evaluation import evaluate

        quality_begin = time.perf_counter()
        evaluate(trainer.model, data, c, "calibration")
        torch.cuda.synchronize()
        quality_seconds = (
            (time.perf_counter() - quality_begin)
            * c["evaluation"]["development_windows"]
            / c["evaluation"]["calibration_windows"]
        )
        phase = "resource-projection"
        arm_seconds = {
            arm: c["resources"]["projection_safety_factor"]
            * (
                pilot["p90_step_seconds"] * c["training"]["steps"]
                + diagnostic_seconds * len(c["evaluation"]["snapshots"])
                + pilot["shadow_rehearsal_seconds"]
                + 2 * alignment_seconds
                + 17 * quality_seconds
            )
            for arm, pilot in pilots.items()
        }
        projection = max(sum(arm_seconds[arm] for arm in queue) for queue in device_queues(c)) + (
            time.perf_counter() - begin
        )
        gates = {
            "gradient_pilot_pass": all(v["status"] == "GPU_PILOT_PASS" for v in pilots.values()),
            "local_alignment_pass": alignment["summary"]["status"] == "LOCAL_ALIGNMENT_PASS",
            "resource_projection_pass": (
                not c["resources"].get("projection_blocks_development", True)
                or projection <= c["resources"]["target_makespan_seconds"]
            ),
        }
        summary = {
            "status": "GRADIENT_PILOT_PASS" if all(gates.values()) else "PREFLIGHT_STOPPED",
            "arms": pilots,
            "gates": gates,
            "beta_selection": selection,
            "projected_total_seconds": projection,
            "resource_target_met": projection <= c["resources"]["target_makespan_seconds"],
            "resource_projection_advisory_only": not c["resources"].get("projection_blocks_development", True),
            "projection_scope": "two queues including measured snapshot diagnostics; no wall-clock termination",
            "scientific_decision": False,
            "training_status": "NOT_TRAINED",
            "source_sha": source,
            "config_hash": canonical_sha256(c),
        }
        seal(output / "gpu-pilot-summary.json", summary)
        if not all(gates.values()):
            print(
                "[R3] preflight did not support development; returning negative/inconclusive evidence",
                flush=True,
            )
            return aggregate(c, output, source)
        active = runtime(c, data, beta, canonical_sha256(summary))
        namespace = {**namespace, "runtime": canonical_sha256(active)}
        publish(output, checkout, source, namespace, preflight_only=True)
        config = output / "runtime-config.json"
        lock = {
            "source_sha": source,
            "configuration_sha256": canonical_sha256(active),
            "protocol_sha256": file_sha256(BASE / "SPEC.md"),
            "assets_sha256": data["asset_sha256"],
            "tokenizer_sha256": data["tokenizer_sha256"],
            "beta": beta,
        }
        lockfile = output / "r3.lock.json"
        if lockfile.exists() and json.loads(lockfile.read_text()) != lock:
            raise RuntimeError("frozen R3 identity changed")
        write_json(config, active)
        write_json(lockfile, lock)
        phase = "development"
        print(
            "[R3] all gates passed; frozen beta=" + str(beta) + "; starting three-arm development",
            flush=True,
        )
        codes = workers(active, config, output, assets)
        completed(active, output, assets)
        result = aggregate(active, output, source)
        result["workers"] = codes
        if any(row["exit_code"] for row in codes):
            raise RuntimeError("R3 development partial failure: " + json.dumps(codes))
        return result
    except Exception as error:
        seal(
            output / "workflow-failure.json",
            {
                "stage": phase,
                "type": type(error).__name__,
                "message": str(error),
                "source_sha": source,
                "scientific_decision": False,
            },
        )
        raise
    finally:
        aggregate(active, output, source)
        if any((output / "arms" / arm / "metrics.json").exists() for arm in c["arms"]):
            from .plots import render

            try:
                render(active, output)
            except Exception as error:
                seal(
                    output / "figures-summary.json",
                    {"status": "PLOT_FAILURE", "message": str(error), "scientific_decision": False},
                )
        if ready:
            try:
                write_json(
                    output / "publication-state.json", publish(output, checkout, source, namespace)
                )
            except Exception as error:
                write_json(
                    output / "publication-state.json",
                    {
                        "status": "PUBLICATION_FAILURE",
                        "message": str(error),
                        "retry_mode": "publish-only",
                        "retrain": False,
                    },
                )
                raise
