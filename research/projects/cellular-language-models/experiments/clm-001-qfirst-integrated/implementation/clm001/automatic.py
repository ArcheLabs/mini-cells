"""One invocation: checks -> assets -> real functional probe -> paired development -> return."""

from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import sys
import torch
from .common import BASE, canonical_sha256, source_sha, write_json, validate
from .data import prepare, load_assets
from .publication import publish
from .report import summarize


def runtime_config(candidate, assets, source):
    c = deepcopy(candidate)
    c["status"] = "AUTOMATIC_EXPLORATORY_DEVELOPMENT"
    if c["policy"]["lambda_cost"] is None:
        c["policy"]["lambda_cost"] = c["policy"]["lambda_candidate"]
    c["gates"]["oracle_cpu_pass"] = True
    c["frozen_assets_sha256"] = assets["asset_sha256"]
    c["frozen_tokenizer_sha256"] = assets["tokenizer_sha256"]
    c["execution"] = {
        "mode": "automatic-development",
        "source_commit": source,
        "configuration_sha256": canonical_sha256(c),
        "checks": {
            "cpu": True,
            "return_path": True,
            "assets": True,
            "gpu_functional": False,
            "local_restore": False,
        },
        "real_kaggle_persistence_verified": False,
    }
    validate(c)
    return c


def validate_pilot(result, c):
    if result.get("source_commit") != source_sha() or result.get(
        "config_sha256"
    ) != canonical_sha256(c):
        raise RuntimeError("GPU preflight source/configuration identity differs")
    if (
        result.get("asset_sha256") != c["frozen_assets_sha256"]
        or result.get("tokenizer_sha256") != c["frozen_tokenizer_sha256"]
    ):
        raise RuntimeError("GPU preflight asset identity differs")
    if set(result.get("arms", {})) != set(c["arms"]):
        raise RuntimeError("both preflight arms are required")
    for arm in c["arms"]:
        item = result["arms"][arm]
        if not item.get("local_restore_pass"):
            raise RuntimeError("checkpoint restart preflight failed")
        if item.get("oracle", {}).get("states", 0) < 1:
            raise RuntimeError("ORACLE_UNTRUSTED: no real-model Oracle states")
        if not item.get("update_rows"):
            raise RuntimeError("no functional update measurements")
        for row in item["update_rows"]:
            if not all(math.isfinite(row[k]) for k in ("live_ce", "synchronized_total_seconds")):
                raise RuntimeError("nonfinite GPU functional preflight")
        seconds = item["projected_500_update_seconds"]
        if not math.isfinite(seconds) or seconds <= 0:
            raise RuntimeError("invalid workload projection")
        if seconds > 7200:
            raise RuntimeError(
                "Projected training workload exceeds two hours; full development was not started. No process was killed or budget reduced."
            )


def workers(c, config, output, assets):
    children = []
    for index, arm in enumerate(c["arms"]):
        command = [
            sys.executable,
            str(BASE / "run.py"),
            "--mode",
            "train-dev",
            "--config",
            str(config),
            "--output",
            str(output / arm),
            "--assets",
            str(assets),
            "--device",
            "cuda:" + str(index),
            "--arm",
            arm,
        ]
        print("[Automatic] starting " + arm + " on cuda:" + str(index), flush=True)
        children.append(subprocess.Popen(command))
    codes = [child.wait() for child in children]
    if any(codes):
        raise RuntimeError(
            "Development worker failed; retained checkpoints can resume: " + str(codes)
        )


def run(candidate, output, assets):
    from .cli import cpu_gate
    from .pilot import run as gpu_preflight

    output = Path(output)
    assets = Path(assets)
    output.mkdir(parents=True, exist_ok=True)
    source = source_sha()
    namespace = {
        "candidate_config": canonical_sha256(candidate),
        "seed": candidate["seed"],
        "flow": "automatic",
    }
    checkout = output.parent / "clm001-automatic-result-checkout"
    phase = "cpu-preflight"
    return_ready = False
    state = {
        "experiment_id": candidate["experiment_id"],
        "source_commit": source,
        "scientific_decision": False,
        "formal_confirmation": False,
        "real_kaggle_persistence_verified": False,
    }

    def record(status):
        state.update(phase=phase, status=status)
        write_json(output / "automatic-state.json", state)
        print("[Automatic] " + phase + ": " + status, flush=True)

    try:
        record("RUNNING")
        cpu_gate(candidate, output)
        phase = "hardware"
        record("RUNNING")
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            raise RuntimeError(
                "This two-arm development run needs two GPUs; select Kaggle T4 x2 and Run All."
            )
        phase = "return-preflight"
        record("RUNNING")
        publish(output, checkout, source, namespace, preflight_only=True)
        return_ready = True
        phase = "assets"
        record("RUNNING")
        if not assets.exists():
            prepare(candidate, assets)
        data = load_assets(candidate, assets)
        c = runtime_config(candidate, data, source)
        config = output / "runtime-config.json"
        write_json(config, c)
        phase = "gpu-preflight"
        record("RUNNING")
        pilot_folder = output / "preflight"
        pilot_file = pilot_folder / "pilot.json"
        result = (
            json.loads(pilot_file.read_text())
            if pilot_file.exists()
            else gpu_preflight(c, assets, pilot_folder, "cuda:0", source)
        )
        validate_pilot(result, c)
        c["gates"]["gpu_pilot_pass"] = True
        c["execution"]["configuration_sha256"] = canonical_sha256(
            {k: v for k, v in c.items() if k != "execution"}
        )
        c["execution"]["checks"].update(gpu_functional=True, local_restore=True)
        write_json(config, c)
        phase = "development"
        record("RUNNING")
        workers(c, config, output, assets)
        phase = "analysis"
        record("RUNNING")
        results = {
            arm: json.loads((output / arm / "metrics.json").read_text()) for arm in c["arms"]
        }
        journals = [
            json.loads((output / arm / "shadow-journal.json").read_text()) for arm in c["arms"]
        ]
        report = summarize(c, results, oracle_pass=True, shadow_journal=journals)
        report["execution_mode"] = "automatic-development"
        report["real_kaggle_persistence_verified"] = False
        write_json(output / "report.json", report)
        phase = "return"
        record("RUNNING")
        publication = publish(output, checkout, source, namespace)
        state["publication"] = publication
        record("COMPLETED")
        return state
    except Exception as error:
        state["error"] = str(error)
        record("FAILED")
        # Retain a useful failure report; never retry training just because publication failed.
        if not (output / "report.json").exists():
            write_json(
                output / "report.json",
                {**state, "status": "INCOMPLETE", "claim_scope": "EXPLORATORY_ONLY"},
            )
        if return_ready and phase != "return":
            try:
                publish(output, checkout, source, namespace)
            except Exception as publication_error:
                state["publication_error"] = str(publication_error)
                record("FAILED")
        raise
