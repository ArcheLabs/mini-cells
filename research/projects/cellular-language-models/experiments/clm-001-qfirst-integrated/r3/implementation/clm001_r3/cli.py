"""R3 one-launch entry. GPU work is performed only by the user's Kaggle Run All."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import json
from .common import BASE, ROOT, load, source_sha, canonical_sha256
from .model import Model
from .accounting import parameters
from .report import seal, aggregate


def cpu_gate(c, output):
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "GIT_ALLOW_PROTOCOL": "file",
    }
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(ROOT / "tests/research/clm001_r3")],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(result.stdout, flush=True)
    report = {
        "status": "CPU_PASS" if result.returncode == 0 else "CPU_FAIL",
        "source_sha": source_sha(),
        "configuration_sha256": canonical_sha256(c),
        "scope": "SYNTHETIC_CPU_ONLY",
        "test_output": result.stdout,
        "scientific_decision": False,
    }
    seal(Path(output) / "cpu-gate.json", report)
    inventory = {}
    models = {arm: Model(c, arm) for arm in c["arms"]}
    initial = models[c["arms"][0]].state_dict()
    import torch

    if not all(
        torch.equal(value, model.state_dict()[name])
        for model in models.values()
        for name, value in initial.items()
    ):
        raise RuntimeError("three-arm initialization mismatch")
    for arm, model in models.items():
        inventory[arm] = {
            **parameters(model),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        }
    if len({r["total_parameters"] for r in inventory.values()}) != 1:
        raise RuntimeError("three-arm parameter mismatch")
    seal(
        Path(output) / "parameter-audit.json",
        {
            "arms": inventory,
            "source_sha": source_sha(),
            "configuration_sha256": canonical_sha256(c),
        },
    )
    if result.returncode:
        raise RuntimeError("CPU Gate failed; no GPU next phase")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "cpu-preflight", "train-dev", "analyze", "publish-only"],
    )
    parser.add_argument("--config", default=str(BASE / "config/proposed.json"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--arm", choices=["Q-R3", "QC-R3", "QS-R3"])
    a = parser.parse_args()
    c = load(a.config)
    a.output.mkdir(parents=True, exist_ok=True)
    if a.mode == "cpu-preflight":
        result = cpu_gate(c, a.output)
    elif a.mode == "auto":
        from .automatic import run

        result = run(c, a.output, a.assets or a.output.parent / "clm001-r3-assets")
    elif a.mode == "train-dev":
        from .runner import development

        if a.arm is None:
            raise ValueError("arm required")
        result = development(c, a.arm, a.output, a.assets, a.device)
    elif a.mode == "analyze":
        result = aggregate(c, a.output, source_sha())
    else:
        from .publication import publish

        # Use the frozen runtime identity when retrying publication; this path never trains.
        runtime = a.output / "runtime-config.json"
        if runtime.exists():
            c = load(runtime)
        lock = a.output / "r3.lock.json"
        source = json.loads(lock.read_text())["source_sha"] if lock.exists() else source_sha()
        aggregate(c, a.output, source)
        candidate = load(a.config)
        namespace = {
            "candidate": c.get("execution", {}).get(
                "candidate_configuration_sha256", canonical_sha256(candidate)
            ),
            "seed": c["seed"],
        }
        if runtime.exists():
            namespace["runtime"] = canonical_sha256(c)
        result = publish(
            a.output,
            a.output.parent / "clm001-r3-result-checkout",
            source,
            namespace,
        )
    if a.mode == "train-dev":
        print(
            json.dumps(
                {
                    "arm": a.arm,
                    "status": result["training_status"],
                    "output": str(a.output),
                    "scientific_decision": False,
                }
            ),
            flush=True,
        )
    else:
        print(json.dumps(result, indent=2), flush=True)
