#!/usr/bin/env python3
"""Analyze Core Validation 001 checkpoints for residual memorization."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import torch

from minicells.knowledge_subsumption import KnowledgeSubsumptionConfig
from minicells.residual_memorization import (
    ResidualMemorizationConfig,
    analyze_checkpoint_pair,
    analyze_oracle,
    summarize_experiment,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "research" / "core-validation-001b-protocol.json"
DEFAULT_SOURCE = (
    ROOT / "results" / "core-validation-001b-residual-memorization" / "source-001"
)
DEFAULT_OUT = ROOT / "results" / "core-validation-001b-residual-memorization"


def _git(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *command], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _tracked_tree_dirty() -> bool | None:
    try:
        unstaged = subprocess.run(
            ["git", "diff", "--quiet"], cwd=ROOT, check=False
        ).returncode
        staged = subprocess.run(
            ["git", "diff", "--cached", "--quiet"], cwd=ROOT, check=False
        ).returncode
        return bool(unstaged or staged)
    except OSError:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    source_raw_path = args.source / "raw.json"
    if not source_raw_path.exists():
        raise FileNotFoundError(source_raw_path)
    source = json.loads(source_raw_path.read_text(encoding="utf-8"))
    if source.get("format") != "minicells.core-validation.knowledge-subsumption.v1":
        raise RuntimeError("001b requires a Core Validation 001 source run")
    if source.get("mode") != "formal":
        raise RuntimeError("001b formal analysis requires a formal Core Validation 001 source")
    if source.get("provenance", {}).get("tracked_tree_dirty") is not False:
        raise RuntimeError("refusing analysis from dirty parent training provenance")

    training_config = KnowledgeSubsumptionConfig(**source["config"])
    residual_config = ResidualMemorizationConfig.from_protocol(args.protocol)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if bool(protocol["hardware"]["gpu_required_for_formal_run"]):
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("formal Core Validation 001b requires CUDA")

    runs = []
    for source_run in source["runs"]:
        print(
            f"[core-001b] analyze task={source_run['task']} seed={source_run['seed']}",
            flush=True,
        )
        run = analyze_checkpoint_pair(
            training_config,
            residual_config,
            source_run,
            device=device,
        )
        runs.append(run)
        coupling = run["late_coupling"]
        print(
            "[core-001b] "
            f"pass={run['gates']['pass']} "
            f"corr={coupling['exclusion_accuracy_correlation']:.4f} "
            f"mean_gap={coupling['mean_absolute_gap']:.4f} "
            f"max_positive_gap={coupling['maximum_positive_gap']:.4f}",
            flush=True,
        )

    oracle_seed = int(
        source.get("oracle_reference", {}).get("seed")
        if source.get("oracle_reference")
        else 73101
    )
    print(f"[core-001b] oracle seed={oracle_seed}", flush=True)
    oracle = analyze_oracle(
        training_config,
        residual_config,
        seed=oracle_seed,
        device=device,
    )
    gates = protocol["gates"]
    decision = summarize_experiment(
        runs,
        oracle,
        positive_status=str(gates["positive_status"]),
        negative_status=str(gates["negative_status"]),
    )
    payload = {
        "format": protocol["format"],
        "experiment_id": protocol["experiment_id"],
        "parent_experiment": protocol["parent_experiment"],
        "mode": "formal",
        "provenance": {
            "code_commit": _git(["rev-parse", "HEAD"]),
            "code_tree": _git(["rev-parse", "HEAD^{tree}"]),
            "tracked_tree_dirty": _tracked_tree_dirty(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0)
            if device.type == "cuda"
            else None,
        },
        "parent_training_provenance": source.get("provenance"),
        "parent_decision": source.get("decision"),
        "training_config": asdict(training_config),
        "residual_config": asdict(residual_config),
        "runs": runs,
        "oracle": oracle,
        "decision": decision,
        "elapsed_seconds": time.time() - started,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "raw.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
