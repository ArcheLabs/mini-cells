#!/usr/bin/env python3
"""Run Native CLM v0 M2 protected-vs-unsafe continual-language validation.

On a two-GPU Kaggle host the parent process assigns the protected arm to GPU0 and the
unsafe arm to GPU1. The two independent 12.15M models run concurrently; no DDP
synchronization is needed because the causal control is the second workload.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from minicells.native_clm_m2 import (
    NativeCLMM2Config,
    aggregate_formal,
    compare_arms,
    run_arm,
)


DEFAULT_PROTOCOL = Path(
    "research/validations/native-clm-v0-m2-continual-language/protocol.json"
)
DEFAULT_OUTPUT = Path(
    "artifacts/experiments/native-clm-v0-m2-continual-language"
)


def _load_protocol(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    protocol = json.loads(raw)
    if protocol.get("format") != "minicells.native-clm-v0.m2-protocol.v1":
        raise RuntimeError("unexpected M2 protocol format")
    return protocol, hashlib.sha256(raw).hexdigest()


def _paths(data_dir: Path):
    train = {
        "B": data_dir / "B-wikitext-train.txt",
        "C": data_dir / "C-code-train.txt",
        "D": data_dir / "D-dolly-train.txt",
    }
    evaluation = {
        "A": data_dir / "A-tinystories-eval.txt",
        "B": data_dir / "B-wikitext-eval.txt",
        "C": data_dir / "C-code-eval.txt",
        "D": data_dir / "D-dolly-eval.txt",
    }
    missing = [str(path) for path in [*train.values(), *evaluation.values()] if not path.exists()]
    if missing:
        raise FileNotFoundError("missing M2 data files: " + ", ".join(missing))
    return train, evaluation


def _run_worker(args, protocol: dict) -> int:
    train, evaluation = _paths(args.data_dir)
    config = NativeCLMM2Config(**protocol["training"])
    summary = run_arm(
        checkpoint_path=args.checkpoint,
        expected_checkpoint_sha256=protocol["parent_checkpoint"]["sha256"],
        train_paths=train,
        eval_paths=evaluation,
        output_dir=args.output_dir / f"seed-{args.seed}" / args.arm,
        arm=args.arm,
        seed=args.seed,
        config=config,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "arm": summary["arm"],
                "seed": summary["seed"],
                "final_checkpoint_sha256": summary["final_checkpoint_sha256"],
                "shared_and_router_frozen": summary["shared_and_router_frozen"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


def _arm_command(args, seed: int, arm: str, device: str) -> tuple[list[str], dict[str, str]]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--arm",
        arm,
        "--seed",
        str(seed),
        "--checkpoint",
        str(args.checkpoint),
        "--data-dir",
        str(args.data_dir),
        "--output-dir",
        str(args.output_dir),
        "--protocol",
        str(args.protocol),
        "--device",
        "cuda" if device != "cpu" else "cpu",
    ]
    env = os.environ.copy()
    if device != "cpu":
        env["CUDA_VISIBLE_DEVICES"] = device
    return command, env


def _run_seed_pair(args, protocol: dict, seed: int, devices: tuple[str, str]) -> dict:
    print(
        f"\n[M2 seed={seed}] GPU/worker 0 = protected ({devices[0]}), "
        f"GPU/worker 1 = unsafe ({devices[1]})",
        flush=True,
    )
    protected_cmd, protected_env = _arm_command(args, seed, "protected", devices[0])
    unsafe_cmd, unsafe_env = _arm_command(args, seed, "unsafe", devices[1])
    protected_proc = subprocess.Popen(protected_cmd, env=protected_env)
    unsafe_proc = subprocess.Popen(unsafe_cmd, env=unsafe_env)
    p_rc = protected_proc.wait()
    u_rc = unsafe_proc.wait()
    if p_rc != 0 or u_rc != 0:
        raise RuntimeError(
            f"M2 seed {seed} arm failure: protected={p_rc}, unsafe={u_rc}"
        )

    seed_dir = args.output_dir / f"seed-{seed}"
    protected = json.loads((seed_dir / "protected/arm-summary.json").read_text())
    unsafe = json.loads((seed_dir / "unsafe/arm-summary.json").read_text())
    result = compare_arms(
        protected,
        unsafe,
        thresholds=protocol["thresholds"],
    )
    (seed_dir / "seed-summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "seed": seed,
                "pass": result["pass"],
                "protected_mean_forgetting": result["protected_mean_forgetting"],
                "unsafe_mean_forgetting": result["unsafe_mean_forgetting"],
                "retention_advantage": result["retention_advantage"],
                "protected_mean_plasticity": result["protected_mean_plasticity"],
            },
            indent=2,
        ),
        flush=True,
    )
    return result


def _write_formal_artifacts(output: Path, decision: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output / "gate-summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed", "gate", "pass"])
        for seed_result in decision["seed_results"]:
            for gate, passed in seed_result["gates"].items():
                writer.writerow([seed_result["seed"], gate, int(bool(passed))])

    rows = []
    for result in decision["seed_results"]:
        rows.append(
            "| {seed} | {pp:.4f} | {up:.4f} | {pf:.4f} | {uf:.4f} | {adv:.4f} | {ok} |".format(
                seed=result["seed"],
                pp=result["protected_mean_plasticity"],
                up=result["unsafe_mean_plasticity"],
                pf=result["protected_mean_forgetting"],
                uf=result["unsafe_mean_forgetting"],
                adv=result["retention_advantage"],
                ok="PASS" if result["pass"] else "FAIL",
            )
        )
    results_md = """# Native CLM v0 M2 — Continual Language Stream\n\n"""
    results_md += f"- Status: `{decision['status']}`\n"
    results_md += f"- Scientific decision: `{decision['scientific_decision']}`\n"
    results_md += f"- Protocol SHA-256: `{decision['protocol_sha256']}`\n"
    results_md += f"- Formal seeds: `{decision['formal_seeds']}`\n"
    results_md += "- Learner replay: `0 bytes`\n"
    results_md += "- Topology: `8 Cells / 2 active / growth disabled`\n"
    results_md += "- Control: `protected certificate projection` vs `unsafe identical Cell-local writes`\n\n"
    results_md += "| seed | protected plasticity | unsafe plasticity | protected forgetting | unsafe forgetting | retention advantage | result |\n"
    results_md += "|---:|---:|---:|---:|---:|---:|---|\n"
    results_md += "\n".join(rows) + "\n\n"
    results_md += "Boundary: shared substrate and router are frozen from M1; M2 does not test autonomous growth or router continual adaptation.\n"
    (output / "RESULTS.md").write_text(results_md, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--arm", choices=["protected", "unsafe"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    protocol, protocol_sha = _load_protocol(args.protocol)
    if args.worker:
        if args.arm is None or args.seed is None:
            parser.error("--worker requires --arm and --seed")
        return _run_worker(args, protocol)

    devices = tuple(part.strip() for part in args.devices.split(","))
    if len(devices) != 2:
        raise ValueError("--devices must name exactly two workers, e.g. 0,1")

    if args.formal:
        seeds = [int(seed) for seed in protocol["formal_seeds"]]
    elif args.seed is not None:
        if args.seed in protocol["formal_seeds"]:
            raise RuntimeError("refusing to touch a formal seed outside --formal")
        seeds = [args.seed]
    else:
        parser.error("choose --formal or --seed <development-seed>")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_results = [_run_seed_pair(args, protocol, seed, devices) for seed in seeds]

    if args.formal:
        decision = aggregate_formal(
            seed_results,
            protocol_sha256=protocol_sha,
            formal_seeds=seeds,
        )
        _write_formal_artifacts(args.output_dir, decision)
        print(json.dumps(decision, indent=2), flush=True)
        return 0 if decision["scientific_decision"] else 2

    print(json.dumps(seed_results[0], indent=2), flush=True)
    return 0 if seed_results[0]["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
