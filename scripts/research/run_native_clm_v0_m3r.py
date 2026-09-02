"""Run frozen Native CLM v0 M3R global-growth vs lineage-growth validation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

from minicells.native_clm_m2 import NativeCLMM2Config
from minicells.native_clm_m3 import NativeCLMM3GrowthConfig
from minicells.native_clm_m3r import (
    aggregate_m3r_formal,
    compare_m3r_arms,
    run_global_growth_control,
    run_lineage_growth_arm,
)

DEFAULT_PROTOCOL = Path(
    "research/validations/native-clm-v0-m3r-read-preserving-growth/protocol.json"
)
DEFAULT_OUTPUT = Path("artifacts/experiments/native-clm-v0-m3r-read-preserving-growth")


def _load_protocol(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    protocol = json.loads(raw)
    if protocol.get("format") != "minicells.native-clm-v0.m3r-protocol.v1":
        raise RuntimeError("unexpected M3R protocol format")
    return protocol, hashlib.sha256(raw).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_data_manifest(data_dir: Path, protocol: dict) -> str:
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("format") != "minicells.native-clm-v0.m3r-data-manifest.v1":
        raise RuntimeError("unexpected M3R data manifest format")
    if manifest.get("stream") != ["B", "C", "D"] or manifest.get("learner_replay_bytes") != 0:
        raise RuntimeError("unexpected M3R stream/replay declaration")

    expected_docs = {
        "A_eval": int(protocol["stream"]["A"]["documents"]),
        "B_train": int(protocol["stream"]["B"]["train_documents"]),
        "B_eval": int(protocol["stream"]["B"]["eval_documents"]),
        "C_train": int(protocol["stream"]["C"]["train_documents"]),
        "C_eval": int(protocol["stream"]["C"]["eval_documents"]),
        "D_train": int(protocol["stream"]["D"]["train_documents"]),
        "D_eval": int(protocol["stream"]["D"]["eval_documents"]),
    }
    records = manifest.get("files", {})
    if set(records) != set(expected_docs):
        raise RuntimeError("M3R data manifest has an unexpected file set")
    for name, expected_count in expected_docs.items():
        record = records[name]
        if int(record["documents"]) != expected_count:
            raise RuntimeError(f"M3R document count mismatch for {name}")
        path = data_dir / record["path"]
        if not path.exists() or path.stat().st_size != int(record["bytes"]):
            raise RuntimeError(f"M3R data file size mismatch for {name}")
        if _sha256(path) != record["sha256"]:
            raise RuntimeError(f"M3R data SHA mismatch for {name}")

    revisions = manifest.get("dataset_revisions", {})
    if set(revisions) != {"A", "B", "C_train", "C_eval", "D"}:
        raise RuntimeError("M3R data manifest lacks exact dataset revisions")
    for name, record in revisions.items():
        if not record.get("repo_id") or not record.get("resolved_revision"):
            raise RuntimeError(f"M3R dataset revision incomplete for {name}")
    return hashlib.sha256(raw).hexdigest()


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
        raise FileNotFoundError("missing M3R data files: " + ", ".join(missing))
    return train, evaluation


def _growth_config(protocol: dict) -> NativeCLMM3GrowthConfig:
    names = {field.name for field in fields(NativeCLMM3GrowthConfig)}
    return NativeCLMM3GrowthConfig(**{key: protocol["growth"][key] for key in names})


def _run_worker(args, protocol: dict) -> int:
    train, evaluation = _paths(args.data_dir)
    common = {
        "checkpoint_path": args.checkpoint,
        "expected_checkpoint_sha256": protocol["parent_checkpoint"]["sha256"],
        "train_paths": train,
        "eval_paths": evaluation,
        "output_dir": args.output_dir / f"seed-{args.seed}" / args.arm,
        "seed": args.seed,
        "train_config": NativeCLMM2Config(**protocol["training"]),
        "growth_config": _growth_config(protocol),
        "device": args.device,
    }
    if args.arm == "global_growth_control":
        summary = run_global_growth_control(**common)
    else:
        summary = run_lineage_growth_arm(**common)
    print(
        json.dumps(
            {
                "arm": summary["arm"],
                "seed": summary["seed"],
                "final_cell_count": summary["final_cell_count"],
                "spawned_cells": summary["spawned_cells"],
                "final_checkpoint_sha256": summary["final_checkpoint_sha256"],
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


def _run_seed_pair(
    args,
    protocol: dict,
    seed: int,
    devices: tuple[str, str],
    *,
    protocol_sha256: str,
    data_manifest_sha256: str,
) -> dict:
    seed_dir = args.output_dir / f"seed-{seed}"
    completed = seed_dir / "seed-summary.json"
    global_summary = seed_dir / "global_growth_control/arm-summary.json"
    lineage_summary = seed_dir / "lineage_growth/arm-summary.json"
    if completed.exists() and global_summary.exists() and lineage_summary.exists():
        cached = json.loads(completed.read_text(encoding="utf-8"))
        if cached.get("protocol_sha256") != protocol_sha256:
            raise RuntimeError(f"seed {seed} cache belongs to another M3R protocol")
        if cached.get("data_manifest_sha256") != data_manifest_sha256:
            raise RuntimeError(f"seed {seed} cache belongs to another M3R data snapshot")
        print(f"[M3R seed={seed}] reusing verified completed seed artifacts", flush=True)
        return cached

    print(
        f"\n[M3R seed={seed}] GPU/worker 0 = global_growth_control ({devices[0]}), "
        f"GPU/worker 1 = lineage_growth ({devices[1]})",
        flush=True,
    )
    global_cmd, global_env = _arm_command(args, seed, "global_growth_control", devices[0])
    lineage_cmd, lineage_env = _arm_command(args, seed, "lineage_growth", devices[1])
    global_proc = subprocess.Popen(global_cmd, env=global_env)
    lineage_proc = subprocess.Popen(lineage_cmd, env=lineage_env)
    global_rc = global_proc.wait()
    lineage_rc = lineage_proc.wait()
    if global_rc != 0 or lineage_rc != 0:
        raise RuntimeError(f"M3R seed {seed} arm failure: global={global_rc}, lineage={lineage_rc}")

    global_result = json.loads(global_summary.read_text(encoding="utf-8"))
    lineage_result = json.loads(lineage_summary.read_text(encoding="utf-8"))
    result = compare_m3r_arms(global_result, lineage_result, thresholds=protocol["thresholds"])
    result["protocol_sha256"] = protocol_sha256
    result["data_manifest_sha256"] = data_manifest_sha256
    seed_dir.mkdir(parents=True, exist_ok=True)
    completed.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "seed": seed,
                "pass": result["pass"],
                "global_A_regression": result["global_A_regression"],
                "lineage_A_regression": result["lineage_A_regression"],
                "A_retention_advantage": result["A_retention_advantage"],
                "birth_max_abs": result["max_birth_logits_max_abs_drift"],
                "A_child_share_reduction": result["A_child_share_reduction_vs_global"],
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
            "| {seed} | {ga:.4f} | {la:.4f} | {adv:.4f} | {gf:.4f} | {lf:.4f} | {drift:.2e} | {leak:.3f} | {ok} |".format(
                seed=result["seed"],
                ga=result["global_A_regression"],
                la=result["lineage_A_regression"],
                adv=result["A_retention_advantage"],
                gf=result["global_mean_forgetting"],
                lf=result["lineage_mean_forgetting"],
                drift=result["max_birth_logits_max_abs_drift"],
                leak=result["A_child_share_reduction_vs_global"],
                ok="PASS" if result["pass"] else "FAIL",
            )
        )
    text = "# Native CLM v0 M3R — Read-Preserving / Lineage-Isolated Growth\n\n"
    text += f"- Status: `{decision['status']}`\n"
    text += f"- Scientific decision: `{decision['scientific_decision']}`\n"
    text += f"- Protocol SHA-256: `{decision['protocol_sha256']}`\n"
    text += f"- Data manifest SHA-256: `{decision['data_manifest_sha256']}`\n"
    text += f"- Formal seeds: `{decision['formal_seeds']}`\n"
    text += "- Learner replay: `0 bytes`\n"
    text += "- Causal arms: frozen M3 global-pool growth vs lineage-isolated growth\n\n"
    text += "| seed | global A reg | lineage A reg | A advantage | global forgetting | lineage forgetting | max birth drift | A child-share reduction | result |\n"
    text += "|---:|---:|---:|---:|---:|---:|---:|---:|---|\n"
    text += "\n".join(rows) + "\n\n"
    text += "Boundary: all M3 pressure-controller numerical thresholds are unchanged; M3R changes only read topology and the leaf-only lineage allocation constraint required by that topology.\n"
    (output / "RESULTS.md").write_text(text, encoding="utf-8")


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
    parser.add_argument("--arm", choices=["global_growth_control", "lineage_growth"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    protocol, protocol_sha = _load_protocol(args.protocol)
    if args.worker:
        if args.arm is None or args.seed is None:
            parser.error("--worker requires --arm and --seed")
        return _run_worker(args, protocol)

    data_manifest_sha = _validate_data_manifest(args.data_dir, protocol)
    print(f"Verified M3R data manifest SHA-256: {data_manifest_sha}", flush=True)
    devices = tuple(part.strip() for part in args.devices.split(","))
    if len(devices) != 2:
        raise ValueError("--devices must name exactly two workers, e.g. 0,1")

    consumed = {int(seed) for seed in protocol["consumed_seeds_forbidden"]}
    if args.formal:
        seeds = [int(seed) for seed in protocol["formal_seeds"]]
    elif args.seed is not None:
        if args.seed in protocol["formal_seeds"]:
            raise RuntimeError("refusing to touch a formal M3R seed outside --formal")
        if args.seed in consumed:
            raise RuntimeError("refusing to reuse a consumed M2/M3 formal seed")
        if args.seed not in protocol["development_seeds"]:
            raise RuntimeError("M3R development runs must use a registered development seed")
        seeds = [args.seed]
    else:
        parser.error("choose --formal or --seed <registered-development-seed>")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_results = [
        _run_seed_pair(
            args,
            protocol,
            seed,
            devices,
            protocol_sha256=protocol_sha,
            data_manifest_sha256=data_manifest_sha,
        )
        for seed in seeds
    ]

    if args.formal:
        decision = aggregate_m3r_formal(
            seed_results,
            protocol_sha256=protocol_sha,
            formal_seeds=seeds,
            data_manifest_sha256=data_manifest_sha,
        )
        _write_formal_artifacts(args.output_dir, decision)
        print(json.dumps(decision, indent=2), flush=True)
        return 0 if decision["scientific_decision"] else 2

    print(json.dumps(seed_results[0], indent=2), flush=True)
    return 0 if seed_results[0]["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
