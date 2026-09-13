from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import native_clm_runtime as base
from native_clm_candidates import VERIFIED_DATASET_REVISION
from native_clm_replication import MODEL_IDS, PARAMETER_TOLERANCE, REPLICATION_MODELS, REPLICATION_SEEDS, parameter_summary, validate_replication_request
from native_clm_replication_runlib import aggregate_results, seed_leaderboard, train_one_replication, write_csv
from native_clm_replication_visualize import write_cross_seed_ppl, write_paired_delta, write_seed_visualizations

base.DATASET_REVISION = VERIFIED_DATASET_REVISION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen Native CLM cross-seed replication")
    sub = parser.add_subparsers(dest="command", required=True)
    sweep = sub.add_parser("sweep")
    sweep.add_argument("--profile", choices=tuple(base.PROFILES), default="baseline")
    sweep.add_argument("--seeds", nargs="+", type=int, default=list(REPLICATION_SEEDS))
    sweep.add_argument("--cache-root", type=Path, required=True)
    sweep.add_argument("--output-root", type=Path, required=True)
    sweep.add_argument("--allow-cpu", action="store_true")
    sweep.add_argument("--no-resume", action="store_true")
    worker = sub.add_parser("worker")
    worker.add_argument("--model", choices=REPLICATION_MODELS, required=True)
    worker.add_argument("--profile", choices=tuple(base.PROFILES), required=True)
    worker.add_argument("--seed", type=int, required=True)
    worker.add_argument("--cache-root", type=Path, required=True)
    worker.add_argument("--output-dir", type=Path, required=True)
    worker.add_argument("--allow-cpu", action="store_true")
    worker.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def _device(allow_cpu: bool) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if allow_cpu:
        return torch.device("cpu")
    raise RuntimeError("replication training requires CUDA; --allow-cpu is smoke/debug only")


def run_worker(args: argparse.Namespace) -> int:
    if args.seed not in REPLICATION_SEEDS:
        raise RuntimeError("worker seed is not part of frozen replication")
    profile = base.PROFILES[args.profile]
    corpus = base.prepare_corpus(args.cache_root, profile, hf_token=os.environ.get("HF_TOKEN") or None)
    summary = train_one_replication(
        args.model, corpus=corpus, profile=profile, output_dir=args.output_dir,
        seed=args.seed, device=_device(args.allow_cpu), resume=not args.no_resume,
    )
    final = summary["final"] or {}
    print(json.dumps({
        "seed": args.seed, "id": MODEL_IDS[args.model], "model": args.model,
        "validation_ppl": final.get("validation_ppl"),
        "auc": summary["validation_log_token_nll_auc"],
        "tokens_per_second": final.get("tokens_per_second"),
    }, indent=2))
    return 0


def _worker_command(args: argparse.Namespace, model: str, seed: int, output_dir: Path) -> list[str]:
    command = [
        sys.executable, str(Path(__file__).resolve()), "worker",
        "--model", model, "--profile", args.profile, "--seed", str(seed),
        "--cache-root", str(args.cache_root), "--output-dir", str(output_dir),
    ]
    if args.no_resume:
        command.append("--no-resume")
    if args.allow_cpu:
        command.append("--allow-cpu")
    return command


def _launch_seed(args: argparse.Namespace, seed: int, output_dir: Path) -> int:
    gpu_count = torch.cuda.device_count()
    models = list(REPLICATION_MODELS)
    if gpu_count == 0:
        if not args.allow_cpu:
            raise RuntimeError("No CUDA GPU visible")
        for model in models:
            subprocess.run(_worker_command(args, model, seed, output_dir), check=True)
        return 0
    width = min(gpu_count, 2, len(models))
    for start in range(0, len(models), width):
        active = []
        for gpu, model in enumerate(models[start:start + width]):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            log_path = output_dir / f"{MODEL_IDS[model]}-seed-{seed}.log"
            handle = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(_worker_command(args, model, seed, output_dir), env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
            active.append((model, process, handle, log_path))
            print(f"started seed={seed} {MODEL_IDS[model]} on physical GPU {gpu}", flush=True)
        failures = []
        for model, process, handle, log_path in active:
            code = process.wait()
            handle.close()
            print(f"--- seed {seed} / {MODEL_IDS[model]} ---", flush=True)
            print(log_path.read_text(encoding="utf-8").rstrip(), flush=True)
            if code:
                failures.append(f"{MODEL_IDS[model]} seed {seed} exited {code}; see {log_path}")
        if failures:
            raise RuntimeError("; ".join(failures))
    return width


def run_sweep(args: argparse.Namespace) -> int:
    seeds = tuple(args.seeds)
    validate_replication_request(seeds)
    profile = base.PROFILES[args.profile]
    params = parameter_summary()
    bad = {name: record for name, record in params.items() if name in REPLICATION_MODELS and float(record["relative_error"]) >= PARAMETER_TOLERANCE}
    if bad:
        raise RuntimeError(f"parameter mismatch: {bad}")
    corpus = base.prepare_corpus(args.cache_root, profile, hf_token=os.environ.get("HF_TOKEN") or None)
    seed_label = "-".join(str(seed) for seed in seeds)
    root = args.output_root / f"replication-{args.profile}-seeds-{seed_label}"
    root.mkdir(parents=True, exist_ok=True)
    protocol = {
        "format": "minicells.native-clm-replication.v1",
        "phase": "cross-seed-development-replication",
        "profile": profile.__dict__, "seeds": list(seeds), "models": list(REPLICATION_MODELS),
        "model_ids": MODEL_IDS, "candidate_parameters": params, "corpus_manifest": corpus.manifest,
        "environment_before_launch": base.environment_record(),
        "decision_rule": "Freeze architecture; retrain T1/M4/X3/X4 on 91002 and 91003; use same-seed paired deltas vs T1; report compute separately; do not consume 91101-91103.",
    }
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    per_seed = {}
    gpus_used = 0
    for seed in seeds:
        seed_dir = root / f"seed-{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        gpus_used = max(gpus_used, _launch_seed(args, seed, seed_dir))
        rows, curves = seed_leaderboard(seed_dir, seed)
        per_seed[seed] = rows
        (seed_dir / f"replication-seed-{seed}-leaderboard.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        write_csv(seed_dir / f"replication-seed-{seed}-leaderboard.csv", rows)
        files = write_seed_visualizations(rows, curves, seed, seed_dir)
        (seed_dir / f"replication-seed-{seed}-summary.json").write_text(json.dumps({"seed": seed, "leaderboard": rows, "visualizations": files}, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    aggregate = aggregate_results(per_seed)
    paired_rows = [{"seed": seed, "id": row["id"], "model": row["model"], "validation_ppl_10m": row["validation_ppl_10m"], "ppl_delta_vs_same_seed_t1": row["ppl_delta_vs_same_seed_t1"]} for seed in seeds for row in per_seed[seed]]
    (root / "replication-aggregate.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(root / "replication-aggregate.csv", aggregate)
    write_csv(root / "replication-paired-results.csv", paired_rows)
    write_cross_seed_ppl(per_seed, root / "replication-cross-seed-ppl.svg")
    write_paired_delta(per_seed, root / "replication-paired-delta-vs-t1.svg")
    summary = {"format": "minicells.native-clm-replication-summary.v1", "profile": args.profile, "seeds": list(seeds), "models": list(REPLICATION_MODELS), "gpus_used": gpus_used, "aggregate": aggregate, "per_seed": {str(seed): per_seed[seed] for seed in seeds}}
    (root / "replication-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("=== Native CLM frozen cross-seed replication ===")
    for row in sorted(aggregate, key=lambda item: float(item["validation_ppl_mean"])):
        print(f"{row['id']:>2} mean_ppl={float(row['validation_ppl_mean']):.6f} range=[{float(row['validation_ppl_min']):.6f}, {float(row['validation_ppl_max']):.6f}] paired_delta={float(row['paired_ppl_delta_vs_t1_mean']):+.6f}")
    return 0


def main() -> int:
    args = parse_args()
    return run_worker(args) if args.command == "worker" else run_sweep(args)


if __name__ == "__main__":
    raise SystemExit(main())
