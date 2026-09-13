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

import native_clm_runtime as runtime

# Verified immutable TinyStories commit. The previous value expanded the correct
# short SHA f54c09f with an incorrect suffix.
runtime.DATASET_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"

from native_clm_runtime import (
    C0_NAME,
    MODEL_NAMES,
    PROFILES,
    T1_NAME,
    environment_record,
    parameter_summary,
    prepare_corpus,
    train_one,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Native CLM long-lived baseline trainer")
    sub = p.add_subparsers(dest="command", required=True)

    pair = sub.add_parser("pair", help="prepare corpus and train C0/T1 fairly")
    pair.add_argument("--profile", choices=tuple(PROFILES), default="baseline")
    pair.add_argument("--seed", type=int, default=91001)
    pair.add_argument("--cache-root", type=Path, required=True)
    pair.add_argument("--output-root", type=Path, required=True)
    pair.add_argument("--allow-cpu", action="store_true")
    pair.add_argument("--no-resume", action="store_true")

    worker = sub.add_parser("worker", help="train one model on the visible device")
    worker.add_argument("--model", choices=MODEL_NAMES, required=True)
    worker.add_argument("--profile", choices=tuple(PROFILES), required=True)
    worker.add_argument("--seed", type=int, required=True)
    worker.add_argument("--cache-root", type=Path, required=True)
    worker.add_argument("--output-dir", type=Path, required=True)
    worker.add_argument("--allow-cpu", action="store_true")
    worker.add_argument("--no-resume", action="store_true")
    return p.parse_args()


def _device(allow_cpu: bool) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if allow_cpu:
        return torch.device("cpu")
    raise RuntimeError("Native CLM baseline training requires CUDA; use --allow-cpu only for smoke/debug")


def run_worker(args: argparse.Namespace) -> int:
    profile = PROFILES[args.profile]
    hf_token = os.environ.get("HF_TOKEN") or None
    corpus = prepare_corpus(args.cache_root, profile, hf_token=hf_token)
    summary = train_one(
        args.model,
        corpus=corpus,
        profile=profile,
        output_dir=args.output_dir,
        seed=args.seed,
        device=_device(args.allow_cpu),
        resume=not args.no_resume,
    )
    print(json.dumps({
        "model": args.model,
        "parameters": summary["parameters"],
        "validation_ppl": summary["final"]["validation_ppl"] if summary["final"] else None,
        "tokens_per_second": summary["final"]["tokens_per_second"] if summary["final"] else None,
    }, indent=2))
    return 0


def _worker_command(args: argparse.Namespace, model: str, output_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "worker",
        "--model", model,
        "--profile", args.profile,
        "--seed", str(args.seed),
        "--cache-root", str(args.cache_root),
        "--output-dir", str(output_dir),
    ]
    if args.no_resume:
        cmd.append("--no-resume")
    if args.allow_cpu:
        cmd.append("--allow-cpu")
    return cmd


def _launch_pair(args: argparse.Namespace, output_dir: Path) -> int:
    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        if not args.allow_cpu:
            raise RuntimeError("No CUDA GPU visible")
        for model in MODEL_NAMES:
            subprocess.run(_worker_command(args, model, output_dir), check=True)
        return 0
    if gpu_count == 1:
        for model in MODEL_NAMES:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = "0"
            subprocess.run(_worker_command(args, model, output_dir), env=env, check=True)
        return 1

    active: list[tuple[str, subprocess.Popen[str], object, Path]] = []
    for gpu, model in enumerate(MODEL_NAMES):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        log_path = output_dir / f"{model}.log"
        handle = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(
            _worker_command(args, model, output_dir),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        active.append((model, proc, handle, log_path))
        print(f"started {model} on physical GPU {gpu}", flush=True)

    failures: list[str] = []
    for model, proc, handle, log_path in active:
        code = proc.wait()
        handle.close()
        print(f"--- {model} ---", flush=True)
        print(log_path.read_text(encoding="utf-8").rstrip(), flush=True)
        if code:
            failures.append(f"{model} exited {code}; see {log_path}")
    if failures:
        raise RuntimeError("; ".join(failures))
    return 2


def _read_summary(output_dir: Path, model: str) -> dict[str, object]:
    path = output_dir / model / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def run_pair(args: argparse.Namespace) -> int:
    profile = PROFILES[args.profile]
    params = parameter_summary()
    if float(params["relative_error"]) >= 0.01:
        raise RuntimeError(f"C0/T1 parameter mismatch exceeds 1%: {params}")

    hf_token = os.environ.get("HF_TOKEN") or None
    corpus = prepare_corpus(args.cache_root, profile, hf_token=hf_token)

    output_dir = args.output_root / f"{args.profile}-seed-{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "protocol.json").write_text(
        json.dumps({
            "format": "minicells.native-clm-protocol.v1",
            "profile": profile.__dict__,
            "seed": args.seed,
            "parameter_summary": params,
            "corpus_manifest": corpus.manifest,
            "environment_before_launch": environment_record(),
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    gpus_used = _launch_pair(args, output_dir)
    c0 = _read_summary(output_dir, C0_NAME)
    t1 = _read_summary(output_dir, T1_NAME)
    c0_final = c0["final"]
    t1_final = t1["final"]
    if c0_final is None or t1_final is None:
        raise RuntimeError("both models must have final validation metrics")

    c0_ppl = float(c0_final["validation_ppl"])
    t1_ppl = float(t1_final["validation_ppl"])
    c0_flops = float(c0["flops"]["train_flops_estimate"])
    t1_flops = float(t1["flops"]["train_flops_estimate"])
    summary = {
        "format": "minicells.native-clm-pair.v1",
        "profile": args.profile,
        "seed": args.seed,
        "gpus_used": gpus_used,
        "parameter_summary": params,
        "C0": c0,
        "T1": t1,
        "comparison": {
            "c0_over_t1_ppl": c0_ppl / t1_ppl,
            "c0_minus_t1_ppl": c0_ppl - t1_ppl,
            "c0_over_t1_train_flops": c0_flops / t1_flops,
            "quality_parameter_winner": C0_NAME if c0_ppl < t1_ppl else T1_NAME,
            "compute_claim_allowed": False,
            "note": "PPL under equal parameter/token budget is not an equal-FLOP comparison.",
        },
    }
    (output_dir / "pair-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("=== Native CLM pair summary ===")
    print(json.dumps({
        "profile": args.profile,
        "seed": args.seed,
        "C0_params": c0["parameters"],
        "T1_params": t1["parameters"],
        "C0_ppl": c0_ppl,
        "T1_ppl": t1_ppl,
        "C0/T1_ppl": c0_ppl / t1_ppl,
        "C0/T1_train_flops_estimate": c0_flops / t1_flops,
        "gpus_used": gpus_used,
        "output": str(output_dir / "pair-summary.json"),
    }, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "worker":
        return run_worker(args)
    return run_pair(args)


if __name__ == "__main__":
    raise SystemExit(main())
