from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import native_clm_runtime as base
from native_clm_candidates import C1_NAME, VERIFIED_DATASET_REVISION
from native_clm_phase_b import (
    M_NAMES,
    N_NAMES,
    PARAMETER_TOLERANCE,
    PHASE_B_NAMES,
    build_phase_b,
    estimate_flops,
    parameter_summary,
    phase_b_config,
    validation_log_token_auc,
)

base.DATASET_REVISION = VERIFIED_DATASET_REVISION
DEFAULT_BASELINE_ANCHOR = HERE / "results/dev/baseline-seed-91001"
DEFAULT_PHASE_A_ANCHOR = HERE / "results/dev/architecture-search-baseline-seed-91001"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Native CLM Phase-B modernization/native two-track search")
    sub = parser.add_subparsers(dest="command", required=True)

    sweep = sub.add_parser("sweep")
    sweep.add_argument("--track", choices=("all", "modernization", "native"), default="all")
    sweep.add_argument("--models", nargs="+", choices=PHASE_B_NAMES)
    sweep.add_argument("--profile", choices=tuple(base.PROFILES), default="baseline")
    sweep.add_argument("--seed", type=int, default=91001)
    sweep.add_argument("--cache-root", type=Path, required=True)
    sweep.add_argument("--output-root", type=Path, required=True)
    sweep.add_argument("--baseline-anchor-dir", type=Path, default=DEFAULT_BASELINE_ANCHOR)
    sweep.add_argument("--phase-a-anchor-dir", type=Path, default=DEFAULT_PHASE_A_ANCHOR)
    sweep.add_argument("--allow-cpu", action="store_true")
    sweep.add_argument("--no-resume", action="store_true")

    worker = sub.add_parser("worker")
    worker.add_argument("--model", choices=PHASE_B_NAMES, required=True)
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
    raise RuntimeError("Phase-B training requires CUDA; --allow-cpu is smoke/debug only")


def _read_rows(path: Path) -> list[dict[str, object]]:
    numeric = {
        "step", "consumed_tokens", "train_loss", "learning_rate", "grad_norm",
        "parameters", "elapsed_seconds", "tokens_per_second", "peak_vram_bytes",
        "validation_nll", "validation_ppl", "validation_tokens",
    }
    with path.open(newline="", encoding="utf-8") as handle:
        rows: list[dict[str, object]] = list(csv.DictReader(handle))
    for row in rows:
        for key in numeric:
            if key in row and row[key] != "":
                row[key] = float(row[key])
    return rows


def _selected_models(args: argparse.Namespace) -> list[str]:
    if args.models:
        return list(args.models)
    if args.track == "modernization":
        return list(M_NAMES)
    if args.track == "native":
        return list(N_NAMES)
    return list(PHASE_B_NAMES)


def train_one_phase_b(
    name: str,
    corpus: base.Corpus,
    profile: base.Profile,
    output_dir: Path,
    seed: int,
    device: torch.device,
    resume: bool,
) -> dict[str, object]:
    params = base.count_parameters(build_phase_b(name))
    t1_params = base.count_parameters(base.build_model(base.T1_NAME))
    if abs(params / t1_params - 1.0) >= PARAMETER_TOLERANCE:
        raise RuntimeError(f"parameter mismatch: {name}={params}, T1={t1_params}")

    old = (base.C0_NAME, base.build_model, base.model_config, base.estimate_flops)
    base.C0_NAME = name
    base.build_model = lambda model_name, vocab_size=base.VOCAB_SIZE: build_phase_b(model_name, vocab_size)
    base.model_config = lambda model_name, vocab_size=base.VOCAB_SIZE: {
        **phase_b_config(model_name),
        "vocab_size": vocab_size,
        "context_length": base.CONTEXT_LENGTH,
    }
    base.estimate_flops = lambda model_name, sequence_length, tokens: estimate_flops(
        model_name, tokens, sequence_length
    )
    try:
        summary = base.train_one(
            name,
            corpus=corpus,
            profile=profile,
            output_dir=output_dir,
            seed=seed,
            device=device,
            resume=resume,
        )
    finally:
        base.C0_NAME, base.build_model, base.model_config, base.estimate_flops = old

    rows = _read_rows(output_dir / name / "checkpoints.csv")
    summary["checkpoints"] = rows
    summary["validation_log_token_nll_auc"] = validation_log_token_auc(rows)
    summary["parameter_ratio_to_t1"] = params / t1_params
    summary["track"] = phase_b_config(name)["track"]
    (output_dir / name / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def run_worker(args: argparse.Namespace) -> int:
    profile = base.PROFILES[args.profile]
    corpus = base.prepare_corpus(
        args.cache_root, profile, hf_token=os.environ.get("HF_TOKEN") or None
    )
    summary = train_one_phase_b(
        args.model,
        corpus,
        profile,
        args.output_dir,
        args.seed,
        _device(args.allow_cpu),
        not args.no_resume,
    )
    final = summary["final"] or {}
    print(json.dumps({
        "model": args.model,
        "track": summary["track"],
        "parameters": summary["parameters"],
        "validation_ppl": final.get("validation_ppl"),
        "auc": summary["validation_log_token_nll_auc"],
        "tokens_per_second": final.get("tokens_per_second"),
    }, indent=2))
    return 0


def _worker_command(args: argparse.Namespace, model: str, output_dir: Path) -> list[str]:
    command = [
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
        command.append("--no-resume")
    if args.allow_cpu:
        command.append("--allow-cpu")
    return command


def _launch(args: argparse.Namespace, models: list[str], output_dir: Path) -> int:
    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        if not args.allow_cpu:
            raise RuntimeError("No CUDA GPU visible")
        for model in models:
            subprocess.run(_worker_command(args, model, output_dir), check=True)
        return 0

    width = min(gpu_count, len(models))
    for start in range(0, len(models), width):
        active: list[tuple[str, subprocess.Popen[str], object, Path]] = []
        for gpu, model in enumerate(models[start:start + width]):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            log_path = output_dir / f"{model}.log"
            handle = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                _worker_command(args, model, output_dir),
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            active.append((model, process, handle, log_path))
            print(f"started {model} on physical GPU {gpu}", flush=True)

        failures: list[str] = []
        for model, process, handle, log_path in active:
            code = process.wait()
            handle.close()
            print(f"--- {model} ---", flush=True)
            print(log_path.read_text(encoding="utf-8").rstrip(), flush=True)
            if code:
                failures.append(f"{model} exited {code}; see {log_path}")
        if failures:
            raise RuntimeError("; ".join(failures))
    return width


def _anchor_row(summary_path: Path, checkpoints_path: Path, label: str) -> dict[str, object]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = _read_rows(checkpoints_path)
    auc = validation_log_token_auc(rows)
    final = summary["final"]
    return {
        "id": label,
        "model": summary["model"],
        "kind": "frozen-anchor",
        "track": "anchor",
        "parameters": int(summary["parameters"]),
        "validation_ppl_10m": float(final["validation_ppl"]),
        "validation_nll_10m": float(final["validation_nll"]),
        "log_token_nll_auc": auc["auc"],
        "log_token_nll_auc_mean": auc["mean_nll"],
        "train_flops_estimate": float(summary["flops"]["train_flops_estimate"]),
        "inference_flops_per_token_estimate": float(summary["flops"]["inference_forward_flops_per_token"]),
        "tokens_per_second": float(final["tokens_per_second"]),
        "peak_vram_bytes": int(final["peak_vram_bytes"]),
    }


def _leaderboard(
    baseline_anchor: Path,
    phase_a_anchor: Path,
    output_dir: Path,
    models: list[str],
) -> list[dict[str, object]]:
    rows = [
        _anchor_row(baseline_anchor / "T1-summary.json", baseline_anchor / "T1-checkpoints.csv", "T1"),
        _anchor_row(baseline_anchor / "C0-summary.json", baseline_anchor / "C0-checkpoints.csv", "C0"),
        _anchor_row(phase_a_anchor / "C1-summary.json", phase_a_anchor / "C1-checkpoints.csv", "C1"),
    ]
    for model in models:
        summary = json.loads((output_dir / model / "summary.json").read_text(encoding="utf-8"))
        final = summary["final"]
        auc = summary["validation_log_token_nll_auc"]
        config = phase_b_config(model)
        rows.append({
            "id": model.split("-")[0],
            "model": model,
            "kind": "phase-B-candidate",
            "track": config["track"],
            "parameters": int(summary["parameters"]),
            "validation_ppl_10m": float(final["validation_ppl"]),
            "validation_nll_10m": float(final["validation_nll"]),
            "log_token_nll_auc": float(auc["auc"]),
            "log_token_nll_auc_mean": float(auc["mean_nll"]),
            "train_flops_estimate": float(summary["flops"]["train_flops_estimate"]),
            "inference_flops_per_token_estimate": float(summary["flops"]["inference_forward_flops_per_token"]),
            "tokens_per_second": float(final["tokens_per_second"]),
            "peak_vram_bytes": int(final["peak_vram_bytes"]),
        })

    t1_params = int(rows[0]["parameters"])
    t1_ppl = float(rows[0]["validation_ppl_10m"])
    c1_ppl = float(rows[2]["validation_ppl_10m"])
    c1_auc = float(rows[2]["log_token_nll_auc_mean"])
    for row in rows:
        row["parameter_ratio_to_t1"] = int(row["parameters"]) / t1_params
        row["ppl_over_t1"] = float(row["validation_ppl_10m"]) / t1_ppl
        row["ppl_over_c1"] = float(row["validation_ppl_10m"]) / c1_ppl
        row["auc_mean_over_c1"] = float(row["log_token_nll_auc_mean"]) / c1_auc
    return rows


def run_sweep(args: argparse.Namespace) -> int:
    if args.seed != 91001:
        raise RuntimeError("Phase-B is frozen to development seed 91001; do not consume held-back seeds")
    models = _selected_models(args)
    if len(set(models)) != len(models):
        raise RuntimeError("duplicate models in sweep")
    profile = base.PROFILES[args.profile]
    parameters = parameter_summary()
    bad = {
        name: record
        for name, record in parameters.items()
        if name in PHASE_B_NAMES and float(record["relative_error"]) >= PARAMETER_TOLERANCE
    }
    if bad:
        raise RuntimeError(f"parameter mismatch: {bad}")

    corpus = base.prepare_corpus(
        args.cache_root, profile, hf_token=os.environ.get("HF_TOKEN") or None
    )
    output_dir = args.output_root / f"phase-b-{args.profile}-seed-{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "format": "minicells.native-clm-phase-b.v1",
        "phase": "B-two-track",
        "track_request": args.track,
        "profile": profile.__dict__,
        "seed": args.seed,
        "models": models,
        "modernization_models": list(M_NAMES),
        "native_models": list(N_NAMES),
        "candidate_parameters": parameters,
        "candidate_configs": {name: phase_b_config(name) for name in models},
        "corpus_manifest": corpus.manifest,
        "environment_before_launch": base.environment_record(),
        "decision_rule": (
            "C1 is the parent anchor. M-series measures transferable modern-LM components; "
            "N-series measures CLM-native mechanisms. Rank 10M PPL and log-token NLL AUC; "
            "report compute separately; do not consume held-back seeds before interpretation."
        ),
    }
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    gpus_used = _launch(args, models, output_dir)
    rows = _leaderboard(args.baseline_anchor_dir, args.phase_a_anchor_dir, output_dir, models)
    (output_dir / "phase-b-leaderboard.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "phase-b-leaderboard.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "format": "minicells.native-clm-phase-b-summary.v1",
        "profile": args.profile,
        "seed": args.seed,
        "models": models,
        "gpus_used": gpus_used,
        "leaderboard": rows,
    }
    (output_dir / "phase-b-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("=== Native CLM Phase-B leaderboard ===")
    for row in sorted(rows, key=lambda item: float(item["validation_ppl_10m"])):
        print(
            f"{row['id']:>3} track={row['track']:<13} "
            f"ppl={float(row['validation_ppl_10m']):8.4f} "
            f"vsC1={float(row['ppl_over_c1']):7.4f} "
            f"auc={float(row['log_token_nll_auc_mean']):7.4f} "
            f"params={int(row['parameters']):,} "
            f"tok/s={float(row['tokens_per_second']):,.0f}"
        )
    print(f"output: {output_dir}")
    return 0


def main() -> int:
    args = parse_args()
    return run_worker(args) if args.command == "worker" else run_sweep(args)


if __name__ == "__main__":
    raise SystemExit(main())
