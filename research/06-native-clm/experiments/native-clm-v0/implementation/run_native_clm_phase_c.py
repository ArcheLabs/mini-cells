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
from native_clm_candidates import VERIFIED_DATASET_REVISION
from native_clm_phase_c import (
    PARAMETER_TOLERANCE,
    PHASE_C_NAMES,
    build_phase_c,
    estimate_flops,
    parameter_summary,
    phase_c_config,
    validation_log_token_auc,
)
from native_clm_visualize import write_phase_c_visualizations

base.DATASET_REVISION = VERIFIED_DATASET_REVISION
DEFAULT_BASELINE = HERE / "results/dev/baseline-seed-91001"
DEFAULT_PHASE_A = HERE / "results/dev/architecture-search-baseline-seed-91001"
DEFAULT_PHASE_B = HERE / "results/dev/phase-b-baseline-seed-91001"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Native CLM Phase-C winner cross-over search")
    sub = parser.add_subparsers(dest="command", required=True)
    sweep = sub.add_parser("sweep")
    sweep.add_argument("--models", nargs="+", choices=PHASE_C_NAMES, default=list(PHASE_C_NAMES))
    sweep.add_argument("--profile", choices=tuple(base.PROFILES), default="baseline")
    sweep.add_argument("--seed", type=int, default=91001)
    sweep.add_argument("--cache-root", type=Path, required=True)
    sweep.add_argument("--output-root", type=Path, required=True)
    sweep.add_argument("--baseline-anchor-dir", type=Path, default=DEFAULT_BASELINE)
    sweep.add_argument("--phase-a-anchor-dir", type=Path, default=DEFAULT_PHASE_A)
    sweep.add_argument("--phase-b-anchor-dir", type=Path, default=DEFAULT_PHASE_B)
    sweep.add_argument("--allow-cpu", action="store_true")
    sweep.add_argument("--no-resume", action="store_true")
    worker = sub.add_parser("worker")
    worker.add_argument("--model", choices=PHASE_C_NAMES, required=True)
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
    raise RuntimeError("Phase-C training requires CUDA; --allow-cpu is smoke/debug only")


def _read_rows(path: Path) -> list[dict[str, object]]:
    numeric = {
        "step", "consumed_tokens", "train_loss", "learning_rate", "grad_norm", "parameters",
        "elapsed_seconds", "tokens_per_second", "peak_vram_bytes", "validation_nll",
        "validation_ppl", "validation_tokens",
    }
    with path.open(newline="", encoding="utf-8") as handle:
        rows: list[dict[str, object]] = list(csv.DictReader(handle))
    for row in rows:
        for key in numeric:
            if key in row and row[key] != "":
                row[key] = float(row[key])
    return rows


def train_one_phase_c(name: str, corpus: base.Corpus, profile: base.Profile, output_dir: Path, seed: int, device: torch.device, resume: bool) -> dict[str, object]:
    params = base.count_parameters(build_phase_c(name))
    t1_params = base.count_parameters(base.build_model(base.T1_NAME))
    if abs(params / t1_params - 1.0) >= PARAMETER_TOLERANCE:
        raise RuntimeError(f"parameter mismatch: {name}={params}, T1={t1_params}")
    old = (base.C0_NAME, base.build_model, base.model_config, base.estimate_flops)
    base.C0_NAME = name
    base.build_model = lambda model_name, vocab_size=base.VOCAB_SIZE: build_phase_c(model_name, vocab_size)
    base.model_config = lambda model_name, vocab_size=base.VOCAB_SIZE: {**phase_c_config(model_name), "vocab_size": vocab_size, "context_length": base.CONTEXT_LENGTH}
    base.estimate_flops = lambda model_name, sequence_length, tokens: estimate_flops(model_name, tokens, sequence_length)
    try:
        summary = base.train_one(name, corpus=corpus, profile=profile, output_dir=output_dir, seed=seed, device=device, resume=resume)
    finally:
        base.C0_NAME, base.build_model, base.model_config, base.estimate_flops = old
    rows = _read_rows(output_dir / name / "checkpoints.csv")
    summary["checkpoints"] = rows
    summary["validation_log_token_nll_auc"] = validation_log_token_auc(rows)
    summary["parameter_ratio_to_t1"] = params / t1_params
    summary["track"] = "cross-over"
    (output_dir / name / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def run_worker(args: argparse.Namespace) -> int:
    profile = base.PROFILES[args.profile]
    corpus = base.prepare_corpus(args.cache_root, profile, hf_token=os.environ.get("HF_TOKEN") or None)
    summary = train_one_phase_c(args.model, corpus, profile, args.output_dir, args.seed, _device(args.allow_cpu), not args.no_resume)
    final = summary["final"] or {}
    print(json.dumps({"model": args.model, "parameters": summary["parameters"], "validation_ppl": final.get("validation_ppl"), "auc": summary["validation_log_token_nll_auc"], "tokens_per_second": final.get("tokens_per_second")}, indent=2))
    return 0


def _worker_command(args: argparse.Namespace, model: str, output_dir: Path) -> list[str]:
    command = [sys.executable, str(Path(__file__).resolve()), "worker", "--model", model, "--profile", args.profile, "--seed", str(args.seed), "--cache-root", str(args.cache_root), "--output-dir", str(output_dir)]
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
        active = []
        for gpu, model in enumerate(models[start:start + width]):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            log_path = output_dir / f"{model}.log"
            handle = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(_worker_command(args, model, output_dir), env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
            active.append((model, process, handle, log_path))
            print(f"started {model} on physical GPU {gpu}", flush=True)
        failures = []
        for model, process, handle, log_path in active:
            code = process.wait()
            handle.close()
            print(f"--- {model} ---\n{log_path.read_text(encoding='utf-8').rstrip()}", flush=True)
            if code:
                failures.append(f"{model} exited {code}; see {log_path}")
        if failures:
            raise RuntimeError("; ".join(failures))
    return width


def _anchor_row(summary_path: Path, checkpoints_path: Path, label: str, track: str = "anchor") -> tuple[dict[str, object], list[dict[str, object]]]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = _read_rows(checkpoints_path)
    auc = validation_log_token_auc(rows)
    final = summary["final"]
    record = {
        "id": label,
        "model": summary["model"],
        "kind": "frozen-anchor",
        "track": track,
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
    return record, rows


def _leaderboard(args: argparse.Namespace, output_dir: Path, models: list[str]) -> tuple[list[dict[str, object]], dict[str, list[dict[str, object]]]]:
    sources = [
        (args.baseline_anchor_dir / "T1-summary.json", args.baseline_anchor_dir / "T1-checkpoints.csv", "T1", "anchor"),
        (args.baseline_anchor_dir / "C0-summary.json", args.baseline_anchor_dir / "C0-checkpoints.csv", "C0", "anchor"),
        (args.phase_a_anchor_dir / "C1-summary.json", args.phase_a_anchor_dir / "C1-checkpoints.csv", "C1", "anchor"),
        (args.phase_b_anchor_dir / "M4-summary.json", args.phase_b_anchor_dir / "M4-checkpoints.csv", "M4", "modernization-winner"),
        (args.phase_b_anchor_dir / "N1-summary.json", args.phase_b_anchor_dir / "N1-checkpoints.csv", "N1", "native-winner"),
    ]
    leaderboard: list[dict[str, object]] = []
    curves: dict[str, list[dict[str, object]]] = {}
    for summary_path, checkpoints_path, label, track in sources:
        row, curve = _anchor_row(summary_path, checkpoints_path, label, track)
        leaderboard.append(row)
        curves[label] = curve
    for model in models:
        summary = json.loads((output_dir / model / "summary.json").read_text(encoding="utf-8"))
        final = summary["final"]
        auc = summary["validation_log_token_nll_auc"]
        model_id = model.split("-")[0]
        curve = _read_rows(output_dir / model / "checkpoints.csv")
        curves[model_id] = curve
        leaderboard.append({
            "id": model_id,
            "model": model,
            "kind": "phase-C-candidate",
            "track": "cross-over",
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
    t1 = next(row for row in leaderboard if row["id"] == "T1")
    m4 = next(row for row in leaderboard if row["id"] == "M4")
    n1 = next(row for row in leaderboard if row["id"] == "N1")
    for row in leaderboard:
        row["parameter_ratio_to_t1"] = int(row["parameters"]) / int(t1["parameters"])
        row["ppl_over_t1"] = float(row["validation_ppl_10m"]) / float(t1["validation_ppl_10m"])
        row["ppl_over_m4"] = float(row["validation_ppl_10m"]) / float(m4["validation_ppl_10m"])
        row["ppl_over_n1"] = float(row["validation_ppl_10m"]) / float(n1["validation_ppl_10m"])
    return leaderboard, curves


def run_sweep(args: argparse.Namespace) -> int:
    if args.seed != 91001:
        raise RuntimeError("Phase-C is frozen to development seed 91001; do not consume held-back seeds")
    models = list(args.models)
    if len(set(models)) != len(models):
        raise RuntimeError("duplicate models in sweep")
    profile = base.PROFILES[args.profile]
    parameters = parameter_summary()
    bad = {name: record for name, record in parameters.items() if name in PHASE_C_NAMES and float(record["relative_error"]) >= PARAMETER_TOLERANCE}
    if bad:
        raise RuntimeError(f"parameter mismatch: {bad}")
    corpus = base.prepare_corpus(args.cache_root, profile, hf_token=os.environ.get("HF_TOKEN") or None)
    output_dir = args.output_root / f"phase-c-{args.profile}-seed-{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "format": "minicells.native-clm-phase-c.v1",
        "phase": "C-winner-cross-over",
        "profile": profile.__dict__,
        "seed": args.seed,
        "models": models,
        "candidate_parameters": parameters,
        "candidate_configs": {name: phase_c_config(name) for name in models},
        "corpus_manifest": corpus.manifest,
        "environment_before_launch": base.environment_record(),
        "decision_rule": "Cross only evidenced winners: N1 residual refinement with SwiGLU/RoPE, and use X4 solely to re-test RMSNorm in the combined state. Rank quality by 10M PPL and log-token NLL AUC; compute remains separate.",
    }
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    gpus_used = _launch(args, models, output_dir)
    leaderboard, curves = _leaderboard(args, output_dir, models)
    (output_dir / "phase-c-leaderboard.json").write_text(json.dumps(leaderboard, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output_dir / "phase-c-leaderboard.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(leaderboard[0]))
        writer.writeheader()
        writer.writerows(leaderboard)
    visualizations = write_phase_c_visualizations(leaderboard, curves, output_dir)
    summary = {"format": "minicells.native-clm-phase-c-summary.v1", "profile": args.profile, "seed": args.seed, "models": models, "gpus_used": gpus_used, "leaderboard": leaderboard, "visualizations": visualizations}
    (output_dir / "phase-c-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("=== Native CLM Phase-C leaderboard ===")
    for row in sorted(leaderboard, key=lambda item: float(item["validation_ppl_10m"])):
        print(f"{row['id']:>3} ppl={float(row['validation_ppl_10m']):8.4f} params={int(row['parameters']):,} FLOPs/T1={float(row['train_flops_estimate']) / float(leaderboard[0]['train_flops_estimate']):.2f}x tok/s={float(row['tokens_per_second']):,.0f}")
    print("visualizations:", ", ".join(visualizations["files"].values()))
    print("output:", output_dir)
    return 0


def main() -> int:
    args = parse_args()
    return run_worker(args) if args.command == "worker" else run_sweep(args)


if __name__ == "__main__":
    raise SystemExit(main())
