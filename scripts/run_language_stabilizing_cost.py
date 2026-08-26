from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import torch

matplotlib.use("Agg")

ROOT = Path.cwd()
if not (ROOT / "research").exists():
    ROOT = Path("/kaggle/working/mini-cells")
if not (ROOT / "research").exists():
    raise RuntimeError("Run from a mini-cells checkout or /kaggle/working/mini-cells")
sys.path.insert(0, str(ROOT / "research"))

from minicells.language_data import encode_story_stream, iter_tinystories, load_tokenizer  # noqa: E402


OUT = ROOT / "results" / "language-stabilizing-cost-v1"
SOURCE_005 = ROOT / "artifacts" / "experiments" / "005-consumer-language-bridge"
SOURCE_009 = ROOT / "artifacts" / "experiments" / "009-2d-latent-tissue"
WORKER = ROOT / "scripts" / "run_language_stabilizing_variant.py"
MODELS = (
    "transformer-s",
    "minicells-v2-fixed",
    "minicells-v2-stable",
    "minicells-2d-k4-fixed",
    "minicells-2d-k4-stable",
)
DISPLAY = {
    "transformer-s": "Transformer-S (LLM baseline)",
    "minicells-v2-fixed": "MiniCells 1D fixed",
    "minicells-v2-stable": "MiniCells 1D stabilizing",
    "minicells-2d-k4-fixed": "MiniCells 2D fixed",
    "minicells-2d-k4-stable": "MiniCells 2D stabilizing",
}
CHECKPOINTS = (250_000, 500_000, 1_000_000, 2_000_000)
TRAIN_STREAM_TOKENS = 3_000_000
VALIDATION_STREAM_TOKENS = 200_000
QUALITY_TARGETS = (100.0, 75.0, 60.0, 50.0)


def tensor_sha256(tensor: torch.Tensor) -> str:
    values = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def prepare_corpus() -> tuple[Path, dict[str, object]]:
    source_tokenizer = SOURCE_005 / "tokenizer.json"
    source_manifest_path = SOURCE_005 / "corpus-manifest.json"
    source_009_manifest_path = SOURCE_009 / "corpus-manifest.json"
    if not source_tokenizer.is_file() or not source_manifest_path.is_file():
        raise FileNotFoundError("Experiment 005 tokenizer/corpus manifest must exist before 011")
    if not source_009_manifest_path.is_file():
        raise FileNotFoundError("Experiment 009 corpus manifest must be merged before 011")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_009_manifest = json.loads(source_009_manifest_path.read_text(encoding="utf-8"))
    source_tokenizer_sha = hashlib.sha256(source_tokenizer.read_bytes()).hexdigest()
    if source_tokenizer_sha != source_manifest.get("tokenizer_sha256"):
        raise RuntimeError("Experiment 005 tokenizer hash mismatch")

    cache = OUT / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    tokenizer_path = cache / "tokenizer.json"
    train_path = cache / "train-tokens.pt"
    validation_path = cache / "validation-tokens.pt"
    manifest_path = cache / "corpus-manifest.json"
    expected = {
        "format": "minicells.language-stabilizing-cost-corpus.v1",
        "source_005_tokenizer_sha256": source_tokenizer_sha,
        "train_stream_tokens": TRAIN_STREAM_TOKENS,
        "validation_stream_tokens": VALIDATION_STREAM_TOKENS,
    }

    if tokenizer_path.exists() and train_path.exists() and validation_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if all(manifest.get(key) == value for key, value in expected.items()):
            train = torch.load(train_path, map_location="cpu")
            validation = torch.load(validation_path, map_location="cpu")
            if (
                tensor_sha256(train) == manifest.get("train_token_sha256")
                and tensor_sha256(validation) == manifest.get("validation_token_sha256")
                and hashlib.sha256(tokenizer_path.read_bytes()).hexdigest() == source_tokenizer_sha
            ):
                return cache, manifest

    shutil.copy2(source_tokenizer, tokenizer_path)
    tokenizer = load_tokenizer(tokenizer_path)
    train, train_stories = encode_story_stream(
        tokenizer,
        iter_tinystories("train"),
        target_tokens=TRAIN_STREAM_TOKENS,
    )
    validation, validation_stories = encode_story_stream(
        tokenizer,
        iter_tinystories("validation"),
        target_tokens=VALIDATION_STREAM_TOKENS,
    )
    train_sha = tensor_sha256(train)
    validation_sha = tensor_sha256(validation)
    if train_sha != source_009_manifest.get("train_token_sha256"):
        raise RuntimeError("Experiment 011 train stream does not reproduce Experiment 009")
    if validation_sha != source_009_manifest.get("validation_token_sha256"):
        raise RuntimeError("Experiment 011 validation stream does not reproduce Experiment 009")

    torch.save(train, train_path)
    torch.save(validation, validation_path)
    manifest = {
        **expected,
        "dataset": source_manifest.get("dataset"),
        "streaming": True,
        "vocab_size_actual": tokenizer.get_vocab_size(),
        "train_stories_consumed": train_stories,
        "validation_stories_consumed": validation_stories,
        "train_token_sha256": train_sha,
        "validation_token_sha256": validation_sha,
        "tokenizer_sha256": source_tokenizer_sha,
        "reproduces_009_corpus": True,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return cache, manifest


def worker_command(model: str, cache_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(WORKER),
        "--model",
        model,
        "--cache-dir",
        str(cache_dir),
        "--output-dir",
        str(OUT),
    ]


def run_models(cache_dir: Path) -> int:
    available = torch.cuda.device_count()
    if available < 1:
        raise RuntimeError("Experiment 011 requires CUDA")
    gpu_count = min(2, available)
    queue = list(MODELS)
    while queue:
        batch = queue[:gpu_count]
        queue = queue[gpu_count:]
        active: list[tuple[str, int, subprocess.Popen[str], Path, object]] = []
        for gpu_index, model in enumerate(batch):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
            log_path = OUT / f"{model}.log"
            handle = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                worker_command(model, cache_dir),
                cwd=ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            active.append((model, gpu_index, process, log_path, handle))
            print(f"started {model:24s} on physical GPU {gpu_index}")
        failures: list[str] = []
        for model, gpu_index, process, log_path, handle in active:
            code = process.wait()
            handle.close()
            print(f"--- {model} / GPU {gpu_index} ---")
            print(log_path.read_text(encoding="utf-8").rstrip())
            if code != 0:
                failures.append(f"{model} exited {code}; see {log_path}")
        if failures:
            raise RuntimeError("; ".join(failures))
    return gpu_count


def summarize() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    checkpoint_frames = [pd.read_csv(OUT / f"{model}-checkpoints.csv") for model in MODELS]
    checkpoints = pd.concat(checkpoint_frames, ignore_index=True)
    checkpoints.to_csv(OUT / "checkpoints.csv", index=False)

    worker_rows: list[dict[str, object]] = []
    for model in MODELS:
        worker = json.loads((OUT / f"{model}-worker.json").read_text(encoding="utf-8"))
        final = checkpoints.loc[checkpoints["model"] == model].sort_values("consumed_tokens").iloc[-1]
        worker_rows.append(
            {
                "model": model,
                "display_name": DISPLAY[model],
                "parameters": int(worker["parameters"]),
                "stable_training": bool(worker["stable_training"]),
                "final_ppl_2m": float(final["validation_ppl"]),
                "training_elapsed_seconds": float(worker["training_elapsed_seconds"]),
                "training_tokens_per_second": float(worker["training_tokens_per_second"]),
                "seconds_per_million_tokens": float(worker["seconds_per_million_tokens"]),
                "projected_t4_hours_per_billion_tokens": float(worker["seconds_per_million_tokens"] * 1000.0 / 3600.0),
                "peak_vram_bytes": int(worker["peak_vram_bytes"]),
                "peak_vram_gib": float(worker["peak_vram_bytes"] / (1024**3)),
                "avg_recurrent_iterations": worker.get("avg_recurrent_iterations"),
                "best_adaptive_steps": (
                    worker.get("best_adaptive_halting_within_1pct_ppl") or {}
                ).get("avg_total_steps"),
                "best_adaptive_iteration_saving": (
                    worker.get("best_adaptive_halting_within_1pct_ppl") or {}
                ).get("theoretical_iteration_saving"),
            }
        )
    summary = pd.DataFrame(worker_rows)
    transformer = summary.loc[summary["model"] == "transformer-s"].iloc[0]
    summary["training_cost_ratio_to_transformer"] = (
        summary["seconds_per_million_tokens"] / float(transformer["seconds_per_million_tokens"])
    )
    summary["ppl_ratio_to_transformer"] = summary["final_ppl_2m"] / float(transformer["final_ppl_2m"])
    summary.to_csv(OUT / "model-summary.csv", index=False)

    quality_rows: list[dict[str, object]] = []
    for model, group in checkpoints.groupby("model"):
        ordered = group.sort_values("consumed_tokens")
        for target in QUALITY_TARGETS:
            reached = ordered.loc[ordered["validation_ppl"] <= target]
            quality_rows.append(
                {
                    "model": model,
                    "display_name": DISPLAY[model],
                    "target_ppl": target,
                    "reached": not reached.empty,
                    "training_seconds_to_target": (
                        float(reached.iloc[0]["training_elapsed_seconds"]) if not reached.empty else math.nan
                    ),
                    "tokens_to_target": (
                        int(reached.iloc[0]["consumed_tokens"]) if not reached.empty else math.nan
                    ),
                }
            )
    quality = pd.DataFrame(quality_rows)
    quality.to_csv(OUT / "cost-to-quality.csv", index=False)

    halting_frames = []
    for model in MODELS:
        path = OUT / f"{model}-halting.csv"
        if path.is_file():
            halting_frames.append(pd.read_csv(path))
    halting = pd.concat(halting_frames, ignore_index=True) if halting_frames else pd.DataFrame()
    if not halting.empty:
        halting.to_csv(OUT / "halting-sweep.csv", index=False)

    return checkpoints, summary, quality, halting


def _pair(summary: pd.DataFrame, fixed_name: str, stable_name: str) -> dict[str, float]:
    by_model = summary.set_index("model")
    fixed = by_model.loc[fixed_name]
    stable = by_model.loc[stable_name]
    return {
        "ppl_ratio_stable_to_fixed": float(stable["final_ppl_2m"] / fixed["final_ppl_2m"]),
        "training_seconds_ratio_stable_to_fixed": float(stable["training_elapsed_seconds"] / fixed["training_elapsed_seconds"]),
        "throughput_ratio_stable_to_fixed": float(stable["training_tokens_per_second"] / fixed["training_tokens_per_second"]),
        "vram_ratio_stable_to_fixed": float(stable["peak_vram_bytes"] / fixed["peak_vram_bytes"]),
        "stable_avg_recurrent_iterations": float(stable["avg_recurrent_iterations"]),
        "stable_training_iteration_fraction": float(stable["avg_recurrent_iterations"] / 12.0),
    }


def make_decision(summary: pd.DataFrame, gpu_count: int) -> dict[str, object]:
    by_model = summary.set_index("model")
    one_d = _pair(summary, "minicells-v2-fixed", "minicells-v2-stable")
    two_d = _pair(summary, "minicells-2d-k4-fixed", "minicells-2d-k4-stable")
    transformer = by_model.loc["transformer-s"]
    one_d_stable = by_model.loc["minicells-v2-stable"]
    two_d_stable = by_model.loc["minicells-2d-k4-stable"]
    pair_ok = (
        one_d["training_seconds_ratio_stable_to_fixed"] < 1.0
        and two_d["training_seconds_ratio_stable_to_fixed"] < 1.0
        and one_d["ppl_ratio_stable_to_fixed"] <= 1.05
        and two_d["ppl_ratio_stable_to_fixed"] <= 1.05
    )
    status = "PASS_COST_SIGNAL" if pair_ok else "MIXED_COST_SIGNAL"
    diagnosis = (
        "RANDOM_DEPTH_STABILIZATION_REDUCES_MEASURED_TRAINING_COST_WITH_BOUNDED_QUALITY_LOSS"
        if pair_ok
        else "SELF_STABILIZING_TRAINING_SHOWS_A_MIXED_QUALITY_COST_TRADEOFF"
    )
    return {
        "format": "minicells.language-stabilizing-cost.v1",
        "experiment": "MINI Cells Experiment 011 — Self-Stabilizing Training and Cost",
        "status": status,
        "diagnosis": diagnosis,
        "question": "Can randomized-depth, stability-regularized cellular training reduce recurrent training cost, and how does the measured cost compare with a parameter-matched Transformer-S LLM baseline?",
        "budget": {
            "tokens_per_model": 2_000_000,
            "checkpoints": list(CHECKPOINTS),
            "context_length": 128,
            "models": list(MODELS),
        },
        "cost_measurement": {
            "hardware": "Tesla T4 on Kaggle when available",
            "metric": "synchronized train-step wall clock only",
            "excludes": ["validation", "plotting", "post-training adaptive-halting probe", "result publication"],
            "gpu_count_used_for_parallel_execution": gpu_count,
            "important_scope": "Transformer-S is a small parameter-matched LLM baseline; these measurements are not a claim about the cost of all LLMs or frontier-scale training.",
        },
        "training_recipe": {
            "fixed_recurrent_depth": [4, 4, 4],
            "stabilizing_random_depth_per_stage": [2, 4],
            "expected_stabilizing_iterations_total": 9.0,
            "fixed_iterations_total": 12,
            "stability_regularizer": "mean relative RMS of each stage's final state update",
            "stability_weight": 0.10,
            "step_embedding_scale_stable": 0.25,
        },
        "one_d": one_d,
        "two_d": two_d,
        "transformer_baseline": {
            "parameters": int(transformer["parameters"]),
            "ppl_2m": float(transformer["final_ppl_2m"]),
            "seconds_per_million_tokens": float(transformer["seconds_per_million_tokens"]),
            "peak_vram_gib": float(transformer["peak_vram_gib"]),
        },
        "stable_vs_transformer": {
            "one_d_training_cost_ratio": float(one_d_stable["training_cost_ratio_to_transformer"]),
            "one_d_ppl_ratio": float(one_d_stable["ppl_ratio_to_transformer"]),
            "two_d_training_cost_ratio": float(two_d_stable["training_cost_ratio_to_transformer"]),
            "two_d_ppl_ratio": float(two_d_stable["ppl_ratio_to_transformer"]),
        },
    }


def save_plots(checkpoints: pd.DataFrame, summary: pd.DataFrame, quality: pd.DataFrame) -> None:
    fig, axis = plt.subplots(figsize=(8.5, 5.3))
    for model, group in checkpoints.groupby("model"):
        ordered = group.sort_values("consumed_tokens")
        axis.plot(ordered["consumed_tokens"], ordered["validation_ppl"], marker="o", label=DISPLAY[model])
    axis.set_xscale("log")
    axis.set_xlabel("Consumed training tokens")
    axis.set_ylabel("Validation perplexity")
    axis.set_title("Experiment 011 — quality vs training tokens")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "ppl-vs-training-tokens.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.5, 5.3))
    for model, group in checkpoints.groupby("model"):
        ordered = group.sort_values("training_elapsed_seconds")
        axis.plot(ordered["training_elapsed_seconds"], ordered["validation_ppl"], marker="o", label=DISPLAY[model])
    axis.set_xlabel("Measured training-step wall time (seconds)")
    axis.set_ylabel("Validation perplexity")
    axis.set_title("Experiment 011 — quality vs measured T4 training time")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "ppl-vs-training-seconds.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    ordered_summary = summary.sort_values("seconds_per_million_tokens")
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.bar(ordered_summary["display_name"], ordered_summary["seconds_per_million_tokens"])
    axis.set_ylabel("Seconds per 1M training tokens")
    axis.set_title("Experiment 011 — measured training cost on T4")
    axis.tick_params(axis="x", rotation=18)
    fig.tight_layout()
    fig.savefig(OUT / "training-cost-per-million.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5))
    axis.bar(summary["display_name"], summary["peak_vram_gib"])
    axis.set_ylabel("Peak allocated VRAM (GiB)")
    axis.set_title("Experiment 011 — peak VRAM")
    axis.tick_params(axis="x", rotation=18)
    fig.tight_layout()
    fig.savefig(OUT / "peak-vram.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.5, 5.3))
    axis.scatter(summary["seconds_per_million_tokens"], summary["final_ppl_2m"])
    for _, row in summary.iterrows():
        axis.annotate(
            str(row["display_name"]),
            (float(row["seconds_per_million_tokens"]), float(row["final_ppl_2m"])),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_xlabel("Seconds per 1M training tokens on T4")
    axis.set_ylabel("Validation PPL at 2M tokens")
    axis.set_title("Experiment 011 — quality–cost frontier")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "quality-cost-frontier.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.5, 5.3))
    for model, group in quality.groupby("model"):
        reached = group.loc[group["reached"] == True].sort_values("target_ppl", ascending=False)  # noqa: E712
        if reached.empty:
            continue
        axis.plot(
            reached["target_ppl"],
            reached["training_seconds_to_target"],
            marker="o",
            label=DISPLAY[model],
        )
    axis.invert_xaxis()
    axis.set_xlabel("Target validation PPL (lower is harder)")
    axis.set_ylabel("Training seconds to first measured checkpoint")
    axis.set_title("Experiment 011 — measured cost to quality")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "cost-to-quality.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    cellular = summary.loc[summary["model"] != "transformer-s"].copy()
    cellular = cellular.dropna(subset=["best_adaptive_steps"])
    if not cellular.empty:
        fig, axis = plt.subplots(figsize=(8.5, 5))
        axis.bar(cellular["display_name"], cellular["best_adaptive_steps"])
        axis.axhline(12.0, linestyle="--", linewidth=1, label="Fixed maximum = 12")
        axis.set_ylabel("Average recurrent iterations")
        axis.set_title("Experiment 011 — best adaptive inference within 1% PPL budget")
        axis.tick_params(axis="x", rotation=18)
        axis.legend()
        fig.tight_layout()
        fig.savefig(OUT / "adaptive-iterations.png", dpi=170, bbox_inches="tight")
        plt.close(fig)


def write_task_files(corpus_manifest: dict[str, object]) -> None:
    task = {
        "format": "minicells.language-stabilizing-cost-task.v1",
        "goal": "Measure whether self-stabilizing randomized-depth training reduces recurrent cost, while comparing quality and measured T4 cost against fixed cellular baselines and a parameter-matched Transformer-S LLM baseline.",
        "models": list(MODELS),
        "budget_tokens_per_model": 2_000_000,
        "checkpoints": list(CHECKPOINTS),
        "same_corpus_as_009": True,
        "fairness": {
            "same_tokenizer": True,
            "same_training_stream": True,
            "same_batch_schedule": True,
            "same_optimizer_family": "AdamW",
            "parameter_matched_transformer": True,
            "fixed_and_stable_pairs_share_initialization_seed": True,
        },
        "cost_definition": "Synchronized train-step wall clock; validation and post-training probes excluded.",
        "warning": "Transformer-S is a small parameter-matched LLM baseline, not a proxy for every LLM or frontier-scale training.",
    }
    (OUT / "task-spec.json").write_text(json.dumps(task, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "corpus-manifest.json").write_text(
        json.dumps(corpus_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cache, corpus_manifest = prepare_corpus()
    write_task_files(corpus_manifest)
    gpu_count = run_models(cache)
    checkpoints, summary, quality, _ = summarize()
    decision = make_decision(summary, gpu_count)
    (OUT / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    save_plots(checkpoints, summary, quality)
    print("=== decision ===")
    print(json.dumps(decision, indent=2))
    print("=== model summary ===")
    print(summary.to_string(index=False))
    print("=== files ===")
    for path in sorted(OUT.iterdir()):
        if path.is_file():
            print(path.name, path.stat().st_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
