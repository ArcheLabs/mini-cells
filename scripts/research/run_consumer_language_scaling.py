from __future__ import annotations

import json
import os
import platform
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

from minicells.language_scaling import (  # noqa: E402
    MODEL_SEED,
    SCALING_CHECKPOINTS,
    SCHEDULE_SEED,
    TRAIN_STREAM_TOKENS,
    TRANSFORMER_SEED,
    VALIDATION_STREAM_TOKENS,
    WARMUP_STEPS,
    make_scaling_decision,
    prepare_scaling_corpus,
    save_scaling_plots,
    summarize_scaling,
    write_generation_progression,
)

OUT = ROOT / "results" / "consumer-language-scaling-v1"
SOURCE_005 = ROOT / "artifacts" / "experiments" / "005-consumer-language-bridge"
SOURCE_005B = ROOT / "artifacts" / "experiments" / "005b-consumer-language-ablation"
WORKER = ROOT / "scripts" / "run_consumer_language_scaling_variant.py"
MODELS = ("minicells-v2", "transformer-s")


def require_sources() -> tuple[dict[str, object], dict[str, object]]:
    source_005 = json.loads((SOURCE_005 / "decision.json").read_text(encoding="utf-8"))
    source_005b = json.loads((SOURCE_005B / "decision.json").read_text(encoding="utf-8"))
    if source_005.get("format") != "minicells.consumer-language-bridge.v1":
        raise RuntimeError("Experiment 005 artifacts have an unexpected format")
    if source_005b.get("format") != "minicells.consumer-language-ablation.v1":
        raise RuntimeError("Experiment 005B artifacts have an unexpected format")
    if source_005b.get("status") != "PASS":
        raise RuntimeError("Experiment 005B must PASS before Experiment 006")
    if source_005b.get("recommended_006_variant") != "ln-c2-a0":
        raise RuntimeError(
            "Experiment 006 is locked to the 005B winner ln-c2-a0; "
            f"found {source_005b.get('recommended_006_variant')!r}"
        )
    best = source_005b.get("best") or {}
    if bool(best.get("rms_norm")) or not bool(best.get("carry_bias")) or bool(best.get("auxiliary_loss")):
        raise RuntimeError("Experiment 005B winner factors no longer match the locked 006 candidate")
    return source_005, source_005b


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


def run_workers(cache_dir: Path) -> int:
    available = torch.cuda.device_count()
    if available < 1:
        raise RuntimeError("Experiment 006 requires at least one CUDA GPU")
    used = min(2, available)
    if used == 1:
        for model in MODELS:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = "0"
            log_path = OUT / f"{model}.log"
            with log_path.open("w", encoding="utf-8") as handle:
                result = subprocess.run(
                    worker_command(model, cache_dir),
                    cwd=ROOT,
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            print(f"--- {model} / GPU 0 ---")
            print(log_path.read_text(encoding="utf-8").rstrip())
            if result.returncode != 0:
                raise RuntimeError(f"{model} worker failed with exit code {result.returncode}")
        return 1

    active: list[tuple[str, int, subprocess.Popen[str], Path]] = []
    for gpu_index, model in enumerate(MODELS):
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
        active.append((model, gpu_index, process, log_path))
        handle.close()
        print(f"started {model:14s} on physical GPU {gpu_index}")

    failures: list[str] = []
    for model, gpu_index, process, log_path in active:
        code = process.wait()
        print(f"--- {model} / GPU {gpu_index} ---")
        print(log_path.read_text(encoding="utf-8").rstrip())
        if code != 0:
            failures.append(f"{model} exited {code}; see {log_path}")
    if failures:
        raise RuntimeError("; ".join(failures))
    return 2


def save_throughput(summary: pd.DataFrame) -> None:
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.bar(summary["model"], summary["tokens_per_second"])
    axis.set_ylabel("Training tokens / second")
    axis.set_title("Experiment 006 — T4 training throughput")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "throughput.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    source_005, source_005b = require_sources()
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    print(
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu_count": torch.cuda.device_count(),
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "budget_tokens_per_model": 10_000_000,
            "train_stream_tokens": TRAIN_STREAM_TOKENS,
            "checkpoints": SCALING_CHECKPOINTS,
        }
    )

    train, validation, tokenizer_path, corpus_manifest = prepare_scaling_corpus(
        ROOT,
        source_005_dir=SOURCE_005,
    )
    cache_dir = tokenizer_path.parent
    shutil.copy2(tokenizer_path, OUT / "tokenizer.json")
    (OUT / "corpus-manifest.json").write_text(
        json.dumps(corpus_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    del train, validation

    gpu_count_used = run_workers(cache_dir)

    frames: list[pd.DataFrame] = []
    generations: list[dict[str, object]] = []
    workers: dict[str, dict[str, object]] = {}
    for model in MODELS:
        checkpoint_path = OUT / f"{model}-checkpoints.csv"
        generation_path = OUT / f"{model}-generations.json"
        worker_path = OUT / f"{model}-worker.json"
        if not checkpoint_path.is_file() or not generation_path.is_file() or not worker_path.is_file():
            raise FileNotFoundError(f"incomplete Experiment 006 worker artifacts for {model}")
        frames.append(pd.read_csv(checkpoint_path))
        generations.extend(json.loads(generation_path.read_text(encoding="utf-8")))
        workers[model] = json.loads(worker_path.read_text(encoding="utf-8"))

    checkpoints = pd.concat(frames, ignore_index=True)
    checkpoints.to_csv(OUT / "checkpoints.csv", index=False)
    summary, ratios = summarize_scaling(checkpoints)
    summary.to_csv(OUT / "model-summary.csv", index=False)
    ratios.to_csv(OUT / "relative-gap.csv", index=False)

    source_005b_ppl = float(source_005b["best"]["validation_ppl"])
    decision = make_scaling_decision(summary, ratios, source_005b_ppl=source_005b_ppl)
    transformer_match = workers["transformer-s"]["transformer_match"]
    decision["parameter_matching"] = {
        "minicells_parameters": int(summary.set_index("model").loc["minicells-v2", "parameters"]),
        "transformer_parameters": int(summary.set_index("model").loc["transformer-s", "parameters"]),
        "relative_error": float(transformer_match["relative_parameter_error"]),
        "within_5_percent": float(transformer_match["relative_parameter_error"]) <= 0.05,
    }
    decision["runtime"] = {
        "gpu_count_used": gpu_count_used,
        "physical_gpus": [torch.cuda.get_device_name(i) for i in range(gpu_count_used)],
    }
    (OUT / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    model_configs = {
        "format": "minicells.consumer-language-scaling-models.v1",
        "shared": {
            "dataset": corpus_manifest["dataset"],
            "vocab_size": corpus_manifest["vocab_size_actual"],
            "context_length": 128,
            "training_budget_tokens": 10_000_000,
            "train_stream_tokens": TRAIN_STREAM_TOKENS,
            "optimizer": "AdamW",
            "learning_rate": 3e-4,
            "warmup_steps": WARMUP_STEPS,
            "gradient_clip": 1.0,
            "schedule_seed": SCHEDULE_SEED,
        },
        "minicells-v2": {
            "source_005b_variant": "ln-c2-a0",
            "seed": MODEL_SEED,
            "dim": 128,
            "heads": 4,
            "ffn_dim": 512,
            "windows": [8, 32, 128],
            "iterations": [4, 4, 4],
            "normalization": "LayerNorm",
            "gru_carry_bias": 2.0,
            "auxiliary_stage_losses": None,
            "parameters": decision["parameter_matching"]["minicells_parameters"],
        },
        "transformer-s": {
            "seed": TRANSFORMER_SEED,
            **transformer_match,
        },
    }
    (OUT / "model-configs.json").write_text(
        json.dumps(model_configs, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    task_spec = {
        "format": "minicells.consumer-language-scaling-task.v1",
        "goal": "Measure whether MiniCells-v2 closes, preserves, or widens its gap to a parameter-matched Transformer through 10M consumed tokens.",
        "budget_tokens_per_model": 10_000_000,
        "checkpoints": list(SCALING_CHECKPOINTS),
        "train_stream_tokens": TRAIN_STREAM_TOKENS,
        "validation_stream_tokens": VALIDATION_STREAM_TOKENS,
        "same_tokenizer_as_005": True,
        "source_005_prefix_reproduced": True,
        "parallel_strategy": "one independent model process per T4 when two GPUs are available",
        "from_random_initialization": True,
    }
    (OUT / "task-spec.json").write_text(
        json.dumps(task_spec, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    (OUT / "generation-samples.json").write_text(
        json.dumps(generations, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_generation_progression(generations, OUT / "generation-progression.md")
    save_scaling_plots(checkpoints, ratios, OUT)
    save_throughput(summary)

    print("=== decision ===")
    print(json.dumps(decision, indent=2))
    print("=== model summary ===")
    print(summary.to_string(index=False))
    print("=== relative gap ===")
    print(ratios.to_string(index=False))
    print("=== files ===")
    for path in sorted(OUT.iterdir()):
        if path.is_file():
            print(path.name, path.stat().st_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
