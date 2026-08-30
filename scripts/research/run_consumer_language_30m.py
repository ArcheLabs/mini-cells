from __future__ import annotations

import argparse
import hashlib
import json
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
if not (ROOT / "src").exists():
    ROOT = Path("/kaggle/working/mini-cells")
if not (ROOT / "src").exists():
    raise RuntimeError("Run from a mini-cells checkout or /kaggle/working/mini-cells")
sys.path.insert(0, str(ROOT / "src"))

from minicells.language_30m import (  # noqa: E402
    BASE_LR,
    BATCH_SIZE,
    CHECKPOINT_TOKENS,
    CONTEXT_LENGTH,
    MINICELLS_CARRY_BIAS,
    MINICELLS_DIM,
    MINICELLS_FFN,
    MINICELLS_HEADS,
    MINICELLS_ITERATIONS,
    MINICELLS_WINDOWS,
    MODEL_NAME,
    SCHEDULE_SEED,
    TARGET_TOKENS,
    TRAIN_SEQUENCE_LENGTH,
    TRAIN_STREAM_TOKENS,
    TRANSFORMER_NAME,
    VALIDATION_STREAM_TOKENS,
    WARMUP_STEPS,
    WEIGHT_DECAY,
    make_30m_decision,
    model_parameter_summary,
    prepare_30m_corpus,
    save_30m_plots,
    summarize_30m,
    write_generation_progression,
)

OUT = ROOT / "results" / "consumer-language-30m-v1"
SOURCE_006 = ROOT / "artifacts" / "experiments" / "006-consumer-language-scaling"
WORKER = ROOT / "scripts" / "run_consumer_language_30m_variant.py"
MODELS = (MODEL_NAME, TRANSFORMER_NAME)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Experiment 007 — MiniCells-30M v0.")
    parser.add_argument(
        "--stop-after-tokens",
        type=int,
        default=int(os.environ.get("MINICELLS_30M_STOP_AFTER", TARGET_TOKENS)),
        help="Stop both models at this consumed-token point while keeping the 100M LR schedule.",
    )
    parser.add_argument(
        "--resume-input",
        type=Path,
        default=Path(os.environ["MINICELLS_30M_RESUME_INPUT"])
        if os.environ.get("MINICELLS_30M_RESUME_INPUT")
        else None,
        help="Optional directory containing prior *-latest.pt resume checkpoints.",
    )
    parser.add_argument(
        "--reset-training",
        action="store_true",
        help="Delete worker/resume outputs but preserve the expensive corpus cache.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_source_006() -> dict[str, object]:
    decision_path = SOURCE_006 / "decision.json"
    if not decision_path.is_file():
        raise FileNotFoundError("Experiment 006 results must be merged before Experiment 007")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("format") != "minicells.consumer-language-scaling.v1":
        raise RuntimeError("unexpected Experiment 006 format")
    if decision.get("status") != "GREEN":
        raise RuntimeError("Experiment 006 must be GREEN before Experiment 007")
    return decision


def reset_training_outputs() -> None:
    if not OUT.exists():
        return
    for path in OUT.iterdir():
        if path.name == "cache":
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def import_resume_checkpoints(source: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    destination = OUT / "resume"
    destination.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        candidates = [
            source / f"{model}-latest.pt",
            source / "resume" / f"{model}-latest.pt",
            source / "consumer-language-30m-v1" / "resume" / f"{model}-latest.pt",
        ]
        match = next((path for path in candidates if path.is_file()), None)
        if match is None:
            raise FileNotFoundError(
                f"resume input does not contain {model}-latest.pt in a supported layout"
            )
        target = destination / f"{model}-latest.pt"
        if match.resolve() != target.resolve():
            shutil.copy2(match, target)
            print(f"imported {model} resume checkpoint from {match}")


def worker_command(model: str, cache_dir: Path, stop_after_tokens: int) -> list[str]:
    return [
        sys.executable,
        str(WORKER),
        "--model",
        model,
        "--cache-dir",
        str(cache_dir),
        "--output-dir",
        str(OUT),
        "--stop-after-tokens",
        str(stop_after_tokens),
    ]


def run_workers(cache_dir: Path, stop_after_tokens: int) -> int:
    available = torch.cuda.device_count()
    if available < 1:
        raise RuntimeError("Experiment 007 requires at least one CUDA GPU")
    used = min(2, available)
    if used == 1:
        for model in MODELS:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = "0"
            log_path = OUT / f"{model}.log"
            with log_path.open("w", encoding="utf-8") as handle:
                result = subprocess.run(
                    worker_command(model, cache_dir, stop_after_tokens),
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
            worker_command(model, cache_dir, stop_after_tokens),
            cwd=ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        active.append((model, gpu_index, process, log_path))
        handle.close()
        print(f"started {model:18s} on physical GPU {gpu_index}")

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


def read_worker_summaries() -> dict[str, dict[str, object]]:
    summaries: dict[str, dict[str, object]] = {}
    for model in MODELS:
        path = OUT / f"{model}-worker.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        summaries[model] = json.loads(path.read_text(encoding="utf-8"))
    return summaries


def save_throughput(summary: pd.DataFrame) -> None:
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.bar(summary["model"], summary["tokens_per_second"])
    axis.set_ylabel("Training tokens / second")
    axis.set_title("Experiment 007 — T4 training throughput")
    axis.tick_params(axis="x", rotation=10)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "throughput.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_model_card(
    decision: dict[str, object],
    *,
    model_artifact: Path,
    tokenizer_sha: str,
) -> None:
    candidate = decision["candidate"]
    comparison = decision["comparison"]
    text = f"""# MiniCells-30M v0

MiniCells-30M v0 is the first retained ~30M-parameter MiniCells language-model artifact.

## Architecture

- Parameters: {candidate['parameters']:,}
- Hidden dimension: {MINICELLS_DIM}
- Heads: {MINICELLS_HEADS}
- FFN dimension: {MINICELLS_FFN}
- Hierarchical causal windows: {list(MINICELLS_WINDOWS)}
- Recurrent iterations: {list(MINICELLS_ITERATIONS)}
- GRU carry bias: {MINICELLS_CARRY_BIAS}
- Context length: {CONTEXT_LENGTH}
- Tokenizer vocabulary: 2,048

## Training

- Dataset: TinyStories
- Consumed training tokens: {TARGET_TOKENS:,}
- Optimizer: AdamW
- Base learning rate: {BASE_LR}
- Warmup steps: {WARMUP_STEPS:,}
- Weight decay: {WEIGHT_DECAY}
- Tokenizer SHA-256: `{tokenizer_sha}`

## Result

- Status: `{decision['status']}`
- Diagnosis: `{decision['diagnosis']}`
- MiniCells PPL @100M: {candidate['ppl_100m']:.4f}
- Transformer PPL @100M: {decision['transformer']['ppl_100m']:.4f}
- PPL ratio @100M: {comparison['ppl_ratio_100m']:.4f}x
- Learning-slope ratio: {comparison['slope_ratio_to_transformer']:.4f}

## Artifact

- File: `{model_artifact.name}`
- Bytes: {model_artifact.stat().st_size:,}
- SHA-256: `{sha256_file(model_artifact)}`

The retained artifact stores FP16 weights plus the architecture configuration. It is intended
for inference and future model work; optimizer/resume checkpoints are intentionally not
published to Git.

## Scope

This model was trained only on TinyStories. It can generate simple story-like English text,
but it is not an instruction-following assistant and should not be described as a general
knowledge, reasoning, coding, or chat model.
"""
    (OUT / "MODEL_CARD.md").write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.stop_after_tokens <= 0 or args.stop_after_tokens > TARGET_TOKENS:
        raise ValueError("--stop-after-tokens must be in (0, 100M]")
    if args.stop_after_tokens % 1_000:
        raise ValueError("--stop-after-tokens must be divisible by 1,000")

    source_006 = require_source_006()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.reset_training:
        reset_training_outputs()
        OUT.mkdir(parents=True, exist_ok=True)

    print(
        {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu_count": torch.cuda.device_count(),
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "target_tokens_per_model": TARGET_TOKENS,
            "stop_after_tokens": args.stop_after_tokens,
            "train_stream_tokens": TRAIN_STREAM_TOKENS,
            "checkpoints": CHECKPOINT_TOKENS,
        }
    )
    params = model_parameter_summary(2048)
    print("model parameters", params)
    if float(params["relative_parameter_error"]) > 0.01:
        raise RuntimeError("Experiment 007 Transformer must parameter-match MiniCells within 1%")

    corpus = prepare_30m_corpus(ROOT, source_006_dir=SOURCE_006)
    cache_dir = corpus.tokenizer_path.parent
    shutil.copy2(corpus.tokenizer_path, OUT / "tokenizer.json")
    (OUT / "corpus-manifest.json").write_text(
        json.dumps(corpus.manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if args.resume_input is not None:
        import_resume_checkpoints(args.resume_input)

    gpu_count_used = run_workers(cache_dir, args.stop_after_tokens)
    workers = read_worker_summaries()
    consumed = {model: int(workers[model]["consumed_tokens"]) for model in MODELS}
    complete = all(bool(workers[model]["complete"]) for model in MODELS)

    progress = {
        "format": "minicells.language-30m-progress.v1",
        "target_tokens": TARGET_TOKENS,
        "requested_stop_after_tokens": args.stop_after_tokens,
        "consumed_tokens": consumed,
        "complete": complete,
        "resume_dir": str(OUT / "resume"),
    }
    (OUT / "progress.json").write_text(
        json.dumps(progress, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if not complete:
        print("=== Experiment 007 partial progress ===")
        print(json.dumps(progress, indent=2))
        print(
            "Training is resumable. Preserve the two files under results/consumer-language-30m-v1/"
            "resume (for example with a Kaggle saved output/dataset), then provide that directory "
            "through --resume-input or MINICELLS_30M_RESUME_INPUT on the next session."
        )
        return 0

    frames = [pd.read_csv(OUT / f"{model}-checkpoints.csv") for model in MODELS]
    checkpoints = pd.concat(frames, ignore_index=True)
    checkpoints.to_csv(OUT / "checkpoints.csv", index=False)

    generations: list[dict[str, object]] = []
    for model in MODELS:
        generations.extend(
            json.loads((OUT / f"{model}-generations.json").read_text(encoding="utf-8"))
        )
    (OUT / "generation-samples.json").write_text(
        json.dumps(generations, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_generation_progression(generations, OUT / "generation-progression.md")

    summary, ratios = summarize_30m(checkpoints)
    summary.to_csv(OUT / "model-summary.csv", index=False)
    ratios.to_csv(OUT / "relative-gap.csv", index=False)
    source_ratio = float(source_006["comparison"]["ppl_ratio_10m"])
    decision = make_30m_decision(summary, ratios, source_006_ratio_10m=source_ratio)
    transformer_match = workers[TRANSFORMER_NAME]["transformer_match"]
    decision["parameter_matching"] = {
        "minicells_parameters": int(params[MODEL_NAME]),
        "transformer_parameters": int(params[TRANSFORMER_NAME]),
        "relative_error": float(params["relative_parameter_error"]),
        "within_1_percent": float(params["relative_parameter_error"]) <= 0.01,
        "transformer_config": transformer_match,
    }
    decision["runtime"] = {
        "gpu_count_used": gpu_count_used,
        "physical_gpus": [torch.cuda.get_device_name(i) for i in range(gpu_count_used)],
    }

    model_artifact = OUT / "minicells-30m-v0-fp16.pt"
    if not model_artifact.is_file():
        raise FileNotFoundError(model_artifact)
    decision["retained_model"] = {
        "path": model_artifact.name,
        "bytes": model_artifact.stat().st_size,
        "sha256": sha256_file(model_artifact),
        "precision": "fp16",
        "optimizer_state_published": False,
    }
    (OUT / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    model_configs = {
        "format": "minicells.language-30m-models.v1",
        "shared": {
            "dataset": corpus.manifest["dataset"],
            "vocab_size": corpus.manifest["vocab_size_actual"],
            "context_length": CONTEXT_LENGTH,
            "training_budget_tokens": TARGET_TOKENS,
            "train_stream_tokens": TRAIN_STREAM_TOKENS,
            "optimizer": "AdamW",
            "learning_rate": BASE_LR,
            "warmup_steps": WARMUP_STEPS,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "sequence_length": TRAIN_SEQUENCE_LENGTH,
            "schedule_seed": SCHEDULE_SEED,
        },
        MODEL_NAME: workers[MODEL_NAME]["model_config"],
        TRANSFORMER_NAME: {
            **workers[TRANSFORMER_NAME]["model_config"],
            **(transformer_match or {}),
        },
    }
    (OUT / "model-configs.json").write_text(
        json.dumps(model_configs, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    task_spec = {
        "format": "minicells.language-30m-task.v1",
        "goal": "Test parameter scaling at ~30M parameters and retain the first MiniCells-30M model artifact.",
        "tokens_per_model": TARGET_TOKENS,
        "checkpoints": list(CHECKPOINT_TOKENS),
        "train_stream_tokens": TRAIN_STREAM_TOKENS,
        "validation_stream_tokens": VALIDATION_STREAM_TOKENS,
        "same_tokenizer_as_006": True,
        "source_006_prefix_reproduced": True,
        "from_random_initialization": True,
        "parallel_strategy": "one independent model process per T4 when two GPUs are available",
        "resume_interval_tokens": 5_000_000,
        "resume_is_exact": True,
        "published_model_precision": "fp16",
    }
    (OUT / "task-spec.json").write_text(
        json.dumps(task_spec, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    save_30m_plots(checkpoints, ratios, OUT)
    save_throughput(summary)
    write_model_card(
        decision,
        model_artifact=model_artifact,
        tokenizer_sha=str(corpus.manifest["tokenizer_sha256"]),
    )

    print("=== decision ===")
    print(json.dumps(decision, indent=2))
    print("=== model summary ===")
    print(summary.to_string(index=False))
    print("=== relative gap ===")
    print(ratios.to_string(index=False))
    print("=== retained model ===")
    print(model_artifact, model_artifact.stat().st_size, sha256_file(model_artifact))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
