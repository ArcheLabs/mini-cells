from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path.cwd()
if not (ROOT / "src").exists():
    ROOT = Path("/kaggle/working/mini-cells")
if not (ROOT / "src").exists():
    raise RuntimeError("Run from a mini-cells checkout or /kaggle/working/mini-cells")
sys.path.insert(0, str(ROOT / "src"))

from minicells.language_data import fixed_validation_starts, load_tokenizer, make_training_schedule  # noqa: E402
from minicells.language_models import count_parameters  # noqa: E402
from minicells.language_scaling import (  # noqa: E402
    MODEL_SEED,
    SCALING_CHECKPOINTS,
    SCHEDULE_SEED,
    TRANSFORMER_SEED,
    WARMUP_STEPS,
    build_scaling_models,
)
from minicells.language_training import train_language_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Experiment 006 model on one visible GPU.")
    parser.add_argument("--model", choices=("minicells-v2", "transformer-s"), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Experiment 006 worker requires CUDA")
    device = torch.device("cuda:0")
    cache_dir = args.cache_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_path = cache_dir / "tokenizer.json"
    train_path = cache_dir / "train-tokens.pt"
    validation_path = cache_dir / "validation-tokens.pt"
    if not tokenizer_path.is_file() or not train_path.is_file() or not validation_path.is_file():
        raise FileNotFoundError("Experiment 006 cache is incomplete")

    tokenizer = load_tokenizer(tokenizer_path)
    train_stream = torch.load(train_path, map_location="cpu")
    validation_stream = torch.load(validation_path, map_location="cpu")
    schedule = make_training_schedule(
        int(train_stream.numel()),
        seed=SCHEDULE_SEED,
        budget_tokens=10_000_000,
        batch_size=8,
        sequence_length=125,
    )
    validation_starts = fixed_validation_starts(
        int(validation_stream.numel()),
        batches=48,
        batch_size=8,
        sequence_length=128,
        seed=5105,
    )

    minicells, transformer, transformer_config = build_scaling_models(tokenizer.get_vocab_size())
    if float(transformer_config["relative_parameter_error"]) > 0.05:
        raise RuntimeError(
            "Unable to parameter-match Transformer within 5%: "
            f"{transformer_config['relative_parameter_error']:.2%}"
        )
    if args.model == "minicells-v2":
        model = minicells
        seed = MODEL_SEED
    else:
        model = transformer
        seed = TRANSFORMER_SEED
    parameters = count_parameters(model)
    del minicells, transformer

    print(
        {
            "model": args.model,
            "visible_gpu": torch.cuda.get_device_name(0),
            "parameters": parameters,
            "budget_tokens": schedule.consumed_tokens,
            "checkpoint_tokens": SCALING_CHECKPOINTS,
            "warmup_steps": WARMUP_STEPS,
        }
    )

    result = train_language_model(
        name=args.model,
        model=model,
        train_stream=train_stream,
        validation_stream=validation_stream,
        schedule=schedule,
        validation_starts=validation_starts,
        tokenizer=tokenizer,
        output_dir=output_dir,
        device=device,
        auxiliary_weights=None,
        seed=seed,
        base_lr=3e-4,
        weight_decay=0.1,
        warmup_steps=WARMUP_STEPS,
        checkpoint_tokens=SCALING_CHECKPOINTS,
        final_checkpoint_label="10m",
    )
    metrics = result.metrics.copy()
    metrics["parameters"] = parameters
    metrics["peak_vram_bytes"] = result.peak_vram_bytes
    metrics.to_csv(output_dir / f"{args.model}-checkpoints.csv", index=False)
    (output_dir / f"{args.model}-generations.json").write_text(
        json.dumps(result.generations, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    worker = {
        "format": "minicells.consumer-language-scaling-worker.v1",
        "model": args.model,
        "parameters": parameters,
        "seed": seed,
        "schedule_seed": SCHEDULE_SEED,
        "warmup_steps": WARMUP_STEPS,
        "checkpoint_path": result.checkpoint_path.name,
        "transformer_match": transformer_config,
        "elapsed_seconds": result.elapsed_seconds,
        "peak_vram_bytes": result.peak_vram_bytes,
    }
    (output_dir / f"{args.model}-worker.json").write_text(
        json.dumps(worker, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
