from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from minicells.language_ablation import AblationSpec  # noqa: E402
from minicells.language_data import (  # noqa: E402
    fixed_validation_starts,
    load_tokenizer,
    make_training_schedule,
)
from minicells.language_models import TextNCALM, count_parameters  # noqa: E402
from minicells.language_training import estimate_learning_slope, train_language_model  # noqa: E402

BUDGET_TOKENS = 500_000
SEQUENCE_LENGTH = 125
BATCH_SIZE = 8
MODEL_SEED = 55005


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Experiment 005B factorial cell.")
    parser.add_argument("--name", required=True)
    parser.add_argument("--rms-norm", type=int, choices=(0, 1), required=True)
    parser.add_argument("--carry-bias", type=int, choices=(0, 1), required=True)
    parser.add_argument("--auxiliary-loss", type=int, choices=(0, 1), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec = AblationSpec(
        args.name,
        bool(args.rms_norm),
        bool(args.carry_bias),
        bool(args.auxiliary_loss),
    )
    cache_dir = args.cache_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = cache_dir / "train-tokens.pt"
    validation_path = cache_dir / "validation-tokens.pt"
    tokenizer_path = cache_dir / "tokenizer.json"
    for path in (train_path, validation_path, tokenizer_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    train_stream = torch.load(train_path, map_location="cpu")
    validation_stream = torch.load(validation_path, map_location="cpu")
    tokenizer = load_tokenizer(tokenizer_path)
    vocab_size = int(tokenizer.get_vocab_size())

    schedule = make_training_schedule(
        int(train_stream.numel()),
        seed=5005,
        budget_tokens=BUDGET_TOKENS,
        batch_size=BATCH_SIZE,
        sequence_length=SEQUENCE_LENGTH,
    )
    validation_starts = fixed_validation_starts(
        int(validation_stream.numel()),
        batches=24,
        batch_size=8,
        sequence_length=128,
    )

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        gpu = torch.cuda.get_device_name(0)
    else:
        device = torch.device("cpu")
        gpu = None

    # Every factorial cell starts from the same RNG seed. The factor switches do
    # not consume additional random draws, so matching-shaped learnable tensors
    # begin from comparable initial values.
    torch.manual_seed(MODEL_SEED)
    model = TextNCALM(
        vocab_size=vocab_size,
        rms_norm=spec.rms_norm,
        carry_bias=spec.carry_bias_value,
        tie_embeddings=True,
        stage_supervision=spec.auxiliary_loss,
    )
    parameters = count_parameters(model)

    result = train_language_model(
        name=spec.name,
        model=model,
        train_stream=train_stream,
        validation_stream=validation_stream,
        schedule=schedule,
        validation_starts=validation_starts,
        tokenizer=tokenizer,
        output_dir=output_dir,
        device=device,
        auxiliary_weights=spec.auxiliary_weights,
        seed=MODEL_SEED,
    )
    metrics = result.metrics.sort_values("consumed_tokens").reset_index(drop=True)
    metrics.to_csv(output_dir / f"{spec.name}-checkpoints.csv", index=False)
    (output_dir / f"{spec.name}-generations.json").write_text(
        json.dumps(result.generations, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    final = metrics.iloc[-1]
    summary = {
        "format": "minicells.consumer-language-ablation-cell.v1",
        "name": spec.name,
        "factors": {
            "rms_norm": spec.rms_norm,
            "carry_bias": spec.carry_bias,
            "carry_bias_value": spec.carry_bias_value,
            "auxiliary_loss": spec.auxiliary_loss,
            "auxiliary_weights": list(spec.auxiliary_weights) if spec.auxiliary_weights else None,
        },
        "parameters": parameters,
        "validation_nll": float(final["validation_nll"]),
        "validation_ppl": float(final["validation_ppl"]),
        "ppl_125k": float(metrics.loc[metrics["consumed_tokens"] == 125_000, "validation_ppl"].iloc[0]),
        "ppl_250k": float(metrics.loc[metrics["consumed_tokens"] == 250_000, "validation_ppl"].iloc[0]),
        "ppl_500k": float(final["validation_ppl"]),
        "learning_slope_alpha": float(estimate_learning_slope(metrics)),
        "elapsed_seconds": float(result.elapsed_seconds),
        "tokens_per_second": float(BUDGET_TOKENS / result.elapsed_seconds),
        "peak_vram_bytes": int(result.peak_vram_bytes),
        "checkpoint_path": str(result.checkpoint_path),
        "device": str(device),
        "gpu": gpu,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    (output_dir / f"{spec.name}-worker.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
