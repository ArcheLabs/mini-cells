from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path.cwd()
if not (ROOT / "research").exists():
    ROOT = Path("/kaggle/working/mini-cells")
if not (ROOT / "research").exists():
    raise RuntimeError("Run from a mini-cells checkout or /kaggle/working/mini-cells")
sys.path.insert(0, str(ROOT / "research"))

from minicells.language_2d import LatentTissueNCALM, build_minicells_2d  # noqa: E402
from minicells.language_data import (  # noqa: E402
    batch_from_starts,
    fixed_validation_starts,
    load_tokenizer,
    make_training_schedule,
)
from minicells.language_models import count_parameters  # noqa: E402
from minicells.language_scaling import build_minicells_v2  # noqa: E402
from minicells.language_training import train_language_model  # noqa: E402

CHECKPOINTS = (250_000, 500_000, 1_000_000, 2_000_000)
BUDGET_TOKENS = 2_000_000
SCHEDULE_SEED = 9009
BASE_SEED = 59009
WARMUP_STEPS = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Experiment 009 model on one visible GPU.")
    parser.add_argument(
        "--model",
        choices=("minicells-v2", "minicells-2d-k2", "minicells-2d-k4"),
        required=True,
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def build_model(name: str, vocab_size: int) -> tuple[torch.nn.Module, int]:
    if name == "minicells-v2":
        torch.manual_seed(BASE_SEED)
        return build_minicells_v2(vocab_size), BASE_SEED
    if name == "minicells-2d-k2":
        torch.manual_seed(BASE_SEED + 2)
        return build_minicells_2d(vocab_size, tissue_height=2), BASE_SEED + 2
    if name == "minicells-2d-k4":
        torch.manual_seed(BASE_SEED + 4)
        return build_minicells_2d(vocab_size, tissue_height=4), BASE_SEED + 4
    raise ValueError(name)


@torch.no_grad()
def evaluate_subset(
    model: LatentTissueNCALM,
    validation_stream: torch.Tensor,
    validation_starts: tuple[tuple[int, ...], ...],
    *,
    device: torch.device,
    ablate_row: int | None = None,
) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for starts in validation_starts[:12]:
        inputs, targets = batch_from_starts(validation_stream, starts, 128, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            output = model(inputs) if ablate_row is None else model.forward_with_ablation(inputs, ablate_row)
            logits = output.logits
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                reduction="sum",
            )
        total_loss += float(loss.item())
        total_tokens += int(targets.numel())
    return math.exp(min(total_loss / total_tokens, 20.0))


@torch.no_grad()
def collect_tissue_diagnostics(
    model: LatentTissueNCALM,
    validation_stream: torch.Tensor,
    validation_starts: tuple[tuple[int, ...], ...],
    *,
    device: torch.device,
) -> dict[str, object]:
    inputs, _ = batch_from_starts(validation_stream, validation_starts[0], 128, device)
    diagnostics = model.diagnose(inputs)
    baseline_subset_ppl = evaluate_subset(
        model,
        validation_stream,
        validation_starts,
        device=device,
    )
    ablations = {
        str(row): evaluate_subset(
            model,
            validation_stream,
            validation_starts,
            device=device,
            ablate_row=row,
        )
        for row in range(1, model.tissue_height)
    }
    relative = {
        row: float(ppl / baseline_subset_ppl - 1.0)
        for row, ppl in ablations.items()
    }
    return {
        "tissue_height": model.tissue_height,
        "row_cosine_to_token": list(diagnostics.row_cosine_to_token),
        "row_update_rms_flat_by_stage": [list(values) for values in diagnostics.row_update_rms],
        "matched_subset_baseline_ppl": baseline_subset_ppl,
        "latent_row_ablation_ppl": ablations,
        "latent_row_ablation_relative_ppl": relative,
    }


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Experiment 009 worker requires CUDA")
    device = torch.device("cuda:0")
    cache_dir = args.cache_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_path = cache_dir / "tokenizer.json"
    train_path = cache_dir / "train-tokens.pt"
    validation_path = cache_dir / "validation-tokens.pt"
    if not tokenizer_path.is_file() or not train_path.is_file() or not validation_path.is_file():
        raise FileNotFoundError("Experiment 009 cache is incomplete")

    tokenizer = load_tokenizer(tokenizer_path)
    train_stream = torch.load(train_path, map_location="cpu")
    validation_stream = torch.load(validation_path, map_location="cpu")
    schedule = make_training_schedule(
        int(train_stream.numel()),
        seed=SCHEDULE_SEED,
        budget_tokens=BUDGET_TOKENS,
        batch_size=8,
        sequence_length=125,
    )
    validation_starts = fixed_validation_starts(
        int(validation_stream.numel()),
        batches=48,
        batch_size=8,
        sequence_length=128,
        seed=9109,
    )

    model, seed = build_model(args.model, tokenizer.get_vocab_size())
    parameters = count_parameters(model)
    print(
        {
            "model": args.model,
            "visible_gpu": torch.cuda.get_device_name(0),
            "parameters": parameters,
            "budget_tokens": schedule.consumed_tokens,
            "checkpoint_tokens": CHECKPOINTS,
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
        checkpoint_tokens=CHECKPOINTS,
        final_checkpoint_label="2m",
    )
    metrics = result.metrics.copy()
    metrics["parameters"] = parameters
    metrics["peak_vram_bytes"] = result.peak_vram_bytes
    metrics.to_csv(output_dir / f"{args.model}-checkpoints.csv", index=False)
    (output_dir / f"{args.model}-generations.json").write_text(
        json.dumps(result.generations, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    diagnostics: dict[str, object] | None = None
    if args.model.startswith("minicells-2d"):
        model, _ = build_model(args.model, tokenizer.get_vocab_size())
        checkpoint = torch.load(result.checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        diagnostics = collect_tissue_diagnostics(
            model,
            validation_stream,
            validation_starts,
            device=device,
        )
        model.to("cpu")
        torch.cuda.empty_cache()
        (output_dir / f"{args.model}-tissue-diagnostics.json").write_text(
            json.dumps(diagnostics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    worker = {
        "format": "minicells.language-2d-worker.v1",
        "model": args.model,
        "parameters": parameters,
        "seed": seed,
        "schedule_seed": SCHEDULE_SEED,
        "warmup_steps": WARMUP_STEPS,
        "checkpoint_path": result.checkpoint_path.name,
        "elapsed_seconds": result.elapsed_seconds,
        "peak_vram_bytes": result.peak_vram_bytes,
        "tissue_diagnostics": diagnostics,
    }
    (output_dir / f"{args.model}-worker.json").write_text(
        json.dumps(worker, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
