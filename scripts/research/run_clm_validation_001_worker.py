from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

ROOT = Path.cwd()
if not (ROOT / "src").exists():
    ROOT = Path("/kaggle/working/mini-cells")
sys.path.insert(0, str(ROOT / "src"))

from minicells.clm_training import (  # noqa: E402
    ComputeDual,
    compute_constraint_loss,
    distillation_loss,
)
from minicells.language_clm_validation import (  # noqa: E402
    NUM_PROGRAMS,
    RoutingRecorder,
    evaluate_arm,
    load_experiment_006_teacher,
    static_topk_masks,
    validate_real_conversion,
)
from minicells.language_data import (  # noqa: E402
    batch_from_starts,
    fixed_validation_starts,
    make_training_schedule,
)
from minicells.textnca_to_clm import convert_textnca_to_sparse_cellular  # noqa: E402


PHASES = (
    ("soft-75", "soft_program", None, 0.75, 250_000),
    ("hard-6", "hard_program", 6, 0.75, 250_000),
    ("soft-50", "soft_program", None, 0.50, 500_000),
    ("hard-4", "hard_program", 4, 0.50, 500_000),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one CLM Validation 001 replicate.")
    parser.add_argument("--replicate", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    return parser.parse_args()


def train_phase(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    train_stream: torch.Tensor,
    *,
    phase_name: str,
    routing_mode: str,
    top_k: int | None,
    target: float,
    budget: int,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    student.set_routing_mode(routing_mode)
    student.set_execution_backend("masked_dense")
    student.set_program_top_k(top_k)
    schedule = make_training_schedule(
        int(train_stream.numel()), seed=seed, budget_tokens=budget,
        batch_size=8, sequence_length=125,
    )
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1
    )
    dual = ComputeDual(target=target, learning_rate=0.02)
    amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    started = time.perf_counter()
    final_loss = 0.0
    for starts in schedule.starts:
        student.train()
        inputs, targets = batch_from_starts(train_stream, starts, 125, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp
        ):
            teacher_output = teacher(inputs)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            student_output = student(inputs)
            program_ratio, cell_ratio = student.routing_ratios()
            loss = distillation_loss(student_output, teacher_output, targets, beta=0.5)
            if routing_mode == "soft_program":
                loss = loss + compute_constraint_loss(
                    program_ratio, cell_ratio,
                    program_dual=dual, cell_dual=ComputeDual(target=1.0),
                )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        dual.update(float(program_ratio.detach()))
        final_loss = float(loss.detach())
    elapsed = time.perf_counter() - started
    return {
        "phase": phase_name, "routing_mode": routing_mode, "top_k": top_k,
        "target_program_ratio": target, "budget_tokens": budget,
        "final_loss": final_loss, "final_program_ratio": float(program_ratio.detach()),
        "cell_ratio": float(cell_ratio.detach()), "dual_lambda_program": dual.value,
        "elapsed_seconds": elapsed,
    }


def result_row(result: object, replicate: int) -> dict[str, object]:
    return {
        "replicate": replicate,
        "arm": result.arm,
        "top_k": result.top_k,
        "validation_nll": result.validation_nll,
        "validation_ppl": result.validation_ppl,
        "dense_executor_flops": result.dense_executor_flops,
        "receptor_flops": result.receptor_flops,
        "active_executor_flops": result.active_executor_flops,
        "executor_ratio": result.executor_ratio,
        "effective_compute_ratio": result.effective_compute_ratio,
        "receptor_ratio": result.receptor_flops / result.dense_executor_flops,
        "tokens_per_second": result.tokens_per_second,
        "milliseconds_per_batch": result.milliseconds_per_batch,
        "peak_vram_bytes": result.peak_vram_bytes,
        "structural_variation": result.structural_variation,
        "temporal_variation": result.temporal_variation,
        "program_usage": json.dumps(result.program_usage.tolist()),
        "program_coactivation": json.dumps(result.program_coactivation.tolist()),
    }


def evaluate_checkpoint(
    student: torch.nn.Module,
    validation_stream: torch.Tensor,
    validation_starts: tuple[tuple[int, ...], ...],
    *,
    top_k: int,
    replicate: int,
    device: torch.device,
) -> list[dict[str, object]]:
    calibration: list[list[torch.Tensor]] = []
    student.eval().set_routing_mode("hard_program")
    student.set_program_top_k(top_k)
    for starts in validation_starts[:8]:
        inputs, _ = batch_from_starts(validation_stream, starts, 128, device)
        with RoutingRecorder(student) as recorder:
            student(inputs)
        calibration.append(recorder.masks)
    static_mask = static_topk_masks(calibration, top_k)
    rows = []
    for arm in ("dense", "dynamic", "static", "shuffled"):
        result = evaluate_arm(
            student, validation_stream, validation_starts, sequence_length=128,
            device=device, arm=arm, top_k=top_k, static_mask=static_mask,
            permutation_seed=91001 + replicate,
        )
        rows.append(result_row(result, replicate))
    return rows


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CLM Validation 001 worker requires CUDA")
    device = torch.device("cuda:0")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_stream = torch.load(
        args.cache_dir / "train-tokens.pt", map_location="cpu", weights_only=True
    )
    validation_stream = torch.load(
        args.cache_dir / "validation-tokens.pt", map_location="cpu", weights_only=True
    )
    validation_starts = fixed_validation_starts(
        int(validation_stream.numel()), batches=24, batch_size=8,
        sequence_length=128, seed=5105,
    )
    teacher = load_experiment_006_teacher(
        str(args.checkpoint), device=device, model_config_path=str(args.model_config)
    )
    torch.manual_seed(71001 + args.replicate)
    torch.cuda.manual_seed_all(71001 + args.replicate)
    student = convert_textnca_to_sparse_cellular(teacher, num_programs=NUM_PROGRAMS).to(device)
    parity = validate_real_conversion(
        teacher, student, validation_stream, validation_starts[:4],
        sequence_length=128, device=device,
    )
    if parity["status"] != "CLM_DENSE_EQUIVALENCE":
        raise RuntimeError(json.dumps(parity, indent=2))
    phase_rows: list[dict[str, object]] = []
    evaluation_rows: list[dict[str, object]] = []
    for phase_index, (name, mode, top_k, target, budget) in enumerate(PHASES):
        phase_rows.append(train_phase(
            teacher, student, train_stream, phase_name=name, routing_mode=mode,
            top_k=top_k, target=target, budget=budget,
            seed=81001 + args.replicate * 100 + phase_index, device=device,
        ))
        if top_k is not None:
            evaluation_rows.extend(evaluate_checkpoint(
                student, validation_stream, validation_starts,
                top_k=top_k, replicate=args.replicate, device=device,
            ))
            torch.save(
                {"format": "minicells.clm-validation-001-checkpoint.v1",
                 "replicate": args.replicate, "phase": name,
                 "model_state": student.state_dict()},
                args.output_dir / f"r{args.replicate}-{name}.pt",
            )
    pd.DataFrame(phase_rows).to_csv(
        args.output_dir / f"r{args.replicate}-phases.csv", index=False
    )
    pd.DataFrame(evaluation_rows).to_csv(
        args.output_dir / f"r{args.replicate}-arms.csv", index=False
    )
    worker = {
        "format": "minicells.clm-validation-001-worker.v1",
        "replicate": args.replicate, "receptor_seed": 71001 + args.replicate,
        "schedule_seed_base": 81001 + args.replicate * 100,
        "source_checkpoint": str(args.checkpoint), "dense_conversion": parity,
        "programs": NUM_PROGRAMS, "cell_sparsity": False,
    }
    (args.output_dir / f"r{args.replicate}-worker.json").write_text(
        json.dumps(worker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
