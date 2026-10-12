from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import pandas as pd
import torch

ROOT = Path.cwd()
if not (ROOT / "src").exists():
    ROOT = Path("/kaggle/working/mini-cells")
sys.path.insert(0, str(ROOT / "src"))

from minicells.clm_training import distillation_loss  # noqa: E402
from minicells.language_clm_validation import (  # noqa: E402
    NUM_PROGRAMS,
    QUALITY_RATIO_THRESHOLD,
    RoutingRecorder,
    configure_hard_program_stage,
    evaluate_arm,
    load_experiment_006_teacher,
    reset_program_routing_logits,
    router_diagnostics,
    static_topk_masks,
    validate_real_conversion,
)
from minicells.language_data import (  # noqa: E402
    batch_from_starts,
    fixed_validation_starts,
    make_training_schedule,
)
from minicells.textnca_to_clm import convert_textnca_to_sparse_cellular  # noqa: E402


STAGES = ((8, 250_000), (7, 250_000), (6, 375_000), (5, 500_000), (4, 500_000))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one CLM Validation 001b replicate.")
    parser.add_argument("--replicate", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    return parser.parse_args()


def program_head_grad_norm(model: torch.nn.Module) -> float:
    gradients = [
        stage.receptor.out_proj.bias.grad[1:].detach().float().reshape(-1)
        for stage in model.stages
        if stage.receptor.out_proj.bias.grad is not None
    ]
    return float(torch.cat(gradients).norm()) if gradients else 0.0


def train_stage(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    train_stream: torch.Tensor,
    *,
    top_k: int,
    budget_tokens: int,
    schedule_seed: int,
    device: torch.device,
) -> dict[str, object]:
    returned = configure_hard_program_stage(student, optimizer, top_k=top_k)
    if returned is not optimizer:
        raise RuntimeError("continuation replaced the persistent optimizer")
    schedule = make_training_schedule(
        int(train_stream.numel()), seed=schedule_seed, budget_tokens=budget_tokens,
        batch_size=8, sequence_length=125,
    )
    amp = device.type == "cuda"
    started = time.perf_counter()
    grad_norms: list[float] = []
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
            loss = distillation_loss(student_output, teacher_output, targets, beta=0.5)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norms.append(program_head_grad_norm(student))
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        final_loss = float(loss.detach())
    return {
        "top_k": top_k,
        "program_ratio": top_k / NUM_PROGRAMS,
        "budget_tokens": budget_tokens,
        "schedule_seed": schedule_seed,
        "final_loss": final_loss,
        "program_router_grad_norm_mean": sum(grad_norms) / len(grad_norms),
        "program_router_grad_norm_last": grad_norms[-1],
        "cell_ratio": 1.0,
        "learning_rate": optimizer.param_groups[0]["lr"],
        "elapsed_seconds": time.perf_counter() - started,
    }


def result_row(result: object, replicate: int) -> dict[str, object]:
    return {
        "replicate": replicate,
        "arm": result.arm,
        "top_k": result.top_k,
        "program_ratio": result.top_k / NUM_PROGRAMS,
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
        "position_variation": result.position_variation,
        "temporal_variation": result.temporal_variation,
        "program_usage": json.dumps(result.program_usage.tolist()),
        "program_coactivation": json.dumps(result.program_coactivation.tolist()),
    }


def average_shuffled(rows: list[dict[str, object]]) -> dict[str, object]:
    averaged = dict(rows[0])
    numeric = (
        "validation_nll", "dense_executor_flops", "receptor_flops",
        "active_executor_flops", "executor_ratio", "effective_compute_ratio",
        "receptor_ratio", "tokens_per_second", "milliseconds_per_batch",
        "peak_vram_bytes", "structural_variation", "position_variation",
        "temporal_variation",
    )
    for key in numeric:
        averaged[key] = sum(float(row[key]) for row in rows) / len(rows)
    averaged["validation_ppl"] = math.exp(float(averaged["validation_nll"]))
    for key in ("program_usage", "program_coactivation"):
        tensors = [torch.tensor(json.loads(str(row[key]))) for row in rows]
        averaged[key] = json.dumps(torch.stack(tensors).mean(0).tolist())
    averaged["shuffle_permutations"] = len(rows)
    return averaged


def evaluate_dense_dynamic(
    student: torch.nn.Module,
    validation_stream: torch.Tensor,
    starts: tuple[tuple[int, ...], ...],
    *,
    top_k: int,
    replicate: int,
    device: torch.device,
) -> tuple[dict[str, object], dict[str, object]]:
    dense = result_row(evaluate_arm(
        student, validation_stream, starts, sequence_length=128, device=device,
        arm="dense", top_k=top_k,
    ), replicate)
    dynamic = result_row(evaluate_arm(
        student, validation_stream, starts, sequence_length=128, device=device,
        arm="dynamic", top_k=top_k,
    ), replicate)
    return dense, dynamic


def evaluate_final_controls(
    student: torch.nn.Module,
    validation_stream: torch.Tensor,
    calibration_starts: tuple[tuple[int, ...], ...],
    formal_starts: tuple[tuple[int, ...], ...],
    *,
    top_k: int,
    replicate: int,
    device: torch.device,
) -> list[dict[str, object]]:
    student.eval()
    student.set_routing_mode("hard_program")
    student.set_program_top_k(top_k)
    calibration: list[list[torch.Tensor]] = []
    for starts in calibration_starts:
        inputs, _ = batch_from_starts(validation_stream, starts, 128, device)
        with RoutingRecorder(student) as recorder:
            student(inputs)
        calibration.append(recorder.masks)
    static_mask = static_topk_masks(calibration, top_k)
    rows = []
    for arm in ("dense", "dynamic", "static"):
        rows.append(result_row(evaluate_arm(
            student, validation_stream, formal_starts, sequence_length=128,
            device=device, arm=arm, top_k=top_k, static_mask=static_mask,
        ), replicate))
    shuffled = [
        result_row(evaluate_arm(
            student, validation_stream, formal_starts, sequence_length=128,
            device=device, arm="shuffled", top_k=top_k, static_mask=static_mask,
            permutation_seed=92001 + replicate * 100 + index,
        ), replicate)
        for index in range(3)
    ]
    rows.append(average_shuffled(shuffled))
    return rows


def save_stage_checkpoint(
    path: Path,
    *,
    student: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    replicate: int,
    top_k: int,
) -> None:
    torch.save({
        "format": "minicells.clm-validation-001b-checkpoint.v1",
        "replicate": replicate,
        "top_k": top_k,
        "model_state": student.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "rng_state": torch.random.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all(),
    }, path)


def main() -> int:
    args = parse_args()
    completed = args.output_dir / f"r{args.replicate}-worker.json"
    if completed.is_file():
        manifest = json.loads(completed.read_text(encoding="utf-8"))
        if manifest.get("complete") is True:
            print(f"replicate {args.replicate} already complete; skipping")
            return 0
    if not torch.cuda.is_available():
        raise RuntimeError("CLM Validation 001b worker requires CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    train_stream = torch.load(
        args.cache_dir / "train-tokens.pt", map_location="cpu", weights_only=True
    )
    validation_stream = torch.load(
        args.cache_dir / "validation-tokens.pt", map_location="cpu", weights_only=True
    )
    gate_starts = fixed_validation_starts(
        int(validation_stream.numel()), batches=16, batch_size=8,
        sequence_length=128, seed=5305,
    )
    calibration_starts = fixed_validation_starts(
        int(validation_stream.numel()), batches=8, batch_size=8,
        sequence_length=128, seed=5205,
    )
    formal_starts = fixed_validation_starts(
        int(validation_stream.numel()), batches=24, batch_size=8,
        sequence_length=128, seed=5405,
    )
    teacher = load_experiment_006_teacher(
        str(args.checkpoint), device=device, model_config_path=str(args.model_config)
    )
    seed = 72001 + args.replicate
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    student = convert_textnca_to_sparse_cellular(teacher, num_programs=NUM_PROGRAMS).to(device)
    parity = validate_real_conversion(
        teacher, student, validation_stream, gate_starts[:4],
        sequence_length=128, device=device,
    )
    stage0 = {"format": "minicells.clm-validation-001b-stage0.v1", **parity}
    (args.output_dir / f"r{args.replicate}-stage0.json").write_text(
        json.dumps(stage0, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if parity["status"] != "CLM_DENSE_EQUIVALENCE":
        raise RuntimeError(json.dumps(parity, indent=2))

    reset_program_routing_logits(student, seed=seed)
    max_steps = sum(budget for _, budget in STAGES) // (8 * 125)
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_steps, eta_min=1e-5
    )
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    progression: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    quality_safe_k = 8
    safe_checkpoint: Path | None = None
    for stage_index, (top_k, budget) in enumerate(STAGES):
        before = router_diagnostics(
            student, validation_stream, calibration_starts[:2],
            sequence_length=128, device=device,
        )
        training = train_stage(
            teacher, student, optimizer, scheduler, scaler, train_stream,
            top_k=top_k, budget_tokens=budget,
            schedule_seed=82001 + args.replicate * 100 + stage_index,
            device=device,
        )
        after = router_diagnostics(
            student, validation_stream, calibration_starts[:2],
            sequence_length=128, device=device,
        )
        diagnostics.extend([
            {"replicate": args.replicate, "top_k": top_k, "when": "before", **before},
            {"replicate": args.replicate, "top_k": top_k, "when": "after", **after,
             "program_router_grad_norm": training["program_router_grad_norm_mean"]},
        ])
        dense, dynamic = evaluate_dense_dynamic(
            student, validation_stream, gate_starts, top_k=top_k,
            replicate=args.replicate, device=device,
        )
        quality_ratio = float(dynamic["validation_ppl"]) / float(dense["validation_ppl"])
        passed = quality_ratio <= QUALITY_RATIO_THRESHOLD
        checkpoint_path = args.output_dir / f"r{args.replicate}-top{top_k}.pt"
        save_stage_checkpoint(
            checkpoint_path, student=student, optimizer=optimizer, scheduler=scheduler,
            scaler=scaler, replicate=args.replicate, top_k=top_k,
        )
        progression.append({
            "replicate": args.replicate, **training,
            "dense_nll": dense["validation_nll"], "dense_ppl": dense["validation_ppl"],
            "dynamic_nll": dynamic["validation_nll"],
            "dynamic_ppl": dynamic["validation_ppl"],
            "executor_ratio": dynamic["executor_ratio"],
            "effective_compute_ratio": dynamic["effective_compute_ratio"],
            "receptor_ratio": dynamic["receptor_ratio"],
            "quality_ratio": quality_ratio, "quality_pass": passed,
            "quality_safe_k": quality_safe_k if not passed else top_k,
            "selected": False,
        })
        if not passed:
            break
        quality_safe_k = top_k
        safe_checkpoint = checkpoint_path
    if safe_checkpoint is None:
        raise RuntimeError("top-8 dense-preserving warmup failed its quality gate")
    safe_payload = torch.load(safe_checkpoint, map_location=device, weights_only=True)
    student.load_state_dict(safe_payload["model_state"])
    for row in progression:
        row["quality_safe_k"] = quality_safe_k
    next(row for row in progression if int(row["top_k"]) == quality_safe_k)["selected"] = True
    arms = evaluate_final_controls(
        student, validation_stream, calibration_starts, formal_starts,
        top_k=quality_safe_k, replicate=args.replicate, device=device,
    )
    pd.DataFrame(progression).to_csv(
        args.output_dir / f"r{args.replicate}-progression.csv", index=False
    )
    pd.DataFrame(arms).to_csv(args.output_dir / f"r{args.replicate}-arms.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(
        args.output_dir / f"r{args.replicate}-router-diagnostics.csv", index=False
    )
    warmup_after = next(
        row for row in diagnostics if row["top_k"] == 8 and row["when"] == "after"
    )
    warmup_before = next(
        row for row in diagnostics if row["top_k"] == 8 and row["when"] == "before"
    )
    learned_preference = (
        float(warmup_after["program_logit_std"])
        > float(warmup_before["program_logit_std"]) + 1e-6
        or float(warmup_after["mean_cross_sample_program_logit_variance"]) > 1e-10
    )
    warmup_ok = (
        learned_preference
        and float(warmup_after["program_router_grad_norm"]) > 0
    )
    worker = {
        "format": "minicells.clm-validation-001b-worker.v1",
        "complete": True,
        "replicate": args.replicate,
        "quality_safe_k": quality_safe_k,
        "router_warmup_ok": warmup_ok,
        "receptor_seed": seed,
        "schedule_seed_base": 82001 + args.replicate * 100,
        "shuffle_seed_base": 92001 + args.replicate * 100,
        "dense_conversion": parity,
        "cell_activation": 1.0,
        "persistent_optimizer": True,
    }
    completed.write_text(json.dumps(worker, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
