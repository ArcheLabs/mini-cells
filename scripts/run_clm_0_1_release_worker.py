from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path.cwd()
if not (ROOT / "research").exists():
    ROOT = Path("/kaggle/working/mini-cells")
sys.path.insert(0, str(ROOT / "research"))

from minicells.clm_conditionality_002 import aligned_route_disagreement  # noqa: E402
from minicells.clm_upcycling_validation import (  # noqa: E402
    UpcyclingRoutingRecorder,
    collect_stage_perceptions,
    evaluate_upcycled,
    geometry_prototypes,
    static_templates,
    validate_upcycled_parity,
)
from minicells.language_clm_validation import load_experiment_006_teacher  # noqa: E402
from minicells.language_data import (  # noqa: E402
    batch_from_starts,
    fixed_validation_starts,
    make_training_schedule,
)
from minicells.upcycled_cellular_textnca import (  # noqa: E402
    UpcyclingConfig,
    convert_textnca_to_upcycled,
)

TRAINING_BLOCK_TOKENS = 250_000
TRAINING_BLOCKS = 4
EXPECTED_GEOMETRY_PPL = {
    0: 17.99729567546806,
    1: 17.97998454539842,
    2: 17.968933276012226,
}
REPRODUCTION_PPL_ATOL = 0.05
BACKEND_PPL_ATOL = 1e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce one CLM-0.1 geometry candidate.")
    parser.add_argument("--replicate", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    return parser.parse_args()


@torch.no_grad()
def evaluate_lm(model, stream, starts_batches, device) -> dict[str, float]:
    model.eval()
    total = 0.0
    tokens = 0
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for starts in starts_batches:
        inputs, targets = batch_from_starts(stream, starts, 128, device)
        output = model(inputs)
        total += float(
            F.cross_entropy(
                output.logits.flatten(0, 1),
                targets.flatten(),
                reduction="sum",
            )
        )
        tokens += targets.numel()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    nll = total / tokens
    return {
        "nll": nll,
        "ppl": math.exp(min(nll, 20)),
        "tokens_per_second": tokens / elapsed,
        "peak_vram_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
    }


def train_block(model, teacher, stream, starts_batches, optimizer, scheduler, scaler, device) -> None:
    amp = device.type == "cuda"
    for starts in starts_batches:
        model.train()
        inputs, targets = batch_from_starts(stream, starts, 125, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp,
        ):
            teacher_logits = teacher(inputs).logits.detach()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            output, stats = model(inputs, return_stats=True)
            logits = output.logits
            ce = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
            kl = F.kl_div(
                logits.reshape(-1, logits.shape[-1]).log_softmax(-1),
                teacher_logits.reshape(-1, teacher_logits.shape[-1]).softmax(-1),
                reduction="batchmean",
            )
            loss = ce + 0.5 * kl + 0.01 * stats.balance_loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()


def optimizer_for(model, steps: int):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=steps,
        eta_min=1e-5,
    )
    return optimizer, scheduler


def collect_masks_and_aligned(model, validation, starts_batches, device) -> float:
    values = []
    for starts in starts_batches:
        inputs, _ = batch_from_starts(validation, starts, 128, device)
        with UpcyclingRoutingRecorder(model) as recorder:
            model(inputs)
        values.append(aligned_route_disagreement(recorder.masks))
    return sum(values) / len(values)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / f"r{args.replicate}-release-worker.json"
    if manifest_path.is_file() and json.loads(manifest_path.read_text()).get("complete"):
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("CLM-0.1 release reproduction requires CUDA")

    device = torch.device("cuda:0")
    train = torch.load(
        args.cache_dir / "train-tokens.pt",
        map_location="cpu",
        weights_only=True,
    )
    validation = torch.load(
        args.cache_dir / "validation-tokens.pt",
        map_location="cpu",
        weights_only=True,
    )
    gate_starts = fixed_validation_starts(
        validation.numel(), batches=12, batch_size=8, sequence_length=128, seed=7105
    )
    geometry_starts = fixed_validation_starts(
        validation.numel(), batches=8, batch_size=8, sequence_length=128, seed=7205
    )
    calibration_starts = fixed_validation_starts(
        validation.numel(), batches=8, batch_size=8, sequence_length=128, seed=7305
    )
    formal_starts = fixed_validation_starts(
        validation.numel(), batches=24, batch_size=8, sequence_length=128, seed=7405
    )
    teacher = load_experiment_006_teacher(
        str(args.checkpoint),
        device=device,
        model_config_path=str(args.model_config),
    )
    teacher.eval().requires_grad_(False)

    schedules = [
        make_training_schedule(
            int(train.numel()),
            seed=121001 + args.replicate * 100 + block,
            budget_tokens=TRAINING_BLOCK_TOKENS,
            batch_size=8,
            sequence_length=125,
        ).starts
        for block in range(TRAINING_BLOCKS)
    ]
    total_steps = sum(len(row) for row in schedules)

    # Matched dense continuation. Keep only a CPU inference copy after training so
    # optimizer state and the frozen teacher cannot contaminate benchmark VRAM.
    dense = copy.deepcopy(teacher).to(device).requires_grad_(True)
    dense_opt, dense_sched = optimizer_for(dense, total_steps)
    dense_scaler = torch.amp.GradScaler("cuda", enabled=True)
    for starts in schedules:
        for batch_starts in starts:
            dense.train()
            inputs, targets = batch_from_starts(train, batch_starts, 125, device)
            dense_opt.zero_grad(set_to_none=True)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                teacher_logits = teacher(inputs).logits.detach()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = dense(inputs).logits
                ce = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
                kl = F.kl_div(
                    logits.reshape(-1, logits.shape[-1]).log_softmax(-1),
                    teacher_logits.reshape(-1, teacher_logits.shape[-1]).softmax(-1),
                    reduction="batchmean",
                )
                loss = ce + 0.5 * kl
            dense_scaler.scale(loss).backward()
            dense_scaler.unscale_(dense_opt)
            torch.nn.utils.clip_grad_norm_(dense.parameters(), 1.0)
            dense_scaler.step(dense_opt)
            dense_scaler.update()
            dense_sched.step()
    dense_cpu = copy.deepcopy(dense).cpu().eval()
    del dense, dense_opt, dense_sched, dense_scaler
    torch.cuda.empty_cache()

    # Geometry initialization is extracted only from frozen-teacher local perceptions.
    stage_samples = collect_stage_perceptions(
        teacher,
        validation,
        geometry_starts,
        sequence_length=128,
        device=device,
        max_samples_per_stage=8192,
        seed=122001 + args.replicate,
    )
    prototypes, geometry_diag = geometry_prototypes(
        stage_samples,
        4,
        seed=123001 + args.replicate * 100,
    )
    torch.manual_seed(131001 + args.replicate * 100 + 1)
    torch.cuda.manual_seed_all(131001 + args.replicate * 100 + 1)
    model = convert_textnca_to_upcycled(
        teacher,
        config=UpcyclingConfig(num_experts=4, top_k=1, router_scale=4.0),
    ).to(device)
    model.set_router_prototypes(prototypes)
    parity = validate_upcycled_parity(
        teacher,
        model,
        validation,
        gate_starts[:4],
        sequence_length=128,
        device=device,
    )
    if parity["status"] != "CLM_UPCYCLING_EQUIVALENCE":
        raise RuntimeError(
            json.dumps({"replicate": args.replicate, "parity": parity}, indent=2)
        )

    optimizer, scheduler = optimizer_for(model, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    for starts in schedules:
        model.set_execution_backend("masked_dense")
        train_block(model, teacher, train, starts, optimizer, scheduler, scaler, device)

    model.set_execution_backend("sparse_dispatch")
    dynamic = evaluate_upcycled(
        model,
        validation,
        formal_starts,
        sequence_length=128,
        device=device,
        arm="dynamic",
        templates=[],
    )
    reproduced = (
        abs(dynamic.ppl - EXPECTED_GEOMETRY_PPL[args.replicate])
        <= REPRODUCTION_PPL_ATOL
    )
    if not reproduced:
        raise RuntimeError(
            f"release reproduction drift: observed PPL={dynamic.ppl:.6f}, "
            f"expected={EXPECTED_GEOMETRY_PPL[args.replicate]:.6f}"
        )

    calibration = []
    for starts in calibration_starts:
        inputs, _ = batch_from_starts(validation, starts, 128, device)
        with UpcyclingRoutingRecorder(model) as recorder:
            model(inputs)
        calibration.append(recorder.masks)
    templates = static_templates(calibration)
    static = evaluate_upcycled(
        model,
        validation,
        formal_starts,
        sequence_length=128,
        device=device,
        arm="static",
        templates=templates,
    )
    shuffled_runs = [
        evaluate_upcycled(
            model,
            validation,
            formal_starts,
            sequence_length=128,
            device=device,
            arm="shuffled",
            templates=templates,
            permutation_seed=143001 + args.replicate * 100 + index,
        )
        for index in range(3)
    ]
    shuffled_nll = sum(row.nll for row in shuffled_runs) / len(shuffled_runs)
    shuffled_ppl = math.exp(shuffled_nll)
    aligned = collect_masks_and_aligned(model, validation, formal_starts, device)

    # Training and control evaluation are complete. Strip training-only allocations
    # before measuring inference memory/throughput.
    del optimizer, scheduler, scaler, teacher
    torch.cuda.empty_cache()

    dense_benchmark_model = dense_cpu.to(device)
    dense_metrics = evaluate_lm(
        dense_benchmark_model,
        validation,
        formal_starts,
        device,
    )
    dense_cpu = dense_benchmark_model.cpu()
    del dense_benchmark_model
    torch.cuda.empty_cache()

    model.set_execution_backend("masked_dense")
    masked_metrics = evaluate_lm(model, validation, formal_starts, device)
    model.set_execution_backend("sparse_dispatch")
    sparse_metrics = evaluate_lm(model, validation, formal_starts, device)
    if abs(masked_metrics["ppl"] - sparse_metrics["ppl"]) > BACKEND_PPL_ATOL:
        raise RuntimeError(
            "masked_dense/sparse_dispatch release parity failed: "
            f"{masked_metrics['ppl']} vs {sparse_metrics['ppl']}"
        )

    torch.save(
        {
            "format": "minicells.clm-0.1-release-candidate.v1",
            "replicate": args.replicate,
            "model_state": model.state_dict(),
            "provenance": model.provenance,
        },
        args.output_dir / f"r{args.replicate}-geometry-release.pt",
    )
    manifest = {
        "format": "minicells.clm-0.1-release-worker.v1",
        "complete": True,
        "replicate": args.replicate,
        "initial_equivalence": parity,
        "expected_geometry_ppl": EXPECTED_GEOMETRY_PPL[args.replicate],
        "observed_geometry_ppl": dynamic.ppl,
        "reproduction_ppl_atol": REPRODUCTION_PPL_ATOL,
        "reproduction_pass": reproduced,
        "dense_nll": dense_metrics["nll"],
        "dense_ppl": dense_metrics["ppl"],
        "dynamic": {
            "nll": dynamic.nll,
            "ppl": dynamic.ppl,
            "usage_entropy": dynamic.usage_entropy,
        },
        "static": {"nll": static.nll, "ppl": static.ppl},
        "shuffled": {"nll": shuffled_nll, "ppl": shuffled_ppl},
        "aligned_route_disagreement": aligned,
        "geometry_init": geometry_diag,
        "backend_parity": {
            "ppl_atol": BACKEND_PPL_ATOL,
            "masked_dense_ppl": masked_metrics["ppl"],
            "sparse_dispatch_ppl": sparse_metrics["ppl"],
            "passed": True,
        },
        "benchmark": {
            "dense": dense_metrics,
            "clm_sparse_dispatch": sparse_metrics,
            "clm_masked_dense": masked_metrics,
        },
        "training_tokens": TRAINING_BLOCK_TOKENS * TRAINING_BLOCKS,
        "same_recipe_as_upcycling_study_001": True,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
