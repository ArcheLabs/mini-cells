from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from torch.nn import functional as F

ROOT = Path.cwd()
if not (ROOT / "src").exists():
    ROOT = Path("/kaggle/working/mini-cells")
sys.path.insert(0, str(ROOT / "src"))

from minicells.clm_upcycling_validation import (  # noqa: E402
    UpcyclingRoutingRecorder,
    collect_stage_perceptions,
    evaluate_upcycled,
    evaluation_dict,
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
DISTILLATION_WEIGHT = 0.5
BALANCE_WEIGHT = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one CLM Upcycling Study 001 replicate.")
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
    for starts in starts_batches:
        inputs, targets = batch_from_starts(stream, starts, 128, device)
        output = model(inputs)
        total += float(
            F.cross_entropy(output.logits.flatten(0, 1), targets.flatten(), reduction="sum")
        )
        tokens += targets.numel()
    nll = total / tokens
    return {"nll": nll, "ppl": math.exp(min(nll, 20))}


@torch.no_grad()
def evaluate_upcycled_current(model, stream, starts_batches, device) -> dict[str, object]:
    model.eval()
    model.set_execution_backend("sparse_dispatch")
    total = 0.0
    tokens = 0
    usages = []
    entropies = []
    variances = []
    for starts in starts_batches:
        inputs, targets = batch_from_starts(stream, starts, 128, device)
        output, stats = model(inputs, return_stats=True)
        total += float(
            F.cross_entropy(output.logits.flatten(0, 1), targets.flatten(), reduction="sum")
        )
        tokens += targets.numel()
        usages.append(stats.program_usage.detach().cpu())
        entropies.append(float(stats.usage_entropy))
        variances.append(float(stats.router_logit_variance))
    nll = total / tokens
    return {
        "nll": nll,
        "ppl": math.exp(min(nll, 20)),
        "usage_entropy": sum(entropies) / len(entropies),
        "router_logit_variance": sum(variances) / len(variances),
        "program_usage": torch.stack(usages).mean(0).tolist(),
    }


def train_block(
    model,
    teacher,
    stream,
    starts_batches,
    optimizer,
    scheduler,
    scaler,
    *,
    device,
    upcycled: bool,
) -> dict[str, float]:
    amp = device.type == "cuda"
    losses = []
    ce_values = []
    kl_values = []
    balance_values = []
    started = time.perf_counter()
    for starts in starts_batches:
        model.train()
        inputs, targets = batch_from_starts(stream, starts, 125, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp
        ):
            teacher_output = teacher(inputs)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            if upcycled:
                output, stats = model(inputs, return_stats=True)
            else:
                output = model(inputs)
                stats = None
            student_logits = output.logits.reshape(-1, output.logits.shape[-1])
            teacher_logits = teacher_output.logits.detach().reshape(-1, teacher_output.logits.shape[-1])
            ce = F.cross_entropy(student_logits, targets.reshape(-1))
            kl = F.kl_div(
                student_logits.log_softmax(-1), teacher_logits.softmax(-1), reduction="batchmean"
            )
            balance = stats.balance_loss if stats is not None else ce.new_zeros(())
            loss = ce + DISTILLATION_WEIGHT * kl + BALANCE_WEIGHT * balance
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        losses.append(float(loss.detach()))
        ce_values.append(float(ce.detach()))
        kl_values.append(float(kl.detach()))
        balance_values.append(float(balance.detach()))
    return {
        "loss": losses[-1],
        "ce": ce_values[-1],
        "kl": kl_values[-1],
        "balance": balance_values[-1],
        "elapsed_seconds": time.perf_counter() - started,
    }


def optimizer_for(model, steps: int):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=1e-5
    )
    return optimizer, scheduler


def expert_diversity(model) -> dict[str, float]:
    cosine = []
    relative_l2 = []
    with torch.no_grad():
        for stage in model.stages:
            vectors = []
            for expert in stage.program_bank.experts:
                vectors.append(torch.cat([p.detach().float().reshape(-1) for p in expert.parameters()]))
            for left in range(len(vectors)):
                for right in range(left + 1, len(vectors)):
                    a = vectors[left]
                    b = vectors[right]
                    cosine.append(float(F.cosine_similarity(a, b, dim=0)))
                    relative_l2.append(float((a - b).norm() / a.norm().clamp_min(1e-8)))
    return {
        "expert_pairwise_cosine": sum(cosine) / len(cosine),
        "expert_pairwise_relative_l2": sum(relative_l2) / len(relative_l2),
    }


def final_controls(model, validation, calibration_starts, formal_starts, method, replicate, device):
    model.set_execution_backend("sparse_dispatch")
    calibration = []
    for starts in calibration_starts:
        inputs, _ = batch_from_starts(validation, starts, 128, device)
        with UpcyclingRoutingRecorder(model) as recorder:
            model(inputs)
        calibration.append(recorder.masks)
    templates = static_templates(calibration)
    dynamic = evaluate_upcycled(
        model, validation, formal_starts, sequence_length=128, device=device,
        arm="dynamic", templates=templates,
    )
    static = evaluate_upcycled(
        model, validation, formal_starts, sequence_length=128, device=device,
        arm="static", templates=templates,
    )
    shuffled_runs = [
        evaluate_upcycled(
            model, validation, formal_starts, sequence_length=128, device=device,
            arm="shuffled", templates=templates,
            permutation_seed=143001 + replicate * 100 + index,
        )
        for index in range(3)
    ]
    rows = [
        {"replicate": replicate, "method": method, **evaluation_dict(dynamic)},
        {"replicate": replicate, "method": method, **evaluation_dict(static)},
    ]
    shuffled_dicts = [evaluation_dict(item) for item in shuffled_runs]
    average = dict(shuffled_dicts[0])
    for key in (
        "nll", "sample_variation", "position_variation", "temporal_variation",
        "usage_entropy", "router_logit_variance", "tokens_per_second",
    ):
        average[key] = sum(float(row[key]) for row in shuffled_dicts) / len(shuffled_dicts)
    average["ppl"] = math.exp(float(average["nll"]))
    for key in ("program_usage", "program_coactivation"):
        average[key] = torch.tensor([row[key] for row in shuffled_dicts]).mean(0).tolist()
    rows.append({
        "replicate": replicate,
        "method": method,
        **average,
        "shuffle_permutations": len(shuffled_runs),
    })
    return rows


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / f"r{args.replicate}-worker.json"
    if manifest_path.is_file() and json.loads(manifest_path.read_text()).get("complete"):
        print(f"replicate {args.replicate} already complete; skipping")
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("CLM Upcycling Study 001 requires CUDA")
    device = torch.device("cuda:0")
    train = torch.load(args.cache_dir / "train-tokens.pt", map_location="cpu", weights_only=True)
    validation = torch.load(
        args.cache_dir / "validation-tokens.pt", map_location="cpu", weights_only=True
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
        str(args.checkpoint), device=device, model_config_path=str(args.model_config)
    )
    teacher.eval().requires_grad_(False)
    source_metrics = evaluate_lm(teacher, validation, formal_starts, device)

    schedules = [
        make_training_schedule(
            int(train.numel()), seed=121001 + args.replicate * 100 + block,
            budget_tokens=TRAINING_BLOCK_TOKENS, batch_size=8, sequence_length=125,
        ).starts
        for block in range(TRAINING_BLOCKS)
    ]
    total_steps = sum(len(starts) for starts in schedules)
    progression: list[dict[str, object]] = []
    controls: list[dict[str, object]] = []

    # Dense continuation control: identical source, identical data schedule and optimizer budget.
    dense = copy.deepcopy(teacher).to(device)
    dense.requires_grad_(True)
    dense_optimizer, dense_scheduler = optimizer_for(dense, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    for block, starts in enumerate(schedules, start=1):
        trained = train_block(
            dense, teacher, train, starts, dense_optimizer, dense_scheduler, scaler,
            device=device, upcycled=False,
        )
        metrics = evaluate_lm(dense, validation, gate_starts, device)
        progression.append({
            "replicate": args.replicate, "method": "dense_continued", "block": block,
            "tokens": block * TRAINING_BLOCK_TOKENS, **trained,
            "validation_nll": metrics["nll"], "validation_ppl": metrics["ppl"],
        })
    dense_formal = evaluate_lm(dense, validation, formal_starts, device)
    del dense, dense_optimizer, dense_scheduler
    torch.cuda.empty_cache()

    # Geometry is extracted only from frozen teacher local perceptions, with no labels.
    stage_samples = collect_stage_perceptions(
        teacher, validation, geometry_starts, sequence_length=128, device=device,
        max_samples_per_stage=8192, seed=122001 + args.replicate,
    )
    prototypes, geometry_diagnostics = geometry_prototypes(
        stage_samples, 4, seed=123001 + args.replicate * 100
    )
    (args.output_dir / f"r{args.replicate}-geometry-init.json").write_text(
        json.dumps({
            "format": "minicells.clm-upcycling-geometry-init.v1",
            "replicate": args.replicate,
            "method": "cosine-kmeans-local-perception",
            "stages": geometry_diagnostics,
        }, indent=2, sort_keys=True) + "\n"
    )

    method_summaries: dict[str, dict[str, object]] = {}
    for method in ("copy_random", "copy_geometry"):
        torch.manual_seed(131001 + args.replicate * 100 + (0 if method == "copy_random" else 1))
        torch.cuda.manual_seed_all(131001 + args.replicate * 100 + (0 if method == "copy_random" else 1))
        model = convert_textnca_to_upcycled(
            teacher, config=UpcyclingConfig(num_experts=4, top_k=1, router_scale=4.0)
        ).to(device)
        if method == "copy_geometry":
            model.set_router_prototypes(prototypes)
        else:
            model.provenance["router_initialization"] = "random"
        parity = validate_upcycled_parity(
            teacher, model, validation, gate_starts[:4], sequence_length=128, device=device
        )
        if parity["status"] != "CLM_UPCYCLING_EQUIVALENCE":
            raise RuntimeError(json.dumps({"method": method, **parity}, indent=2))
        initial = evaluate_upcycled_current(model, validation, gate_starts, device)
        initial_diversity = expert_diversity(model)
        optimizer, scheduler = optimizer_for(model, total_steps)
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        for block, starts in enumerate(schedules, start=1):
            model.set_execution_backend("masked_dense")
            trained = train_block(
                model, teacher, train, starts, optimizer, scheduler, scaler,
                device=device, upcycled=True,
            )
            metrics = evaluate_upcycled_current(model, validation, gate_starts, device)
            diversity = expert_diversity(model)
            progression.append({
                "replicate": args.replicate, "method": method, "block": block,
                "tokens": block * TRAINING_BLOCK_TOKENS, **trained,
                "validation_nll": metrics["nll"], "validation_ppl": metrics["ppl"],
                "usage_entropy": metrics["usage_entropy"],
                "router_logit_variance": metrics["router_logit_variance"],
                "program_usage": json.dumps(metrics["program_usage"]),
                **diversity,
            })
        formal_dynamic = evaluate_upcycled_current(model, validation, formal_starts, device)
        final_diversity = expert_diversity(model)
        controls.extend(
            final_controls(
                model, validation, calibration_starts, formal_starts,
                method, args.replicate, device,
            )
        )
        method_summaries[method] = {
            "parity": parity,
            "initial": initial,
            "initial_diversity": initial_diversity,
            "final": formal_dynamic,
            "final_diversity": final_diversity,
        }
        torch.save(
            {
                "format": "minicells.clm-upcycling-study-001-checkpoint.v1",
                "replicate": args.replicate,
                "method": method,
                "model_state": model.state_dict(),
                "provenance": model.provenance,
            },
            args.output_dir / f"r{args.replicate}-{method}.pt",
        )
        del model, optimizer, scheduler
        torch.cuda.empty_cache()

    pd.DataFrame(progression).to_csv(
        args.output_dir / f"r{args.replicate}-progression.csv", index=False
    )
    pd.DataFrame(controls).to_csv(
        args.output_dir / f"r{args.replicate}-controls.csv", index=False
    )
    manifest = {
        "format": "minicells.clm-upcycling-study-001-worker.v1",
        "complete": True,
        "replicate": args.replicate,
        "source_nll": source_metrics["nll"],
        "source_ppl": source_metrics["ppl"],
        "dense_nll": dense_formal["nll"],
        "dense_ppl": dense_formal["ppl"],
        "random_parity": method_summaries["copy_random"]["parity"]["status"],
        "geometry_parity": method_summaries["copy_geometry"]["parity"]["status"],
        "random_initial_ppl": method_summaries["copy_random"]["initial"]["ppl"],
        "geometry_initial_ppl": method_summaries["copy_geometry"]["initial"]["ppl"],
        "random_final_ppl": method_summaries["copy_random"]["final"]["ppl"],
        "geometry_final_ppl": method_summaries["copy_geometry"]["final"]["ppl"],
        "random_expert_relative_l2": method_summaries["copy_random"]["final_diversity"]["expert_pairwise_relative_l2"],
        "geometry_expert_relative_l2": method_summaries["copy_geometry"]["final_diversity"]["expert_pairwise_relative_l2"],
        "training_tokens_per_arm": TRAINING_BLOCK_TOKENS * TRAINING_BLOCKS,
        "same_continuation_schedule_across_arms": True,
        "cell_activation": 1.0,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
