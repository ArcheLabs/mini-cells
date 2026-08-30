from __future__ import annotations

import argparse
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

from minicells.clm_v2_closed_loop_validation import (  # noqa: E402
    FINAL_HANDOFF_RATIO_MAX,
    handoff_stage_pass,
    local_imitation_pass,
)
from minicells.clm_v2_training import (  # noqa: E402
    HANDOFF_STAGES,
    K_STAGES,
    handoff_loss,
    latest_stage_checkpoint,
    normalized_local_loss,
    save_v2_checkpoint,
)
from minicells.clm_v2_validation import (  # noqa: E402
    V2RoutingRecorder,
    evaluate_v2,
    evaluation_dict,
    static_mask,
    v2_router_diagnostics,
    validate_v2_scaffold_parity,
)
from minicells.language_clm_validation import load_experiment_006_teacher  # noqa: E402
from minicells.language_data import (  # noqa: E402
    batch_from_starts,
    fixed_validation_starts,
    make_training_schedule,
)
from minicells.textnca_to_clm_v2 import convert_textnca_to_clm_v2  # noqa: E402

ARM_COLUMNS = [
    "replicate", "arm", "top_k", "nll", "ppl", "sample_variation",
    "position_variation", "temporal_variation", "program_usage",
    "program_coactivation", "receptor_ratio", "active_ffn_ratio",
    "effective_ffn_ratio", "tokens_per_second", "shuffle_permutations",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one CLM v2 closed-loop handoff replicate.")
    parser.add_argument("--replicate", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    return parser.parse_args()


@torch.no_grad()
def evaluate_current(model, stream, starts_batches, device) -> dict[str, float]:
    model.eval()
    total = tokens = 0
    relative = cosine = 0.0
    for starts in starts_batches:
        inputs, targets = batch_from_starts(stream, starts, 128, device)
        output, stats = model(inputs, return_stats=True, return_local_imitation=True)
        total += float(F.cross_entropy(
            output.logits.flatten(0, 1), targets.flatten(), reduction="sum"
        ))
        tokens += targets.numel()
        relative += float(stats.local_relative_mse)
        cosine += float(stats.local_cosine_similarity)
    count = len(starts_batches)
    nll = total / tokens
    return {
        "nll": nll,
        "ppl": math.exp(min(nll, 20)),
        "local_relative_mse": relative / count,
        "local_cosine": cosine / count,
    }


@torch.no_grad()
def evaluate_teacher(teacher, stream, starts_batches, device) -> dict[str, float]:
    teacher.eval()
    total = tokens = 0
    for starts in starts_batches:
        inputs, targets = batch_from_starts(stream, starts, 128, device)
        output = teacher(inputs)
        total += float(F.cross_entropy(
            output.logits.flatten(0, 1), targets.flatten(), reduction="sum"
        ))
        tokens += targets.numel()
    nll = total / tokens
    return {"nll": nll, "ppl": math.exp(min(nll, 20))}


def train_tokens(
    teacher, student, optimizer, scheduler, scaler, stream, *, budget, seed,
    device, local_only: bool, local_weight: float,
) -> dict[str, float]:
    schedule = make_training_schedule(
        int(stream.numel()), seed=seed, budget_tokens=budget,
        batch_size=8, sequence_length=125,
    )
    amp = device.type == "cuda"
    losses: list[float] = []
    relative: list[float] = []
    grad_norms: list[float] = []
    started = time.perf_counter()
    for starts in schedule.starts:
        student.train()
        inputs, targets = batch_from_starts(stream, starts, 125, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp
        ):
            teacher_output = teacher(inputs)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            output, stats = student(inputs, return_local_imitation=True)
            loss = normalized_local_loss(stats) if local_only else handoff_loss(
                output, teacher_output, targets, stats, local_weight=local_weight
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        router_gradients = [
            stage.program_bank.receptor.out_proj.weight.grad.detach().float().reshape(-1)
            for stage in student.stages
            if stage.program_bank.receptor.out_proj.weight.grad is not None
        ]
        grad_norms.append(
            float(torch.cat(router_gradients).norm()) if router_gradients else 0.0
        )
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        losses.append(float(loss.detach()))
        relative.append(float(stats.local_relative_mse.detach()))
    return {
        "loss": losses[-1],
        "local_relative_mse": relative[-1],
        "router_grad_norm": sum(grad_norms) / len(grad_norms),
        "elapsed_seconds": time.perf_counter() - started,
    }


def checkpoint_payload(
    student, optimizer, scheduler, scaler, replicate, stage_index,
    rows, diagnostics, status="RUNNING",
):
    return {
        "format": "minicells.clm-v2-validation-001b-checkpoint.v1",
        "replicate": replicate,
        "stage_index": stage_index,
        "status": status,
        "model_state": student.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "rows": rows,
        "diagnostics": diagnostics,
        "rng_state": torch.random.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all(),
        "scaffold_provenance": student.scaffold_provenance,
    }


def final_controls(
    student, validation, calibration_starts, formal_starts,
    top_k, replicate, device,
):
    student.set_scaffold_alpha(0)
    student.set_program_top_k(top_k)
    calibration = []
    for starts in calibration_starts:
        inputs, _ = batch_from_starts(validation, starts, 128, device)
        with V2RoutingRecorder(student) as recorder:
            student(inputs)
        calibration.append(recorder.masks)
    fixed = static_mask(calibration, top_k)
    results = [evaluate_v2(
        student, validation, formal_starts, sequence_length=128, device=device,
        arm=arm, top_k=top_k, fixed_mask=fixed,
    ) for arm in ("dense", "dynamic", "static")]
    shuffled = [evaluate_v2(
        student, validation, formal_starts, sequence_length=128, device=device,
        arm="shuffled", top_k=top_k, fixed_mask=fixed,
        permutation_seed=112001 + replicate * 100 + index,
    ) for index in range(3)]
    rows = [{"replicate": replicate, **evaluation_dict(result)} for result in results]
    shuffled_rows = [evaluation_dict(result) for result in shuffled]
    average = dict(shuffled_rows[0])
    for key in (
        "nll", "sample_variation", "position_variation", "temporal_variation",
        "receptor_ratio", "active_ffn_ratio", "effective_ffn_ratio",
        "tokens_per_second",
    ):
        average[key] = sum(float(row[key]) for row in shuffled_rows) / 3
    average["ppl"] = math.exp(float(average["nll"]))
    for key in ("program_usage", "program_coactivation"):
        average[key] = torch.tensor([row[key] for row in shuffled_rows]).mean(0).tolist()
    rows.append({"replicate": replicate, **average, "shuffle_permutations": 3})
    return rows


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / f"r{args.replicate}-worker.json"
    if manifest_path.is_file() and json.loads(manifest_path.read_text()).get("complete"):
        print(f"replicate {args.replicate} already complete; skipping")
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("CLM v2 closed-loop validation requires CUDA")

    device = torch.device("cuda:0")
    train = torch.load(args.cache_dir / "train-tokens.pt", map_location="cpu", weights_only=True)
    validation = torch.load(
        args.cache_dir / "validation-tokens.pt", map_location="cpu", weights_only=True
    )
    gate_starts = fixed_validation_starts(
        validation.numel(), batches=16, batch_size=8, sequence_length=128, seed=7105
    )
    calibration_starts = fixed_validation_starts(
        validation.numel(), batches=8, batch_size=8, sequence_length=128, seed=7205
    )
    formal_starts = fixed_validation_starts(
        validation.numel(), batches=24, batch_size=8, sequence_length=128, seed=7305
    )

    teacher = load_experiment_006_teacher(
        str(args.checkpoint), device=device, model_config_path=str(args.model_config)
    )
    teacher_metrics = evaluate_teacher(teacher, validation, gate_starts, device)
    seed = 103001 + args.replicate
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    student = convert_textnca_to_clm_v2(teacher).to(device)
    parity = validate_v2_scaffold_parity(
        teacher, student, validation, gate_starts[:4], sequence_length=128, device=device
    )
    if parity["status"] != "CLMV2_SCAFFOLD_EQUIVALENCE":
        raise RuntimeError(json.dumps(parity, indent=2))
    (args.output_dir / f"r{args.replicate}-stage0.json").write_text(
        json.dumps({"format": "minicells.clm-v2-validation-001b-stage0.v1", **parity}, indent=2)
        + "\n"
    )

    student.freeze_inherited_backbone()
    optimizer = torch.optim.AdamW(
        student.sparse_parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=2_000_000 // 1000, eta_min=1e-5
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    stage_index = 0

    latest = latest_stage_checkpoint(args.output_dir, args.replicate)
    if latest is not None:
        payload = torch.load(latest, map_location=device, weights_only=True)
        student.load_state_dict(payload["model_state"])
        rows = list(payload["rows"])
        diagnostics = list(payload.get("diagnostics", []))
        stage_index = int(payload["stage_index"])
        crossed_boundary = any(
            row["phase"] == "consolidation-k6" or str(row["phase"]).startswith("top-")
            for row in rows
        )
        if crossed_boundary:
            student.unfreeze_sparse_backbone()
            optimizer = torch.optim.AdamW(
                [parameter for parameter in student.parameters() if parameter.requires_grad],
                lr=5e-5, betas=(0.9, 0.95), weight_decay=0.1,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=(500_000 + sum(tokens for _, tokens in K_STAGES)) // 1000,
                eta_min=5e-6,
            )
        optimizer.load_state_dict(payload["optimizer_state"])
        scheduler.load_state_dict(payload["scheduler_state"])
        scaler.load_state_dict(payload["scaler_state"])
        torch.random.set_rng_state(payload["rng_state"].cpu())
        torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
        print(f"resumed replicate {args.replicate} from {latest.name}")

    completed_phases = {str(row["phase"]) for row in rows}
    if not rows:
        student.set_scaffold_alpha(1)
        student.set_program_top_k(6)

    imitation_pass = any(
        str(row["phase"]).startswith("imitation-")
        and bool(row.get("local_gate_pass", False))
        for row in rows
    )
    for attempt in range(2):
        phase_name = f"imitation-{attempt + 1}"
        if phase_name in completed_phases or imitation_pass:
            continue
        stage_index += 1
        student.set_scaffold_alpha(1)
        student.set_program_top_k(6)
        trained = train_tokens(
            teacher, student, optimizer, scheduler, scaler, train,
            budget=500_000,
            seed=104001 + args.replicate * 100 + stage_index,
            device=device, local_only=True, local_weight=1.0,
        )
        dense_metrics = evaluate_current(student, validation, gate_starts, device)
        local_gate = local_imitation_pass(
            dense_metrics["local_relative_mse"], dense_metrics["local_cosine"]
        )
        student.set_scaffold_alpha(0)
        zero_scaffold = evaluate_current(student, validation, gate_starts, device)
        zero_scaffold_ratio = zero_scaffold["ppl"] / teacher_metrics["ppl"]
        student.set_scaffold_alpha(1)
        rows.append({
            "replicate": args.replicate,
            "phase": phase_name,
            "alpha": 1.0,
            "top_k": 6,
            "quality_ratio": dense_metrics["ppl"] / teacher_metrics["ppl"],
            "local_gate_pass": local_gate,
            "zero_scaffold_ppl_ratio_telemetry": zero_scaffold_ratio,
            **trained,
            **{f"validation_{key}": value for key, value in dense_metrics.items()},
        })
        diagnostics.append({
            "replicate": args.replicate,
            "phase": phase_name,
            "router_grad_norm": trained["router_grad_norm"],
            **v2_router_diagnostics(
                student, validation, calibration_starts[:2], sequence_length=128, device=device
            ),
        })
        save_v2_checkpoint(
            args.output_dir / f"r{args.replicate}-stage-{stage_index}.pt",
            checkpoint_payload(
                student, optimizer, scheduler, scaler, args.replicate,
                stage_index, rows, diagnostics,
            ),
        )
        if local_gate:
            imitation_pass = True
            break

    status = "RUNNING" if imitation_pass else "CLMV2_CLOSED_LOOP_LOCAL_APPROXIMATION_FAILURE"

    if imitation_pass:
        for handoff in HANDOFF_STAGES:
            existing = next((row for row in rows if row["phase"] == handoff.name), None)
            if existing is not None:
                if not bool(existing.get("handoff_gate_pass", False)):
                    status = "CLMV2_CLOSED_LOOP_HANDOFF_FAILURE"
                    break
                student.set_scaffold_alpha(handoff.alpha)
                continue

            stage_index += 1
            student.set_scaffold_alpha(handoff.alpha)
            before = evaluate_current(student, validation, gate_starts, device)
            trained = train_tokens(
                teacher, student, optimizer, scheduler, scaler, train,
                budget=handoff.tokens,
                seed=104001 + args.replicate * 100 + stage_index,
                device=device, local_only=False, local_weight=handoff.local_weight,
            )
            after = evaluate_current(student, validation, gate_starts, device)
            safety_ratio = after["ppl"] / teacher_metrics["ppl"]
            progress_ratio = after["ppl"] / before["ppl"]
            passed = handoff_stage_pass(
                before_ppl=before["ppl"], after_ppl=after["ppl"],
                teacher_ppl=teacher_metrics["ppl"],
            )
            rows.append({
                "replicate": args.replicate,
                "phase": handoff.name,
                "alpha": handoff.alpha,
                "top_k": 6,
                "quality_ratio": safety_ratio,
                "handoff_before_ppl": before["ppl"],
                "handoff_before_ratio": before["ppl"] / teacher_metrics["ppl"],
                "handoff_progress_ratio": progress_ratio,
                "handoff_safety_ratio": safety_ratio,
                "handoff_gate_pass": passed,
                **trained,
                **{f"validation_{key}": value for key, value in after.items()},
            })
            diagnostics.append({
                "replicate": args.replicate,
                "phase": handoff.name,
                "router_grad_norm": trained["router_grad_norm"],
                **v2_router_diagnostics(
                    student, validation, calibration_starts[:2],
                    sequence_length=128, device=device,
                ),
            })
            save_v2_checkpoint(
                args.output_dir / f"r{args.replicate}-stage-{stage_index}.pt",
                checkpoint_payload(
                    student, optimizer, scheduler, scaler, args.replicate,
                    stage_index, rows, diagnostics,
                    "RUNNING" if passed else "CLMV2_CLOSED_LOOP_HANDOFF_FAILURE",
                ),
            )
            if not passed:
                status = "CLMV2_CLOSED_LOOP_HANDOFF_FAILURE"
                break

    quality_safe_k = 6
    reference_k6_ppl: float | None = None
    consolidation_row = next(
        (row for row in rows if row["phase"] == "consolidation-k6"), None
    )
    if consolidation_row is not None:
        reference_k6_ppl = float(consolidation_row["validation_ppl"])
        status = (
            "CLMV2_CLOSED_LOOP_HANDOFF_SIGNAL"
            if float(consolidation_row["quality_ratio"]) <= FINAL_HANDOFF_RATIO_MAX
            else "CLMV2_CLOSED_LOOP_HANDOFF_FAILURE"
        )
    elif status == "RUNNING" and student.config.scaffold_alpha == 0:
        student.unfreeze_sparse_backbone()
        optimizer = torch.optim.AdamW(
            [parameter for parameter in student.parameters() if parameter.requires_grad],
            lr=5e-5, betas=(0.9, 0.95), weight_decay=0.1,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=(500_000 + sum(tokens for _, tokens in K_STAGES)) // 1000,
            eta_min=5e-6,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        stage_index += 1
        trained = train_tokens(
            teacher, student, optimizer, scheduler, scaler, train,
            budget=500_000,
            seed=104001 + args.replicate * 100 + stage_index,
            device=device, local_only=False, local_weight=0.25,
        )
        metrics = evaluate_current(student, validation, gate_starts, device)
        ratio = metrics["ppl"] / teacher_metrics["ppl"]
        reference_k6_ppl = metrics["ppl"]
        rows.append({
            "replicate": args.replicate,
            "phase": "consolidation-k6",
            "alpha": 0.0,
            "top_k": 6,
            "quality_ratio": ratio,
            **trained,
            **{f"validation_{key}": value for key, value in metrics.items()},
        })
        diagnostics.append({
            "replicate": args.replicate,
            "phase": "consolidation-k6",
            "router_grad_norm": trained["router_grad_norm"],
            **v2_router_diagnostics(
                student, validation, calibration_starts[:2], sequence_length=128, device=device
            ),
        })
        status = (
            "CLMV2_CLOSED_LOOP_HANDOFF_SIGNAL"
            if ratio <= FINAL_HANDOFF_RATIO_MAX
            else "CLMV2_CLOSED_LOOP_HANDOFF_FAILURE"
        )
        save_v2_checkpoint(
            args.output_dir / f"r{args.replicate}-stage-{stage_index}.pt",
            checkpoint_payload(
                student, optimizer, scheduler, scaler, args.replicate,
                stage_index, rows, diagnostics, status,
            ),
        )

    if status == "CLMV2_CLOSED_LOOP_HANDOFF_SIGNAL":
        assert reference_k6_ppl is not None
        safe_state = {
            key: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
            for key, value in student.state_dict().items()
        }
        for top_k, budget in K_STAGES:
            existing = next((row for row in rows if row["phase"] == f"top-{top_k}"), None)
            if existing is not None:
                if float(existing["quality_ratio"]) > FINAL_HANDOFF_RATIO_MAX:
                    break
                quality_safe_k = top_k
                continue
            stage_index += 1
            student.set_program_top_k(top_k)
            trained = train_tokens(
                teacher, student, optimizer, scheduler, scaler, train,
                budget=budget,
                seed=104001 + args.replicate * 100 + stage_index,
                device=device, local_only=False, local_weight=0.25,
            )
            metrics = evaluate_current(student, validation, gate_starts, device)
            ratio = metrics["ppl"] / reference_k6_ppl
            passed = ratio <= FINAL_HANDOFF_RATIO_MAX
            rows.append({
                "replicate": args.replicate,
                "phase": f"top-{top_k}",
                "alpha": 0.0,
                "top_k": top_k,
                "quality_ratio": ratio,
                **trained,
                **{f"validation_{key}": value for key, value in metrics.items()},
            })
            diagnostics.append({
                "replicate": args.replicate,
                "phase": f"top-{top_k}",
                "router_grad_norm": trained["router_grad_norm"],
                **v2_router_diagnostics(
                    student, validation, calibration_starts[:2],
                    sequence_length=128, device=device,
                ),
            })
            save_v2_checkpoint(
                args.output_dir / f"r{args.replicate}-stage-{stage_index}.pt",
                checkpoint_payload(
                    student, optimizer, scheduler, scaler, args.replicate,
                    stage_index, rows, diagnostics, status,
                ),
            )
            if not passed:
                student.load_state_dict(safe_state)
                break
            quality_safe_k = top_k
            safe_state = {
                key: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
                for key, value in student.state_dict().items()
            }

    arms: list[dict[str, object]] = []
    if status == "CLMV2_CLOSED_LOOP_HANDOFF_SIGNAL":
        arms = final_controls(
            student, validation, calibration_starts, formal_starts,
            quality_safe_k, args.replicate, device,
        )

    pd.DataFrame(rows).to_csv(
        args.output_dir / f"r{args.replicate}-progression.csv", index=False
    )
    pd.DataFrame(arms, columns=ARM_COLUMNS).to_csv(
        args.output_dir / f"r{args.replicate}-arms.csv", index=False
    )
    pd.DataFrame(diagnostics).to_csv(
        args.output_dir / f"r{args.replicate}-router-diagnostics.csv", index=False
    )
    manifest = {
        "format": "minicells.clm-v2-validation-001b-worker.v1",
        "complete": True,
        "replicate": args.replicate,
        "status": status,
        "quality_safe_k": quality_safe_k,
        "seed": seed,
        "stage_level_checkpoints": True,
        "cell_activation": 1.0,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
