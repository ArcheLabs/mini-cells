#!/usr/bin/env python3
"""Run one resumable member of the preregistered CLM-0.3 matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from minicells.clm_growth import ProgressiveGrowthCLM, next_growth_event, phase_for_tokens, replicate_seed, stop_target
from minicells.clm_release import build_release_model
from minicells.growth_checkpoint import GlobalLRScheduler, load_growth_checkpoint, save_growth_checkpoint, verify_base_release_hash
from minicells.growth_pressure import calibrate_model_pressure, select_pressure_parent, select_random_parent, write_pressure_table
from minicells.growth_reporting import write_growth_history, write_ppl_history
from minicells.growth_validation import clm_growth_loss, evaluate_nll, make_ppl_row, newborn_causal_diagnostics
from minicells.language_clm_validation import load_experiment_006_teacher
from minicells.language_data import batch_from_starts, make_training_schedule


PROGRESS_INTERVAL = 25_000
EVAL_INTERVAL = 100_000


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run/resume one CLM-0.3 growth worker")
    result.add_argument("--release-dir", type=Path, default=Path("artifacts/releases/clm-0.1"))
    result.add_argument("--source-005-dir", type=Path, default=Path("artifacts/experiments/005-consumer-language-bridge"))
    result.add_argument("--textnca-checkpoint", type=Path, default=Path("artifacts/experiments/006-consumer-language-scaling/minicells-v2-10m.pt"))
    result.add_argument("--textnca-config", type=Path, default=Path("artifacts/experiments/006-consumer-language-scaling/model-configs.json"))
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--arm", choices=("fixed4", "pressure_growth", "random_growth"), required=True)
    result.add_argument("--replicate", type=int, choices=range(3), required=True)
    result.add_argument("--resume-input", type=Path)
    result.add_argument("--stop-after-tokens", type=int)
    result.add_argument("--target-tokens", type=int, default=1_500_000)
    result.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    result.add_argument("--execute", action="store_true")
    result.add_argument("--batch-size", type=int, default=8)
    result.add_argument("--sequence-length", type=int, default=125)
    result.add_argument("--eval-batches", type=int, default=4)
    return result


def schedule_digest(starts: tuple[tuple[int, ...], ...]) -> str:
    return hashlib.sha256(json.dumps(starts, separators=(",", ":")).encode()).hexdigest()


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Telemetry:
    def __init__(self, output: Path, arm: str, replicate: int) -> None:
        self.output, self.arm, self.replicate = output, arm, replicate
        output.mkdir(parents=True, exist_ok=True)
        self.handle = (output / "events.jsonl").open("a", encoding="utf-8")

    def write(self, event: dict[str, object]) -> None:
        event = {"arm": self.arm, "replicate": self.replicate, "time": time.time(), **event}
        self.handle.write(json.dumps(event, sort_keys=True) + "\n"); self.handle.flush()
        if event["type"] == "training_progress":
            self.output.joinpath("progress.json").write_text(json.dumps(event, indent=2, sort_keys=True) + "\n")
            print(f"[r{self.replicate} {self.arm}] {event['consumed_tokens']:,}/{event['target_tokens']:,} {event['phase']}", flush=True)

    def close(self) -> None:
        self.handle.close()


def release_teacher(release_dir: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(release_dir / "model.pt", map_location="cpu", weights_only=False)
    model = build_release_model(num_experts=int(checkpoint["num_experts"]), router_scale=float(checkpoint["router_scale"]))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.to(device).eval().requires_grad_(False)


def validation_batches(stream: torch.Tensor, args: argparse.Namespace, device: torch.device) -> list[tuple[torch.Tensor, torch.Tensor]]:
    width = args.sequence_length + 1
    starts = tuple(range(0, args.eval_batches * args.batch_size * width, width))
    return [batch_from_starts(stream, starts[i:i + args.batch_size], args.sequence_length, device)
            for i in range(0, len(starts), args.batch_size)]


def checkpoint(path: Path, model: ProgressiveGrowthCLM, optimizer: torch.optim.Optimizer,
               scheduler: GlobalLRScheduler, consumed: int, step: int, digest: str) -> None:
    save_growth_checkpoint(path, model=model, optimizer=optimizer, scheduler=scheduler,
                           consumed_tokens=consumed, training_step=step,
                           data_schedule_state={"current_step": step, "consumed_tokens": consumed,
                                                "schedule_sha256": digest})


def run(args: argparse.Namespace, telemetry: Telemetry, observed_hash: str) -> int:
    from minicells.language_scaling import prepare_scaling_corpus

    device = torch.device(args.device)
    seed = replicate_seed(args.replicate)
    seed_all(seed)
    train, validation_stream, _, corpus_manifest = prepare_scaling_corpus(
        Path("."), source_005_dir=args.source_005_dir,
        train_stream_tokens=args.target_tokens + args.sequence_length + 2,
        validation_stream_tokens=max(2048, args.eval_batches * args.batch_size * (args.sequence_length + 1)))
    schedule = make_training_schedule(len(train), seed=seed, budget_tokens=args.target_tokens,
                                      batch_size=args.batch_size, sequence_length=args.sequence_length)
    digest = schedule_digest(schedule.starts)
    target = stop_target(args.target_tokens, args.stop_after_tokens, schedule.tokens_per_step)
    model = ProgressiveGrowthCLM.from_clm01_release(str(args.release_dir), device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    lr = lambda step: 3e-4 * min(1.0, (step + 1) / 100.0)
    scheduler = GlobalLRScheduler(optimizer, lr)
    consumed = start_step = 0
    if args.resume_input:
        model, payload = load_growth_checkpoint(args.resume_input, model=model, optimizer=optimizer,
                                                scheduler=scheduler, map_location=device)
        consumed, start_step = int(payload["consumed_tokens"]), int(payload["training_step"])
        saved_digest = payload.get("data_schedule_state", {}).get("schedule_sha256")
        if saved_digest and saved_digest != digest:
            raise RuntimeError("resume data schedule does not match this replicate")
    fixed_teacher = release_teacher(args.release_dir, device)
    textnca = load_experiment_006_teacher(str(args.textnca_checkpoint), device=device,
                                         model_config_path=str(args.textnca_config))
    validation = validation_batches(validation_stream, args, device)
    clm01_ppl = math.exp(evaluate_nll(fixed_teacher, validation))
    textnca_ppl = math.exp(evaluate_nll(textnca, validation))
    telemetry.write({"type": "worker_started", "consumed_tokens": consumed, "target_tokens": target,
                     "base_model_sha256": observed_hash, "replicate_seed": seed,
                     "schedule_sha256": digest, "corpus_manifest": corpus_manifest})
    rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    started = time.time()
    for step in range(start_step, schedule.steps):
        if consumed >= target:
            break
        due = next_growth_event(consumed, model.growth_history)
        if args.arm != "fixed4" and due is not None:
            _, scheduled_token = due
            calibration = [batch_from_starts(train, schedule.starts[i], schedule.sequence_length, device)
                           for i in range(max(0, step - 8), step)]
            candidates, perceptions = calibrate_model_pressure(model, calibration)
            birth_index = len(model.growth_history) + 1
            write_pressure_table(args.output_dir / f"pressure-scan-{birth_index}.csv", candidates)
            parent = (select_pressure_parent(candidates) if args.arm == "pressure_growth" else
                      select_random_parent(candidates, seed=seed + birth_index))
            telemetry.write({"type": "pressure_scan", "birth_index": birth_index,
                             "scheduled_token": scheduled_token,
                             "candidates": [item.to_row() for item in candidates],
                             "selected_expert": parent.expert_id})
            checkpoint(args.output_dir / f"before-birth-{birth_index}.pt", model, optimizer,
                       scheduler, consumed, step, digest)
            event = model.birth(stage=parent.stage, parent_id=parent.expert_id,
                                routed_perceptions=perceptions[parent.expert_id].to(device), token=consumed,
                                validation_inputs=validation[0][0], validation_targets=validation[0][1],
                                selection_method="pressure" if args.arm == "pressure_growth" else "random",
                                pressure={"usage": parent.usage, "gradient_disagreement": parent.grad_conflict,
                                          "score": parent.pressure}, optimizer=optimizer)
            telemetry.write({"type": "birth", "birth_index": birth_index, "scheduled_token": scheduled_token,
                             "actual_token": consumed, "stage": parent.stage, "parent": parent.expert_id,
                             "child": event["child"], "parity": event["parity"],
                             "parity_status": event["parity"]["status"]})
            checkpoint(args.output_dir / f"after-birth-{birth_index}.pt", model, optimizer,
                       scheduler, consumed, step, digest)
            diag = newborn_causal_diagnostics(model, validation, stage=parent.stage,
                                              parent_id=parent.expert_id, child_id=event["child"])
            diagnostics.append({"birth_index": birth_index, "offset_tokens": 0, **diag})
            telemetry.write({"type": "newborn_diagnostic", **diagnostics[-1]})
        inputs, targets = batch_from_starts(train, schedule.starts[step], schedule.sequence_length, device)
        optimizer.zero_grad(set_to_none=True)
        student, stats = model(inputs, execution_backend="masked_dense", return_stats=True)
        with torch.no_grad():
            teacher_output = fixed_teacher(inputs)
        loss = clm_growth_loss(student, teacher_output, targets, root_usage=stats.root_usage)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        consumed = (step + 1) * schedule.tokens_per_step
        scheduler.step(step + 1)
        if consumed % PROGRESS_INTERVAL == 0 or consumed >= target:
            elapsed = max(time.time() - started, 1e-9)
            telemetry.write({"type": "training_progress", "consumed_tokens": consumed,
                             "target_tokens": target, "phase": phase_for_tokens(consumed),
                             "train_loss": float(loss.detach()), "lr": scheduler.optimizer.param_groups[0]["lr"],
                             "tokens_per_second": (consumed - start_step * schedule.tokens_per_step) / elapsed,
                             "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0})
        if consumed % EVAL_INTERVAL == 0 or consumed >= target:
            nll = evaluate_nll(model, validation); ppl = math.exp(nll)
            row = make_ppl_row(replicate=args.replicate, arm=args.arm, tokens=consumed,
                               phase=phase_for_tokens(consumed), ppl=ppl, nll=nll,
                               fixed4_ppl=clm01_ppl, clm01_start_ppl=clm01_ppl,
                               textnca_frozen_ppl=textnca_ppl)
            rows.append(row)
            telemetry.write({"type": "evaluation", **row, "raw_model_ppl": ppl,
                             "clm01_start_ppl": clm01_ppl, "textnca_frozen_ppl": textnca_ppl})
            checkpoint(args.output_dir / f"checkpoint-{consumed}.pt", model, optimizer,
                       scheduler, consumed, step + 1, digest)
        for event in model.growth_history:
            offset = consumed - int(event["token"])
            if offset in (100_000, 250_000) or consumed >= target and not any(
                    d["birth_index"] == event["birth_index"] and d["offset_tokens"] == offset for d in diagnostics):
                diag = newborn_causal_diagnostics(model, validation, stage=int(event["stage"]),
                                                  parent_id=str(event["parent"]), child_id=str(event["child"]))
                diagnostics.append({"birth_index": event["birth_index"], "offset_tokens": offset, **diag})
                telemetry.write({"type": "newborn_diagnostic", **diagnostics[-1]})
    write_ppl_history(args.output_dir / "ppl-history.csv", rows)
    write_growth_history(args.output_dir / "growth-history.json", model.growth_history)
    args.output_dir.joinpath("newborn-diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    checkpoint(args.output_dir / "final.pt", model, optimizer, scheduler, consumed,
               consumed // schedule.tokens_per_step, digest)
    telemetry.write({"type": "worker_complete", "consumed_tokens": consumed, "target_tokens": target,
                     "completed_births": len(model.growth_history), "base_model_sha256": observed_hash})
    return 0


def main() -> int:
    args = parser().parse_args()
    observed = verify_base_release_hash(args.release_dir / "model.pt")
    telemetry = Telemetry(args.output_dir, args.arm, args.replicate)
    try:
        if not args.execute:
            telemetry.write({"type": "worker_started", "consumed_tokens": 0,
                             "target_tokens": args.target_tokens, "base_model_sha256": observed,
                             "mode": "preflight_only"})
            telemetry.write({"type": "training_progress", "consumed_tokens": 0,
                             "target_tokens": args.target_tokens, "phase": "preflight_only",
                             "train_loss": None, "lr": None, "tokens_per_second": 0.0,
                             "peak_vram_bytes": 0})
            telemetry.write({"type": "worker_complete", "consumed_tokens": 0,
                             "target_tokens": args.target_tokens, "mode": "preflight_only"})
            return 0
        return run(args, telemetry, observed)
    finally:
        telemetry.close()


if __name__ == "__main__":
    raise SystemExit(main())
