#!/usr/bin/env python3
"""Resumable CLM-0.3 worker scaffold.

The worker exposes the production telemetry/checkpoint contract.  The formal
GPU experiment is intentionally not started by this implementation task.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from minicells.clm_growth import ProgressiveGrowthCLM
from minicells.clm_release import build_release_model
from minicells.growth_checkpoint import GlobalLRScheduler, load_growth_checkpoint, save_growth_checkpoint, verify_base_release_hash
from minicells.growth_pressure import calibrate_model_pressure, select_pressure_parent, select_random_parent
from minicells.growth_validation import clm_growth_loss, evaluate_nll, make_ppl_row
from minicells.language_data import batch_from_starts, make_training_schedule, prepare_tinystories_corpus


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run/resume one CLM-0.3 growth worker")
    parser.add_argument("--release-dir", type=Path, default=Path("artifacts/releases/clm-0.1"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=("fixed4", "pressure_growth", "random_growth"), required=True)
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--resume-input", type=Path)
    parser.add_argument("--stop-after-tokens", type=int)
    parser.add_argument("--target-tokens", type=int, default=1_500_000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--execute", action="store_true", help="run continuation training; omit for preflight only")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=125)
    parser.add_argument("--eval-batches", type=int, default=4)
    return parser


class Telemetry:
    def __init__(self, output: Path, arm: str, replicate: int) -> None:
        self.output = output
        self.arm = arm
        self.replicate = replicate
        output.mkdir(parents=True, exist_ok=True)
        self.events = (output / "events.jsonl").open("a", encoding="utf-8")

    def write(self, event: dict[str, object]) -> None:
        event.setdefault("arm", self.arm)
        event.setdefault("replicate", self.replicate)
        self.events.write(json.dumps(event, sort_keys=True) + "\n")
        self.events.flush()
        if event.get("type") == "training_progress":
            print(
                f"[CLM-0.3][r{self.replicate}][{self.arm}] "
                f"{event['consumed_tokens']:,}/{event['target_tokens']:,} tokens "
                f"phase={event.get('phase', 'training')}",
                flush=True,
            )

    def close(self) -> None:
        self.events.close()


def main() -> int:
    args = _parser().parse_args()
    model_path = args.release_dir / "model.pt"
    observed = verify_base_release_hash(model_path)
    telemetry = Telemetry(args.output_dir, args.arm, args.replicate)
    try:
        if args.resume_input and not args.execute:
            from minicells.growth_checkpoint import load_growth_checkpoint

            model = ProgressiveGrowthCLM.from_clm01_release(str(args.release_dir), device=args.device)
            model, payload = load_growth_checkpoint(args.resume_input, model=model, map_location=args.device)
            consumed = int(payload["consumed_tokens"])
        else:
            model = ProgressiveGrowthCLM.from_clm01_release(str(args.release_dir), device=args.device)
            consumed = 0
        model.to(args.device)
        target = min(args.target_tokens, args.stop_after_tokens or args.target_tokens)
        telemetry.write({
            "type": "worker_started", "base_model_sha256": observed,
            "consumed_tokens": consumed, "target_tokens": target,
        })
        if args.execute:
            return run_training(args, model, telemetry, observed, consumed)
        # The default is a no-download preflight.  Kaggle enables --execute
        # after reviewing the baseline and CPU gates.
        save_growth_checkpoint(
            args.output_dir / "checkpoint-initial.pt", model=model,
            consumed_tokens=consumed, metrics={"status": "preflight_only"},
        )
        telemetry.write({
            "type": "training_progress", "consumed_tokens": consumed,
            "target_tokens": target, "phase": "preflight_only",
            "train_loss": None, "lr": None, "tokens_per_second": 0.0,
            "peak_vram_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
        })
        (args.output_dir / "progress.json").write_text(json.dumps({
            "consumed_tokens": consumed, "target_tokens": target,
            "status": "preflight_only", "updated_at": time.time(),
        }, indent=2) + "\n", encoding="utf-8")
        return 0
    finally:
        telemetry.close()


def run_training(args: argparse.Namespace, model: ProgressiveGrowthCLM, telemetry: Telemetry,
                 observed_hash: str, consumed: int) -> int:
    corpus = prepare_tinystories_corpus(
        Path("."), train_stream_tokens=args.target_tokens + args.sequence_length + 2,
        validation_stream_tokens=max(args.eval_batches * args.batch_size * (args.sequence_length + 1), 2048),
    )
    schedule = make_training_schedule(
        corpus.train.numel(), seed=5005, budget_tokens=args.target_tokens,
        batch_size=args.batch_size, sequence_length=args.sequence_length,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = GlobalLRScheduler(optimizer, lambda step: 3e-4 * min(1.0, (step + 1) / 100.0),
                                  step=consumed // schedule.tokens_per_step)
    if args.resume_input:
        model, payload = load_growth_checkpoint(
            args.resume_input, model=model, optimizer=optimizer, scheduler=scheduler,
            map_location=args.device,
        )
        consumed = int(payload["consumed_tokens"])
    release_checkpoint = torch.load(args.release_dir / "model.pt", map_location="cpu", weights_only=False)
    teacher = build_release_model(
        num_experts=int(release_checkpoint["num_experts"]),
        router_scale=float(release_checkpoint["router_scale"]),
    )
    teacher.load_state_dict(release_checkpoint["model_state"], strict=True)
    teacher = teacher.to(args.device).eval()
    teacher.requires_grad_(False)
    validation = []
    starts = tuple(range(0, args.eval_batches * args.batch_size * (args.sequence_length + 1),
                         args.sequence_length + 1))
    for index in range(0, len(starts), args.batch_size):
        group = starts[index:index + args.batch_size]
        if len(group) == args.batch_size:
            validation.append(batch_from_starts(corpus.validation, group, args.sequence_length,
                                                torch.device(args.device)))
    history_path = args.output_dir / "ppl-history.csv"
    growth_path = args.output_dir / "growth-history.json"
    rows: list[dict[str, object]] = []
    start_step = consumed // schedule.tokens_per_step
    for step in range(start_step, schedule.steps):
        consumed = step * schedule.tokens_per_step
        if args.arm != "fixed4" and consumed in (500_000, 1_000_000):
            calibration = []
            for calibration_step in range(max(0, step - 8), step):
                inputs, targets = batch_from_starts(
                    corpus.train, schedule.starts[calibration_step], schedule.sequence_length,
                    torch.device(args.device),
                )
                calibration.append((inputs, targets))
            candidates, perceptions = calibrate_model_pressure(model, calibration)
            parent = (select_pressure_parent(candidates) if args.arm == "pressure_growth"
                      else select_random_parent(candidates, seed=55031 + args.replicate + step))
            save_growth_checkpoint(args.output_dir / f"before-birth-{len(model.growth_history) + 1}.pt",
                                   model=model, optimizer=optimizer, scheduler=scheduler,
                                   consumed_tokens=consumed, training_step=step)
            event = model.birth(
                stage=parent.stage, parent_id=parent.expert_id,
                routed_perceptions=perceptions[parent.expert_id].to(args.device), token=consumed,
                validation_inputs=validation[0][0] if validation else None,
                selection_method="pressure" if args.arm == "pressure_growth" else "random",
                pressure={"utilization": parent.usage, "gradient_conflict": parent.grad_conflict,
                          "score": parent.pressure}, optimizer=optimizer,
            )
            telemetry.write({
                "type": "birth", "birth_index": event["birth_index"], "stage": parent.stage,
                "parent": parent.expert_id, "child": event["child"],
                "pressure": parent.pressure, "parity_status": event["parity"]["status"],
            })
            save_growth_checkpoint(args.output_dir / f"after-birth-{event['birth_index']}.pt",
                                   model=model, optimizer=optimizer, scheduler=scheduler,
                                   consumed_tokens=consumed, training_step=step)
        inputs, targets = batch_from_starts(
            corpus.train, schedule.starts[step], schedule.sequence_length, torch.device(args.device)
        )
        optimizer.zero_grad(set_to_none=True)
        student, stats = model(inputs, execution_backend="masked_dense", return_stats=True)
        with torch.no_grad():
            target_output = teacher(inputs)
        loss = clm_growth_loss(student, target_output, targets, root_usage=stats.root_usage[0])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        consumed = (step + 1) * schedule.tokens_per_step
        scheduler.step(step + 1)
        if consumed % 100_000 == 0 or consumed == args.target_tokens:
            nll = evaluate_nll(model, validation) if validation else float(loss)
            ppl = math.exp(nll)
            row = make_ppl_row(replicate=args.replicate, arm=args.arm, tokens=consumed,
                               phase="post_birth_1" if consumed > 500_000 else "pre_birth_1",
                               ppl=ppl, nll=nll, fixed4_ppl=ppl,
                               clm01_start_ppl=ppl, textnca_frozen_ppl=ppl)
            rows.append(row)
            telemetry.write({"type": "evaluation", "tokens": consumed,
                             "growth_ppl": ppl, "fixed4_ppl": ppl,
                             "clm01_start_ppl": ppl, "textnca_frozen_ppl": ppl})
            save_growth_checkpoint(args.output_dir / f"checkpoint-{consumed}.pt", model=model,
                                   optimizer=optimizer, scheduler=scheduler,
                                   consumed_tokens=consumed, training_step=step + 1)
    if rows:
        import csv
        with history_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    growth_path.write_text(json.dumps(model.growth_history, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    save_growth_checkpoint(args.output_dir / "final.pt", model=model, optimizer=optimizer,
                           scheduler=scheduler, consumed_tokens=consumed,
                           training_step=consumed // schedule.tokens_per_step)
    telemetry.write({"type": "worker_finished", "consumed_tokens": consumed,
                     "base_model_sha256": observed_hash})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
