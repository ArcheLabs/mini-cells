#!/usr/bin/env python3
"""Run one resumable CLM-0.3b marginal-growth worker."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from minicells.clm_growth import ProgressiveGrowthCLM, replicate_seed
from minicells.growth_checkpoint import GlobalLRScheduler, load_growth_checkpoint, verify_base_release_hash
from minicells.growth_experiment_utils import (
    Telemetry,
    checkpoint,
    git_provenance,
    load_diagnostics,
    load_ppl_history,
    persist_diagnostics,
    release_teacher,
    schedule_digest,
    seed_all,
    validation_batches,
    validation_starts,
    value_digest,
)
from minicells.growth_marginal import (
    calibrate_marginal_candidates,
    detect_saturation,
    mergeback_bootstrap_ci,
    select_marginal_parent,
    select_random_parent,
    write_marginal_table,
)
from minicells.growth_reporting import write_growth_history, write_ppl_history
from minicells.growth_validation import clm_growth_loss, evaluate_nll, make_ppl_row, newborn_causal_diagnostics
from minicells.language_clm_validation import load_experiment_006_teacher
from minicells.language_data import batch_from_starts, make_training_schedule


EXPERIMENT_FORMAT = "minicells.clm-0.3b-marginal-growth-worker.v1"
PROGRESS_INTERVAL = 25_000
EVAL_INTERVAL = 100_000
DEFAULT_MIN_SATURATION_TOKENS = 1_500_000
DEFAULT_MAX_PREBIRTH_TOKENS = 3_000_000
DEFAULT_POST_BIRTH_TOKENS = 1_000_000
DEFAULT_EVAL_BATCHES = 32
DEFAULT_CALIBRATION_BATCHES = 16
DIAGNOSTIC_AGES = (0, 100_000, 250_000, 500_000, 1_000_000)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run/resume one CLM-0.3b marginal-growth worker")
    result.add_argument("--release-dir", type=Path, default=Path("artifacts/releases/clm-0.1"))
    result.add_argument("--source-005-dir", type=Path, default=Path("artifacts/experiments/005-consumer-language-bridge"))
    result.add_argument("--textnca-checkpoint", type=Path, default=Path("artifacts/experiments/006-consumer-language-scaling/minicells-v2-10m.pt"))
    result.add_argument("--textnca-config", type=Path, default=Path("artifacts/experiments/006-consumer-language-scaling/model-configs.json"))
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--arm", choices=("fixed4", "marginal_growth", "random_growth"), required=True)
    result.add_argument("--replicate", type=int, choices=range(3), required=True)
    result.add_argument("--resume-input", type=Path)
    result.add_argument("--stop-after-tokens", type=int)
    result.add_argument("--min-saturation-tokens", type=int, default=DEFAULT_MIN_SATURATION_TOKENS)
    result.add_argument("--max-prebirth-tokens", type=int, default=DEFAULT_MAX_PREBIRTH_TOKENS)
    result.add_argument("--post-birth-tokens", type=int, default=DEFAULT_POST_BIRTH_TOKENS)
    result.add_argument("--batch-size", type=int, default=8)
    result.add_argument("--sequence-length", type=int, default=125)
    result.add_argument("--eval-batches", type=int, default=DEFAULT_EVAL_BATCHES)
    result.add_argument("--calibration-batches", type=int, default=DEFAULT_CALIBRATION_BATCHES)
    result.add_argument("--balance-weight", type=float, default=0.0)
    result.add_argument("--bootstrap-samples", type=int, default=1000)
    result.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    result.add_argument("--execute", action="store_true")
    return result


def _phase(*, saturation_token: int | None, arm: str, born: bool) -> str:
    if saturation_token is None:
        return "saturation_search"
    if arm == "fixed4":
        return "matched_control"
    if born:
        return "post_birth"
    return "birth_pending"


def _write_saturation(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _record_diagnostic(
    model: ProgressiveGrowthCLM,
    validation: list[tuple[torch.Tensor, torch.Tensor]],
    diagnostics: list[dict[str, object]],
    *,
    event: dict[str, object],
    age: int,
    seed: int,
    bootstrap_samples: int,
    output_dir: Path,
    telemetry: Telemetry,
) -> None:
    if any(
        int(row["birth_index"]) == int(event["birth_index"])
        and int(row["offset_tokens"]) == age
        for row in diagnostics
    ):
        return
    diagnostic = newborn_causal_diagnostics(
        model,
        validation,
        stage=int(event["stage"]),
        parent_id=str(event["parent"]),
        child_id=str(event["child"]),
    )
    diagnostic.update(mergeback_bootstrap_ci(
        model,
        validation,
        stage=int(event["stage"]),
        child_id=str(event["child"]),
        seed=seed + int(event["birth_index"]) * 1_000_003 + age,
        bootstrap_samples=bootstrap_samples,
    ))
    row = {"birth_index": int(event["birth_index"]), "offset_tokens": age, **diagnostic}
    diagnostics.append(row)
    persist_diagnostics(output_dir / "newborn-diagnostics.json", diagnostics)
    telemetry.write({"type": "newborn_diagnostic", **row})


def run(args: argparse.Namespace, telemetry: Telemetry, observed_hash: str) -> int:
    from minicells.language_scaling import prepare_scaling_corpus

    root = Path(".").resolve()
    provenance = git_provenance(root)
    if provenance["tracked_tree_dirty"]:
        raise RuntimeError("formal CLM-0.3b execution requires a clean tracked Git tree")
    device = torch.device(args.device)
    seed = replicate_seed(args.replicate)
    seed_all(seed)
    max_total_tokens = args.max_prebirth_tokens + args.post_birth_tokens

    train, validation_stream, _, corpus_manifest = prepare_scaling_corpus(
        Path("."),
        source_005_dir=args.source_005_dir,
        train_stream_tokens=max_total_tokens + args.sequence_length + 2,
        validation_stream_tokens=max(
            100_000,
            args.eval_batches * args.batch_size * (args.sequence_length + 1),
        ),
    )
    schedule = make_training_schedule(
        len(train),
        seed=seed,
        budget_tokens=max_total_tokens,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
    )
    validation_schedule = validation_starts(
        eval_batches=args.eval_batches,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
    )
    base_schedule_state: dict[str, object] = {
        "format": EXPERIMENT_FORMAT,
        **provenance,
        "replicate_seed": seed,
        "schedule_seed": seed,
        "schedule_sha256": schedule_digest(schedule.starts),
        "validation_schedule_sha256": value_digest(validation_schedule),
        "tokens_per_step": schedule.tokens_per_step,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "eval_batches": args.eval_batches,
        "calibration_batches": args.calibration_batches,
        "min_saturation_tokens": args.min_saturation_tokens,
        "max_prebirth_tokens": args.max_prebirth_tokens,
        "post_birth_tokens": args.post_birth_tokens,
        "balance_weight": args.balance_weight,
        "bootstrap_samples": args.bootstrap_samples,
    }
    schedule_state = dict(base_schedule_state)

    model = ProgressiveGrowthCLM.from_clm01_release(str(args.release_dir), device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    lr = lambda step: 3e-4 * min(1.0, (step + 1) / 100.0)
    scheduler = GlobalLRScheduler(optimizer, lr)

    # Construct frozen references before RNG restoration; constructors may consume RNG.
    fixed_teacher = release_teacher(args.release_dir, device)
    textnca = load_experiment_006_teacher(
        str(args.textnca_checkpoint),
        device=device,
        model_config_path=str(args.textnca_config),
    )

    consumed = 0
    start_step = 0
    saturation_token: int | None = None
    if args.resume_input:
        model, payload = load_growth_checkpoint(
            args.resume_input,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device,
        )
        consumed = int(payload["consumed_tokens"])
        start_step = int(payload["training_step"])
        saved_state = payload.get("data_schedule_state", {})
        for key, expected in base_schedule_state.items():
            observed = saved_state.get(key)
            if observed is not None and observed != expected:
                raise RuntimeError(
                    f"resume experiment semantics mismatch for {key}: {observed!r} != {expected!r}"
                )
        if saved_state.get("saturation_token") is not None:
            saturation_token = int(saved_state["saturation_token"])
            schedule_state["saturation_token"] = saturation_token
            schedule_state["saturation_detected"] = True

    validation = validation_batches(
        validation_stream,
        eval_batches=args.eval_batches,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        device=device,
    )
    clm01_ppl = math.exp(evaluate_nll(fixed_teacher, validation))
    textnca_ppl = math.exp(evaluate_nll(textnca, validation))
    telemetry.write({
        "type": "worker_started",
        "format": EXPERIMENT_FORMAT,
        "consumed_tokens": consumed,
        "max_total_tokens": max_total_tokens,
        "base_model_sha256": observed_hash,
        **provenance,
        "replicate_seed": seed,
        "schedule_sha256": base_schedule_state["schedule_sha256"],
        "validation_schedule_sha256": base_schedule_state["validation_schedule_sha256"],
        "corpus_manifest": corpus_manifest,
        "balance_weight": args.balance_weight,
    })

    evidence_source = args.resume_input.parent if args.resume_input else args.output_dir
    rows = load_ppl_history(evidence_source / "ppl-history.csv", through_tokens=consumed)
    diagnostics = load_diagnostics(
        evidence_source / "newborn-diagnostics.json",
        through_tokens=consumed,
        growth_history=model.growth_history,
    )
    write_ppl_history(args.output_dir / "ppl-history.csv", rows)
    write_growth_history(args.output_dir / "growth-history.json", model.growth_history)
    persist_diagnostics(args.output_dir / "newborn-diagnostics.json", diagnostics)

    if saturation_token is not None:
        _write_saturation(args.output_dir / "saturation.json", {
            "detected": True,
            "token": saturation_token,
            "restored_from_checkpoint": True,
            "min_saturation_tokens": args.min_saturation_tokens,
            "max_prebirth_tokens": args.max_prebirth_tokens,
        })

    started = time.time()
    latest_ppl = float(rows[-1]["ppl"]) if rows else None
    no_saturation = False

    for step in range(start_step, schedule.steps):
        final_target = saturation_token + args.post_birth_tokens if saturation_token is not None else None
        if args.stop_after_tokens is not None and consumed >= args.stop_after_tokens:
            break
        if final_target is not None and consumed >= final_target:
            break
        if saturation_token is None and consumed >= args.max_prebirth_tokens:
            no_saturation = True
            break

        # Birth exactly once at the first step after the saturation boundary.
        if saturation_token is not None and args.arm != "fixed4" and not model.growth_history:
            calibration = [
                batch_from_starts(train, schedule.starts[index], schedule.sequence_length, device)
                for index in range(max(0, step - args.calibration_batches), step)
            ]
            candidates, perceptions = calibrate_marginal_candidates(model, calibration)
            write_marginal_table(args.output_dir / "marginal-scan-1.csv", candidates)
            parent = (
                select_marginal_parent(candidates)
                if args.arm == "marginal_growth"
                else select_random_parent(candidates, seed=seed + 1)
            )
            telemetry.write({
                "type": "marginal_scan",
                "birth_index": 1,
                "saturation_token": saturation_token,
                "candidates": [
                    {"rank": rank, **item.to_row()}
                    for rank, item in enumerate(candidates, start=1)
                ],
                "selected_expert": parent.expert_id,
                "selection_method": "marginal" if args.arm == "marginal_growth" else "random",
            })
            checkpoint(
                args.output_dir / "before-birth-1.pt",
                model,
                optimizer,
                scheduler,
                consumed,
                step,
                schedule_state,
                telemetry=telemetry,
                reason="before_birth_1",
            )
            event = model.birth(
                stage=parent.stage,
                parent_id=parent.expert_id,
                routed_perceptions=perceptions[parent.expert_id].to(device),
                token=consumed,
                validation_inputs=validation[0][0],
                validation_targets=validation[0][1],
                selection_method="marginal" if args.arm == "marginal_growth" else "random",
                pressure={
                    "usage": parent.usage,
                    "gradient_disagreement": parent.gradient_disagreement,
                    "legacy_pressure": parent.legacy_pressure,
                    "fisher_per_route": parent.fisher_per_route,
                    "weight_grad_saliency": parent.weight_grad_saliency,
                    "geometry_separation": parent.geometry_separation,
                    "marginal_score": parent.marginal_score,
                },
                optimizer=optimizer,
            )
            telemetry.write({
                "type": "birth",
                "birth_index": 1,
                "actual_token": consumed,
                "stage": parent.stage,
                "parent": parent.expert_id,
                "child": event["child"],
                "parity": event["parity"],
                "parity_status": event["parity"]["status"],
                "selection_method": event["selection_method"],
            })
            write_growth_history(args.output_dir / "growth-history.json", model.growth_history)
            checkpoint(
                args.output_dir / "after-birth-1.pt",
                model,
                optimizer,
                scheduler,
                consumed,
                step,
                schedule_state,
                telemetry=telemetry,
                reason="after_birth_1",
            )
            _record_diagnostic(
                model,
                validation,
                diagnostics,
                event=event,
                age=0,
                seed=seed,
                bootstrap_samples=args.bootstrap_samples,
                output_dir=args.output_dir,
                telemetry=telemetry,
            )

        inputs, targets = batch_from_starts(train, schedule.starts[step], schedule.sequence_length, device)
        optimizer.zero_grad(set_to_none=True)
        student, stats = model(inputs, execution_backend="masked_dense", return_stats=True)
        with torch.no_grad():
            teacher_output = fixed_teacher(inputs)
        loss = clm_growth_loss(
            student,
            teacher_output,
            targets,
            root_usage=stats.root_usage,
            balance_weight=args.balance_weight,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        consumed = (step + 1) * schedule.tokens_per_step
        scheduler.step(step + 1)

        if consumed % EVAL_INTERVAL == 0:
            nll = evaluate_nll(model, validation)
            ppl = math.exp(nll)
            latest_ppl = ppl
            row = make_ppl_row(
                replicate=args.replicate,
                arm=args.arm,
                tokens=consumed,
                phase=_phase(
                    saturation_token=saturation_token,
                    arm=args.arm,
                    born=bool(model.growth_history),
                ),
                ppl=ppl,
                nll=nll,
                fixed4_ppl=clm01_ppl,
                clm01_start_ppl=clm01_ppl,
                textnca_frozen_ppl=textnca_ppl,
            )
            row["fixed4_ppl"] = None
            row["ppl_vs_fixed4"] = None
            rows = [existing for existing in rows if int(existing["tokens"]) != consumed]
            rows.append(row)
            rows.sort(key=lambda item: int(item["tokens"]))
            write_ppl_history(args.output_dir / "ppl-history.csv", rows)
            telemetry.write({"type": "evaluation", **row, "raw_model_ppl": ppl})

            if saturation_token is None:
                saturation = detect_saturation(rows, min_tokens=args.min_saturation_tokens)
                telemetry.write({"type": "saturation_probe", **saturation.to_dict()})
                if saturation.detected:
                    saturation_token = int(saturation.token)
                    schedule_state["saturation_token"] = saturation_token
                    schedule_state["saturation_detected"] = True
                    saturation_payload = {
                        **saturation.to_dict(),
                        "min_saturation_tokens": args.min_saturation_tokens,
                        "max_prebirth_tokens": args.max_prebirth_tokens,
                        "post_birth_tokens": args.post_birth_tokens,
                        "eval_interval": EVAL_INTERVAL,
                    }
                    _write_saturation(args.output_dir / "saturation.json", saturation_payload)
                    telemetry.write({"type": "saturation_detected", **saturation_payload})
                    checkpoint(
                        args.output_dir / f"saturation-{saturation_token}.pt",
                        model,
                        optimizer,
                        scheduler,
                        consumed,
                        step + 1,
                        schedule_state,
                        telemetry=telemetry,
                        reason="saturation_detected",
                    )

            checkpoint(
                args.output_dir / f"checkpoint-{consumed}.pt",
                model,
                optimizer,
                scheduler,
                consumed,
                step + 1,
                schedule_state,
                telemetry=telemetry,
                reason="evaluation",
            )

        if consumed % PROGRESS_INTERVAL == 0:
            elapsed = max(time.time() - started, 1e-9)
            telemetry.write({
                "type": "training_progress",
                "consumed_tokens": consumed,
                "target_tokens": (
                    saturation_token + args.post_birth_tokens
                    if saturation_token is not None
                    else args.max_prebirth_tokens
                ),
                "phase": _phase(
                    saturation_token=saturation_token,
                    arm=args.arm,
                    born=bool(model.growth_history),
                ),
                "train_loss": float(loss.detach()),
                "lr": scheduler.optimizer.param_groups[0]["lr"],
                "ppl": latest_ppl,
                "tokens_per_second": (consumed - start_step * schedule.tokens_per_step) / elapsed,
                "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
            })

        if model.growth_history:
            event = model.growth_history[0]
            age = consumed - int(event["token"])
            if age in DIAGNOSTIC_AGES:
                _record_diagnostic(
                    model,
                    validation,
                    diagnostics,
                    event=event,
                    age=age,
                    seed=seed,
                    bootstrap_samples=args.bootstrap_samples,
                    output_dir=args.output_dir,
                    telemetry=telemetry,
                )

    if saturation_token is None:
        no_saturation = True
        saturation_payload = {
            "detected": False,
            "token": None,
            "reached_max_prebirth_tokens": consumed >= args.max_prebirth_tokens,
            "consumed_tokens": consumed,
            "min_saturation_tokens": args.min_saturation_tokens,
            "max_prebirth_tokens": args.max_prebirth_tokens,
            "post_birth_tokens": args.post_birth_tokens,
        }
        _write_saturation(args.output_dir / "saturation.json", saturation_payload)
        telemetry.write({"type": "saturation_not_established", **saturation_payload})

    write_ppl_history(args.output_dir / "ppl-history.csv", rows)
    write_growth_history(args.output_dir / "growth-history.json", model.growth_history)
    persist_diagnostics(args.output_dir / "newborn-diagnostics.json", diagnostics)
    checkpoint(
        args.output_dir / "final.pt",
        model,
        optimizer,
        scheduler,
        consumed,
        consumed // schedule.tokens_per_step,
        schedule_state,
        telemetry=telemetry,
        reason="final",
    )
    expected_births = 0 if args.arm == "fixed4" else 1
    final_target = saturation_token + args.post_birth_tokens if saturation_token is not None else args.max_prebirth_tokens
    formal_eligible = bool(
        not no_saturation
        and consumed >= final_target
        and len(model.growth_history) == expected_births
        and (args.stop_after_tokens is None or consumed < args.stop_after_tokens)
    )
    telemetry.write({
        "type": "worker_complete",
        "format": EXPERIMENT_FORMAT,
        "consumed_tokens": consumed,
        "target_tokens": final_target,
        "saturation_token": saturation_token,
        "saturation_detected": saturation_token is not None,
        "completed_births": len(model.growth_history),
        "base_model_sha256": observed_hash,
        "formal_eligible": formal_eligible,
        **provenance,
    })
    return 0


def main() -> int:
    args = parser().parse_args()
    observed = verify_base_release_hash(args.release_dir / "model.pt")
    telemetry = Telemetry(args.output_dir, args.arm, args.replicate)
    try:
        if not args.execute:
            telemetry.write({
                "type": "worker_started",
                "consumed_tokens": 0,
                "max_total_tokens": args.max_prebirth_tokens + args.post_birth_tokens,
                "base_model_sha256": observed,
                "mode": "preflight_only",
            })
            telemetry.write({
                "type": "training_progress",
                "consumed_tokens": 0,
                "target_tokens": args.max_prebirth_tokens,
                "phase": "preflight_only",
                "train_loss": None,
                "lr": None,
                "ppl": None,
                "tokens_per_second": 0.0,
                "peak_vram_bytes": 0,
            })
            telemetry.write({
                "type": "worker_complete",
                "consumed_tokens": 0,
                "target_tokens": args.max_prebirth_tokens,
                "formal_eligible": False,
                "mode": "preflight_only",
            })
            return 0
        return run(args, telemetry, observed)
    finally:
        telemetry.close()


if __name__ == "__main__":
    raise SystemExit(main())
