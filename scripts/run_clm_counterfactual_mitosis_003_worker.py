#!/usr/bin/env python3
"""Run one formal CLM-0.3c counterfactual-mitosis replicate."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from minicells.clm_growth import ProgressiveGrowthCLM, replicate_seed
from minicells.growth_checkpoint import GlobalLRScheduler, load_growth_checkpoint, verify_base_release_hash
from minicells.growth_counterfactual import (
    SplitRegretCandidate,
    calibrate_split_regret,
    paired_bootstrap_utility,
    select_counterfactual_action,
    spearman_rank_correlation,
    write_split_regret_table,
)
from minicells.growth_experiment_utils import (
    Telemetry,
    checkpoint,
    git_provenance,
    release_teacher,
    schedule_digest,
    seed_all,
    validation_starts,
    value_digest,
)
from minicells.growth_validation import clm_growth_loss
from minicells.language_data import batch_from_starts, make_training_schedule


EXPERIMENT_FORMAT = "minicells.clm-0.3c-counterfactual-worker.v1"
DEFAULT_DECISION_TOKENS = 1_500_000
DEFAULT_PROBE_TOKENS = 100_000
DEFAULT_CONFIRM_TOKENS = 500_000
DEFAULT_EVAL_BATCHES = 32
DEFAULT_CALIBRATION_BATCHES = 16
DEFAULT_BOOTSTRAP_SAMPLES = 2_000
EVAL_INTERVAL = 100_000
PROGRESS_INTERVAL = 25_000


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run one CLM-0.3c counterfactual replicate")
    result.add_argument("--release-dir", type=Path, default=Path("artifacts/releases/clm-0.1"))
    result.add_argument("--source-005-dir", type=Path, default=Path("artifacts/experiments/005-consumer-language-bridge"))
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--replicate", type=int, choices=range(3), required=True)
    result.add_argument("--decision-tokens", type=int, default=DEFAULT_DECISION_TOKENS)
    result.add_argument("--probe-tokens", type=int, default=DEFAULT_PROBE_TOKENS)
    result.add_argument("--confirm-tokens", type=int, default=DEFAULT_CONFIRM_TOKENS)
    result.add_argument("--batch-size", type=int, default=8)
    result.add_argument("--sequence-length", type=int, default=125)
    result.add_argument("--eval-batches", type=int, default=DEFAULT_EVAL_BATCHES)
    result.add_argument("--calibration-batches", type=int, default=DEFAULT_CALIBRATION_BATCHES)
    result.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    result.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    result.add_argument("--execute", action="store_true")
    return result


def _json_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _json_read(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _checkpoint_commit(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    state = payload.get("data_schedule_state", {}) if isinstance(payload, dict) else {}
    return str(state.get("code_commit")) if state.get("code_commit") else None


def _latest_trunk_checkpoint(output_dir: Path, code_commit: str, decision_tokens: int) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for path in output_dir.glob("trunk-checkpoint-*.pt"):
        if _checkpoint_commit(path) != code_commit:
            continue
        try:
            token = int(path.stem.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            continue
        if 0 <= token <= decision_tokens:
            candidates.append((token, path))
    decision = output_dir / "decision-trunk.pt"
    if decision.exists() and _checkpoint_commit(decision) == code_commit:
        candidates.append((decision_tokens, decision))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _batches_from_starts(
    stream: torch.Tensor,
    starts: tuple[int, ...],
    *,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [
        batch_from_starts(stream, starts[index:index + batch_size], sequence_length, device)
        for index in range(0, len(starts), batch_size)
    ]


def _batch_nlls(model: ProgressiveGrowthCLM, validation: list[tuple[torch.Tensor, torch.Tensor]]) -> list[float]:
    was_training = model.training
    model.eval()
    rows: list[float] = []
    try:
        with torch.no_grad():
            for inputs, targets in validation:
                output = model(inputs, execution_backend="sparse_dispatch")
                rows.append(float(F.cross_entropy(output.logits.flatten(0, 1), targets.reshape(-1))))
    finally:
        if was_training:
            model.train()
    return rows


def _fresh_optimizer(model: ProgressiveGrowthCLM) -> tuple[torch.optim.AdamW, GlobalLRScheduler]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    lr = lambda step: 3e-4 * min(1.0, (step + 1) / 100.0)
    return optimizer, GlobalLRScheduler(optimizer, lr)


def _restore_branch(
    release_dir: Path,
    trunk: Path,
    device: torch.device,
) -> tuple[ProgressiveGrowthCLM, torch.optim.AdamW, GlobalLRScheduler, dict[str, Any]]:
    model = ProgressiveGrowthCLM.from_clm01_release(str(release_dir), device=device)
    optimizer, scheduler = _fresh_optimizer(model)
    model, payload = load_growth_checkpoint(
        trunk,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        map_location=device,
    )
    return model, optimizer, scheduler, payload


def _train_segment(
    model: ProgressiveGrowthCLM,
    optimizer: torch.optim.Optimizer,
    scheduler: GlobalLRScheduler,
    teacher: torch.nn.Module,
    train: torch.Tensor,
    schedule: Any,
    *,
    start_step: int,
    end_step: int,
    device: torch.device,
    telemetry: Telemetry,
    phase: str,
) -> None:
    started = time.time()
    start_tokens = start_step * schedule.tokens_per_step
    for step in range(start_step, end_step):
        inputs, targets = batch_from_starts(train, schedule.starts[step], schedule.sequence_length, device)
        optimizer.zero_grad(set_to_none=True)
        student, stats = model(inputs, execution_backend="masked_dense", return_stats=True)
        with torch.no_grad():
            teacher_output = teacher(inputs)
        loss = clm_growth_loss(
            student,
            teacher_output,
            targets,
            root_usage=stats.root_usage,
            balance_weight=0.0,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step(step + 1)
        consumed = (step + 1) * schedule.tokens_per_step
        if consumed % PROGRESS_INTERVAL == 0:
            elapsed = max(time.time() - started, 1e-9)
            telemetry.write({
                "type": "training_progress",
                "consumed_tokens": consumed,
                "target_tokens": end_step * schedule.tokens_per_step,
                "phase": phase,
                "train_loss": float(loss.detach()),
                "lr": scheduler.optimizer.param_groups[0]["lr"],
                "tokens_per_second": (consumed - start_tokens) / elapsed,
                "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
            })


def _candidate_payload(candidate: SplitRegretCandidate, rank: int) -> dict[str, object]:
    return {"analytic_rank": rank, **candidate.to_row()}


def run(args: argparse.Namespace, telemetry: Telemetry, observed_hash: str) -> int:
    from minicells.language_scaling import prepare_scaling_corpus

    root = Path(".").resolve()
    provenance = git_provenance(root)
    if provenance["tracked_tree_dirty"]:
        raise RuntimeError("formal CLM-0.3c execution requires a clean tracked Git tree")
    code_commit = str(provenance["code_commit"])
    device = torch.device(args.device)
    seed = replicate_seed(args.replicate)
    seed_all(seed)

    tokens_per_step = args.batch_size * args.sequence_length
    if args.decision_tokens % tokens_per_step != 0:
        raise ValueError("decision token boundary must align with a training step")
    if args.probe_tokens % tokens_per_step != 0 or args.confirm_tokens % tokens_per_step != 0:
        raise ValueError("probe/confirmation horizons must align with a training step")
    if args.probe_tokens <= 0 or args.confirm_tokens < args.probe_tokens:
        raise ValueError("probe/confirmation horizons are invalid")

    max_total = args.decision_tokens + args.confirm_tokens
    holdout_targets = 2 * args.eval_batches * args.batch_size * (args.sequence_length + 1)
    train, validation_stream, _, corpus_manifest = prepare_scaling_corpus(
        Path("."),
        source_005_dir=args.source_005_dir,
        train_stream_tokens=max_total + args.sequence_length + 2,
        validation_stream_tokens=max(100_000, holdout_targets),
    )
    schedule = make_training_schedule(
        len(train),
        seed=seed,
        budget_tokens=max_total,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
    )
    probe_validation_starts = validation_starts(
        eval_batches=args.eval_batches,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
    )
    holdout_offset = args.eval_batches * args.batch_size * (args.sequence_length + 1)
    confirm_validation_starts = tuple(holdout_offset + value for value in probe_validation_starts)
    schedule_state: dict[str, object] = {
        "format": EXPERIMENT_FORMAT,
        **provenance,
        "replicate_seed": seed,
        "schedule_seed": seed,
        "schedule_sha256": schedule_digest(schedule.starts),
        "probe_validation_schedule_sha256": value_digest(probe_validation_starts),
        "confirm_validation_schedule_sha256": value_digest(confirm_validation_starts),
        "tokens_per_step": schedule.tokens_per_step,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "eval_batches": args.eval_batches,
        "calibration_batches": args.calibration_batches,
        "decision_tokens": args.decision_tokens,
        "probe_tokens": args.probe_tokens,
        "confirm_tokens": args.confirm_tokens,
        "bootstrap_samples": args.bootstrap_samples,
        "balance_weight": 0.0,
    }
    identity_path = args.output_dir / "run-provenance.json"
    existing_identity = _json_read(identity_path, None)
    formal_identity = {**schedule_state, "base_model_sha256": observed_hash}
    if existing_identity is not None and existing_identity != formal_identity:
        raise RuntimeError(
            "existing CLM-0.3c evidence uses different code or formal semantics; "
            "restart the replicate instead of mixing evidence"
        )
    _json_write(identity_path, formal_identity)

    probe_validation = _batches_from_starts(
        validation_stream,
        probe_validation_starts,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        device=device,
    )
    confirm_validation = _batches_from_starts(
        validation_stream,
        confirm_validation_starts,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        device=device,
    )
    teacher = release_teacher(args.release_dir, device)

    telemetry.write({
        "type": "worker_started",
        "format": EXPERIMENT_FORMAT,
        "base_model_sha256": observed_hash,
        "corpus_manifest": corpus_manifest,
        **schedule_state,
    })

    # Phase 1: build or resume the common decision trunk.
    trunk_path = args.output_dir / "decision-trunk.pt"
    latest = _latest_trunk_checkpoint(args.output_dir, code_commit, args.decision_tokens)
    model = ProgressiveGrowthCLM.from_clm01_release(str(args.release_dir), device=device)
    optimizer, scheduler = _fresh_optimizer(model)
    start_step = 0
    if latest is not None:
        model, payload = load_growth_checkpoint(
            latest,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device,
        )
        saved = payload.get("data_schedule_state", {})
        for key, expected in schedule_state.items():
            observed = saved.get(key)
            if observed is not None and observed != expected:
                raise RuntimeError(f"trunk resume semantics mismatch for {key}: {observed!r} != {expected!r}")
        start_step = int(payload["training_step"])
        telemetry.write({"type": "trunk_resume", "path": str(latest), "training_step": start_step})

    decision_step = args.decision_tokens // schedule.tokens_per_step
    trunk_history: list[dict[str, float | int]] = _json_read(args.output_dir / "trunk-history.json", [])
    if start_step < decision_step:
        started = time.time()
        for step in range(start_step, decision_step):
            inputs, targets = batch_from_starts(train, schedule.starts[step], schedule.sequence_length, device)
            optimizer.zero_grad(set_to_none=True)
            student, stats = model(inputs, execution_backend="masked_dense", return_stats=True)
            with torch.no_grad():
                teacher_output = teacher(inputs)
            loss = clm_growth_loss(
                student,
                teacher_output,
                targets,
                root_usage=stats.root_usage,
                balance_weight=0.0,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step(step + 1)
            consumed = (step + 1) * schedule.tokens_per_step
            if consumed % EVAL_INTERVAL == 0:
                losses = _batch_nlls(model, probe_validation)
                nll = sum(losses) / len(losses)
                trunk_history = [row for row in trunk_history if int(row["tokens"]) != consumed]
                trunk_history.append({"tokens": consumed, "nll": nll, "ppl": math.exp(nll)})
                trunk_history.sort(key=lambda row: int(row["tokens"]))
                _json_write(args.output_dir / "trunk-history.json", trunk_history)
                checkpoint(
                    args.output_dir / f"trunk-checkpoint-{consumed}.pt",
                    model,
                    optimizer,
                    scheduler,
                    consumed,
                    step + 1,
                    schedule_state,
                    telemetry=telemetry,
                    reason="trunk_evaluation",
                )
            if consumed % PROGRESS_INTERVAL == 0:
                elapsed = max(time.time() - started, 1e-9)
                telemetry.write({
                    "type": "training_progress",
                    "consumed_tokens": consumed,
                    "target_tokens": args.decision_tokens,
                    "phase": "decision_trunk",
                    "train_loss": float(loss.detach()),
                    "lr": scheduler.optimizer.param_groups[0]["lr"],
                    "tokens_per_second": (consumed - start_step * schedule.tokens_per_step) / elapsed,
                    "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                })
        checkpoint(
            trunk_path,
            model,
            optimizer,
            scheduler,
            args.decision_tokens,
            decision_step,
            schedule_state,
            telemetry=telemetry,
            reason="decision_trunk",
        )
    elif latest != trunk_path:
        checkpoint(
            trunk_path,
            model,
            optimizer,
            scheduler,
            args.decision_tokens,
            decision_step,
            schedule_state,
            telemetry=telemetry,
            reason="decision_trunk_normalized",
        )

    # Reload the immutable trunk before calibration so resume history cannot perturb the scan.
    model, optimizer, scheduler, _ = _restore_branch(args.release_dir, trunk_path, device)
    calibration = [
        batch_from_starts(train, schedule.starts[index], schedule.sequence_length, device)
        for index in range(decision_step - args.calibration_batches, decision_step)
    ]
    candidates, perceptions, _prototypes = calibrate_split_regret(model, optimizer, calibration)
    eligible = [item for item in candidates if item.eligible]
    if len(eligible) != 12:
        raise RuntimeError(f"formal CLM-0.3c requires all 12 root lineages eligible; observed {len(eligible)}")
    write_split_regret_table(args.output_dir / "split-regret.csv", candidates)
    analytic_rank = {item.expert_id: rank for rank, item in enumerate(candidates, start=1)}
    candidate_by_id = {item.expert_id: item for item in candidates}
    telemetry.write({
        "type": "split_regret_scan",
        "candidates": [_candidate_payload(item, rank) for rank, item in enumerate(candidates, start=1)],
    })

    # Phase 2: common no-growth 100K probe, evaluated only on holdout A.
    probe_step = decision_step + args.probe_tokens // schedule.tokens_per_step
    control_probe_path = args.output_dir / "probe-control.json"
    control_probe = _json_read(control_probe_path, None)
    if control_probe is None:
        branch, branch_optimizer, branch_scheduler, _ = _restore_branch(args.release_dir, trunk_path, device)
        _train_segment(
            branch, branch_optimizer, branch_scheduler, teacher, train, schedule,
            start_step=decision_step, end_step=probe_step, device=device,
            telemetry=telemetry, phase="probe_no_growth",
        )
        batch_losses = _batch_nlls(branch, probe_validation)
        control_probe = {
            "code_commit": code_commit,
            "validation_schedule_sha256": schedule_state["probe_validation_schedule_sha256"],
            "batch_nlls": batch_losses,
            "nll": sum(batch_losses) / len(batch_losses),
            "ppl": math.exp(sum(batch_losses) / len(batch_losses)),
        }
        _json_write(control_probe_path, control_probe)

    # Phase 3: probe every lineage from the same trunk. Each candidate is durable and replayable.
    probe_dir = args.output_dir / "probes"
    probe_dir.mkdir(parents=True, exist_ok=True)
    probe_rows: list[dict[str, object]] = []
    parity_rows: list[dict[str, object]] = []
    for candidate in candidates:
        result_path = probe_dir / f"{candidate.expert_id}.json"
        existing = _json_read(result_path, None)
        if existing is not None:
            if existing.get("code_commit") != code_commit:
                raise RuntimeError(f"stale probe evidence for {candidate.expert_id}")
            probe_rows.append(existing["probe"])
            parity_rows.append(existing["parity"])
            continue
        branch, branch_optimizer, branch_scheduler, _ = _restore_branch(args.release_dir, trunk_path, device)
        event = branch.birth(
            stage=candidate.stage,
            parent_id=candidate.expert_id,
            routed_perceptions=perceptions[candidate.expert_id].to(device),
            token=args.decision_tokens,
            validation_inputs=probe_validation[0][0],
            validation_targets=probe_validation[0][1],
            selection_method="counterfactual_probe",
            pressure={
                "split_regret": candidate.split_regret,
                "adam_metric_disagreement": candidate.adam_metric_disagreement,
                "usage": candidate.usage,
                "pi0": candidate.pi0,
                "pi1": candidate.pi1,
            },
            optimizer=branch_optimizer,
        )
        _train_segment(
            branch, branch_optimizer, branch_scheduler, teacher, train, schedule,
            start_step=decision_step, end_step=probe_step, device=device,
            telemetry=telemetry, phase=f"probe_{candidate.expert_id}",
        )
        batch_losses = _batch_nlls(branch, probe_validation)
        utility = paired_bootstrap_utility(
            control_probe["batch_nlls"],
            batch_losses,
            seed=seed + 100_003 * (analytic_rank[candidate.expert_id] + 1),
            bootstrap_samples=args.bootstrap_samples,
        )
        probe = {
            "expert_id": candidate.expert_id,
            "stage": candidate.stage,
            "analytic_rank": analytic_rank[candidate.expert_id],
            "split_regret": candidate.split_regret,
            "candidate_nll": utility.candidate_nll,
            "candidate_ppl": math.exp(utility.candidate_nll),
            **utility.to_dict(),
        }
        parity = {
            "expert_id": candidate.expert_id,
            "stage": candidate.stage,
            "parity": event["parity"],
        }
        _json_write(result_path, {
            "code_commit": code_commit,
            "probe_validation_schedule_sha256": schedule_state["probe_validation_schedule_sha256"],
            "probe": probe,
            "parity": parity,
            "birth": event,
        })
        probe_rows.append(probe)
        parity_rows.append(parity)
        telemetry.write({"type": "counterfactual_probe", **probe})

    probe_rows.sort(key=lambda row: int(row["analytic_rank"]))
    parity_rows.sort(key=lambda row: analytic_rank[str(row["expert_id"])])
    _json_write(args.output_dir / "probe-results.json", probe_rows)
    _json_write(args.output_dir / "growth-equivalence.json", parity_rows)

    rho = spearman_rank_correlation(
        [float(row["split_regret"]) for row in probe_rows],
        [float(row["relative_improvement"]) for row in probe_rows],
    )
    policy = select_counterfactual_action(probe_rows)
    policy.update({
        "spearman_split_regret_vs_probe_utility": rho,
        "decision_tokens": args.decision_tokens,
        "probe_tokens": args.probe_tokens,
        "probe_validation_schedule_sha256": schedule_state["probe_validation_schedule_sha256"],
    })
    _json_write(args.output_dir / "policy-decision.json", policy)
    telemetry.write({"type": "policy_decision", **policy})

    # Phase 4: independent-holdout 500K confirmation from the original trunk.
    confirm_step = decision_step + args.confirm_tokens // schedule.tokens_per_step
    confirm_control_path = args.output_dir / "confirm-control.json"
    confirm_control = _json_read(confirm_control_path, None)
    if confirm_control is None:
        branch, branch_optimizer, branch_scheduler, _ = _restore_branch(args.release_dir, trunk_path, device)
        _train_segment(
            branch, branch_optimizer, branch_scheduler, teacher, train, schedule,
            start_step=decision_step, end_step=confirm_step, device=device,
            telemetry=telemetry, phase="confirm_no_growth",
        )
        batch_losses = _batch_nlls(branch, confirm_validation)
        confirm_control = {
            "code_commit": code_commit,
            "validation_schedule_sha256": schedule_state["confirm_validation_schedule_sha256"],
            "batch_nlls": batch_losses,
            "nll": sum(batch_losses) / len(batch_losses),
            "ppl": math.exp(sum(batch_losses) / len(batch_losses)),
        }
        _json_write(confirm_control_path, confirm_control)

    selected_id = str(policy["selected_expert"])
    selected = candidate_by_id[selected_id]
    confirm_candidate_path = args.output_dir / "confirm-candidate.json"
    confirm_candidate = _json_read(confirm_candidate_path, None)
    if confirm_candidate is None or confirm_candidate.get("expert_id") != selected_id:
        branch, branch_optimizer, branch_scheduler, _ = _restore_branch(args.release_dir, trunk_path, device)
        event = branch.birth(
            stage=selected.stage,
            parent_id=selected.expert_id,
            routed_perceptions=perceptions[selected.expert_id].to(device),
            token=args.decision_tokens,
            validation_inputs=confirm_validation[0][0],
            validation_targets=confirm_validation[0][1],
            selection_method="counterfactual_confirm",
            pressure={
                "split_regret": selected.split_regret,
                "adam_metric_disagreement": selected.adam_metric_disagreement,
                "usage": selected.usage,
                "pi0": selected.pi0,
                "pi1": selected.pi1,
            },
            optimizer=branch_optimizer,
        )
        _train_segment(
            branch, branch_optimizer, branch_scheduler, teacher, train, schedule,
            start_step=decision_step, end_step=confirm_step, device=device,
            telemetry=telemetry, phase=f"confirm_{selected_id}",
        )
        batch_losses = _batch_nlls(branch, confirm_validation)
        utility = paired_bootstrap_utility(
            confirm_control["batch_nlls"],
            batch_losses,
            seed=seed + 9_000_001,
            bootstrap_samples=args.bootstrap_samples,
        )
        confirm_candidate = {
            "code_commit": code_commit,
            "confirm_validation_schedule_sha256": schedule_state["confirm_validation_schedule_sha256"],
            "expert_id": selected_id,
            "stage": selected.stage,
            "analytic_rank": analytic_rank[selected_id],
            "split_regret": selected.split_regret,
            "control_ppl": float(confirm_control["ppl"]),
            "candidate_ppl": math.exp(utility.candidate_nll),
            "ppl_ratio": math.exp(utility.candidate_nll) / float(confirm_control["ppl"]),
            "practical_improvement": 1.0 - math.exp(utility.candidate_nll) / float(confirm_control["ppl"]),
            "parity": event["parity"],
            **utility.to_dict(),
        }
        _json_write(confirm_candidate_path, confirm_candidate)
    elif confirm_candidate.get("code_commit") != code_commit:
        raise RuntimeError("stale confirmation evidence from a different commit")

    action = str(policy["action"])
    calibrated = bool(
        (action == "GROW" and float(confirm_candidate["ci95_low"]) > 0.0)
        or (action == "NO_GROW" and float(confirm_candidate["ci95_high"]) <= 0.0)
    )
    inconclusive = bool(
        not calibrated
        and float(confirm_candidate["ci95_low"]) <= 0.0 < float(confirm_candidate["ci95_high"])
    )
    final = {
        "format": "minicells.clm-0.3c-counterfactual-replicate.v1",
        "replicate": args.replicate,
        "policy": policy,
        "confirm": confirm_candidate,
        "decision_calibrated": calibrated,
        "decision_inconclusive": inconclusive,
        "confirmed_positive_capacity_value": float(confirm_candidate["ci95_low"]) > 0.0,
        "practical_growth_pass": float(confirm_candidate["ppl_ratio"]) <= 0.995,
        "births_checked": len(parity_rows) + 1,
        "births_equivalent": sum(
            row["parity"].get("status") == "CLM_GROWTH_EQUIVALENCE" for row in parity_rows
        ) + int(confirm_candidate["parity"].get("status") == "CLM_GROWTH_EQUIVALENCE"),
        **provenance,
    }
    _json_write(args.output_dir / "replicate-result.json", final)
    telemetry.write({
        "type": "worker_complete",
        "format": EXPERIMENT_FORMAT,
        "consumed_tokens": args.decision_tokens + args.confirm_tokens,
        "target_tokens": args.decision_tokens + args.confirm_tokens,
        "formal_eligible": True,
        **provenance,
        "decision_calibrated": calibrated,
        "confirmed_positive_capacity_value": final["confirmed_positive_capacity_value"],
    })
    return 0


def main() -> int:
    args = parser().parse_args()
    observed = verify_base_release_hash(args.release_dir / "model.pt")
    telemetry = Telemetry(args.output_dir, "counterfactual", args.replicate)
    try:
        if not args.execute:
            telemetry.write({
                "type": "worker_started",
                "mode": "preflight_only",
                "base_model_sha256": observed,
            })
            telemetry.write({
                "type": "training_progress",
                "consumed_tokens": 0,
                "target_tokens": args.decision_tokens,
                "phase": "preflight_only",
                "train_loss": None,
                "lr": None,
                "tokens_per_second": 0.0,
                "peak_vram_bytes": 0,
            })
            telemetry.write({
                "type": "worker_complete",
                "mode": "preflight_only",
                "consumed_tokens": 0,
                "target_tokens": args.decision_tokens,
                "formal_eligible": False,
            })
            return 0
        return run(args, telemetry, observed)
    finally:
        telemetry.close()


if __name__ == "__main__":
    raise SystemExit(main())
