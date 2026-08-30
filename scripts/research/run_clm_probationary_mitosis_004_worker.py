#!/usr/bin/env python3
"""Run one formal CLM-0.3d probationary-mitosis replicate."""

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
from minicells.growth_counterfactual import calibrate_split_regret, paired_bootstrap_utility
from minicells.growth_experiment_utils import (
    Telemetry,
    checkpoint,
    git_provenance,
    release_teacher,
    schedule_digest,
    seed_all,
    value_digest,
)
from minicells.growth_probationary import (
    FORMAL_CONDITIONS,
    FORMAL_HORIZONS,
    PRACTICAL_PPL_RATIO_THRESHOLD,
    SHORTLIST_HORIZON,
    SHORTLIST_K,
    STORY_RETENTION_RATIO_THRESHOLD,
    ProbationPoint,
    absorption_diagnostic,
    condition_domains,
    independent_confirmation,
    maturation_rescue,
    select_promotion_candidate,
    shortlist_candidates,
    summarize_probation,
)
from minicells.growth_validation import clm_growth_loss
from minicells.language_conflict_differentiation import deterministic_starts, prepare_arithmetic_cache
from minicells.language_data import batch_from_starts, make_training_schedule


EXPERIMENT_FORMAT = "minicells.clm-0.3d-probationary-worker.v1"
DEFAULT_DECISION_TOKENS = 1_500_000
DEFAULT_EVAL_BATCHES = 32
DEFAULT_CALIBRATION_BATCHES = 16
DEFAULT_BOOTSTRAP_SAMPLES = 2_000
PROGRESS_INTERVAL = 25_000
TRUNK_CHECKPOINT_INTERVAL = 100_000


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run one CLM-0.3d probationary-mitosis replicate")
    result.add_argument("--release-dir", type=Path, default=Path("artifacts/releases/clm-0.1"))
    result.add_argument(
        "--source-005-dir",
        type=Path,
        default=Path("artifacts/experiments/005-consumer-language-bridge"),
    )
    result.add_argument("--cache-dir", type=Path, default=Path("results/.clm-0.3d-cache"))
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--replicate", type=int, choices=range(3), required=True)
    result.add_argument("--decision-tokens", type=int, default=DEFAULT_DECISION_TOKENS)
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


def _fresh_optimizer(model: ProgressiveGrowthCLM) -> tuple[torch.optim.AdamW, GlobalLRScheduler]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    lr = lambda step: 3e-4 * min(1.0, (step + 1) / 100.0)
    return optimizer, GlobalLRScheduler(optimizer, lr)


def _restore_branch(
    release_dir: Path,
    path: Path,
    device: torch.device,
) -> tuple[ProgressiveGrowthCLM, torch.optim.AdamW, GlobalLRScheduler, dict[str, Any]]:
    model = ProgressiveGrowthCLM.from_clm01_release(str(release_dir), device=device)
    optimizer, scheduler = _fresh_optimizer(model)
    model, payload = load_growth_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        map_location=device,
    )
    return model, optimizer, scheduler, payload


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty list")
    return float(sum(values) / len(values))


def _bundle_from_refs(
    refs: list[dict[str, object]],
    streams: dict[str, torch.Tensor],
    *,
    sequence_length: int,
    device: torch.device,
) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    result = []
    for row in refs:
        domain = str(row["domain"])
        starts = tuple(int(value) for value in row["starts"])
        inputs, targets = batch_from_starts(
            streams[domain], starts, sequence_length, device
        )
        result.append((domain, inputs, targets))
    return result


def _validation_refs(
    condition: str,
    *,
    holdout: str,
    eval_batches: int,
    batch_size: int,
    sequence_length: int,
) -> list[dict[str, object]]:
    if holdout not in ("A", "B"):
        raise ValueError("holdout must be A or B")
    if condition == "stationary_story":
        counts = {"story": eval_batches}
    elif condition == "story_arithmetic_shift":
        if eval_batches % 2:
            raise ValueError("shift evaluation requires an even number of batches")
        counts = {"story": eval_batches // 2, "arithmetic": eval_batches // 2}
    else:
        raise ValueError(condition)
    width = sequence_length + 1
    refs: list[dict[str, object]] = []
    for domain in ("story", "arithmetic"):
        count = counts.get(domain, 0)
        if not count:
            continue
        sequences_per_holdout = count * batch_size
        base_sequence = 0 if holdout == "A" else sequences_per_holdout
        for batch_index in range(count):
            sequence_index = base_sequence + batch_index * batch_size
            starts = tuple((sequence_index + offset) * width for offset in range(batch_size))
            refs.append({"domain": domain, "starts": starts})
    return refs


def _parity_ref(*, batch_size: int, sequence_length: int) -> list[dict[str, object]]:
    width = sequence_length + 1
    base_sequence = 600
    starts = tuple((base_sequence + offset) * width for offset in range(batch_size))
    return [{"domain": "story", "starts": starts}]


def _evaluate(
    model: ProgressiveGrowthCLM,
    bundle: list[tuple[str, torch.Tensor, torch.Tensor]],
) -> dict[str, object]:
    was_training = model.training
    model.eval()
    all_rows: list[float] = []
    by_domain: dict[str, list[float]] = {}
    try:
        with torch.no_grad():
            for domain, inputs, targets in bundle:
                output = model(inputs, execution_backend="sparse_dispatch")
                value = float(
                    F.cross_entropy(output.logits.flatten(0, 1), targets.reshape(-1))
                )
                all_rows.append(value)
                by_domain.setdefault(domain, []).append(value)
    finally:
        if was_training:
            model.train()
    nll = _mean(all_rows)
    return {
        "batch_nlls": all_rows,
        "nll": nll,
        "ppl": math.exp(nll),
        "domain_batch_nlls": by_domain,
        "domain_nll": {key: _mean(values) for key, values in by_domain.items()},
    }


def _future_refs(
    condition: str,
    streams: dict[str, torch.Tensor],
    *,
    steps: int,
    batch_size: int,
    sequence_length: int,
    seed: int,
) -> list[dict[str, object]]:
    domains = condition_domains(condition, steps=steps, seed=seed)
    story_starts = deterministic_starts(
        len(streams["story"]),
        steps=steps,
        batch_size=batch_size,
        sequence_length=sequence_length,
        seed=seed + 1,
    )
    arithmetic_starts = deterministic_starts(
        len(streams["arithmetic"]),
        steps=steps,
        batch_size=batch_size,
        sequence_length=sequence_length,
        seed=seed + 2,
    )
    return [
        {
            "domain": domain,
            "starts": story_starts[index] if domain == "story" else arithmetic_starts[index],
        }
        for index, domain in enumerate(domains)
    ]


def _train_trunk(
    model: ProgressiveGrowthCLM,
    optimizer: torch.optim.Optimizer,
    scheduler: GlobalLRScheduler,
    teacher: torch.nn.Module,
    train: torch.Tensor,
    schedule: Any,
    *,
    start_step: int,
    decision_step: int,
    schedule_state: dict[str, object],
    output_dir: Path,
    device: torch.device,
    telemetry: Telemetry,
) -> list[dict[str, object]]:
    history: list[dict[str, object]] = _json_read(output_dir / "trunk-history.json", [])
    started = time.time()
    start_tokens = start_step * schedule.tokens_per_step
    for step in range(start_step, decision_step):
        inputs, targets = batch_from_starts(
            train, schedule.starts[step], schedule.sequence_length, device
        )
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
        if consumed % TRUNK_CHECKPOINT_INTERVAL == 0:
            history = [row for row in history if int(row["tokens"]) != consumed]
            history.append({"tokens": consumed, "train_nll": float(loss.detach())})
            history.sort(key=lambda row: int(row["tokens"]))
            _json_write(output_dir / "trunk-history.json", history)
            checkpoint(
                output_dir / f"trunk-checkpoint-{consumed}.pt",
                model,
                optimizer,
                scheduler,
                consumed,
                step + 1,
                schedule_state,
                telemetry=telemetry,
                reason="trunk_periodic",
            )
        if consumed % PROGRESS_INTERVAL == 0:
            elapsed = max(time.time() - started, 1e-9)
            telemetry.write({
                "type": "training_progress",
                "consumed_tokens": consumed,
                "target_tokens": decision_step * schedule.tokens_per_step,
                "phase": "decision_trunk",
                "train_loss": float(loss.detach()),
                "lr": scheduler.optimizer.param_groups[0]["lr"],
                "tokens_per_second": (consumed - start_tokens) / elapsed,
                "peak_vram_bytes": (
                    torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
                ),
            })
    return history


def _train_future(
    model: ProgressiveGrowthCLM,
    optimizer: torch.optim.Optimizer,
    scheduler: GlobalLRScheduler,
    teacher: torch.nn.Module,
    streams: dict[str, torch.Tensor],
    refs: list[dict[str, object]],
    *,
    start_age_tokens: int,
    end_age_tokens: int,
    decision_tokens: int,
    tokens_per_step: int,
    sequence_length: int,
    device: torch.device,
    telemetry: Telemetry,
    phase: str,
) -> None:
    start_local = start_age_tokens // tokens_per_step
    end_local = end_age_tokens // tokens_per_step
    decision_step = decision_tokens // tokens_per_step
    started = time.time()
    for local_step in range(start_local, end_local):
        row = refs[local_step]
        domain = str(row["domain"])
        inputs, targets = batch_from_starts(
            streams[domain],
            tuple(int(value) for value in row["starts"]),
            sequence_length,
            device,
        )
        optimizer.zero_grad(set_to_none=True)
        student, stats = model(inputs, execution_backend="masked_dense", return_stats=True)
        if domain == "story":
            with torch.no_grad():
                teacher_output = teacher(inputs)
            loss = clm_growth_loss(
                student,
                teacher_output,
                targets,
                root_usage=stats.root_usage,
                balance_weight=0.0,
            )
        else:
            loss = F.cross_entropy(student.logits.flatten(0, 1), targets.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step(decision_step + local_step + 1)
        age = (local_step + 1) * tokens_per_step
        if age % PROGRESS_INTERVAL == 0:
            elapsed = max(time.time() - started, 1e-9)
            telemetry.write({
                "type": "training_progress",
                "consumed_tokens": decision_tokens + age,
                "target_tokens": decision_tokens + end_age_tokens,
                "phase": phase,
                "train_loss": float(loss.detach()),
                "lr": scheduler.optimizer.param_groups[0]["lr"],
                "tokens_per_second": (age - start_age_tokens) / elapsed,
                "peak_vram_bytes": (
                    torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
                ),
            })


def _point(
    *,
    tokens: int,
    control: dict[str, object],
    candidate: dict[str, object],
    seed: int,
    bootstrap_samples: int,
) -> ProbationPoint:
    utility = paired_bootstrap_utility(
        control["batch_nlls"],
        candidate["batch_nlls"],
        seed=seed,
        bootstrap_samples=bootstrap_samples,
    )
    return ProbationPoint(
        tokens=tokens,
        utility=utility,
        control_ppl=float(control["ppl"]),
        candidate_ppl=float(candidate["ppl"]),
    )


def _expert_number(expert_id: str) -> int:
    stage_text, expert_text = expert_id.split("-")
    return int(stage_text[1:]) * 100 + int(expert_text[1:])


def _candidate_seed(base: int, condition_index: int, expert_id: str, horizon: int) -> int:
    return (
        int(base)
        + 1_000_003 * (condition_index + 1)
        + 10_007 * (_expert_number(expert_id) + 1)
        + int(horizon // 1000)
    )


def _control_trajectory(
    *,
    args: argparse.Namespace,
    condition: str,
    condition_dir: Path,
    trunk_path: Path,
    teacher: torch.nn.Module,
    streams: dict[str, torch.Tensor],
    future_refs: list[dict[str, object]],
    validation_a: list[tuple[str, torch.Tensor, torch.Tensor]],
    validation_b: list[tuple[str, torch.Tensor, torch.Tensor]],
    schedule_state: dict[str, object],
    code_commit: str,
    device: torch.device,
    telemetry: Telemetry,
) -> tuple[list[dict[str, object]], dict[str, object], Path]:
    trajectory_path = condition_dir / "control-trajectory.json"
    final_path = condition_dir / "control-500000.pt"
    final_control_path = condition_dir / "final-control.json"
    existing = _json_read(trajectory_path, None)
    final_control = _json_read(final_control_path, None)
    if (
        existing is not None
        and existing.get("code_commit") == code_commit
        and final_path.exists()
        and _checkpoint_commit(final_path) == code_commit
        and final_control is not None
        and final_control.get("code_commit") == code_commit
    ):
        return list(existing["rows"]), final_control, final_path

    branch, optimizer, scheduler, _ = _restore_branch(args.release_dir, trunk_path, device)
    rows: list[dict[str, object]] = []
    start_age = 0
    for horizon in FORMAL_HORIZONS:
        _train_future(
            branch,
            optimizer,
            scheduler,
            teacher,
            streams,
            future_refs,
            start_age_tokens=start_age,
            end_age_tokens=horizon,
            decision_tokens=args.decision_tokens,
            tokens_per_step=args.batch_size * args.sequence_length,
            sequence_length=args.sequence_length,
            device=device,
            telemetry=telemetry,
            phase=f"{condition}_control_{horizon}",
        )
        evaluation = _evaluate(branch, validation_a)
        rows.append({"tokens": horizon, **evaluation})
        cp = condition_dir / f"control-{horizon}.pt"
        checkpoint(
            cp,
            branch,
            optimizer,
            scheduler,
            args.decision_tokens + horizon,
            args.decision_tokens // (args.batch_size * args.sequence_length)
            + horizon // (args.batch_size * args.sequence_length),
            {**schedule_state, "condition": condition, "branch": "control"},
            telemetry=telemetry,
            reason=f"{condition}_control_{horizon}",
        )
        _json_write(trajectory_path, {"code_commit": code_commit, "rows": rows})
        start_age = horizon

    final_b = _evaluate(branch, validation_b)
    final_control = {
        "code_commit": code_commit,
        "holdout": "B",
        "tokens": FORMAL_HORIZONS[-1],
        **final_b,
    }
    _json_write(final_control_path, final_control)
    return rows, final_control, final_path


def _run_condition(
    *,
    args: argparse.Namespace,
    condition: str,
    condition_index: int,
    trunk_path: Path,
    candidates: list[Any],
    perceptions: dict[str, torch.Tensor],
    teacher: torch.nn.Module,
    streams: dict[str, torch.Tensor],
    future_refs: list[dict[str, object]],
    validation_a: list[tuple[str, torch.Tensor, torch.Tensor]],
    validation_b: list[tuple[str, torch.Tensor, torch.Tensor]],
    parity_batch: tuple[torch.Tensor, torch.Tensor],
    schedule_state: dict[str, object],
    code_commit: str,
    device: torch.device,
    telemetry: Telemetry,
) -> dict[str, object]:
    condition_dir = args.output_dir / condition
    condition_dir.mkdir(parents=True, exist_ok=True)
    existing_promotion = _json_read(condition_dir / "promotion-decision.json", None)
    if existing_promotion is not None and existing_promotion.get("code_commit") == code_commit:
        return existing_promotion

    control_rows, final_control, _ = _control_trajectory(
        args=args,
        condition=condition,
        condition_dir=condition_dir,
        trunk_path=trunk_path,
        teacher=teacher,
        streams=streams,
        future_refs=future_refs,
        validation_a=validation_a,
        validation_b=validation_b,
        schedule_state=schedule_state,
        code_commit=code_commit,
        device=device,
        telemetry=telemetry,
    )
    control_by_horizon = {int(row["tokens"]): row for row in control_rows}

    trunk_model, _, _, _ = _restore_branch(args.release_dir, trunk_path, device)
    baseline_b = _evaluate(trunk_model, validation_b)
    _json_write(
        condition_dir / "baseline-evaluation.json",
        {"code_commit": code_commit, "holdout": "B", **baseline_b},
    )
    del trunk_model

    parity_rows: list[dict[str, object]] = []
    initial_rows: list[dict[str, object]] = []
    candidate_meta = {item.expert_id: item for item in candidates}
    initial_dir = condition_dir / "shadows"
    initial_dir.mkdir(parents=True, exist_ok=True)

    for candidate in candidates:
        evidence_path = initial_dir / f"{candidate.expert_id}-initial.json"
        checkpoint_100 = initial_dir / f"{candidate.expert_id}-100000.pt"
        existing = _json_read(evidence_path, None)
        if (
            existing is not None
            and existing.get("code_commit") == code_commit
            and checkpoint_100.exists()
            and _checkpoint_commit(checkpoint_100) == code_commit
        ):
            parity_rows.append(existing["parity"])
            initial_rows.append(existing["summary"])
            continue

        branch, optimizer, scheduler, _ = _restore_branch(args.release_dir, trunk_path, device)
        event = branch.birth(
            stage=int(candidate.stage),
            parent_id=str(candidate.expert_id),
            routed_perceptions=perceptions[candidate.expert_id].to(device),
            token=args.decision_tokens,
            validation_inputs=parity_batch[0],
            validation_targets=parity_batch[1],
            selection_method="probationary_shadow",
            pressure={
                "usage": float(candidate.usage),
                "pi0": float(candidate.pi0),
                "pi1": float(candidate.pi1),
                "geometry_separation": float(candidate.geometry_separation),
            },
            optimizer=optimizer,
        )
        points: list[dict[str, object]] = []
        start_age = 0
        for horizon in FORMAL_HORIZONS[:2]:
            _train_future(
                branch,
                optimizer,
                scheduler,
                teacher,
                streams,
                future_refs,
                start_age_tokens=start_age,
                end_age_tokens=horizon,
                decision_tokens=args.decision_tokens,
                tokens_per_step=args.batch_size * args.sequence_length,
                sequence_length=args.sequence_length,
                device=device,
                telemetry=telemetry,
                phase=f"{condition}_{candidate.expert_id}_{horizon}",
            )
            evaluation = _evaluate(branch, validation_a)
            point = _point(
                tokens=horizon,
                control=control_by_horizon[horizon],
                candidate=evaluation,
                seed=_candidate_seed(
                    args.replicate + 55031,
                    condition_index,
                    candidate.expert_id,
                    horizon,
                ),
                bootstrap_samples=args.bootstrap_samples,
            )
            points.append({
                **point.to_dict(),
                "candidate_batch_nlls": evaluation["batch_nlls"],
            })
            start_age = horizon
        checkpoint(
            checkpoint_100,
            branch,
            optimizer,
            scheduler,
            args.decision_tokens + SHORTLIST_HORIZON,
            args.decision_tokens // (args.batch_size * args.sequence_length)
            + SHORTLIST_HORIZON // (args.batch_size * args.sequence_length),
            {
                **schedule_state,
                "condition": condition,
                "branch": f"shadow:{candidate.expert_id}",
            },
            telemetry=telemetry,
            reason=f"{condition}_{candidate.expert_id}_100k_shadow",
        )
        parity = {
            "expert_id": candidate.expert_id,
            "stage": int(candidate.stage),
            "equivalent": event["parity"].get("status") == "CLM_GROWTH_EQUIVALENCE",
            "parity": event["parity"],
        }
        summary = {
            "expert_id": candidate.expert_id,
            "stage": int(candidate.stage),
            "relative_improvement": float(points[-1]["relative_improvement"]),
            "ci95_low": float(points[-1]["ci95_low"]),
            "ci95_high": float(points[-1]["ci95_high"]),
            "ppl_ratio": float(points[-1]["ppl_ratio"]),
        }
        evidence = {
            "code_commit": code_commit,
            "condition": condition,
            "expert_id": candidate.expert_id,
            "stage": int(candidate.stage),
            "birth": event,
            "parity": parity,
            "points": points,
            "summary": summary,
        }
        _json_write(evidence_path, evidence)
        parity_rows.append(parity)
        initial_rows.append(summary)

    parity_rows.sort(key=lambda row: (int(row["stage"]), str(row["expert_id"])))
    initial_rows.sort(key=lambda row: (int(row["stage"]), str(row["expert_id"])))
    _json_write(condition_dir / "growth-equivalence.json", parity_rows)
    _json_write(condition_dir / "initial-shadow-results.json", initial_rows)

    shortlist = shortlist_candidates(initial_rows, k=SHORTLIST_K)
    shortlist_ids = [str(row["expert_id"]) for row in shortlist]
    _json_write(
        condition_dir / "shortlist.json",
        {
            "code_commit": code_commit,
            "selection_horizon": SHORTLIST_HORIZON,
            "k": SHORTLIST_K,
            "rule": "top point-estimate relative improvement at 100K; analytic scores excluded",
            "experts": shortlist_ids,
        },
    )
    telemetry.write({
        "type": "probation_shortlist",
        "condition": condition,
        "experts": shortlist_ids,
    })

    all_trajectories: dict[str, list[dict[str, object]]] = {}
    decisions = []
    for expert_id in shortlist_ids:
        candidate = candidate_meta[expert_id]
        initial = _json_read(initial_dir / f"{expert_id}-initial.json", None)
        if initial is None:
            raise RuntimeError(f"missing 100K shadow evidence for {expert_id}")
        trajectory_path = initial_dir / f"{expert_id}-trajectory.json"
        final_checkpoint = initial_dir / f"{expert_id}-500000.pt"
        existing = _json_read(trajectory_path, None)
        if (
            existing is not None
            and existing.get("code_commit") == code_commit
            and final_checkpoint.exists()
            and _checkpoint_commit(final_checkpoint) == code_commit
        ):
            points_rows = list(existing["points"])
        else:
            checkpoint_100 = initial_dir / f"{expert_id}-100000.pt"
            branch, optimizer, scheduler, _ = _restore_branch(
                args.release_dir, checkpoint_100, device
            )
            points_rows = list(initial["points"])
            start_age = SHORTLIST_HORIZON
            for horizon in FORMAL_HORIZONS[2:]:
                _train_future(
                    branch,
                    optimizer,
                    scheduler,
                    teacher,
                    streams,
                    future_refs,
                    start_age_tokens=start_age,
                    end_age_tokens=horizon,
                    decision_tokens=args.decision_tokens,
                    tokens_per_step=args.batch_size * args.sequence_length,
                    sequence_length=args.sequence_length,
                    device=device,
                    telemetry=telemetry,
                    phase=f"{condition}_{expert_id}_{horizon}",
                )
                evaluation = _evaluate(branch, validation_a)
                point = _point(
                    tokens=horizon,
                    control=control_by_horizon[horizon],
                    candidate=evaluation,
                    seed=_candidate_seed(
                        args.replicate + 55031,
                        condition_index,
                        expert_id,
                        horizon,
                    ),
                    bootstrap_samples=args.bootstrap_samples,
                )
                points_rows.append({
                    **point.to_dict(),
                    "candidate_batch_nlls": evaluation["batch_nlls"],
                })
                cp = initial_dir / f"{expert_id}-{horizon}.pt"
                checkpoint(
                    cp,
                    branch,
                    optimizer,
                    scheduler,
                    args.decision_tokens + horizon,
                    args.decision_tokens // (args.batch_size * args.sequence_length)
                    + horizon // (args.batch_size * args.sequence_length),
                    {
                        **schedule_state,
                        "condition": condition,
                        "branch": f"shadow:{expert_id}",
                    },
                    telemetry=telemetry,
                    reason=f"{condition}_{expert_id}_{horizon}",
                )
                _json_write(
                    trajectory_path,
                    {
                        "code_commit": code_commit,
                        "condition": condition,
                        "expert_id": expert_id,
                        "points": points_rows,
                    },
                )
                start_age = horizon
        all_trajectories[expert_id] = points_rows
        point_objects = [
            ProbationPoint(
                tokens=int(row["tokens"]),
                utility=paired_bootstrap_utility(
                    control_by_horizon[int(row["tokens"])]["batch_nlls"],
                    row["candidate_batch_nlls"],
                    seed=_candidate_seed(
                        args.replicate + 55031,
                        condition_index,
                        expert_id,
                        int(row["tokens"]),
                    ),
                    bootstrap_samples=args.bootstrap_samples,
                ),
                control_ppl=float(row["control_ppl"]),
                candidate_ppl=float(row["candidate_ppl"]),
            )
            for row in points_rows
        ]
        decisions.append(summarize_probation(expert_id, point_objects))

    _json_write(condition_dir / "probation-trajectories.json", all_trajectories)
    _json_write(
        condition_dir / "probation-decisions.json",
        [item.to_dict() for item in decisions],
    )

    winner = select_promotion_candidate(decisions)
    confirmation = None
    rescued = False
    if winner is not None:
        expert_id = winner.expert_id
        branch, _, _, _ = _restore_branch(
            args.release_dir, initial_dir / f"{expert_id}-500000.pt", device
        )
        candidate_b = _evaluate(branch, validation_b)
        utility_b = paired_bootstrap_utility(
            final_control["batch_nlls"],
            candidate_b["batch_nlls"],
            seed=args.replicate + 9_000_001 + condition_index * 100_003,
            bootstrap_samples=args.bootstrap_samples,
        )
        story_control_nll = None
        story_candidate_nll = None
        if condition == "story_arithmetic_shift":
            story_control_nll = float(final_control["domain_nll"]["story"])
            story_candidate_nll = float(candidate_b["domain_nll"]["story"])
        confirmation = independent_confirmation(
            utility=utility_b,
            control_ppl=float(final_control["ppl"]),
            candidate_ppl=float(candidate_b["ppl"]),
            story_control_nll=story_control_nll,
            story_candidate_nll=story_candidate_nll,
        )
        _json_write(
            condition_dir / "final-candidate.json",
            {
                "code_commit": code_commit,
                "expert_id": expert_id,
                "stage": int(candidate_meta[expert_id].stage),
                "holdout": "B",
                **candidate_b,
                "confirmation": confirmation,
            },
        )
        early_row = next(
            row
            for row in all_trajectories[expert_id]
            if int(row["tokens"]) == SHORTLIST_HORIZON
        )
        early_point = ProbationPoint(
            tokens=SHORTLIST_HORIZON,
            utility=paired_bootstrap_utility(
                control_by_horizon[SHORTLIST_HORIZON]["batch_nlls"],
                early_row["candidate_batch_nlls"],
                seed=_candidate_seed(
                    args.replicate + 55031,
                    condition_index,
                    expert_id,
                    SHORTLIST_HORIZON,
                ),
                bootstrap_samples=args.bootstrap_samples,
            ),
            control_ppl=float(early_row["control_ppl"]),
            candidate_ppl=float(early_row["candidate_ppl"]),
        )
        rescued = maturation_rescue(early_point, confirmation)

    if condition == "story_arithmetic_shift":
        absorption = absorption_diagnostic(
            baseline_story_nll=float(baseline_b["domain_nll"]["story"]),
            baseline_arithmetic_nll=float(baseline_b["domain_nll"]["arithmetic"]),
            control_story_nll=float(final_control["domain_nll"]["story"]),
            control_arithmetic_nll=float(final_control["domain_nll"]["arithmetic"]),
        )
        _json_write(condition_dir / "absorption-diagnostic.json", absorption)
    else:
        absorption = None

    independent_confirmed = bool(confirmation and confirmation.get("confirmed"))
    if winner is None:
        action = "REJECT"
    elif independent_confirmed:
        action = "PROMOTE"
    else:
        action = "REJECT_OVERFIT"
    promotion = {
        "code_commit": code_commit,
        "condition": condition,
        "action": action,
        "selected_expert": winner.expert_id if winner is not None else None,
        "probe_accepted": winner is not None,
        "independent_confirmed": independent_confirmed,
        "maturation_rescue": bool(rescued),
        "final_ppl_ratio": (
            float(confirmation["ppl_ratio"]) if confirmation is not None else None
        ),
        "confirmation": confirmation,
        "absorption": absorption,
    }
    _json_write(condition_dir / "promotion-decision.json", promotion)
    telemetry.write({"type": "promotion_decision", **promotion})
    return promotion


def run(args: argparse.Namespace, telemetry: Telemetry, observed_hash: str) -> int:
    from minicells.language_scaling import prepare_scaling_corpus

    root = Path(".").resolve()
    provenance = git_provenance(root)
    if provenance["tracked_tree_dirty"]:
        raise RuntimeError("formal CLM-0.3d execution requires a clean tracked Git tree")
    code_commit = str(provenance["code_commit"])
    device = torch.device(args.device)
    seed = replicate_seed(args.replicate)
    seed_all(seed)

    tokens_per_step = args.batch_size * args.sequence_length
    if args.decision_tokens % tokens_per_step:
        raise ValueError("decision token boundary must align with a training step")
    if any(horizon % tokens_per_step for horizon in FORMAL_HORIZONS):
        raise ValueError("all probation horizons must align with a training step")
    if args.eval_batches % 2:
        raise ValueError("formal eval-batches must be even")

    max_horizon = FORMAL_HORIZONS[-1]
    train, validation_stream, tokenizer, corpus_manifest = prepare_scaling_corpus(
        Path("."),
        source_005_dir=args.source_005_dir,
        train_stream_tokens=args.decision_tokens + max_horizon + args.sequence_length + 2,
        validation_stream_tokens=max(100_000, 80_000),
    )
    arithmetic = prepare_arithmetic_cache(args.cache_dir, tokenizer)
    streams = {"story": train, "arithmetic": arithmetic["train"]}
    validation_streams = {
        "story": validation_stream,
        "arithmetic": arithmetic["validation"],
    }

    trunk_schedule = make_training_schedule(
        len(train),
        seed=seed,
        budget_tokens=args.decision_tokens,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
    )
    max_future_steps = max_horizon // tokens_per_step
    condition_refs: dict[str, list[dict[str, object]]] = {}
    validation_a_refs: dict[str, list[dict[str, object]]] = {}
    validation_b_refs: dict[str, list[dict[str, object]]] = {}
    validation_a: dict[str, list[tuple[str, torch.Tensor, torch.Tensor]]] = {}
    validation_b: dict[str, list[tuple[str, torch.Tensor, torch.Tensor]]] = {}
    for index, condition in enumerate(FORMAL_CONDITIONS):
        refs = _future_refs(
            condition,
            streams,
            steps=max_future_steps,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            seed=seed + 700_000 + 10_000 * index,
        )
        condition_refs[condition] = refs
        a_refs = _validation_refs(
            condition,
            holdout="A",
            eval_batches=args.eval_batches,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
        )
        b_refs = _validation_refs(
            condition,
            holdout="B",
            eval_batches=args.eval_batches,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
        )
        validation_a_refs[condition] = a_refs
        validation_b_refs[condition] = b_refs
        validation_a[condition] = _bundle_from_refs(
            a_refs,
            validation_streams,
            sequence_length=args.sequence_length,
            device=device,
        )
        validation_b[condition] = _bundle_from_refs(
            b_refs,
            validation_streams,
            sequence_length=args.sequence_length,
            device=device,
        )
    parity_refs = _parity_ref(
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
    )
    parity_bundle = _bundle_from_refs(
        parity_refs,
        validation_streams,
        sequence_length=args.sequence_length,
        device=device,
    )
    parity_batch = (parity_bundle[0][1], parity_bundle[0][2])

    conditions_identity = {
        condition: {
            "future_schedule_sha256": value_digest(condition_refs[condition]),
            "holdout_a_sha256": value_digest(validation_a_refs[condition]),
            "holdout_b_sha256": value_digest(validation_b_refs[condition]),
        }
        for condition in FORMAL_CONDITIONS
    }
    schedule_state: dict[str, object] = {
        "format": EXPERIMENT_FORMAT,
        **provenance,
        "replicate_seed": seed,
        "schedule_seed": seed,
        "trunk_schedule_sha256": schedule_digest(trunk_schedule.starts),
        "tokens_per_step": tokens_per_step,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "eval_batches": args.eval_batches,
        "calibration_batches": args.calibration_batches,
        "decision_tokens": args.decision_tokens,
        "probation_horizons": list(FORMAL_HORIZONS),
        "shortlist_k": SHORTLIST_K,
        "bootstrap_samples": args.bootstrap_samples,
        "practical_ppl_ratio_threshold": PRACTICAL_PPL_RATIO_THRESHOLD,
        "story_retention_ratio_threshold": STORY_RETENTION_RATIO_THRESHOLD,
        "balance_weight": 0.0,
        "conditions": conditions_identity,
        "parity_validation_sha256": value_digest(parity_refs),
        "story_corpus_manifest_sha256": value_digest(corpus_manifest),
        "arithmetic_manifest_sha256": value_digest(arithmetic["manifest"]),
    }
    identity_path = args.output_dir / "run-provenance.json"
    formal_identity = {**schedule_state, "base_model_sha256": observed_hash}
    existing_identity = _json_read(identity_path, None)
    if existing_identity is not None and existing_identity != formal_identity:
        raise RuntimeError(
            "existing CLM-0.3d evidence uses different code or formal semantics; "
            "restart the replicate instead of mixing evidence"
        )
    _json_write(identity_path, formal_identity)

    teacher = release_teacher(args.release_dir, device)
    telemetry.write({
        "type": "worker_started",
        "format": EXPERIMENT_FORMAT,
        "base_model_sha256": observed_hash,
        "corpus_manifest": corpus_manifest,
        "arithmetic_manifest": arithmetic["manifest"],
        **schedule_state,
    })

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
        if saved.get("code_commit") != code_commit:
            raise RuntimeError("trunk checkpoint belongs to another code commit")
        start_step = int(payload["training_step"])
        telemetry.write({
            "type": "trunk_resume",
            "path": str(latest),
            "training_step": start_step,
        })
    decision_step = args.decision_tokens // tokens_per_step
    if start_step < decision_step:
        _train_trunk(
            model,
            optimizer,
            scheduler,
            teacher,
            train,
            trunk_schedule,
            start_step=start_step,
            decision_step=decision_step,
            schedule_state=schedule_state,
            output_dir=args.output_dir,
            device=device,
            telemetry=telemetry,
        )
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
    if not (args.output_dir / "trunk-history.json").exists():
        _json_write(args.output_dir / "trunk-history.json", [])

    model, optimizer, _, _ = _restore_branch(args.release_dir, trunk_path, device)
    calibration = [
        batch_from_starts(
            train,
            trunk_schedule.starts[index],
            trunk_schedule.sequence_length,
            device,
        )
        for index in range(decision_step - args.calibration_batches, decision_step)
    ]
    raw_candidates, perceptions, _ = calibrate_split_regret(model, optimizer, calibration)
    candidates = sorted(raw_candidates, key=lambda item: (item.stage, item.expert_id))
    if len(candidates) != 12 or not all(item.eligible for item in candidates):
        raise RuntimeError("formal CLM-0.3d requires all 12 root lineages geometry-eligible")
    _json_write(
        args.output_dir / "geometry-calibration.json",
        [
            {
                "stage": int(item.stage),
                "expert_id": item.expert_id,
                "usage": float(item.usage),
                "routed_samples": int(item.routed_samples),
                "pi0": float(item.pi0),
                "pi1": float(item.pi1),
                "geometry_separation": float(item.geometry_separation),
                "eligible": bool(item.eligible),
                "split_regret_diagnostic_only": float(item.split_regret),
            }
            for item in candidates
        ],
    )
    del model, optimizer

    condition_results: dict[str, object] = {}
    for index, condition in enumerate(FORMAL_CONDITIONS):
        condition_results[condition] = _run_condition(
            args=args,
            condition=condition,
            condition_index=index,
            trunk_path=trunk_path,
            candidates=candidates,
            perceptions=perceptions,
            teacher=teacher,
            streams=streams,
            future_refs=condition_refs[condition],
            validation_a=validation_a[condition],
            validation_b=validation_b[condition],
            parity_batch=parity_batch,
            schedule_state=schedule_state,
            code_commit=code_commit,
            device=device,
            telemetry=telemetry,
        )

    births_checked = 0
    births_equivalent = 0
    for condition in FORMAL_CONDITIONS:
        parity = _json_read(args.output_dir / condition / "growth-equivalence.json", [])
        births_checked += len(parity)
        births_equivalent += sum(bool(row.get("equivalent")) for row in parity)
    final = {
        "format": "minicells.clm-0.3d-probationary-replicate.v1",
        "replicate": args.replicate,
        "conditions": condition_results,
        "births_checked": births_checked,
        "births_equivalent": births_equivalent,
        **provenance,
    }
    _json_write(args.output_dir / "replicate-result.json", final)
    telemetry.write({
        "type": "worker_complete",
        "format": EXPERIMENT_FORMAT,
        "consumed_tokens": args.decision_tokens + 2 * FORMAL_HORIZONS[-1],
        "target_tokens": args.decision_tokens + 2 * FORMAL_HORIZONS[-1],
        "formal_eligible": True,
        **provenance,
        "births_checked": births_checked,
        "births_equivalent": births_equivalent,
    })
    return 0


def main() -> int:
    args = parser().parse_args()
    observed = verify_base_release_hash(args.release_dir / "model.pt")
    telemetry = Telemetry(args.output_dir, "probationary", args.replicate)
    try:
        if not args.execute:
            telemetry.write({
                "type": "worker_started",
                "mode": "preflight_only",
                "base_model_sha256": observed,
            })
            telemetry.write({
                "type": "worker_complete",
                "mode": "preflight_only",
                "formal_eligible": False,
            })
            return 0
        return run(args, telemetry, observed)
    finally:
        telemetry.close()


if __name__ == "__main__":
    raise SystemExit(main())
