#!/usr/bin/env python3
"""Run one Experiment-025 arm on one visible GPU.

LLM arm:
  reproduce the matched Experiment-007 30M Transformer to 100M TinyStories,
  reset optimizer, then train the deterministic Story→Math shift.

CLM arm:
  load the retained 30M TextNCA@100M artifact, function-preservingly upcycle,
  reset optimizer, then run the same shift with at most two budgeted
  counterfactual birth decisions.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from minicells.clm_growth import ProgressiveGrowthCLM  # noqa: E402
from minicells.clm_sparse_runtime import install_optimized_runtime  # noqa: E402
from minicells.growth_checkpoint import (  # noqa: E402
    load_growth_checkpoint,
    save_growth_checkpoint,
)
from minicells.language_30m import (  # noqa: E402
    TARGET_TOKENS as STORY_PRETRAIN_TOKENS,
    TRANSFORMER_NAME,
    build_transformer_30m,
    open_memmap,
)
from minicells.language_data import load_tokenizer  # noqa: E402
from minicells.story_math_shift_30m import (  # noqa: E402
    BOOTSTRAP_SEED,
    EVAL_TOKENS,
    EXPECTED_SOURCE_TOKENS,
    FORMAT,
    GRAD_CLIP,
    GROWTH_DECISION_TOKENS,
    MATH_VALIDATION_BATCHES,
    MAX_PROMOTIONS,
    PROBATION_TOKENS,
    SHIFT_BASE_LR,
    SHIFT_BETAS,
    SHIFT_STEPS,
    SHIFT_TOKENS,
    SHIFT_WEIGHT_DECAY,
    SOURCE_007_ARTIFACT,
    STORY_VALIDATION_BATCHES,
    STORY_VALIDATION_SEED,
    MATH_VALIDATION_SEED,
    TOKENS_PER_STEP,
    WORKER_FORMAT,
    build_30m_clm,
    build_math_exact_batches,
    collect_growth_proposal,
    evaluate_domains,
    fixed_validation_starts,
    parameter_snapshot,
    promotion_decision,
    schedule_manifest,
    shift_batch,
    shift_lr,
)


RESUME_INTERVAL_TOKENS = 2_000_000
PROGRESS_INTERVAL_TOKENS = 250_000
DEFAULT_MAX_WALL_HOURS = 9.25
MIN_PROBATION_REMAINING_SECONDS = 30 * 60
CLM_TRAIN_BACKEND = "batched_dense"
CLM_EVAL_BACKEND = "masked_dense"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run one Experiment-025 GPU arm")
    result.add_argument("--arm", choices=("llm", "clm"), required=True)
    result.add_argument("--story-cache-dir", type=Path, required=True)
    result.add_argument("--math-cache-dir", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--shift-tokens", type=int, default=SHIFT_TOKENS)
    result.add_argument("--max-wall-hours", type=float, default=DEFAULT_MAX_WALL_HOURS)
    result.add_argument("--reset", action="store_true")
    return result.parse_args()


def _json_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def git_identity() -> dict[str, object]:
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        text=True,
    ).strip()
    identity = {
        "code_commit": _git("rev-parse", "HEAD"),
        "code_tree_sha": _git("rev-parse", "HEAD^{tree}"),
        "tracked_tree_dirty": bool(dirty),
    }
    if identity["tracked_tree_dirty"]:
        raise RuntimeError("Experiment 025 refuses to run from a dirty tracked tree")
    return identity


def atomic_torch_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _load_llm_artifact(
    path: Path,
    *,
    vocab_size: int,
    device: torch.device,
) -> torch.nn.Module:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "minicells.language-inference.v1":
        raise RuntimeError(f"unexpected Transformer artifact format: {payload.get('format')!r}")
    if payload.get("model_name") != TRANSFORMER_NAME:
        raise RuntimeError(f"unexpected Transformer artifact model: {payload.get('model_name')!r}")
    if int(payload.get("consumed_tokens", -1)) != STORY_PRETRAIN_TOKENS:
        raise RuntimeError("Experiment 025 LLM must begin at the 100M Story checkpoint")
    model, _ = build_transformer_30m(vocab_size)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device)


def ensure_llm_story_pretrain(
    *,
    story_cache_dir: Path,
    arm_dir: Path,
    deadline: float,
) -> Path:
    pretrain_dir = arm_dir / "pretrain"
    artifact = pretrain_dir / "transformer-30m-fp16.pt"
    if artifact.is_file():
        return artifact
    if time.monotonic() >= deadline:
        raise RuntimeError("wall-time budget expired before LLM Story pretraining")
    worker = ROOT / "scripts" / "run_consumer_language_30m_variant.py"
    command = [
        sys.executable,
        str(worker),
        "--model",
        TRANSFORMER_NAME,
        "--cache-dir",
        str(story_cache_dir),
        "--output-dir",
        str(pretrain_dir),
        "--stop-after-tokens",
        str(STORY_PRETRAIN_TOKENS),
    ]
    print("reproducing matched 30M Transformer Story checkpoint")
    subprocess.run(command, cwd=ROOT, check=True)
    if not artifact.is_file():
        raise FileNotFoundError(artifact)
    return artifact


def fresh_optimizer(model: torch.nn.Module) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        model.parameters(),
        lr=SHIFT_BASE_LR,
        betas=SHIFT_BETAS,
        weight_decay=SHIFT_WEIGHT_DECAY,
    )


def _set_lr(optimizer: torch.optim.Optimizer, step: int, total_steps: int) -> float:
    lr = shift_lr(step, total_steps=total_steps)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def _eval_row(
    model: torch.nn.Module,
    *,
    arm: str,
    shift_step: int,
    story_validation,
    math_validation,
    story_starts,
    math_starts,
    exact_batches,
    device: torch.device,
) -> dict[str, object]:
    backend = CLM_EVAL_BACKEND if arm == "clm" else None
    metrics = evaluate_domains(
        model,
        story_validation=story_validation,
        math_validation=math_validation,
        story_starts=story_starts,
        math_starts=math_starts,
        math_exact_batches=exact_batches,
        device=device,
        clm_backend=backend,
    )
    return {
        "arm": arm,
        "shift_step": int(shift_step),
        "shift_tokens": int(shift_step * TOKENS_PER_STEP),
        "total_experience_tokens": int(EXPECTED_SOURCE_TOKENS + shift_step * TOKENS_PER_STEP),
        **parameter_snapshot(model),
        **metrics,
    }


def _train_one(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    *,
    arm: str,
    step: int,
    total_steps: int,
    story_train,
    math_train,
    device: torch.device,
) -> tuple[float, str, float]:
    domain, inputs, targets = shift_batch(
        step,
        story_stream=story_train,
        math_stream=math_train,
        device=device,
    )
    lr = _set_lr(optimizer, step + 1, total_steps)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        if arm == "clm":
            output = model(inputs, execution_backend=CLM_TRAIN_BACKEND)
        else:
            output = model(inputs)
        loss = F.cross_entropy(output.logits.flatten(0, 1), targets.reshape(-1))
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    scaler.step(optimizer)
    scaler.update()
    return float(loss.detach().item()), domain, lr


def _save_llm_resume(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    shift_step: int,
    history: list[dict[str, object]],
    events: list[dict[str, object]],
    physical_training_tokens: int,
    provenance: dict[str, object],
) -> None:
    atomic_torch_save(
        {
            "format": "minicells.story-math-shift-llm-resume.v1",
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "shift_step": shift_step,
            "history": history,
            "events": events,
            "physical_training_tokens": physical_training_tokens,
            "provenance": provenance,
        },
        path,
    )


def _load_llm_resume(
    path: Path,
    *,
    artifact: Path,
    vocab_size: int,
    device: torch.device,
) -> tuple[
    torch.nn.Module,
    torch.optim.AdamW,
    torch.amp.GradScaler,
    int,
    list[dict[str, object]],
    list[dict[str, object]],
    int,
]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format") != "minicells.story-math-shift-llm-resume.v1":
        raise RuntimeError("unsupported Experiment-025 LLM resume format")
    model = _load_llm_artifact(artifact, vocab_size=vocab_size, device=device)
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer = fresh_optimizer(model)
    optimizer.load_state_dict(payload["optimizer_state"])
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    scaler.load_state_dict(payload["scaler_state"])
    return (
        model,
        optimizer,
        scaler,
        int(payload["shift_step"]),
        list(payload.get("history", [])),
        list(payload.get("events", [])),
        int(payload.get("physical_training_tokens", 0)),
    )


def _save_clm_resume(
    path: Path,
    *,
    model: ProgressiveGrowthCLM,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    shift_step: int,
    history: list[dict[str, object]],
    events: list[dict[str, object]],
    physical_training_tokens: int,
    provenance: dict[str, object],
    reason: str,
) -> None:
    total_tokens = EXPECTED_SOURCE_TOKENS + shift_step * TOKENS_PER_STEP
    save_growth_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        consumed_tokens=total_tokens,
        training_step=shift_step,
        data_schedule_state={
            "format": FORMAT,
            "current_step": shift_step,
            "consumed_tokens": total_tokens,
            "reason": reason,
            "provenance": provenance,
        },
        metrics={
            "scaler_state": scaler.state_dict(),
            "history": history,
            "events": events,
            "physical_training_tokens": physical_training_tokens,
        },
    )


def _restore_clm_resume(
    path: Path,
    *,
    source_artifact: Path,
    vocab_size: int,
    device: torch.device,
) -> tuple[
    ProgressiveGrowthCLM,
    torch.optim.AdamW,
    torch.amp.GradScaler,
    int,
    list[dict[str, object]],
    list[dict[str, object]],
    int,
]:
    model, _ = build_30m_clm(
        source_artifact,
        vocab_size=vocab_size,
        device=device,
    )
    install_optimized_runtime(model)
    optimizer = fresh_optimizer(model)
    model, payload = load_growth_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        map_location=device,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    metrics = dict(payload.get("metrics", {}))
    if metrics.get("scaler_state") is not None:
        scaler.load_state_dict(metrics["scaler_state"])
    return (
        model,
        optimizer,
        scaler,
        int(payload["training_step"]),
        list(metrics.get("history", [])),
        list(metrics.get("events", [])),
        int(metrics.get("physical_training_tokens", 0)),
    )


def _train_range(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    *,
    arm: str,
    start_step: int,
    end_step: int,
    total_steps: int,
    story_train,
    math_train,
    device: torch.device,
) -> tuple[int, float]:
    last_loss = float("nan")
    for step in range(start_step, end_step):
        last_loss, _, _ = _train_one(
            model,
            optimizer,
            scaler,
            arm=arm,
            step=step,
            total_steps=total_steps,
            story_train=story_train,
            math_train=math_train,
            device=device,
        )
    return (end_step - start_step) * TOKENS_PER_STEP, last_loss


def _age_zero_parity(
    clm: ProgressiveGrowthCLM,
    *,
    source_artifact: Path,
    vocab_size: int,
    story_validation,
    story_starts,
    device: torch.device,
) -> dict[str, float]:
    from minicells.story_math_shift_30m import load_30m_textnca_source

    source, _ = load_30m_textnca_source(
        source_artifact,
        vocab_size=vocab_size,
        device=device,
    )
    source.eval()
    clm.eval()
    starts = story_starts[0]
    from minicells.language_30m import memmap_batch

    inputs, _ = memmap_batch(story_validation, starts, 128, device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        source_logits = source(inputs).logits.float()
        clm_logits = clm(inputs, execution_backend=CLM_EVAL_BACKEND).logits.float()
    diff = source_logits - clm_logits
    result = {
        "max_abs_diff": float(diff.abs().max().item()),
        "rms_diff": float(diff.square().mean().sqrt().item()),
        "argmax_agreement": float(
            (source_logits.argmax(-1) == clm_logits.argmax(-1)).float().mean().item()
        ),
    }
    del source
    torch.cuda.empty_cache()
    return result


def _probation(
    *,
    model: ProgressiveGrowthCLM,
    optimizer: torch.optim.AdamW,
    scaler: torch.amp.GradScaler,
    shift_step: int,
    total_steps: int,
    history: list[dict[str, object]],
    events: list[dict[str, object]],
    physical_training_tokens: int,
    story_train,
    math_train,
    story_validation,
    math_validation,
    story_starts,
    math_starts,
    exact_batches,
    source_artifact: Path,
    vocab_size: int,
    arm_dir: Path,
    provenance: dict[str, object],
    device: torch.device,
) -> tuple[
    ProgressiveGrowthCLM,
    torch.optim.AdamW,
    torch.amp.GradScaler,
    int,
    list[dict[str, object]],
    list[dict[str, object]],
    int,
]:
    decision_tokens = shift_step * TOKENS_PER_STEP
    proposal, pressure_table = collect_growth_proposal(
        model,
        story_stream=story_train,
        math_stream=math_train,
        start_step=shift_step,
        device=device,
        execution_backend=CLM_TRAIN_BACKEND,
    )
    event: dict[str, object] = {
        "type": "growth_decision",
        "decision_shift_tokens": decision_tokens,
        "decision_total_experience_tokens": EXPECTED_SOURCE_TOKENS + decision_tokens,
        "pressure_table": pressure_table,
        "proposal": proposal.to_dict() if proposal is not None else None,
    }
    if proposal is None:
        event["outcome"] = "NO_ELIGIBLE_PROPOSAL"
        events.append(event)
        return model, optimizer, scaler, shift_step, history, events, physical_training_tokens

    start_checkpoint = arm_dir / f"decision-{decision_tokens}-start.pt"
    _save_clm_resume(
        start_checkpoint,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        shift_step=shift_step,
        history=history,
        events=events,
        physical_training_tokens=physical_training_tokens,
        provenance=provenance,
        reason="probation_start",
    )
    end_step = min(total_steps, shift_step + PROBATION_TOKENS // TOKENS_PER_STEP)
    parity_starts = story_starts[0]
    from minicells.language_30m import memmap_batch

    parity_inputs, parity_targets = memmap_batch(
        story_validation,
        parity_starts,
        128,
        device,
    )

    # Control: no birth, exact same future schedule.
    control, control_opt, control_scaler, _, _, _, control_physical = _restore_clm_resume(
        start_checkpoint,
        source_artifact=source_artifact,
        vocab_size=vocab_size,
        device=device,
    )
    added, _ = _train_range(
        control,
        control_opt,
        control_scaler,
        arm="clm",
        start_step=shift_step,
        end_step=end_step,
        total_steps=total_steps,
        story_train=story_train,
        math_train=math_train,
        device=device,
    )
    control_physical += added
    control_eval = evaluate_domains(
        control,
        story_validation=story_validation,
        math_validation=math_validation,
        story_starts=story_starts,
        math_starts=math_starts,
        math_exact_batches=exact_batches,
        device=device,
        clm_backend=CLM_EVAL_BACKEND,
    )
    control_checkpoint = arm_dir / f"decision-{decision_tokens}-control.pt"
    _save_clm_resume(
        control_checkpoint,
        model=control,
        optimizer=control_opt,
        scaler=control_scaler,
        shift_step=end_step,
        history=history,
        events=events,
        physical_training_tokens=control_physical,
        provenance=provenance,
        reason="probation_control",
    )
    del control, control_opt, control_scaler
    torch.cuda.empty_cache()

    # Shadow: exact parent clone + local split, then the same future schedule.
    shadow, shadow_opt, shadow_scaler, _, _, _, shadow_physical = _restore_clm_resume(
        start_checkpoint,
        source_artifact=source_artifact,
        vocab_size=vocab_size,
        device=device,
    )
    birth = shadow.birth(
        stage=proposal.stage,
        parent_id=proposal.expert_id,
        routed_perceptions=proposal.perceptions.to(device),
        token=EXPECTED_SOURCE_TOKENS + decision_tokens,
        validation_inputs=parity_inputs,
        validation_targets=parity_targets,
        selection_method="budgeted_route_usage_shadow",
        pressure={
            "usage": proposal.usage,
            "routed_samples": float(proposal.routed_samples),
        },
        optimizer=shadow_opt,
    )
    added, _ = _train_range(
        shadow,
        shadow_opt,
        shadow_scaler,
        arm="clm",
        start_step=shift_step,
        end_step=end_step,
        total_steps=total_steps,
        story_train=story_train,
        math_train=math_train,
        device=device,
    )
    shadow_physical += added
    shadow_eval = evaluate_domains(
        shadow,
        story_validation=story_validation,
        math_validation=math_validation,
        story_starts=story_starts,
        math_starts=math_starts,
        math_exact_batches=exact_batches,
        device=device,
        clm_backend=CLM_EVAL_BACKEND,
    )
    decision = promotion_decision(
        control_eval,
        shadow_eval,
        seed=BOOTSTRAP_SEED + decision_tokens,
    )
    shadow_checkpoint = arm_dir / f"decision-{decision_tokens}-shadow.pt"
    _save_clm_resume(
        shadow_checkpoint,
        model=shadow,
        optimizer=shadow_opt,
        scaler=shadow_scaler,
        shift_step=end_step,
        history=history,
        events=events,
        physical_training_tokens=shadow_physical,
        provenance=provenance,
        reason="probation_shadow",
    )
    event.update(
        {
            "birth": birth,
            "control": control_eval,
            "shadow": shadow_eval,
            "decision": decision,
            "probation_end_shift_tokens": end_step * TOKENS_PER_STEP,
            "outcome": "PROMOTE" if decision["promote"] else "REJECT",
        }
    )
    events = [*events, event]

    selected_path = shadow_checkpoint if decision["promote"] else control_checkpoint
    del shadow, shadow_opt, shadow_scaler
    torch.cuda.empty_cache()
    selected, selected_opt, selected_scaler, selected_step, _, _, selected_physical = _restore_clm_resume(
        selected_path,
        source_artifact=source_artifact,
        vocab_size=vocab_size,
        device=device,
    )
    selected_eval = shadow_eval if decision["promote"] else control_eval
    row = {
        "arm": "clm",
        "shift_step": selected_step,
        "shift_tokens": selected_step * TOKENS_PER_STEP,
        "total_experience_tokens": EXPECTED_SOURCE_TOKENS + selected_step * TOKENS_PER_STEP,
        "event": "post_probation_promotion" if decision["promote"] else "post_probation_rejection",
        **parameter_snapshot(selected),
        **selected_eval,
    }
    history = [item for item in history if int(item["shift_tokens"]) != int(row["shift_tokens"])]
    history.append(row)
    history.sort(key=lambda item: int(item["shift_tokens"]))
    # Physical cost includes both counterfactual branches, not only the selected path.
    physical_training_tokens = physical_training_tokens + 2 * (end_step - shift_step) * TOKENS_PER_STEP
    return (
        selected,
        selected_opt,
        selected_scaler,
        selected_step,
        history,
        events,
        physical_training_tokens,
    )


def main() -> int:
    args = parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Experiment 025 requires CUDA")
    if args.shift_tokens <= 0 or args.shift_tokens > SHIFT_TOKENS:
        raise ValueError("--shift-tokens must be in (0, 50M]")
    if args.shift_tokens % TOKENS_PER_STEP:
        raise ValueError("--shift-tokens must align to whole training steps")
    if args.max_wall_hours <= 0:
        raise ValueError("--max-wall-hours must be positive")

    provenance = git_identity()
    device = torch.device("cuda")
    arm_dir = args.output_dir / args.arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    resume_path = arm_dir / f"{args.arm}-latest.pt"
    if args.reset:
        resume_path.unlink(missing_ok=True)

    session_started = time.monotonic()
    deadline = session_started + args.max_wall_hours * 3600.0
    story_train_path = args.story_cache_dir / "train.u16"
    story_validation_path = args.story_cache_dir / "validation.u16"
    tokenizer_path = args.story_cache_dir / "tokenizer.json"
    math_root = args.math_cache_dir / "math-30m-shift"
    math_train_path = math_root / "train.u16"
    math_validation_path = math_root / "validation.u16"
    for path in (
        story_train_path,
        story_validation_path,
        tokenizer_path,
        math_train_path,
        math_validation_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    tokenizer = load_tokenizer(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    story_train = open_memmap(story_train_path)
    story_validation = open_memmap(story_validation_path)
    math_train = open_memmap(math_train_path)
    math_validation = open_memmap(math_validation_path)
    story_starts = fixed_validation_starts(
        len(story_validation),
        batches=STORY_VALIDATION_BATCHES,
        seed=STORY_VALIDATION_SEED,
    )
    math_starts = fixed_validation_starts(
        len(math_validation),
        batches=MATH_VALIDATION_BATCHES,
        seed=MATH_VALIDATION_SEED,
    )
    exact_batches = build_math_exact_batches(tokenizer)
    total_steps = args.shift_tokens // TOKENS_PER_STEP

    age_zero_parity: dict[str, float] | None = None
    source_identity: dict[str, object] | None = None
    llm_artifact: Path | None = None

    if args.arm == "llm":
        llm_artifact = ensure_llm_story_pretrain(
            story_cache_dir=args.story_cache_dir,
            arm_dir=arm_dir,
            deadline=deadline,
        )
        if resume_path.is_file():
            model, optimizer, scaler, shift_step, history, events, physical_training_tokens = _load_llm_resume(
                resume_path,
                artifact=llm_artifact,
                vocab_size=vocab_size,
                device=device,
            )
        else:
            model = _load_llm_artifact(llm_artifact, vocab_size=vocab_size, device=device)
            optimizer = fresh_optimizer(model)
            scaler = torch.amp.GradScaler("cuda", enabled=True)
            shift_step = 0
            history: list[dict[str, object]] = []
            events: list[dict[str, object]] = [
                {
                    "type": "optimizer_reset",
                    "shift_tokens": 0,
                    "reason": "fair Story→Math boundary; both arms start fresh AdamW",
                }
            ]
            physical_training_tokens = 0
    else:
        source_artifact = ROOT / SOURCE_007_ARTIFACT
        if resume_path.is_file():
            model, optimizer, scaler, shift_step, history, events, physical_training_tokens = _restore_clm_resume(
                resume_path,
                source_artifact=source_artifact,
                vocab_size=vocab_size,
                device=device,
            )
        else:
            model, source_identity = build_30m_clm(
                source_artifact,
                vocab_size=vocab_size,
                device=device,
            )
            install_optimized_runtime(model)
            age_zero_parity = _age_zero_parity(
                model,
                source_artifact=source_artifact,
                vocab_size=vocab_size,
                story_validation=story_validation,
                story_starts=story_starts,
                device=device,
            )
            optimizer = fresh_optimizer(model)
            scaler = torch.amp.GradScaler("cuda", enabled=True)
            shift_step = 0
            history = []
            events = [
                {
                    "type": "function_preserving_upcycle",
                    "shift_tokens": 0,
                    "source": source_identity,
                    "age_zero_parity": age_zero_parity,
                    "program_cells": int(model.expert_count),
                },
                {
                    "type": "optimizer_reset",
                    "shift_tokens": 0,
                    "reason": "fair Story→Math boundary; both arms start fresh AdamW",
                },
            ]
            physical_training_tokens = 0

    torch.cuda.reset_peak_memory_stats()
    shift_started = time.monotonic()

    def evaluate_if_needed(force: bool = False, event: str | None = None) -> None:
        consumed = shift_step * TOKENS_PER_STEP
        if not force and consumed not in set(EVAL_TOKENS):
            return
        if any(int(row["shift_tokens"]) == consumed for row in history):
            return
        row = _eval_row(
            model,
            arm=args.arm,
            shift_step=shift_step,
            story_validation=story_validation,
            math_validation=math_validation,
            story_starts=story_starts,
            math_starts=math_starts,
            exact_batches=exact_batches,
            device=device,
        )
        if event is not None:
            row["event"] = event
        history.append(row)
        history.sort(key=lambda item: int(item["shift_tokens"]))
        print(
            f"{args.arm} shift={consumed/1e6:5.1f}M "
            f"story_ppl={row['story_ppl']:.3f} "
            f"math_exact={100*row['math_exact_answer_accuracy']:.1f}% "
            f"cells={row['program_cells']}"
        )

    evaluate_if_needed(force=True, event="shift_start")
    last_loss = float("nan")
    stopped_for_budget = False
    handled_decisions = {
        int(event["decision_shift_tokens"])
        for event in events
        if event.get("type") == "growth_decision" and event.get("decision_shift_tokens") is not None
    }

    while shift_step < total_steps:
        consumed = shift_step * TOKENS_PER_STEP
        if time.monotonic() >= deadline - 5 * 60:
            stopped_for_budget = True
            break

        if (
            args.arm == "clm"
            and consumed in GROWTH_DECISION_TOKENS
            and consumed not in handled_decisions
            and sum(1 for event in events if event.get("outcome") == "PROMOTE") < MAX_PROMOTIONS
        ):
            if deadline - time.monotonic() < MIN_PROBATION_REMAINING_SECONDS:
                stopped_for_budget = True
                break
            (
                model,
                optimizer,
                scaler,
                shift_step,
                history,
                events,
                physical_training_tokens,
            ) = _probation(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                shift_step=shift_step,
                total_steps=total_steps,
                history=history,
                events=events,
                physical_training_tokens=physical_training_tokens,
                story_train=story_train,
                math_train=math_train,
                story_validation=story_validation,
                math_validation=math_validation,
                story_starts=story_starts,
                math_starts=math_starts,
                exact_batches=exact_batches,
                source_artifact=ROOT / SOURCE_007_ARTIFACT,
                vocab_size=vocab_size,
                arm_dir=arm_dir,
                provenance=provenance,
                device=device,
            )
            handled_decisions.add(consumed)
            evaluate_if_needed(force=True)
            continue

        last_loss, domain, lr = _train_one(
            model,
            optimizer,
            scaler,
            arm=args.arm,
            step=shift_step,
            total_steps=total_steps,
            story_train=story_train,
            math_train=math_train,
            device=device,
        )
        shift_step += 1
        physical_training_tokens += TOKENS_PER_STEP
        consumed = shift_step * TOKENS_PER_STEP
        evaluate_if_needed()

        if consumed % PROGRESS_INTERVAL_TOKENS == 0:
            elapsed = max(time.monotonic() - shift_started, 1e-9)
            print(
                f"{args.arm} progress={consumed/1e6:5.1f}M "
                f"domain={domain:5s} loss={last_loss:.4f} lr={lr:.2e} "
                f"main_tok/s={consumed/elapsed:.0f}"
            )

        if consumed % RESUME_INTERVAL_TOKENS == 0:
            if args.arm == "llm":
                _save_llm_resume(
                    resume_path,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    shift_step=shift_step,
                    history=history,
                    events=events,
                    physical_training_tokens=physical_training_tokens,
                    provenance=provenance,
                )
            else:
                _save_clm_resume(
                    resume_path,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    shift_step=shift_step,
                    history=history,
                    events=events,
                    physical_training_tokens=physical_training_tokens,
                    provenance=provenance,
                    reason="periodic_resume",
                )

    evaluate_if_needed(force=True, event="final_or_budget_stop")
    if args.arm == "llm":
        _save_llm_resume(
            resume_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            shift_step=shift_step,
            history=history,
            events=events,
            physical_training_tokens=physical_training_tokens,
            provenance=provenance,
        )
    else:
        _save_clm_resume(
            resume_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            shift_step=shift_step,
            history=history,
            events=events,
            physical_training_tokens=physical_training_tokens,
            provenance=provenance,
            reason="final_or_budget_stop",
        )

    elapsed = time.monotonic() - session_started
    complete = shift_step >= total_steps
    pd.DataFrame(history).to_csv(arm_dir / "metrics.csv", index=False)
    _json_write(arm_dir / "events.json", events)
    summary = {
        "format": WORKER_FORMAT,
        "protocol_format": FORMAT,
        "arm": args.arm,
        "complete": complete,
        "stopped_for_wall_budget": stopped_for_budget,
        "shift_tokens": shift_step * TOKENS_PER_STEP,
        "target_shift_tokens": args.shift_tokens,
        "physical_training_tokens": physical_training_tokens,
        "elapsed_seconds": elapsed,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "fresh_optimizer_at_shift": True,
        "training_objective": "standard next-token cross entropy",
        "clm_training_backend": CLM_TRAIN_BACKEND if args.arm == "clm" else None,
        "growth_promotions": sum(1 for event in events if event.get("outcome") == "PROMOTE"),
        "growth_rejections": sum(1 for event in events if event.get("outcome") == "REJECT"),
        "final_parameters": parameter_snapshot(model),
        "source_identity": source_identity,
        "age_zero_parity": age_zero_parity,
        "llm_story_artifact": str(llm_artifact) if llm_artifact is not None else None,
        "schedule": schedule_manifest(args.shift_tokens),
        "provenance": provenance,
    }
    _json_write(arm_dir / "worker-summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
