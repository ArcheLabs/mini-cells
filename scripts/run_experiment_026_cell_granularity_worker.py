#!/usr/bin/env python3
"""Run one formal Experiment-026 granularity arm on one visible CUDA GPU."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from minicells.granularity_30m import (  # noqa: E402
    BASE_LR,
    BETAS,
    CONTINUATION_TOKENS,
    DOMAINS,
    EVAL_TOKENS,
    FORMAT,
    GRANULARITIES,
    GRAD_CLIP,
    TOKENS_PER_STEP,
    WEIGHT_DECAY,
    WORKER_FORMAT,
    build_granularity_model,
    collect_cell_diagnostics,
    continuation_batch,
    continuation_lr,
    diagnostic_batches,
    evaluate_domains,
    fixed_validation_starts,
    model_structure,
    summarize_diagnostics,
)
from minicells.language_data import load_tokenizer  # noqa: E402
from minicells.local_plasticity import (  # noqa: E402
    LocalPlasticityConfig,
    build_local_plasticity_optimizer,
    plasticity_summary,
    set_global_lr,
    update_local_plasticity,
)
from minicells.story_math_shift_30m import (  # noqa: E402
    SOURCE_007_ARTIFACT,
    build_30m_clm,
)

RESUME_INTERVAL_TOKENS = 2_000_000
PROGRESS_INTERVAL_TOKENS = 250_000
DEFAULT_MAX_WALL_HOURS = 2.5
AGE_ZERO_MAX_ABS_TOLERANCE = 5e-4
PLASTICITY_CONFIG = LocalPlasticityConfig()


def parser() -> argparse.Namespace:
    result = argparse.ArgumentParser(
        description="Run one Experiment-026 granularity arm"
    )
    result.add_argument(
        "--granularity",
        type=int,
        choices=GRANULARITIES,
        required=True,
    )
    result.add_argument("--tokenizer-path", type=Path, required=True)
    result.add_argument("--story-train", type=Path, required=True)
    result.add_argument("--story-validation", type=Path, required=True)
    result.add_argument("--math-train", type=Path, required=True)
    result.add_argument("--math-validation", type=Path, required=True)
    result.add_argument("--symbolic-train", type=Path, required=True)
    result.add_argument("--symbolic-validation", type=Path, required=True)
    result.add_argument("--facts-train", type=Path, required=True)
    result.add_argument("--facts-validation", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument(
        "--continuation-tokens",
        type=int,
        default=CONTINUATION_TOKENS,
    )
    result.add_argument(
        "--max-wall-hours",
        type=float,
        default=DEFAULT_MAX_WALL_HOURS,
    )
    result.add_argument("--reset", action="store_true")
    return result.parse_args()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def git_identity() -> dict[str, object]:
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        text=True,
    ).strip()
    if dirty:
        raise RuntimeError("Experiment 026 refuses to run from a dirty tracked tree")
    return {
        "code_commit": _git("rev-parse", "HEAD"),
        "code_tree_sha": _git("rev-parse", "HEAD^{tree}"),
        "tracked_tree_dirty": False,
    }


def _json_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_torch_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _memmap(path: Path) -> np.memmap:
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.memmap(path, dtype=np.uint16, mode="r")


def _age_zero_parity(
    tissue_model: torch.nn.Module,
    *,
    vocab_size: int,
    inputs: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    reference, _ = build_30m_clm(
        ROOT / SOURCE_007_ARTIFACT,
        vocab_size=vocab_size,
        device=device,
    )
    reference.eval()
    tissue_model.eval()
    try:
        with torch.no_grad():
            expected = reference(inputs).logits.float()
            actual = tissue_model(inputs).logits.float()
        difference = actual - expected
        result = {
            "max_abs_diff": float(difference.abs().max().item()),
            "rms_diff": float(difference.square().mean().sqrt().item()),
            "argmax_agreement": float(
                (actual.argmax(-1) == expected.argmax(-1)).float().mean().item()
            ),
        }
    finally:
        del reference
        torch.cuda.empty_cache()
    if result["max_abs_diff"] > AGE_ZERO_MAX_ABS_TOLERANCE:
        raise RuntimeError(f"age-zero tissue parity failed: {result}")
    return result


def _save_tables(
    arm_dir: Path,
    metrics: list[dict[str, object]],
    cell_rows: list[dict[str, object]],
    tissue_rows: list[dict[str, object]],
) -> None:
    pd.DataFrame(metrics).to_csv(arm_dir / "metrics.csv", index=False)
    pd.DataFrame(cell_rows).to_csv(arm_dir / "cell-diagnostics.csv", index=False)
    pd.DataFrame(tissue_rows).to_csv(
        arm_dir / "tissue-diagnostics.csv",
        index=False,
    )


def _evaluate_checkpoint(
    model: torch.nn.Module,
    *,
    granularity: int,
    step: int,
    validation_streams: dict[str, np.memmap],
    validation_starts: dict[str, tuple[tuple[int, ...], ...]],
    probe_batches,
    baseline_profiles,
    device: torch.device,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, dict[str, float]],
]:
    domain_metrics = evaluate_domains(
        model,
        validation_streams=validation_streams,
        validation_starts=validation_starts,
        device=device,
    )
    domain_nlls = {
        domain: float(domain_metrics[f"{domain}_nll"])
        for domain in DOMAINS
    }
    cells, tissues, profiles = collect_cell_diagnostics(
        model,
        batches=probe_batches,
        domain_nlls=domain_nlls,
        baseline_profiles=baseline_profiles,
    )
    metric = {
        "granularity": granularity,
        "continuation_step": step,
        "continuation_tokens": step * TOKENS_PER_STEP,
        "total_experience_tokens": 100_000_000 + step * TOKENS_PER_STEP,
        **model_structure(model),
        **domain_metrics,
        **summarize_diagnostics(cells, tissues),
    }
    tokens = step * TOKENS_PER_STEP
    for row in cells:
        row.update({"granularity": granularity, "continuation_tokens": tokens})
    for row in tissues:
        row.update({"granularity": granularity, "continuation_tokens": tokens})
    return metric, cells, tissues, profiles


def _resume_payload(
    *,
    granularity: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    step: int,
    metrics: list[dict[str, object]],
    cell_rows: list[dict[str, object]],
    tissue_rows: list[dict[str, object]],
    baseline_profiles: dict[str, dict[str, float]] | None,
    age_zero: dict[str, float] | None,
    elapsed: float,
    provenance: dict[str, object],
) -> dict[str, object]:
    return {
        "format": WORKER_FORMAT,
        "granularity": granularity,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "step": step,
        "metrics": metrics,
        "cell_rows": cell_rows,
        "tissue_rows": tissue_rows,
        "baseline_profiles": baseline_profiles,
        "age_zero_parity": age_zero,
        "accumulated_elapsed_seconds": elapsed,
        "provenance": provenance,
    }


def main() -> int:
    args = parser()
    if args.continuation_tokens <= 0 or args.continuation_tokens > CONTINUATION_TOKENS:
        raise ValueError("--continuation-tokens must be in (0, 20M]")
    if args.continuation_tokens % TOKENS_PER_STEP != 0:
        raise ValueError(
            "--continuation-tokens must be a multiple of TOKENS_PER_STEP"
        )
    if args.max_wall_hours <= 0:
        raise ValueError("--max-wall-hours must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("formal Experiment 026 worker requires CUDA")

    provenance = git_identity()
    device = torch.device("cuda")
    granularity = int(args.granularity)
    arm_dir = args.output_dir / f"g{granularity}"
    arm_dir.mkdir(parents=True, exist_ok=True)
    resume_path = arm_dir / "resume.pt"
    if args.reset and resume_path.exists():
        resume_path.unlink()

    tokenizer = load_tokenizer(args.tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    train_streams = {
        "story": _memmap(args.story_train),
        "math": _memmap(args.math_train),
        "symbolic": _memmap(args.symbolic_train),
        "facts": _memmap(args.facts_train),
    }
    validation_streams = {
        "story": _memmap(args.story_validation),
        "math": _memmap(args.math_validation),
        "symbolic": _memmap(args.symbolic_validation),
        "facts": _memmap(args.facts_validation),
    }
    validation_starts = {
        domain: fixed_validation_starts(len(stream), domain)
        for domain, stream in validation_streams.items()
    }
    probe_batches = diagnostic_batches(validation_streams, device=device)

    model, source_identity = build_granularity_model(
        ROOT / SOURCE_007_ARTIFACT,
        vocab_size=vocab_size,
        granularity=granularity,
        device=device,
    )
    optimizer, cell_group_index = build_local_plasticity_optimizer(
        model,
        lr=BASE_LR,
        betas=BETAS,
        weight_decay=WEIGHT_DECAY,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    target_steps = args.continuation_tokens // TOKENS_PER_STEP
    metrics: list[dict[str, object]] = []
    all_cell_rows: list[dict[str, object]] = []
    all_tissue_rows: list[dict[str, object]] = []
    baseline_profiles: dict[str, dict[str, float]] | None = None
    step = 0
    accumulated_elapsed = 0.0
    age_zero: dict[str, float] | None = None

    if resume_path.is_file():
        payload = torch.load(resume_path, map_location=device, weights_only=False)
        if payload.get("format") != WORKER_FORMAT:
            raise RuntimeError("unsupported Experiment-026 resume format")
        if int(payload.get("granularity", -1)) != granularity:
            raise RuntimeError("resume granularity mismatch")
        model.load_state_dict(payload["model_state"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state"])
        scaler.load_state_dict(payload["scaler_state"])
        step = int(payload["step"])
        metrics = list(payload.get("metrics", []))
        all_cell_rows = list(payload.get("cell_rows", []))
        all_tissue_rows = list(payload.get("tissue_rows", []))
        baseline_profiles = payload.get("baseline_profiles")
        accumulated_elapsed = float(
            payload.get("accumulated_elapsed_seconds", 0.0)
        )
        age_zero = payload.get("age_zero_parity")
        print(
            f"resumed g{granularity} at "
            f"{step * TOKENS_PER_STEP:,} continuation tokens"
        )

    invocation_started = time.monotonic()
    deadline = invocation_started + args.max_wall_hours * 3600.0
    torch.cuda.reset_peak_memory_stats(device)

    if age_zero is None:
        age_zero = _age_zero_parity(
            model,
            vocab_size=vocab_size,
            inputs=probe_batches["story"].inputs,
            device=device,
        )
        _json_write(arm_dir / "age-zero-parity.json", age_zero)

    eval_steps = {
        token // TOKENS_PER_STEP
        for token in EVAL_TOKENS
        if token <= args.continuation_tokens
    }
    if step == 0 and not metrics:
        metric, cells, tissues, profiles = _evaluate_checkpoint(
            model,
            granularity=granularity,
            step=0,
            validation_streams=validation_streams,
            validation_starts=validation_starts,
            probe_batches=probe_batches,
            baseline_profiles=None,
            device=device,
        )
        baseline_profiles = profiles
        metrics.append(metric)
        all_cell_rows.extend(cells)
        all_tissue_rows.extend(tissues)
        _save_tables(arm_dir, metrics, all_cell_rows, all_tissue_rows)
        print(
            f"g{granularity} age0 balanced_nll={metric['balanced_nll']:.4f} "
            f"specialization={metric['median_cell_specialization']:.4f}"
        )

    last_progress_tokens = step * TOKENS_PER_STEP
    last_resume_tokens = step * TOKENS_PER_STEP
    last_loss = float("nan")
    last_domain = ""
    last_plasticity = {
        "mean_plasticity": 1.0,
        "min_plasticity": 1.0,
        "max_plasticity": 1.0,
        "mean_local_pressure": 1.0,
    }
    stop_reason = "complete"

    while step < target_steps:
        if time.monotonic() >= deadline:
            stop_reason = "worker_soft_wall_limit"
            break
        domain, inputs, targets = continuation_batch(
            step,
            streams=train_streams,
            device=device,
        )
        scheduled_lr = continuation_lr(step + 1, total_steps=target_steps)
        set_global_lr(
            optimizer,
            cell_group_index,
            base_lr=scheduled_lr,
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            output = model(inputs)
            loss = F.cross_entropy(
                output.logits.reshape(-1, output.logits.shape[-1]),
                targets.reshape(-1),
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        plasticity_rows = update_local_plasticity(
            model,
            optimizer,
            cell_group_index,
            base_lr=scheduled_lr,
            config=PLASTICITY_CONFIG,
        )
        last_plasticity = plasticity_summary(plasticity_rows)
        scaler.step(optimizer)
        scaler.update()

        step += 1
        last_loss = float(loss.detach().item())
        last_domain = domain
        tokens = step * TOKENS_PER_STEP

        if (
            tokens - last_progress_tokens >= PROGRESS_INTERVAL_TOKENS
            or step == target_steps
        ):
            elapsed = accumulated_elapsed + (time.monotonic() - invocation_started)
            print(
                f"g{granularity} tokens={tokens:,}/{args.continuation_tokens:,} "
                f"domain={domain} loss={last_loss:.4f} lr={scheduled_lr:.2e} "
                f"plasticity=[{last_plasticity['min_plasticity']:.3f},"
                f"{last_plasticity['max_plasticity']:.3f}] "
                f"elapsed={elapsed / 3600:.2f}h"
            )
            last_progress_tokens = tokens

        if step in eval_steps and (
            not metrics
            or int(metrics[-1]["continuation_tokens"]) != tokens
        ):
            metric, cells, tissues, _ = _evaluate_checkpoint(
                model,
                granularity=granularity,
                step=step,
                validation_streams=validation_streams,
                validation_starts=validation_starts,
                probe_batches=probe_batches,
                baseline_profiles=baseline_profiles,
                device=device,
            )
            metrics.append(metric)
            all_cell_rows.extend(cells)
            all_tissue_rows.extend(tissues)
            _save_tables(arm_dir, metrics, all_cell_rows, all_tissue_rows)
            print(
                f"g{granularity} eval@{tokens:,}: "
                f"balanced_nll={metric['balanced_nll']:.4f} "
                f"specialization={metric['median_cell_specialization']:.4f} "
                f"plasticity_std={metric['plasticity_std']:.4f}"
            )

        if (
            tokens - last_resume_tokens >= RESUME_INTERVAL_TOKENS
            or step == target_steps
        ):
            elapsed = accumulated_elapsed + (time.monotonic() - invocation_started)
            _atomic_torch_save(
                _resume_payload(
                    granularity=granularity,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    step=step,
                    metrics=metrics,
                    cell_rows=all_cell_rows,
                    tissue_rows=all_tissue_rows,
                    baseline_profiles=baseline_profiles,
                    age_zero=age_zero,
                    elapsed=elapsed,
                    provenance=provenance,
                ),
                resume_path,
            )
            last_resume_tokens = tokens

    complete = step >= target_steps
    total_elapsed = accumulated_elapsed + (time.monotonic() - invocation_started)
    if not complete:
        _atomic_torch_save(
            _resume_payload(
                granularity=granularity,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                step=step,
                metrics=metrics,
                cell_rows=all_cell_rows,
                tissue_rows=all_tissue_rows,
                baseline_profiles=baseline_profiles,
                age_zero=age_zero,
                elapsed=total_elapsed,
                provenance=provenance,
            ),
            resume_path,
        )

    _save_tables(arm_dir, metrics, all_cell_rows, all_tissue_rows)
    summary = {
        "format": WORKER_FORMAT,
        "experiment_format": FORMAT,
        "granularity": granularity,
        "complete": complete,
        "reason": stop_reason,
        "continuation_tokens": step * TOKENS_PER_STEP,
        "target_continuation_tokens": args.continuation_tokens,
        "last_loss": last_loss,
        "last_domain": last_domain,
        "last_plasticity": last_plasticity,
        "local_plasticity_config": {
            "ema_decay": PLASTICITY_CONFIG.ema_decay,
            "pressure_exponent": PLASTICITY_CONFIG.pressure_exponent,
            "minimum": PLASTICITY_CONFIG.minimum,
            "maximum": PLASTICITY_CONFIG.maximum,
        },
        "elapsed_seconds": total_elapsed,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
        "source": source_identity,
        "age_zero_parity": age_zero,
        "structure": model_structure(model),
        "final_metrics": metrics[-1] if metrics else None,
        "provenance": provenance,
    }
    _json_write(arm_dir / "worker-summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
