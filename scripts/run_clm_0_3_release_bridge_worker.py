#!/usr/bin/env python3
"""Run one TextNCA-to-CLM machinery-bridge arm on one visible GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
if str(RESEARCH) not in sys.path:
    sys.path.insert(0, str(RESEARCH))

from minicells.clm_growth import ProgressiveGrowthCLM  # noqa: E402
from minicells.clm_release_benchmark import (  # noqa: E402
    AGE_ZERO_MAX_LOGITS_DIFF,
    AGE_ZERO_PPL_RATIO_TOLERANCE,
    BRIDGE_ARMS,
    BRIDGE_BASE_LR,
    BRIDGE_BATCH_SIZE,
    BRIDGE_BETAS,
    BRIDGE_BUDGET_TOKENS,
    BRIDGE_CHECKPOINT_TOKENS,
    BRIDGE_GRAD_CLIP,
    BRIDGE_KL_BETA,
    BRIDGE_MODEL_SEED,
    BRIDGE_SCHEDULE_SEED,
    BRIDGE_SEQUENCE_LENGTH,
    BRIDGE_STATE_INTERVAL_TOKENS,
    BRIDGE_TRAIN_PREFIX_OFFSET,
    BRIDGE_VALIDATION_BATCHES,
    BRIDGE_VALIDATION_BATCH_SIZE,
    BRIDGE_VALIDATION_SEED,
    BRIDGE_VALIDATION_SEQUENCE_LENGTH,
    BRIDGE_WEIGHT_DECAY,
    BRIDGE_WORKER_FORMAT,
    SOURCE_006_CHECKPOINT,
    bridge_lr,
    build_bridge_model,
    clm_parameter_breakdown,
    dense_parameter_breakdown,
    json_digest,
    load_source_textnca,
    sha256_file,
    verify_source_checkpoint,
)
from minicells.growth_validation import student_teacher_kl  # noqa: E402
from minicells.language_data import (  # noqa: E402
    batch_from_starts,
    fixed_validation_starts,
    load_tokenizer,
    make_training_schedule,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run one CLM-0.3 release bridge arm")
    result.add_argument("--arm", choices=BRIDGE_ARMS, required=True)
    result.add_argument("--cache-dir", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    result.add_argument("--execute", action="store_true")
    return result


def _git_provenance() -> dict[str, object]:
    def run(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    return {
        "code_commit": run("rev-parse", "HEAD"),
        "code_tree_sha": run("rev-parse", "HEAD^{tree}"),
        "tracked_tree_dirty": bool(run("status", "--porcelain", "--untracked-files=no")),
    }


def _json_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _forward(model: torch.nn.Module, inputs: torch.Tensor, *, train: bool) -> Any:
    if isinstance(model, ProgressiveGrowthCLM):
        return model(inputs, execution_backend="masked_dense" if train else "sparse_dispatch")
    return model(inputs)


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    validation_stream: torch.Tensor,
    starts: tuple[tuple[int, ...], ...],
    *,
    device: torch.device,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    amp = device.type == "cuda"
    total_loss = 0.0
    total_tokens = 0
    for batch_starts in starts:
        inputs, targets = batch_from_starts(
            validation_stream,
            batch_starts,
            BRIDGE_VALIDATION_SEQUENCE_LENGTH,
            device,
        )
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            output = _forward(model, inputs, train=False)
            loss = F.cross_entropy(
                output.logits.reshape(-1, output.logits.shape[-1]),
                targets.reshape(-1),
                reduction="sum",
            )
        total_loss += float(loss.item())
        total_tokens += int(targets.numel())
    if was_training:
        model.train()
    nll = total_loss / max(total_tokens, 1)
    return {
        "validation_nll": nll,
        "validation_ppl": math.exp(min(nll, 20.0)),
        "validation_tokens": total_tokens,
    }


@torch.no_grad()
def _age_zero_parity(
    clm: ProgressiveGrowthCLM,
    source: torch.nn.Module,
    validation_stream: torch.Tensor,
    starts: tuple[tuple[int, ...], ...],
    *,
    device: torch.device,
) -> dict[str, object]:
    batch_starts = starts[0]
    inputs, targets = batch_from_starts(
        validation_stream,
        batch_starts,
        BRIDGE_VALIDATION_SEQUENCE_LENGTH,
        device,
    )
    clm.eval()
    source.eval()
    source_logits = source(inputs).logits
    clm_logits = clm(inputs, execution_backend="masked_dense").logits
    logits_diff = float((source_logits - clm_logits).abs().max().item())
    source_nll = F.cross_entropy(source_logits.flatten(0, 1), targets.reshape(-1))
    clm_nll = F.cross_entropy(clm_logits.flatten(0, 1), targets.reshape(-1))
    ppl_ratio = math.exp(float(clm_nll - source_nll))
    passed = (
        abs(ppl_ratio - 1.0) <= AGE_ZERO_PPL_RATIO_TOLERANCE
        and logits_diff <= AGE_ZERO_MAX_LOGITS_DIFF
    )
    return {
        "status": "CLM_RELEASE_BRIDGE_EQUIVALENCE" if passed else "CLM_RELEASE_BRIDGE_EQUIVALENCE_FAILURE",
        "ppl_ratio": ppl_ratio,
        "max_logits_abs_diff": logits_diff,
        "ppl_ratio_tolerance": AGE_ZERO_PPL_RATIO_TOLERANCE,
        "max_logits_diff_threshold": AGE_ZERO_MAX_LOGITS_DIFF,
    }


def _save_state(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    step: int,
    train_elapsed_seconds: float,
    rows: list[dict[str, object]],
    identity: dict[str, object],
) -> None:
    torch.save(
        {
            "format": "minicells.clm-0.3-release-bridge-state.v1",
            "identity": identity,
            "step": int(step),
            "train_elapsed_seconds": float(train_elapsed_seconds),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "rows": rows,
        },
        path,
    )


def _load_state(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    identity: dict[str, object],
) -> tuple[int, float, list[dict[str, object]]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "minicells.clm-0.3-release-bridge-state.v1":
        raise RuntimeError("unsupported bridge resume state")
    if payload.get("identity") != identity:
        raise RuntimeError("bridge resume state uses different formal semantics or code provenance")
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    scaler.load_state_dict(payload["scaler_state"])
    return (
        int(payload["step"]),
        float(payload.get("train_elapsed_seconds", 0.0)),
        list(payload.get("rows", [])),
    )


def _benchmark_inference(
    model: torch.nn.Module,
    validation_stream: torch.Tensor,
    starts: tuple[tuple[int, ...], ...],
    *,
    device: torch.device,
    warmup_iterations: int = 20,
    timed_iterations: int = 100,
) -> tuple[float, int]:
    model.eval()
    amp = device.type == "cuda"
    prepared = [
        batch_from_starts(
            validation_stream,
            batch_starts,
            BRIDGE_VALIDATION_SEQUENCE_LENGTH,
            device,
        )[0]
        for batch_starts in starts[: min(8, len(starts))]
    ]
    if not prepared:
        raise RuntimeError("no inference benchmark batches")

    with torch.no_grad():
        for index in range(warmup_iterations):
            inputs = prepared[index % len(prepared)]
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                _forward(model, inputs, train=False)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.no_grad():
        for index in range(timed_iterations):
            inputs = prepared[index % len(prepared)]
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                _forward(model, inputs, train=False)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = max(time.perf_counter() - started, 1e-9)
    tokens = timed_iterations * int(prepared[0].numel())
    throughput = tokens / elapsed
    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    return throughput, peak


def run(args: argparse.Namespace) -> int:
    provenance = _git_provenance()
    if provenance["tracked_tree_dirty"]:
        raise RuntimeError("formal release bridge requires a clean tracked Git tree")
    if not args.execute:
        print({"mode": "preflight_only", "arm": args.arm, **provenance})
        return 0

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(BRIDGE_MODEL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(BRIDGE_MODEL_SEED)

    cache = args.cache_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    tokenizer_path = cache / "tokenizer.json"
    train_path = cache / "train-tokens.pt"
    validation_path = cache / "validation-tokens.pt"
    manifest_path = cache / "corpus-manifest.json"
    for path in (tokenizer_path, train_path, validation_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"release bridge cache is incomplete: {path}")

    tokenizer = load_tokenizer(tokenizer_path)
    train_stream = torch.load(train_path, map_location="cpu")
    validation_stream = torch.load(validation_path, map_location="cpu")
    corpus_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(train_stream.numel()) <= BRIDGE_TRAIN_PREFIX_OFFSET + BRIDGE_SEQUENCE_LENGTH + 1:
        raise RuntimeError("materialized TinyStories stream does not contain a post-006 suffix")
    suffix = train_stream[BRIDGE_TRAIN_PREFIX_OFFSET:]
    schedule = make_training_schedule(
        int(suffix.numel()),
        seed=BRIDGE_SCHEDULE_SEED,
        budget_tokens=BRIDGE_BUDGET_TOKENS,
        batch_size=BRIDGE_BATCH_SIZE,
        sequence_length=BRIDGE_SEQUENCE_LENGTH,
    )
    validation_starts = fixed_validation_starts(
        int(validation_stream.numel()),
        batches=BRIDGE_VALIDATION_BATCHES,
        batch_size=BRIDGE_VALIDATION_BATCH_SIZE,
        sequence_length=BRIDGE_VALIDATION_SEQUENCE_LENGTH,
        seed=BRIDGE_VALIDATION_SEED,
    )
    source_checkpoint = ROOT / SOURCE_006_CHECKPOINT
    observed_source_sha = verify_source_checkpoint(source_checkpoint)

    identity = {
        "format": BRIDGE_WORKER_FORMAT,
        "arm": args.arm,
        **provenance,
        "source_checkpoint_sha256": observed_source_sha,
        "source_checkpoint_path": SOURCE_006_CHECKPOINT.as_posix(),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "full_train_token_sha256": corpus_manifest.get("train_token_sha256"),
        "validation_token_sha256": corpus_manifest.get("validation_token_sha256"),
        "train_suffix_offset": BRIDGE_TRAIN_PREFIX_OFFSET,
        "train_suffix_sha256": _tensor_sha256(suffix),
        "schedule_sha256": json_digest(schedule.starts),
        "validation_schedule_sha256": json_digest(validation_starts),
        "budget_tokens": BRIDGE_BUDGET_TOKENS,
        "batch_size": BRIDGE_BATCH_SIZE,
        "sequence_length": BRIDGE_SEQUENCE_LENGTH,
        "objective": "CE + 0.5 * KL(student || frozen source TextNCA)",
        "kl_beta": BRIDGE_KL_BETA,
        "optimizer": "AdamW",
        "optimizer_betas": list(BRIDGE_BETAS),
        "weight_decay": BRIDGE_WEIGHT_DECAY,
        "base_lr": BRIDGE_BASE_LR,
    }
    identity_path = output / "run-provenance.json"
    if identity_path.exists():
        existing = json.loads(identity_path.read_text(encoding="utf-8"))
        if existing != identity:
            raise RuntimeError("existing bridge evidence uses different code or formal semantics; restart it")
    _json_write(identity_path, identity)

    model = build_bridge_model(
        args.arm,
        source_checkpoint,
        vocab_size=tokenizer.get_vocab_size(),
        device=device,
    )
    teacher = load_source_textnca(
        source_checkpoint,
        vocab_size=tokenizer.get_vocab_size(),
        device=device,
    ).eval().requires_grad_(False)

    if isinstance(model, ProgressiveGrowthCLM):
        parameters = clm_parameter_breakdown(model)
        age_zero_equivalence = _age_zero_parity(
            model, teacher, validation_stream, validation_starts, device=device
        )
        if age_zero_equivalence["status"] != "CLM_RELEASE_BRIDGE_EQUIVALENCE":
            _json_write(output / "age-zero-equivalence.json", age_zero_equivalence)
            raise RuntimeError(f"CLM release bridge equivalence failed: {age_zero_equivalence}")
    else:
        parameters = dense_parameter_breakdown(model)
        age_zero_equivalence = None

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=BRIDGE_BASE_LR,
        betas=BRIDGE_BETAS,
        weight_decay=BRIDGE_WEIGHT_DECAY,
    )
    amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    state_path = output / "bridge-state.pt"
    rows: list[dict[str, object]] = []
    start_step = 0
    train_elapsed = 0.0
    if state_path.exists():
        start_step, train_elapsed, rows = _load_state(
            state_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            identity=identity,
        )
        model.to(device)
        for state in optimizer.state.values():
            for key, value in list(state.items()):
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        print(f"[{args.arm}] resumed at {start_step * BRIDGE_TOKENS_PER_STEP:,} tokens", flush=True)

    if not any(int(row["consumed_tokens"]) == 0 for row in rows):
        evaluation = _evaluate(model, validation_stream, validation_starts, device=device)
        rows.append(
            {
                "arm": args.arm,
                "step": 0,
                "consumed_tokens": 0,
                "train_elapsed_seconds": train_elapsed,
                "train_tokens_per_second": 0.0,
                **evaluation,
            }
        )
        rows.sort(key=lambda row: int(row["consumed_tokens"]))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    total_steps = schedule.steps
    checkpoint_steps = {
        tokens // BRIDGE_TOKENS_PER_STEP
        for tokens in BRIDGE_CHECKPOINT_TOKENS
        if tokens > 0
    }
    state_interval_steps = BRIDGE_STATE_INTERVAL_TOKENS // BRIDGE_TOKENS_PER_STEP
    segment_started: float | None = None
    if start_step < total_steps:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        segment_started = time.perf_counter()

    for zero_step in range(start_step, total_steps):
        step = zero_step + 1
        model.train()
        lr = bridge_lr(step, total_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        inputs, targets = batch_from_starts(
            suffix,
            schedule.starts[zero_step],
            BRIDGE_SEQUENCE_LENGTH,
            device,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            student = _forward(model, inputs, train=True)
            with torch.no_grad():
                teacher_output = teacher(inputs)
            ce = F.cross_entropy(
                student.logits.reshape(-1, student.logits.shape[-1]), targets.reshape(-1)
            )
            kl = student_teacher_kl(student.logits, teacher_output.logits)
            loss = ce + BRIDGE_KL_BETA * kl
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), BRIDGE_GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        should_evaluate = step in checkpoint_steps
        should_save = should_evaluate or step % state_interval_steps == 0 or step == total_steps
        if should_save:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if segment_started is not None:
                train_elapsed += time.perf_counter() - segment_started
                segment_started = None
            consumed = step * BRIDGE_TOKENS_PER_STEP
            if should_evaluate:
                evaluation = _evaluate(model, validation_stream, validation_starts, device=device)
                rows = [row for row in rows if int(row["consumed_tokens"]) != consumed]
                rows.append(
                    {
                        "arm": args.arm,
                        "step": step,
                        "consumed_tokens": consumed,
                        "train_loss": float(loss.detach().item()),
                        "learning_rate": lr,
                        "train_elapsed_seconds": train_elapsed,
                        "train_tokens_per_second": consumed / max(train_elapsed, 1e-9),
                        **evaluation,
                    }
                )
                rows.sort(key=lambda row: int(row["consumed_tokens"]))
                print(
                    f"[{args.arm}] {consumed:,} tokens | "
                    f"PPL={evaluation['validation_ppl']:.4f} | "
                    f"train={consumed / max(train_elapsed, 1e-9):.0f} tok/s",
                    flush=True,
                )
            _save_state(
                state_path,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                step=step,
                train_elapsed_seconds=train_elapsed,
                rows=rows,
                identity=identity,
            )
            if step < total_steps:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                segment_started = time.perf_counter()

    if len({int(row["consumed_tokens"]) for row in rows}) != len(BRIDGE_CHECKPOINT_TOKENS):
        raise RuntimeError("bridge worker did not produce every formal evaluation age")
    train_peak_vram = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    final_row = next(row for row in rows if int(row["consumed_tokens"]) == BRIDGE_BUDGET_TOKENS)

    del optimizer, scaler, teacher
    if device.type == "cuda":
        torch.cuda.empty_cache()
    inference_tps, inference_peak_vram = _benchmark_inference(
        model, validation_stream, validation_starts, device=device
    )

    checkpoints_path = output / "bridge-checkpoints.json"
    _json_write(checkpoints_path, rows)
    if age_zero_equivalence is not None:
        _json_write(output / "age-zero-equivalence.json", age_zero_equivalence)
    runtime = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "train_tokens_per_second": BRIDGE_BUDGET_TOKENS / max(train_elapsed, 1e-9),
        "train_elapsed_seconds": train_elapsed,
        "train_peak_vram_bytes": train_peak_vram,
        "inference_tokens_per_second": inference_tps,
        "inference_peak_vram_bytes": inference_peak_vram,
        "inference_backend": "sparse_dispatch" if isinstance(model, ProgressiveGrowthCLM) else "dense",
        "inference_warmup_iterations": 20,
        "inference_timed_iterations": 100,
    }
    result = {
        "format": BRIDGE_WORKER_FORMAT,
        "arm": args.arm,
        "formal_gpu_experiment_run": device.type == "cuda",
        "final_ppl": float(final_row["validation_ppl"]),
        "final_nll": float(final_row["validation_nll"]),
        "parameters": parameters,
        "runtime": runtime,
        "age_zero_equivalence": age_zero_equivalence,
        "checkpoint_ages": list(BRIDGE_CHECKPOINT_TOKENS),
        "source_checkpoint_sha256": observed_source_sha,
        **provenance,
    }
    _json_write(output / "worker-result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


def main() -> int:
    args = parser().parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
