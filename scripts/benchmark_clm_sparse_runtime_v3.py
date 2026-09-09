#!/usr/bin/env python3
"""Benchmark calibrated CLM sparse runtime v3 on the frozen release source."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
if str(RESEARCH) not in sys.path:
    sys.path.insert(0, str(RESEARCH))

from minicells.clm_release_benchmark import SOURCE_006_CHECKPOINT, build_bridge_model
from minicells.clm_sparse_runtime_v3 import install_optimized_runtime, runtime_status


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Benchmark CLM sparse runtime v3")
    result.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    result.add_argument("--batch-size", type=int, default=8)
    result.add_argument("--sequence-length", type=int, default=128)
    result.add_argument("--warmup", type=int, default=20)
    result.add_argument("--iterations", type=int, default=100)
    result.add_argument("--train-iterations", type=int, default=20)
    result.add_argument("--output", type=Path)
    return result


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _forward(model, inputs: torch.Tensor, backend: str | None):
    if backend is None:
        return model(inputs).logits
    return model(inputs, execution_backend=backend).logits


def _numerical_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    ref = reference.float()
    cand = candidate.float()
    diff = cand - ref
    abs_diff = diff.abs()
    max_abs = float(abs_diff.max().item()) if diff.numel() else 0.0
    rms = float(diff.square().mean().sqrt().item()) if diff.numel() else 0.0
    rel_l2 = float(diff.norm().item()) / max(float(ref.norm().item()), 1e-12)
    top1 = float((cand.argmax(dim=-1) == ref.argmax(dim=-1)).float().mean().item())
    return {
        "max_abs_diff": max_abs,
        "rms_diff": rms,
        "relative_l2": rel_l2,
        "argmax_agreement": top1,
    }


def _bench_inference(
    model,
    inputs: torch.Tensor,
    backend: str | None,
    *,
    warmup: int,
    iterations: int,
) -> dict[str, float]:
    model.eval()
    amp = inputs.device.type == "cuda"
    with torch.inference_mode():
        for _ in range(warmup):
            with torch.autocast(device_type=inputs.device.type, dtype=torch.float16, enabled=amp):
                _forward(model, inputs, backend)
    _sync(inputs.device)
    if inputs.device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(inputs.device)
    started = time.perf_counter()
    with torch.inference_mode():
        for _ in range(iterations):
            with torch.autocast(device_type=inputs.device.type, dtype=torch.float16, enabled=amp):
                _forward(model, inputs, backend)
    _sync(inputs.device)
    elapsed = max(time.perf_counter() - started, 1e-9)
    tokens = int(inputs.numel()) * iterations
    peak = int(torch.cuda.max_memory_allocated(inputs.device)) if inputs.device.type == "cuda" else 0
    return {
        "tokens_per_second": tokens / elapsed,
        "time_per_token_us": elapsed * 1e6 / tokens,
        "peak_vram_bytes": peak,
    }


def _bench_training(
    model,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    backend: str | None,
    *,
    iterations: int,
) -> dict[str, float]:
    model.train()
    amp = inputs.device.type == "cuda"

    def step() -> None:
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type=inputs.device.type, dtype=torch.float16, enabled=amp):
            logits = _forward(model, inputs, backend)
            loss = F.cross_entropy(logits.flatten(0, 1), targets.reshape(-1))
        loss.backward()

    for _ in range(3):
        step()
    _sync(inputs.device)
    if inputs.device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(inputs.device)
    started = time.perf_counter()
    for _ in range(iterations):
        step()
    _sync(inputs.device)
    elapsed = max(time.perf_counter() - started, 1e-9)
    tokens = int(inputs.numel()) * iterations
    peak = int(torch.cuda.max_memory_allocated(inputs.device)) if inputs.device.type == "cuda" else 0
    return {
        "tokens_per_second": tokens / elapsed,
        "time_per_token_us": elapsed * 1e6 / tokens,
        "peak_vram_bytes": peak,
    }


def _build_models(source_checkpoint: Path, device: torch.device):
    textnca = build_bridge_model(
        "textnca_continuation",
        source_checkpoint,
        vocab_size=2048,
        device=device,
    )
    clm = build_bridge_model(
        "clm_fixed4",
        source_checkpoint,
        vocab_size=2048,
        device=device,
    )
    install_optimized_runtime(clm)
    return textnca, clm


def main() -> int:
    args = parser().parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if min(args.batch_size, args.sequence_length, args.warmup, args.iterations, args.train_iterations) <= 0:
        raise ValueError("benchmark dimensions and iteration counts must be positive")

    source_checkpoint = ROOT / SOURCE_006_CHECKPOINT
    dense, clm = _build_models(source_checkpoint, device)
    generator = torch.Generator(device=device).manual_seed(63003)
    inputs = torch.randint(
        0,
        2048,
        (args.batch_size, args.sequence_length),
        generator=generator,
        device=device,
    )
    targets = torch.randint(
        0,
        2048,
        (args.batch_size, args.sequence_length),
        generator=generator,
        device=device,
    )

    amp = device.type == "cuda"
    dense.eval()
    clm.eval()
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=amp,
    ):
        textnca_logits = dense(inputs).logits
        masked_logits = clm(inputs, execution_backend="masked_dense").logits
        reference_logits = clm(inputs, execution_backend="reference_sparse").logits
        optimized_logits = clm(inputs, execution_backend="sparse_dispatch").logits

    numerical = {
        "age_zero_clm_vs_textnca": _numerical_metrics(textnca_logits, masked_logits),
        "reference_sparse_vs_masked": _numerical_metrics(masked_logits, reference_logits),
        "optimized_sparse_vs_masked": _numerical_metrics(masked_logits, optimized_logits),
        "optimized_sparse_vs_reference": _numerical_metrics(reference_logits, optimized_logits),
    }
    numerical["optimized_vs_reference_nll_delta"] = float(
        F.cross_entropy(
            optimized_logits.float().flatten(0, 1),
            targets.reshape(-1),
        ).item()
        - F.cross_entropy(
            reference_logits.float().flatten(0, 1),
            targets.reshape(-1),
        ).item()
    )

    inference = {
        "textnca_dense": _bench_inference(
            dense, inputs, None, warmup=args.warmup, iterations=args.iterations
        ),
        "clm_masked_dense": _bench_inference(
            clm, inputs, "masked_dense", warmup=args.warmup, iterations=args.iterations
        ),
        "clm_reference_sparse": _bench_inference(
            clm, inputs, "reference_sparse", warmup=args.warmup, iterations=args.iterations
        ),
        "clm_sparse_dispatch_v3": _bench_inference(
            clm, inputs, "sparse_dispatch", warmup=args.warmup, iterations=args.iterations
        ),
    }
    inference_runtime_status = runtime_status(clm)

    dense_train, _ = _build_models(source_checkpoint, device)
    _, clm_masked_train = _build_models(source_checkpoint, device)
    _, clm_batched_train = _build_models(source_checkpoint, device)
    training = {
        "textnca_dense": _bench_training(
            dense_train, inputs, targets, None, iterations=args.train_iterations
        ),
        "clm_masked_dense": _bench_training(
            clm_masked_train,
            inputs,
            targets,
            "masked_dense",
            iterations=args.train_iterations,
        ),
        "clm_batched_dense": _bench_training(
            clm_batched_train,
            inputs,
            targets,
            "batched_dense",
            iterations=args.train_iterations,
        ),
    }

    textnca_inf = inference["textnca_dense"]["tokens_per_second"]
    masked_inf = inference["clm_masked_dense"]["tokens_per_second"]
    reference_inf = inference["clm_reference_sparse"]["tokens_per_second"]
    optimized_inf = inference["clm_sparse_dispatch_v3"]["tokens_per_second"]
    textnca_train = training["textnca_dense"]["tokens_per_second"]
    masked_train = training["clm_masked_dense"]["tokens_per_second"]
    batched_train = training["clm_batched_dense"]["tokens_per_second"]

    selected_backends: dict[str, int] = {}
    total_profiles = 0
    for stage in inference_runtime_status:
        for profile in stage.get("calibration_profiles", []):
            selected = str(profile["selected"])
            selected_backends[selected] = selected_backends.get(selected, 0) + 1
            total_profiles += 1

    result = {
        "format": "minicells.clm-sparse-runtime-benchmark.v3",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "tokens_per_forward": int(inputs.numel()),
        "numerical": numerical,
        "inference": inference,
        "training": training,
        "speedups": {
            "optimized_over_reference_sparse": optimized_inf / reference_inf,
            "optimized_over_clm_masked_dense": optimized_inf / masked_inf,
            "optimized_over_textnca_dense": optimized_inf / textnca_inf,
            "clm_inference_time_ratio_vs_textnca": textnca_inf / optimized_inf,
            "batched_dense_train_over_masked_dense": batched_train / masked_train,
            "batched_clm_train_over_textnca": batched_train / textnca_train,
            "clm_train_time_ratio_vs_textnca": textnca_train / batched_train,
        },
        "calibration_summary": {
            "total_profiles": total_profiles,
            "selected_backends": selected_backends,
        },
        "runtime_status": inference_runtime_status,
    }

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
