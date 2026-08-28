#!/usr/bin/env python3
"""Microbenchmark the CLM execution backends on the frozen Experiment-006 source.

This script requires no dataset download. It loads the repository-pinned 10M
TextNCA checkpoint, function-preservingly upcycles it to CLM fixed4, and
compares the old and new execution paths on identical synthetic token batches.

The benchmark is intentionally separate from release evidence: it measures
runtime engineering only and does not make a language-quality claim.
"""

from __future__ import annotations

import argparse
import copy
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
from minicells.clm_sparse_runtime import install_optimized_runtime, runtime_status


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Benchmark CLM sparse runtime")
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


def _bench_inference(model, inputs: torch.Tensor, backend: str, *, warmup: int, iterations: int) -> dict[str, float]:
    model.eval()
    amp = inputs.device.type == "cuda"
    with torch.no_grad():
        for _ in range(warmup):
            with torch.autocast(device_type=inputs.device.type, dtype=torch.float16, enabled=amp):
                model(inputs, execution_backend=backend)
    _sync(inputs.device)
    if inputs.device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(inputs.device)
    started = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            with torch.autocast(device_type=inputs.device.type, dtype=torch.float16, enabled=amp):
                model(inputs, execution_backend=backend)
    _sync(inputs.device)
    elapsed = max(time.perf_counter() - started, 1e-9)
    tokens = int(inputs.numel()) * iterations
    peak = int(torch.cuda.max_memory_allocated(inputs.device)) if inputs.device.type == "cuda" else 0
    return {
        "tokens_per_second": tokens / elapsed,
        "time_per_token_us": elapsed * 1e6 / tokens,
        "peak_vram_bytes": peak,
    }


def _bench_training(model, inputs: torch.Tensor, targets: torch.Tensor, backend: str, *, iterations: int) -> dict[str, float]:
    model.train()
    amp = inputs.device.type == "cuda"
    # Forward/backward only: optimizer updates would make the compared models
    # diverge and would measure AdamW rather than the expert execution backend.
    for _ in range(3):
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type=inputs.device.type, dtype=torch.float16, enabled=amp):
            logits = model(inputs, execution_backend=backend).logits
            loss = F.cross_entropy(logits.flatten(0, 1), targets.reshape(-1))
        loss.backward()
    _sync(inputs.device)
    if inputs.device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(inputs.device)
    started = time.perf_counter()
    for _ in range(iterations):
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type=inputs.device.type, dtype=torch.float16, enabled=amp):
            logits = model(inputs, execution_backend=backend).logits
            loss = F.cross_entropy(logits.flatten(0, 1), targets.reshape(-1))
        loss.backward()
    _sync(inputs.device)
    elapsed = max(time.perf_counter() - started, 1e-9)
    tokens = int(inputs.numel()) * iterations
    peak = int(torch.cuda.max_memory_allocated(inputs.device)) if inputs.device.type == "cuda" else 0
    return {
        "tokens_per_second": tokens / elapsed,
        "time_per_token_us": elapsed * 1e6 / tokens,
        "peak_vram_bytes": peak,
    }


def main() -> int:
    args = parser().parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if min(args.batch_size, args.sequence_length, args.warmup, args.iterations, args.train_iterations) <= 0:
        raise ValueError("benchmark dimensions and iteration counts must be positive")

    torch.manual_seed(61003)
    source_checkpoint = ROOT / SOURCE_006_CHECKPOINT
    model = build_bridge_model(
        "clm_fixed4",
        source_checkpoint,
        vocab_size=2048,
        device=device,
    )
    install_optimized_runtime(model)
    inputs = torch.randint(0, 2048, (args.batch_size, args.sequence_length), device=device)
    targets = torch.randint(0, 2048, (args.batch_size, args.sequence_length), device=device)

    # Numerical parity before timing.
    model.eval()
    with torch.no_grad():
        dense = model(inputs, execution_backend="masked_dense").logits.float()
        reference = model(inputs, execution_backend="reference_sparse").logits.float()
        optimized = model(inputs, execution_backend="sparse_dispatch").logits.float()
    parity = {
        "reference_sparse_max_abs_diff": float((dense - reference).abs().max().item()),
        "optimized_sparse_max_abs_diff": float((dense - optimized).abs().max().item()),
    }

    inference = {}
    for backend in ("masked_dense", "reference_sparse", "sparse_dispatch"):
        inference[backend] = _bench_inference(
            model,
            inputs,
            backend,
            warmup=args.warmup,
            iterations=args.iterations,
        )

    baseline_train = copy.deepcopy(model)
    batched_train = copy.deepcopy(model)
    install_optimized_runtime(baseline_train)
    install_optimized_runtime(batched_train)
    training = {
        "masked_dense": _bench_training(
            baseline_train, inputs, targets, "masked_dense", iterations=args.train_iterations
        ),
        "batched_dense": _bench_training(
            batched_train, inputs, targets, "batched_dense", iterations=args.train_iterations
        ),
    }

    reference_tps = inference["reference_sparse"]["tokens_per_second"]
    grouped_tps = inference["sparse_dispatch"]["tokens_per_second"]
    masked_train_tps = training["masked_dense"]["tokens_per_second"]
    batched_train_tps = training["batched_dense"]["tokens_per_second"]
    result = {
        "format": "minicells.clm-sparse-runtime-benchmark.v1",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "tokens_per_forward": int(inputs.numel()),
        "parity": parity,
        "inference": inference,
        "training": training,
        "speedups": {
            "grouped_sparse_over_reference_sparse": grouped_tps / reference_tps,
            "batched_dense_train_over_masked_dense": batched_train_tps / masked_train_tps,
        },
        "runtime_status": runtime_status(model),
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
