#!/usr/bin/env python3
"""Compatibility and optimized-runtime entrypoint for the CLM-0.3 bridge worker.

The historical release-benchmark result remains pinned to its original code.
This branch keeps the same model semantics while installing the new optional
CLM runtime for engineering measurements:

1. Re-export the canonical ``BRIDGE_TOKENS_PER_STEP`` constant into the loaded
   worker module.
2. Install the autotuned CLM runtime on ProgressiveGrowthCLM instances.
   Inference requests ``sparse_dispatch`` and chooses the fastest supported
   exact-forward runtime for the current GPU/shape.
3. Training remains ``masked_dense`` by default. Setting
   ``MINICELLS_CLM_TRAIN_BACKEND=batched_dense`` opts into the mathematically
   equivalent batched expert implementation after its gradient regression
   tests pass in the target environment.
4. Remove residual parameter gradients and benchmark under inference mode.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESEARCH = ROOT / "research"
if str(RESEARCH) not in sys.path:
    sys.path.insert(0, str(RESEARCH))

from minicells.clm_release_benchmark import BRIDGE_TOKENS_PER_STEP
from minicells.clm_sparse_runtime import install_optimized_runtime

WORKER = HERE / "run_clm_0_3_release_bridge_worker.py"
spec = importlib.util.spec_from_file_location("clm_0_3_release_bridge_worker_core", WORKER)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load release bridge worker: {WORKER}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

module.BRIDGE_TOKENS_PER_STEP = BRIDGE_TOKENS_PER_STEP
if module.BRIDGE_TOKENS_PER_STEP != module.BRIDGE_BATCH_SIZE * module.BRIDGE_SEQUENCE_LENGTH:
    raise RuntimeError("release bridge tokens-per-step constant is inconsistent")

TRAIN_BACKEND = os.environ.get("MINICELLS_CLM_TRAIN_BACKEND", "masked_dense")
if TRAIN_BACKEND not in {"masked_dense", "batched_dense"}:
    raise RuntimeError(
        "MINICELLS_CLM_TRAIN_BACKEND must be 'masked_dense' or 'batched_dense'"
    )

_original_build_bridge_model = module.build_bridge_model


def _build_bridge_model_with_runtime(*args, **kwargs):
    model = _original_build_bridge_model(*args, **kwargs)
    if isinstance(model, module.ProgressiveGrowthCLM):
        install_optimized_runtime(model)
    return model


module.build_bridge_model = _build_bridge_model_with_runtime

_original_forward = module._forward


def _runtime_forward(model, inputs, *, train: bool):
    if isinstance(model, module.ProgressiveGrowthCLM):
        backend = TRAIN_BACKEND if train else "sparse_dispatch"
        return model(inputs, execution_backend=backend)
    return _original_forward(model, inputs, train=train)


module._forward = _runtime_forward

_original_benchmark = module._benchmark_inference


def _clean_inference_benchmark(model, validation_stream, starts, *, device, **kwargs):
    model.zero_grad(set_to_none=True)
    with torch.inference_mode():
        return _original_benchmark(
            model,
            validation_stream,
            starts,
            device=device,
            **kwargs,
        )


module._benchmark_inference = _clean_inference_benchmark

if __name__ == "__main__":
    raise SystemExit(module.main())
