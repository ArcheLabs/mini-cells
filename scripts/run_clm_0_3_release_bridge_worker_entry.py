#!/usr/bin/env python3
"""Formal compatibility entrypoint for the CLM-0.3 release bridge worker.

The core worker intentionally retains its resumable training state until the
final benchmark.  This entrypoint removes residual parameter gradients before
measuring inference VRAM so the public runtime metric describes inference
rather than the final training step.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
WORKER = HERE / "run_clm_0_3_release_bridge_worker.py"
spec = importlib.util.spec_from_file_location("clm_0_3_release_bridge_worker_core", WORKER)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load release bridge worker: {WORKER}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

_original_benchmark = module._benchmark_inference


def _clean_inference_benchmark(model, validation_stream, starts, *, device, **kwargs):
    model.zero_grad(set_to_none=True)
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
