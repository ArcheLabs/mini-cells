#!/usr/bin/env python3
"""Formal compatibility entrypoint for the CLM-0.3 release bridge worker.

The core worker intentionally retains its resumable training state until the
final benchmark. This entrypoint keeps the formal worker boundary strict while
applying two compatibility fixes that do not change experiment semantics:

1. Re-export the canonical ``BRIDGE_TOKENS_PER_STEP`` constant into the loaded
   worker module. The worker uses the symbol for checkpoint/resume accounting,
   but an earlier release-benchmark commit forgot to import it.
2. Remove residual parameter gradients before measuring inference VRAM so the
   public runtime metric describes inference rather than the final training
   step.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESEARCH = ROOT / "research"
if str(RESEARCH) not in sys.path:
    sys.path.insert(0, str(RESEARCH))

from minicells.clm_release_benchmark import BRIDGE_TOKENS_PER_STEP

WORKER = HERE / "run_clm_0_3_release_bridge_worker.py"
spec = importlib.util.spec_from_file_location("clm_0_3_release_bridge_worker_core", WORKER)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load release bridge worker: {WORKER}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# The canonical value is derived from BRIDGE_BATCH_SIZE * BRIDGE_SEQUENCE_LENGTH
# in clm_release_benchmark.py. Injecting the missing import here preserves the
# already-frozen worker semantics while making the formal entrypoint executable.
module.BRIDGE_TOKENS_PER_STEP = BRIDGE_TOKENS_PER_STEP
if module.BRIDGE_TOKENS_PER_STEP != module.BRIDGE_BATCH_SIZE * module.BRIDGE_SEQUENCE_LENGTH:
    raise RuntimeError("release bridge tokens-per-step constant is inconsistent")

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
