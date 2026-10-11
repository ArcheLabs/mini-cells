"""Verify Kaggle GPU execution and emit a small, machine-readable result."""

from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle did not provide a usable CUDA device.")

    device = torch.device("cuda:0")
    torch.cuda.init()
    properties = torch.cuda.get_device_properties(device)

    size = 512
    left = torch.randn((size, size), device=device)
    right = torch.randn((size, size), device=device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    product = left @ right
    torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - started) * 1000
    all_finite = bool(torch.isfinite(product).all().item())
    if not all_finite:
        raise RuntimeError("GPU smoke operation produced non-finite values.")

    metrics = {
        "cuda_available": True,
        "device_count": torch.cuda.device_count(),
        "gpu_name": properties.name,
        "gpu_memory_bytes": properties.total_memory,
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "matrix_size": size,
        "matmul_latency_ms": round(elapsed_ms, 3),
        "all_finite": all_finite,
    }
    experiment = {
        "schema_version": 1,
        "experiment_id": "kaggle-gpu-smoke-v1",
        "kind": "infrastructure_smoke_test",
        "status": "passed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "scope": (
            "Confirms Kaggle GPU allocation and artifact output only; "
            "does not train a language model or evaluate a routing architecture."
        ),
    }

    working_dir = Path("/kaggle/working")
    if not working_dir.is_dir():
        working_dir = Path.cwd()
    output_dir = working_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "experiment.json").write_text(
        json.dumps(experiment, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"experiment": experiment, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
