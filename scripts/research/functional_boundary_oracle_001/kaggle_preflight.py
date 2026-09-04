from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=False)


def _gpu_memory() -> dict[str, int | str]:
    result = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ]
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {(result.stderr or result.stdout).strip()}")
    first = result.stdout.strip().splitlines()[0]
    parts = [part.strip() for part in first.split(",")]
    if len(parts) != 4:
        raise RuntimeError(f"unexpected nvidia-smi output: {first!r}")
    return {
        "name": parts[0],
        "total_mb": int(parts[1]),
        "used_mb": int(parts[2]),
        "free_mb": int(parts[3]),
    }


def _compute_apps() -> list[dict[str, str]]:
    result = _run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if result.returncode != 0:
        return []
    rows: list[dict[str, str]] = []
    for line in result.stdout.strip().splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",", 2)]
        if len(parts) == 3:
            rows.append({"pid": parts[0], "process_name": parts[1], "used_memory_mb": parts[2]})
    return rows


def _matching_processes() -> list[str]:
    result = _run(["ps", "-eo", "pid,ppid,etimes,cmd"])
    if result.returncode != 0:
        return []
    needles = ("moe_mutation_001", "functional_boundary_oracle_001")
    return [line for line in result.stdout.splitlines() if any(needle in line for needle in needles)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Kaggle preflight for Functional Boundary Oracle 001")
    parser.add_argument("--minimum-free-mb", type=int, default=12000)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    gpu = _gpu_memory()
    report = {
        "python": sys.version.split()[0],
        "pid": os.getpid(),
        "gpu": gpu,
        "compute_apps": _compute_apps(),
        "matching_processes": _matching_processes(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    free_mb = int(gpu["free_mb"])
    if free_mb < args.minimum_free_mb:
        raise SystemExit(
            f"GPU preflight failed: only {free_mb} MiB free; "
            f"formal FP32 run requires at least {args.minimum_free_mb} MiB free. "
            "A prior Kaggle subprocess may still own GPU memory. Restart the Kaggle session/runtime "
            "rather than killing an unknown process, then rerun from the top."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
