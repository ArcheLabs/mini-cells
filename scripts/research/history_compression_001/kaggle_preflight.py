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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Kaggle preflight for History Compression 001")
    parser.add_argument(
        "--minimum-free-mb",
        "--min-free-mib",
        dest="minimum_free_mb",
        type=int,
        default=12000,
        help="minimum free GPU memory in MiB; legacy --min-free-mib is accepted as an alias",
    )
    parser.add_argument("--json-out", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    gpu = _gpu_memory()
    report = {
        "python": sys.version.split()[0],
        "pid": os.getpid(),
        "gpu": gpu,
    }
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(
        json.dumps(
            {
                "gpu": gpu["name"],
                "free_mb": gpu["free_mb"],
                "required_mb": args.minimum_free_mb,
            },
            sort_keys=True,
        )
    )
    free_mb = int(gpu["free_mb"])
    if free_mb < args.minimum_free_mb:
        raise SystemExit(
            f"GPU preflight failed: {free_mb} MiB free < {args.minimum_free_mb} MiB required; "
            "restart the hosted runtime if a stale subprocess owns GPU memory"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
