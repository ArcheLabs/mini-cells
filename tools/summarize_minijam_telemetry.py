#!/usr/bin/env python3
"""Summarize measured MiniJAM execution telemetry.

Input is a JSON array, or an object containing a ``samples`` array. Each
sample records measured_refine_gas, measured_accumulate_gas, wall_time_ms,
peak_memory_bytes, batch_count, and sample_count. Gas headroom is calculated
against the canonical MiniJamSpec v1 limits.
"""
import argparse
import json
import math
from pathlib import Path
from typing import List


REFINE_LIMIT = 1_000_000_000
ACCUMULATE_LIMIT = 1_000_000_000
NUMERIC_FIELDS = {
    "measured_refine_gas": "refine_gas",
    "measured_accumulate_gas": "accumulate_gas",
    "wall_time_ms": "wall_time_ms",
    "peak_memory_bytes": "peak_memory_bytes",
}


def percentile(values: List[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def summarize(payload: object) -> dict:
    samples = payload["samples"] if isinstance(payload, dict) else payload
    if not isinstance(samples, list) or not samples:
        raise ValueError("telemetry input must contain a non-empty samples array")
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("every telemetry sample must be an object")
        missing = [field for field in NUMERIC_FIELDS if field not in sample]
        if missing:
            raise ValueError("telemetry sample missing: " + ", ".join(missing))

    metrics = {}
    for source, name in NUMERIC_FIELDS.items():
        values = [float(sample[source]) for sample in samples]
        metrics[name] = {
            "p50": percentile(values, 0.50),
            "p95": percentile(values, 0.95),
            "max": max(values),
        }
        for key, value in metrics[name].items():
            if value.is_integer():
                metrics[name][key] = int(value)

    refine_p95 = metrics["refine_gas"]["p95"]
    accumulate_p95 = metrics["accumulate_gas"]["p95"]
    return {
        "schema": "minicells.minijam-telemetry.v1",
        "minijam_spec": "v1",
        "sample_count": sum(int(sample["sample_count"]) for sample in samples),
        "batch_count": sum(int(sample["batch_count"]) for sample in samples),
        "run_count": len(samples),
        "metrics": metrics,
        "headroom": {
            "refine": {
                "limit": REFINE_LIMIT,
                "p95_margin": REFINE_LIMIT - refine_p95,
                "p95_fraction": refine_p95 / REFINE_LIMIT,
            },
            "accumulate": {
                "limit": ACCUMULATE_LIMIT,
                "p95_margin": ACCUMULATE_LIMIT - accumulate_p95,
                "p95_fraction": accumulate_p95 / ACCUMULATE_LIMIT,
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(json.loads(args.input.read_text()))
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
