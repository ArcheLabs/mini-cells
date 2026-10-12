#!/usr/bin/env python3
"""Run and aggregate Shadow Cell Validation 001 across registered seeds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from minicells.shadow_cell_validation_001 import aggregate_shadow_validation

DEFAULT_PROTOCOL = Path(
    "research/validations/shadow-cell-validation-001-copy-on-write-functional-isolation/protocol.json"
)
DEFAULT_IMPLEMENTATION = Path(
    "research/validations/shadow-cell-validation-001-copy-on-write-functional-isolation/implementation.json"
)
DEFAULT_OUTPUT = Path(
    "artifacts/experiments/shadow-cell-validation-001-copy-on-write-functional-isolation"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_seed_process(
    *,
    seed: int,
    device: int,
    output_dir: Path,
    protocol: Path,
    implementation: Path,
) -> subprocess.Popen[str]:
    command = [
        sys.executable,
        "scripts/research/run_shadow_cell_validation_001_seed.py",
        "--seed",
        str(seed),
        "--device",
        f"cuda:{device}",
        "--output-dir",
        str(output_dir / f"seed-{seed}"),
        "--protocol",
        str(protocol),
        "--implementation",
        str(implementation),
    ]
    print("+", " ".join(command), flush=True)
    return subprocess.Popen(command, text=True)


def _run_parallel(
    *,
    seeds: list[int],
    devices: list[int],
    output_dir: Path,
    protocol: Path,
    implementation: Path,
) -> None:
    if not devices:
        raise ValueError("at least one CUDA device is required")
    pending = list(seeds)
    active: dict[int, tuple[int, subprocess.Popen[str]]] = {}
    while pending or active:
        for device in devices:
            if device in active or not pending:
                continue
            seed = pending.pop(0)
            active[device] = (
                seed,
                _run_seed_process(
                    seed=seed,
                    device=device,
                    output_dir=output_dir,
                    protocol=protocol,
                    implementation=implementation,
                ),
            )
        finished: list[int] = []
        for device, (seed, process) in active.items():
            code = process.poll()
            if code is None:
                continue
            if code != 0:
                raise RuntimeError(f"Shadow Cell seed {seed} failed with return code {code}")
            finished.append(device)
        for device in finished:
            del active[device]
        if active:
            time.sleep(2.0)


def _load_seed_results(
    output_dir: Path,
    seeds: list[int],
    *,
    protocol_sha256: str,
    implementation_sha256: str,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for seed in seeds:
        path = output_dir / f"seed-{seed}" / "seed-result.json"
        if not path.exists():
            raise FileNotFoundError(path)
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("format") != "minicells.shadow-cell-validation-001.seed-result.v1":
            raise RuntimeError(f"unexpected seed result format for {seed}")
        if int(result.get("seed")) != seed:
            raise RuntimeError(f"seed identity mismatch in {path}")
        if result.get("protocol_sha256") != protocol_sha256:
            raise RuntimeError(f"protocol identity mismatch in {path}")
        if result.get("implementation_sha256") != implementation_sha256:
            raise RuntimeError(f"implementation identity mismatch in {path}")
        results.append(result)
    return results


def _write_curve_csv(path: Path, seed_results: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for result in seed_results:
        seed = int(result["seed"])
        for arm, points in result["curves"].items():
            for point in points:
                rows.append({"seed": seed, "arm": arm, **point})
    fields = [
        "seed",
        "arm",
        "maturity",
        "A_regression",
        "B_gain",
        "B_gain_fraction_of_direct",
        "A_accuracy",
        "B_accuracy",
        "A_nll",
        "B_nll",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_markdown(path: Path, aggregate: dict[str, Any]) -> None:
    lines = [
        "# Shadow Cell Validation 001 — Copy-on-Write Functional Isolation",
        "",
        f"- Classification: `{aggregate['classification']}`",
        f"- Phase: `{aggregate['phase']}`",
        f"- Scientific decision: `{aggregate['scientific_decision']}`",
        "- Independent of Native CLM M2/M3 conclusion chain: `True`",
        f"- Protocol SHA-256: `{aggregate['protocol_sha256']}`",
        f"- Implementation SHA-256: `{aggregate['implementation_sha256']}`",
        "",
        "| seed | base A acc | parent A share | parent B share | direct B gain | gate AUC | primary m | primary A damage | primary B gain/direct | HV gain vs direct | shuffled A damage advantage |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in aggregate["seed_results"]:
        primary = result["primary_conditional"] or {}
        causal = result["causal_control"]
        lines.append(
            "| {seed} | {base:.4f} | {pa:.4f} | {pb:.4f} | {gain:.4f} | {auc:.4f} | {m} | {areg} | {bg} | {hv:.4f} | {shuffle} |".format(
                seed=result["seed"],
                base=result["base_metrics"]["A"]["accuracy"],
                pa=result["parent"]["top1_share_A"],
                pb=result["parent"]["top1_share_B"],
                gain=result["direct_tx"]["B_gain"],
                auc=result["gate"]["heldout_auc"],
                m="-" if not primary else f"{primary['maturity']:.4f}",
                areg="-" if not primary else f"{primary['A_regression']:.4f}",
                bg="-" if not primary else f"{primary['B_gain_fraction_of_direct']:.4f}",
                hv=result["hypervolume"]["conditional_improvement_vs_direct_interp"],
                shuffle=(
                    "-"
                    if causal["correct_vs_shuffled_A_regression_advantage"] is None
                    else f"{causal['correct_vs_shuffled_A_regression_advantage']:.4f}"
                ),
            )
        )
    lines.extend(
        [
            "",
            "Boundary: this experiment uses fresh synthetic data, fresh base checkpoints and fresh seeds. A/B calibration data may train the expression probe, but old examples never enter Shadow/direct operator weight training. The experiment does not modify any Native CLM M2/M3 scientific decision.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("development", "formal"), required=True)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--implementation", type=Path, default=DEFAULT_IMPLEMENTATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-completed", action="store_true")
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    implementation = json.loads(args.implementation.read_text(encoding="utf-8"))
    if protocol.get("format") != "minicells.shadow-cell-validation-001.protocol.v1":
        raise RuntimeError("unexpected protocol format")
    if protocol.get("status") != "FROZEN_UNRUN":
        raise RuntimeError("Shadow Cell Validation 001 protocol is not frozen/unrun")
    if implementation.get("format") != "minicells.shadow-cell-validation-001.implementation.v1":
        raise RuntimeError("unexpected implementation format")
    if implementation.get("status") != "FROZEN_UNRUN":
        raise RuntimeError("Shadow Cell Validation 001 implementation is not frozen/unrun")
    protocol_sha = sha256_file(args.protocol)
    implementation_sha = sha256_file(args.implementation)
    seed_key = "development_seeds" if args.phase == "development" else "formal_seeds"
    seeds = [int(value) for value in protocol["fresh_evidence"][seed_key]]
    devices = [int(value.strip()) for value in args.devices.split(",") if value.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    to_run = seeds
    if args.skip_completed:
        to_run = [
            seed
            for seed in seeds
            if not (args.output_dir / f"seed-{seed}" / "seed-result.json").exists()
        ]
    if to_run:
        _run_parallel(
            seeds=to_run,
            devices=devices,
            output_dir=args.output_dir,
            protocol=args.protocol,
            implementation=args.implementation,
        )

    seed_results = _load_seed_results(
        args.output_dir,
        seeds,
        protocol_sha256=protocol_sha,
        implementation_sha256=implementation_sha,
    )
    aggregate = aggregate_shadow_validation(
        seed_results,
        thresholds=protocol["thresholds"],
        protocol_sha256=protocol_sha,
        phase=args.phase,
    )
    aggregate["implementation_sha256"] = implementation_sha
    decision_path = args.output_dir / "decision.json"
    decision_path.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_curve_csv(args.output_dir / "pareto-points.csv", seed_results)
    _write_summary_markdown(args.output_dir / "RESULTS.md", aggregate)
    print(
        json.dumps(
            {
                "classification": aggregate["classification"],
                "phase": aggregate["phase"],
                "scientific_decision": aggregate["scientific_decision"],
                "seeds": aggregate["seeds"],
                "protocol_sha256": aggregate["protocol_sha256"],
                "implementation_sha256": aggregate["implementation_sha256"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
