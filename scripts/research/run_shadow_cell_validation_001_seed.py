#!/usr/bin/env python3
"""Run one fresh Shadow Cell Validation 001 seed."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from minicells.shadow_cell_validation_001 import (
    ShadowValidationConfig,
    run_shadow_validation_seed,
)

DEFAULT_PROTOCOL = Path(
    "research/validations/shadow-cell-validation-001-copy-on-write-functional-isolation/protocol.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("format") != "minicells.shadow-cell-validation-001.protocol.v1":
        raise RuntimeError("unexpected Shadow Cell Validation 001 protocol format")
    registered = set(protocol["fresh_evidence"]["development_seeds"]) | set(
        protocol["fresh_evidence"]["formal_seeds"]
    )
    if args.seed not in registered:
        raise RuntimeError(f"seed {args.seed} is not registered")

    base = protocol["base_training"]
    adapt = protocol["B_adaptation"]
    config = ShadowValidationConfig(
        base_steps=int(base["steps"]),
        base_batch_size=int(base["batch_size"]),
        base_lr=float(base["lr"]),
        base_weight_decay=float(base["weight_decay"]),
        base_warmup_steps=int(base["warmup_steps"]),
        adapt_steps=int(adapt["steps"]),
        adapt_batch_size=int(adapt["batch_size"]),
        adapt_lr=float(adapt["lr"]),
        adapt_warmup_steps=int(adapt["warmup_steps"]),
        grad_clip=float(adapt["grad_clip"]),
        precision=str(base["precision"]),
    )
    result = run_shadow_validation_seed(
        seed=args.seed,
        counts={key: int(value) for key, value in protocol["data_counts_per_seed"].items()},
        maturity_grid=[float(value) for value in protocol["maturity_grid"]],
        thresholds=protocol["thresholds"],
        output_dir=args.output_dir,
        device=args.device,
        config=config,
    )
    summary = {
        "seed": result["seed"],
        "protocol_sha256": sha256_file(args.protocol),
        "base_A_accuracy": result["base_metrics"]["A"]["accuracy"],
        "parent_share_A": result["parent"]["top1_share_A"],
        "parent_share_B": result["parent"]["top1_share_B"],
        "direct_B_gain": result["direct_tx"]["B_gain"],
        "gate_AUC": result["gate"]["heldout_auc"],
        "primary_conditional": result["primary_conditional"],
        "identity": result["identity"],
        "gates": result["gates"],
    }
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
