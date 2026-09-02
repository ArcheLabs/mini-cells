#!/usr/bin/env python3
"""Run Native CLM v0 M1 next-token training from a frozen JSON config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from minicells.native_clm_train import load_configs, train_m1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("research/stages/06-native-clm/configs/native-clm-v0-m1-12m.json"),
    )
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--validation-file", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/experiments/native-clm-v0-m1-next-token"),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="development override; canonical M1 uses config value",
    )
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default=None)
    args = parser.parse_args()

    model_config, train_config = load_configs(args.config)
    if args.max_steps is not None:
        train_config = type(train_config)(**{**train_config.__dict__, "max_steps": args.max_steps})
    if args.precision is not None:
        train_config = type(train_config)(**{**train_config.__dict__, "precision": args.precision})

    summary = train_m1(
        model_config=model_config,
        train_config=train_config,
        train_path=args.train_file,
        validation_path=args.validation_file,
        output_dir=args.output_dir,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "pass": summary["pass"],
                "parameters": summary["parameter_count"]["total"],
                "initial_eval_loss": summary["initial_eval"]["loss"],
                "final_eval_loss": summary["final_eval"]["loss"],
                "cell_count": summary["cell_count"],
                "active_cells": summary["active_cells"],
                "checkpoint_sha256": summary["final_checkpoint_sha256"],
            },
            indent=2,
        )
    )
    return 0 if summary["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
