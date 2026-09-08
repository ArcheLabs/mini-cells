from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = ROOT / "src"
LOCAL_ROOT = ROOT / "scripts" / "research" / "granite_hybrid_clm_v01"
RUNNER = LOCAL_ROOT / "run_milestone.py"
RELOAD = LOCAL_ROOT / "verify_reload.py"
VALIDATOR = LOCAL_ROOT / "validate_result.py"

_BOOTSTRAP = r"""
import importlib.util
import runpy
import sys
from pathlib import Path

script = Path(sys.argv[1]).resolve()
forwarded = sys.argv[2:]
local_dataset = script.parent / "dataset.py"
if local_dataset.exists():
    spec = importlib.util.spec_from_file_location("dataset", local_dataset)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Hybrid CLM dataset from {local_dataset}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["dataset"] = module
    spec.loader.exec_module(module)
sys.argv = [str(script), *forwarded]
runpy.run_path(str(script), run_name="__main__")
"""


def _load_hf_token() -> None:
    if os.environ.get("HF_TOKEN"):
        print("[granite-hybrid-clm-v0.1] hf_token_loaded=True", flush=True)
        return
    try:
        from kaggle_secrets import UserSecretsClient

        token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as exc:
        raise RuntimeError(
            "HF_TOKEN is required. Configure it as a Kaggle Secret or environment variable."
        ) from exc
    if not token:
        raise RuntimeError("HF_TOKEN is configured but empty")
    os.environ["HF_TOKEN"] = token
    print("[granite-hybrid-clm-v0.1] hf_token_loaded=True", flush=True)


def _run(script: Path, *args: str) -> None:
    command = [sys.executable, "-c", _BOOTSTRAP, str(script), *args]
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{SRC_ROOT}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else str(SRC_ROOT)
    )
    print("+", sys.executable, script, *args, flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def _cell_diagnostic(cell: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "cell_id": cell.get("cell_id"),
        "status": cell.get("status"),
    }
    address = cell.get("address")
    if isinstance(address, dict):
        payload["address"] = {
            "passed": address.get("passed"),
            "positive_recall": address.get("positive_recall"),
            "negative_false_positive_rate": address.get("negative_false_positive_rate"),
            "heldout_positive_recall": address.get("heldout_positive_recall"),
            "history_false_positive_rate": address.get("history_false_positive_rate"),
            "minimum_positive_probability": address.get("minimum_positive_probability"),
            "maximum_negative_probability": address.get("maximum_negative_probability"),
            "heldout_minimum_positive_probability": address.get(
                "heldout_minimum_positive_probability"
            ),
            "history_maximum_probability": address.get("history_maximum_probability"),
        }
    transform = cell.get("transform")
    if isinstance(transform, dict):
        candidates = list(transform.get("candidates", []))
        payload["transform"] = {
            "passed": transform.get("passed"),
            "best_nll_gain": transform.get("best_nll_gain"),
            "candidates": candidates,
        }
        if candidates:
            payload["transform_summary"] = {
                "maximum_nll_gain": max(
                    float(item.get("nll_gain", float("-inf"))) for item in candidates
                ),
                "minimum_history_kl": min(
                    float(item.get("history_kl", float("inf"))) for item in candidates
                ),
                "maximum_choice_accuracy": max(
                    float(item.get("choice_accuracy", 0.0)) for item in candidates
                ),
                "eligible_steps": [
                    item.get("step") for item in candidates if item.get("eligible")
                ],
            }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Granite Hybrid CLM v0.1 on Kaggle")
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=26090471)
    args = parser.parse_args()
    _load_hf_token()

    if args.mode == "smoke":
        facts = 3
        address_steps = 80
        transform_steps = 12
        output_dir = ROOT / "results" / "granite-hybrid-clm-v0.1-smoke"
    else:
        facts = 50
        address_steps = 240
        transform_steps = 40
        output_dir = ROOT / "results" / "granite-hybrid-clm-v0.1"

    _run(
        RUNNER,
        "--device",
        args.device,
        "--facts",
        str(facts),
        "--seed",
        str(args.seed),
        "--address-steps",
        str(address_steps),
        "--transform-steps",
        str(transform_steps),
        "--output-dir",
        str(output_dir),
    )

    result = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    if result["status"] != "GRANITE_HYBRID_CLM_V01_SUPPORTED":
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "result_status": result["status"],
                    "committed_facts": result["committed_facts"],
                    "retention_choice_accuracy": result["retention_choice_accuracy"],
                    "reload_status": "SKIPPED_RUNNER_NOT_SUPPORTED",
                    "routing": result.get("routing"),
                    "cells": [_cell_diagnostic(cell) for cell in result.get("cells", [])],
                    "contextual_child": result.get("contextual_child", {}),
                    "output_dir": str(output_dir.relative_to(ROOT)),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    _run(RELOAD, "--device", args.device, "--result-dir", str(output_dir))
    if args.mode == "full":
        _run(VALIDATOR, "--result-dir", str(output_dir))

    reload_report = json.loads(
        (output_dir / "reload_verification.json").read_text(encoding="utf-8")
    )
    print(
        json.dumps(
            {
                "mode": args.mode,
                "result_status": result["status"],
                "committed_facts": result["committed_facts"],
                "retention_choice_accuracy": result["retention_choice_accuracy"],
                "reload_status": reload_report["status"],
                "output_dir": str(output_dir.relative_to(ROOT)),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
