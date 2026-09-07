from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "scripts" / "research" / "granite_hybrid_clm_v01" / "run_milestone.py"
RELOAD = ROOT / "scripts" / "research" / "granite_hybrid_clm_v01" / "verify_reload.py"
VALIDATOR = ROOT / "scripts" / "research" / "granite_hybrid_clm_v01" / "validate_result.py"


def _run(*args: str) -> None:
    command = [sys.executable, *args]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Granite Hybrid CLM v0.1 on Kaggle")
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=26090471)
    args = parser.parse_args()

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
        str(RUNNER),
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
    _run(str(RELOAD), "--device", args.device, "--result-dir", str(output_dir))
    if args.mode == "full":
        _run(str(VALIDATOR), "--result-dir", str(output_dir))

    result = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
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
